'''
Created on May 3, 2021

@author: janis
'''

import numpy as np

from .base import PredicateOrIndexCollectionType
from .closures import ClosureConjunctionLanguageBase, ClosureConjunctionSelector
from sergio.language.utils import EffectiveValiditiesTracker,\
    EffectiveValiditiesView, CandidateRestriction, Indices
from typing import Iterator
from colito.logging import getModuleLogger

log = getModuleLogger(__name__)

class ClosureConjunctionLanguagePrefixed(ClosureConjunctionLanguageBase):
    __collection_tag__ = 'closure-conjunctions-prefixed'

    def refine(self, selector: ClosureConjunctionSelector, blacklist: PredicateOrIndexCollectionType=None) -> Iterator[ClosureConjunctionSelector]:
        
        
        num_predicates = len(self.predicates)
        if blacklist is not None:
            blacklist = np.fromiter(self.predicate_indices(blacklist),int)
            whitelist = Indices.invert(blacklist, num_predicates)
        else:
            blacklist = np.zeros(0, int)
            whitelist = np.arange(num_predicates)
                
        restrictions = []
        ev_tracker = EffectiveValiditiesTracker(self._predicate_validities, effective_validity=selector.validity)
        ev_view = EffectiveValiditiesView(tracker=ev_tracker)
        
        is_root_superset = ev_tracker.supports == selector.validity.sum()
        if is_root_superset[blacklist].any():
            return
        
        root_supersets = np.where(is_root_superset)[0]
        whitelist = whitelist[~is_root_superset[whitelist]] # no need to parse these. They are always in the closure.
        whitelist = whitelist[ev_tracker.supports[whitelist] != 0]
        blacklist = blacklist[ev_tracker.supports[blacklist] != 0]
        ev_view = ev_view.restrict_candidates(Indices.union((whitelist, blacklist)))
        root_cr = CandidateRestriction(ev_view=ev_view, 
                                       extension = whitelist,
                                       closure=whitelist,
                                       blacklist=blacklist)
        
        if root_cr.extension.size:
            restrictions.append(root_cr)
       
        while restrictions:
            restriction = restrictions.pop()
            
            min_ext = restriction.minsup_extension
            validity = restriction.validity_of(min_ext)
            supersets = restriction.supersets(validity)
            
            ' ### Handle closure printing'
            if not supersets.blacklist:
                ' => Yield a closure comprising all <min_ext> supersets'
                # closure = selector.extend(extension, refinement_predicates)
                # yield refinement
                closure = Indices.union((root_supersets,supersets.closure.indices))
                refinement = selector.extend(min_ext, _closure=closure)
                yield refinement

            ' ### Upkeeping candidates'
            '''remove all {extension}, {closure} and {blacklist} candidates that are:
                - equal to <minexp>
                - have a support less than <minexp>
                NOTE: All those sets are active when they are supersets of an extension.
                      If their support is smaller, there is no need to keep them around.
            '''
            
            ' ### Handle superset candidates'
            '''The new state is:
             * A Candidates object with:
                 > extension: only the PROPER supersets of <min_ext> in the {extension} candidates
                 > closure: only the PROPER supersets of <min_ext> in the {closure} candidates
                 > blacklist: only the PROPER supersets of <min_ext> in {blacklist} candidates
             * A value of predicates extended by 
                 > all equal {candidate} supersets of <minext>  
             '''
            
            restr_ss = restriction.restrict(ev=~validity,
                                            extension=supersets.extension.proper,
                                            closure=supersets.closure.proper,
                                            blacklist=supersets.blacklist.proper)
            if restr_ss.extension.size:
                restrictions.append(restr_ss)
                # print(restr_ss.dump())                
                # print(f'SS restriction: {restr_ss}')
            
            ' == Handle non-superset candidates'
            ''' Create a Candidates object with:
             * extension: all extension candidates except:
                          - all supersets of <min_ext>
                             * the equal sets are already contained in the closure
                             * the proper supersets are handled by the superset candidates
             * closure:   all {closure} candidates
                          (- except: all sets equal to <min_ext>) 
             * blacklist: all {blacklist} candidates
                          (- except: all sets equal to <min_ext>) 
            '''

            restr_non_ss = restriction.remove_candidates(extension=supersets.extension.indices,
                                                         closure=supersets.closure.equal,
                                                         blacklist=supersets.blacklist.equal)
            if restr_non_ss.extension.size:
                restrictions.append(restr_non_ss)
                # print(restr_non_ss.dump())                
                # print(f'NSS restriction: {restr_non_ss}')
