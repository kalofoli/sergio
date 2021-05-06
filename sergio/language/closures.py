'''
Created on May 3, 2021

@author: janis
'''

import numpy as np
from .base import ConjunctionSelector, LanguageSelector, Cache, \
    property_predicate_objects, Predicate, \
    PredicateOrIndexCollectionType, PredicateOrIndexType, CacheSpecType
    

from .utils import EffectiveValiditiesTracker, EffectiveValiditiesView, CandidateRestriction, Indices

from typing import Tuple, List, Union, cast, NamedTuple, Iterator, Sequence
import itertools
from sergio.language.base import PredicateCollectionType, Selector,\
    ConjunctionLanguage
from colito.summaries import SummaryOptions
from colito.logging import getModuleLogger
from sergio.language.utils import indices_remove

log = getModuleLogger(__name__)


class ClosureConjunctionSelector(ConjunctionSelector, LanguageSelector):

    def __init__(self, language: 'ClosureConjunctionLanguageBase',
                 predicates: PredicateOrIndexCollectionType,
                 cache: CacheSpecType=True,
                 _closure_indices: np.ndarray=None,
                 _index_needed_end:int = None,
                 _indices_sorted: np.ndarray=None) -> None:
        super().__init__(language=language, predicates=predicates, cache=cache, _indices_sorted=_indices_sorted)
        self._closure_indices = _closure_indices
        if _index_needed_end is not None:
            self._cache.update(index_needed_end = _index_needed_end)

    @Cache.cached_property
    def indices_compact(self) -> Tuple[int, ...]:
        indices_compact = self.language.indices_minimal_approximation(self.validity)
        return indices_compact
        
    @property_predicate_objects
    def predicates_compact(self) -> Tuple[int, ...]:
        '''Predicate indices in the closure'''
        return self.indices_compact

    #@Cache.cached_property
    @property
    def validity(self) -> np.ndarray:
        validity = self.language.validate(self.predicates)
        return validity

    @Cache.cached_property
    def index_needed_end(self):
        '''The end of the needed predicates, without which which the selector changes'''
        index = self._language.index_needed_end(self.validity, self.indices)
        return index

    @property
    def predicate_max(self) -> Predicate:
        index_max = self.index_max
        return self.language.predicate_object(index_max) if index_max is not None else None
    
    @Cache.cached_property
    def tail_indices(self):
        index_max = self.index_max
        tail_indices = tuple(filter(lambda x: x < index_max, self.indices))
        return tail_indices

    @property
    def closure_indices(self) -> np.ndarray:
        if self._closure_indices is None:
            indices_closure = self._language.indices_closure(self.validity, blacklist=self.indices)
            self._closure_indices = np.sort(np.r_[indices_closure, self.indices])
        return self._closure_indices
    
    def __repr__(self):
        spreds = ' AND '.join(map(repr, self.predicates))
        sclos = f'{len(self.closure_indices)}' if self._closure_indices is not None else '?'
        return f'<{__class__.__name__} {len(self.indices)}/{sclos}: {spreds}>'

    class Digest(NamedTuple):  # pylint: disable = too-few-public-methods
        indices: List[int]
        indices_path: List[int]
    
    def serialise(self, json=False):
        digest = ClosureConjunctionSelector.Digest(indices=self.indices, indices_path=self.indices_path)
        if json:
            digest = digest._asdict()  # pylint: disable=no-member
        return digest
    
    @classmethod
    def deserialise(cls, language, digest):
        if isinstance(digest, dict):
            digest = ClosureConjunctionSelector.Digest(**digest)
        return cls(language, predicates_path=digest.indices_path, _closure=digest.indices)
    
    def __summary_dict__(self,options:SummaryOptions):
        dct = super().__summary_dict__(options)
        dct['path_description'] = self.path_description
        #if options.parts & SummaryParts.PREDICATE_INDICES:
        #    dct['path_indices'] = self.indices_path
        #    dct['compact_indices'] = self.indices_compact
        return dct


class ClosureConjunctionLanguageBase(ConjunctionLanguage):

    __collection_tag__ = None
    __selector_class__ = ClosureConjunctionSelector

    def __init__(self, data, predicates: PredicateCollectionType=None) -> None:
        super(ClosureConjunctionLanguageBase, self).__init__(data=data, predicates=predicates)
        self._predicate_supports: np.ndarray = self._predicate_validities.sum(axis=0)
    
    @property
    def predicate_validities(self): return self._predicate_validities
        
    def refine(self, selector: ClosureConjunctionSelector, greater_only: bool=False,
               blacklist: PredicateOrIndexCollectionType=None) -> Iterator[ClosureConjunctionSelector]:
        raise NotImplementedError()
    
    def indices_minimal_approximation(self, validity: np.ndarray, indices_closure=None) -> Tuple[int, ...]:
        '''Compute an approximation to the minimal predicate list describing the closure'''
        if indices_closure is None:
            indices_closure = np.fromiter(self.indices_closure(validity), int)
        minimal_indices_closure = []
        remainder = ~validity
        closure_validities = self._predicate_validities[:, indices_closure]
        while remainder.sum():
            minimal_index = np.argmin(closure_validities[remainder, :].sum(axis=0))
            remainder &= closure_validities[:, minimal_index]
            minimal_indices_closure.append(minimal_index)
        minimal_indices = indices_closure[minimal_indices_closure]
        return tuple(map(int, minimal_indices))
    
    def index_needed_end(self, validity: np.ndarray, indices: Tuple[int, ...]) -> int:
        '''Return the predicate with the maximal index that is necessary to not change the selector support''' 
        if indices:
            support = validity.sum()
            indices_sorted = sorted(indices)
            buffer = self._predicate_validities[:, indices_sorted]
            ## TODO: Speedup
            running_coverage = np.cumprod(buffer, out=buffer, axis=1, dtype=bool).sum(axis=0)
            closure_index = np.where(running_coverage == support)[0][0]
            index_end = int(indices_sorted[closure_index]) + 1
        else:
            index_end = 0
        return index_end
        
    def indices_closure(self, validity: np.ndarray, blacklist=None) -> Tuple[int, ...]:
        '''Return the predicate indices forming the closure of the selector'''
        from .utils import ValidityPrincipalMatrix
        if blacklist is not None:
            vpm = ValidityPrincipalMatrix(self._predicate_validities, validity, blacklist=blacklist)
            is_superset = vpm._supports == vpm.nrows
            indices = vpm.candidates2predicates(is_superset)
        else:
            indices = np.where(self._predicate_validities[validity, :].all(axis=0))[0]
        return indices

    def _get_validity_and_support(self, what:Union[Predicate, Selector]):
        if isinstance(what, Predicate):
            master_idx = self.predicate_index(what)
            validity = self._predicate_validities[:, master_idx]
            support = self._predicate_supports[master_idx]
        elif isinstance(what, Selector):
            validity = what.validity
            support = validity.sum()
        elif isinstance(what, np.ndarray):
            validity = cast(np.ndarray, what)
            support = validity.sum()
        else:
            raise TypeError(f'Can not infer validity from object {what} of type {what.__class__}.')
        return validity, support 
    
    def predicate_is_superset(self, master:Union[Predicate, Selector], candidates:Sequence[Predicate], only_proper:bool=False) -> np.ndarray:
        master_validity, master_support = self._get_validity_and_support(master)
        candidate_idx = list(self.predicate_indices(candidates))
        cover_sums = np.bitwise_and(master_validity[:,None], self._predicate_validities[:, candidate_idx]).sum(axis=0)
        return (cover_sums > master_support) if only_proper else (cover_sums >= master_support)

    def predicate_is_subset(self, master:Union[Predicate, Selector], candidates:Sequence[Predicate], only_proper:bool=False) -> np.ndarray:
        master_validity, master_support = self._get_validity_and_support(master)
        candidate_idx = list(self.predicate_indices(candidates))
        cover_sums = np.bitwise_and(master_validity[:,None], self._predicate_validities[:, candidate_idx]).sum(axis=0)
        return (cover_sums < master_support) if only_proper else (cover_sums <= master_support)

    class Digest(NamedTuple):  # pylint: disable = too-few-public-methods
        language: str
        selectors: List[ClosureConjunctionSelector.Digest]
        
    def serialise_selectors(self, selectors):
        digests = list(s.serialise() for s in selectors)
        return ClosureConjunctionLanguageBase.Digest(language=self.tag, selectors=digests) # pylint: disable = no-value-for-parameter)
    
    def deserialise_selectors(self, digest):
        selectors = list(ClosureConjunctionSelector.deserialise(self, d) for d in digest.selectors)
        return selectors

    def deserialise_selector(self, digest):
        return ClosureConjunctionSelector.deserialise(self, digest)

class ClosureConjunctionLanguageRestricted(ClosureConjunctionLanguageBase):

    __collection_tag__ = 'closure-conjunctions-restricted'

    def refine(self, selector: ClosureConjunctionSelector, blacklist: PredicateOrIndexCollectionType=None) -> Iterator[ClosureConjunctionSelector]:
        import warnings
        warnings.warn('Incomplete code. Untested. Avoid using.')
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
            if not supersets.blacklist.indices.size:
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
        
#         if False:
#             '''DEVELOPMENT CODE re-implementing the above more correctly, but NOT COMPLETE'''
#             ev_pm = ValidityPrincipalMatrix(self.predicate_validities, selector.validity, idl_whitelist)
#             
#             if blacklist is not None:
#                 idx_blacklist = np.fromiter(self.predicate_indices(blacklist),int)
#                 idl_whitelist[idx_blacklist] = False
#             else:
#                 idx_blacklist = np.zeros(0, int)
#             
#             idc_blacklist = ev_pm.predicates2candidates(idx_blacklist)
#             idc_whitelist = ev_pm.predicates2candidates(idl_whitelist)
#             
#             is_root_superset = ev_pm.candidate_supports == sup_root
#             if is_root_superset[idc_blacklist].any():
#                 return
#             
#             'Remove whitelisted with empty support and those with full support' 
#             is_ok = ~(is_root_superset | (ev_pm.candidate_supports == 0) )
#             ev_pm = ev_pm.select_candidates(is_ok)
#             idc_whitelist,idc_blacklist = indices_slice(is_ok, (idc_whitelist, idc_blacklist))
#             
#             states = []
#             if len(idc_whitelist):
#                 state0 = State(candidate=np.argmin(ev_pm.candidate_supports), whitelist=idc_whitelist, blacklist=idc_blacklist)
#                 states.append(state0)
#             
#             while states:
#                 state = states.pop()
#                 
#                 ev_cand = ev_pm.get_candidate_ev(None, state.candidate)
#                 sup_cand = ev_pm.candidate_supports[state.candidate]
#                 bl_supports = ev_pm.get_candidate_supports(ev_cand, state.blacklist)
#                 'Ignore blacklisted'
#                 if np.max(bl_supports, initial=-1) == sup_cand:
#                     continue
#                 
#                 if not len(state.whitelist):
#                     continue
#                 
#                 wl_supports_inner = ev_pm.get_candidate_supports(ev_cand, state.whitelist)
#                 wl_supports_outer = ev_pm.candidate_supports - wl_supports_inner
#                 
#                 is_super = wl_supports_inner == sup_cand
#                 is_cover = is_super & (wl_supports_outer == 0)
#                 
#                 new_states = []
#                 ''' Add each proper superset with the current predicate as covered'''
#     
#                 proper_cands = np.where(is_super & ~is_cover)[0]
#                 
#                 for proper_cand in proper_cands:
#                     idc_wl_new = indices_remove(idc_whitelist, state.candidate)
#                     new_state = State(proper_cand, idc_wl_new, idc_blacklist)
#                     new_states.append(new_state)
#                 
#                 if wl_supports_inner[:state.candidate].max() == 0:
#                     'We are the lowest index for this intersection. Add.'
#                     refinement = selector.extend(state.predicate, covered=state.covered)
#                     yield refinement 
#                 
#                 states += sorted(new_states, key = lambda s:-s.support)
#             

        
class ClosureConjunctionLanguageFull(ClosureConjunctionLanguageBase):

    __collection_tag__ = 'closure-conjunctions-full'

    def refine(self, selector: ClosureConjunctionSelector, blacklist: PredicateOrIndexCollectionType=None) -> Iterator[ClosureConjunctionSelector]:
        raise NotImplementedError()  # this code is wrong
        num_predicates = len(self.predicates)

        extension_indices = tuple(self._indices_refinement_extensions(selector=selector, blacklist=blacklist))
        num_predicates = len(self.predicates)
        selector_predicate_supports = self._predicate_validities[selector.validity, :].sum(axis=0)
        extension_indices_remain = np.ones(num_predicates, bool)
        extension_indices_remain[list(selector.indices)] = False
        refinement_covered = np.zeros(num_predicates, bool)  # dbgremove
        for extension_index in extension_indices:
            if extension_indices_remain[extension_index]:
                refinement = selector.extend(extension_index)
                refinement_indices = list(refinement.indices)
                support = refinement.validity.sum()
                refinement_covered[:] = False  # dbgremove
                refinement_covered[refinement_indices] = True  # dbgremove
                refinement_covered &= selector_predicate_supports == support  # dbgremove
                extension_indices_remain &= ~refinement_covered  # dbgremove
                # Debug unfriendly faster equivalent:
                extension_indices_remain[refinement_indices] &= selector_predicate_supports[refinement_indices] != support
                yield refinement


class ClosureConjunctionLanguageSlow(ClosureConjunctionLanguageBase):

    __collection_tag__ = 'closure-conjunctions-slow'
    
    def refine(self, selector: ClosureConjunctionSelector, blacklist: PredicateOrIndexCollectionType=None) -> Iterator[ClosureConjunctionSelector]:

        ''' We do not consider (w.r.t. the current validity):
         * completely empty predicates
         * completely full predicates
        '''
        is_nz = self._predicate_validities[selector.validity,:].any(axis=0)
        is_full = self._predicate_validities[selector.validity,:].all(axis=0)
        extension_indices = np.where(is_nz & ~is_full)[0]
        
        refinements = set(selector.extend(idx) for idx in extension_indices)
        
        iter_refinements = iter(refinements)
        if blacklist is not None:
            blacklist_set = set(blacklist)

            def not_blacklisted(selector: ConjunctionSelector):
                common = selector.index_set & blacklist_set
                return not common

            iter_refinements = filter(not_blacklisted, iter_refinements)
            
        return iter_refinements

class ClosureConjunctionLanguageTester(ClosureConjunctionLanguageRestricted):

    __collection_tag__ = 'closure-conjunctions-tester'
    
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self._slow = ClosureConjunctionLanguageSlow(*args, **kwargs)
        
    def refine(self, selector: ClosureConjunctionSelector, blacklist: PredicateOrIndexCollectionType=None) -> Iterator[ClosureConjunctionSelector]:
        refinements = sorted(super().refine(selector, blacklist), key=str)
        refinements_slow = sorted(self._slow.refine(selector, blacklist), key=str)
        if refinements_slow != refinements:
            log.error(f'Refinements do not match, while refining {selector} with blacklist {blacklist}. \nSlow: {refinements_slow}\nRstr: {refinements}')
        return refinements


