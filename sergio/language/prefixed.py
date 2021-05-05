'''
Created on May 3, 2021

@author: janis
'''

import numpy as np

from .base import PredicateOrIndexCollectionType, CacheSpecType
from .closures import ClosureConjunctionLanguageBase, ClosureConjunctionSelector
from .utils import EffectiveValiditiesTracker,\
    EffectiveValiditiesView, Indices, ValidityPrincipalMatrix, indices_slice, indices_remove
from typing import Iterator, NamedTuple
from colito.logging import getModuleLogger

log = getModuleLogger(__name__)

class CoveredClosureConjunctionSelector(ClosureConjunctionSelector):
    def __init__(self, language:'ClosureConjunctionLanguageBase', 
        predicates_path:PredicateOrIndexCollectionType, 
        cache:CacheSpecType=True, covered:PredicateOrIndexCollectionType=())->None:
        ClosureConjunctionSelector.__init__(self, language, predicates_path, cache=cache)
        self._covered = tuple(language.predicate_indices(covered))
    
    @property
    def covered(self): return self._covered
    

class CandidateRestriction:
    '''Restricting the remaining superset candidates
    
    Serves as a state furing the refinement creation of closure languages. 
    
    :param extension ndarray: the indices of the remaining extensions to consider
    :param closure: the indices of the potential closures each of the extensions might have
    :param blacklist ndarray: the indices of the active blacklisted predicates (i.e. potential supersets)
    '''

    def __init__(self, ev_view, extension:np.ndarray, closure:np.ndarray, blacklist:np.ndarray):
        self._ev_view: EffectiveValiditiesView = ev_view
        self._blacklist = blacklist
        self._extension = extension
        self._closure = closure
        self._min_sup_extension = None
    
    def supersets(self, effective_validity):
        supersets = self._ev_view.superset_sequence(effective_validity, (self._extension, self._closure, self._blacklist))
        supersets_set = SupersetCollection(extension=supersets[0], closure=supersets[1], blacklist=supersets[2])
        return supersets_set
    
    @property
    def extension(self):
        return self._extension
    
    @property
    def minsup(self):
        '''The support of the extension with the minimum support'''
        return self._ev_view.supports_of(self.minsup_extension)
    
    @property
    def minsup_extension(self):
        '''The extension with the minimum support'''
        if self._min_sup_extension is None:
            ext_supports = self._ev_view.supports_of(self._extension)
            self._min_sup_extension = self._extension[np.argmin(ext_supports)]
        return self._min_sup_extension

    def restrict_effective_validity(self, ev, supersets=None):
        '''Create a restriction whose effective validity is the one specified'''
        raise NotImplementedError('Deprecated')
        ev_new = self._ev_view.restrict_effective_validity(ev)
        if supersets is None:
            supersets = self.supersets(ev)
        restriction_new = CandidateRestriction(ev_view=ev_new,
                                               extension=supersets.extension.indices,
                                               closure=supersets.closure.indices,
                                               blacklist=supersets.blacklist.indices)
        return restriction_new

    def restrict(self, ev=None, extension=None, closure=None, blacklist=None):
        '''Create a restriction where the specified elements are replaced'''
        extension_new = extension if extension is not None else self._extension
        closure_new = closure if closure is not None else self._closure
        blacklist_new = blacklist if blacklist is not None else self._blasklist
        candidates = Indices.union((extension_new, closure_new, blacklist_new))
        if ev is not None:
            ev_new = self._ev_view.restrict_effective_validity(ev, candidates,drop_empty=True)
        else:
            ev_new = self._ev_view
        restriction_new = CandidateRestriction(ev_view=ev_new,
                                               extension=extension_new,
                                               closure=closure_new,
                                               blacklist=blacklist_new)
        return restriction_new
    
    def remove_candidates(self, extension=None, closure=None, blacklist=None):
        remove_candidates = Indices.remove
        extension_new = remove_candidates(self._extension, extension) if extension is not None else self._extension
        closure_new = remove_candidates(self._closure, closure) if closure is not None else self._closure
        blacklist_new = remove_candidates(self._blacklist, blacklist) if blacklist is not None else self._blacklist
        candidates = Indices.union((extension_new, closure_new, blacklist_new))
        ev_new = self._ev_view.restrict_candidates(candidates)
        restriction = CandidateRestriction(ev_view=ev_new,
                                           extension=extension_new,
                                           closure=closure_new,
                                           blacklist=blacklist_new)
        return restriction
    
    def validity_of(self, predicate):
        return self._ev_view.validity_of(predicate)

    def __repr__(self):
        return f'<{self.__class__.__name__}\nE:{self._extension}\nC:{self._closure}\nB:{self._blacklist}>'
    
    def dump(self):
        from sdcore.utils import print_array
        data = self.validity_of(self._ev_view.candidates)
        sel = self._ev_view.map_predicate2candidate(self.minsup_extension)
        txt = print_array(data,select=(sel,), no_print=True)                
        union = np.array(sorted(set(self._extension) | set(self._closure) | set(self._blacklist)))
        return (f'<{self.__class__.__name__}\n'
                f' E:{self._extension}\n'
                f' C:{self._closure}\n'
                f' B:{self._blacklist}\n'
                f' U:{union}\n'
                f'{txt}\n>')
        
class State(NamedTuple):
    candidate: np.ndarray
    whitelist: np.ndarray
    blacklist: np.ndarray


class ClosureConjunctionLanguagePrefixed(ClosureConjunctionLanguageBase):
    __collection_tag__ = 'closure-conjunctions-prefixed'

    __selector_class__ = CoveredClosureConjunctionSelector

    def refine(self, selector: ClosureConjunctionSelector, blacklist: PredicateOrIndexCollectionType=None) -> Iterator[ClosureConjunctionSelector]:
        
        sup_sel = selector.validity.sum()
        num_predicates = len(self.predicates)
        
        evt = EffectiveValiditiesTracker(self.predicate_validities, selector.validity)

        idl_whitelist = np.ones(num_predicates, bool)
        if blacklist is not None:
            idx_blacklist = np.fromiter(self.predicate_indices(blacklist),int)
            idl_whitelist[idx_blacklist] = False
        else:
            idx_blacklist = np.zeros(0, int)
        
        is_superset_prd = evt.supports == sup_sel
        if np.any(is_superset_prd[idx_blacklist]):
            return
        
        'Remove non-consequential'
        idl_whitelist[evt.supports == 0] = False
        idl_whitelist[evt.supports == sup_sel] = False
        
        idx_whitelist = np.arange(num_predicates)[idl_whitelist]
        
        end_needed_pr = selector.index_end
        end_needed_wl = np.searchsorted(idx_whitelist, end_needed_pr)
        sel_prefix_wl = is_superset_prd[idx_whitelist[:end_needed_wl]]
        
        for cand, pred in enumerate(idx_whitelist):
            sup_cur = evt.supports[pred]
            sup_bl = evt.intersection_support_of(pred, idx_blacklist)
            if np.max(sup_bl, initial=-1) == sup_cur:
                continue
            sup_wl = evt.intersection_support_of(pred, idx_whitelist)
            cur_prefix_wl = sup_wl[:end_needed_wl] == sup_cur
            if np.any(cur_prefix_wl != sel_prefix_wl):
                continue
            #last_in_closure = np.max(np.where(sup_wl==sup_sel)[0])
            #if last_in_closure != cand:
            #    continue
            refinement = selector.extend(pred)
            yield refinement
        
        return
        
        
        ev_pm = ValidityPrincipalMatrix(self.predicate_validities, selector.validity, idl_whitelist)
        
        if blacklist is not None:
            idx_blacklist = np.fromiter(self.predicate_indices(blacklist),int)
            idl_whitelist[idx_blacklist] = False
        else:
            idx_blacklist = np.zeros(0, int)
        
        idc_blacklist = ev_pm.predicates2candidates(idx_blacklist)
        idc_whitelist = ev_pm.predicates2candidates(idl_whitelist)
        
        is_root_superset = ev_pm.candidate_supports == sup_root
        if is_root_superset[idc_blacklist].any():
            return
        
        'Remove whitelisted with empty support and those with full support' 
        is_ok = ~(is_root_superset | (ev_pm.candidate_supports == 0) )
        ev_pm = ev_pm.select_candidates(is_ok)
        idc_whitelist,idc_blacklist = indices_slice(is_ok, (idc_whitelist, idc_blacklist))
        
        states = []
        if len(idc_whitelist):
            state0 = State(candidate=np.argmin(ev_pm.candidate_supports), whitelist=idc_whitelist, blacklist=idc_blacklist)
            states.append(state0)
        
        while states:
            state = states.pop()
            
            ev_cand = ev_pm.get_candidate_ev(None, state.candidate)
            sup_cand = ev_pm.candidate_supports[state.candidate]
            bl_supports = ev_pm.get_candidate_supports(ev_cand, state.blacklist)
            'Ignore blacklisted'
            if np.max(bl_supports, initial=-1) == sup_cand:
                continue
            
            if not len(state.whitelist):
                continue
            
            wl_supports_inner = ev_pm.get_candidate_supports(ev_cand, state.whitelist)
            wl_supports_outer = ev_pm.candidate_supports - wl_supports_inner
            
            is_super = wl_supports_inner == sup_cand
            is_cover = is_super & (wl_supports_outer == 0)
            
            new_states = []
            ''' Add each proper superset with the current predicate as covered'''

            proper_cands = np.where(is_super & ~is_cover)[0]
            
            for proper_cand in proper_cands:
                idc_wl_new = indices_remove(idc_whitelist, state.candidate)
                new_state = State(proper_cand, idc_wl_new, idc_blacklist)
                new_states.append(new_state)
            
            if wl_supports_inner[:state.candidate].max() == 0:
                'We are the lowest index for this intersection. Add.'
                refinement = selector.extend(state.predicate, covered=state.covered)
                yield refinement 
            
            states += sorted(new_states, key = lambda s:-s.support)
        
