'''
Created on May 3, 2021

@author: janis
'''

import numpy as np

from .base import PredicateOrIndexCollectionType, CacheSpecType
from .closures import ClosureConjunctionLanguageBase, ClosureConjunctionSelector
from .utils import EffectiveValiditiesTracker
from typing import Iterator
from colito.logging import getModuleLogger

log = getModuleLogger(__name__)

class ClosureConjunctionSelectorPrefixed(ClosureConjunctionSelector):
    
    def get_parent(self, normalise=True):
        indices = np.array(self.indices)
        if len(indices):
            indices_head = indices[indices<self.index_needed_end-1]
            candidate = self.language.selector(indices_head)
            if len(candidate.indices) and normalise:
                parent_needed_end = candidate.index_needed_end
                if parent_needed_end == candidate.indices[-1]+1:
                    parent = candidate
                else:
                    indices_norm = indices[indices<parent_needed_end]
                    parent = self.language.selector(indices_norm, _index_needed_end = parent_needed_end)
            else:
                parent = candidate
        else:
            parent = None
        return parent
    
class ClosureConjunctionLanguagePrefixed(ClosureConjunctionLanguageBase):
    '''An LCM based guaranteed tree over the closed sets.
    
    .. note:
        
        This results in selectors with level k which will not be visited by any the k-level extension.
        In other words, some selectors which have a minimum generator will not be seen with this generator, but only in deeper states. 
        
    ''' 
    
    
    __collection_tag__ = 'closure-conjunctions-prefixed'
    __collection_title__ = 'Prefixed Closure Conjunctions Language - LCM Based'
    

    __selector_class__ = ClosureConjunctionSelectorPrefixed

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
        
        end_needed_pr = selector.index_needed_end
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
            refinement = selector.extend(pred, _index_needed_end=pred+1)
            yield refinement
        
        return

