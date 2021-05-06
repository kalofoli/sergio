'''
Created on May 27, 2018

@author: janis
'''

from typing import NamedTuple, List, Iterable

import numpy as np
from numpy import ndarray
from functools import reduce
from builtins import property


   

class Indices:

    @classmethod
    def remove(cls, base, remove):
        return np.array(sorted(set(base) - set(remove)), int)
    
    @classmethod
    def invert(cls, base, max_index):
        return np.array(sorted(set(range(max_index)) - set(base)), int)
    
    @classmethod
    def logical(cls, absolute, length):
        '''Convert an absolute index vector into a logican index one'''
        idl = np.zeros(length,bool)
        idl[absolute] = True
        return idl
    
    @classmethod
    def make_inverse_map(cls, indices, max_index=None):
        '''Given an ordering {0,...,max_index}, creates an inverse map imap.
        
        That is, if index[k]=m, then imap[m] = k.'''
        if max_index is None:
            max_index = indices.max()
        imap = np.zeros(max_index + 1, int) - 1
        imap[indices] = np.arange(len(indices))
        return imap
    
    @classmethod
    def merge_logicals(cls, base, selector, output=None):
        '''Combines a logical w.r.t. the base into base itself.
        
        That is, returns a logical l s.t. A[(np.arange(len(base))[base])][selector] = A[l]
        '''
        if output is None:
            output = base.copy()
        else:
            output[:] = base
        output[output] = selector
        return output

    @classmethod
    def absolute2logical(cls, base, length, output=None):
        '''convert an absolute index to a logical one'''
        if output is None:
            output = np.zeros(length, bool)
        output[base] = True
        return output
    
    @classmethod
    def union(cls, index_collection, map_orig2merged=None):
        '''Merge a sequence of index lists.
        
        @param map_orig2merged
            None or an empty list, or a of (at least) N elements.
            If not None, it will be filled with the indices of the original elements within the merged list.
            If some element is False, the corresponding entry is not computed. If empty, all entries are computed.
        @return The union of all elements in the index_collection 
        '''
        merged = np.array(reduce(np.union1d, index_collection))
        if map_orig2merged is not None:
            if not map_orig2merged:
                map_orig2merged += list(list() for _ in range(len(index_collection)))
            for i in range(len(index_collection)):
                if map_orig2merged[i] is not False:
                    map_orig2merged[i] = np.searchsorted(merged, index_collection[i])
        return merged

    @classmethod
    def member_index(cls, base_srt, v):
        '''Returns a vector of the positions of the elements of v within the base array. If the element does not exist, -1 is returned.
        
        @param base_srt: a sorted int array
        '''
        idx = np.searchsorted(base_srt, v)
        
        is_member = idx < base_srt.size
        is_member[is_member] &= base_srt[idx[is_member]] == v[is_member]
        idx[~is_member] = -1
        return idx

def _from_predicates2(fn):
    def predicated(self, what, predicate):
        candidates = self.predicates2candidates(predicate)
        return fn(self, what, candidates)
    return predicated

def indices_slice(slicer, indices):
    '''Returns a sliced list of a sliced version of each element of indices, where the slicing is done according to the slicer.
    
    The elements are assumed to be integer indices from a collection.
    '''
    if isinstance(indices, np.ndarray):
        indices = [indices]
        unwrap = True
    else: unwrap = False
    sliced_args = []
    if slicer.dtype == bool:
        for arg in indices:
            if arg.dtype == bool:
                sliced_arg = arg[slicer]
            else:
                sliced_arg = arg[slicer[arg]]
            sliced_args.append(sliced_arg)
    else:
        raise NotImplementedError()
    return sliced_args[0] if unwrap else sliced_args

def indices_remove(indices, which, ignore_missing=False):
    '''Remove from a sorted index set a list of indices'''
    x = np.searchsorted(indices, which)
    if np.ndim(x)!=0 and not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    try:
        is_found = indices[x] == which
    except IndexError: is_found = False
    if not ignore_missing and not np.all(is_found):
        if np.ndim(x):
            pos = np.where(is_found)[0]
            fail = np.array(which)[pos]
            raise KeyError(f'Could not remove non-existent entries {fail} at positions {pos}.')
        else:
            raise KeyError(f'Could not remove non-existent entry {which}.')
    else:
        return np.delete(indices, x[is_found])
    

class ValidityPrincipalMatrix:
    '''Holds a tighter copy of a subset of the validities''' 

    def __init__(self, predicate_validities, effective_validity, candidates=None, blacklist=None):
        if candidates is None:
            candidates = np.arange(predicate_validities.shape[1])
        if not isinstance(candidates, np.ndarray):
            candidates = np.array(candidates)
        if blacklist is not None:
            candidates = indices_remove(candidates, blacklist, ignore_missing=True)
            
        # self._validities = np.array(predicate_validities[effective_validity, :], order='F')
        self._predicate_validities = predicate_validities
        self._effective_validity = effective_validity
        self._candidate_validities = predicate_validities[np.ix_(effective_validity, candidates)]
        self._supports = self._candidate_validities.sum(axis=0)
        self._candidates = candidates
        from colito.indexing import Indexer
        self._candidate_indexer = Indexer(predicates=candidates,candidates=None)

    @property
    def ncols(self):
        '''Number of predicates; or equivalently columns'''
        return self._candidate_validities.shape[1]
    
    @property
    def nrows(self):
        '''Number of effectively selected instances; or equivalently rows'''
        return self._candidate_validities.shape[0]
    
    @property
    def candidate_ev(self):
        '''Candidate effective validities. The selected validity subset'''
        return self._candidate_validities
    
    @property
    def candidate_supports(self):
        return self._supports
    
    def get_candidate_supports(self, ev, candidates=None):
        '''Return the supports of contained candidates.
        
        Candidates are a subset of predicates, and their indices correspond to the block column indices.'''
        if ev.mean()>.5:
            sup_neg = self._block_index(~ev, candidates).sum(axis=0)
            sup = self._supports - sup_neg
        else:
            sup = self._block_index(ev, candidates).sum(axis=0)
        return sup

    def get_candidate_ev(self, sel, candidate):
        return self._block_index(sel, candidate)
    
    def _block_index(self, idl, what):
        if idl is None:
            idl = slice(None, None)
        if isinstance(what, slice) or not isinstance(what, Iterable):
            index = (idl, what)
        else:
            index = np.ix_(idl, what)
        return self._candidate_validities[index]

    def candidate_validity(self, candidate):
        return self._candidate_validities[:, candidate]
    
    def predicates2candidates(self, predicates):
        return self._candidate_indexer.predicates2candidates_value[predicates]
    
    def candidates2predicates(self, candidates):
        return self._candidate_indexer.candidates2predicates_value[candidates]
    
    def candidate_all(self, sel, candidates):
        return self._block_index(sel, candidates).all(axis=0)

    def candidate_any(self, sel, candidates):
        return self._block_index(sel, candidates).any(axis=0)

    def select_candidates(self, which):
        candidates = indices_slice(which, self._candidates)
        predicates = self.candidates2predicates(candidates)
        res = ValidityPrincipalMatrix(
            self._predicate_validities,
            effective_validity=self._effective_validity,
            candidates=predicates
        )
        return res
     
    predicate_all = _from_predicates2(candidate_all)
    predicate_any = _from_predicates2(candidate_any)
    get_predicate_supports = _from_predicates2(get_candidate_supports)
    get_predicate_ev = _from_predicates2(get_candidate_ev)



class EffectiveValiditiesTracker:
    '''Holds a tighter copy of the current effective validities''' 

    def __init__(self, predicate_validities, effective_validity):
        # self._validities = np.array(predicate_validities[effective_validity, :], order='F')
        self._validities = predicate_validities[effective_validity, :]
        self._supports = self._validities.sum(axis=0)

    @property
    def validities(self):
        return self._validities
    
    @property
    def ncols(self):
        '''Number of predicates; or equivalently columns'''
        return self._validities.shape[1]
    
    @property
    def nrows(self):
        '''Number of effectively selected instances; or equivalently rows'''
        return self._validities.shape[0]
    
    @property
    def supports(self):
        return self._supports

    def supports_of(self, ev, predicates):
        # cythonise?
        return self._validities[np.ix_(ev, predicates)].sum(axis=0)

    def intersection_support_of(self, predicate, predicates):
        ev = self.validities[:,predicate]
        return self.supports_of(ev, predicates)
    
    def validity_of(self, ev, predicate):
        # cythonise?
        if isinstance(predicate, Iterable):
            index = np.ix_(ev, predicate)
        else:
            index = (ev, predicate)
        return self._validities[index]

    def all(self, cover, predicates):
        return self._validities[np.ix_(cover, predicates)].all(axis=0)

    def any(self, cover, predicates):
        return self._validities[np.ix_(cover, predicates)].any(axis=0)

    
class EffectiveValiditiesView:
    '''Handles validity overlaps adapting the underlying Tracker''' 

    def __init__(self, tracker, ev_full=None, candidates=None, supports=None):
        self._tracker = tracker
        self._ev_full = ev_full if ev_full is not None else np.ones(self._tracker.nrows, bool)
        
        if (candidates is None) ^ (supports is None):
            raise ValueError('How did you compute the supports without knowing the candidates?')
        elif candidates is None:  # and also supports is None
            candidates = np.arange(self._tracker.ncols)
            if ev_full is None:
                supports = self._tracker.supports
            elif self._ev_full.mean() > .5:  # more True than False: better to count the False rows
                supports_false = self._tracker.supports_of(~self._ev_full, candidates)
                supports -= supports_false
            else:
                supports = self._tracker.supports_of(self._ev_full)
            is_support_positive = supports > 0
            candidates = candidates[is_support_positive]
            supports = supports[is_support_positive]
        self._candidates = candidates
        self._supports = supports
        
        self._map_predicate2candidate = Indices.make_inverse_map(candidates, tracker.ncols - 1)
    
    def superset_sequence(self, effective_validity, indices_collection):
        map_orig2merged = []
        all_indices = Indices.union(indices_collection, map_orig2merged=map_orig2merged)
        is_superset = self.all(effective_validity, all_indices)
        is_support_equal = self.supports_of(all_indices) == effective_validity.sum()
        is_equal = is_support_equal & is_superset
        # Split
        supersets = []
        for i in range(len(indices_collection)):
            idx_o2m = map_orig2merged[i]
            idl_o2m = Indices.absolute2logical(idx_o2m, length=len(all_indices))
            idl_o2m_ss = idl_o2m & is_superset 
            seq_indices = all_indices[idl_o2m_ss]
            seq_is_equal = is_equal[idl_o2m_ss]
            ss = Supersets(indices=seq_indices, is_equal=seq_is_equal)
            supersets.append(ss)
            
        return supersets
        
    @classmethod
    def remove_indices(cls, full, removed):
        '''Remove a (sorted) array of indices from another (sorted) array of indices'''
        # TODO: faster
        restricted = sorted(set(full) - set(removed))
        return np.array(restricted, int)
        
    def make_view(self, ev, candidates):
        
        evt_view = EffectiveValiditiesView(self, ev, candidates)
        return evt_view
    
    def map_predicate2candidate(self, predicates):
        '''convert predicates indices into a local index within the currently used candidates''' 
        return self._map_predicate2candidate[predicates]
        
    def restrict_effective_validity(self, ev, predicates=None, drop_empty=True):
        '''Create a new view with a restricted effective validity (and only the surviving candidates)'''
        if predicates is not None:
            idx_pred = self.map_predicate2candidate(predicates)
            is_kept = Indices.logical(idx_pred, len(self._candidates))
            candidates_new = self._candidates[is_kept]
            supports_old = self._supports[is_kept]
        else:
            candidates_new = self._cadidates
            supports_old = self._supports
        
        ev_new = Indices.merge_logicals(self._ev_full, ev)
        if ev.mean() > .5:  # more True than False: better to count the newly False rows
            ev_false = Indices.merge_logicals(self._ev_full, ~ev)
            supports_false = self._tracker.supports_of(ev_false, candidates_new)
            supports_new = supports_old - supports_false
        else:
            supports_new = self._tracker.supports_of(ev_new, candidates_new)
        if drop_empty:
            is_support_positive = supports_new > 0
            candidates_new = candidates_new[is_support_positive]
            supports_new = supports_new[is_support_positive]
        return EffectiveValiditiesView(tracker=self._tracker,
                                       ev_full=ev_new,
                                       candidates=candidates_new,
                                       supports=supports_new)
    
    def restrict_candidates(self, predicates):
        '''Create a new view with a restricted candidate set'''
        is_selected = np.zeros(len(self._candidates), bool)
        
        candidates = self.map_predicate2candidate(predicates)
        is_selected[candidates] = True
        supports_new = self._supports[is_selected]
        candidates_new = self._candidates[is_selected]  # this ensures candidates will remain sorted. (provides O(n) sorting)
        assert np.all(self._candidates[is_selected] == np.array(sorted(predicates), int)), 'selection failed'
        return EffectiveValiditiesView(tracker=self._tracker,
                                       ev_full=self._ev_full,
                                       candidates=candidates_new,
                                       supports=supports_new)

    @property
    def candidates(self):
        return self._candidates
    
    def supports(self):
        '''Return candidate supports'''
        return self._supports
    
    @property
    def minsup_predicate(self):
        '''Predicate (index) with the minimum support among the candidates'''
        return self.candidates[np.argmin(self.supports)]
        
    def supports_of(self, predicates):
        candidates = self._map_predicate2candidate[predicates]
        return self._supports[candidates]

    def all(self, cover, predicates):
        cover_full = Indices.merge_logicals(self._ev_full, cover)
        res = self._tracker.all(cover_full, predicates)
        assert np.all(res == self._tracker._validities[np.ix_(cover_full, predicates)].all(axis=0)), 'all failed'
        return res

    def any(self, cover, predicates):
        cover_full = Indices.merge_logicals(self._ev_full, cover)
        return self._tracker.all(cover_full, predicates)

    def validity_of(self, predicate):
        return self._tracker.validity_of(self._ev_full, predicate)
    

class Supersets(NamedTuple):
    indices: ndarray
    is_equal: ndarray

    @property
    def equal(self):
        return self.indices[self.is_equal]
    
    @property
    def proper(self):
        return self.indices[~self.is_equal]

    def __repr__(self):
        return ' '.join(f'{i}{"Pe"[e]}' for i, e in 
                        zip(self.indices, self.is_equal))
        
    def __bool__(self): return len(self.indices)!=0

class SupersetCollection(NamedTuple):
    extension: Supersets
    closure: Supersets
    blacklist: Supersets

    def __repr__(self):
        return f'<{self.__class__.__name__}:\n E:{self.extension}\n C:{self.closure}\n B:{self.blacklist}\n>'

       
class CandidateRestriction:
    '''Restricting the remaining superset candidates
    
    Serves as a state furing the refinement creation of closure languages. 
    
    :param extension ndarray: the indices of the remaining extensions to consider
    :param closure: the indices of the potential closures each of the extensions might have
    :param blacklist ndarray: the indices of the active blacklisted predicates (i.e. potential supersets)
    '''

    def __init__(self, ev_view, extension:ndarray, closure:ndarray, blacklist:ndarray):
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
        
