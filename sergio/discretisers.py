'''
Created on Aug 9, 2018

@author: janis
'''

import numpy as np
from math import inf
from typing import NamedTuple, Tuple, Optional, Sequence, List 
from collections import namedtuple
import enum

import colito.matlab as mtl

from colito.summaries import Summarisable, SummaryOptions
from colito.resolvers import make_enum_resolver

class Minimisers:
    class IntResult(NamedTuple):
        x:int
        value:float
    
    @classmethod
    def sequence_bisect(cls, fn, a, b):
        '''Integer sequence (derivative-less) bisection search method.
        
        @param fn Callable[int,float]: The function to minimise. Must be a quasiconvex function.
        @param a int: First valid integer to search (inclusive)
        @param b int: Last valid integer to search (inclusive)
        @return IntResult: The result of the minimisation 
        '''
        mid = lambda a,b:int(round((a+b)/2))
        if b==a:
            return cls.IntResult(a,fn(a))
        fa = fn(a)
        fb = fn(b)
        m = fm = None
        while b-a>2:
            if m is None:
                m = mid(a,b)
                fm = fn(m)
            #print(f'IBISECT: {a}={fa},{m}={fm},{b}={fb}')
            if fm>fa:
                b = m
                fb = fm
                m = None
            elif fm>fb:
                a = m
                fa = fm
                m = None
            else: # fm<fa and fm<fb or f is a non q.convex function
                l = mid(a,m)
                fl = fn(l)
                if fl < fm: # use a l m
                    b = m
                    m = fl
                    fb = fm
                    fm = fl
                else: # fl>fm and fl<fa or f is a non q.convex function
                    # use l X b
                    a = l
                    fa = fl
                    m = None
        if m!=mid(a,b):
            m = mid(a,b)
            fm = fn(m)
        #print(f'IBISECT: {a}={fa},{m}={fm},{b}={fb}')
        values = [fa,fm,fb]
        idx_min = np.argmin(values)
        return cls.IntResult(x=[a,m,b][idx_min], value=values[idx_min])

class Interval(NamedTuple):
    lower:float
    upper:float
    leq: bool
    ueq: bool
    name:str
    
    @property
    def range(self):
        '''The range of this interval. Cannot be None.'''
        return (
            self.lower if self.lower is not None else -inf,
            self.upper if self.upper is not None else inf
            )
        
    @property
    def bounds(self):
        '''The range of this interval. Non existent bounds can be None.'''
        return (self.lower, self.upper)
    
    def __repr__(self):
        rng = self.range
        return f'{self.name}:{"[" if self.leq else "("}{rng[0]},{rng[1]}{"]" if self.ueq else ")"}'

class DiscretiserRanges(enum.Flag):
    INTERVALS = enum.auto()
    SLABS_POSITIVE = enum.auto()
    SLABS_NEGATIVE = enum.auto()
    SLABS = SLABS_NEGATIVE | SLABS_POSITIVE

DISCRETISER_RANGE_RESOLVER = make_enum_resolver(DiscretiserRanges, allow_multiple=True, collapse_multipart=True)

class Discretiser(Summarisable):
    
    
    default_names = {
        2:("low", "high"),
        3:("low", "med", "high"),
        4:("v.low", "low", "high", "v.high"),
        5:("v.low", "low", "med", "high", "v.high")
    }
    level_tag = 'LVL'
    
    def __init__(self, cut_count: int=3, ranges:DiscretiserRanges=None) -> None:
        self._cut_count: int = cut_count
        self._ranges = DISCRETISER_RANGE_RESOLVER.resolve(ranges) if ranges is not None else DiscretiserRanges.INTERVALS
        
    cut_count = property(lambda self:self._cut_count, None, 'The number of cuts to use.')
    ranges = property(lambda self:self._ranges, None, 'The type of ranges to create.')
    
    def discretise(self, data: np.ndarray) -> List['Interval']:
        raise NotImplementedError()
    
    def level_names(self, level_count: int) -> Tuple[str, ...]:
        null = object()
        decorators = self.default_names.get(level_count, null)
        if decorators is null:
            decorators = tuple(f'{self.level_tag}:{i+1}/{level_count}' for i in range(level_count))
        return decorators

    def cuts2levels(self, cuts:np.ndarray) -> List['Interval']:
        cuts = list(map(float,cuts))
        levels = []
        def mk_level(rng,leq,ueq,name):
            lower,upper = rng
            if lower is None:
                lower = None
                leq = False
            if upper is None:
                upper = None
                ueq = False
            return Interval(lower=lower,upper=upper,leq=leq,ueq=ueq,name=name)
        if len(cuts) == 1:
            cut = cuts[0]
            names = self.level_names(2)
            levels = [Interval(lower=None,upper=cut, leq=True,ueq=False,name=names[0]),
                      Interval(lower=cut,upper=None, leq=True,ueq=False,name=names[1])]
            return levels
        if self.ranges & DiscretiserRanges.INTERVALS:
            breaks = [None]+cuts+[None]
            names = self.level_names(len(breaks)-1)
            levels += list(mk_level(r, True,False, names[idx]) for idx,r
                                    in enumerate(zip(breaks[:-1], breaks[1:])))
        if self.ranges & DiscretiserRanges.SLABS_NEGATIVE:
            names = self.level_names(len(cuts)+1)
            levels += list(mk_level(r,False, True, names[idx]) for idx,r
                           in enumerate(zip([None]*len(cuts), cuts)))
        if self.ranges & DiscretiserRanges.SLABS_POSITIVE:
            names = self.level_names(len(cuts)+1)
            levels += list(mk_level(r,True, False, names[idx]) for idx,r
                           in enumerate(zip(cuts, [None]*len(cuts))))
        return levels
    
    def summary_dict(self, options:SummaryOptions):
        dct = self.summary_from_fields(('cut_count',))
        dct['ranges'] = DISCRETISER_RANGE_RESOLVER.flag2str(self.ranges)
        return dct
    
    def __repr__(self):
        return f'<{self.__class__.__name__}:{self.cut_count} cuts of type {self.ranges!s}>'
    
    def __str__(self):
        return f'<{self.__class__.__name__} C:{self.cut_count} R:{DISCRETISER_RANGE_RESOLVER.flag2str(self.ranges)}>'

class FrequencyDiscretiser(Discretiser):
    level_tag = 'QNT'
    def __init__(self, *args, midpoints:bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._midpoints = midpoints
        
    midpoints = property(lambda self:self._midpoints,None,'Whether the cuts hould be shifted to the mean between consecutive available data points.')
        
    State = namedtuple('State',('num_elements','num_unique','cum_count_unique'))
    @classmethod
    def cuts2probs(cls, cuts:np.ndarray, state:'FrequencyDiscretiser.State') -> np.ndarray:
        cdfs_inner = state.cum_count_unique[cuts]/state.num_elements
        cdfs = np.concatenate(([0],cdfs_inner,[1]))
        probs = np.diff(cdfs)
        return probs
    
    @classmethod
    def entropy(cls, cuts, state:'FrequencyDiscretiser.State', eps=None) -> float:
        if eps is None:
            eps = 1/state.num_elements*1e-2
        probs = cls.cuts2probs(cuts, state=state)
        is_zero = probs==0 
        num_zeros = sum(is_zero)
        probs_disc = (probs + eps*is_zero)/(1+num_zeros*eps)
        ent = -np.sum(probs_disc*np.log(probs_disc))
        return ent
    
    @classmethod
    def modcuts(cls, cuts, index, value, state:'FrequencyDiscretiser.State', step=False) -> np.ndarray:
        cuts = cuts.copy()
        cuts[index] = cuts[index]+value if step else value
        if cuts[index] > state.num_unique-1 or cuts[index] <= 0: # Allow greater or equal slabs. Cut on the last index is allowed.
            return None
        for idx_r in range(index+1,len(cuts)):
            if cuts[idx_r] < cuts[idx_r-1]:
                cuts[idx_r] = cuts[idx_r-1]
            else:
                break
        for idx_l in range(index,0,-1):
            if cuts[idx_l] < cuts[idx_l-1]:
                cuts[idx_l-1] = cuts[idx_l]
            else:
                break
        return cuts

    Candidate = namedtuple('Candidate',('cuts','entropy'))
    
    @classmethod
    def best_cut(cls,cuts, index, state:'FrequencyDiscretiser.State'):
        def fn(value):
            cuts_new = cls.modcuts(cuts,index,value,state=state,step=False)
            #print(f'index={index}, value={value}, cuts={cuts}, cuts_new={cuts_new}')
            if cuts_new is None:
                return inf
            ent = cls.entropy(cuts_new,state=state)
            return -ent
        res =  Minimisers.sequence_bisect(fn,1,state.num_unique-1)
        best_value = res.x
        return cls.modcuts(cuts,index,best_value,state=state,step=False)

    def discretise(self, data:  np.ndarray) -> List['Discretiser.Interval']:
        cls = self.__class__
        trail = []
        
        unq = mtl.unique(data)
        cnt_unq = np.bincount(unq.entry_labels)
        cum_cnt_unq = np.concatenate(([0],np.cumsum(cnt_unq)))
        
        state = cls.State(
            num_elements=len(data),
            num_unique=len(unq.unique_entries),
            cum_count_unique=cum_cnt_unq
            )
        if self.cut_count >= state.num_unique-1:
            cuts = np.arange(state.num_unique-1)+1
        else:
            quantiles = np.linspace(0,1,self.cut_count+2)[1:-1]
            cuts = np.searchsorted(cum_cnt_unq,quantiles)
            
            while True:
                candidates = []
                for i in range(self.cut_count):
                    cuts_new = cls.best_cut(cuts,i, state=state)
                    ent_new = cls.entropy(cuts_new, state=state)
                    candidates.append(cls.Candidate(cuts_new,ent_new))
                idx_max = np.argmax(list(candidate.entropy for candidate in candidates))
                best_candidate = candidates[idx_max]
                # print('Current candidates:\n {0}'.format('\n '.join(f'{idx}:{c}' for idx,c in enumerate(candidates))))
                if np.all(best_candidate.cuts == cuts):
                    break
                cuts = best_candidate.cuts
                trail.append(best_candidate)
        if self.midpoints:
            values = (unq.unique_entries[cuts] + unq.unique_entries[cuts+1])/2
        else:
            values = unq.unique_entries[cuts]
        levels = self.cuts2levels(values)
        return levels

    def summary_dict(self, options:SummaryOptions):
        dct = super().summary_dict(options)
        dct['midpoints'] = self.midpoints
        return dct


class DiscretiserFixed(Discretiser):
    level_tag = 'FXT'

    def __init__(self, ranges: Sequence[Tuple[float, float]], level_names: Optional[Sequence[str]]=None) -> None:
        super(DiscretiserFixed, self).__init__(level_count=len(ranges))
        if level_names is None:
            level_names = self.level_names()
        self._level_names: Tuple[str, ...] = tuple(level_names)
        self._ranges: Tuple[Tuple[float, float], ...] = tuple(ranges)
        assert len(self._level_names) == len(self._ranges), ValueError("Intervals must be as many as the ranges.")
        
    def discretise(self, data: np.ndarray) -> Tuple[Tuple[float, float], ...]:
        return self._ranges

class DiscretiserUniform(Discretiser):
    level_tag = 'UNI'
    
    def discretise(self, data: np.ndarray) -> Tuple[Tuple[float, float], ...]:
        '''Discretise a given series into num_levels bins.'''
        min_: float = min(data[~np.isinf(data)])
        max_: float = max(data[~np.isinf(data)])
        cuts: np.ndarray = np.linspace(min_, max_, self.cut_count + 2, dtype=float)[1:-1]
        intervals = self.cuts2levels(cuts)
        return intervals



DEFAULT_DISCRETISER = FrequencyDiscretiser(cut_count=5, ranges=DiscretiserRanges.SLABS_POSITIVE)

