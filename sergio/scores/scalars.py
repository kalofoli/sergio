'''
Created on Mar 9, 2021

@author: janis
'''

import numpy as np
import pandas as pd

from sergio.language import Selector
from sergio.data import EntityAttributes, EntityAttributesWithAttributeTarget
from sergio.scores import Measure, OptimisticEstimator, CachingScoreMixin
from colito.summaries import SummarisableFromFields, SummaryFieldsAppend
from types import SimpleNamespace
from colito.collection import CollectionMemberWithKeywordResolver
from colito import NamedUniqueConstant



_UNSET = NamedUniqueConstant('Unset')
class ScalarEvaluatorMixin(SummarisableFromFields, CollectionMemberWithKeywordResolver):
    __summary_fields__ = SummaryFieldsAppend(('target_name'))
    def __init__(self, *args, target_data, target_name, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_data: np.ndarray = np.array(target_data)
        self._target_name: str = target_name
    
    @property
    def data(self):
        '''The data associated with this Measure'''
        return self._data
    @property
    def target_data(self) -> np.ndarray: return self._target_data
    @property
    def target_name(self) -> str: return self._target_name
    
    @classmethod
    def from_components(cls, dataset, target=None, **kwargs):
        if target is None:
            if isinstance(dataset, EntityAttributesWithAttributeTarget):
                target_data = dataset.target_data
                target_name = dataset.target_name
            else:
                raise TypeError(f'No target specified and data has no default.')
        else:
            target_data = dataset.lookup_attribute(target).data
        return cls(target_data=target_data, target_name=target_name, **kwargs)
    
    @classmethod
    def __kwargs_preprocess__(cls, resolver, dataset=_UNSET, target_data=_UNSET, target_name=_UNSET, **kwargs):
        if target_data is _UNSET or target_name is _UNSET:
            if dataset is _UNSET:
                dataset = resolver('dataset', None)
            if isinstance(dataset, EntityAttributesWithAttributeTarget):
                target_data = dataset.target_data if target_data is _UNSET else target_data
                target_name = dataset.target_name if target_name is _UNSET else target_name
            else:
                raise TypeError(f'No target specified and data has no default.')
        kwargs['target_data'] = target_data
        kwargs['target_name'] = target_name
        return kwargs
    
class ScalarMeasure(ScalarEvaluatorMixin, Measure):
    __collection_tag__ = None
class ScalarOptimisticEstimator(ScalarEvaluatorMixin, OptimisticEstimator):
    __collection_tag__ = None

class JaccardScoreMixin(CachingScoreMixin):
    __collection_title__ = 'Jaccard'
    __summary_fields__ = ('target_name',)
    def __init__(self, target_data, target_name:str) -> None:
        super().__init__(target_data=target_data, target_name=target_name)
        if self.target_data.dtype != bool:
            raise TypeError(f"Target must be of type bool, but targt {self.target_name} is of type {self.target_data.dtype}.")
        self._target_size = self.target_data.sum()
    
class MeasureJaccard(JaccardScoreMixin, ScalarMeasure):
    __collection_tag__ = 'jaccard'
    
    def evaluate_uncached(self, selector: Selector) -> float:
        idl: np.ndarray = selector.validity
        target: np.ndarray = self._target_data
        
        return np.sum(idl & target)/np.sum(idl | target)
    
    
class OptimisticEstimatorJaccard(JaccardScoreMixin, ScalarOptimisticEstimator):
    __collection_tag__ = 'jaccard'
    
    def evaluate_uncached(self, selector: Selector) -> float:
        idl: np.ndarray = selector.validity
        target: np.ndarray = self._target_data
        
        return np.sum(idl & target)/self._target_size
    

class CoverageMeanShiftScoreMixin(CachingScoreMixin):
    __collection_title__ = 'Coverage times (positive) mean shift'
    __collection_tag__ = None
    __summary_fields__ = ['coverage_exponent']
    def __init__(self, target_data, target_name, coverage_exponent:float=1.) -> None:
        super().__init__(target_data=target_data, target_name = target_name)
        self._coverage_exponent:float = coverage_exponent
        max_exp = max(coverage_exponent, 1)
        self._exps = SimpleNamespace(coverage=coverage_exponent/max_exp, shift=1./max_exp)
        self._population_mean = target_data.mean()
    @property
    def coverage_exponent(self) -> float: return self._coverage_exponent
    
class MeasureCoverageMeanShift(CoverageMeanShiftScoreMixin, ScalarMeasure):
    __collection_tag__ = 'coverage-mean-shift'
    def evaluate_uncached(self, selector: Selector) -> float:
        idl: np.ndarray = selector.validity
        target: np.ndarray = self._target_data
        m,n = np.sum(idl), len(idl)
        if m:
            coverage = m/n
            shift = target[idl].mean() - self._population_mean
            value = coverage**self._exps.coverage*shift**self._exps.shift
        else:
            value = 0
        return value
    
class OptimisticEstimatorCoverageMeanShift(CoverageMeanShiftScoreMixin, ScalarOptimisticEstimator):
    __collection_tag__ = 'coverage-mean-shift'
    
    def evaluate_uncached(self, selector: Selector) -> float:
        idl: np.ndarray = selector.validity
        n = len(idl)
        y = self.target_data
        y_sel = y[idl]
        p = np.argsort(y_sel)[::-1]
        y_sel_run = np.cumsum(y_sel[p])
        m = idl.sum()
        m_run = np.arange(1,m+1)
        coverages = (m_run/n)**self._exps.coverage
        shifts = (y_sel_run/m_run - self._population_mean)**self._exps.shift
        values = np.r_[0,coverages*shifts]
        idx_max = np.argmax(values)
        value = values[idx_max]
        return value

class OptimisticEstimatorCoverageMeanShiftGroupped(CoverageMeanShiftScoreMixin, ScalarOptimisticEstimator):
    __collection_tag__ = 'coverage-mean-shift-groupped'
    def __init__(self, validities, target_data, target_name, coverage_exponent:float=1.) -> None:
        super().__init__(target_data=target_data, target_name = target_name, coverage_exponent=coverage_exponent)
        self._validities = validities
        
    @property
    def validities(self): return self._validities
        
    def evaluate_uncached(self, selector: Selector) -> float:
        return self._evaluate_raw(selector.validity)

    def _evaluate_raw(self, idl_sel, full=False) -> float:
        idx_group, grp_size = self.get_group_index(idl_sel)
        y_sel = self.target_data[idl_sel]
        grp_sum = np.bincount(idx_group, y_sel)
        p = np.argsort(grp_sum/grp_size)[::-1]
        y_grp_run, m_grp_run = np.cumsum(grp_sum[p]), np.cumsum(grp_size[p])
        
        n = len(idl_sel)
        coverages = (m_grp_run/n)**self._exps.coverage
        shifts = (y_grp_run/m_grp_run - self._population_mean)**self._exps.shift
        values = coverages*shifts
        f = max(0, values.max())
        if full:
            return values
        else:
            return float(f)

    def get_group_index(self, idl_sel):
        V_full = self._validities
        idl_nonred = np.where(V_full[idl_sel,:].sum(0) < idl_sel.sum())[0]
        C = np.packbits(V_full[np.ix_(idl_sel,idl_nonred)],axis=1)
        
        C_unq, map_sel2grp, grp_cnt = np.unique(C,axis=0, return_inverse=True, return_counts=True)
        np.testing.assert_array_equal(C_unq[map_sel2grp,:], C, 'Mapping unique to selection')
        return map_sel2grp, grp_cnt
        
    def __get_groups_deprecated(self, idl_sel, index_local=True):
        n = len(idl_sel)
        V_full = self._validities
        idl_nonred = np.where(V_full[idl_sel,:].sum(0) < idl_sel.sum())[0]
        map_sel2full = np.arange(n)[idl_sel]
        C = np.packbits(V_full[np.ix_(idl_sel,idl_nonred)],axis=1)
        
        C_unq, map_sel2grp, grp_cnt = np.unique(C,axis=0, return_inverse=True, return_counts=True)
        map_seg2sel = np.argsort(map_sel2grp)
        np.testing.assert_array_equal(C_unq[map_sel2grp,:], C, 'Mapping unique to selection')
        # Since sorted, the segment ids will be consequtive
        
        if index_local==False:
            map_seg2full = map_sel2full[map_seg2sel]
            seg_groups = np.split(map_seg2full, np.cumsum(grp_cnt)[:-1])
        else:
            seg_groups = np.split(map_seg2sel, np.cumsum(grp_cnt)[:-1])
        return seg_groups
        
        
    @classmethod
    def __kwargs_preprocess__(cls, kwarg_resolver, language=_UNSET, validities=_UNSET, **kwargs):
        kwargs = super().__kwargs_preprocess__(kwarg_resolver, **kwargs)
        if validities is _UNSET:
            if language is _UNSET:
                language = kwarg_resolver('language')
            validities = language.predicate_validities
        kwargs['validities'] = validities
        return kwargs
