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



class ScalarEvaluatorMixin(SummarisableFromFields):
    __summary_fields__ = SummaryFieldsAppend(('target_name'))
    def __init__(self, *args, target_data, target_name, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_data: np.ndarray = target_data
        self._target_name: str = target_name
    
    @property
    def data(self):
        '''The data associated with this Measure'''
        return self._data
    @property
    def target_data(self) -> pd.Series: return self._target_data
    @property
    def target_name(self) -> str: return self._target_name
    
    @classmethod
    def from_dataset(cls, dataset, target=None, **kwargs):
        if target is None:
            if isinstance(dataset, EntityAttributesWithAttributeTarget):
                target_data = dataset.target_data
                target_name = dataset.target_name
            else:
                raise TypeError(f'No target specified and data has no default.')
        else:
            target_data = dataset.lookup_attribute(target).data
        return cls(target_data=target_data, target_name=target_name, **kwargs)
    
    
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
        target: np.ndarray = self._target_data.values
        
        return np.sum(idl & target)/np.sum(idl | target)
    
    
class OptimisticEstimatorJaccard(JaccardScoreMixin, ScalarOptimisticEstimator):
    __collection_tag__ = 'jaccard'
    
    def evaluate_uncached(self, selector: Selector) -> float:
        idl: np.ndarray = selector.validity
        target: np.ndarray = self._target_data.values
        
        return np.sum(idl & target)/self._target_size
    

class CoverageMeanShiftScoreMixin(CachingScoreMixin):
    __collection_title__ = 'Coverage times (positive) mean shift'
    __collection_tag__ = None
    __summary_fields__ = ['coverage_exponent']
    def __init__(self, target_data, target_name, coverage_exponent:float=1., target=None) -> None:
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

