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



class ScalarEvaluatorMixin(SummarisableFromFields):
    __summary_fields__ = SummaryFieldsAppend(('target_name'))
    def __init__(self, *args, target=None, **kwargs):
        super().__init__(*args, **kwargs)
        if target is None:
            if isinstance(self.data, EntityAttributesWithAttributeTarget):
                target_data = self.data.target_data()
            else:
                raise TypeError(f'No target specified and data has no default.')
        else:
            target_data = self.data.lookup_attribute(target).data
        self._target_data = target_data
    @property
    def target_data(self) -> pd.Series: return self._target_data
    @property
    def target_name(self) -> str: return self._target_data.name
    
class ScalarMeasure(ScalarEvaluatorMixin, Measure):
    __collection_tag__ = None
class ScalarOptimisticEstimator(ScalarEvaluatorMixin, OptimisticEstimator):
    __collection_tag__ = None

class JaccardScoreMixin(CachingScoreMixin):
    __collection_title__ = 'Jaccard'
    __summary_fields__ = ('target',)
    def __init__(self, data: EntityAttributes, target:str=None) -> None:
        super().__init__(data=data, target=target)
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
    __summary_fields__ = ['gamma']
    def __init__(self, data: EntityAttributes, gamma:float=1., target=None) -> None:
        super().__init__(data=data, target=target)
        self._gamma:float = gamma
    @property
    def gamma(self) -> float: return self._gamma
    
class MeasureCoverageMeanShift(CoverageMeanShiftScoreMixin, ScalarMeasure):
    __collection_tag__ = 'coverage-mean-shift'
    def evaluate_uncached(self, selector: Selector) -> float:
        idl: np.ndarray = selector.validity
        target: np.ndarray = self._target_data.values
        
        return np.sum(idl & target)/np.sum(idl | target)
    
class OptimisticEstimatorCoverageMeanShift(CoverageMeanShiftScoreMixin, ScalarOptimisticEstimator):
    __collection_tag__ = 'coverage-mean-shift'
    def evaluate_uncached(self, selector: Selector) -> float:
        raise NotImplementedError()
        idl: np.ndarray = selector.validity
        target: np.ndarray = self._target_data.values
        return np.sum(idl & target)/self._target_size

