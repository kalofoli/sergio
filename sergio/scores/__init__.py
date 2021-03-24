'''
Measures

Created on Jan 8, 2018

@author: janis
'''
# pylint: disable=bad-whitespace

from typing import Optional, Type, Callable, NamedTuple, Iterable
from math import inf, ceil

import re
import enum

import numpy as np
from numpy import ndarray

from colito.collection import ClassCollection, ClassCollectionFactoryRegistrar
from colito.summaries import Summarisable, SummaryOptions, SummarisableList,\
    DEFAULT_SUMMARY_OPTIONS, SummarisableFromFields
from colito.resolvers import make_enum_resolver

from builtins import staticmethod, property, classmethod
from collections import namedtuple
from pandas.core.series import Series
from colito.collection import ClassCollectionRegistrar

from sergio.language import Selector
from sergio.data import EntityAttributes


MEASURES = ClassCollection('Measures')
OPTIMISTIC_ESTIMATORS = ClassCollection('OptimisticEstimators')


class Measure(Summarisable, ClassCollectionFactoryRegistrar):
    '''Abstract measure class'''
    __collection_tag__ = None
    __collection_factory__ = MEASURES
    
    def evaluate(self, selector: Selector) -> float:
        '''Evaluate the measure on a given selector'''
        raise NotImplementedError()

    @property
    def __summary_name__(self):
        return f'meas-{self.__class__.tag}'
    
    @classmethod
    def from_kwargs(cls, **kwargs):
        super().__init__()

class OptimisticEstimator(Summarisable, ClassCollectionFactoryRegistrar):
    '''Abstract Optimistic Estimator class'''
    __collection_tag__ = None
    __collection_factory__ = OPTIMISTIC_ESTIMATORS
    
    def evaluate(self, selector: Selector) -> float:
        '''Evaluate an upper bound for the values of all refinements of the given selector'''
        raise NotImplementedError()

    @property
    def __summary_name__(self):
        return f'oest-{self.__class__.tag}'
    

class CachingScoreMixin:
    '''Caching framework for selector-based measures and optimistic estimators'''
    
    __cache_variable_prefix__ = "__cache"  # Override this
    
    @classmethod
    def selector_property_name(cls):
        '''Selector property name under which the value is cached''' 
        return f"{cls.__cache_variable_prefix__}_{cls.__name__}"
    
    def evaluate_uncached(self, selector) -> float:
        '''Perform actual computation'''
        raise NotImplementedError()
    
    def evaluate(self, selector: Selector) -> float:
        '''Memoise the evaluation'''
        cls: Type[CachingScoreMixin] = self.__class__
        selector_property_name = cls.selector_property_name()
        val: float = selector.cache.get(selector_property_name, None)
        if val is None:
            val = self.evaluate_uncached(selector)
            selector.cache[selector_property_name] = val
        return val

    @classmethod
    def cache_isset(cls, selector: Selector) -> bool:
        return cls.selector_property_name() in selector.cache

    @classmethod
    def set_cache(cls, selector: Selector, value: float) -> None:
        selector.cache[cls.selector_property_name()] = value

class MeasureCoverage(CachingScoreMixin, SummarisableFromFields, Measure):
    __collection_title__ = 'Coverage'
    __collection_tag__ = 'coverage'

    def evaluate_uncached(self, selector: Selector) -> float:
        idl: np.ndarray = selector.validity
        return idl.mean()

class OptimisticEstimatorCoverage(CachingScoreMixin, SummarisableFromFields, OptimisticEstimator):
    __collection_title__ = 'Coverage'
    __collection_tag__ = 'coverage'
    
    def evaluate_uncached(self, selector: Selector) -> float:
        idl: np.ndarray = selector.validity
        return idl.mean()
