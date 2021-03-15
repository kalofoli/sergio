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

from colito.collection import ClassCollection
from colito.summaries import Summarisable, SummaryOptions, SummarisableList,\
    DEFAULT_SUMMARY_OPTIONS
from colito.resolvers import make_enum_resolver
from colito.factory import resolve_arguments, NoConversionError

from builtins import staticmethod, property, classmethod
from collections import namedtuple
from pandas.core.series import Series
from colito.collection import ClassCollectionRegistrar

from sergio.language import Selector
from sergio.data import EntityAttributes


MEASURES = ClassCollection('Measures')
OPTIMISTIC_ESTIMATORS = ClassCollection('OptimisticEstimators')


class Measure(Summarisable, ClassCollectionRegistrar):
    '''Abstract measure class'''
    __collection_tag__ = None
    __collection_factory__ = MEASURES
    
    def __init__(self, data: EntityAttributes) -> None:
        self._data = data

    def evaluate(self, selector: Selector) -> float:
        '''Evaluate the measure on a given selector'''
        raise NotImplementedError()

    def summary_dict(self, options:SummaryOptions):
        return dict()
    
    @property
    def summary_name(self):
        return f'meas-{self.__class__.tag}'
    
    @property
    def data(self):
        '''The data associated with this Measure'''
        return self._data
    
    def __repr__(self):
        dct = self.summary_dict(DEFAULT_SUMMARY_OPTIONS)
        params_txt = ','.join(f'{key}={value}' for key,value in dct.items())
        return f'<{self.__class__.__name__}({params_txt})>'

    @classmethod
    def parse_argument(cls, name, value, parameter):
        raise NoConversionError()

    @classmethod
    def make_from_strings(cls, name, *args, **kwargs):
        measure_cls = MEASURES.tags[name]
        args_p, kwargs_p = resolve_arguments(measure_cls.__init__, args, kwargs, handler=measure_cls.parse_argument)
        measure = measure_cls(*args_p[1:], **kwargs_p)
        return measure
        

class OptimisticEstimator(Summarisable, ClassCollectionRegistrar):
    '''Abstract Optimistic Estimator class'''
    __collection_tag__ = None
    __collection_factory__ = OPTIMISTIC_ESTIMATORS
    
    def __init__(self, data: EntityAttributes) -> None:
        self._data = data

    def evaluate(self, selector: Selector) -> float:
        '''Evaluate an upper bound for the values of all refinements of the given selector'''
        raise NotImplementedError()

    def summary_dict(self, options:SummaryOptions):
        return dict()
    
    @property
    def summary_name(self):
        return f'oest-{self.__class__.tag}'
    
    @property
    def data(self) -> EntityAttributes:
        '''The data associated with this OptimisticEstimator'''
        return self._data

    @classmethod
    def parse_argument(cls, name, value, parameter):
        raise NoConversionError()

    @classmethod
    def make_from_strings(cls, name, *args, **kwargs):
        oest_cls = OPTIMISTIC_ESTIMATORS.tags[name]
        args_p, kwargs_p = resolve_arguments(oest_cls.__init__, args, kwargs, handler=oest_cls.parse_argument)
        oest = oest_cls(*args_p[1:], **kwargs_p)
        return oest

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

class MeasureCoverage(CachingScoreMixin, Measure):
    __collection_title__ = 'Coverage'
    __collection_tag__ = 'coverage'

    def evaluate_uncached(self, selector: Selector) -> float:
        idl: np.ndarray = selector.validity
        return idl.mean()

class OptimisticEstimatorCoverage(CachingScoreMixin, OptimisticEstimator):
    __collection_title__ = 'Coverage'
    __collection_tag__ = 'coverage'
    
    def evaluate_uncached(self, selector: Selector) -> float:
        idl: np.ndarray = selector.validity
        return idl.mean()
