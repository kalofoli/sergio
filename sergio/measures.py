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

from colito.indexing import ClassCollection
from colito.summarisable import Summarisable, SummaryOptions, SummarisableList
from colito.resolvers import make_enum_resolver
from colito.factory import resolve_arguments, NoConversionError

from .language import Selector
from builtins import staticmethod, property, classmethod
from collections import namedtuple
from pandas.core.series import Series


class GraphData: pass

class MeasureKind(enum.Enum):
    '''Available measures to use'''
    AVERAGE_CORENESS = enum.auto


class Exponents(NamedTuple):
    gamma : float
    common: float
    density: float
    coverage: float

class Measure(Summarisable):
    '''Abstract measure class'''

    def __init__(self, data) -> None:
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
        dct = self.summary_dict(SummaryOptions.default())
        params_txt = ','.join(f'{key}={value}' for key,value in dct.items())
        return f'<{self.__class__.__name__}({params_txt})>'

    @classmethod
    def parse_argument(cls, name, value, parameter):
        raise NoConversionError()

    @classmethod
    def make_from_strings(cls, name, *args, **kwargs):
        measure_cls = MEASURES.get_class_from_tag(name)
        args_p, kwargs_p = resolve_arguments(measure_cls.__init__, args, kwargs, handler=measure_cls.parse_argument)
        measure = measure_cls(*args_p[1:], **kwargs_p)
        return measure
        

class OptimisticEstimator(Summarisable):
    '''Abstract Optimistic Estimator class'''

    def __init__(self, data: GraphData) -> None:
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
    def data(self) -> GraphData:
        '''The data associated with this OptimisticEstimator'''
        return self._data

    def __repr__(self):
        dct = self.summary_dict(SummaryOptions.default())
        params_txt = ','.join(f'{key}={value}' for key,value in dct.items())
        return f'<{self.__class__.__name__}({params_txt})>'

    @classmethod
    def parse_argument(cls, name, value, parameter):
        raise NoConversionError()

    @classmethod
    def make_from_strings(cls, name, *args, **kwargs):
        oest_cls = OPTIMISTIC_ESTIMATORS.get_class_from_tag(name)
        args_p, kwargs_p = resolve_arguments(oest_cls.__init__, args, kwargs, handler=oest_cls.parse_argument)
        oest = oest_cls(*args_p[1:], **kwargs_p)
        return oest

class CachingEvaluator:
    '''Caching framework for selector-based measures and optimistic estimators'''
    
    __evaluator_type__ = "evaluator"  # Override this
    
    @classmethod
    def selector_property_name(cls):
        '''Selector property name under which the value is cached''' 
        return f"{cls.__name__}_{cls.__evaluator_type__}"
    
    def evaluate_uncached(self, selector) -> float:
        '''Perform actual computation'''
        raise NotImplementedError()
    
    def evaluate(self, selector: Selector) -> float:
        '''Memoise the evaluation'''
        cls: Type[CachingEvaluator] = self.__class__
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


class CachingMeasure(CachingEvaluator, Measure):
    '''Abstract measure class'''
    __evaluator_type__ = "measure"  # for use in cache property naming
    
    def evaluate_uncached(self, selector: Selector) -> float:
        '''Evaluate the measure on a given selector'''
        raise NotImplementedError()


class CachingOptimisticEstimator(CachingEvaluator, OptimisticEstimator):
    '''Abstract Optimistic Estimator class that caches the computed values'''
    __evaluator_type__ = "optimistic_estimator"  # for use in cache property naming

    def evaluate_uncached(self, selector: Selector) -> float:
        '''Evaluate an upper bound for the values of all refinements of the given selector'''
        raise NotImplementedError()

        
class MeasureCoverage(CachingMeasure):
    name = 'Coverage'
    tag = 'coverage'
    
    def __init__(self, data: GraphData) -> None:
        super().__init__(data=data)

    def evaluate_uncached(self, selector: Selector) -> float:
        idl: ndarray = selector.validity
        return idl.mean()

class MeasureJaccard(CachingMeasure):
    name = 'Jaccard'
    tag = 'jaccard'
    
    def __init__(self, data: GraphData, target:str) -> None:
        super().__init__(data=data)
        target_data: Series = self.data.get_series(target, collapse=True, selection=False)
        if target_data.dtype != bool:
            raise TypeError(f"Target must be of type bool, but targt {target_data.name} is of type {target_data.dtype}.")
        self._target_name: str = target_data.name
        self._target_data: Series = target_data

    def evaluate_uncached(self, selector: Selector) -> float:
        idl: ndarray = selector.validity
        target: ndarray = self._target_data.values
        
        return np.sum(idl & target)/np.sum(idl | target)
    
    def summary_dict(self, options:SummaryOptions):
        return {'target':self._target_name}
    
class OptimisticEstimatorJaccard(CachingOptimisticEstimator):
    name = 'Jaccard'
    tag = 'jaccard'
    
    def __init__(self, data: GraphData, target:str) -> None:
        super().__init__(data=data)
        target_data: Series = self.data.get_series(target, collapse=True, selection=False)
        if target_data.dtype != bool:
            raise TypeError(f"Target must be of type bool, but target {target_data.name} is of type {target_data.dtype}.")
        self._target_name: str = target_data.name
        self._target_data: Series = target_data
        self._target_size: int = int(target_data.sum())

    def evaluate_uncached(self, selector: Selector) -> float:
        idl: ndarray = selector.validity
        target: ndarray = self._target_data.values
        
        return np.sum(idl & target)/self._target_size
    
    def summary_dict(self, options:SummaryOptions):
        return {'target':self._target_name, 'target_size': self._target_size}

class MeasureGeometricMean(CachingMeasure):
    name = 'Geometric mean of sub-measures'
    tag = 'geometric-mean'
    
    def __init__(self, data: GraphData, exponents:Iterable[float], measures: Iterable[Measure]) -> None:
        super().__init__(data=data)
        self._measures = _Utils.instantiate_subclass_collections(measures, data, Measure)
        self._exponents = tuple(exponents)
        if len(self._exponents) != len(self._measures):
            raise ValueError(f'The number of exponents ({len(self._exponents)} must match the number of measures ({len(self._measures)}).')
        self._exponent_sum = sum(self._exponents)
        
        

    def evaluate_uncached(self, selector: Selector) -> float:
        values_raw = tuple(measure.evaluate(selector) for measure in self._measures)
        values_wei = tuple(value**exp for value,exp in zip(values_raw, self._exponents))
        
        return float(np.prod(values_wei) ** (1/self._exponent_sum))
    
    def summary_dict(self, options:SummaryOptions):
        return {'exponents': self._exponents, 'measures': SummarisableList(self._measures)}

    @classmethod
    def parse_argument(cls, name, value, parameter):
        if name == 'measures':
            measures = _Utils.parse_subclass_from_string(value)
            return measures
        if name == 'exponents':
            exponents = _Utils.parse_exponents_from_string(value)
            return exponents
        raise NoConversionError()

class _Utils:
    rex_split = re.compile('\s+')
    rex_spec = re.compile('(?P<tag>[a-zA-Z-_]+)\((?P<params>[^)]+)\)')
    rex_params = re.compile('(?P<tag>[a-zA-Z-_]+)\((?P<params>[^)]+)\)')
    rex_param_split = re.compile('\s*=\s*')
    
    ParsedSubclass = namedtuple('ParsedSubclass', ('tag', 'args', 'kwargs'))
    
    @classmethod
    def instantiate_subclass_collections(cls, collection, data, subcls):
        def instantiate(subinst):
            if isinstance(subinst, subcls):
                return subinst
            elif isinstance(subinst, _Utils.ParsedSubclass):
                ps = subinst
                return subcls.make_from_strings(ps.tag, data, *ps.args, **ps.kwargs)
            else:
                raise TypeError(f'Cannot create {subcls} from element {subinst}.')
        return tuple(map(instantiate, collection))
        
    @classmethod
    def parse_subclass_from_string(cls, txt):
        from itertools import takewhile
        specs = cls.rex_split.split(txt)
        def parse_spec(spec):
            m = cls.rex_spec.match(spec)
            if not m:
                raise ValueError(f'Could not parse subclass from {spec}. Must be in tag(val,...,key=val) format.')
            tag = m.groupdict()['tag']
            params = m.groupdict()['params'].split(',')
            param_pairs = tuple(map(cls.rex_param_split.split, params))
            args = tuple(p[0] for p in takewhile(lambda p:len(p)==1,param_pairs))
            kwargs = dict(param_pairs[len(args):])
            return cls.ParsedSubclass(tag=tag, args=args, kwargs=kwargs)
        
        return tuple(map(parse_spec,specs))
    
    @classmethod
    def parse_exponents_from_string(cls, txt):
        return tuple(map(float, txt.split(' ')))
    
class OptimisticEstimatorGeometricMean(CachingMeasure):
    name = 'Geometric mean of sub-optimistic-estimators'
    tag = 'geometric-mean'
    
    def __init__(self, data: GraphData, exponents:Iterable[float], optimistic_estimators: Iterable[OptimisticEstimator]) -> None:
        super().__init__(data=data)
        self._oests = _Utils.instantiate_subclass_collections(optimistic_estimators, data, OptimisticEstimator)
        self._exponents = tuple(exponents)
        if len(self._exponents) != len(self._oests):
            raise ValueError(f'The number of exponents ({len(self._exponents)} must match the number of optimistic estimators ({len(self._oests)}).')
        self._exponent_sum = sum(self._exponents)
        
        

    def evaluate_uncached(self, selector: Selector) -> float:
        values_raw = tuple(oest.evaluate(selector) for oest in self._oests)
        values_wei = tuple(value**exp for value,exp in zip(values_raw, self._exponents))
        
        return float(np.prod(values_wei) ** (1 / self._exponent_sum))
    
    def summary_dict(self, options:SummaryOptions):
        return {'exponents': self._exponents, 'optimistic_estimators': Summarisable(self._oests)}

    @classmethod
    def parse_argument(cls, name, value, parameter):
        if name == 'optimistic_estimators':
            oests = _Utils.parse_subclass_from_string(value)
            return oests
        if name == 'exponents':
            exponents = _Utils.parse_exponents_from_string(value)
            return exponents
        raise NoConversionError()


MEASURES = ClassCollection('Measures', (MeasureCoverage,))

MEASURES_DEFAULT_CONSTRUCTIBLE = ClassCollection('Measures', (MeasureCoverage,))

OPTIMISTIC_ESTIMATORS = ClassCollection('Optimistic Estimators', ())
OPTIMISTIC_ESTIMATORS_DEFAULT_CONSTRUCTIBLE = ClassCollection('Optimistic Estimators', ())

   
    