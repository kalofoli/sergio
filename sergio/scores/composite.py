'''
Created on Mar 9, 2021

@author: janis
'''

from . import CachingMeasure, CachingEvaluator
from colito.factory import NoConversionError

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
