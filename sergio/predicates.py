'''Validate data entities based on properties'''

from typing import Collection, Tuple, List, Sequence, TYPE_CHECKING, Optional, cast, Union, \
    Type

import re
from collections import OrderedDict
import enum
from itertools import chain
from functools import reduce
import operator

from pandas.core.series import Series, isna, notna
import numpy as np

from colito.summaries import SummaryOptions, SummarisableAsDict,\
    summary_from_fields
from sergio.attributes import AttributeKind, AttributeInfo as Attribute, AttributeCategorical, AttributeNumerical, \
    AttributeBoolean
from sergio.discretisers import Discretiser, Interval, DEFAULT_DISCRETISER
from colito.resolvers import make_enum_resolver


EntityIndexType = Union[slice, np.ndarray, Sequence[int]]

class PredicateKinds(enum.IntFlag):
    BOOLEAN = enum.auto()
    CATEGORICAL = enum.auto()
    RANGED = enum.auto()
    ALL = BOOLEAN|CATEGORICAL|RANGED
    NONE = 0

PREDICATE_KINDS_RESOLVER = make_enum_resolver(PredicateKinds)

class Predicate:
    '''A logical predicate mapping entities to a boolean'''
     
    def __init__(self, data: 'GraphData', negated=False) -> None:
        self._negated: bool = False
        self.negated = negated
        self._data: 'GraphData' = data

    @property
    def negated(self):
        '''Whether the predicate value should be negated'''
        return self._negated

    @negated.setter
    def negated(self, value: bool):
        self._negated = value

    @property
    def data(self):
        '''Data which the predicate refers'''
        return self._data

    @property
    def name(self):
        '''A name for pretty printing of the current predicate'''
        raise NotImplementedError()

#     @property
#     def validity(self):
#         #TODO: fill this
#         '''Return the full validity of this object'''
#         raise NotImplementedError()

    def validate(self, entity_index:EntityIndexType=slice(None, None, None)) -> np.ndarray:
        '''Decide if data rows validate the predicate'''
        raise NotImplementedError()

    def __repr__(self):
        negated_str = "!" if self.negated else ""
        return "<Predicate:{1}{0.name}>".format(self, negated_str)

    def __lt__(self, other):
        if other.__class__ == self.__class__:
            return self.__class__.__name__ < other.__class__.__name__


class AttributePredicate(SummarisableAsDict, Predicate):
    '''Predicate based on a single attribute'''
    
    name_rex = re.compile('[a-zA-Z_][a-zA-Z0-9_]*')
    
    def __init__(self, attribute: Attribute, negated: bool=False) -> None:
        super(AttributePredicate, self).__init__(data=attribute.data, negated=negated)
        self._attribute: Attribute = attribute
        self._attribute_index: int = attribute.index

    @property
    def name(self):
        return self.series.name

    @property
    def series(self) -> Series:
        '''Series data this predicate uses'''
        return self.attribute.series

    @property
    def index(self) -> int:
        '''Index of the involved attribute'''
        return self._attribute_index

    @property
    def attribute(self) -> Attribute:
        '''The attribute for this Predicate'''
        return self._attribute

    def validate_true(self, entity_index) -> np.ndarray:
        raise NotImplementedError()
        
    def validate(self, entity_index: EntityIndexType=slice(None, None, None)) -> np.ndarray:
        val = self.validate_true(entity_index=entity_index)
        if self.negated:
            val = ~val
        return val
    
    def __lt__(self, other) -> bool:
        raise NotImplementedError()
    
    @classmethod
    def escape(cls, what, allow_unescaped:bool=True) -> str:
        token = str(what) if notna(what) else '<NA>'
        if cls.name_rex.match(token) is not None:
            res = token if allow_unescaped else f'"{token}"'
        else:
            res = '"' + token.replace('\\', r'\\').replace('"', r'\"') + '"'
        return res
    
    def __repr__(self):
        return f'<{self.__class__.__name__}:{self.to_string(False)}>'

    def __str__(self):
        return self.to_string(compact=True)
    
    def to_string(self, compact=True) -> str:
        '''Create a string representation of this predicate'''
        raise NotImplementedError()
    
    def __summary_dict__(self, options:SummaryOptions):
        return OrderedDict([('description', str(self)),
                            ('name', self.name),
                            ('negated', self.negated),
                            ('attribute_index', self.index)])


class PredicateCategorical(AttributePredicate):
    '''Test if an entity has a categorical attribute equal to a category'''
    kind = PredicateKinds.CATEGORICAL

    def __init__(self, attribute: Attribute, category: str, negated: bool=False) -> None:
        super(PredicateCategorical, self).__init__(attribute=attribute, negated=negated)
        self._category = category

    @property
    def category(self):
        '''Category this predicate tests against'''
        return self._category

    def validate_true(self, entity_index:EntityIndexType=slice(None, None, None)) -> np.ndarray:
        idl = self.series[entity_index] == self.category
        return idl.values

    def to_string(self, compact:bool=False) -> str:
        cls: Type[AttributePredicate] = self.__class__
        name = cls.escape(self.name, allow_unescaped=compact)
        category = cls.escape(self.category, allow_unescaped=compact)
        negated = '!' if self.negated else ''
        return f'[{name}{negated}={category}]'

    def __lt__(self, other: 'PredicateCategorical') -> bool:
        if isinstance(other, AttributePredicate):
            other_index = other.index
            self_index = self.index
            if self_index < other_index:
                res = True
            elif self_index == other_index:
                if isna(self.category):
                    res = False
                elif isna(other.category):
                    res = True
                else:
                    res = self.category < other.category
            else:
                res = False  
            return res
        else:
            return super(PredicateCategorical, self).__lt__(other)

    def __summary_dict__(self, options:SummaryOptions):
        dct = super().__summary_dict__(options)
        dct['category'] = self.category
        return dct
    
    summary_name = 'predicate-categorical'

        
class PredicateRanged(AttributePredicate):
    '''Test if an entity numeric value falls in given range'''
    kind = PredicateKinds.RANGED

    def __init__(self, attribute: Attribute, interval: Interval,
                 negated: bool=True) -> None:
        super(PredicateRanged, self).__init__(attribute=attribute, negated=negated)
        self._interval = interval

    label = property(lambda self:self._interval.name,None,'The label of the current range')
    interval = property(lambda self:self._interval,None, 'The interval of the current range')
    bounds = property(lambda self: self._interval.bounds, None, 'The bounds provided. Might be None.')
    range = property(lambda self: self._interval.range, None, 'The range covered. Might not be None.')

    def to_string(self, compact: bool=False) -> str:
        cls: Type[AttributePredicate] = self.__class__
        interval = self.interval
        name = cls.escape(self.name, allow_unescaped=compact)
        if compact:
            s = f'[{name} {"not" if self.negated else "is"} {interval.name}]'
        else:
            str_lo = f'{interval.lower:.3g}{"<=" if interval.leq else "<"}' if interval.lower is not None else ''
            str_up = f'{"<=" if interval.ueq else "<"}{interval.upper:.3g}' if interval.upper is not None else ''
            negated = '!' if self.negated else ''
            s = f'{negated}[{str_lo}{name}{str_up}]'
        return s 

    def validate_true(self, entity_index:EntityIndexType=slice(None, None, None)) -> np.ndarray:
        interval:Interval = self.interval
        series = self.series[entity_index]
        idl = np.ones(len(series), dtype=bool)
        if interval.lower is not None:
            idl_ok = (interval.lower <= series) if interval.leq else (interval.lower < series)
            idl = idl & idl_ok
        if interval.upper is not None:
            idl_ok = (series <= interval.upper) if interval.ueq else (series < interval.upper)
            idl = idl & idl_ok
        return idl.values

    def __lt__(self, other: 'PredicateRanged') -> bool:
        if isinstance(other, AttributePredicate):
            other_index = other.index
            self_index = self.index
            res = self_index < other_index or (self_index == other_index and self.range < other.range)
            return res
        else:
            return super(PredicateRanged, self).__lt__(other)

    def __summary_dict__(self, options:SummaryOptions):
        dct = super().__summary_dict__(options)
        dct['bounds'] = self.bounds
        return dct
    
    summary_name = 'predicate-ranged'

    
class PredicateBoolean(AttributePredicate):
    '''Test if an entity numeric value falls in given range'''
    kind = PredicateKinds.BOOLEAN
    
    @property
    def label(self) -> str:
        '''The label of the current range'''
        return 'F' if self.negated else 'T'

    def to_string(self, compact:bool=False) -> str:
        cls:Type[AttributePredicate] = self.__class__
        name = cls.escape(self.name, allow_unescaped=compact)
        negated = '!' if self.negated else ''
        if compact and name[0] != '"':
            text = f'{negated}{name}'
        else:
            text = f'{negated}[{name}]'
        return text

    def validate_true(self, entity_index:EntityIndexType=slice(None, None, None)) -> np.ndarray:
        series = self.series[entity_index].astype(bool).values
        return series

    def __lt__(self, other: 'PredicateBoolean') -> bool:
        if isinstance(other, AttributePredicate):
            other_index = other.index
            self_index = self.index
            res = self_index < other_index or (self_index == other_index and self.negated < other.negated)
            return res
        else:
            return super(PredicateBoolean, self).__lt__(other)

    summary_name = 'predicate-boolean'


class Prediciser(SummarisableAsDict):
    
    def __init__(self, discretiser: Discretiser=None, negate:PredicateKinds=PredicateKinds.ALL) -> None:
        self._discretiser: Discretiser = DEFAULT_DISCRETISER if discretiser is None else discretiser
        self._negate = PREDICATE_KINDS_RESOLVER.resolve(negate)
    
    discretiser = property(lambda self:self._discretiser, None, 'The discretiser to use for ranged attributes')
    negate = property(lambda self:self._negate, None, 'The kind of predicates for which negations are also appended during predicisation')
    
    def predicates_from_attribute(self, attribute: Attribute) -> List[Predicate]:
        '''Create an AttributePredicate based on the dtype of a series'''
        if isinstance(attribute, AttributeCategorical):
            preds = self.make_categorical_from_attribute(attribute)
        elif isinstance(attribute, AttributeNumerical):
            preds = self.make_ranged_from_attribute(attribute)
        elif isinstance(attribute, AttributeBoolean):
            preds = self.make_boolean_from_attribute(attribute)
        else:
            raise TypeError(f'Unknown attribute {attribute} of class {attribute.__class__.__name__}.')
        return preds
    
    def make_boolean_from_attribute(self, attribute: AttributeBoolean) -> List[PredicateBoolean]:
        '''Create boolean predicate based on a numeric data series'''
        
        preds = [PredicateBoolean(attribute, False)]
        if self.negate & PredicateKinds.BOOLEAN:
            preds.append(PredicateBoolean(attribute, True))
        return preds

    def make_ranged_from_attribute(self, attribute: AttributeNumerical) -> List[PredicateBoolean]:
        '''Create ranged predicate based on a numeric data series'''
        series: Series = attribute.series
        
        intervals = self.discretiser.discretise(series.values)
        negate = self.negate & PredicateKinds.BOOLEAN
        
        def mk_preds(interval:Interval) -> PredicateRanged:
            '''Make ranged predicates from an attribute'''
            nonlocal negate
            preds = [PredicateRanged(attribute=attribute, interval=interval, negated=False)]
            if negate:
                pred_neg = PredicateRanged(attribute=attribute, interval=interval, negated=True)
                preds.append(pred_neg) 
            return preds
        
        preds = reduce(operator.add,map(mk_preds, intervals),[])
        return preds

    def make_categorical_from_attribute(self, attribute: AttributeCategorical) -> List[PredicateBoolean]:
        '''Create categorical predicate based on a string data series'''
        series = attribute.series
        categories = set(series)
        
        negate = len(categories) != 2 and PredicateKinds.CATEGORICAL & self.negate
            
        def mk_preds(category):
            '''Make categorical predicates from an attribute'''
            nonlocal negate
            preds = [PredicateCategorical(attribute=attribute, category=category, negated=False)]
            if negate:
                pred_neg = PredicateCategorical(attribute=attribute, category=category, negated=True)
                preds.append(pred_neg) 
            return preds

        preds = reduce(operator.add,map(mk_preds, categories),[])
        return preds
    
    def __summary_dict__(self, options:SummaryOptions):
        dct = summary_from_fields(('discretiser',))
        dct['negate'] = PREDICATE_KINDS_RESOLVER.flag2str(self.negate)
        return dct

    def __repr__(self):
        return f'<{self.__class__.__name__}:with discretiser {self.discretiser!r} negating {self.negate!r}>'

    def __str__(self):
        return f'<{self.__class__.__name__} D:{self.discretiser!s} N:{PREDICATE_KINDS_RESOLVER.flag2str(self.negate)}>'


DEFAULT_PREDICISER = Prediciser(discretiser=DEFAULT_DISCRETISER, negate=PredicateKinds.ALL)
