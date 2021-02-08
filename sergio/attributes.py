'''
Created on Jan 19, 2018

@author: janis
'''

from typing import TypeVar, Union, Container, Dict, cast, Optional, \
    TYPE_CHECKING, Sequence, Type
from enum import Enum
from pandas import Series
from builtins import property

IndexType = TypeVar('IndexType', str, int)
KindType = Union[str, 'AttributeKind']
KindCollection = Union[Dict[IndexType, KindType], Container[KindType]]
SelectionCollection = Union[Dict[IndexType, bool], Container[bool]]

if TYPE_CHECKING:
    from sdcore.data import GraphData


class AttributeKind(Enum):
    '''Attribute interpretation kind'''
    NUMERICAL = 1
    CATEGORICAL = 2
    BOOLEAN = 3
    INDEX = 4
    NAME = 5


class AttributeFactory:

    map_kind_to_class: Dict['AttributeKind', Type] = {}
    
    @classmethod
    def lookup_kind(cls, kind: KindType) -> "AttributeKind":
        '''Lookup a given kind, possibly 'auto'.'''
        if isinstance(kind, AttributeKind):
            return cast(AttributeKind, kind)
        else:
            kind_str = cast(str, kind).upper()
            if kind_str not in AttributeKind.__members__:
                keys_str: str = ','.join(AttributeKind.__members__.keys())
                raise KeyError(("No such AttributeKind: {0}. " + 
                                "Choose one of: [{1}].")
                                .format(kind, keys_str))
            return AttributeKind[kind_str]

    @classmethod
    def get_class_from_kind(cls, kind: KindType) -> Type:
        kind_type: 'AttributeKind' = cls.lookup_kind(kind)
        return cls.map_kind_to_class[kind_type]
    
    PandasTypes = (({'int16', 'int32', 'int64', 'float64', 'float32'}, AttributeKind.NUMERICAL),
                   ({'bool'}, AttributeKind.BOOLEAN),
                   ({'category', 'object'}, AttributeKind.CATEGORICAL))

    @classmethod
    def infer_kind_from_series(cls, series):
        '''Infer AttributeKind from series dtype.'''
        dtype = series.dtype
        for dtypes, kind in cls.PandasTypes:
            if str(dtype) in dtypes:
                return kind
        raise ValueError("Could not match kind for dtype: {0}".format(dtype))

    @classmethod
    def _register_classes(cls, classes: Sequence[Type['Attribute']]):
        for attribute_class in classes:
            cls.map_kind_to_class[attribute_class.KIND] = attribute_class

    @classmethod    
    def make(self, data: 'GraphData', index: int, kind: Optional[KindType]=None):
        '''Make an attribute from GraphData and a column index'''
        
        series = data.get_series(index, collapse=True)
        attribute_kind: AttributeKind
        if kind is None:
            attribute_kind = AttributeFactory.infer_kind_from_series(series)
        else:
            attribute_kind = AttributeFactory.lookup_kind(kind)
        
        attribute_cls = AttributeFactory.get_class_from_kind(attribute_kind)
        attribute = attribute_cls(data=data, index=index)
        return attribute


class Attribute:
    '''Defines the interpretation of a series'''
    KIND: AttributeKind
    
    def __init__(self, data: 'GraphData', index: int) -> None:
        self._data = data
        self._index = index

    @property
    def kind(self):
        '''The kind of this attribute'''
        return self.__class__.KIND

    @property
    def index(self):
        '''Index within the data of the series this attribute refers to'''
        return self._index

    @property
    def series(self) -> Series:
        '''Series data for the specified index'''
        return self.data.get_series(self.index)

    @property
    def name(self) -> str:
        '''Name of this attribute'''
        return self.series.name

    @property
    def data(self) -> 'GraphData':
        '''The graph data this attribute refers to'''
        return self._data

    def __repr__(self):
        return "<Attribute[{0.kind.name}]:{0.name}>".format(self)


class AttributeCategorical(Attribute):
    '''An attribute to be interpreted as categorical'''
    KIND = AttributeKind.CATEGORICAL

    def __init__(self, data: 'GraphData', index: int) -> None:
        super(AttributeCategorical, self).__init__(data=data, index=index)


class AttributeNumerical(Attribute):
    '''An attribute to be interpreted as numerical'''
    KIND = AttributeKind.NUMERICAL

    def __init__(self, data: 'GraphData', index: int) -> None:
        super(AttributeNumerical, self).__init__(data=data, index=index)

class AttributeBoolean(Attribute):
    '''An attribute to be interpreted as boolean'''
    KIND = AttributeKind.BOOLEAN

    def __init__(self, data: 'GraphData', index: int) -> None:
        super(AttributeBoolean, self).__init__(data=data, index=index)


class AttributeIndex(Attribute):
    '''An attribute containing numerical instance indices'''
    KIND = AttributeKind.INDEX

    def __init__(self, data: 'GraphData', index: int) -> None:
        super(AttributeIndex, self).__init__(data=data, index=index)


class AttributeName(Attribute):
    '''An attribute containing a list of instance names'''
    KIND = AttributeKind.NAME

    def __init__(self, data: 'GraphData', index: int) -> None:
        super(AttributeName, self).__init__(data=data, index=index)


AttributeFactory._register_classes([AttributeCategorical, AttributeNumerical, AttributeBoolean, AttributeIndex, AttributeName])
