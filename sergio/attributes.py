'''
Created on Jan 19, 2018

@author: janis
'''

from typing import Dict, cast, Sequence, Type
import enum
from pandas import Series, DataFrame

class AttributeKind(enum.Enum):
    '''Attribute interpretation kind'''
    NUMERICAL = enum.auto()
    CATEGORICAL = enum.auto()
    BOOLEAN = enum.auto()
    INDEX = enum.auto()
    NAME = enum.auto()


class AttributeInfoFactory:

    __map_kind_to_class: Dict['AttributeKind', Type] = {}
    
    def __init__(self, data: DataFrame):
        self._data = data
    
    @classmethod
    def resolve_kind(cls, kind) -> "AttributeKind":
        if isinstance(kind, AttributeKind):
            return kind
        else:
            kind_str = kind.upper()
            if kind_str not in AttributeKind.__members__:
                keys_str: str = ','.join(AttributeKind.__members__.keys())
                raise KeyError(f"Could not resolve kind {kind}. Choose one of: [{keys_str}].")
            return AttributeKind[kind_str]

    PandasTypes = (({'int16', 'int32', 'int64', 'float64', 'float32'}, AttributeKind.NUMERICAL),
                   ({'bool'}, AttributeKind.BOOLEAN),
                   ({'category', 'object'}, AttributeKind.CATEGORICAL))

    @classmethod
    def _infer_kind(cls, series):
        '''Infer AttributeKind from series dtype.'''
        dtype = series.dtype
        for dtypes, kind in cls.PandasTypes:
            if str(dtype) in dtypes:
                return kind
        raise ValueError("Could not match kind for dtype: {0}".format(dtype))

    def __call__(self, index: int, spec=None):
        '''Make an attribute from column index and hints
        
        >>> import numpy as np
        >>> df = DataFrame({'a':np.r_[True, False, True, True],'b':['one','two','three','five'],'c':[3,2,4,6]})
        >>> aif = AttributeInfoFactory(df)
        >>> aif(0)
        <Attribute[BOOLEAN]:a>
        >>> aif(1)
        <Attribute[CATEGORICAL]:b>
        >>> aif(2)
        <Attribute[NUMERICAL]:c>
        '''
        data = self._data
        series = data.iloc[:,index]
        if isinstance(spec, AttributeInfo):
            cls_attr = spec.__class__
        else:
            if spec is None:
                kind = self._infer_kind(series)
            else:
                kind = self.resolve_kind(spec)
            cls_attr = self.__map_kind_to_class[kind]
        return cls_attr(data, index)

    @classmethod
    def _register_classes(cls, classes: Sequence[Type['Attribute']]):
        for attribute_class in classes:
            cls.__map_kind_to_class[attribute_class.__kind__] = attribute_class


class AttributeInfo:
    '''Defines the interpretation of a series'''
    __kind__: AttributeKind
    __selectable__ = True
    
    def __init__(self, data: DataFrame, index: int) -> None:
        self._data = data
        self._index = index

    @property
    def kind(self):
        '''The kind of this attribute'''
        return self.__class__.__kind__

    @property
    def index(self):
        '''Index within the data of the series this attribute refers to'''
        return self._index

    @property
    def series(self) -> Series:
        '''Series data for the specified index'''
        return self._data.iloc[:, self.index]

    @property
    def name(self) -> str:
        '''Name of this attribute'''
        return self.series.name

    @property
    def data(self):
        '''The data frame this attribute describes'''
        return self._data

    def __repr__(self):
        return "<Attribute[{0.kind.name}]:{0.name}>".format(self)


class AttributeCategorical(AttributeInfo):
    '''An attribute to be interpreted as categorical'''
    __kind__ = AttributeKind.CATEGORICAL

class AttributeNumerical(AttributeInfo):
    '''An attribute to be interpreted as numerical'''
    __kind__ = AttributeKind.NUMERICAL

class AttributeBoolean(AttributeInfo):
    '''An attribute to be interpreted as boolean'''
    __kind__ = AttributeKind.BOOLEAN


class AttributeIndex(AttributeInfo):
    '''An attribute containing numerical instance indices'''
    __kind__ = AttributeKind.INDEX
    __selectable__ = False


class AttributeName(AttributeInfo):
    '''An attribute containing a list of instance names'''
    __kind__ = AttributeKind.NAME
    __selectable__ = False

AttributeInfoFactory._register_classes([AttributeCategorical, AttributeNumerical, AttributeBoolean, AttributeIndex, AttributeName])

if __name__ == '__main__':
    import doctest
    doctest.testmod()



