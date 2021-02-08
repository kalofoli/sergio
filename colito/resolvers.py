'''
Created on Aug 20, 2018

@author: janis
'''
import enum
from collections import OrderedDict
from typing import Union, cast
from functools import reduce
import operator

class EnumResolver:
    '''Convert text or integer inputs to enum members
    
    >>> class Test(enum.Enum):
    ...     ONE = 1
    ...     TWO = 2
    ...     THREE = 3
    >>> r = EnumResolver(Test)
    >>> val = r.resolve(1)
    >>> f'{type(val).__name__},{val.name}'
    'Test,ONE'
    >>> val = r.resolve("ONE")
    >>> f'{type(val).__name__},{val.name}'
    'Test,ONE'
    '''
    
    def __init__(self, cls, ignore_case=True):
        self._cls = cls
        self._is_flag = issubclass(cls, enum.Flag)
        self._ignore_case = ignore_case
        
        self._fn_normalise=lambda x:x.lower() if ignore_case else lambda x:x
        self._name2object = {self._fn_normalise(key):value for key,value in cls.__members__.items()}
        values = cls.__members__.values()
        self._int2value = OrderedDict((value.value,value) for value in values)
        
    def resolve(self, what:Union[str,int], **kwargs):
        '''Retrieve a member of the enum.'''
        if isinstance(what, str):
            text = cast(str, what)
            result = self.resolve_text(text, **kwargs)
        elif isinstance(what, int):
            num = cast(int, what)
            result = self.resolve_int(num, **kwargs)
        elif isinstance(what, self._cls):
            result = what
        else:
            raise TypeError(f'Unknown type {what.__class__.__name__} for object {what}. Use either string or int.')
        return result
    

    def resolve_int(self, num:int):
        return self._int2value[num]

    def resolve_text(self, text:str):
        return self._resolve_text_part(text)
    
    def _resolve_text_part(self, text:str):
        part_text = self._fn_normalise(text.strip())
        missing = object()
        value = self._name2object.get(part_text, missing)
        if value is missing:
            raise KeyError(f'Part {text} could not be tracked to one of the members of {self._cls.__name__}. Available options are: {self.options()}.')
        return value

    def options(self,sep=',',equal=':',pad_names=None):
        if pad_names is not None:
            pad_size = max(map(len,self._cls.__members__)) + pad_names
            fmt_key = f'{{key:{pad_size}s}}'
        else:
            fmt_key = '{key!s}'
        fmt = fmt_key + f'{equal}{{value}}'
        return sep.join(fmt.format(key=key,value=value.value) for key,value in self._cls.__members__.items())

class FlagResolver(EnumResolver):
    """Resolve a flag enum
    @param allow_multiple bool: Allow resolving to yield more than one flags.
    @param allow_composite bool: Allow lookup of flags consisting of more than one active bits.
    
    >>> class Test(enum.Flag):
    ...    ONE = 1
    ...    TWO = 2
    ...    FOUR = 4
    ...    EIGHT = 8
    ...    FIVE = ONE|FOUR
    >>> r = FlagResolver(Test)
    >>> r.resolve('one')
    <Test.ONE: 1>
    >>> r.resolve('one|two').value
    3
    >>> r.resolve('one|eight').value
    9
    >>> r.resolve('one|four')
    <Test.FIVE: 5>
    >>> r.resolve('one|four', allow_composite=False)
    <Test.FIVE: 5>
    >>> r.resolve('five', allow_composite=False)
    Traceback (most recent call last):
     ...
    KeyError: 'Part five gave composite value Test.FIVE. Available options are: ONE:1,TWO:2,FOUR:4,EIGHT:8,FIVE:5.'
    >>> list(r.resolve('one|four', collapse_multipart=False))
    [<Test.ONE: 1>, <Test.FOUR: 4>]
    >>> list(r.resolve('five', collapse_multipart=False))
    [<Test.FIVE: 5>]
    """
    def __init__(self, cls, ignore_case=True, allow_multiple = True, part_separator='|', collapse_multipart=True, allow_composite=True):
        super().__init__(cls, ignore_case=ignore_case)
        
        self._allow_multiple = allow_multiple
        self._allow_composite = allow_composite
        self._part_separator = part_separator
        self._collapse_multipart = collapse_multipart
        
        values = cls.__members__.values()
        self._values_single = {v.value for v in values if self.count_bits(v.value) == 1}
    
    @classmethod
    def count_bits(cls, value):
        '''Count the number of set bits in a given number.
        >>> FlagResolver.count_bits(4)
        1
        >>> FlagResolver.count_bits(5)
        2
        '''
        result = cls.count_bits_slow(value) if value > 1<<32 else cls.count_bits_fast32(value)
        return result
        
    @staticmethod
    def count_bits_fast32(value):
        if value > 1<<32:
            raise IndexError(f'Value {value} out of bounds. Only up to 32 bit integers are supported.')
        value = (value & 0x55555555) + ((value >> 1) & 0x55555555)
        value = (value & 0x33333333) + ((value >> 2) & 0x33333333)
        value = (value & 0x0f0f0f0f) + ((value >> 4) & 0x0f0f0f0f)
        value = (value & 0x00ff00ff) + ((value >> 8) & 0x00ff00ff)
        value = (value & 0x0000ffff) + ((value >> 16) &0x0000ffff)
        return value
    
    @staticmethod
    def count_bits_slow(value):
        cnt = 0
        while value:
            cnt += value& 0x1
            value >>= 1
        return cnt
    
    def number2parts(self, num, allow_remainder=True, allow_composite=True):
        '''Convert a number to a sequence of parts.
        '''
        remaining = num
        parts = []
        if remaining:
            if allow_composite is None:
                allow_composite = self._allow_composite
            if allow_composite:
                items = self._int2value.items()
            else:
                items = ((i,self._int2value[i]) for i in self._values_single)
            for value,name in items:
                if remaining & value == value:
                    parts.append(name)
                    remaining &= ~value
                if not remaining:
                    break
        else:
            parts = [self._cls(0)] if 0 in self._int2value else []
        if allow_remainder:
            return parts, remaining
        else:
            if remaining:
                raise ValueError(f'After adding {parts}, the remainder {remaining} of value {num} cannot be further described by {self._cls.__name__}.') 
            return parts
    
    def flag2str(self, what):
        if isinstance(what, int):
            num = cast(int, what)
        elif isinstance(what, self._cls):
            num = what.value
        else:
            raise TypeError(f'Can only convert integers and {self._cls.__name__} instances.')
        parts = self.number2parts(num, allow_remainder=False)
        result = self._part_separator.join(part.name for part in parts)
        return result

    def _handle_multipart(self, parts, collapse_multipart):
        if collapse_multipart is None:
            collapse_multipart = self._collapse_multipart 
        if collapse_multipart:
            result = reduce(operator.or_, parts, self._cls(0))
        else:
            result = parts
        return result

    def resolve_int(self, num:int, allow_multiple=None, collapse_multipart=None, allow_composite=None):
        if num in self._int2value:
            result = self._int2value[num]
        else:
            if allow_multiple is None:
                allow_multiple = self._allow_multiple
            if not allow_multiple:
                raise KeyError(f'No direct mapping for value {num}. Available options are {self.options()}.')
            else:
                if allow_composite is None:
                    allow_composite = self._allow_composite
                parts = self.number2parts(num, allow_remainder=False, allow_composite=allow_composite)
                result = self._handle_multipart(parts, collapse_multipart)
        return result

        
    def resolve_text(self, text:str, allow_multiple=None, collapse_multipart=None, allow_composite=None):
        if allow_multiple is None:
            allow_multiple = self._allow_multiple
        if allow_composite is None:
            allow_composite = self._allow_composite
        if allow_multiple:
            parts = text.split(self._part_separator)
            part_values = (self._resolve_text_part(part, allow_composite) for part in parts)
            result = self._handle_multipart(part_values, collapse_multipart)
        else:
            result = self._resolve_text_part(text, allow_composite=allow_composite)
        return result
        
    def resolve_parts(self, what:str, allow_multiple=None, allow_composite=None):
        '''Resolve the given input to a list of values.

        @note The only difference between this function and resolve(...,collapse_multipart=False)
        is when handling composite values. This function will only yield the single-bit constituents, whereas
        resolve(...collapse_multipart=False,allow_composite=False) will allow composite in the output,
        but will not accept them in the provided input.
    
        >>> class Test(enum.Flag):
        ...    ONE = 1
        ...    TWO = 2
        ...    FOUR = 4
        ...    FIVE = ONE|FOUR
        >>> r = FlagResolver(Test)
        >>> r.resolve_parts('five', allow_multiple=False, allow_composite=False)
        [<Test.ONE: 1>, <Test.FOUR: 4>]
        '''
        if allow_composite is None:
            allow_composite = self._allow_composite
        if allow_composite:
            parts = self.resolve(what, allow_multiple=allow_multiple,collapse_multipart=False, allow_composite=True)
        else:
            value = self.resolve(what, allow_multiple=allow_multiple,collapse_multipart=True, allow_composite=True)
            num = value.value
            parts = self.number2parts(num, allow_remainder=False, allow_composite=False)
        return parts

    def _resolve_text_part(self, text:str, allow_composite):
        value = super()._resolve_text_part(text)
        if not allow_composite and value.value not in self._values_single:
            raise KeyError(f'Part {text} gave composite value {value}. Available options are: {self.options()}.')
        return value
    

def make_enum_resolver(cls, **kwargs):
    if issubclass(cls, enum.Flag):
        return FlagResolver(cls, **kwargs)
    else:
        return EnumResolver(cls, **kwargs)

if __name__ == "__main__":
    import sys
    del sys.path[0]
    import doctest
    doctest.testmod()

