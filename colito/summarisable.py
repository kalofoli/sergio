'''
Created on Jun 23, 2018

@author: janis
'''

from typing import NamedTuple, Dict, Any, List
from enum import Flag, auto
from collections import OrderedDict
from functools import reduce
import operator
import enum



class SummaryParts(Flag):
    BASIC = 0
    UNCOMPRESSED_INDICES = auto()
    COMPRESSED_INDICES = auto()
    INDICES = COMPRESSED_INDICES
    
    @classmethod
    def from_string(cls, txt:str) -> 'SummaryParts':
        try:
            return cls(int(txt))
        except ValueError:
            pass
        if txt == 'ALL':
            val = reduce(operator.or_, cls.__members__.values())
        else:
            parts = txt.split('|')
            val = cls(0)
            for part in parts:
                if part.upper() not in cls.__members__:
                    raise ValueError(f'No member {part} in {cls.__name__}. Try one of {cls.members_desc}.')
                val |= cls.__members__[part]
        return val
    
    @classmethod
    def get_parts(cls, what) -> 'SummaryParts':
        if isinstance(what, str):
            return cls.from_string(what)
        elif isinstance(what, SummaryParts):
            return what
        else:
            return SummaryParts(what)
        
    @classmethod
    def members_desc(cls):
        return ','.join(f'{name}({e.value})' for name,e in cls.__members__.items()) 
        
class SummaryOptions(dict):
    def __init__(self, parts:SummaryParts, is_compact:bool, **kwargs):
        self.parts: SummaryParts
        self.is_compact: bool
        self.update(parts=parts, is_compact=is_compact,**kwargs)
    def __getattr__(self, key): return self[key]
    def __setattr__(self, key, val): self[key] = val
    def __dir__(self): return object.__dir__(self) + list(self.keys())
    
DEFAULT_SUMMARY_OPTIONS = SummaryOptions(parts=SummaryParts.BASIC, is_compact=False)
COMPACT_SUMMARY_OPTIONS = SummaryOptions(parts=SummaryParts.BASIC, is_compact=True)


class OnMissing(enum.Enum):
    OMMIT = enum.auto()
    RAISE = enum.auto()
    USE_NONE = enum.auto()

# this is required due to a cyclical dependency
ON_MISSING_RESOLVER = None
try:
    from cofi.utils.resolvers import EnumResolver
    ON_MISSING_RESOLVER = EnumResolver(OnMissing)
except ImportError as e:
    pass

class Summarisable:
    summary_sibling_priority:int = 0
    def summary(self, options:SummaryOptions):
        '''The parameters to be included in the summary as a dict'''
        raise NotImplementedError()

    @property
    def summary_name(self) -> str:
        '''The name of this summary object'''
        return self.__class__.__name__
    
    @property
    def summary_short(self) -> str:
        '''The short summary of this object'''
        return str(self)
        
    def summary_from_fields(self, fields, instance=None, missing=OnMissing.RAISE):
        '''Create a dict from given fields'''
        global ON_MISSING_RESOLVER
        if ON_MISSING_RESOLVER is None:
            from cofi.utils.resolvers import EnumResolver
            ON_MISSING_RESOLVER = EnumResolver(OnMissing)
        missing = ON_MISSING_RESOLVER.resolve(missing)
        if instance is None:
            instance = self
        def getattrof(field):
            return getattr(instance,field)
        flds = fields
        if missing == OnMissing.RAISE:
            fieldvalue = getattrof
        elif missing == OnMissing.OMMIT:
            fieldvalue = getattrof
            flds = tuple(f for f in fields if hasattr(instance, f))
        elif missing == OnMissing.USE_NONE:
            def fieldvalue(field):
                try:
                    return getattrof(field)
                except AttributeError:
                    return None
        vals = (fieldvalue(fld) for fld in flds)
        return OrderedDict(zip(flds, vals))

class SummarisableAsDict(Summarisable):
    _summary_convert = {}
    def summary(self, options:SummaryOptions):
        smr = self.summary_dict(options)
        for key,fn in self._summary_convert.items():
            if key in smr:
                smr[key] = fn(smr[key])
            elif isinstance(key, type):
                for smr_key, smr_val in smr.items():
                    if isinstance(smr_val, key):
                        smr[smr_key] = fn(smr_val)
        return SummarisableDict(smr)

    def summary_dict(self, options:SummaryOptions) -> Dict[str, Any]:
        raise NotImplementedError()

    def __repr__(self):
        dct = self.summary_dict(COMPACT_SUMMARY_OPTIONS)
        params_txt = ','.join(f'{key}={value}' for key,value in dct.items())
        return f'<{self.__class__.__name__}({params_txt})>'

    def __str__(self):
        if hasattr(self, '_summary_compact_fields'):
            dct = self.summary_dict(COMPACT_SUMMARY_OPTIONS)
            def stringify(val):
                if isinstance(val, float):
                    return '{0:g}'.format(val)
                return str(val)
            params_txt = ','.join(stringify(dct[value]) for value in self._summary_compact_fields)
            return f'{self.__class__.__name__}({params_txt})'
        else:
            return repr(self)

class SummarisableDict(dict, SummarisableAsDict):
    def summary_dict(self, options:SummaryOptions) -> Dict[str, Any]:
        return self

def summarise_exception(exc, summary_options:SummaryOptions):
    import traceback as tb
    dct = {'message':str(exc), 'type':type(exc).__name__}
    if hasattr(exc, '__traceback__') and exc.__traceback__ is not None:
        frames = tb.extract_tb(exc.__traceback__)
        dct_frames = []
        for frame in frames:
            dct_frames.append({'name':frame.name,'lineno':frame.lineno,'line':frame.line,'file':frame.file})
        dct['frames'] = dct_frames
        if frames:
            dct.update({f'last_{k}':v for k,v in dct['frames'][0].items()})
    if hasattr(exc,'__context__') and exc.__context__ is not None:
        dct['context'] = summarise_exception(exc.__context__, summary_options)
    return dct
    
    

class SummarisableException(Exception, SummarisableAsDict):
    def summary_dict(self, summary_options:SummaryOptions):
        return summarise_exception(self, summary_options)
        

class SummarisableAsList(Summarisable):
    def summary(self, options:SummaryOptions):
        smr = self.summary_list(options)
        return SummarisableList(smr)

    def summary_list(self, options:SummaryOptions) -> List:
        raise NotImplementedError()

class SummarisableList(list, SummarisableAsList):
    def summary_list(self, options:SummaryOptions) -> List:
        return self

class CompressionScheme(enum.Enum):
    BASE64 = enum.auto()
    B64GZIP = enum.auto()
    GZIP = enum.auto()

class CompressionToString(enum.Enum):
    AUTO = enum.auto()
    JSON = enum.auto()
    STR = enum.auto()
    REPR = enum.auto()
    RAW = enum.auto()

'''Tagging class as summary compressible. Also usable as a mixin.'''
class SummaryCompressible(Summarisable):
    summary_compressible_encoding = 'utf8'
    summary_compressible_enable:bool = True
    summary_compressible_tostring:CompressionToString = None
    summary_compressible_scheme:CompressionScheme = CompressionScheme.B64GZIP
    def __init__(self, value, enable:bool=None, tostring:CompressionToString=None, scheme:CompressionScheme=None):
        if enable is not None:
            self.summary_compressible_enable = enable
        if tostring is not None:
            self.summary_compressible_tostring = tostring
        if scheme is not None:
            self.summary_compressible_scheme = scheme
        self.summary_value = value
    def __repr__(self):
        return f'{self.__class__.__name__}({self.summary_value!r})'
    def __str__(self):
        return str(self.value)
    def summary(self, options:SummaryOptions):
        return self.summary_value

'''Tagging class as summary grouppable. Also usable as a mixin.'''
class SummaryGroupable(Summarisable):
    summary_group_enable:bool = True
    summary_group_name:str = None
    def __init__(self, value, name=None, enable:bool = True):
        if enable is not None:
            self.summary_group_enable:bool = enable
        if name is not None:
            self.summary_group_name:str = name
        if self.summary_group_name is None:
            self.summary_group_name = self.__class__.__name__
        if not hasattr(self.__class__,'summary_value'):
            self.summary_value = value
    def summary(self, options:SummaryOptions):
        return self.summary_value

class SummaryGroupCompressible(SummaryGroupable):
    summary_group_compressible = SummaryCompressible # class to use to compress group
