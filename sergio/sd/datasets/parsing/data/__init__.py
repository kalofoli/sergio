from types import SimpleNamespace
from collections import deque

from .. import Properties
import re
from sergio.sd.datasets.parsing import FormattingError
from itertools import chain

class FormatterData:
    def __init__(self):
        pass
    
    def __call__(self):
        pass
    def __repr__(self): return f'<{type(self).__name__}({self})>'
    
class FormatterSequence(FormatterData):
    def __init__(self, data):
        self._data = deque(data)
    
    def __bool__(self):
        return self._data.__bool__()
    
    def __len__(self): return len(self._data)
    
    def consume(self):
        return self._data.popleft()
    def __str__(self): return str(self._data)

class FormatterPool(FormatterData):
    def __init__(self, data):
        self._data = data
    
    def __bool__(self):
        return True
    def __len__(self): return None
    
    def consume(self):
        return self._data
    def __str__(self): return str(self._data)

def sequence(what):
    if isinstance(what, FormatterData):
        return what
    else:
        return FormatterSequence(what)

def pool(what):
    if isinstance(what, FormatterPool):
        return what
    else:
        return FormatterPool(what)

def props(**kwargs):
    return pool(Properties(**kwargs))
    

rex_attr = re.compile('^[_a-zA-Z][0-9a-zA-Z_]+')
def consume(what, name=None):
    if isinstance(what, FormatterData):
        data = what.consume()
    else:
        data = what
    if isinstance(name, str):
        m = rex_attr.match(name)
        if m is not None:
            idx = m.end(0)
            key, expand = name[:idx],name[idx:]
            try:
                data = data[key]
            except Exception as e:
                raise FormattingError(f'Could not parse key {key} from data of type {type(data).__name__} with value: {data}.') from e
        else:
            expand = name
        if expand == '*':
            args, kwargs = data, {}
        elif expand == '**':
            args, kwargs = (), dict(data)
        elif expand == '.':
            args, kwargs = (Properties(**data),),{}
        elif not expand:
            args, kwargs = (data,),{}
        else:
            raise FormattingError(f'Invalid expand option {expand}.')
    else:
        args, kwargs = (data,),{}
    return args, kwargs
    
class ParserData:
    def __call__(self): pass
    @property
    def ignore(self): return False


class ParserDataContainer(): 
    def __call__(self, unpack):
        pass

class ParserSequence(list, ParserDataContainer):
    def __call__(self, pack):
        data = (d.unpack() for d in self if not d.ignore)
        if pack == '{}':
            params = dict(chain(*data))
            res = ParserProperties(**params)
        elif pack == '[]':
            res = list(data)
        elif pack == '[0]':
            res = next(data)
        else:
            res = list(self)
        return res

class ParserProperties(Properties, ParserData):
    def unpack(self):
        return Properties(**self.__dict__)

class ParserDataRaw(ParserData):
    def __init__(self, value):
        self.value = value
    def __str__(self): return str(self.value)
    def __repr__(self): return f'{type(self).__name__}({self!s})'
    def unpack(self):
        return self.value
    @property
    def ignore(self): return False
    
class ParserDataResult(ParserData):
    def __init__(self, value, index, symbol):
        self.value = value
        self.index = index
        self.symbol = symbol
    def __str__(self): return str(self.value)
    def __repr__(self): return f'{type(self).__name__}({type(self.symbol).__name__}@{self.index}: {self!s})'
    def unpack(self):
        value = self.value
        while isinstance(value, ParserData):
            value = value.unpack()
        return value
    @property
    def ignore(self): return not isinstance(self.value, ParserData)



def produce(data, name):
    if isinstance(name, str):
        m = rex_attr.match(name)
        if m is not None:
            idx = m.end(0)
            key, pack = name[:idx],name[idx:]
        else:
            key = None
            pack = name
        if isinstance(data, ParserDataContainer):
            res = data(pack)
        elif pack == '{}':
            res = {key:data.value}
        elif pack == '=':
            res = ParserDataRaw(data)
        else:
            if name:
                res = ParserProperties(**{name:data})
            else:
                res = data
    else:
        res = data
    return res

    
    
    