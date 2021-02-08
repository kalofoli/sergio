'''
Created on Feb 5, 2021

@author: janis
'''
from io import StringIO
import re
from .data import *
from . import FormattingError, MissingData
from typing import Callable
import pandas
import numpy
from . import Properties as _Properties
from sergio.sd.datasets.parsing import _get_default_class_name

class Symbol:
    def dumps(self, *args,**kwargs):
        s = StringIO()
        _ = self.dump(s, *args, **kwargs)
        return s.getvalue()
    
    def dump(self, s, *args,**kwargs):
        it = self.__generate__(*args, **kwargs)
        sz = 0
        try:
            while True:
                sz += s.write(next(it))
        except StopIteration:
            pass
        return sz
    
class DataSymbol(Symbol):
    def __init__(self, data=None):
        self._data = data if data is not None else _get_default_class_name(type(self))
    def __generate__(self, data):
        args, kwargs = consume(data, self._data)
        yield from self._generate(*args, **kwargs)
    @property
    def data(self): return self._data

class Value(DataSymbol):
    '''
    @param quote: Controls quoting.
        True quotes always,
        False never quotes
        'str' quotes strings,
        'name' quotes strings when necessary
    @param fmt Controls formatting.
        If a string, it follows the python formatting rules (e.q., "!s", ":.3f", etc).
        If callable, it is invoked with the value in question to yield the result.
        If a dict, the keys are classes mapping to formatting options. A key of None works as the default.
    @note Only doubles quotes are supported.
    '''
    
    def __init__(self, quote='str', fmt='!s', **kwargs):
        super().__init__(**kwargs)
        self._quote = quote
        self._fmt = fmt
    from . import rex_uquot
    def _generate(self, value):
        if value is None:
            raise MissingData(f'Required parameter {self.data} is of invalid type {value}.')
        fmt = self._fmt
        cls = type(value)
        if isinstance(fmt, dict):
            fn = fmt.get(cls, None)
            if fn is None:
                fn = fmt.get(None, '!s')
        else:
            fn = '!s' if fmt is None else fmt
        if isinstance(fn, str):
            text = f'{{{fn}}}'.format(value)
        elif isinstance(fn, Callable):
            text = fn(value)
        else:
            raise FormattingError(f'Unsupported format {fn} of type {type(fn).__name__} for value {value} of type {type(value).__name__}.')
        quote = self._quote
        if quote is True:
            must_quote = True
        elif quote is False:
            must_quote = False
        elif quote == 'str':
            must_quote = issubclass(cls, str)
        elif quote == 'name':
            must_quote = self.rex_uquot.match(text) is None
        else:
            raise FormattingError(f'Unsupported quoting option {quote} of type {type(quote).__name__}. Choose a boolean of ["str", "name"].')
        yield f'"{text}"' if must_quote else text
    
class Name(Value):
    def __init__(self, quote='name', **kwargs):
        super().__init__(quote=quote, **kwargs) 


class Indices(DataSymbol):
    def __init__(self, base=0, data=None):
        super().__init__(data=data)
        self._base = base
    
    def _generate(self, indices):
        import numpy as np
        parts = []
        if len(indices)>0:
            diff = np.diff(indices)
            idx_non = np.where(diff!=1)[0]
            idx_chg = np.r_[idx_non,len(indices)]
            cnt_chg = np.diff(np.r_[0,idx_chg])
            beg = 0
            for end,cnt in zip(idx_chg, cnt_chg):
                offset = indices[beg] + self._base
                if cnt == 1:
                    part = str(offset)
                else:
                    part = f'{offset}-{offset+cnt-1}'
                beg = end
                parts.append(part)
        return ','.join(parts)


class Repeat(DataSymbol):
    
    def __init__(self, symbol, count=None, sep=' ', data=''):
        super().__init__(data=data)
        self._symbol = symbol
        self._count = count
        self._sep = sep
    
    def __generate__(self, data):
        sym = self._symbol
        sep = self._sep
        seq = sequence(data)
        if self._count is None:
            count = len(seq)
            if count is None:
                raise FormattingError(f'Unbounded repeat')
        else:
            count = self._count
        if count>0:
            yield from sym.__generate__(seq)
        for _ in range(1,count):
            if sep: yield sep
            yield from sym.__generate__(seq)

class Token(Symbol):
    '''
    @param unwrap For regular expression matches, returns the specific group.
    '''
    def __init__(self, text):
        self._text = text
    def __generate__(self, data):
        yield self._text
    def __str__(self):
        return f"{self._text!r}"
    
class Space(Token):
    def __init__(self):
        super().__init__(' ')

class Chain(Symbol):
    '''Parse in sequence all given symbols
    
    @param separate Controls consuming separators in between.
        False performs no consuming.
        True consumes whitespace
        Otherwise, the parameter is fed into rid.consume.
    '''
    def __init__(self, symbols, sep=' '):
        self._symbols = symbols
        self._sep = sep
    def __generate__(self, data):
        symbols = self._symbols
        sep = self._sep
        seq = sequence(data)
        try:
            if self._symbols:
                sym = symbols[0]
                yield from sym.__generate__(seq)
            for sym in symbols[1:]:
                if sep: yield sep
                yield from sym.__generate__(seq)
        except Exception as e:
            raise FormattingError(f'In {self} while generating {sym}.') from e

class Number(Symbol):
    def __init__(self, fmt='!s'):
        self._fmt = fmt
    def __generate__(self, data):
        yield f'{{0{self._fmt}}}'.format(data)

class Array(DataSymbol):
    '''Array symbol
    ''' 
    def __init__(self, quote=None, data=None):
        super().__init__(data=data)
        self._quote = quote
        
    def _generate(self, data):
        if data.dtype.name == 'str':
            sub = self.rex_escape.sub
            def fmt(value):
                escaped = sub(r'\"',value)
                return f'"{escaped}"'
        else:
            fmt = repr
        return ','.join(map(fmt, data))

class NewLine(str): pass

newline = NewLine('\n')

class LineChain(Chain):
    def __init__(self, symbols, sep=' ', end=newline):
        super().__init__(symbols, sep=sep)
        self._end = end
    
    def __generate__(self, *args, **kwargs):
        yield from super().__generate__(*args, **kwargs)
        yield self._end

class Header(Token):
    def __init__(self, name):
        super().__init__(f'@{name}')
        self.__name = name
    def __str__(self): return f'{self.__name}'

class Property(DataSymbol):
    def __init__(self, data='*'):
        super().__init__(data=data)
    SYMBOL = Chain((Name(),Token('='),Value(quote='name')),sep='')
    def _generate(self, pair):
        name, value = pair
        yield from self.SYMBOL.__generate__(props(name=name, value=value))
        
    def __str__(self): return self._name

class Properties(DataSymbol):
    def __init__(self, data=''):
        super().__init__(data=data)
    SYMBOL = Repeat(Property(data=''),sep=' ')
    def _generate(self, *pairs):
        yield from self.SYMBOL.__generate__(pairs)



class Payload(DataSymbol):
    def __init__(self, sparse=False, by_columns=False, dtype=None, data=None):
        super().__init__(data=data)
        self._dtype = dtype
        self._sparse = sparse
        self._by_columns = by_columns
    SYM_HDR = LineChain((Header('data'), Properties(data='*')))
    SYM_END = LineChain((Header('end'),Token('data')))
    
    class SparsePayloadLine(Symbol):
        SYMBOL = LineChain((Value(quote=False),Token(' ('), Indices(1),Token(') '),Array()),sep='')
        def __generate__(self, index, indices, values):
            yield from self.SYMBOL.__generate__(props(value=index+1, indices=indices, array=values))
    class DensePayloadLine(Symbol):
        SYMBOL = LineChain((Array(),))
        def __generate__(self, values):
            yield from self.SYMBOL.__generate__(props(array=values))
            
    def __generate__(self, payload):
        if self._by_columns is False:
            raise NotImplementedError('Can only write column-wise data.')
        
#        idl_sparse = np.r_[[isinstance(dtype, pd.SparseDtype) for dtype in data.dtypes]]
        if self._sparse:
            if isinstance(payload, pandas.DataFrame):
                def get_data(i):
                    arr = payload.iloc[:,i]
                    indices = arr.sp_index.to_int_index().indices
                    if len(indices) == 0:
                        return None
                    values = arr.sp_values
                    return (i, indices, values)
            else:
                raise NotImplementedError(f'No sparse matrices supported')
            sym = self.SparsePayloadLine()
        else:
            if isinstance(payload, pandas.DataFrame):
                get_data = lambda i: payload.iloc[:,i]
            else:
                get_data = lambda i: payload[:,i]
            sym = self.DensePayloadLine()
        num_data = payload.shape[1]
        mode = 'sparse' if self._sparse else 'dense'
        order = 'columns' if self._by_columns else 'rows'
        yield from self.SYM_HDR.__generate__(props(type=mode, order=order))
        for i in range(num_data):
            data = get_data(i)
            yield from sym.__generate__(data)
        yield from self.SYM_END.__generate__(pool(None))
        

class Matrix(DataSymbol):
    def __init__(self, sparse=None, by_columns=False, data='**'):
        super().__init__(data=data)
        self._by_columns = by_columns
        self._sym_data = Payload(sparse=sparse, by_columns=by_columns, dtype='int')
    SYM_META = LineChain((Header('matrix'), Name()))
    SYM_END = LineChain((Header('end'),Token('matrix'),Name()))
    def _generate(self, data, name):
        yield from self.SYM_META.__generate__(props(name=name))
        yield from self._sym_data.__generate__(data)
        yield from self.SYM_END.__generate__(props(name=name))

class Relation(DataSymbol):
    def __init__(self, data='**'):
        super().__init__(data=data)
    
    SYM_COLUMN = LineChain((Header('columns'), Repeat(Name(data=''))))
    ATTRIBUTE_TYPES = {'i': 'numeric', 'f':'numeric', 'U': 'string', 'S':'string'}
    STORAGE_TYPES = {'f': 'float', 'i': 'int'}
    def _generate_attr(self, kind, identifiers):
        try:
            attr_type = self.ATTRIBUTE_TYPES[kind]
        except KeyError: 
            raise NotImplementedError(f'Kind {kind} is not supported.')
        storage_type = self.STORAGE_TYPES.get(kind)
        str_storage = '' if storage_type is None else f' storage={storage_type}'
        str_attr = f'@attribute {identifiers} {attr_type}{str_storage}'
        return str_attr
    SYM_ATTR = LineChain((Header('attribute'), Indices(1), Name(data='attr_type'), Properties(data='properties*')))
    def _generate_meta(self, df, name):
        yield from self.SYM_COLUMN.__generate__(sequence(df.columns))
        kinds = numpy.r_[[d.kind for d in df.dtypes]]
        for kind in numpy.unique(kinds):
            indices = numpy.where(kind == kinds)[0]
            try:
                attr_type = self.ATTRIBUTE_TYPES[kind]
            except KeyError: 
                raise NotImplementedError(f'Kind {kind} is not supported.')
            storage_type = self.STORAGE_TYPES.get(kind)
            yield from self.SYM_ATTR.__generate__(props(indices=indices, attr_type=attr_type, properties=_Properties(storage=storage_type)))
    SYM_HDR = LineChain((Header('relation'), Name()))
    SYM_END = LineChain((Header('end'), Token('relation'), Name()))
    def _generate(self, data, name):
        yield from self.SYM_HDR.__generate__(props(data=data, name=name))
        yield from self._generate_meta(data, name)
        yield from Payload(by_columns=True).__generate__(data)
        yield from self.SYM_END.__generate__(props(name=name))
        
    
        

class DataSource(DataSymbol):
    def __init__(self, symbol, name, data='**'):
        super().__init__(data=data)
        self._symbol = symbol
        self._name = name
        
    SYMBOL = Name(data='')
    def _generate(self, name, data, manager, **kwargs):
        manager.add(name=name, symbol=self._symbol, data=data)
        yield from self.SYMBOL.__generate__(name)
    
    
    
    