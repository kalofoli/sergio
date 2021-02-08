'''
Created on Feb 5, 2021

@author: janis
'''
import re
from typing import NamedTuple, Any
import numpy
import pandas
from . import _get_default_class_name, PATTERN_UQUOT
from .data import produce, ParserSequence
from colito.resolvers import make_enum_resolver
import enum
from . import Properties as _Properties
from .data import ParserDataResult as Result
import scipy
import warnings
import numpy as np

class ParseException(Exception): pass
class ExpectationError(ParseException): pass
class ParseEOF(ParseException): pass

rex_space = re.compile('[\t ]+')
rex_newline = re.compile('[\t ]*(\r\n|\n\r|\n)')

class _BufferedReaderBase:
    class State:
        MAX_PRINT_WIDTH = 100
        def __init__(self, pos_fid, off_buf, buf, last_line=0):
            self._pos_fid = pos_fid
            self._off_buf = off_buf
            self._buf = buf
            self._last_line = last_line
        def copy(self):
            return self.__class__(pos_fid=self._pos_fid, off_buf=self._off_buf, buf=self._buf, last_line=self._last_line)
        @property
        def pos_emulated(self): return self._pos_fid-len(self._buf) + self._off_buf
        @pos_emulated.setter
        def pos_emulated(self, pos):
            self._off_buf = pos - self.pos_min_seekable
        @property
        def pos_min_seekable(self): return self._pos_fid - len(self._buf) 
        @property
        def pos_max_seekable(self): return self._pos_fid - 1
        @property
        def buffer(self): return self._buf[self._off_buf:]
        @property
        def readable(self):
            '''Number of readable characters'''
            return len(self._buf) - self._off_buf 
        def peek(self, n):
            '''Return the next buffers to read in the buffer.
            
            The result is truncated to the available size.
            '''
            return self._buf[self._off_buf: self._off_buf+n]
        @property
        def next_line(self): return self.peek(self.len_next_line)
        @property
        def len_next_line(self): return self.buffer.find('\n')+1
        def _append(self, buf):
            self._buf += buf
            self._pos_fid += len(buf)
        def advance(self, n):
            '''Move the offset pointer forward.
            
            The buffer must be big enough to support this operation.
            '''
            if self._off_buf + n > self.pos_max_seekable:
                raise ParseEOF()
            buf_drop = self.peek(n)
            next_line = buf_drop.find('\n')
            if next_line >= 0:
                self._last_line = self._off_buf + next_line + 1
            self._off_buf += n
            return buf_drop
        def drop(self):
            '''Drop all lines before the current offset'''
            off = self._last_line
            self._buf = self._buf[off:]
            self._off_buf -= off
            self._last_line = 0
            return off
        def __str__(self):
            prn_buf = lambda buf,off: f'{buf[:off]}<{buf[off] if off<len(buf) else ""}>{buf[off+1:]}'
            if len(self._buf) < self.MAX_PRINT_WIDTH:
                str_buf = prn_buf(self._buf, self._off_buf)
            else:
                beg = max(self._off_buf-int(self.MAX_PRINT_WIDTH/4*1),0)
                off = self._off_buf - beg
                pbuf = self._buf[beg:beg+self.MAX_PRINT_WIDTH]
                str_buf = f'...{prn_buf(pbuf, off)}...'
            return f'pos:[min: {self.pos_min_seekable} emu: {self.pos_emulated} real: {self._pos_fid}] buf: {len(self._buf)} {str_buf!r}'
        def __repr__(self):
            return f'<{type(self).__name__} {self!s}>' 
    def __init__(self, fid, state=None):
        self._fid = fid
        self.__state = self._make_state(state)
    @property
    def state(self): return self.__state
    def _make_state(self, state):
        fid = self._fid
        if state is None:
            new_state = self.State(off_buf=0, pos_fid=fid.tell(), buf='')
            self._populate(state=new_state)
        else:
            if state._pos_fid != fid.tell():
                raise AssertionError(f'Stream has position {fid.tell()} but we were told it is at {state._pos_fid}')
            new_state = state.copy()
        return new_state
    def _populate(self, n=1, state=None):
        '''Fill the buffer to a minimum number of (readable) characters
        The loaded size ends in a newline.
        '''
        state = self.state if state is None else state
        readable = state.readable
        read = 0
        lines = []
        while readable + read < n:
            ln = self._fid.readline()
            if not ln:
                break 
            read += len(ln)
            lines.append(ln)
        buf = ''.join(lines)
        state._append(buf)
        assert state._pos_fid == self._fid.tell(), f'After read of {read} chars File is at {self._fid.tell()} instead of expected {state._pos_fid}.'
        return read
    def readline(self):
        return self.read(self.state.len_next_line)
    def read(self, n):
        _ = self._populate(n+1)
        state = self.state
        res = state.advance(n)
        state.drop()
        return res
    def peek(self, n):
        self._populate(n)
        res = self.state.peek(n)
        return res
    def peekline(self):
        return self.state.next_line
    def seek(self, pos):
        state = self.state
        if pos < state.pos_min_seekable or pos > state.pos_max_seekable:
            self._fid.seek(pos)
            self._state = self._make_state()
        else:
            state.pos_emulated = pos
    def tell(self):
        return self.state.pos_emulated
    @property
    def buf(self): return self.state._buf
    @property
    def fid(self): return self._fid
    def __str__(self):
        return f'state: "{self.state}"'
    def _rewind(self):
        self._fid.seek(self.state._pos_fid)
    def _set_state(self, state):
        self.__state = state
        
class _ParsingMixin:
    def __init__(self, *args, ignore_case=False, **kwargs):
        self._ignore_case = ignore_case
        super().__init__(*args, **kwargs)
    @property
    def ignore_case(self): return self._ignore_case
    def _compare(self, a, b, ignore_case=None):
        if ignore_case is None:
            ignore_case = self._ignore_case
        if ignore_case:
            return a.lower() == b.lower()
        else:
            return a == b 
    def _consume(self, what, ignore_case, err):
        line = self.peekline()
        if not line:
            if err:
                raise ParseEOF()
            else:
                return
        res = None
        if isinstance(what, re.Pattern):
            m = what.match(line)
            if m is not None:
                res = m
                self.read(m.end(0))
            else:
                if err:
                    raise ExpectationError(f'Could not match {line} against {what}.')
        else:
            n = len(what)
            txt = self.peek(n)
            if self._compare(txt, what, ignore_case):
                self.read(n)
                res = txt
            else:
                raise ExpectationError(f'Expected {what} but found {txt}.')
        return res
    rex_ws = re.compile('^\s+')
    def expect(self, what, ignore_case=False):
        return self._consume(what, ignore_case, err=True)
    def consume_ws(self, ignore_case=None):
        return self.consume(self.rex_ws, ignore_case=ignore_case)
    def consume(self, what, ignore_case=None):
        return self._consume(what, ignore_case, err=False)
    def __str__(self):
        sself = super().__str__()
        return f'{sself} Case: {"IGN" if self._ignore_case else "SAME"}'
    def parse(self, sym, ignore_case=None):
        with self.rewinder(ignore_case=ignore_case) as rid:
            res = sym.__parse__(rid)
            return res
    def rewinder(self, ignore_case = None):
        if ignore_case is None:
            ignore_case = self.ignore_case
        rid = RewindableReader(fid=self.fid, state=self.state, ignore_case=ignore_case, parent=self)
        return rid
    
    def _notify_end(self, expected, found):
        if expected != found:
            warnings.warn(f'Expected ending {expected} but found {found}')
        
class RewindableReader(_ParsingMixin, _BufferedReaderBase):
    def __init__(self, *args, parent, **kwargs):
        self._parent = parent
        super().__init__(*args, **kwargs)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._parent._rewind()
        else:
            self._parent._set_state(self.state)
        self._set_state(None)
        
    def __repr__(self):
        return f'<{type(self).__name__}({self!s})'
    

class ParserReader(_ParsingMixin, _BufferedReaderBase):
    def __repr__(self):
        return f'<{type(self).__name__}({self!s})>'
            

class Symbol:
    rex_name = re.compile('("(?P<qname>.*)"|(?P<name>[a-zA-Z_][a-zA-Z0-9]*))')
    def __init__(self, line_sep = '\n'):
        self._line_sep = line_sep
    def write(self, s, *args,**kwargs):
        it = self.generate(*args, **kwargs)
        l = 0
        try:
            while True:
                l += s.write(next(it) + self._line_sep)
        except StopIteration:
            pass
        return l

    def _make_separator(self, separate):
        if separate is True:
            separator = lambda rid: rid.consume_ws()
        elif separate is False:
            def separator(rid): pass
        else:
            separator = lambda rid : rid.consume(separate)
        return separator

class _Result(NamedTuple): # TODO: remove
    symbol: Symbol
    value: Any
    index: int


class DataSymbol(Symbol):
    def __init__(self, data=None):
        self._data = data if data is not None else _get_default_class_name(type(self))
        
    def __parse__(self, rid):
        res = self._parse(rid)
        data = produce(res, self._data)
        return data
    
    @property
    def data(self): return self._data


class _Value(DataSymbol): # TODO: remove
    '''
    @param quote: Controls quoting.
        For parsing: None to accept both quoted/unquoted, bool to enforce a choice.
        For writing it must be set to a boolean.
    @param to_int: Controls parsing as int during parsing.
        None: Try to silently convert if the value is not quoted.
        False: No conversion attempted: aalways returns a str.
        True: Enforce conversion and raise if not able to convert.
    @note Only doubles quotes are supported.
    '''
    
    def __init__(self, quote=None, to_int=None, data=None):
        super().__init__(data=data)
        self._quote = quote
        self._to_int = to_int
    rex_name_q = re.compile(r'"(?P<qname>(?:\\.|[^"\\]))*"')
    rex_name_quq = re.compile(r'"(?P<qname>(?:\\.|[^"\\]))*"|(?P<name>[a-zA-Z_][a-zA-Z0-9]*)|(?P<index>[+-]?[0-9]+)')
    rex_name_uq = re.compile('(?P<name>[a-zA-Z_][a-zA-Z0-9]*)|(?P<index>[+-]?[0-9]+)')
    def _parse(self, rid):
        to_int = self._to_int
        if self._quote is None:
            rex = self.rex_name_quq
        elif self._quote is True:
            rex = self.rex_name_q
        elif self._quote is False:
            rex = self.rex_name_uq
        m = rid.expect(rex)
        dct = m.groupdict()
        name = dct.get('name')
        if name is not None:
            if to_int is not False:
                name = self._try_convert(name, fail=False)
        else:
            name = dct.get('qname')
            if name is not None: # unquoted key
                if to_int is True:
                    name = self._try_convert(name, fail=True)
            else:
                name = dct['index']
                if to_int is False:
                    raise ParseException(f'Expected key but found {name}.')
                else:
                    name = self._try_convert(name, fail=True)
        return name

    @staticmethod
    def _try_convert(name,fail):
        try:
            name = int(name)
        except ValueError:
            if fail:
                raise ParseException(f'Could not __parse__ {name} as integer.')
        return name

class Indices(DataSymbol):
    def __init__(self, base=0, data=None):
        super().__init__(data=data)
        self._base = base
    rex = re.compile('([0-9]+(\s*-\s*[0-9]+)?)(\s*,\s*([0-9]+(\s*-\s*[0-9]+)?))*')
    def _parse(self, rid):
        indices = []
        m = rid.expect(self.rex)
        text = m.group(0)
        try:
            for rng in text.split(','):
                parts = rng.split('-')
                if len(parts) == 1:
                    indices.append([int(rng)])
                elif len(parts) == 2:
                    begin,last = int(parts[0]),int(parts[1])+1
                    indices.append(list(range(begin,last)))
                else:
                    raise ParseException(f'More than 2 endpoints in range {parts}')
        except Exception as e:
            raise ParseException(f'Near "{rng}"') from e
        return numpy.concatenate(indices) - self._base

class Repeat(DataSymbol):
    
    def __init__(self, symbol, lower=0, higher=None, sep=True, data=None):
        super().__init__(data=data)
        self._symbol = symbol
        self._lower = lower
        self._higher = higher
        self._sep = sep
    def _parse(self, rid):
        lo, hi = self._lower, self._higher
        count = 0
        separator = self._make_separator(self._sep)
        symbol = self._symbol
        results = ParserSequence()
        try:
            res = rid.parse(symbol)
            results.append(Result(symbol=symbol, index=0, value=res))
            count = 1
        except ParseException as _:
            pass
        while hi is None or count <= hi:
            try:
                with rid.rewinder() as subrid:
                    separator(subrid)
                    res = symbol.__parse__(subrid)
                    results.append(Result(symbol=symbol, index=count, value=res))
                    count += 1
            except ParseException as _:
                break
        if (lo is not None and lo > count) or (hi is not None and count > hi):
            l = []
            if lo is not None and lo>count: l.append(f'lower bound {lo}') 
            if hi is not None and hi<count: l.append(f'higher bound {hi}')
            raise ParseException(f'Found {count} instances of {symbol} which violates bounds {" and ".join(l)}.')
        return results

class Optional(DataSymbol):
    def __init__(self, symbol, data=None):
        self._symbol = symbol
        super().__init__(data=data)
    def _parse(self, rid):
        symbol = self._symbol
        result = None
        try:
            with rid.rewinder() as subrid:
                res = symbol.__parse__(subrid)
                result = Result(value=res,index=0,symbol=symbol)
        except ParseException as _:
            pass
        return result

class Token(Symbol):
    '''
    @param unwrap For regular expression matches, returns the specific group.
    '''
    def __init__(self, what, ignore_case=False, unwrap=False):
        self.__what = what
        self._ignore_case = ignore_case
        self._unwrap = unwrap
    @property
    def what(self): return self.__what
    def __parse__(self, rid):
        res = rid.expect(self.__what)
        if self._unwrap is not False:
            res = res.group(self._unwrap)
        return res
    def __str__(self):
        return f"R{self._what.pattern!r}" if isinstance(self._what, re.Pattern) else f"{self._what!r}"
    
class Header(Token):
    def __init__(self, text):
        super().__init__(f'@{text}')
        self._text = text
        
    def __str__(self): return self._text

class Space(Token):
    def __init__(self):
        super().__init__(rex_space)

class Union(DataSymbol):
    def __init__(self, symbols, data=''):
        self._symbols = symbols
        super().__init__(data=data)
    def _parse(self, rid):
        res = None
        for i,sym in enumerate(self._symbols):
            try:
                res = rid.parse(sym)
                break
            except ParseException as _:
                pass
        if res is None:
            raise ParseException(f'Could not reolve any of the given symbols: {self._symbols} near {rid}.')
        return Result(symbol=sym, value=res, index=i)
            
class Chain(DataSymbol):
    '''Parse in sequence all given symbols
    
    @param separate Controls consuming separators in between.
        False performs no consuming.
        True consumes whitespace
        Otherwise, the parameter is fed into rid.consume.
    '''
    def __init__(self, symbols, sep=True, data=''):
        self._symbols = symbols
        self._sep = sep
        super().__init__(data=data)
    def _parse(self, rid):
        symbols = self._symbols
        results = ParserSequence()
        if symbols:
            separator = self._make_separator(self._sep)
            sym = symbols[0]
            res = sym.__parse__(rid)
            results.append(Result(symbol=sym,value=res,index=0))
            for sidx,sym in enumerate(symbols[1:],1):
                separator(rid)
                res = sym.__parse__(rid)
                results.append(Result(symbol=sym,value=res, index=sidx))
        return results

class LineChain(Chain):
    def __init__(self, symbols, sep=rex_space, end=rex_newline, data=None):
        super().__init__(symbols=symbols, sep = sep, data=data)
        self._end = end
    def __parse__(self, rid):
        res = super().__parse__(rid)
        rid.expect(self._end)
        return res

class ValueType(enum.Flag): # Order MATTERS! Lower are matched first.
    NAME = enum.auto()
    SNQUOTED = enum.auto()
    DBQUOTED = enum.auto()
    UNQUOTED = enum.auto() # Only for arrays. Eats until "," or "\n"
    FLOAT = enum.auto()
    INT = enum.auto()
    
    NUMERIC = INT|FLOAT
    
    QUOTED = SNQUOTED|DBQUOTED
    STRING = QUOTED|NAME
    
    ANY = STRING|NUMERIC

VALUE_TYPE_RESOLVER = make_enum_resolver(ValueType)

class Value(DataSymbol):
    r'''Array symbol
    @param dtype If specified, this kind is expected during parsing.
    If the value 'detect' is specified, then the same format as the first entry is assumed throughout the array.
    
    >>> from io import StringIO
    >>> s = StringIO('3 4.4 g4 "5" It\'s all one,and another\n')
    >>> rid = ParserReader(s)
    >>> rid.parse(Value(accept=int))
    ParserProperties(value=3)
    >>> _ = rid.consume_ws()
    >>> rid.parse(Value(accept=ValueType.ANY))
    ParserProperties(value=4.4)
    >>> _ = rid.consume_ws()
    >>> res = rid.parse(Value(accept=ValueType.NAME,data=''))
    >>> type(res).__name__, res
    ('str', 'g4')
    >>> _ = rid.consume_ws()
    >>> res = rid.parse(Value(accept=ValueType.ANY,data=''))
    >>> type(res).__name__, res
    ('str', '5')
    >>> _ = rid.consume_ws()
    >>> res = rid.parse(Value(accept=ValueType.FLOAT)) # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    parsing.parsers.ParseException: Could not parse value for FLOAT.
    >>> res = rid.parse(Value(accept=ValueType.UNQUOTED,data=''))
    >>> type(res).__name__, res
    ('str', "It's all one")
    >>> _ = rid.consume_ws()
    '''
        
    def __init__(self, accept=ValueType.ANY, quoted=None, data=None):
        super().__init__(data=data)
        self._value_type = self._parse_type(accept=accept, quoted=quoted)
        self._parser = self._get_parser(self._value_type)

    stripper = lambda s: s[1:-1]
    TYPE2CONF = {
        ValueType.NAME: (r'''(?P<NAME>[a-zA-Z_][a-zA-Z_0-9]*)''',None),
        ValueType.DBQUOTED: (r'''"(?P<DBQUOTED>(\\.|[^"\\]))*"''',stripper),
        ValueType.SNQUOTED: (r"""'(?P<SNQUOTED>(\\.|[^'\\]))*'""",stripper),
#         ValueType.FLOAT: (r'(?P<FLOAT>[+-]?[0-9]*(\.[0-9][0-9]*|[0-9]+)([eE][+-]?[0-9]+)?)',float),
        ValueType.FLOAT: (r'(?P<FLOAT>[+-]?([0-9]+\.[0-9]*|[.]?[0-9]+)([eE][+-]?[0-9]+)?)',float),
        ValueType.INT: (r'(?P<INT>[+-]?[0-9]+)(?=[^.])',int),
        ValueType.UNQUOTED: (PATTERN_UQUOT, None)
    }
    TYPE2REXCONV = {}
    
    KIND2TYPE = {
        'f': ValueType.FLOAT,
        'i': ValueType.INT,
#                 'b': (rex_bool,bool),
        'S': ValueType.QUOTED,
        'U': ValueType.QUOTED
    }
    
    TAG2CONV = {k.name:v[1] for k,v in TYPE2CONF.items()}
    TYPE2PARSER = {}
    
    @property
    def value_type(self): return self._value_type
    @property
    def parser(self): return self._parser
    @classmethod
    def _get_tag_dependent(cls, value_type):
        '''If the parser converter depends on the rex group tag.'''
        return len(cls._get_rexconv(value_type)[1])>1

    @classmethod
    def _parse_type(cls, accept, quoted=None):
        if isinstance(accept, (str, ValueType)):
            value_type = VALUE_TYPE_RESOLVER.resolve(accept)
        else:
            accept = numpy.dtype(accept)
            value_type = cls.KIND2TYPE.get(accept.kind)
            if value_type is None: 
                raise NotImplementedError(f'No special handling for dtype {accept}.')
        if quoted is True and accept&ValueType.QUOTED:
            value_type |= ValueType.QUOTED
            value_type &= ~ValueType.UNQUOTED
        elif quoted is False:
            value_type &= ~ValueType.QUOTED
        return value_type
    
    @classmethod
    def _tag2type(cls, tag):
        return ValueType[tag]
        
    @classmethod
    def _get_rexconv(cls, value_type):
        res = cls.TYPE2REXCONV.get(value_type)
        if res is None:
            parts = VALUE_TYPE_RESOLVER.resolve_parts(value_type, allow_composite=False)
            pattern = '|'.join(cls.TYPE2CONF[part][0] for part in parts)
            rex = re.compile(pattern)
            convs = set(cls.TYPE2CONF[part][1] for part in parts)
            res = rex,convs
            cls.TYPE2REXCONV[value_type] = res
        return res
    
    def _get_parser(self, value_type):
        parser = self.TYPE2PARSER.get(value_type)
        if parser is None:
            TAG2CONV = self.TAG2CONV
            rex,convs = self._get_rexconv(value_type)
            if len(convs) == 1:
                conv = next(iter(convs))
                if conv is None:
                    def parser(rid):
                        m = rid.expect(rex)
                        if m is None:
                            return
                        return m.group(0),m
                else:
                    def parser(rid):
                        m = rid.consume(rex)
                        if m is None:
                            return
                        return conv(m.group(0)),m
            else:
                def parser(rid):
                    m = rid.expect(rex)
                    if m is None: return
                    dct = m.groupdict()
                    key = next(k for k,v in dct.items() if v is not None)
                    text = m.group(0)
                    conv = TAG2CONV.get(key)
                    value = text if conv is None else conv(text)
                    return value, m
            self.TYPE2PARSER[parser] = parser
        return parser
    
    def _parse(self, rid):
        res = self._parser(rid)
        if res is None:
            raise ParseException(f'Could not parse value for {self._value_type.name}.')
        value = res[0]
        return value
    
class Array(Value):
    r'''Parse an array of values
    
    >>> from io import StringIO
    >>> rid = ParserReader(StringIO("4,5,6,7,5,7 3.,4,5 '5','two' 5,here there,3.4,you are, end\n"))
    >>> rid.parse(Array(detect=True,data=''))
    [4, 5, 6, 7, 5, 7]
    >>> _ = rid.consume_ws()
    >>> with rid.rewinder():
    ...    rid.parse(Array(accept=ValueType.NUMERIC, data=''))
    [3.0, 4, 5]
    >>> rid.parse(Array(detect=True, accept=ValueType.NUMERIC, data=''))
    [3.0, 4.0, 5.0]
    >>> _ = rid.consume_ws()
    >>> rid.parse(Array(accept=int, data=''))
    Traceback (most recent call last):
    ...
    parsing.parsers.ParseException: Empty array encountered.
    >>> rid.parse(Array(accept='snquoted', data=''))
    ['5', 'two']
    
    >>> _ = rid.consume_ws()
    >>> rid.parse(Array(detect=False,accept='unquoted|numeric',data=''))
    [5, 'here there', 3.4, 'you are', 'end']
    '''
    def __init__(self, detect=None, accept=ValueType.ANY, allow_empty = False, data=None):
        super().__init__(accept=accept, data=data)
        self._detect = detect
        self._allow_empty = allow_empty
    
    rex_sep = re.compile('\s*,\s*')
    def _parse(self, rid):
        parser = self.parser
        rex_sep = self.rex_sep
        
        data = []
        res = parser(rid)
        if res is None:
            if not self._allow_empty:
                raise ParseException(f'Empty array encountered.')
            return data
        else:
            data.append(res[0])
        if self._detect and self._get_tag_dependent(self.value_type):
            dct = res[1].groupdict()
            tag, = (k for k,v in dct.items() if v is not None)
            value_type = self._tag2type(tag)
            parser = self._get_parser(value_type)
        while True:
            res = rid.consume(rex_sep)
            if not res:
                break
            res = parser(rid)
            if res is None:
                break
            value = res[0]
            data.append(value)
        return data

class Name(Value):
    def __init__(self, accept=ValueType.NAME, **kwargs):
        super().__init__(accept=accept, **kwargs) 

class Number(Value):
    def __init__(self, accept=ValueType.NUMERIC, **kwargs):
        super().__init__(accept=accept, **kwargs)
class Index(Number):
    def __init__(self, base=0, accept=ValueType.INT, **kwargs):
        super().__init__(accept=accept, **kwargs)
        self._base = base
    def _parse(self, rid):
        res = super()._parse(rid)
        return res - self._base
        

class Properties(DataSymbol):
    r'''Parse Properties
    
    >>> from io import StringIO
    >>> rid = ParserReader(StringIO("value=5 tag='name' name=g4\n"))
    >>> rid.parse(Properties(data=''))
    Properties(value=5,tag='name',name='g4')
    '''
    SYMBOL = Repeat(
        Chain((
                Name(),Token('='),Value()
            ), data='{}'
        ), data='[]')
    
    def _parse(self, rid):
        props = rid.parse(self.SYMBOL)
        res = _Properties(**{p.name:p.value for p in props})
        return res

class PayloadData(_Properties): pass
class SparseColumn(_Properties):
    sparse = True
    @property
    def rows(self): return numpy.max(self.indices)+1
    def __len__(self): return len(self.indices)
    def apply_df(self, df):
        dtype = df.iloc[:,self.index].dtype
        rows = df.shape[0]
        M = scipy.sparse.csc_matrix(
            (self.values,self.indices,(0,len(self))),
            dtype=dtype, shape=(rows,1))
        s = pandas.arrays.SparseArray.from_spmatrix(M)
        df.iloc[:,self.index] = s
        return df
    def get_triplets(self):
        n = len(self)
        i = numpy.r_[self.indices]
        v = numpy.r_[self.values]
        return i,(self.index,n),v
    def apply_array(self,M):
        M[np.r_[self.indices],self.index] = self.values
class DenseColumn(_Properties):
    sparse = False
    @property
    def rows(self): return len(self.values)
    def __len__(self): return len(self.values)
    def apply_df(self, df):
        df.iloc[:,self.index] = self.values
        return df
    def get_triplets(self):
        n = len(self)
        i = numpy.repeat(self.index,n), numpy.arange(n)
        v = numpy.r_[self.values]
        return i,(self.index,n),v
    def apply_array(self,M):
        M[:, self.index] = self.values
class Payload(DataSymbol):
    r'''
    
    >>> from io import StringIO
    >>> rid = ParserReader(StringIO(\
    ... """@data type=sparse order=columns missing=0
    ... 1 (1,2,5) "5","hello","five"
    ... 3 (2,4,9) 20,40,90
    ... @end data
    ... """))
    >>> pl = rid.parse(Payload(data=''))
    >>> len(pl.data), pl.props.type, pl.shape, pl.data[0].values
    (2, 'sparse', (9, 3), ['5', 'hello', 'five'])
    
    >>> rid = ParserReader(StringIO(\
    ... """@data order=columns
    ... 1 "5","hello",0,0,"five",0,0,0,0
    ... 3 0,20,0,40,0,0,0,0,90
    ... @end data
    ... """))
    >>> pl = rid.parse(Payload(data=''))
    >>> len(pl.data), pl.props.order, pl.shape, pl.data[0].values
    (2, 'columns', (9, 3), ['5', 'hello', 0, 0, 'five', 0, 0, 0, 0])
    '''
    def __init__(self, sparse=False, by_columns=False, dtype=None, data=None):
        super().__init__(data=data)
        self._dtype = dtype
        self._sparse = sparse
        self._by_columns = by_columns
    SYM_HDR = LineChain((Header('data'), Properties(data='=')),data='[0]')
    SYM_END = LineChain((Header('end'),Token('data')))
    class SparseColumnLine(DataSymbol):
        SYMBOL = LineChain((Index(1),Token('('), Indices(1),Token(')'),Array()),sep=rex_space,data='{}')
        def _parse(self, rid):
            res = rid.parse(self.SYMBOL)
            return _Properties(index=res.index, indices=numpy.r_[res.indices], values=res.array)
    class DenseColumnLine(DataSymbol):
        SYMBOL = LineChain((Index(1),Array(),),data='{}')
        def _parse(self, rid):
            res = rid.parse(self.SYMBOL)
            return _Properties(index=res.index, values=res.array)
    
    storage_types_allowed = {'dense','sparse','mixed'}            
    def _parse(self, rid):
#        idl_sparse = np.r_[[isinstance(dtype, pd.SparseDtype) for dtype in data.dtypes]]
        props = self.SYM_HDR.__parse__(rid)
        
        if props.order == 'columns':
            missing = props._get('missing', 0)
            syms = [self.SYM_END]
            store_type = props._get('type', 'mixed')
            if store_type not in self.storage_types_allowed:
                raise ValueError(f'Invalid property type specfied. Can only be one of {self.storage_types_allowed}.')
            cls_data = {}
            if store_type == 'sparse' or store_type == 'mixed':
                cls_data[len(syms)] = SparseColumn
                syms.append(self.SparseColumnLine(data=''))
            if store_type == 'dense' or store_type == 'mixed':
                cls_data[len(syms)] = DenseColumn
                syms.append(self.DenseColumnLine(data=''))
            us = Union(syms)
            data = []
            while True:
                res = us.__parse__(rid)
                if res.index == 0:
                    break
                data_cls = cls_data[res.index]
                val = res.unpack()
                d = data_cls(**val.__dict__, missing=missing)
                data.append(d)
            
            if 'rows' not in props:
                rows = max(l.rows for l in data)
            if 'columns' not in props:
                cols = max(l.index for l in data)+1
            pl = PayloadData(shape=(rows,cols), data = data, props=props)
        else:
            raise NotImplementedError()
        return pl


class Matrix(DataSymbol):
    r'''
    
    
    >>> from io import StringIO
    >>> rid = ParserReader(StringIO(\
    ... """@matrix edges
    ... @data type=sparse order=columns
    ... 1 (1,2,5) 5,4,2
    ... 3 (2,4,9) 20,40,90
    ... @end data
    ... @end matrix edges
    ... """))
    >>> M = rid.parse(Matrix(data=''))
    >>> M.shape, M.dtype, M.todense()[1,:]
    ((9, 3), dtype('int64'), matrix([[ 4,  0, 20]]))

    >>> rid = ParserReader(StringIO(\
    ... """@matrix edges
    ... @data type=dense order=columns
    ... 1 0,2,3,4,5
    ... 3 3,4,3,7,7
    ... @end data
    ... @end matrix edges
    ... """))
    >>> M = rid.parse(Matrix(data=''))
    >>> M.shape, M.dtype, M[3,:]
    ((5, 3), dtype('int64'), array([4, 0, 7]))
    '''
    
    def __init__(self, sparse=None, by_columns=False, data='**'):
        super().__init__(data=data)
        self._by_columns = by_columns
        self._sym_data = Payload(data='')
    SYM_HDR = LineChain((Header('matrix'), Name(data='=')),data='[0]')
    SYM_END = LineChain((Header('end'),Token('matrix'),Name(data='=')),data='[0]')
    def _parse(self, rid):
        from scipy.sparse import csc_matrix, csr_matrix
        name = self.SYM_HDR.__parse__(rid)
        data  = self._sym_data.__parse__(rid)
        end_name = self.SYM_END.__parse__(rid)
        rid._notify_end(name, end_name)
        props = data.props
        if 'type' in props:
            sparse = props.type in {'sparse', 'mixed'}
        else:
            sparse = any(d.sparse for d in data.data)
        if sparse:
            I,J,V = [],[], []
            for d in data.data:
                i,j,v = d.get_triplets()
                I.append(i)
                J.append(j)
                V.append(v)
            def get_ptr(X, n):
                offsets = np.zeros(n+1,int)
                idx,val = zip(*X)
                offsets[np.r_[idx]+1] = val
                ptr = np.cumsum(offsets)
                return ptr
            if isinstance(I[0], tuple): #row-wise
                idx,V = np.concatenate(J), np.concatenate(V)
                ptr = get_ptr(I, data.shape[0])
                cls = csr_matrix
            else:
                idx,V = np.concatenate(I), np.concatenate(V)
                ptr = get_ptr(J, data.shape[1])
                cls = csc_matrix
            
            sp_data = (V, idx, ptr)
            M = cls(sp_data, shape=data.shape)
        else:
            arrs = [np.dtype(type(d.values[0])) for d in data.data]
            dtype = np.find_common_type(arrs,[])
            M = np.zeros(shape=data.shape, dtype=dtype)
            for d in data.data:
                d.apply_array(M)
        return M


class AttributeHeader(Symbol):
    rex_kind = re.compile('(?i)(nominal|numeric|categorical|boolean|index)')
    SYMBOL = Chain([
            Token('@attribute'),
            Token(re.compile('\s+\(')), Indices(1), Token(re.compile('\)\s+')), 
            Token(rex_kind,unwrap=0),
            Optional(
                Chain((Space(), Properties()), sep=False)
            )
        ], sep=False)
    class AttrEntry(NamedTuple):
        kind : str
        dtype : numpy.dtype
        properties: dict 
    def __parse__(self, rid):
        vals = self.SYMBOL.__parse__(rid)
        indices, kind = vals[3].value,vals[6].value
        res_opts = vals[7].value
        if res_opts is not None:
            props = res_opts.value[1].value
        else:
            props = {}
        dtype = self.guess_dtype(kind, props)
        return indices, self.AttrEntry(kind, dtype=dtype, properties=props)
    KINDS2TYPE = {
        'nominal': int,
        'numeric': float,
        'categorical' : pandas.Categorical,
        'boolean': numpy.bool,
        'index': int}
    @classmethod
    def guess_dtype(cls, kind, props):
        dtype = cls.KINDS2TYPE[kind.lower()]
        if 'storage' in props:
            dtype = numpy.dtype(props.storage)
        return dtype     


        
class MetaDataParser(DataSymbol):
    rex_rel = re.compile('@relation\s')
    class ColumnHeader(DataSymbol):
        SYMBOL = LineChain((Header('columns'), Repeat(Name(),lower=1)))
        def __parse__(self, rid):
            res = self.SYMBOL.__parse__(rid)
            return res[1].value
    SYMBOL = Union((ColumnHeader(), AttributeHeader()))
        
    def __parse__(self, rid):
        rid.consume_ws()
        m = rid.expect(self.rex_rel)
        name = Name().__parse__(rid)
        rid.consume_ws()
        columns = None 
        attr_headers = []
        try:
            while True:
                res = us.__parse__(rid)
                if res is None:
                    break
                if res.index == 0:
                    columns = res.value
                else:
                    attr_headers.append(res.value)
                rid.consume_ws()
        except ParseException as _:
            pass
        if columns is None:
            raise ParseException(f'Columns entry is missing.')
        
        attrs = [None]*len(columns)
        for indices, attr in attr_headers:
            for i in indices:
                attrs[i] = attr
        meta = SimpleNamespace(columns = columns, attrs = attrs, name=name)
        return meta
    

        
   
class DataParser(Symbol):
    header = Chain((Token('@data'), Optional(Properties())))
    end = Token(re.compile('@end relation[^\n]*'))
    def __parse__(self, rid):
        res = self.header.__parse__(rid)
        if res[1].value is not None:
            props = res[1].value.value
        else:
            props = Properties.Properties()
        rid.consume_ws()
        if props.type == 'sparse':
            if 'rows' not in props:
                raise ParseException(f'Sparse data must specify a rows= property in the @data keyword.')
        if props.order == 'column':
            series = {}
            uc = Union([SparseColumn(), self.end])
            while True:
                res = rid.__parse__(uc)
                if res.index == 0:
                    sp_col = res.value
                    dtype = self._meta.attrs[sp_col.index].dtype
                    M = sp.sparse.csc_matrix(
                        (sp_col.values,sp_col.indices,(0,len(sp_col.indices))),
                        dtype=dtype,shape=(props.rows,1))
                    s = pd.arrays.SparseArray.from_spmatrix(M)
                    series[sp_col.index] = s
                else:
                    break
                rid.consume_ws()
            columns = self._meta.columns
            nrow,ncol = props.rows, len(columns)
            M = sp.sparse.csc_matrix(((),(),[0]*(ncol+1)),shape=(nrow,ncol))
            df = pd.DataFrame.sparse.from_spmatrix(M, columns=columns)
            convs = {}
            for sidx,attr in enumerate(self._meta.attrs):
                ser = series.get(sidx)
                if ser is not None:
                    df.iloc[:,sidx] = ser
                else:
                    convs.update({columns[sidx]:attr.dtype})
            df = df.astype(convs)
            df.name = self._meta.name
        return df


class Relation(DataSymbol):
    @classmethod
    def _parse(cls, rid):
        ms = MetaData()
        meta = ms.__parse__(rid)
        rid.consume_ws()
        data = Data(meta=meta).__parse__(rid)
        return data 


class DataSource(DataSymbol):
    def __init__(self, symbol, name, data=None):
        super().__init__(data=data)
        self._symbol = symbol
        self._name = name
        
    SYMBOL = Name(data='')
    def _parse(self, name, data, manager, **kwargs):
        manager.add(name=name, symbol=self._symbol, data=data)
        yield from self.SYMBOL.__generate__(name)


class DataBundle(Symbol):
    SYM_HDR = Chain((
        LineChain(())
                     ))
    def __parse__(self, rid):
        res = rid.parse(self.SYMBOL)
        

if __name__ == '__main__':
    import doctest
    doctest.testmod()

