'''
Created on May 24, 2018

@author: janis
'''

import typing
import logging
from collections import namedtuple
from logging import getLogger, _levelToName, _nameToLevel, getLevelName as _getLevelName,\
    LoggerAdapter
from typing import Callable
from _collections import defaultdict
from colito import NamedUniqueConstant
import functools

def format_multiline(table, *, col_sep=' ', row_sep='\n', row_indices=False, index_base=0, header=False, value_formatters = {}, text_formatters={}):
    '''Format tabular data, respecting column widths.
    @param header Specify headers.
        A boolean value of False indicates no header, while True treats the first row of the data as one.
        Otherwise, it must be an iterable of strings containing the column headers.
    @param value_formatters Specifies hwo to convert the value of each cell to a string.
        A dict of one entry for each cell index, or None for all unspecified indices.
        The value of each entry may either be a Callable of the form
            format(x) -> str
        or one of the python printing formats.
        The indices are zero-based, unless the row index is output, in which case the index 0 refers to the latter and the cell indices are 1-based.
        The default is '!s' for str conversion.
    @param text_formatters Specifies how to align the string of each cell to its available space.
        A dict of one entry for each cell index, or None for all unspecified indices.
        The value of each entry may either be a Callable of the form
            format(x, width:int) -> str
        or one of the alignment characters ">","<".,"^" as per the python printing rules.
        The indices are zero-based, unless the row index is output, in which case the index 0 refers to the latter and the cell indices are 1-based.
        The default is '>' for right-justified.
    @return The string of the formatted data.
        
    '''
    cell_widths = defaultdict(int)
    if header is True:
        num_header = 1
    elif header is False or header is None:
        num_header = 0
    else:
        table = [list(header)] + list(table)
        num_header = 1
    default_value_formatter = value_formatters.get(None, '!s')
    def make_value_formatter(cidx):
        formatter = value_formatters.get(cidx, default_value_formatter)
        if isinstance(formatter, str):
            if formatter and formatter[0] not in {'!',':'}:
                raise ValueError(f'Invalid formatter {formatter}. The first character must be either ":" or "!".')
            return f'{{0{formatter}}}'.format
        elif isinstance(formatter, Callable):
            return formatter
        else:
            raise TypeError(f'Provided invalid cell value formatter {formatter} for index {cidx} which is neither string not Callable, but instead {type(formatter).__name__}.')
        return value_formatters.get(cidx, default_value_formatter)
    _value_formatters = {}
    MISSING = object()
    def get_value_formatter(cidx):
        formatter = _value_formatters.get(cidx, MISSING)
        if formatter is MISSING:
            formatter = make_value_formatter(cidx)
            _value_formatters[cidx] = formatter
        return formatter
    
    lines = []
    cidx_offset = 1 if row_indices else 0
    for lidx, row in enumerate(table):
        if lidx<num_header:
            row_str = row
        else:
            row_str = [get_value_formatter(cidx+cidx_offset)(cell) for cidx,cell in enumerate(row)]
        if row_indices:
            str_idx = '' if lidx<num_header else get_value_formatter(0)(lidx-num_header+index_base)
            row_str = [str_idx] + row_str
        for cidx,str_cell in enumerate(row_str):
            cell_widths[cidx] = max(cell_widths[cidx], len(str_cell))
        lines.append(row_str)
    default_text_formatter = text_formatters.get(None, '>')
    def get_text_formatter(cidx):
        formatter = text_formatters.get(cidx, default_text_formatter)
        cell_width = cell_widths[cidx]
        if isinstance(formatter, str):
            return f'{{:{formatter}{cell_width}}}'.format
        elif isinstance(formatter, Callable):
            return lambda x: formatter(x, width=cell_width)
        else:
            raise TypeError(f'Provided invalid cell text formatter {formatter} for index {cidx} which is neither string not Callable, but instead {type(formatter).__name__}.')
        return formatter
    _text_formatters = [get_text_formatter(cidx) for cidx in range(len(lines))]
    lines = (col_sep.join(_text_formatters[cidx](cell) for cidx,cell in enumerate(line)) for line in lines)
    output = row_sep.join(lines)
    return output

def describe_levels(multiline=False):
    lvls = sorted(_levelToName.items())
    if multiline:
        msg = format_multiline(lvls)
    else:
        msg = ','.join(f'{k}: {v}' for k,v in lvls)
    return msg

def resolve_level_name(lvl, ignore_case=False) -> str:
    try:
        if isinstance(lvl, int):
            level_name = _getLevelName(lvl)
        elif isinstance(lvl, str):
            level_name = lvl.upper() if ignore_case else lvl
            if level_name not in _nameToLevel:
                raise KeyError(f'Invalid level name.')
        else:
            raise TypeError('Invalid type')
    except Exception as exc:
        raise TypeError(f'Could not parse {lvl} as a logging level. Available levels are:\n{describe_levels(True)}.') from exc
    return level_name

def resolve_level_value(lvl, exact=False, ignore_case=False) -> int:
    '''Resolve a level value.
    @param exact If True, it must correspond exactly to a level value.
    '''
    if isinstance(lvl, int):
        if exact and lvl not in _levelToName:
            raise ValueError(f'Level value {lvl} does not correspond to a given level name. Available levels are:\n{describe_levels(True)}.')
    else:
        try:
            if ignore_case:
                lvl = str(lvl).upper()
            lvl = _nameToLevel[lvl]
        except KeyError as e:
            raise ValueError(f'Level value {lvl} is not a valid level. Available levels are:\n{describe_levels(True)}.') from e
    return lvl



import sys

DefaultStreams = namedtuple('DefaultStreams', ('stdout','stderr'))
DEFAULT_STREAMS = DefaultStreams(stdout=sys.stdout, stderr=sys.stderr)

def _now_in_seconds():
    import time
    return time.time()


class _RateTracker:
    class LevelEntry:
        def __init__(self, level):
            self._delay = 0
            self._last = 0
            self._level = level
        @property
        def delay(self): return self._delay
        @delay.setter
        def delay(self, value):
            try: delay = float(value)
            except: raise ValueError(f'Cannot set rate for level {self.level_name} to {value} of type {type(value).__name__}.')
            if delay < 0:
                raise ValueError(f'Cannot set rate for level {self.level_name} to negative value.')
            self._delay = delay
        
        @property
        def level_name(self): return self._level if self._level is not None else 'default'
        @property
        def time_until_on(self):
            now = _now_in_seconds()
            return now - self.last
        @property
        def ison(self):
            now = _now_in_seconds()
            return now - self.last > self.delay
        @property
        def last(self): return self._last
        def mark(self, now = None):
            if now is None:
                now = _now_in_seconds()
            self._last = now
        def __repr__(self):
            return f'<{self.__class__.__name__} {self!s}>'
        def __str__(self):
            if self._delay:
                if self.ison:
                    txt = f'Y /{self.delay:g}s'
                else:
                    rem = self.time_until_on
                    txt = f'N {rem:.4g}/{self.delay:g}s({rem/self.delay*100:.2g}%)'
            else:
                txt = 'Y'
            return f'{self.level_name}:{txt}'
    
    def __init__(self):
        self._delays = {None:self.LevelEntry(None)}
        self._default = self._delays[None]
        
    def __setitem__(self, lvl, delay):
        self.set_delay(delay, lvl)
    
    def __getitem__(self, lvl):
        return self._resolve_level_entry(lvl)
    
    def set_delay(self, delay, level=None):
        if level is None:
            self._delays[None].delay = delay
        else:
            level_name = resolve_level_name(level, ignore_case=True)
            entry = self._delays.get(level_name)
            if entry is None:
                entry = self._delays[level_name] = self.LevelEntry(level_name)
            entry.delay = delay
    
    def mark_level_name(self, level_name):
        entry = self._delays.get(level_name)
        if entry is not None:
            entry.mark()
        self._default.mark()
    
    def mark_level(self, lvl):
        level_entry = self._resolve_level_entry(lvl)
        level_entry.mark()
    
    def _resolve_level_entry(self, lvl):
        level_name = resolve_level_name(lvl)
        entry = self._delays.get(level_name, self._default)
        return entry
    
    def ison(self, lvl):
        return self._resolve_level_entry(lvl).ison
    
    def __str__(self):
        return ', '.join(map(str, self._delays.values()))
    def __repr__(self):
        return f'<{self.__class__.__name__}:{self!s}>'


GLOBAL_RATE_TRACKER = _RateTracker()

class StreamLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
    
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
            
    def flush(self):
        pass

class LevelIsOn:

    def __init__(self, logger):
        self._logger = logger
    
    def __getattr__(self, level_name):
        lvl = resolve_level_value(level_name.upper())
        return self._logger.isEnabledFor(lvl)
    
    def __dir__(self): return tuple(map(str.lower, _nameToLevel.keys()))
    def __str__(self):
        return ','.join(f'{name}:{"Y" if getattr(self, name) else "N"}' for name in _nameToLevel)
    def __repr__(self):
        return f'<{self.__class__.__name__}:{self!s}>'

class RateLimitedLogger(LoggerAdapter):

    def __init__(self, logger, extra=None, *args, **kwargs) -> None:
        super().__init__(logger, extra=extra, *args, **kwargs)
        self._trackers = {}
        self._register_overrides()
        self.ison = LevelIsOn(self)
        
    def _register_overrides(self):
        for key, value in self.__class__.__bases__[0].__dict__.items():
            if key.upper() in _nameToLevel:
                tracker = self._make_tracker(key)
                tracker = functools.update_wrapper(tracker, value)
                setattr(self, key, tracker)
    
    def __getattr__(self, level_name):
        if level_name.upper() not in _nameToLevel or level_name.lower() != level_name:
            raise AttributeError(f"AttributeError: '{self}' object has no attribute '{level_name}'")
        self.level_name = self._make_tracker(level_name)
        return self.level_name
        
    def __dir__(self): return sorted(tuple(map(str.lower, _nameToLevel.keys())) + tuple(self.__class__.__dict__.keys()))

    def isEnabledFor(self, level):
        return self.delay[level].ison

    def _make_tracker(self, fn_name):
        level_name = fn_name.upper()
        def tracker(*args, **kwargs):
            self.delay.mark_level_name(level_name)
            getattr(self.logger, fn_name)(*args, **kwargs)
        tracker.__name__ = fn_name
        return tracker
    delay = GLOBAL_RATE_TRACKER
    
    def __str__(self):
        return str(self.delay)
    def __repr__(self):
        return f'<{self.__class__.__name__} {self!s}>'

class ColitoLogger(LoggerAdapter):
    '''A wrapper function for a logger, giving some convenience methods
    
    >>> log = getModuleLogger(__name__, factory=ColitoLogger)
    >>> log
    <ColitoLogger __main__ (WARNING)>
    >>> log.setLevel(10)
    >>> log
    <ColitoLogger __main__ (DEBUG)>
    >>> log.ison
    <LevelIsOn:CRITICAL:Y,FATAL:Y,ERROR:Y,WARN:Y,WARNING:Y,INFO:Y,DEBUG:Y,NOTSET:N>
    >>> dir(log.ison)
    ['critical', 'debug', 'error', 'fatal', 'info', 'notset', 'warn', 'warning']
    >>> log.rlim
    <RateLimitedLogger default:Y>
    >>> log.rlim.delay['info']=.100
    >>> log.rlim
    <RateLimitedLogger default:Y, INFO:Y /0.1s>
    >>> log.rlim.info("Hello")
    >>> log.rlim.ison.info
    False
    >>> import time
    >>> time.sleep(.100)
    >>> log.rlim.ison.info
    True
    '''
    LOGGER_NAMES:typing.Set[str] = set()
    
    def __init__(self, logger, extra=None, *args, **kwargs) -> None:
        super().__init__(logger, extra=extra, *args, **kwargs)
        self.ison = LevelIsOn(self)
        self._last_standard = None
        self.rlim = RateLimitedLogger(self)
        self.LOGGER_NAMES.add(logger.name)
        
    def add_stderr(self, fmt=None):
        from logging import StreamHandler
        logger = self.logger
        hdlrs = tuple(h for h in logger.handlers if isinstance(h, StreamHandler))
        if not hdlrs:
            hdlr = StreamHandler()
            fmt = self.default_format() if fmt is None else fmt 
            hdlr.setFormatter(logging.Formatter(fmt))
            hdlr.level = 0
            logger.addHandler(hdlr)
        else:
            hdlr = hdlrs[0]
        return hdlr

    def add_file(self, file, fmt=None):
        from logging import FileHandler
        logger = self.logger
        fh = FileHandler(file)
        fmt = self.default_format() if fmt is None else fmt 
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
        return fh

    def setFormatter(self, fmt=None):
        if fmt is None:
            fmt = logging.Formatter(self.default_format())
        for handler in self.handlers:
            handler.setFormatter(fmt)
                
    def __getattr__(self, level_name):
        if level_name.upper() not in _nameToLevel or level_name.lower() != level_name:
            raise AttributeError(f"AttributeError: '{self}' object has no attribute '{level_name}'")
        self.level_name = self._make_fn(level_name)
        return self.level_name

    def _make_fn(self, fn_name):
        level = resolve_level_value(fn_name, exact=True, ignore_case = True)
        def tracker(*args, **kwargs):
            self.log(level, *args, **kwargs)
        tracker.__name__ = fn_name
        return tracker

    
    @classmethod
    def default_format(cls):
        len_level = max(len(key) for key in _nameToLevel.keys())
        len_name = max(map(len, cls.LOGGER_NAMES)) if cls.LOGGER_NAMES else ''
        return f'%(levelname)-{len_level}s %(name)-{len_name}s %(asctime)s||%(message)s'

    def attach_standard(self, level_out = logging.INFO, level_err=logging.ERROR):
        last_standard = sys.stdout, sys.stderr
        sys.stdout = StreamLogger(self, level_out)
        sys.stderr = StreamLogger(self, level_err)
        return last_standard
        
    @staticmethod
    def detach_standard(self):
        if self._last_standard is None:
            sys.stdout = DEFAULT_STREAMS.stdout 
            sys.stderr = DEFAULT_STREAMS.stderr
        else:
            sys.stdout, sys.stderr = self._last_standard

    def log(self, level, msg, *args, **kwargs):
        """
        Delegate a log call to the underlying logger, after adding
        contextual information from this adapter instance.
        """
        if self.rlim.isEnabledFor(level):
            self.rlim.delay.mark_level(level)
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)
        
    def __enter__(self):
        level = self.getLevel()
        self._last_standard = self.attach_standard(level, level)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach_standard()
        

_USE_DEFAULT_FACTORY = NamedUniqueConstant('USE_DEFAULT_FACTORY')
DEFAULT_ADAPTER_FACTORY = ColitoLogger
def getModuleLogger(module_name, factory = _USE_DEFAULT_FACTORY):
    '''Create a Logger instance for the given module name.
    @param module_name The name of the module, typically __name__.
    '''
    tag = module_name.split('.')[-1]
    logger = getLogger(tag)
    if factory is _USE_DEFAULT_FACTORY:
        factory = DEFAULT_ADAPTER_FACTORY
    if factory is not None:
        return factory(logger)
    else:
        return logger

def getRootLogger(factory = _USE_DEFAULT_FACTORY):
    '''Create a Logger instance for the given module name.
    @param module_name The name of the module, typically __name__.
    '''
    logger = getLogger()
    if factory is _USE_DEFAULT_FACTORY:
        factory = DEFAULT_ADAPTER_FACTORY
    if factory is not None:
        return factory(logger)
    else:
        return logger

def setAdapterFactory(adapter_factory=None):
    '''Set an adapter factory that will wrap returned loggers.
    This affects the getModuleLogger function.
    @param adapter_factory A callable that is initialised with a Logger instance or None.
    '''
    global DEFAULT_ADAPTER_FACTORY
    DEFAULT_ADAPTER_FACTORY = adapter_factory


import typing
def to_stuple(what, sort:bool=False, join:str=None, formatter:typing.Callable[[typing.Any], str]=str):
    out = tuple(map(formatter, what))
    if sort:
        if callable(sort):
            key = sort
        else:
            key = None
        out = tuple(sorted(out, key=key))
    if join:
        out = join.join(out)
    return out


if __name__ == '__main__':
    import sys
    del sys.path[0]
    import doctest
    doctest.testmod()
    