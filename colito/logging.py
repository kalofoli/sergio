'''
Created on May 24, 2018

@author: janis
'''

import logging
from collections import namedtuple
from logging import getLogger, _levelToName, _nameToLevel, getLevelName as _getLevelName,\
    LoggerAdapter
from typing import Callable
from _collections import defaultdict

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

def formatLevelNames(multiline=False):
    lvls = sorted(_levelToName.items())
    if multiline:
        msg = format_multiline(lvls)
    else:
        msg = ','.join(f'{k}: {v}' for k,v in lvls)
    return msg

def getLevelName(lvl) -> str:
    try:
        if isinstance(lvl, int):
            lvl_name = _getLevelName(lvl)
        elif isinstance(lvl, str):
            if lvl_name not in _nameToLevel:
                raise KeyError(f'Invalid level name.')
            lvl_name = lvl
        else:
            raise TypeError('Invalid type')
    except Exception as exc:
        raise TypeError(f'Could not parse {lvl} as a logging level. Available levels are:\n{formatLevelNames(True)}.') from exc
    return lvl_name

def getLevelValue(lvl, exact=False) -> int:
    '''Resolve a level value.
    @param exact If True, it must correspond exactly to a level value.
    '''
    if isinstance(lvl, int):
        if exact and lvl not in _levelToName:
            raise ValueError(f'Level value {lvl} does not correspond to a given level name. Available levels are:\n{formatLevelNames(True)}.')
    else:
        try:
            lvl = _nameToLevel[lvl]
        except KeyError as e:
            raise ValueError(f'Level value {lvl} is not a valid level. Available levels are:\n{formatLevelNames(True)}.') from e
    return lvl



import sys

DefaultStreams = namedtuple('DefaultStreams', ('stdout','stderr'))
DEFAULT_STREAMS = DefaultStreams(stdout=sys.stdout, stderr=sys.stderr)

class StreamToLogger(object):
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
    
    def attach_standard(self, level_out = logging.INFO, level_err=logging.ERROR):
        sys.stdout = self.StreamToLogger(self, level_out)
        sys.stderr = self.StreamToLogger(self, level_err)
        

    def detach_standard(self):
        sys.stdout = DEFAULT_STREAMS.stdout
        sys.stderr = DEFAULT_STREAMS.stderr


class _LevelIsOn:

    def __init__(self, logger):
        self._logger = logger
    
    def __getattr__(self, lvl_name):
        lvl = getLevelValue(lvl_name)
        return self._logger.isEnabledFor(lvl)
    
    def __str__(self):
        return ','.join(f'{name}:{"Y" if getattr(self, name) else "N"}' for name in _nameToLevel)
    
    def __repr__(self):
        return f'<{self.__class__.__name__}:{self!s}>'


class SergioLogger(LoggerAdapter):
    '''A wrapper function for a logger, giving some convenience methods'''
    
    def __init__(self, logger, extra=None, *args, **kwargs) -> None:
        super().__init__(logger, extra=extra, *args, **kwargs)
        self.LOGGERS[logger.name] = logger
        #self.rlim = LoggerRateLimiter(self)
        self.ison = _LevelIsOn(self)
        
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

    def setFormatter(self, fmt=None):
        if fmt is None:
            fmt = logging.Formatter(self.default_format())
        for handler in self.handlers:
            handler.setFormatter(fmt)
                

    LOGGERS = dict()
    
    @classmethod
    def default_format(cls):
        len_level = max(len(key) for key in _nameToLevel.keys())
        if cls.LOGGERS:
            len_name = max(len(logger.name) for logger in cls.LOGGERS.values())
        else:
            len_name = ''
        return f'%(levelname)-{len_level}s %(name)-{len_name}s %(asctime)s||%(message)s'
        
DEFAULT_ADAPTER_FACTORY = None
def getModuleLogger(module_name):
    '''Create a Logger instance for the given module name.
    @param module_name The name of the module, typically __name__.
    '''
    tag = module_name.split('.')[-1]
    logger = getLogger(tag)
    if DEFAULT_ADAPTER_FACTORY is not None:
        return DEFAULT_ADAPTER_FACTORY(logger)
    else:
        return logger

def setAdapterFactory(adapter_factory=None):
    '''Set an adapter factory that will wrap returned loggers.
    This affects the getModuleLogger function.
    @param adapter_factory A callable that is initialised with a Logger instance or None.
    '''
    global DEFAULT_ADAPTER_FACTORY
    DEFAULT_ADAPTER_FACTORY = adapter_factory


from typing import Callable, Any
def to_stuple(what, sort:bool=False, join:str=None, formatter:Callable[[Any], str]=str):
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
