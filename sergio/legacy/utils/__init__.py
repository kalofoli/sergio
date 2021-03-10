'''
Created on Dec 18, 2017

@author: janis
'''
import re
import operator
import logging
from array import array

from typing import Iterable, Type, Dict, Any, List, cast, Sequence, Callable, \
    Generic, TypeVar, Tuple, TYPE_CHECKING, NamedTuple
from builtins import property, staticmethod
from numpy import ndarray, fromiter, zeros
from collections import OrderedDict

from sdcore.summarisable import SummaryOptions, Summarisable
from sdcore.logging import getLogger
import functools
import enum
import numpy
from itertools import takewhile

log = getLogger(__name__.split('.')[-1])

if TYPE_CHECKING:
    from sdcore.predicates import Predicate


class Utils:
    '''Convenience utilities'''

    @classmethod
    def wrap(cls, what, allow_single=False, container=tuple, fun=None):
        '''Wrap a list of elements or just an element into the given container type'''
        if isinstance(what, str) or not isinstance(what, Iterable):
            if allow_single:
                if fun is not None:
                    res = fun(what)
                else:
                    res = what
        else:
            if fun is not None:
                what = map(fun, what)
            res = container(what)
        return res
    
    GT_VALUE_TYPE_REX = re.compile('[0-9]+')

    @classmethod
    def get_gt_value_type(cls, value_type):
        from numpy import dtype
        if isinstance(value_type, str):
            return value_type
        elif isinstance(value_type, Type):
            return value_type.__name__
        elif isinstance(value_type, dtype):
            return cls.GT_VALUE_TYPE_REX.sub('', value_type)
        else:
            raise TypeError(f'Could not convert {value_type} to a type.')


class StatisticsMeta(type):

    class Counter:

        def __init__(self, initial_value=0):
            self.initial_value = initial_value
            self.name = 'unnamed'

        @property
        def name_private(self):
            return '_' + self.name

        def __repr__(self):
            return f'<Counter for "{self.name}" with initial value {self.initial_value}>'
        
        def to_getter(self):
            return self.getter
        
        def getter(self, inst):
            return getattr(inst, self.name_private)
            
        def setter(self, inst, value):
            return setattr(inst, self.name_private, value)
        
    class Updater:
        '''Updater'''

        def __init__(self, counter: 'StatisticsMeta.Counter', step=1, fun=operator.iadd) -> None:
            self.counter: StatisticsMeta.Counter = counter
            self.step = step
            self.fun = fun

        def __repr__(self):
            return f'<Updater for {self.counter} with step {self.step} and function {self.fun}>'
        
        def to_method(self):

            def update(inst, steps:int=1):
                setattr(inst, self.counter.name_private, self.fun(getattr(inst, self.counter.name_private), steps * self.step))
                return inst

            return update
    
    @classmethod
    def updater(mcs, counter: 'Counter', step=1, fun=operator.iadd):

        def decorate(decoratee):  # pylint: disable=unused-argument
            return mcs.Updater(counter=counter, step=step, fun=fun)

        return decorate
        
    def __new__(cls, cls_name, bases, dct):
        cls_new = super(StatisticsMeta, cls).__new__(cls, cls_name, bases, dct)
        counters = dict(filter(lambda x:isinstance(x[1], cls.Counter), dct.items()))
        updaters = dict(filter(lambda x:isinstance(x[1], cls.Updater), dct.items()))
        # Make a property with just a getter
        for name, counter in counters.items():
            counter.name = name
            setattr(cls_new, name, property(counter.to_getter()))
        for name, updater in updaters.items():
            setattr(cls_new, name, updater.to_method())
        cls_new.__counters__ = counters
        updaters = tuple(map(lambda x:x[0], filter(lambda x:isinstance(x[1], cls.Counter), dct.items())))
        return cls_new

    def __call__(self, *args, **kwargs):
        inst = super().__call__(*args, **kwargs)
        for counter in inst.__counters__.values():
            setattr(inst, counter.name_private, counter.initial_value)
        return inst


class StatisticsBase(Summarisable, metaclass=StatisticsMeta):
    __counters__: Dict[str, StatisticsMeta.Counter] = {}

    def __repr__(self):
        text = ' '.join(map(lambda name:'{0}:{{0.{0}}}'.format(name).format(self), self.__counters__.keys()))
        return f'<{self.__class__.__name__}: {text}>'
    
    def _reset(self):
        cls: Type[StatisticsBase] = self.__class__
        counters: Dict[str, StatisticsMeta.Counter] = cls.__counters__
        for counter in counters.values():
            setattr(self, counter.name_private, counter.initial_value)
        return self
    
    def summary_dict(self, options:SummaryOptions):
        return self.summary_from_fields(self.__counters__)

    def _asdict(self):
        return OrderedDict((name,counter.getter(self)) for name,counter in self.__counters__.items())
    
    def _copy(self):
        statistics = self.__class__()
        statistics += self
        return statistics

    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f'Can only add same-class statistics and you are adding a {other.__class__.__name__} object.')
        for name, counter in self.__counters__.items():
            # Counters are class-level objects. Therefore, they need to be provided the instance to function
            val_self = counter.getter(self)
            val_other = other.__counters__[name].getter(other)
            counter.setter(self, val_self + val_other)
        return self
    
    def __add__(self, other):
        instance = self.__class__()
        instance += self
        instance += other
        return instance
    
    def __eq__(self, other):
        return self._asdict() == other._asdict()

        
def as_predicate_objects(index_producer: Callable[..., Tuple[int, ...]]) -> Callable[..., Tuple['Predicate', ...]]:

    def predicate_producer(self, *args, **kwargs) -> Tuple['Predicate', ...]:
        predicate_indices = index_producer(self, *args, **kwargs)
        if predicate_indices is None:
            predicate_objects = None
        else:
            predicate_objects = tuple(self.language.predicate_objects(predicate_indices))
        return predicate_objects

    return predicate_producer


def property_predicate_objects(index_producer: Callable[..., Tuple[int, ...]]):
    predicate_producer = as_predicate_objects(index_producer)
    return property(predicate_producer)


class Cache(dict):
    
    class Attributes:

        def __init__(self, cache: 'Cache') -> None:
            dict.__setattr__(self, '_cache', cache)
            self._cache: Cache = cache  # for pylint
        
        def __dir__(self):
            dir_orig = super(Cache.Attributes, self).__dir__()
            return list(self._cache.keys()) + dir_orig
        
        def __repr__(self):
            return f'<{self.__class__.__name__} with {len(self._cache)} key(s): {list(self._cache.keys())}>'
        
        def __setattr__(self, name, value):
            if name in self.__dict__:
                dict.__setattr__(self, name, value)
            else:
                self._cache[name] = value
        
        def __getattr__(self, name):
            return self._cache[name]
            
        def __delattr__(self, name):
            del self._cache[name]
    
    def __init__(self, *args, enabled=True, **kwargs):
        self._enabled = enabled
        super(Cache, self).__init__(*args, **kwargs)
        self._attributes = Cache.Attributes(self)
    
    def __setitem__(self, *args, **kwargs):
        if self._enabled:
            super(Cache, self).__setitem__(*args, **kwargs)
    
    @property
    def a(self):
        return self._attributes
    
    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        
    @classmethod
    def from_spec(cls, spec):
        if isinstance(spec, Cache):
            return spec
        elif isinstance(spec, bool):
            return Cache(enabled=spec)
        elif isinstance(spec, dict):
            return Cache(spec)
        elif spec is None:
            return Cache()
        else:
            raise RuntimeError('Cannot create cache. Specify either a cache, a dict or a bool')
        
    def __repr__(self):
        return f'<{self.__class__.__name__}:{"E" if self.enabled else "D"} ' + super(Cache, self).__repr__() + '>'
    
    def get(self, key, other=None):
        if self._enabled:
            return super().get(key, other)
        else:
            return other

    def __getitem__(self, key):
        if self._enabled:
            return super().__getitem__(key)
        else:
            raise KeyError(key)

    @classmethod
    def cached_property(cls, value_getter):
        name = value_getter.__name__

        def wrapped_getter(self):
            enabled = self.cache.enabled
            if enabled:
                try:
                    value = self.cache[name]
                    return value
                except KeyError:
                    pass
            value = value_getter(self)
            if enabled:
                self.cache[name] = value
            return value

        return property(wrapped_getter) 


class property_eval_once:
    '''Property that gets evaluated only once. Can also decorate a classmethod, a staticmethod or a method.'''
    def __init__(self, wrapped):
        if isinstance(wrapped, classmethod):
            self._instanceless = True
            self._static = False
            self._func = wrapped.__func__
        elif isinstance(wrapped, staticmethod):
            self._instanceless = True
            self._static = True
            self._func = wrapped.__func__
        else:
            self._instanceless = False
            self._static = False
            self._func = wrapped
        # camouflage as the the wrapped function (docstring, annotations, etc)
        functools.update_wrapper(self, self._func)
        
    def __get__(self, inst, cls):
        instanceless = self._instanceless
        if not instanceless and inst is None:
            return self
        
        func = self._func
        obj = cls if instanceless else inst
        value = func() if self._static else func(obj)
        setattr(obj, func.__name__, value)
        return value


ET = TypeVar('ET')

from sys import stderr


def print_array(arr, select=(), no_print=False):
    nrows, ncols = arr.shape
    hdr_body = ''.join(f'{":" if (i%10)==0 and i>0 else ""}{i%10}' for i in range(ncols))
    header = f' {hdr_body} '
    selected = set(select)

    def print_char(v, ir, ic):
        if ic in selected:
            fmt = '.O' 
        else:
            fmt = ' *'
        txt = fmt[v]
        if ic % 10 == 0 and ic > 0:
            txt = f'|{txt}'
        return txt

    def print_row(row, ir):
        body = ''.join(print_char(char, ir, ic) for ic, char in enumerate(row))
        extra = ("\n" + header) if (ir % 5 == 4) else ""
        return f'[{body}]{extra}'

    body = '\n'.join(print_row(arr[ir, :], ir) for ir in range(nrows))
    txt = '\n'.join([header, body])
    if not no_print:
        stderr.write(txt + '\n')
    return txt


class Indexer(Generic[ET]):
    
    class MissingKeyResolution(enum.Enum):
        COMPLAIN = enum.auto()
        IGNORE = enum.auto()
        DEFAULT = enum.auto()
    
    
    def __init__(self, index=None, missing=MissingKeyResolution.COMPLAIN):
        obj2id: Dict[ET, int] 
        id2obj: List[ET]
        if index is None:
            obj2id = {}
            id2obj = []
        else:
            if isinstance(index, Indexer):
                indexer = cast(Indexer, index)
                obj2id = indexer._obj2id 
                id2obj = indexer._id2obj
            elif isinstance(index, dict):
                obj2id = cast(Dict[Any, int], index)
                id2obj = Indexer.dict2index(obj2id)
            elif isinstance(index, (Sequence, Iterable)):
                id2obj = list(index)
                obj2id = dict(zip(id2obj, range(len(id2obj))))
            else:
                raise RuntimeError('Cannot parse index argument')
        self._obj2id: Dict[Any, int] = obj2id
        self._id2obj: List[Any] = id2obj
        self._missing = missing
        
    @property
    def items(self) -> List[ET]:
        return self._id2obj
    
    def clear(self) -> 'Indexer[ET]':
        self._obj2id.clear()
        self._id2obj.clear()
        return self
   
    def update(self, object_: ET) -> 'Indexer[ET]':
        return self.update_iterable((object_,))
    
    def update_iterable(self, objects:Iterable[ET]) -> 'Indexer[ET]':
        new = list(filter(lambda obj: obj not in self._obj2id, objects))
        new_pairs = zip(new, range(len(self), len(self) + len(new)))
        self._id2obj += new
        self._obj2id.update(dict(new_pairs))
        return self

    def __len__(self) -> int:
        return len(self._id2obj)

    def get_index(self, what, missing=MissingKeyResolution.DEFAULT) -> int:
        '''Get the index of an object or fail if the key does not exist.
        
        If a default value is specified, return this instead of an error in case of a missing key.'''
        missing_tag = object()
        res = self._obj2id.get(what, missing_tag)
        if res is missing_tag:
            MKR = Indexer.MissingKeyResolution
            if missing is MKR.DEFAULT:
                missing = self._missing
            if res is MKR.COMPLAIN:
                raise KeyError(f'Could not find key {what}.')
            else:
                res = missing
        return res
    
    def get_indices(self, what) -> Iterable[int]:
        return map(self._obj2id.__getitem__, what)

    def get_index_array(self, what) -> ndarray:
        return fromiter(self.get_indices(what), int)
    
    def get_object(self, what) -> ET:
        return self._id2obj[what]

    def get_objects(self, what) -> Iterable[ET]:
        return map(self._id2obj.__getitem__, what)

    def asdict(self) -> Dict[ET, int]:
        return self._obj2id.copy()
        
    __call__ = get_index
    __getitem__ = get_object
    
    def __hasitem__(self, what):
        return what in self._obj2id 
    
    @classmethod
    def dict2index(self, dct):
        index = fromiter(dct.values(), int)
        items_tmp = array(tuple(dct.keys()), object)
        items = zeros(len(index), object)
        items[index] = items_tmp
        return list(items)
    
    class IndexerMapping(NamedTuple):
        indexer:'Indexer'
        map_new2old:numpy.ndarray
        map_old2new:numpy.ndarray
        is_old_selected = property(lambda self:self.map_old2new != -1, None, 'A boolean array indicating if the corresponding old element is selected.')
        
    def select_indices(self, indices):
        index_local = numpy.arange(len(self))
        map_new2old = index_local[indices]
        obj = self.get_objects(map_new2old)
        indexer = Indexer(obj)
        map_old2new = numpy.empty(len(self), int)
        map_old2new[:] = -1
        map_old2new[map_new2old] = numpy.arange(len(map_new2old))
        
        return self.IndexerMapping(indexer=indexer, map_new2old=map_new2old, map_old2new=map_old2new)
    
    def select_objects(self, what):
        indices = numpy.fromiter(self.get_indices(what), int)
        return self.select_indices(indices)
    
    def __repr__ (self):
        return f'<{self.__class__.__name__} with {len(self)} objects>'

    def to_data_frame(self, column_index='index', column_items='item', object_index=True):
        '''Return a pandas DataFrame with the given index data'''
        import pandas
        if object_index:
            index = pandas.Series(self.items, name=column_index)
            df = pandas.DataFrame(numpy.arange(len(self)),index = index, columns=[column_items])
        else:
            index = pandas.RangeIndex(0, len(self), name=column_index)
            df = pandas.DataFrame(self.items,index = index, columns=[column_items])
        return df
   
def to_dict(classes, sort=False, fn=str):
    '''Deprecated. Use to_dict_values'''
    if sort:
        classes = sorted(classes, key=fn)
        cls_dict = OrderedDict
    else:
        cls_dict = dict
    names = list(map(fn, classes))
    return cls_dict(zip(names, classes))


def to_dict_values(values, key_fn=str, sort=False, ordered=True):
    '''Make a dict from a (computed key) to the provided values
    
    :param values: An iterable of elements to use as dict values.
    :param key_fn: function to create the keys of the dict based on each provided element
    :param sort bool: Whether the keys should be sorted. Implies ordered.
    :param ordered bool: If true, an OrderedDict is created.
    :return: a dict of keys computed by key_fn over each element. If sort is true, the dict is ordered.
    '''
    pairs = list((key_fn(value), value) for value in values)
    if sort:
        pairs = pairs.sort(key=lambda x:x[0])
        if ordered is None:
            ordered = True
    cls_dict = OrderedDict if ordered is None or ordered else dict
    return cls_dict(pairs)


def to_class_dict(what, sort=False):
    return to_dict(what, sort=sort, fn=lambda cls: cls.__name__)


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


class ClassCollection:

    def __init__(self, name, classes):
        self._classes = tuple(classes)
        self._name: str = str(name)
        self._tags: Dict[str, type] = to_dict_values(classes, key_fn=self._get_tag, ordered=True)
        self._class_names: Dict[str, type] = to_dict_values(classes, key_fn=lambda cls:cls.__name__, ordered=True)
    
    @classmethod
    def _get_tag(self, cls):
        if hasattr(cls, 'tag'):
            return cls.tag
        else:
            return cls.__name__
        
    @property
    def tags(self) -> Tuple[str, ...]:
        '''List available tags'''
        return tuple(self._tags.keys())
    
    @property
    def class_names(self) -> Tuple[str, ...]:
        '''List available class names'''
        return tuple(self._class_names.keys())
    
    @property
    def classes(self) -> Tuple[type, ...]:
        return self._classes
    
    def has_tag(self, tag) -> bool:
        return tag in self._tags
    
    def get_class_from_tag(self, tag) -> type:
        if not self.has_tag(tag):
            raise KeyError(f'Collection {self._name} has no tag {tag}. Try one of: {",".join(self.tags)}.')
        return self._tags[tag]
    
    def has_class_name(self, tag) -> bool:
        return tag in self._tags

    def get_class_from_name(self, class_name):
        if not self.has_class_name(class_name):
            raise KeyError(f'Collection {self._name} has no class_name {class_name}. Try one of: {",".join(self.class_names)}.')
        return self._class_names[class_name]
    
    def get_class_title(self, cls) -> str:
        '''Get a friendly name from a class'''
        # this could also have been a classmethod/staticmethod, but kept as a member in case overrides or state is added later. (E.g.: headers, camelisation, etc)
        if hasattr(cls, 'name'):
            return cls.name
        else:
            return cls.__name__

    def get_class_tag(self, cls) -> str:
        '''Get a tag name from a class'''
        # this could also have been a classmethod/staticmethod, but kept as a member in case overrides or state is added later. (E.g.: headers, camelisation, etc)
        if hasattr(cls, 'tag'):
            return cls.tag
        else:
            return cls.__name__
    
    def __repr__(self):
        return f'<{self.__class__.__name__}[{self._name}] with {len(self._tags)} tags>'


class MultilinePaddingFilter(logging.Filter):
    '''A logging filter that appends a padding on the beginning of each line continuation in the logged message''' 

    def __init__(self, padding=' '):
        ''':param indent str: the string to prepend in the line continuations'''
        super(MultilinePaddingFilter, self).__init__()
        self._padding = padding

    def filter(self, record):
        if hasattr(record, 'msg') and record.msg:
            msgs = record.msg.split('\n')
            msgs = msgs[:1] + list(self._padding + txt for txt in msgs[1:])
            msg = '\n'.join(msgs)
            record.msg = msg
        return super(MultilinePaddingFilter, self).filter(record)

    
class DateTime(Summarisable):
    summary_name = 'date'
    _fields = ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond')

    def __init__(self, dt=None):
        import datetime
        super().__init__()
        self._datetime = dt if dt is not None else datetime.datetime.now()
        
    def summary_dict(self, options: SummaryOptions):
        return self.summary_from_fields(self._fields, self._datetime)


import os
class RuntimeEnvironment(Summarisable):
    _fields = ('date', 'hostname', 'username', 'pid', 'cwd', 'cpu_count', 'git_version')
    
    def __init__(self):
        self.date = DateTime()

    @property
    def pid(self):
        return os.getpid()

    @property
    def cwd(self):    
        return os.getcwd()

    @property_eval_once
    @staticmethod
    def cpu_count():
        return os.cpu_count()
    
    @property_eval_once
    @staticmethod
    def username():
        try:
            import pwd
            return pwd.getpwuid(os.getuid()).pw_name
        except:
            return 'unknown'
    
    @property_eval_once
    @staticmethod
    def hostname():
        try:
            import socket
            return socket.gethostname()
        except:
            return 'unknown'
    
    @property_eval_once
    @staticmethod
    def git_version():
        try:
            import subprocess
            cwd = os.path.dirname(os.path.realpath(__file__))
            sp = subprocess.Popen('git describe --tags --always --dirty --long'.split(' '),
                                  cwd=cwd,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            txt_out, txt_err = sp.communicate()
            if sp.returncode != 0:
                log.warning(f'Failed to get git version: {txt_err.decode("latin")}')
                return f'git-return-{sp.returncode}'
            else:
                return txt_out.decode('latin').strip()
        except Exception as e:
            return f'error-{e[0]}'
    
    def summary_dict(self, options: SummaryOptions):
        return self.summary_from_fields(self._fields)


import numpy as np
from typing import NamedTuple
class RemainingIndexMapper:
    '''
    Get item goes rem -> ful
    Call goes ful -> rem
    '''
    class Removal(NamedTuple):
        idx_rem: int
        size:int
        map_rem_old2new: np.ndarray
        mapper: 'RemainingIndexMapper'
        @property
        def has_many(self):
            return self.map_rem_old2new is not None
        @property
        def idx_ful(self):
            if self.has_many:
                return self.mapper[slice(self.size+len(self.idx_rem)-1,self.size-1,-1)]
            else:
                return self.mapper[self.size]
        def apply_numpy(self, arr_rem, axis=0):
            def cut(i):
                idx = [slice(None)]*arr_rem.ndim
                idx[axis] = i
                return tuple(idx)
            idx_rem = self.idx_rem
            if self.has_many:
                size_move = len(self.map_rem_old2new)
                idx_rem_dst = self.map_rem_old2new
                arr_rem[cut(idx_rem_dst)] = np.array(arr_rem[cut(slice(size_move))])
                # old = np.array(arr_rem[cut(idx_rem)])
                old = arr_rem[cut(idx_rem_dst[idx_rem])]
            else:
                idx_rem_lst = self.size
                old = np.array(arr_rem[cut(idx_rem)])
                arr_rem[cut(idx_rem)] = arr_rem[cut(idx_rem_lst)]
                arr_rem[cut(idx_rem_lst)] = old
            return old
        def apply_sequence(self, seq_rem):
            idx_rem = self.idx_rem
            idx_rem_lst = self.size
            old = seq_rem[idx_rem]
            seq_rem[idx_rem] = seq_rem[idx_rem_lst]
            seq_rem[idx_rem_lst] = old
            return old
            
    
    def __init__(self, number:int, remaining:ndarray=None, invalid_index = -1):
        """@param number: count of full object collection.
            @param remain: an array of remaining indices mapping them to collection indices
        """
        self._invalid_index: int = int(invalid_index)
        map_ful2rem: ndarray
        map_rem2ful: ndarray
        if remaining is None:
            map_ful2rem = np.arange(0,number)
            map_rem2ful = np.arange(0,number)
        else:
            map_ful2rem = np.empty(number, int)
            map_rem2ful = np.array(remaining, int)
            map_ful2rem[:] = self.invalid_index
            map_ful2rem[map_rem2ful] = np.arange(len(map_rem2ful))
        self._map_ful2rem: ndarray = map_ful2rem
        self._map_rem2ful: ndarray = map_rem2ful
        self._num_rem = len(map_rem2ful)
    
    @property
    def num_full(self):
        return len(self._map_ful2rem)
    
    @property
    def num_remain(self):
        return len(self)
    
    @property
    def invalid_index(self):
        return self._invalid_index
    
    def __len__(self):
        return self._num_rem
    
    def remove(self, idx_rem):
        if idx_rem > len(self):
            raise ValueError(f'Index exceeds bounds (was {idx_rem}>={len(self)}).')
        idx_ful = self[idx_rem]
        idx_rem_lst = len(self)-1
        idx_ful_lst = self[idx_rem_lst]
        self._map_ful2rem[idx_ful_lst] = idx_rem
        self._map_rem2ful[idx_rem] = idx_ful_lst
        self._map_rem2ful[idx_rem_lst] = idx_ful
        self._map_ful2rem[idx_ful] = self._invalid_index
        self._num_rem -= 1
        return self.Removal(idx_rem=int(idx_rem), mapper = self, size=len(self), map_rem_old2new=None)
    
    @staticmethod
    def _generate_move_indices(idx_del, size, k):
        """Finds the last k remaining indices that are not marked for deletion"""
        is_mov = np.ones(size, bool)
        is_mov[idx_del] = False
        num_mov_remain = k
        for idx in range(size)[::-1]:
            if is_mov[idx]:
                yield(idx)
                num_mov_remain -= 1
            if num_mov_remain == 0:
                break
    
    
    def remove_many(self, idx_rem):
        idx_rem = np.array(idx_rem)
        size = len(self)
        len_del = len(idx_rem)
        size_new = size - len_del
        
        idx_rem_surv = np.ones(size, bool)
        idx_rem_surv[idx_rem] = False
        idx_ful = np.array(self[idx_rem])

        map_rem_old2new = np.empty(size, int)
        map_rem_old2new[idx_rem_surv] = np.arange(size_new)
        map_rem_old2new[~idx_rem_surv] = np.arange(size-1,size_new-1,-1)
        
        self._map_rem2ful[map_rem_old2new] = np.array(self._map_rem2ful[:size])
        # Replace the deleted rem indices with their ful targets
        # self._map_rem2ful[size_new:size] = self._map_ful2rem[self._map_rem2ful[size_new:size]]
        # Invalidate deleted ful indices
        self._map_ful2rem[idx_ful] = self._invalid_index
        # Fix the target indices in the surviving ful targets
        idx_rem_surv_new = self._map_rem2ful[:size_new]
        self._map_ful2rem[idx_rem_surv_new] = np.arange(size_new)
        
        self._num_rem -= len_del
        return self.Removal(idx_rem=idx_rem, mapper = self, size=len(self), map_rem_old2new=map_rem_old2new)
    
    def full2remain(self, idx_ful):
        return self._map_ful2rem[idx_ful]
    def remain2full(self, idx_rem):
        return self._map_rem2ful[idx_rem]
    def as_dataframe(self):
        from pandas import DataFrame
        rem = np.arange(len(self))
        ful = self[:len(self)]
        return DataFrame({'rem':rem,'ful':ful})
    
    def __bool__(self):
        return len(self)>0
    
    @property
    def remain(self):
        return np.array(self[:len(self)])

    @property
    def removed(self):
        return np.array(self[len(self):])[::-1]
    
    def copy(self):
        return self.__class__(size=self.number, remain=self(slice(None)))
    __getitem__ = remain2full
    __call__ = full2remain
    
    def __repr__ (self):
        return f'<{self.__class__.__name__} with {self.num_full} objects ({len(self)} remain)>'
