'''
Created on Dec 11, 2017

@author: janis
'''

from os import path
from contextlib import contextmanager
from pickle import UnpicklingError, load, dump

from typing import Callable, Any, Optional, BinaryIO, Iterator
import re


class FileCache:
    '''Cache loaded data using the pickle system'''
    __max_tag_size__ = None
    def __init__(self, folder : str='cache') -> None:
        self._folder: str = folder
        self._store_active : bool = True
        self._load_active : bool = True

    '''
    Helpers
    '''

    def open(self, tag: str, *args, **kwargs):
        '''Open a file given the cache tag'''
        file_name = path.join(self.folder, "cache-{0}.dat".format(tag))
        return open(file_name, *args, **kwargs)
    

    '''
    Properties
    '''

    @property
    def store_active(self):
        '''Control storing the result after computing it'''
        return self._store_active

    @store_active.setter
    def store_active(self, value: bool):
        self._store_active = value

    @property
    def load_active(self):
        '''Control consulting cache for a given tag'''
        return self._load_active

    @load_active.setter
    def load_active(self, value: bool):
        self._load_active = value

    @property
    def folder(self):
        '''Folder to save cache files into.'''
        return self._folder

    @folder.setter
    def folder(self, value: str):
        if path.exists(value):
            if not path.isdir(value):
                raise ValueError("Path {0} is not a directory.".format(value))
        self._folder = value

    '''
    Methods
    '''

    def store(self, tag: str, data: Any) -> 'Cache':
        '''Store a given data to the file designated by the given tag'''
        with self.open(tag, 'wb') as fid:
            dump(data, file=fid)
        return self

    def load(self, tag: str, loader: Callable[[], Any],
             store: Optional[bool]=None, reload: Optional[bool]=False) -> Any:
        '''Load a previously saved data from the file designated by the given tag'''
        reloaded: bool = False
        if not reload and self.load_active:
            try:
                with self.open(tag, 'rb') as fid:
                    data = load(fid)
                reloaded = False
            except (UnpicklingError, OSError) as _:
                data = loader()
                reloaded = True
        else:
            data = loader()
        if store is None:
            store = self.store_active
        if store and reloaded:
            self.store(tag, data)
        return data
    
    if False:
        rex_keys_1 = re.compile('[_]')
        rex_keys_2 = re.compile('[aoueiAOUEI_]')
        def make_tag_from_dict(self, dct, pressure):
            if pressure == 0:
                spars = ','.join(f'{k}:{v!r}' for k,v in dct.items())
            else:
                pars = []
                for k,v in dct.items():
                    if v is None:
                        continue
                    if isinstance(v, float):
                        sv = f'{v:.6g}'
                    elif isinstance(v,bool):
                        sv = 'T' if v else 'F'
                    else:
                        sv = f'{v!s}'
                    if pressure >= 2:
                        rex = self.rex_keys_1 if pressure<3 else self.rex_keys_2
                        k = rex.sub('',k)
                    pars.append(f'{k}:{sv}')
                spars = ','.join(pars)
                if pressure >= 4:
                    spars = spars[:self.__max_tag_size__]
            return self.make_tag(spars)
        rex_normalise = re.compile('^[]a-zA-Z0-9:,(){}[]+')
        def make_tag(self, what):
            tag = self.rex_normalise.sub('_', what)
            return tag


DEFAULT_FILE_CACHE = FileCache()

class ValueCache(dict):
    
    class Attributes:

        def __init__(self, cache: 'ValueCache') -> None:
            dict.__setattr__(self, '_cache', cache)
            self._cache: ValueCache = cache  # for pylint
        
        def __dir__(self):
            dir_orig = super().__dir__()
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
        super(ValueCache, self).__init__(*args, **kwargs)
        self._attributes = ValueCache.Attributes(self)
    
    def __setitem__(self, *args, **kwargs):
        if self._enabled:
            super(ValueCache, self).__setitem__(*args, **kwargs)
    
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
        if isinstance(spec, ValueCache):
            return spec
        elif isinstance(spec, bool):
            return cls(enabled=spec)
        elif isinstance(spec, dict):
            return cls(spec)
        elif spec is None:
            return cls()
        else:
            raise RuntimeError('Cannot create cache. Specify either a cache, a dict or a bool')
        
    def __repr__(self):
        return f'<{self.__class__.__name__}:{"E" if self.enabled else "D"} ' + super(ValueCache, self).__repr__() + '>'
    
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

