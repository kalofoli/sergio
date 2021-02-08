'''
Created on Dec 11, 2017

@author: janis
'''

from os import path
from contextlib import contextmanager
from pickle import UnpicklingError, load, dump

from typing import Callable, Any, Optional, BinaryIO, Iterator


class FileCache:
    '''Cache loaded data using the pickle system'''

    def __init__(self, folder : str='cache') -> None:
        self._folder: str = folder
        self._store_active : bool = True
        self._load_active : bool = True

    '''
    Helpers
    '''

    @contextmanager
    def get_file(self, tag: str, write: bool=False) -> Iterator[BinaryIO]:
        '''Open a file given the cache tag'''
        file_name = path.join(self.folder, "cache-{0}.dat".format(tag))
        if write:
            with open(file_name, "wb") as fid:
                yield fid
        else:
            with open(file_name, "rb") as fid:
                yield fid

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
        with self.get_file(tag, write=True) as fid:
            dump(data, file=fid)
        return self

    def load(self, tag: str, loader: Callable[[], Any],
             store: Optional[bool]=None, reload: Optional[bool]=False) -> Any:
        '''Load a previously saved data from the file designated by the given tag'''
        reloaded: bool = False
        if not reload and self.load_active:
            try:
                with self.get_file(tag, write=False) as fid:
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

