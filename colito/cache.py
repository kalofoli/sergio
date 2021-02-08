'''
Created on Dec 11, 2017

@author: janis
'''

from os import path
from contextlib import contextmanager
from pickle import UnpicklingError, load, dump

from typing import Callable, Any, Optional, BinaryIO, Iterator


class Cache:
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


DEFAULT_CACHE = Cache()
