'''
Created on Dec 11, 2017

@author: janis
'''

from os import path
from contextlib import contextmanager
import pickle

from typing import Callable, Any, Optional, BinaryIO, Iterator
import re
import datetime
import typing

class CacheMiss(Exception): pass

class CacheBackend:
    pass

class CacheBackendSQLite(CacheBackend):
    def __init__(self, file):
        import sqlite3
        self._file = file
        self._db = sqlite3.connect(file)
        self._init()
        
    def __del__(self):
        self._db.close()
    def _reset(self):
        self._execute('drop table if exists cache_data;')
        self._init()
    def _init(self):
        self._execute('create table if not exists cache_data(tag VARCHAR primary key not null, data blob not null, created date, last_accessed date)')

    def _execute(self, sql, *args):
        return self._db.execute(sql, *args)
    def _fetch_one(self, sql, *args):
        cur = self._execute(sql, *args)
        return cur.fetchone()
    def _fetch_all(self, sql, *args):
        cur = self._execute(sql, *args)
        return cur.fetchall()
    
    def __contains__(self, tag):
        row = self._fetch_one("select count(tag) from cache_data where tag=?;", tag)
        return row[0] == 1
    
    def store(self, tag, data, date=None, overwrite=True):
        if date is None:
            date = datetime.datetime.now()
        method = 'insert or replace' if overwrite else 'insert'
        self._execute(f'{method} into cache_data (tag, data, created, last_accessed) values(?,?,?,?)', [tag, data, date, None])
        self._db.commit()
    def times(self, tag=None):
        if tag is not None:
            cond, args = ' where tag=:tag', [{'tag':tag}]
        else:
            cond, args = '',[]
        rows = self._fetch_all(f"select tag, created, last_accessed from cache_data {cond}", *args)
        as_date = lambda x: datetime.datetime.fromisoformat(x) if x is not None else None
        as_time = lambda row: (row[0], as_date(row[1]), as_date(row[2]))
        times = [as_time(row) for row in rows]
        if tag is not None:
            times = times[0]
        return times
    def tags(self):
        cur = self._execute('select tag from cache_data;')
        return [row[0] for row in cur.fetchall()]
    def rm(self, tag):
        cur = self._execute('delete from cache_data where tag=:tag',{'tag':tag})
        self._db.commit()
        return cur.rowcount==1
    def load(self, tag):
        rows = self._fetch_one("select data from cache_data where tag=:tag;",{'tag':tag})
        if rows:
            self._execute("update cache_data set last_accessed=:now where tag=:tag;", {'tag':tag, 'now':datetime.datetime.now()})
            self._db.commit()
            return rows[0]
        else:
            raise CacheMiss(f'No tag "{tag}" in cache.')

class PersistentCache:
    def __init__(self, backend):
        self._backend: CacheBackend = backend
        self._store_active : bool = True
        self._load_active : bool = True
    
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
    
    def _encode(self, data):
        return data
    
    def _decode(self, data):
        return data
    
    def store(self, tag: str, data) -> 'Cache':
        '''Store a given data under the given tag'''
        encoded = self._encode(data)
        self._backend.store(tag, encoded)
        return self

    def load(self, tag: str, loader: typing.Callable[[], typing.Any],
             store: bool=None, reload: bool=False):
        '''Load or retrieve a given tag.'''
        reloaded: bool = False
        if not reload and self.load_active:
            try:
                encoded = self._backend.load(tag)
                data = self._decode(encoded)
                reloaded = False
            except CacheMiss as _:
                data = loader()
                reloaded = True
        else:
            data = loader()
        if store is None:
            store = self.store_active
        if store and reloaded:
            self.store(tag, data)
        return data
    
    def __contains__(self, tag):
        return self._backend.__contains__(tag)

class ObjectCache(PersistentCache):
    
    def _decode(self, encoded):
        return pickle.loads(encoded)
    def _encode(self, data):
        return pickle.dumps(data)

import os
class FileBackend:
    def __init__(self, folder, prefix=''):
        self._folder = folder
        self._prefix = ''
    
    def get_filename(self, tag):
        return os.path.join(self._folder, f'{self._prefix}{tag}')
    
    def open(self, tag):
        file = self.get_filename(tag)
        return open(file, 'rb')

    def load(self, tag):
        try:
            with self.open(tag) as fid:
                data = fid.read()
            return data
        except OSError as e:
            raise CacheMiss(f'Failed to open file for tag "{tag}"') from e
    def store(self, tag, data, overwrite=True):
        file = self.get_filename(tag)
        mode = 'wb' if overwrite else 'wb+'
        with open(file, mode) as fid:
            fid.write(data)
        
    def __contains__(self, tag):
        try:
            with self.open(tag):
                return True
        except OSError:
            return False

class CacheMixinFile:
    '''Cache loaded data using the pickle system'''
    def __init__(self, folder:str='cache', prefix='cache',*args, **kwargs):
        backend = FileBackend(folder=folder, prefix=prefix)
        super().__init__(backend = backend, *args, **kwargs)
    
    def open(self, tag: str, *args, **kwargs):
        '''Open a file given the cache tag'''
        return self._backend.open(tag)
    
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

def _curl_get_raw(url, decode=None, gunzip=None):
    import pycurl
    from io import BytesIO
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(pycurl.URL, url)
    c.setopt(pycurl.WRITEFUNCTION, buffer.write)
    c.perform()
    return c, buffer.getvalue()

def curl_get(url, decode=None, gunzip=None):
    _, data = _curl_get_raw(url)
    if gunzip is True:
        import gzip
        data = gzip.decompress(data)
    if decode is not None:
        if decode is True:
            decode = 'latin'
        data = data.decode(decode)
    return data

class URLFileCache(CacheMixinFile, PersistentCache):
    def __init__(self, folder:str='cache', prefix='',*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    class URLLoader():
        '''A loader for URLs'''
        def __init__(self, url, gunzip=None, decode=None):
            self._url = url
            self._gunzip = gunzip
            self._decode = decode
        def __call__(self):
            return curl_get(
                url=self._url,
                gunzip=self._gunzip,
                decode=self._decode
            )
        def __repr__(self):
            return f'<{self.__class__.__name__} for "{self._url}">'

    def load_url(self, url, tag=None, store=True, reload=False, gunzip=None, decode=None):
        if tag is None:
            tag = os.path.basename(url)
        return self.load(tag, loader=self.URLLoader(url, gunzip=gunzip, decode=decode), store=store, reload=reload)
    
class FileObjectCache(CacheMixinFile, ObjectCache): pass


DEFAULT_FILE_CACHE = FileObjectCache()

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
            try:
                return self._cache[name]
            except KeyError as e:
                raise AttributeError( "{self.__class__.__name__} object has no attribute '{name}'")
            
            
        def __delattr__(self, name):
            try:
                del self._cache[name]
            except KeyError as e:
                raise AttributeError()
    
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

class CacheException(Exception): pass
def cache_to_disk(cache_dir, file=None, fmt=None, recompute=False):
    '''A decorator that caches the value of this function to disk
    
    @param file Either a filename to use (possibly formatted with the arguments of the function)
        or a callable witht he same signature as fn that returns a file name.
    @param fmt A boolean to specify whether the file parameter is to be formatted.
    @param recompute Controls computing the result. When True no loading is attempted,
        when False the value is computed whenever loading fails. Otherwise it is assumed
        to be a Callable[Any, List[Any], Dict[str,Any]] with signature:
            recompute(data, args, kwargs)
        invoked with the loaded data, a bool indicated whether a loading has been sucessfull,
        and the parameters of the function to cache.
    '''
    if isinstance(fmt, str) and file is None:
        file = fmt
        fmt = True
    def decorator(fn):
        nonlocal file, fmt
        if cache_dir is None:
            return fn
        if fmt is False and file is None:
            raise ValueError('To avoid formatting a file must be specified.')
        if isinstance(file, str) and fmt is False:
            get_file = lambda *args, **kwargs: file
        elif isinstance(file, Callable):
            get_file = file
            if fmt is True:
                raise NotImplementedError('Cannot format the output of a callable file argument.')
        else:
            import inspect
            sig = inspect.signature(fn)
            
            if file is None:
                spars = ','.join(f'{p}={{{p}!r}}' for p in sig.parameters)
                file = f'{fn.__name__}({spars}).pickle'
            sig = inspect.signature(fn)
            def get_file(*args, **kwargs):
                pars = sig.bind(*args, **kwargs)
                pars.apply_defaults()
                arguments = pars.arguments
                return file.format(**arguments)
        from functools import wraps
        @wraps(fn)
        def wrapper(*args, **kwargs):
            fname = get_file(*args, **kwargs)
            path = os.path.join(cache_dir, fname)
            must_store = must_load = True
            if recompute is not True:
                loaded = False
                data = None
                try:
                    with open(path, 'rb') as fid:
                        data = pickle.load(fid)
                        loaded = True
                        must_store = False
                except (FileNotFoundError, EOFError):
                    pass
                except Exception as e:
                    raise CacheException(f'While reading file {fname}') from e
                if loaded:
                    if recompute is False:
                        must_load = False
                    else:
                        must_load = recompute(data, args, kwargs)
                        must_store = True
                else:
                    must_load = True
            if must_load:
                data = fn(*args, **kwargs)
            if must_store:
                with open(path, 'wb') as fid:
                    pickle.dump(data, fid)
            return data
        return wrapper
    return decorator

def _make_file_getter(fn, file, fmt):
    if fmt is False and file is None:
        raise ValueError('To avoid formatting a file must be specified.')
    if isinstance(file, str) and fmt is False:
        get_file = lambda *args, **kwargs: file
    elif isinstance(file, Callable):
        get_file = file
        if fmt is True:
            raise NotImplementedError('Cannot format the output of a callable file argument.')
    else:
        import inspect
        sig = inspect.signature(fn)

        if file is None:
            spars = ','.join(f'{p}={{{p}!r}}' for p in sig.parameters)
            file = f'{fn.__name__}({spars}).pickle'
        sig = inspect.signature(fn)
        def get_file(*args, **kwargs):
            pars = sig.bind(*args, **kwargs)
            pars.apply_defaults()
            arguments = pars.arguments
            return file.format(**arguments)
    return get_file

class LockedException(Exception):
    def __init__(self, msg, fname=None, pid=None, since=None, hostname=None):
        self.pid = pid
        self.fname = fname
        self.since = since
        self.hostname = hostname
        super().__init__(msg)
def file_locked(lock_dir, file=None, fmt=None, ignore_invalid=False, ignore_races=True):
    def decorator(fn):
        if lock_dir is None:
            return fn
        from functools import wraps
        import socket
        hostname = socket.gethostname()
        pid = os.getpid()
        def lock(fname):
            found = False
            try:
                with open(fname, 'r') as fid:
                    data = fid.read()
                found = True
            except OSError: pass
            cur_pid = cur_host = cur_date = None
            try:
                valid = False
                cur_pid, cur_host, cur_date = data.split('\n')
                cur_pid = int(cur_pid)
                cur_date = datetime.datetime.fromisoformat(cur_date)
                valid = True
            except Exception: pass
            if found:
                if valid or not ignore_invalid:
                    if valid:
                        td = datetime.datetime.now() - cur_date
                        smsg = f'Lock found by pid {cur_pid} at host {cur_host} since {td}'
                    else:
                        smsg = 'Invalid lock'
                    raise LockedException(f'{smsg} at file {fname}.', fname=fname, pid=cur_pid, since=cur_date, hostname=cur_host)
            with open(fname, 'w+') as fid:
                fid.write(f'{pid}\n{hostname}\n{datetime.datetime.now()}')
        def unlock(fname):
            try:
                os.unlink(fname)
            except FileNotFoundError:
                if not ignore_races: raise
        get_file = _make_file_getter(fn, file, fmt)
        @wraps(fn)
        def wrapper(*args, **kwargs):
            fname = get_file(*args, **kwargs)
            path = os.path.join(lock_dir, fname)
            try:
                must_unlock = False
                lock(path)
                must_unlock = True
                data = fn(*args, **kwargs)
            except LockedException:
                must_unlock = False
                raise
            finally:
                if must_unlock:
                    unlock(path)
            return data
        return wrapper
    return decorator





class _FileCacheDeprecated:
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
        from pickle import dump
        with self.open(tag, 'wb') as fid:
            dump(data, file=fid)
        return self

    def load(self, tag: str, loader: Callable[[], Any],
             store: Optional[bool]=None, reload: Optional[bool]=False) -> Any:
        '''Load a previously saved data from the file designated by the given tag'''
        from pickle import UnpicklingError, load
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


