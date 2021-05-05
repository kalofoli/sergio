'''
Created on Feb 1, 2021

@author: janis
'''


import os
import enum
from typing import Dict



""" Manage file paths based on their kind.

This offers an abstraction to avoid hard-conding of file paths,
but instead allow for configurable folders depending on the kind of the requested file.
"""
class FileManager:
    class _Symbol:
        def __init__(self, name): self._name = name
        def __repr__(self): return f'[{self._name}]'
    DEFAULT = _Symbol('DEFAULT')
    NO_RESOLVE = _Symbol('NO_RESOLVE') # Do not resolve kind 
    
    def __init__(self, default_path = None, paths={}, allow_default=True) -> None:
        self._paths = dict()
        for kind,path in paths.items():
            self.set_kind_path(kind, path)
        if default_path is None:
            default_path = os.path.curdir
        self._default_path = default_path
        self._allow_default = allow_default

    def _resolve_kind(self, kind):
        raise NotImplementedError('Override')
    
    @property
    def default_path(self): return self._default_path
            
    def set_kind_path(self, kind, path):
        kind_res = self._resolve_kind(kind)
        self._paths[kind_res] = path
        
    def get_kind_path(self, kind, allow_default=None):
        kind_res = self._resolve_kind(kind)
        missing = object()
        path = self._paths.get(kind_res, missing)
        if path is self.DEFAULT:
            res = self.default_path
        elif path is missing:
            if allow_default is None:
                allow_default = self._allow_default
            if not allow_default:
                raise KeyError(f"No kind '{kind}'.")
            res = self.default_path
        else:
            res = path
        return res
    
    def __call__(self, file, kind=None):
        if kind is self.NO_RESOLVE:
            res = file
        else:
            base = self.get_kind_path(kind)
            res = os.path.join(base, file)
        return res
    
    def open(self, file, *args, kind=None, **kwargs):
        path = self(file, kind=kind)
        return open(path, *args, **kwargs)
    
    def __str__(self):
        txt = ",".join(f'{k}:"{v}"' for k,v in self._paths.items())
        return txt
    def __repr__(self):
        return f'<{self.__class__.__name__}:{self}>'


class EnumFileManager(FileManager):
    class Kinds(enum.Enum): pass
    __default_kind__ = None
    def __init__(self, default_path = None, paths={}, allow_default=True, ignore_case=True) -> None:
        """
        >>> import enum
        >>> class Kinds(enum.Enum):
        ...     DATA = enum.auto()
        ...     RESULT = enum.auto()
        ...     DEFAULT = enum.auto()
        >>> class Derived(EnumFileManager):
        ...     Kinds = Kinds
        ...     __default_kind__ = 'default'
        >>> fm = Derived()
        >>> fm.kind_description
        'DATA, RESULT, DEFAULT'
        >>> Derived(paths={'data':'./data'})
        <Derived:DATA:"./data", DEFAULT:"[DEFAULT]">
        >>> fm = Derived(paths={'DATA':'./data'}, default_path='/default/')
        >>> fm('here', kind=FileManager.NO_RESOLVE)
        'here'
        >>> fm('here', kind='data')
        './data/here'
        >>> fm('here', kind=Kinds.DATA)
        './data/here'
        >>> fm('here', kind=Kinds.DATA)
        './data/here'
        >>> fm('here')
        '/default/here'
        >>> fm.open('here','r')
        Traceback (most recent call last):
        ...
        FileNotFoundError: [Errno 2] No such file or directory: '/default/here'
        >>> str(fm)
        'DATA:"./data", DEFAULT:"[DEFAULT]"'
        """
        if ignore_case:
            self._kinds_locase = {key.lower():member for key,member in self.Kinds.__members__.items()}
            self._member_mapper = self._locase_getitem
        else:
            self._member_mapper = self.Kinds.__getitem__
        super().__init__(default_path=default_path, paths=paths, allow_default=allow_default)
        if self.default_kind is not None:
            self.set_kind_path(self.default_kind, self.DEFAULT)
        self._ignore_case = ignore_case
    def _locase_getitem(self, what):
        '''Helpr for case-insensitive lookup.
        
        Promoted to member to aid pickling.'''
        return self._kinds_locase[what.lower()]
    @property
    def default_kind(self): return self.__default_kind__
    @property
    def kind_description(self): return ', '.join(self.Kinds.__members__)
    def _resolve_kind(self, kind):
        if isinstance(kind, str):
            kind_enum = self._member_mapper(kind)
        elif isinstance(kind, self.Kinds):
            kind_enum = kind
        elif kind is None:
            kind_enum = FileManager.DEFAULT
        else:
            raise TypeError(f'Only kinds {self.kind_descrption} are allowed.')
        return kind_enum
            
    def __str__(self):
        return ", ".join(f'{k.name}:"{v}"' for k,v in self._paths.items())

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
