'''
Created on Feb 1, 2021

@author: janis
'''


import os
import enum
from typing import Dict
import datetime

class FileManager:
    class Kind(enum.Enum):
        LOG = enum.auto()
        WORK = enum.auto()
        CACHE = enum.auto()
        DATA = enum.auto()
        RESULT = enum.auto()
        DEFAULT = enum.auto()

    FILE_KINDS_TEXT = ', '.join(Kind.__members__.keys())
        
    def __init__(self, paths: Dict[str,str]={}, default_path=None, bare_only:bool=False) -> None:
        self._paths:Dict[str,str] = {}
        for kind,path in paths.items():
            self.set_kind_path(kind, path)
        if default_path is None:
            default_path = os.path.curdir
        self.set_kind_path(None, default_path)
        self._bare_only = bare_only

    def _kind_enum(self, kind):
        if isinstance(kind, str):
            kind_enum = FileManager.Kind[kind]
        elif isinstance(kind, FileManager.Kind):
            kind_enum = kind
        elif kind is None:
            kind_enum =FileManager.Kind.DEFAULT
        else:
            raise TypeError(f'Only kinds {self.FILE_KINDS_TEXT} are allowed.')
        return kind_enum
            
    def set_kind_path(self, kind, path):
        kind_enum = self._kind_enum(kind)
        self._paths[kind_enum] = path
        
    def get_kind_path(self, kind):
        kind_enum = self._kind_enum(kind)
        return self._paths.get(kind_enum, self._paths[FileManager.Kind.DEFAULT])
    
    def get(self, file, kind=None, bare_only=None):
        if bare_only is None:
            bare_only = self._bare_only
        if bare_only and os.path.dirname(file):
            return file
        else:
            base = self.get_kind_path(kind)
            return os.path.join(base, file)
    
    def __repr__(self):
        txt = ",".join(f'{k.name}:"{v}"' for k,v in self._paths.items())
        return f'<{self.__class__.__name__}:{txt}>'

DefaultFileManager = FileManager(default_path='data')

