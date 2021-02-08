'''
Created on Feb 5, 2021

@author: janis
'''

import numpy as np
import re
from types import SimpleNamespace
from typing import NamedTuple, Any
import pandas

class Properties(SimpleNamespace):
    def __contains__(self, attr): return hasattr(self, attr)
    def __getitem__(self, what): return self.__dict__[what]
    def __iter__(self): return iter(self.__dict__.items())
    def __str__(self):
        return ','.join(f'{k}={v!r}' for k,v in self.__dict__.items())
    def __repr__(self):
        return f'{type(self).__name__}({self!s})'
    def _get(self, what, default=None): return self.__dict__.get(what, default)

class FormattingError(Exception): pass
class MissingData(FormattingError): pass

PATTERN_UQUOT = r'''(?P<UNQUOTED>[a-zA-Z_][^,\n]*)'''
rex_uquot = re.compile(PATTERN_UQUOT)


_REX_CLASS_NAME = re.compile(r'(?<!^)(?=[A-Z])')
def _get_default_class_name(cls):
    return _REX_CLASS_NAME.sub('_', cls.__name__).lower()
