'''
Created on May 14, 2021

@author: janis
'''
from collections import namedtuple
import typing


class CloneFields:
    __slots__ = ('_clone', '_getters')
    def __init__(self, clone=None, getters=None):
        '''
        :param getters: Either None, or a dict whose each key is a field name
        and the corresponding value a callable with a single argument: the instance of which
        field the value to get. 
        '''
        self._clone = set(clone) if clone is not None else None
        self._getters = {} if getters is None else {**getters} 
    
    Field = namedtuple('Field', ('name','value','must_clone'))
    def __call__(self, inst):
        '''Must return a list of elements of type CloneFields.Field'''
        raise NotImplementedError('Override')
    def _must_clone(self, name, value):
        if self._clone is None:
            return isinstance(value, Cloneable)
        else:
            return name in self._clone
    def _get_value(self, inst, name):
        getter = self._getters.get(name, None)
        if getter is None:
            value = getattr(inst, name)
        else:
            value = getter(inst)
        return value
    
    
class CloneFieldsFromFunction(CloneFields):
    __slots__ = ('_fn',)
    def __init__(self, fn = '__init__', clone=None, getters=None):
        super().__init__(clone=clone, getters=getters)
        self._fn = fn
    def __call__(self, inst):
        if isinstance(self._fn, str):
            fn = getattr(inst, self._fn)
        elif isinstance(self._fn, typing.Callable):
            fn = self._fn
        else:
            raise TypeError(f'Cannot get callable from {self._fn} of type {type(self._fn).__name__}.')
        
        import inspect
        sig = inspect.signature(fn)
        fields = []
        for name,par in sig.parameters.items():
            if par.kind in (inspect._VAR_KEYWORD, inspect._VAR_POSITIONAL):
                continue
            try:
                value = self._get_value(inst=inst, name=name)
            except AttributeError:
                if par.default is not inspect.Parameter.empty:
                    continue
                else:
                    raise AttributeError((f'During cloning of {inst!r} of type {type(inst).__class__}:',
                                          f' missing required attribute "{name}".'))
            must_clone = self._must_clone(name=name, value=value)
            fields.append(self.Field(name=name, value=value, must_clone=must_clone))
        return fields
        


DEFAULT_CLONE_FIELDS = CloneFieldsFromFunction()
class Cloneable:
    __slots__ = ()
    __clone_fields__ = DEFAULT_CLONE_FIELDS
    def __clone__(self):
        
        field_spec = getattr(self, '__clone_fields__')
        fields = field_spec(self)
        kwargs = {}
        for field in fields:
            value = field.value.__clone__() if field.must_clone else field.value
            kwargs[field.name] = value
        return self.__class__(**kwargs)
    
    
if __name__ == '__main__':
    
    class B(Cloneable):
        def __init__(self, val):
            self._val = val
        @property
        def val(self): return self._val
    
    class Test(Cloneable):
        '''
        >>> c = Test('a',B('b'),'c')
        >>> c2 = c.__clone__()
        >>> c is c2
        False
        >>> c.b is c2.b
        False
        >>> c.a is c2.a
        True
        '''
        def __init__(self, a, b, c=None):
            self._a = a
            self._b = b
            self._c = c
        @property
        def a(self): return self._a
        @property
        def b(self): return self._b
        
    
    
    
    
    
    import sys
    del sys.path[0] # remove logging circular dependency
    import doctest
    doctest.testmod()