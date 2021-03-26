import functools



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


class NamedUniqueConstant:
    '''A named object to use as a unique constant'''
    __slots__ = ('_name',)
    def __init__(self, name):
        self._name = name
    def __repr__(self): return self._name
    def __eq__(self, other): return self is other
    def __hash__(self): return id(self)