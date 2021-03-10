'''
Created on Mar 11, 2021

@author: janis
'''


def with_field_printers(fields, kv_sep='=', pair_sep=', ', fmt_field=str, fmt_repr=repr, fmt_str=str):
    """Add to a class printers that output the values of selected fields.
    
    >>> class H:
    ...    def __repr__(self): return 'repr'
    ...    def __str__(self): return 'str'
    >>> @with_field_printers('ab')
    ... class Test:
    ...    def __init__(self):
    ...        self.a = 5
    ...        self.b = H()
    >>> Test()
    Test(a=5, b=repr)
    >>> str(Test())   
    'Test(a=5, b=str)'
    """
    def decorator(cls):
        def __repr__(self):
            pairs = ((field,getattr(self, field)) for field in fields)
            spairs = pair_sep.join(f'{fmt_field(field)}{kv_sep}{fmt_repr(value)}'
                               for field,value in pairs)
            return f'{self.__class__.__name__}({spairs})'
        def __str__(self):
            pairs = ((field,getattr(self, field)) for field in fields)
            spairs = pair_sep.join(f'{fmt_field(field)}{kv_sep}{fmt_str(value)}'
                               for field,value in pairs)
            return f'{self.__class__.__name__}({spairs})'
        cls.__repr__ = __repr__
        cls.__str__ = __str__
        return cls
    return decorator


if __name__ == '__main__':
    import doctest
    doctest.testmod()