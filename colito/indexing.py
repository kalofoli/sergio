'''
Created on Feb 2, 2021

@author: janis
'''

from numpy import arange
from pandas import Series


class Indexer:
    '''Provides fast indexing to and from a sequence of elements and its indices.
    
    Provides the indexable objects a2b and b2a, where a and b are the names of the provided sequences.
    >>> i = Indexer(letter=["one","two","five"],value=[1,2,5])
    >>> i.letter2value[["one","five"]]
    [1, 5]
    >>> i.value2letter[[2,5]]
    ["two", "five"] 
    
        The indexing works using two pandas Series objects.
    ''' 
    def __init__(self,*,with_original=False,**kwargs):
        '''Gets one or two keyword arguments and uses their names as property names.
        
        If only one key-value pair is given, the second property is "index".
        If a keyword is given with a value of None, then the value is a 0-based range
        equal to the size of the other property.
        >>> i = Indexer(letters='abcdefghijklmnopqrstuvwxyz')
        >>> i.letter2index['d']
        4
        >>> i = Indexer(letters='abcdefghijklmnopqrstuvwxyz', values=None)
        >>> i.letter2values['d']
        4
        @param with_original If True it will keep and use a reference to the original data.
        >>> l = [1,5,7]
        >>> i = Indexer(numbers=l, with_original=True)
        >>> i.numbers is l
        True
        >>> i = Indexer(numbers=l, with_original=False)
        >>> i.number is l
        False
        ''' 
        if len(kwargs) == 2:
            (name_a,vals_a),(name_b,vals_b) = kwargs.items()
        elif len(kwargs) == 1:
            (name_a,vals_a) = next(kwargs.items())
            name_b,vals_b = 'indices',None
        else:
            raise ValueError(f'Must specify one or two key-value pairs.')
        if vals_a is None:
            vals_a = arange(len(vals_b))
        if vals_b is None:
            vals_b = arange(len(vals_a))
        map_a2b = Series(vals_b, index=vals_a)
        map_b2a = Series(vals_a, index=vals_b)
        a2b = f'{name_a}2{name_b}'
        b2a = f'{name_b}2{name_a}'
        dct = {a2b:map_a2b.loc, b2a:map_b2a.loc,'__len__':len(map_a2b),'_names':(name_a,name_b)}
        if with_original:
            dct.update({name_a:vals_a,name_b:vals_b})
        else:
            dct.update({name_b:map_a2b,name_a:map_b2a})
        dct.update({'_'+a2b:map_a2b,'_'+b2a:map_b2a})
        self.__dict__ = dct
        self._with_original = with_original
    def __len__(self): return self.__dict__['__len__']
    def __str__(self):
        sori = "[ORIG] " if self._with_original else ""
        return f'({len(self)} pairs) {sori}{self._names[0]}-{self._names[1]}'
    def __repr__(self): return f'<{type(self).__name__}{self}>'
    