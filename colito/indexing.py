'''
Created on Feb 2, 2021

@author: janis
'''

from numpy import arange
from pandas import Series
from typing import Dict, Tuple


def to_dict_values(values, key_fn=str, sort=False, ordered=True):
    '''Make a dict from a (computed key) to the provided values
    
    :param values: An iterable of elements to use as dict values.
    :param key_fn: function to create the keys of the dict based on each provided element
    :param sort bool: Whether the keys should be sorted. Implies ordered.
    :param ordered bool: If true, an OrderedDict is created.
    :return: a dict of keys computed by key_fn over each element. If sort is true, the dict is ordered.
    '''
    pairs = list((key_fn(value), value) for value in values)
    if sort:
        pairs = pairs.sort(key=lambda x:x[0])
        if ordered is None:
            ordered = True
    return dict(pairs)


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
    


class ClassCollection:

    def __init__(self, name, classes):
        self._classes = tuple(classes)
        self._name: str = str(name)
        self._tags: Dict[str, type] = to_dict_values(classes, key_fn=self._get_tag, ordered=True)
        self._class_names: Dict[str, type] = to_dict_values(classes, key_fn=lambda cls:cls.__name__, ordered=True)
    
    @classmethod
    def _get_tag(self, cls):
        if hasattr(cls, 'tag'):
            return cls.tag
        else:
            return cls.__name__
        
    @property
    def tags(self) -> Tuple[str, ...]:
        '''List available tags'''
        return tuple(self._tags.keys())
    
    @property
    def class_names(self) -> Tuple[str, ...]:
        '''List available class names'''
        return tuple(self._class_names.keys())
    
    @property
    def classes(self) -> Tuple[type, ...]:
        return self._classes
    
    def has_tag(self, tag) -> bool:
        return tag in self._tags
    
    def get_class_from_tag(self, tag) -> type:
        if not self.has_tag(tag):
            raise KeyError(f'Collection {self._name} has no tag {tag}. Try one of: {",".join(self.tags)}.')
        return self._tags[tag]
    
    def has_class_name(self, tag) -> bool:
        return tag in self._tags

    def get_class_from_name(self, class_name):
        if not self.has_class_name(class_name):
            raise KeyError(f'Collection {self._name} has no class_name {class_name}. Try one of: {",".join(self.class_names)}.')
        return self._class_names[class_name]
    
    def get_class_title(self, cls) -> str:
        '''Get a friendly name from a class'''
        # this could also have been a classmethod/staticmethod, but kept as a member in case overrides or state is added later. (E.g.: headers, camelisation, etc)
        if hasattr(cls, 'name'):
            return cls.name
        else:
            return cls.__name__

    def get_class_tag(self, cls) -> str:
        '''Get a tag name from a class'''
        # this could also have been a classmethod/staticmethod, but kept as a member in case overrides or state is added later. (E.g.: headers, camelisation, etc)
        if hasattr(cls, 'tag'):
            return cls.tag
        else:
            return cls.__name__
    
    def __repr__(self):
        return f'<{self.__class__.__name__}[{self._name}] with {len(self._tags)} tags>'
