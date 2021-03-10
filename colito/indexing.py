'''
Created on Feb 2, 2021

@author: janis
'''

import numpy as np
from pandas import Series
from typing import Dict, Tuple, Iterable, Generic, TypeVar, List, Any, cast,\
    Sequence, NamedTuple
import enum


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
    >>> list(i.letter2value[["one","five"]])
    [1, 5]
    >>> list(i.value2letter[[2,5]])
    ['two', 'five']
    
        The indexing works using two pandas Series objects.
    ''' 
    def __init__(self,*,with_original=False,**kwargs):
        '''Gets one or two keyword arguments and uses their names as property names.
        
        If only one key-value pair is given, the second property is "index".
        If a keyword is given with a value of None, then the value is a 0-based range
        equal to the size of the other property.
        >>> i = Indexer(letters=tuple('abcdefghijklmnopqrstuvwxyz'))
        >>> i.letters2indices['d']
        3
        >>> i = Indexer(letters=tuple('abcdefghijklmnopqrstuvwxyz'), values=None)
        >>> i.letters2values['d']
        3
        >>> i.values2letters[4]
        'e'
        >>> ''.join(i.values2letters[[4,2]])
        'ec'
        
        @param with_original If True it will keep and use a reference to the original data.
        >>> l = [1,5,7]
        >>> i = Indexer(numbers=l, with_original=True)
        >>> i.numbers is l
        True
        >>> i = Indexer(numbers=l, with_original=False)
        >>> i.numbers is l
        False
        ''' 
        if len(kwargs) == 2:
            (name_a,vals_a),(name_b,vals_b) = kwargs.items()
        elif len(kwargs) == 1:
            (name_a,vals_a) = next(iter(kwargs.items()))
            name_b,vals_b = 'indices',None
        else:
            raise ValueError(f'Must specify one or two key-value pairs.')
        if vals_a is None:
            vals_a = np.arange(len(vals_b))
        if vals_b is None:
            vals_b =np.arange(len(vals_a))
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
    

ET = TypeVar('ET')
class SimpleIndexer(Generic[ET]):
    
    class MissingKeyResolution(enum.Enum):
        COMPLAIN = enum.auto()
        IGNORE = enum.auto()
        DEFAULT = enum.auto()
    
    
    def __init__(self, index=None, missing=MissingKeyResolution.COMPLAIN):
        obj2id: Dict[ET, int] 
        id2obj: List[ET]
        if index is None:
            obj2id = {}
            id2obj = []
        else:
            if isinstance(index, Indexer):
                indexer = cast(Indexer, index)
                obj2id = indexer._obj2id 
                id2obj = indexer._id2obj
            elif isinstance(index, dict):
                obj2id = cast(Dict[Any, int], index)
                id2obj = Indexer.dict2index(obj2id)
            elif isinstance(index, (Sequence, Iterable)):
                id2obj = list(index)
                obj2id = dict(zip(id2obj, range(len(id2obj))))
            else:
                raise RuntimeError('Cannot parse index argument')
        self._obj2id: Dict[Any, int] = obj2id
        self._id2obj: List[Any] = id2obj
        self._missing = missing
        
    @property
    def items(self) -> List[ET]:
        return self._id2obj
    
    def clear(self) -> 'Indexer[ET]':
        self._obj2id.clear()
        self._id2obj.clear()
        return self
   
    def update(self, object_: ET) -> 'Indexer[ET]':
        return self.update_iterable((object_,))
    
    def update_iterable(self, objects:Iterable[ET]) -> 'Indexer[ET]':
        new = list(filter(lambda obj: obj not in self._obj2id, objects))
        new_pairs = zip(new, range(len(self), len(self) + len(new)))
        self._id2obj += new
        self._obj2id.update(dict(new_pairs))
        return self

    def __len__(self) -> int:
        return len(self._id2obj)

    def get_index(self, what, missing=MissingKeyResolution.DEFAULT) -> int:
        '''Get the index of an object or fail if the key does not exist.
        
        If a default value is specified, return this instead of an error in case of a missing key.'''
        missing_tag = object()
        res = self._obj2id.get(what, missing_tag)
        if res is missing_tag:
            MKR = Indexer.MissingKeyResolution
            if missing is MKR.DEFAULT:
                missing = self._missing
            if res is MKR.COMPLAIN:
                raise KeyError(f'Could not find key {what}.')
            else:
                res = missing
        return res
    
    def get_indices(self, what) -> Iterable[int]:
        return map(self._obj2id.__getitem__, what)

    def get_index_array(self, what) -> np.ndarray:
        return np.fromiter(self.get_indices(what), int)
    
    def get_object(self, what) -> ET:
        return self._id2obj[what]

    def get_objects(self, what) -> Iterable[ET]:
        return map(self._id2obj.__getitem__, what)

    def asdict(self) -> Dict[ET, int]:
        return self._obj2id.copy()
        
    __call__ = get_index
    __getitem__ = get_object
    
    def __hasitem__(self, what):
        return what in self._obj2id 
    
    @classmethod
    def dict2index(self, dct):
        index = np.fromiter(dct.values(), int)
        items_tmp = np.array(tuple(dct.keys()), object)
        items = np.zeros(len(index), object)
        items[index] = items_tmp
        return list(items)
    
    class IndexerMapping(NamedTuple):
        indexer:'Indexer'
        map_new2old:np.ndarray
        map_old2new:np.ndarray
        is_old_selected = property(lambda self:self.map_old2new != -1, None, 'A boolean array indicating if the corresponding old element is selected.')
        
    def select_indices(self, indices):
        index_local = np.arange(len(self))
        map_new2old = index_local[indices]
        obj = self.get_objects(map_new2old)
        indexer = Indexer(obj)
        map_old2new = np.empty(len(self), int)
        map_old2new[:] = -1
        map_old2new[map_new2old] = np.arange(len(map_new2old))
        
        return self.IndexerMapping(indexer=indexer, map_new2old=map_new2old, map_old2new=map_old2new)
    
    def select_objects(self, what):
        indices = np.fromiter(self.get_indices(what), int)
        return self.select_indices(indices)
    
    def __repr__ (self):
        return f'<{self.__class__.__name__} with {len(self)} objects>'

    def to_data_frame(self, column_index='index', column_items='item', object_index=True):
        '''Return a pandas DataFrame with the given index data'''
        import pandas
        if object_index:
            index = pandas.Series(self.items, name=column_index)
            df = pandas.DataFrame(np.arange(len(self)),index = index, columns=[column_items])
        else:
            index = pandas.RangeIndex(0, len(self), name=column_index)
            df = pandas.DataFrame(self.items,index = index, columns=[column_items])
        return df


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

if __name__ == '__main__':
    import sys
    del sys.path[0]
    import doctest
    doctest.testmod()
    