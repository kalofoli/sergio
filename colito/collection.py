'''
Created on Mar 10, 2021

@author: janis
'''


import typing
EntryType = typing.TypeVar('EntryType')
class NamedDictView(typing.Generic[EntryType]):
    def __init__(self, name, dct, fmt_key=repr, fmt_val=repr):
        self._dict = dct
        self._name = name
        self._fmts = fmt_key, fmt_val
    name = property(lambda self: self._name, None, 'Name of the dict view.')
    def __contains__(self, what) -> bool: return self._dict.__contains__(what)
    def __getitem__(self, what) -> EntryType:
        try:
            return self._dict.__getitem__(what)
        except KeyError:
            raise KeyError(f'Dictionary contains no {self.name} named {self._fmts[0](what)}. Available {self.name} are: {self.describe_keys()}.')
    def __len__(self): return len(self._dict)
    def get(self, what, default=None) -> EntryType: return self._dict.get(what, default)
    def keys(self): return self._dict.keys()
    def values(self): return self._dict.values()
    def items(self): return self._dict.items()
    def describe_keys(self): return ', '.join(map(self._fmts[0], self.keys()))
    def describe_items(self): return ', '.join([f'{self._fmts[0](key)}: {self._fmts[1](value)}'
                                                for key, value in self.items()])
    def __str__(self):
        return f'{self.name}({self.describe_keys()})'
    def __repr__(self):
        return f'<{type(self).__name__}[{self.name}] {self.describe_items()}>'

m = {'a':5}
str(NamedDictView('Measures',m))


import re
class ClassCollection:

    def __init__(self, name):
        self._name: str = str(name)
        self._tags: typing.Dict[str, typing.Type] = {}
        self._titles: typing.Dict[str, typing.Type] = {}
        self._classes = self._tags.keys()
        self._base = None
    
    @property
    def base(self): return self._base
    @base.setter
    def base(self, what):
        if self._base is not None:
            raise ValueError(f'While trying to overwrite base {self._base.__name__} with {what} of type {type(what).__name__}: the base class of {self.name} can only be set once.')
        self._base = what
    @property
    def name(self) -> str: return self._name
    @property
    def tags(self) -> NamedDictView[typing.Type]:
        '''Access available tags'''
        return NamedDictView('tags', self._tags, fmt_val='{0.__name__}'.format)
    
    @property
    def titles(self) -> NamedDictView[typing.Tuple]:
        '''Access available titles'''
        return NamedDictView('titles', self._titles, fmt_val='{0.__name__}'.format)
    
    @property
    def classes(self) -> typing.Tuple[typing.Type, ...]:
        return tuple(self._tags)
    
    def __getitem__(self, what):
        if isinstance(what, type):
            if issubclass(what, self.base):
                return what
            else:
                raise ValueError(f'Cannot resolve class from collection {self.name} from type {what.__name__} which is not a subclass of {self.base.__name__}.')
        elif what in self._tags:
            return self._tags[what]
        elif what in self._titles:
            return self._titles[what]
        else:
            raise TypeError(f'Could not resolve a valid member of collection {self.name} from value {what!r} of type {type(what).__name__}. Available tags: {self.tags.describe_items()}. Available titles: {self.titles.describe_keys()}.')            
    rex_title = re.compile(r'(?<!^)(?=[A-Z])')
    @classmethod
    def _get_title(cls, c):
        title = c.__dict__.get('__collection_title__')
        if title is None:
            title = cls.rex_title.sub(' ', c.__name__)
        return title
    def _register_subclass(self, cls):
        if cls in self._classes:
            raise KeyError(f'Attempting to register already existing class {cls}.')
        tag = cls.__collection_tag__
        if tag in self._tags:
            raise KeyError(f'Attempting to register {cls} which has an already existing tag {tag}->{self.tags[tag]}.')
        title = self._get_title(cls)
        if title in self._titles:
            raise KeyError(f'Attempting to register {cls} which has an already existing title {title}->{self.titles[title]}.')
        self._tags[tag] = cls
        self._titles[title] = cls
    def _register_base(self, cls):
        self.base = cls
    
    def __repr__(self):
        return f'<{self.__class__.__name__}[{self._name}] with {len(self.tags)} tags>'

class ClassCollectionRegistrarMeta(type):
    
    def __new__(cls, cls_name, bases, dct):
        cls_new = super(ClassCollectionRegistrarMeta, cls).__new__(cls, cls_name, bases, dct)
        if not '__collection_tag__' in cls_new.__dict__:
            raise ValueError(f'Class {cls_name} does not have an own member __collection_tag__.')
        tag = cls_new.__dict__['__collection_tag__']
        factory = cls_new.__dict__.get('__collection_factory__')
        if factory is not None: # own member: base class
            factory._register_base(cls_new)
        else:
            try:
                factory = cls_new.__collection_factory__
            except KeyError:
                raise KeyError(f'Class {cls_name} has no member __collection_factory__.')
        if tag is not None:
            factory._register_subclass(cls_new)
        return cls_new

class ClassCollectionRegistrar(metaclass=ClassCollectionRegistrarMeta):
    ''' Inherited from the base class of a collection
    
    The sub-classes of this collection are registered and indexable based on their __collection_tag__ string.
    The class collection class is read from the __collection_factory__ attribute of the collection base class.
    @note: Only the base class must have a __collection_factory__ member   
    
    Example:
    >>> ANIMALS = ClassCollection('Animals')
    >>> class Animal(ClassCollectionRegistrar):
    ...     __collection_tag__ = None
    ...     __collection_factory__ = ANIMALS
    >>> class Pig(Animal):
    ...     __collection_tag__ = 'swine'
    >>> class RacingHorse(Animal):
    ...     __collection_tag__ = 'racing-horse'
    ...     __collection_title__ = 'horse used for racing'
    >>> ANIMALS.tags
    <NamedDictView[tags] 'swine': Pig, 'racing-horse': RacingHorse>
    >>> ANIMALS.titles
    <NamedDictView[titles] 'Pig': Pig, 'horse used for racing': RacingHorse>
    >>> ANIMALS.tags['racing-horse'] is RacingHorse
    True
    >>> ANIMALS.name
    'Animals'
    >>> ANIMALS.tags['pig']
    Traceback (most recent call last):
    ...
    KeyError: "Dictionary contains no tags named 'pig'. Available tags are: 'swine', 'racing-horse'."
    >>> ANIMALS['Pig'] is Pig
    True
    >>> ANIMALS[Pig] is Pig
    True
    '''
    __collection_tag__ = None
    __collection_factory__ = None

    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    