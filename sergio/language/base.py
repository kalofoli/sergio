'''
Created on May 3, 2021

@author: janis
'''

import re
from typing import Sequence, Optional, Tuple, List, Mapping, Union, Iterator, Dict, Any, \
    Set, Iterable, Callable, cast, TypeVar, Generic, NamedTuple
from itertools import chain

import numpy as np

from colito.cache import ValueCache as Cache
from colito.collection import ClassCollection
from colito.indexing import SimpleIndexer
from colito.summaries import SummaryOptions, SummarisableList,\
    SummarisableAsDict, Summarisable
from colito.logging import getModuleLogger
from ..predicates import Predicate
from colito.collection import ClassCollectionFactoryRegistrar
from sergio.language.utils import indices_remove

PredicateCollectionType = Iterable[Predicate]  # pylint: disable=invalid-name
PredicateOrIndexType = Union[Predicate, int]  # pylint: disable=invalid-name
PredicateOrIndexCollectionType = Union[PredicateCollectionType, Iterable[int]]  # pylint: disable=invalid-name
CacheSpecType = Union[bool, Cache, Dict]

SelectorType = TypeVar('SelectorType')
LanguageType = TypeVar('LanguageType')

log = getModuleLogger(__name__)


class Selector():
    '''Selector for a set of entities'''
    def __init__(self, cache: CacheSpecType=True):
        self._cache: Cache = Cache.from_spec(cache)
        
    @property
    def validity(self) -> np.ndarray:
        '''A logical of those entities that validate the selector'''
        raise NotImplementedError()

class StaticSelector(Selector):

    def __init__(self, validity:np.ndarray, cache: CacheSpecType=True) -> None:
        super().__init__(cache=cache)
        self._validity:np.ndarray = validity
        
    @property
    def validity(self) -> np.ndarray:
        return self._validity


def property_predicate_objects(index_producer: Callable[..., Tuple[int, ...]]):
    predicate_producer = as_predicate_objects(index_producer)
    return property(predicate_producer)

        
def as_predicate_objects(index_producer: Callable[..., Tuple[int, ...]]) -> Callable[..., Tuple['Predicate', ...]]:

    def predicate_producer(self, *args, **kwargs) -> Tuple['Predicate', ...]:
        predicate_indices = index_producer(self, *args, **kwargs)
        if predicate_indices is None:
            predicate_objects = None
        else:
            predicate_objects = tuple(self.language.predicate_objects(predicate_indices))
        return predicate_objects

    return predicate_producer



class SelectorParserBase(Generic[SelectorType, LanguageType]):
    '''A parser of subgroups of a given language'''

    class ParseError(RuntimeError): pass
    
    def __init__(self, language: LanguageType) -> None:
        self._language: LanguageType = language

    @property
    def language(self) -> LanguageType:
        return self._language
        
    def parse(self, string:str) -> SelectorType:
        raise NotImplementedError()


class LanguageSelector(Generic[LanguageType], Selector):
    '''Language-aware selector for a set of entities'''

    def __init__(self, language: LanguageType, cache: CacheSpecType=True) -> None:
        super().__init__(cache=cache)
        self._language: LanguageType = language

    @property
    def refinements(self) -> Sequence['SelectorBase']:
        '''List all selectors that refine this one'''
        raise NotImplementedError()

    @property
    def language(self) -> LanguageType:
        '''Reference to the creating language'''
        return self._language

    @property
    def cache(self) -> Cache:
        '''Return the current cache object'''
        return self._cache


LANGUAGES = ClassCollection('Language')

class Language(Summarisable, ClassCollectionFactoryRegistrar):
    '''Language of selectors'''

    __collection_tag__ = None
    __collection_factory__ = LANGUAGES
    
    def __init__(self, data) -> None:
        self._data = data

    @property
    def root(self) -> SelectorType:
        '''The root selector'''
        raise NotImplementedError()

    @property
    def data(self):
        '''Data on which the language operates'''
        return self._data

    def make_parser(self) -> SelectorParserBase['LanguageBase', SelectorType]:
        '''Create a language parser object'''
        raise NotImplementedError()

ConjunctionLanguageType = TypeVar('ConjunctionLanguageType')
class ConjunctionSelectorBase(SummarisableAsDict, LanguageSelector[ConjunctionLanguageType]):
    '''Selector taking the conjunction of a set of predicates'''

    def __init__(self, language: 'ConjunctionLanguage', predicates: PredicateOrIndexCollectionType, cache: CacheSpecType=True, _indices_sorted=None) -> None:
        super().__init__(language=language, cache=cache)
        self._predicate_indices: Tuple[int, ...] = tuple(self.language.predicate_indices(predicates))
        self._indices_sorted = _indices_sorted

    @Cache.cached_property
    def indices_sorted(self):
        '''Return the indices in a sorted order, according the language ordering'''
        if self._indices_sorted is not None:
            return self._indices_sorted
        else:
            return np.sort(self.indices)

    @property_predicate_objects
    def predicates(self) -> Tuple[int, ...]:
        '''Predicates the current selector has'''
        return self._predicate_indices
    
    @property
    def indices(self) -> Tuple[int, ...]:
        '''Predicates the current selector has'''
        return self._predicate_indices

    indices_path = indices
    
    @property
    def language(self) -> ConjunctionLanguageType:
        return cast(ConjunctionLanguageType, self._language)
    
    @Cache.cached_property
    def validity(self) -> np.ndarray:
        validity = self.language.validate(self.predicates)
        return validity

    @property
    def pruning_indices(self):
        '''Return a list of indices that would not allow this selector to be created again'''
        return self._predicate_indices[-1:] 

    @property
    def refinements(self) -> Sequence['ConjunctionSelectorBase']:
        return list(self._language.refine(self))

    def extend(self, predicate: PredicateOrIndexType, **kwargs) -> 'ConjunctionSelector':
        '''Create a new selector with an extra predicate appended to the current one
        
        :param predicate: A single predicate with which to extend the current selector.
        :param kwargs: Keyword arguments to pass to the constructor. 
        '''
        raise NotImplementedError()
    
    def __contains__(self, predicate: PredicateOrIndexType) -> bool:
        if self._predicate_indices:
            index = self.language.predicate_index(predicate)
            indices_sorted = self.indices_sorted
            pos = np.searchsorted(indices_sorted, index)
            return index == indices_sorted[pos]
        else:
            return False

    def __repr__(self):
        predicates = ' AND '.join(map(repr, self.predicates))
        return '<{0.__class__.__name__}: {1}>'.format(self, predicates)

    def __str__(self):
        predicates = '^'.join(map(str, self.predicates))
        return f'{{{predicates}}}'

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.indices))
    
    def __eq__(self, other) -> bool:
        return (self.__class__.__name__, self.indices) == (other.__class__.__name__, other.indices)

    def __summary_dict__(self, options:SummaryOptions):
        dct = {'description':str(self)}
        #if options.parts & SummaryParts.PREDICATE_INDICES:
        #    dct['predicate_indices'] = self.indices
        return dct
        

class ConjunctionSelector(ConjunctionSelectorBase):
    '''Selector taking the conjunction of a set of predicates'''

    def extend(self, predicate: PredicateOrIndexType, **kwargs) -> 'ConjunctionSelector':
        extension_index = self.language.predicate_index(predicate)
        predicate_indices = chain(self.indices, [extension_index])
        return self.__class__(self.language, predicate_indices, **kwargs)
    
    
    class Digest(NamedTuple):  # pylint: disable = too-few-public-methods
        indices: List[int]
    
    def serialise(self, json=False):
        dig = ConjunctionSelector.Digest(indices=self.indices)  # pylint: disable = no-value-for-parameter
        if json:
            dig = dig._asdict()  # pylint: disable=no-member
        return dig
    
    @classmethod
    def deserialise(cls, language, digest):
        if isinstance(digest, dict):
            digest = ConjunctionSelector.Digest(**digest)
        return cls(language, predicates=digest.indices)


class ConjunctionLanguage(SummarisableAsDict, Language):
    '''Language of predicate conjunctions'''

    __collection_tag__ = 'conjunctions'
    __selector_class__ = ConjunctionSelector
    
    def __init__(self, data, predicates) -> None:
        super(ConjunctionLanguage, self).__init__(data=data)
        self._data = data
        predicates_srt = tuple(p.copy(index=index) for index,p in enumerate(sorted(predicates)))
        self._predicate_indexer = SimpleIndexer(predicates_srt)
        self._predicate_validities: np.ndarray = np.array(np.stack([predicate.validate() for predicate in self.predicates], axis=1), order='F')
        self._root: Selector = self.__selector_class__(self, [])
    
    def validate(self, predicates: PredicateOrIndexCollectionType, out: np.ndarray=None):
        '''Compute the validity of a set of predicates'''
        predicate_objects = self.predicate_objects(predicates)
        nrows = self.data.num_entities
        if out is not None:
            assert len(out) == nrows, f'Output vector size mismatch (size was {out.__len__()} instead of {nrows}).'
            validity = out
        else:
            validity = np.ones(nrows, dtype=bool)
        for predicate in predicate_objects:
            validity = validity & predicate.validate()
        return validity

    @property
    def root(self):
        '''The root selector of the language'''
        return self._root

    @property
    def predicates(self) -> List[Predicate]:
        '''Tuple of predicates this language uses'''
        return self._predicate_indexer.items

    @predicates.setter
    def predicates(self, predicates: PredicateOrIndexCollectionType) -> None:
        predicate_objects = sorted(self.predicate_objects(predicates))
        self._predicate_indexer.clear().update_iterable(predicate_objects)

    def __repr__(self):
        return '<{0.__class__.__name__}: of {1} predicates>'.format(self, len(self.predicates))

    def refine(self, selector: ConjunctionSelectorBase, blacklist: Optional[PredicateOrIndexCollectionType]=None) -> Iterator[ConjunctionSelectorBase]:
        '''Return a refinement of the specified selector'''
        num_predicates = len(self.predicates)
        index_end = np.max(selector.indices, initial=-1)+1
        extension_indices = np.arange(index_end, num_predicates)
        if blacklist is not None:
            blacklist = tuple(self.predicate_indices(blacklist))
            extension_indices = indices_remove(extension_indices, blacklist, ignore_missing=True)
        return map(selector.extend, extension_indices)

    def predicate_index(self, predicate: PredicateOrIndexType) -> int:
        '''Convert a predicate (or index) to the respective index'''
        index:int
        if isinstance(predicate, Predicate):
            index = self._predicate_indexer.get_index(predicate)
        else:
            try:
                index = int(predicate)
            except:
                raise ValueError(f'Cannot convert {predicate} to predicate index')
        return index

    def predicate_object(self, predicate: PredicateOrIndexType) -> Predicate:
        '''Convert a predicate (or index) to the respective predicate'''
        obj: Predicate
        if isinstance(predicate, (int,np.int,np.int64)):
            obj = self._predicate_indexer.get_object(predicate)
        elif isinstance(predicate, Predicate):
            obj = predicate
        else:
            raise ValueError(f'Cannot convert {predicate} to predicate object. Must be int or Predicate')
        return obj

    def predicate_indices(self, predicates: PredicateOrIndexCollectionType) -> Iterator[int]:
        '''Convert a sequence of predicates (or indices) to one of predicate indices'''
        return map(self.predicate_index, iter(predicates))
    
    def predicate_objects(self, predicates: PredicateOrIndexCollectionType) -> Iterator[Predicate]:
        '''Convert a sequence of predicates (or indices) to one of predicate objects'''
        return map(self.predicate_object, iter(predicates))
    
    class SelectorParser(SelectorParserBase):

        REGEX = {'body':re.compile(r'^{((?:[^^]+\^)*(?:[^^]+)?)}$'), 'subgroup':re.compile(r'{((?:[^^]+\^)*(?:[^^]+)?)}')}

        def __init__(self, language, selector_ctor):
            super().__init__(language=language)
            self._predicate_dict: Dict[str, Predicate] = dict((str(p), p) for p in language.predicates)
            self._selector_ctor = selector_ctor
            
        def parse(self, string):
            REGEX = ConjunctionLanguage.SelectorParser.REGEX
            ParseError = ConjunctionLanguage.SelectorParser.ParseError
            try:
                match = REGEX['body'].match(string)
                if match is None:
                    raise ParseError('String does not match: {body} format.')
                body = match.groups()[0]
                parts = body.split('^')
                if parts and not parts[0]:
                    parts = []
                try:
                    predicates = tuple(map(self._predicate_dict.__getitem__, parts))
                except KeyError as err:
                    key = err.args[0]
                    raise ParseError(f'Could not find a predicate with name {key}.')
                selector = self._selector_ctor(predicates)
            except ParseError as err:
                msg = err.args[0]
                raise ParseError(f'While parsing string {string}: {msg}.')
            return selector
        
        @classmethod
        def get_subgroups(cls, string):
            matches = cls.REGEX['subgroup'].findall(string)
            return matches
            
    def make_parser(self) -> 'ConjunctionLanguage.SelectorParser':

        def make_selector(predicates):
            return self.__selector_class__(self, predicates)

        return self.SelectorParser(self, make_selector)
    
    def selector(self, predicates:PredicateOrIndexCollectionType, **kwargs):
        '''Create a selector from given predicates'''
        return self.__selector_class__(self, predicates, **kwargs)
    
    def deserialise_selector(self, digest):
        return ConjunctionSelector.deserialise(self, digest)

    def __summary_dict__(self, options:SummaryOptions):
        dct = {'data-name': self.data.name, 'num-predicates':len(self.predicates)}
        #if options.parts & SummaryParts.LANGUAGE_PREDICATES:
        #    dct['predicates'] = SummarisableList(self.predicates)
        return dct
    
