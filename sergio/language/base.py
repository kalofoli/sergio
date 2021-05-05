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

    def __init__(self, language: 'ConjunctionLanguage', predicates: PredicateOrIndexCollectionType, cache: CacheSpecType=True) -> None:
        super().__init__(language=language, cache=cache)
        predicate_objects = self.language.predicate_objects(predicates)
        self._predicate_indices: Tuple[int, ...] = tuple(language.predicate_indices(predicate_objects))

    @property_predicate_objects
    def predicates(self) -> Tuple[int, ...]:
        '''Predicates the current selector has'''
        return self._predicate_indices
    predicates_path = predicates

    @property
    def indices(self) -> Tuple[int, ...]:
        '''Predicates the current selector has'''
        return self._predicate_indices

    indices_path = indices
    @property
    def language(self) -> ConjunctionLanguageType:
        return cast(ConjunctionLanguageType, self._language)
    
    @Cache.cached_property
    def index_set(self) -> Set[int]:
        return set(self.indices)

    @Cache.cached_property
    def validity(self) -> np.ndarray:
        validity = self.language.validate(self.predicates)
        return validity

    @property
    def refinements(self) -> Sequence['ConjunctionSelectorBase']:
        return list(self._language.refine(self))

    @property
    def index_max(self) -> int:
        '''The index of the largest contained predicate w.r.t. the implied total predicate ordering'''
        indices = self._predicate_indices
        return indices[-1] if indices else None
    
    @property
    def predicate_max(self) -> Predicate:
        '''The largest contained predicate w.r.t. the implied total predicate ordering'''
        max_index = self.index_max
        return None if max_index is None else self.language.predicate_object(max_index)
    
    def extend(self, predicate: PredicateOrIndexType) -> 'ConjunctionSelector':
        '''Create a new selector with an extra predicate appended to the current one'''
        raise NotImplementedError()
    
    def __contains__(self, predicate: PredicateOrIndexType) -> bool:
        index = self.language.predicate_index(predicate)
        return index in self.index_set

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

    def __init__(self, language: 'ConjunctionLanguage', predicates: PredicateOrIndexCollectionType, cache: CacheSpecType=True) -> None:
        predicates = tuple(predicates)
        super().__init__(language=language, predicates=predicates, cache=cache)
        self._index_last_extension = self.language.predicate_index(predicates[-1]) if predicates else None

    def extend(self, predicate: PredicateOrIndexType) -> 'ConjunctionSelector':
        '''Create a new selector with an extra predicate appended to the current one'''
        extension_index = self.language.predicate_index(predicate)
        predicate_indices = chain(self.indices, [extension_index])
        return ConjunctionSelector(self.language, predicate_indices)
    
    @property
    def index_last_extension(self):
        '''The index of the last predicate used to refine this selector'''
        return self._index_last_extension

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

    def _indices_refinement_extensions(self, selector: ConjunctionSelectorBase, greater_only: Optional[bool]=False,
                                       blacklist: Optional[PredicateOrIndexCollectionType]=None) -> Iterator[int]:
        '''Iterate through those predicate indices that create refinements through extending a current selector'''
        max_index = selector.index_max
        num_predicates = len(self.predicates)
        indices: Iterator[int]
        if greater_only:
            index = max_index if max_index is not None and greater_only else -1
            indices = iter(range(index + 1, num_predicates))
        else:
            indices = filter(lambda p: p not in selector, range(num_predicates))
        if blacklist is not None:
            blacklist_set = set(self.predicate_indices(blacklist))
            indices = filter(lambda x: self.predicate_index(x) not in blacklist_set, indices)
        return iter(indices)
        
    def refine(self, selector: ConjunctionSelectorBase, greater_only: Optional[bool]=True,
               blacklist: Optional[PredicateOrIndexCollectionType]=None) -> Iterator[ConjunctionSelectorBase]:
        '''Return a refinement of the specified selector'''
        extension_indices = self._indices_refinement_extensions(selector=selector, greater_only=greater_only, blacklist=blacklist)
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
            return ConjunctionSelector(self, predicates)

        return ConjunctionLanguage.SelectorParser(self, make_selector)
    
    def deserialise_selector(self, digest):
        return ConjunctionSelector.deserialise(self, digest)

    def __summary_dict__(self, options:SummaryOptions):
        dct = {'data-name': self.data.name, 'num-predicates':len(self.predicates)}
        #if options.parts & SummaryParts.LANGUAGE_PREDICATES:
        #    dct['predicates'] = SummarisableList(self.predicates)
        return dct
    
