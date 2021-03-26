'''
Created on Dec 8, 2017

@author: janis
'''
 
# pylint: disable=invalid-name, pointless-string-statement
 
import re
from typing import Sequence, Optional, Tuple, List, Mapping, Union, Iterator, Dict, Any, \
    Set, Iterable, Callable, cast, TypeVar, Generic, NamedTuple
from itertools import chain, islice, repeat

import numpy as np

from colito.cache import ValueCache as Cache
from colito.indexing import Indexer, ClassCollection, SimpleIndexer
from colito.summaries import Summarisable, SummaryOptions, SummarisableList
from colito.logging import getModuleLogger
from .utils import EffectiveValiditiesTracker, EffectiveValiditiesView, CandidateRestriction, Indices
from ..predicates import Predicate

PredicateCollectionType = Iterable[Predicate]  # pylint: disable=invalid-name
PredicateOrIndexType = Union[Predicate, int]  # pylint: disable=invalid-name
PredicateOrIndexCollectionType = Union[PredicateCollectionType, Iterable[int]]  # pylint: disable=invalid-name
CacheSpecType = Union[bool, Cache, Dict]

SelectorType = TypeVar('SelectorType')
LanguageType = TypeVar('LanguageType')

log = getModuleLogger(__name__)


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

    class ParseError(RuntimeError):
        pass
    
    def __init__(self, language: LanguageType) -> None:
        self._language: LanguageType = language

    @property
    def language(self) -> LanguageType:
        return self._language
        
    def parse(self, string:str) -> SelectorType:
        raise NotImplementedError()


class Selector():
    '''Selector for a set of entities'''
    def __init__(self, cache: CacheSpecType=True):
        self._cache: Cache = Cache.from_spec(cache)
        
    @property
    def validity(self) -> np.ndarray:
        '''A logical of those entities that validate the selector'''
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


class StaticSelector(Selector):

    def __init__(self, validity:np.ndarray, cache: CacheSpecType=True) -> None:
        super().__init__(cache=cache)
        self._validity:np.ndarray = validity
        
    @property
    def validity(self) -> np.ndarray:
        return self._validity


class Language(Generic[SelectorType]):
    '''Language of selectors'''

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

       
class ConjunctionSelectorBase(Summarisable, LanguageSelector['ConjunctionLanguage']):
    '''Selector taking the conjunction of a set of predicates'''

    def __init__(self, language: 'ConjunctionLanguage', predicates: PredicateOrIndexCollectionType, cache: CacheSpecType=True) -> None:
        super().__init__(language=language, cache=cache)
        predicate_objects = sorted(self.language.predicate_objects(predicates))
        self._predicate_indices: Tuple[int, ...] = tuple(language.predicate_indices(predicate_objects))

    @property_predicate_objects
    def predicates(self) -> Tuple[int, ...]:
        '''Predicates the current selector has'''
        return self._predicate_indices

    @property
    def indices(self) -> Tuple[int, ...]:
        '''Predicates the current selector has'''
        return self._predicate_indices

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

    def summary_dict(self, options:SummaryOptions):
        dct = {'description':str(self)}
        if options.parts & SummaryParts.PREDICATE_INDICES:
            dct['predicate_indices'] = self.indices
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


class ConjunctionLanguage(Summarisable, Language):
    '''Language of predicate conjunctions'''
    tag = 'conjunctions'

    def __init__(self, data, predicates) -> None:
        super(ConjunctionLanguage, self).__init__(data=data)
        self._data = data
        self._predicate_indexer = SimpleIndexer(predicates)
        self._root: ConjunctionSelector = ConjunctionSelector(self, [])
    
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
        
    def refine(self, selector: ConjunctionSelectorBase, greater_only: Optional[bool]=False,
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

    def summary_dict(self, options:SummaryOptions):
        dct = {'data-name': self.data.name, 'num-predicates':len(self.predicates)}
        if options.parts & SummaryParts.LANGUAGE_PREDICATES:
            dct['predicates'] = SummarisableList(self.predicates)
        return dct
    
class ClosureConjunctionSelector(ConjunctionSelector, LanguageSelector['ClosureConjunctionLanguageBase']):

    def __init__(self, language: 'ClosureConjunctionLanguageBase',
                 predicates_path: PredicateOrIndexCollectionType,
                 cache: CacheSpecType=True,
                 _closure: Optional[PredicateOrIndexCollectionType]=None) -> None:
        self._language: ClosureConjunctionLanguageBase
        self._indices_path: Tuple[int, ...] = tuple(language.predicate_indices(predicates_path))
        local_cache = Cache.from_spec(cache)
        validity = language.validate(self._indices_path)
        if _closure is None:
            closure_indices = language.indices_closure(validity)
        else:
            closure_indices_raw = language.predicate_indices(_closure)
            closure_indices = tuple(set(closure_indices_raw) | set(self._indices_path))
        local_cache.a.validity = validity
        super(ClosureConjunctionSelector, self).__init__(language=language, predicates=closure_indices, cache=local_cache)

    @property
    def language(self) -> 'ClosureConjunctionLanguageBase':
        return cast(ClosureConjunctionLanguageBase, super(ClosureConjunctionSelector, self).language)

    @property_predicate_objects
    def predicates_path(self) -> Tuple[int, ...]:
        '''Predicates created as the current selector was a descendant of during its creation.'''
        return self._indices_path

    @property
    def indices_path(self) -> Tuple[int, ...]:
        '''Indices of predicates created as the current selector was a descendant of during its creation.'''
        return self._indices_path
    
    @property
    def index_last_extension(self):
        '''The index of the last predicate used to refine this selector'''
        return self.indices_path[-1] if self.indices_path else None
    
    @Cache.cached_property
    def indices_compact(self) -> Tuple[int, ...]:
        indices_compact = self.language.indices_minimal_approximation(self.validity)
        return indices_compact
        
    @property_predicate_objects
    def predicates_compact(self) -> Tuple[int, ...]:
        '''Predicate indices in the closure'''
        return self.indices_compact

    @Cache.cached_property
    def validity(self) -> np.ndarray:
        validity = self.language.validate(self.predicates)
        return validity

    @Cache.cached_property
    def index_max(self):
        '''The largest contained predicate without which the selector changes'''
        index = self._language.index_last_needed(self.validity, self.indices)
        return index

    @property
    def path_description(self) -> str:
        path_selector = ConjunctionSelector(self.language, predicates=self.predicates_path, cache=False)
        return str(path_selector)

    
    @property
    def predicate_max(self) -> Predicate:
        index_max = self.index_max
        return self.language.predicate_object(index_max) if index_max is not None else None
    
    @Cache.cached_property
    def tail_indices(self):
        index_max = self.index_max
        tail_indices = tuple(filter(lambda x: x < index_max, self.indices))
        return tail_indices

    def extend(self, predicate: PredicateOrIndexType, _closure=None) -> 'ClosureConjunctionSelector':
        '''Create a new selector with an extra predicate appended to the current one
        
        @param greater_only: If True, only the predicates after the extension are candidates for closure search. 
        '''
        extension_index = self.language.predicate_index(predicate)
        predicates = chain(self._indices_path, [extension_index])
        return ClosureConjunctionSelector(self.language, predicates, _closure=_closure)
    
    @Cache.cached_property
    def closure_indices(self):
        return self._language.indices_closure(self)
    
    def __repr__(self):
        predicates = ' AND '.join(map(repr, self.predicates))
        return '<{0.__class__.__name__}: {1}>'.format(self, predicates)

    def __str__(self):
        predicates = '^'.join(map(str, self.predicates))
        return f'{{{predicates}}}'

    class Digest(NamedTuple):  # pylint: disable = too-few-public-methods
        indices: List[int]
        indices_path: List[int]
    
    def serialise(self, json=False):
        digest = ClosureConjunctionSelector.Digest(indices=self.indices, indices_path=self.indices_path)
        if json:
            digest = digest._asdict()  # pylint: disable=no-member
        return digest
    
    @classmethod
    def deserialise(cls, language, digest):
        if isinstance(digest, dict):
            digest = ClosureConjunctionSelector.Digest(**digest)
        return cls(language, predicates_path=digest.indices_path, _closure=digest.indices)
    
    def summary_dict(self,options:SummaryOptions):
        dct = super().summary_dict(options)
        dct['path_description'] = self.path_description
        if options.parts & SummaryParts.PREDICATE_INDICES:
            dct['path_indices'] = self.indices_path
            dct['compact_indices'] = self.indices_compact
        return dct

class ClosureConjunctionLanguageBase(ConjunctionLanguage):

    def __init__(self, data, predicates: Optional[PredicateCollectionType]=None) -> None:
        super(ClosureConjunctionLanguageBase, self).__init__(data=data, predicates=predicates)
        self._predicate_validities: np.ndarray = np.array(np.stack([predicate.validate() for predicate in self.predicates], axis=1), order='F')
        self._root: ClosureConjunctionSelector = ClosureConjunctionSelector(self, [])
        self._predicate_supports: np.ndarray = self._predicate_validities.sum(axis=0)
        
    def refine(self, selector: ClosureConjunctionSelector, greater_only: Optional[bool]=False,
               blacklist: Optional[PredicateOrIndexCollectionType]=None) -> Iterator[ClosureConjunctionSelector]:
        raise NotImplementedError()
    
    def indices_minimal_approximation(self, validity: np.ndarray, indices_closure=None) -> Tuple[int, ...]:
        '''Compute an approximation to the minimal predicate list describing the closure'''
        if indices_closure is None:
            indices_closure = np.fromiter(self.indices_closure(validity), int)
        minimal_indices_closure = []
        remainder = ~validity
        closure_validities = self._predicate_validities[:, indices_closure]
        while remainder.sum():
            minimal_index = np.argmin(closure_validities[remainder, :].sum(axis=0))
            remainder &= closure_validities[:, minimal_index]
            minimal_indices_closure.append(minimal_index)
        minimal_indices = indices_closure[minimal_indices_closure]
        return tuple(map(int, minimal_indices))
    
    def select(self, predicates, _closure=None):
        return ClosureConjunctionSelector(self, predicates_path=predicates, _closure=_closure)
    
    def index_last_needed(self, validity: np.ndarray, indices: Tuple[int, ...]) -> int:
        '''Return the predicate with the maximal index that is necessary to not change the selector support''' 
        if indices:
            support = validity.sum()
            indices_sorted = sorted(indices)
            buffer = self._predicate_validities[:, indices_sorted]
            running_coverage = np.cumprod(buffer, out=buffer, axis=1, dtype=bool).sum(axis=0)
            closure_index = np.where(running_coverage == support)[0][0]
            minimum_index = int(indices_sorted[closure_index])
        else:
            minimum_index = None
        return minimum_index
        
    def indices_closure(self, validity: np.ndarray, candidate_indices=slice(None, None, None)) -> Tuple[int, ...]:
        '''Return the predicate indices forming the closure of the selector'''
        num_indices = len(self.predicates)
        indices = np.arange(num_indices)[candidate_indices]
        closure_index_candidates = np.where(self._predicate_validities[np.ix_(validity, indices)].all(axis=0))[0]
        closure_index = indices[closure_index_candidates]
        return tuple(map(int, closure_index))

    def _get_validity_and_support(self, what:Union[Predicate, Selector]):
        if isinstance(what, Predicate):
            master_idx = self.predicate_index(what)
            validity = self._predicate_validities[:, master_idx]
            support = self._predicate_supports[master_idx]
        elif isinstance(what, Selector):
            validity = what.validity
            support = validity.sum()
        elif isinstance(what, np.ndarray):
            validity = cast(np.ndarray, what)
            support = validity.sum()
        else:
            raise TypeError(f'Can not infer validity from object {what} of type {what.__class__}.')
        return validity, support 
    
    def predicate_is_superset(self, master:Union[Predicate, Selector], candidates:Sequence[Predicate], only_proper:bool=False) -> np.ndarray:
        master_validity, master_support = self._get_validity_and_support(master)
        candidate_idx = list(self.predicate_indices(candidates))
        cover_sums = np.bitwise_and(master_validity[:,None], self._predicate_validities[:, candidate_idx]).sum(axis=0)
        return (cover_sums > master_support) if only_proper else (cover_sums >= master_support)

    def predicate_is_subset(self, master:Union[Predicate, Selector], candidates:Sequence[Predicate], only_proper:bool=False) -> np.ndarray:
        master_validity, master_support = self._get_validity_and_support(master)
        candidate_idx = list(self.predicate_indices(candidates))
        cover_sums = np.bitwise_and(master_validity[:,None], self._predicate_validities[:, candidate_idx]).sum(axis=0)
        return (cover_sums < master_support) if only_proper else (cover_sums <= master_support)

    class Digest(NamedTuple):  # pylint: disable = too-few-public-methods
        language: str
        selectors: List[ClosureConjunctionSelector.Digest]
        
    def serialise_selectors(self, selectors):
        digests = list(s.serialise() for s in selectors)
        return ClosureConjunctionLanguageBase.Digest(language=self.tag, selectors=digests) # pylint: disable = no-value-for-parameter)
    
    def deserialise_selectors(self, digest):
        selectors = list(ClosureConjunctionSelector.deserialise(self, d) for d in digest.selectors)
        return selectors

    def deserialise_selector(self, digest):
        return ClosureConjunctionSelector.deserialise(self, digest)

class ClosureConjunctionLanguageRestricted(ClosureConjunctionLanguageBase):
    tag = 'closure-conjunctions-restricted'

    def refine(self, selector: ClosureConjunctionSelector, blacklist: Optional[PredicateOrIndexCollectionType]=None) -> Iterator[ClosureConjunctionSelector]:
        
        num_predicates = len(self.predicates)
        if blacklist is not None:
            blacklist = np.fromiter(self.predicate_indices(blacklist),int)
            whitelist = Indices.invert(blacklist, num_predicates)
        else:
            blacklist = np.zeros(0, int)
            whitelist = np.arange(num_predicates)
                
        restrictions = []
        ev_tracker = EffectiveValiditiesTracker(self._predicate_validities, effective_validity=selector.validity)
        ev_view = EffectiveValiditiesView(tracker=ev_tracker)
        
        is_root_superset = ev_tracker.supports == selector.validity.sum()
        if is_root_superset[blacklist].any():
            return
        
        root_supersets = np.where(is_root_superset)[0]
        whitelist = whitelist[~is_root_superset[whitelist]] # no need to parse these. They are always in the closure.
        whitelist = whitelist[ev_tracker.supports[whitelist] != 0]
        blacklist = blacklist[ev_tracker.supports[blacklist] != 0]
        ev_view = ev_view.restrict_candidates(Indices.union((whitelist, blacklist)))
        root_cr = CandidateRestriction(ev_view=ev_view, 
                                       extension = whitelist,
                                       closure=whitelist,
                                       blacklist=blacklist)
        
        if root_cr.extension.size:
            restrictions.append(root_cr)
       
        while restrictions:
            restriction = restrictions.pop()
            
            min_ext = restriction.minsup_extension
            validity = restriction.validity_of(min_ext)
            supersets = restriction.supersets(validity)
            
            ' ### Handle closure printing'
            if not supersets.blacklist.indices.size:
                ' => Yield a closure comprising all <min_ext> supersets'
                # closure = selector.extend(extension, refinement_predicates)
                # yield refinement
                closure = Indices.union((root_supersets,supersets.closure.indices))
                refinement = selector.extend(min_ext, _closure=closure)
                yield refinement

            ' ### Upkeeping candidates'
            '''remove all {extension}, {closure} and {blacklist} candidates that are:
                - equal to <minexp>
                - have a support less than <minexp>
                NOTE: All those sets are active when they are supersets of an extension.
                      If their support is smaller, there is no need to keep them around.
            '''
            
            ' ### Handle superset candidates'
            '''The new state is:
             * A Candidates object with:
                 > extension: only the PROPER supersets of <min_ext> in the {extension} candidates
                 > closure: only the PROPER supersets of <min_ext> in the {closure} candidates
                 > blacklist: only the PROPER supersets of <min_ext> in {blacklist} candidates
             * A value of predicates extended by 
                 > all equal {candidate} supersets of <minext>  
             '''
            
            restr_ss = restriction.restrict(ev=~validity,
                                            extension=supersets.extension.proper,
                                            closure=supersets.closure.proper,
                                            blacklist=supersets.blacklist.proper)
            if restr_ss.extension.size:
                restrictions.append(restr_ss)
                # print(restr_ss.dump())                
                # print(f'SS restriction: {restr_ss}')
            
            ' == Handle non-superset candidates'
            ''' Create a Candidates object with:
             * extension: all extension candidates except:
                          - all supersets of <min_ext>
                             * the equal sets are already contained in the closure
                             * the proper supersets are handled by the superset candidates
             * closure:   all {closure} candidates
                          (- except: all sets equal to <min_ext>) 
             * blacklist: all {blacklist} candidates
                          (- except: all sets equal to <min_ext>) 
            '''

            restr_non_ss = restriction.remove_candidates(extension=supersets.extension.indices,
                                                         closure=supersets.closure.equal,
                                                         blacklist=supersets.blacklist.equal)
            if restr_non_ss.extension.size:
                restrictions.append(restr_non_ss)
                # print(restr_non_ss.dump())                
                # print(f'NSS restriction: {restr_non_ss}')
        
class ClosureConjunctionLanguageFull(ClosureConjunctionLanguageBase):
    tag = 'closure-conjunctions-full'

    def refine(self, selector: ClosureConjunctionSelector, blacklist: Optional[PredicateOrIndexCollectionType]=None) -> Iterator[ClosureConjunctionSelector]:
        raise NotImplementedError()  # this code is wrong
        num_predicates = len(self.predicates)

        extension_indices = tuple(self._indices_refinement_extensions(selector=selector, blacklist=blacklist))
        num_predicates = len(self.predicates)
        selector_predicate_supports = self._predicate_validities[selector.validity, :].sum(axis=0)
        extension_indices_remain = np.ones(num_predicates, bool)
        extension_indices_remain[list(selector.indices)] = False
        refinement_covered = np.zeros(num_predicates, bool)  # dbgremove
        for extension_index in extension_indices:
            if extension_indices_remain[extension_index]:
                refinement = selector.extend(extension_index)
                refinement_indices = list(refinement.indices)
                support = refinement.validity.sum()
                refinement_covered[:] = False  # dbgremove
                refinement_covered[refinement_indices] = True  # dbgremove
                refinement_covered &= selector_predicate_supports == support  # dbgremove
                extension_indices_remain &= ~refinement_covered  # dbgremove
                # Debug unfriendly faster equivalent:
                extension_indices_remain[refinement_indices] &= selector_predicate_supports[refinement_indices] != support
                yield refinement


class ClosureConjunctionLanguageSlow(ClosureConjunctionLanguageBase):
    tag = 'closure-conjunctions-slow'
    
    def refine(self, selector: ClosureConjunctionSelector, blacklist: Optional[PredicateOrIndexCollectionType]=None) -> Iterator[ClosureConjunctionSelector]:

        ''' We do not consider (w.r.t. the current validity):
         * completely empty predicates
         * completely full predicates
        '''
        is_nz = self._predicate_validities[selector.validity,:].any(axis=0)
        is_full = self._predicate_validities[selector.validity,:].all(axis=0)
        extension_indices = np.where(is_nz & ~is_full)[0]
        
        refinements = set(selector.extend(idx) for idx in extension_indices)
        
        iter_refinements = iter(refinements)
        if blacklist is not None:
            blacklist_set = set(blacklist)

            def not_blacklisted(selector: ConjunctionSelector):
                common = selector.index_set & blacklist_set
                return not common

            iter_refinements = filter(not_blacklisted, iter_refinements)
            
        return iter_refinements

class ClosureConjunctionLanguageTester(ClosureConjunctionLanguageRestricted):
    tag = 'closure-conjunctions-tester'
    
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self._slow = ClosureConjunctionLanguageSlow(*args, **kwargs)
        
    def refine(self, selector: ClosureConjunctionSelector, blacklist: Optional[PredicateOrIndexCollectionType]=None) -> Iterator[ClosureConjunctionSelector]:
        refinements = sorted(super().refine(selector, blacklist), key=str)
        refinements_slow = sorted(self._slow.refine(selector, blacklist), key=str)
        if refinements_slow != refinements:
            log.error(f'Refinements do not match, while refining {selector} with blacklist {blacklist}. \nSlow: {refinements_slow}\nRstr: {refinements}')
        return refinements

LANGUAGES = ClassCollection('Languages', (ClosureConjunctionLanguageRestricted, ClosureConjunctionLanguageFull,
                                          ClosureConjunctionLanguageSlow, ConjunctionLanguage,
                                          ClosureConjunctionLanguageTester))
