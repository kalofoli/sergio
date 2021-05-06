'''
Created on May 3, 2021

@author: janis
'''
from typing import NamedTuple, Iterable, Tuple

import time
import enum
import numpy as np

from sergio.language import Selector, ConjunctionLanguage, ConjunctionSelector
from sergio.scores import Measure, OptimisticEstimator

from colito.summaries import SummarisableFromFields, SummarisableList
from colito.queues import Entry, TopKQueue
from colito.logging import getModuleLogger
from colito.collection import ClassCollectionFactoryRegistrar, ClassCollection
from colito.factory import factorymethod
import typing

log = getModuleLogger(__name__)

class SearchStatus(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    COMPLETED = enum.auto()
    ABORTED = enum.auto() 


class Result(NamedTuple):
    selector: Selector
    optimistic_estimate: float
    objective_value:float
    
    def __str__(self):
        return fr'(f:{self.objective_value:5.3f}/{self.optimistic_estimate:5.3f}|{self.selector})'

    def __repr__(self):
        return f'<{self.__class__.__name__}{self!s}>'

class AddResultOutcome(NamedTuple):
    entry_out: Entry
    was_added: bool
    objective_value: float


class SearchState:
    __slots__ = ('search','selector','pruned','depth','_objective_value','_optimistic_estimate')
    
    def __init__(self, search:'LanguageTopKBranchAndBound', selector: Selector, pruned: typing.Set[int], depth: int):
        self.search:'LanguageTopKBranchAndBound' = search
        self.selector: Selector = selector
        self.pruned: typing.Set[int] = pruned
        self.depth:int = depth
        self._optimistic_estimate = None
        self._objective_value = None
    
    @property
    def optimistic_estimate(self)-> float:
        if self._optimistic_estimate is None:
            self._optimistic_estimate = self.search.optimistic_estimate(self.selector)
        return self._optimistic_estimate
    
    @property
    def objective_value(self)-> float:
        if self._objective_value is None:
            self._objective_value = self.search.objective_value(self.selector)
        return self._objective_value
    
    @property
    def blacklist(self):
        return self.pruned

    @staticmethod
    def set2str(indices, total, sep='+-'):
        must_invert = len(indices)>total
        if must_invert:
            inverse = np.arange(total,int)
            idl_inverse = np.ones(total,bool)
            idl_inverse[indices] = False
            inverse = np.arange(total)[idl_inverse]
            indices = inverse
        return '{'+sep[must_invert]+sep[must_invert].join(map(str, indices))+'}'
        
    def __str__(self):
        set2str = self.set2str
        num_preds = len(self.selector.language.predicates)
        prn = lambda v: f'{v:5.3f}' if v is not None else 'XXXXX'
        return (fr'({prn(self._objective_value)}/{prn(self._optimistic_estimate)}'
                fr'|@{self.depth}\{len(self.pruned)}'
                fr'|{self.selector}{self.selector.indices_path if hasattr(self.selector,"indices_path") else ""}'
                fr'\({set2str(self.pruned,num_preds,sep=",")})')

    def __repr__(self):
        return f'<{self.__class__.__name__}{self!s}>'
    
    def serialise(self, json=False):
        dig = self._replace(selector=self.selector.serialise(json=json))
        if json:
            dig = dig._replace(pruned=list(dig.pruned), covered=list(dig.covered))._asdict()
        return dig
    
    @classmethod
    def deserialise(cls, language, digest):
        if isinstance(digest, dict):
            digest = cls(**digest)
            digest._replace(pruned=set(digest.pruned), covered=set(digest.covered))
        state = digest._replace(selector=language.deserialise_selector(digest.selector))
        return state

    def __summary_dict__(self, selector_dict=None, suffix=''):
        dct = {f'selector{suffix}':self.selector if selector_dict is None else selector_dict[self.selector],
               f'pruned{suffix}': list(self.pruned),
               f'covered{suffix}': list(self.covered),
               f'selector_depth{suffix}': self.depth,
               f'objective_value{suffix}': self.objective_value,
               f'optimistic_estimate{suffix}': self.optimistic_estimate,
               }
        return dct


class SearchVisitor:
    def start(self, dfs, root_state:SearchState):pass
    def stop(self, dfs):pass
    def state_popped(self, dfs, state:SearchState): pass
    def state_expanded(self, dfs, state:SearchState, valid_states, new_states): pass
    def result_added(self, state:SearchState, result_old:'LanguageTopKBranchAndBound.Result'): pass



SUBGROUP_SEARCHES = ClassCollection('Subgroup Searches')
class SubgroupSearch(SummarisableFromFields,ClassCollectionFactoryRegistrar):
    __collection_tag__ = None
    __collection_factory__ = SUBGROUP_SEARCHES

    __summary_fields__ = ['subgroups','approximation_factor','k','language','measure','optimistic_estimator', 'time_elapsed', 'time_started', 'status']
    __summary_convert__ = {'subgroups': SummarisableList}

    def __init__(self, language:ConjunctionLanguage, measure:Measure, optimistic_estimator:OptimisticEstimator, k:int=1, max_best:bool=True, approximation_factor:float=1.) -> None:
        self._language:ConjunctionLanguage = language
        self._measure:Measure = measure
        self._optimistic_estimator:OptimisticEstimator = optimistic_estimator
        self._results: TopKQueue[Result] = TopKQueue(k=k, max_best=max_best)
        self._approximation_factor = approximation_factor
        
        self._last_exception = None
        self._status: SearchStatus = SearchStatus.IDLE
        self._time_started:float = None
        self._time_elapsed:float = None
    
    def _start(self):
        self._time_started = time.time()
        self._time_elapsed = None
        self._status = SearchStatus.RUNNING

    def _stop(self) -> float:
        if not self.is_running:
            raise RuntimeError('Algorithm not running.')
        self._time_elapsed = time.time() - self._time_started
        self._status = SearchStatus.COMPLETED
        return self._time_elapsed
    
    def _abort(self, exception=None):
        '''Abort run (possibly due to an exception)
        
        @return: the elapsed time so far.
        '''
        if exception is not None:
            self._last_exception = exception
        self._time_elapsed = time.time() - self._time_started
        self._status = SearchStatus.ABORTED
        return self._time_elapsed
    
    @property
    def time_elapsed(self):
        return self._time_elapsed
    
    @property
    def time_started(self):
        return self._time_started
    
    @property
    def k(self):
        '''The number of topmost (k) results to track'''
        return self._results.k
        
    @property
    def status(self) -> SearchStatus:
        return self._status
    
    @property
    def is_running(self) -> bool:
        '''Whether the algorithm is running'''
        return self._status == SearchStatus.RUNNING
    
    @property
    def last_exception(self) -> Exception:
        return self._last_exception
    
    @property
    def results(self) -> TopKQueue[Result]:
        return self._results
    
    @property
    def subgroups(self) -> Tuple[ConjunctionSelector, ...]:
        return tuple(element.data.selector for element in self.results.entries())
    
    @property
    def language(self) -> ConjunctionLanguage:
        '''The used language'''
        return self._language
    
    @property
    def measure(self) -> Measure:
        '''The used measure'''
        return self._measure
    
    @property
    def optimistic_estimator(self) -> OptimisticEstimator:
        '''The used optimistic estimator'''
        return self._optimistic_estimator

    def objective_value(self, selector:Selector) -> float:
        '''Compute objective value
        Used as callback from SearchState'''
        return self.measure.evaluate(selector)
    
    def optimistic_estimate(self, selector:Selector) -> float:
        '''Compute optimistic estimate
        Used as callback from SearchState'''
        return self.optimistic_estimator.evaluate(selector)
        
    @property
    def approximation_factor(self):
        '''The approximation factor to use while optimising. The result found has at least this big a score times that of the optimum.''' 
        return self._approximation_factor
        
    def _run(self):
        raise NotImplementedError()
    
    def run(self, *args, **kwargs):
        self._start()
        try:
            result = self._run(*args, **kwargs)
            self._stop()
        except BaseException as e:
            result = self._abort(e)
            raise
        return result
        
    def try_add_selector(self, selector, objective_value: float=None, optimistic_estimate: float=None, quiet=False) -> 'LanguageTopKBranchAndBound.AddResultOutcome':
        if objective_value is None:
            objective_value = self.measure.evaluate(selector)
        result = Result(selector=selector, objective_value=objective_value, optimistic_estimate=optimistic_estimate)
        entry_out, was_added = self.results.add(data=result, value=objective_value)
        if was_added and not quiet:
            # if optimistic_estimate is not None or (isinstance(self.optimistic_estimator, CachingEvaluator) and self.optimistic_estimator.cache_isset(selector)):
            #    fest = optimistic_estimate if optimistic_estimate is not None else self.optimistic_estimator.evaluate(selector)
            #    text = f', Opt.Est.={fest:7.4f}'
            # else:
            #    text = ''
            # log.rlim.info and log.info(f'Added selector (Value={objective_value:7.4f}{text}): {selector!s}')
            log.rlim.info and log.info(f'Added new result {result}')
            log.rlim.debug and log.debug(f'Current results: {self.results.elements()}')
        
        return AddResultOutcome(entry_out, was_added, objective_value)
    
    def try_add_selectors(self, selectors, quiet=False) -> 'LanguageTopKBranchAndBound.AddResultOutcome':
        make_result = lambda selector: Result(selector=selector,
                                              objective_value=self.measure.evaluate(selector),
                                              optimistic_estimate=self.optimistic_estimator.evaluate(selector))
        return self.try_add_results(map(make_result, selectors), quiet=quiet)
    
    def try_add_result(self, result: 'LanguageTopKBranchAndBound.Result', quiet=False) -> 'LanguageTopKBranchAndBound.AddResultOutcome':
        return self.try_add_selector(**result._asdict(), quiet=quiet)

    def try_add_results(self, results: Iterable['LanguageTopKBranchAndBound.Result'], quiet=False) -> 'LanguageTopKBranchAndBound.AddResultOutcome':
        # TODO: Improve/faster? Needs support by queue
        for result in results:
            self.try_add_result(result, quiet=quiet)
        return None
    

