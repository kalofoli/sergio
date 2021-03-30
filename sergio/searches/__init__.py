#! /usr/bin/env python3

import logging

import enum
import math
import time
import itertools
import numpy as np
import operator
import functools

from typing import Set, Callable, Tuple, NamedTuple, cast, Iterator, Collection,\
    List, Iterable, Union
from collections import namedtuple, Counter

from colito.logging import getModuleLogger, ColitoLogger
from colito.statistics import StatisticsBase, StatisticsCounter, updater as stats_updater,\
    StatisticsUpdater
from colito.summaries import SummaryOptions, SummarisableList, Summarisable,\
    SummarisableDict, SummarisableAsDict, summary_from_fields

from ..scores import OptimisticEstimator, Measure
from ..language import Language, Selector, ConjunctionSelector, ConjunctionLanguage
from ..queues import PriorityQueue, TopKQueue, MaxEntry, Entry
from ..predicates import Predicate
from builtins import classmethod, property
from colito.factory import FactoryBase, factorymethod, FactoryGetter, \
    ProductBundle, FactoryDescription

log = getModuleLogger(__name__, factory=ColitoLogger)
logging.addLevelName(25, 'PROGRESS')

class CandidateScoreFunctions:
    '''List of score functions ictating the order of a candidate in the priority queue'''
    
    @classmethod
    def optimistic_estimate(cls, candidate: 'Candidate') -> float:
        return candidate.optimistic_estimate

    @classmethod
    def opus_optimistic_estimate(cls, candidate: 'OPUSCandidate') -> float:
        return cls.optimistic_estimate(candidate)


class Candidate:

    def __init__(self, bnb: 'BranchAndBound', selector: Selector) -> None:
        self._selector: Selector = selector
        self._optimistic_estimate:float = bnb.optimistic_estimator.evaluate(selector)
        self._objective_value:float = bnb.measure.evaluate(selector)

    @property
    def optimistic_estimate(self) -> float:
        return self._optimistic_estimate

    @property
    def objective_value(self) -> float:
        return self._objective_value

    @property
    def selector(self) -> Selector:
        return self._selector

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self!s}>'

    def __str__(self):
        return f'(f:{self.objective_value:.4g} fmax:{self.optimistic_estimate:.4g} {self.selector})'


class Statistics(StatisticsBase):
    increase_popped = StatisticsUpdater('popped', 1, doc='Popped nodes')
    increase_queued = StatisticsUpdater('queued', 1, doc='Queued nodes')
    increase_results = StatisticsUpdater('results', 1, doc='Number of results')
    increase_survived = StatisticsUpdater('survived', 1, doc='Survived nodes')

    
class BranchAndBound:
    
    def __init__(self, language: Language,
                 measure: Measure, optimistic_estimator: OptimisticEstimator, k:int=1,
                 candidate_scoring:Callable[[Candidate], float]=CandidateScoreFunctions.optimistic_estimate) -> None:
        self._measure: Measure = measure
        self._optimistic_estimator: OptimisticEstimator = optimistic_estimator
        self._language: Language = language
        self._queue: PriorityQueue = PriorityQueue(entry_type=MaxEntry)
        self._results : TopKQueue = TopKQueue(k=k)
        self._candidate_scoring: Callable[[Candidate], float] = candidate_scoring
        self._statistics: Statistics = Statistics()
        self._root_candidate: Candidate = Candidate(self, self.language.root)
        
    @property
    def measure(self) -> Measure:
        return self._measure

    @property
    def optimistic_estimator(self) -> OptimisticEstimator:
        return self._optimistic_estimator

    @property
    def language(self) -> Language:
        return self._language

    @property
    def results(self) -> TopKQueue:
        '''Return found results'''
        return self._results
    
    def subgroups(self):
        '''Return the subgroups of the current result set, in descending order of fitness'''
        return tuple(e.data for e in self.results.entries())
    
    def _update_results(self, candidate: Candidate) -> None:
        """Callback invoked when a new candidate needs to be added to the results"""
        entry = self._results.make_entry(data=candidate.selector, value=candidate.objective_value)
        self._results.add(entry=entry)
        log.info("Added to top-%d: %.4f (%s)", self._results.k, candidate.objective_value, candidate.selector)

    def _generate_refinements(self, subgroup: Candidate) -> Iterator[Candidate]:
        return map(lambda refinement: Candidate(bnb=self, selector=refinement), subgroup.selector.refinements)

    def run(self) -> Tuple[Entry, ...]:
        self._queue.clear()
        self._queue.push(data=self._root_candidate, priority=0)
        subgroup: Candidate
        while self._queue:
            subgroup, priority_value = tuple(self._queue.pop())  # pylint: disable=unused-variable
            log.progress((f'Popped subgroup (Q:%6d) (f={subgroup.objective_value:-8f} '
                          f'fest={subgroup.optimistic_estimate:-8f}, t={self._results.threshold:-8f}, '
                          f'r\'={self._root_candidate.optimistic_estimate:.8}): {subgroup.selector}'),
                     len(self._queue))
            self._statistics.increase_popped()
            
            if subgroup.optimistic_estimate > self._results.threshold:
                self._statistics.increase_survived()

                refinements = self._generate_refinements(subgroup)
                for refinement in refinements:
                    if refinement.objective_value > self._results.threshold:
                        self._update_results(refinement)
                        self._statistics.increase_results()
                    priority = self._candidate_scoring(refinement)
                    self._queue.push(data=refinement, priority=priority)
                    self._statistics.increase_queued()
        return self.results.entries()


class OPUSCandidate(Candidate):

    def __init__(self, opus: 'OPUS', selector: Selector, blacklist: Collection[Predicate]) -> None:
        super(OPUSCandidate, self).__init__(bnb=opus, selector=selector)
        self._blacklist: Set[Predicate] = set(blacklist)
    
    @property
    def blacklist(self) -> Set[Predicate]:
        return self._blacklist


class OPUS(BranchAndBound):

    def __init__(self, language: ConjunctionLanguage,
                 measure: Measure, optimistic_estimator: OptimisticEstimator, k:int=1,
                 candidate_scoring:Callable[[OPUSCandidate], float]=CandidateScoreFunctions.opus_optimistic_estimate) -> None:  # pylint: disable=too-many-arguments
        super(OPUS, self).__init__(language=language, measure=measure, optimistic_estimator=optimistic_estimator, k=k,
                                   candidate_scoring=cast(Callable[[Candidate], float], candidate_scoring))
        self._root_candidate: OPUSCandidate = OPUSCandidate(opus=self, selector=self.language.root, blacklist=[])
    
    def _generate_opus_refinements(self, subgroup: OPUSCandidate) -> Tuple[Iterator[OPUSCandidate], Set[Predicate]]:
        blacklist: Set[Predicate] = set(subgroup.blacklist)
        make_candidate = lambda refinement: OPUSCandidate(opus=self, selector=refinement, blacklist=blacklist)
        refinements = self.language.refine(subgroup.selector, blacklist=blacklist, greater_only=True)
        return map(make_candidate, refinements), blacklist
    
    def run(self) -> Tuple[Entry, ...]:
        self._queue.clear()
        self._queue.push(data=self._root_candidate, priority=0)
        subgroup: OPUSCandidate
        while self._queue:
            subgroup, priority_value = tuple(self._queue.pop())  # pylint: disable=unused-variable
            log.progress((f'Popped subgroup (Q:%6d) (f={subgroup.objective_value:-8f} '
                          f'fest={subgroup.optimistic_estimate:-8f}, t={self._results.threshold:-8f}, '
                          f'r\'={self._root_candidate.optimistic_estimate:.8}): {subgroup.selector}'),
                     len(self._queue))
            self._statistics.increase_popped()
            
            if subgroup.optimistic_estimate > self._results.threshold:
                self._statistics.increase_survived()

                refinements, blacklist = self._generate_opus_refinements(subgroup)
                for refinement in refinements:
                    if refinement.objective_value > self._results.threshold:
                        self._update_results(refinement)
                        self._statistics.increase_results()
                    else:
                        blacklist |= set(refinement.selector.tail_indices)
                    priority = self._candidate_scoring(refinement)
                    self._queue.push(data=refinement, priority=priority)
                    self._statistics.increase_queued()
        return self.results.entries()


class BlacklistCandidate(Candidate):

    def __init__(self, cpid: 'IterativeDeepening', selector: Selector, blacklist: Collection[Predicate]) -> None:
        super(BlacklistCandidate, self).__init__(bnb=cpid, selector=selector)
        self._blacklist: Set[Predicate] = set(blacklist)
    
    @property
    def blacklist(self) -> Set[Predicate]:
        return self._blacklist


class DFSState(NamedTuple):
    search:'LanguageTopKBranchAndBound'
    selector: Selector
    pruned: Set[int]
    covered: Set[int]
    depth:int
    
    @property
    def optimistic_estimate(self)-> float:
        return self.search.optimistic_estimator.evaluate(self.selector)
    
    @property
    def objective_value(self)-> float:
        return self.search.measure.evaluate(self.selector)
    
    @property
    def blacklist(self):
        return self.pruned | self.covered

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
        return (fr'({self.objective_value:5.3f}/{self.optimistic_estimate:5.3f}'
                fr'|@{self.depth}\{len(self.pruned)}+{len(self.covered):3d}'
                fr'|{self.selector}{self.selector.indices_path if hasattr(self.selector,"indices_path") else ""}'
                fr'\({set2str(self.covered,num_preds,sep="-+")}U{set2str(self.pruned,num_preds,sep="-+")})')

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
               f'search_max_depth{suffix}': self.search.max_depth,
               f'objective_value{suffix}': self.objective_value,
               f'optimistic_estimate{suffix}': self.optimistic_estimate,
               }
        return dct

class ScoringFunctions(FactoryBase):
    
    @factorymethod(name='value_addition')
    @classmethod
    def make_value_addition(cls, mult_oest:float=1., mult_fval:float=1.):

        def score(state:DFSState) -> float:
            return (state.optimistic_estimate * mult_oest if mult_oest else 0) + (state.objective_value * mult_fval if mult_fval else 0)

        return score
    
    @factorymethod(name='coverage')
    @classmethod
    def make_coverage(cls, increasing:bool=True):
        mult = 1 if increasing else -1

        def score(state:DFSState) -> float:
            nonlocal mult
            return mult * state.selector.validity.mean()
        
        return score

    @factorymethod(name='selector')
    @classmethod
    def make_selector(cls, record:str='indices'):
        def score(state:DFSState) -> float:
            nonlocal record
            return getattr(state.selector, record)
        
        return score
    
    @factorymethod(name='optimistic_estimate')
    @classmethod
    def make_optimistic_estimate(cls, increasing:bool=True):
        mult = 1 if increasing else -1

        def score(state:DFSState) -> float:
            nonlocal mult
            return mult * state.optimistic_estimate
        
        return score
    
    @factorymethod(name='hash')
    @classmethod
    def make_hash(cls, salt:str = '', volatile:bool=False):
        
        def score_non_volatile(state:DFSState) -> float:
            nonlocal salt
            return hash((state.selector,salt))
        def score_volatile(state:DFSState) -> float:
            nonlocal salt
            return hash((state.selector,state.depth,tuple(state.blacklist), tuple(state.covered), state.optimistic_estimate, salt))
        
        return score_volatile if volatile else score_non_volatile
    
    get_score = FactoryGetter(member='product', default='value_addition', classmethod=True)
    get = FactoryGetter(default='value_addition', classmethod=True)
    
    description = FactoryDescription()


class SearchStatus(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    COMPLETED = enum.auto()
    ABORTED = enum.auto() 


class LanguageTopKBranchAndBound:
        
    class Result(NamedTuple):
        selector: Selector
        optimistic_estimate: float
        objective_value:float
        
        def __str__(self):
            return fr'(f:{self.objective_value:5.3f}/{self.optimistic_estimate:5.3f}|{self.selector})'

        def __repr__(self):
            return f'<{self.__class__.__name__}{self!s}>'

    def __init__(self, language:ConjunctionLanguage, measure:Measure, optimistic_estimator:OptimisticEstimator, k:int=1, max_best:bool=True, approximation_factor:float=1.) -> None:
        self._language:ConjunctionLanguage = language
        self._measure:Measure = measure
        self._optimistic_estimator:OptimisticEstimator = optimistic_estimator
        self._results: TopKQueue[LanguageTopKBranchAndBound.Result] = TopKQueue(k=k, max_best=max_best)
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
    def results(self) -> TopKQueue['LanguageTopKBranchAndBound.Result']:
        return self._results
    
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
    
    @property
    def approximation_factor(self):
        '''The approximation factor to use while optimising. The result found has at least this big a score times that of the optimum.''' 
        return self._approximation_factor
        
    AddResultOutcome = namedtuple('AddResultOutcome', ('entry_out', 'was_added', 'objective_value'))

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
        result = self.Result(selector=selector, objective_value=objective_value, optimistic_estimate=optimistic_estimate)
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
        
        return LanguageTopKBranchAndBound.AddResultOutcome(entry_out, was_added, objective_value)
    
    def try_add_selectors(self, selectors, quiet=False) -> 'LanguageTopKBranchAndBound.AddResultOutcome':
        make_result = lambda selector: self.Result(selector=selector,
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
    
    def subgroups(self) -> Tuple[ConjunctionSelector, ...]:
        return tuple(element.data.selector for element in self.results.entries())

class DFSVisitor:
    def start(self, dfs, root_state:DFSState):pass
    def stop(self, dfs):pass
    def state_popped(self, dfs, state:DFSState): pass
    def state_expanded(self, dfs, state:DFSState, valid_states, new_states): pass
    def result_added(self, state:DFSState, result_old:'LanguageTopKBranchAndBound.Result'): pass

class DFSResultLogger(DFSVisitor):
    Update = namedtuple('Update',('new','old','statistics','time'))
    class UpdateList(list, SummarisableAsDict):
        def __summary_dict__(self, options:SummaryOptions):
            selector2index = {}
            update_dicts = SummarisableList()
            for update in self:
                if update.new.selector not in selector2index:
                    selector2index[update.new.selector] = len(selector2index)
                state_new = update.new.__summary_dict__(selector_dict=selector2index)
                index_old = -1 if update.old is None else selector2index[update.old.selector]
                update_dict = SummarisableDict(state_new)
                update_dict['index_old'] = index_old
                update_dict['statistics'] = update.statistics
                update_dict['time'] = update.time
                update_dicts.append(update_dict)
            dct = {'selectors': SummarisableList(selector2index.keys()),
                   'updates':update_dicts
                   }
            return dct
    def __init__(self, results=None):
        if results is None:
            results = []
        self._result_history = self.UpdateList(results)
    def result_added(self, state:DFSState, result_old:'LanguageTopKBranchAndBound.Result'):
        update = self.Update(new=state, old=result_old, statistics=state.search.statistics.copy(), time=time.time())
        self._result_history.append(update)
    result_history = property(lambda self:self._result_history, None, 'Results tracked so far.')

class DepthFirstSearch(LanguageTopKBranchAndBound, SummarisableAsDict):
    
    def __init__(self, language: ConjunctionLanguage,
                 measure: Measure, optimistic_estimator: OptimisticEstimator,
                 k:int=1, max_best:bool=True,
                 approximation_factor:float=1., max_depth:float=math.inf,
                 state_scoring=None,refinement_scoring='optimistic_estimate', visitor = DFSVisitor()) -> None:
        
        super().__init__(language=language, measure=measure, optimistic_estimator=optimistic_estimator, k=k, max_best=max_best, approximation_factor=approximation_factor)
        self._stats = DepthFirstSearch.Statistics()
        self._states = None
        self._state_scoring: ProductBundle = ScoringFunctions.get(state_scoring)
        self._refinement_scoring: ProductBundle = ScoringFunctions.get(refinement_scoring)
        self._objective_attainable:float = None
        self._max_depth = max_depth
        self._visitor = visitor
           
    class Statistics(StatisticsBase):
        '''Statistics for the DFS algorithm
        
        :var popped: Times a selector was popped
        :var added: Times a result was added that superseded in fitness the so-far maintained list
        :var created: Times a search state was created.
        :var queued: Times a predicate was queued.
        :var pruned: Times a predicate was pruned during refinement (and propagated to all siblings)
        :var ignored: Times a predicate was ignored immediately after popping (late pruning)
        :var deep: Number of unreachable top-nodes due to depth limit.
        '''
        
        increase_popped= StatisticsUpdater('popped', 1, doc='Times a selector was popped')
        increase_added= StatisticsUpdater('added', 1, doc='Times a result was added that superseded in fitness the so-far maintained list')
        increase_created= StatisticsUpdater('created', 1, doc='Times a search state was created.')
        increase_queued= StatisticsUpdater('queued', 1, doc='Times a predicate was queued.')
        increase_pruned= StatisticsUpdater('pruned', 1, doc='Times a predicate was pruned during refinement (and propagated to all siblings)')
        increase_covered= StatisticsUpdater('covered', 1, doc='Number of covered nodes')
        increase_ignored= StatisticsUpdater('ignored', 1, doc='Times a predicate was ignored immediately after popping (late pruning)')
        increase_deep= StatisticsUpdater('deep', 1, doc='Number of unreachable top-nodes due to depth limit.')
        increase_measures= StatisticsUpdater('measures', 1, doc='Number of measures evaluated')
        increase_oests= StatisticsUpdater('oests', 1, doc='Number of oests evaluated')

        
    @property
    def statistics(self):
        return self._stats
    
    @property
    def state_scoring(self):
        '''the function used to sort the states. Higher values designate earlier popping.'''
        return self._state_scoring

    @property
    def refinement_scoring(self):
        '''the function used to sort the refinements. Lower values designate heavier pruning.'''
        return self._refinement_scoring
    
    @property
    def states(self):
        return self._states
    
    @property
    def reached_max_depth(self):
        '''Whether the maximum depth left some states unvisited'''
        return self._reached_max_depth
    
    @property
    def objective_attainable(self):
        '''The maximum possible score attainable by the (unvisited) deeper states'''
        return self._objective_attainable
    
    @property
    def max_depth(self):
        return self._max_depth
    
    def _make_state(self, selector:Selector, depth: int, pruned:Set[int]=None, covered: Set[int]=None) -> DFSState:
        if pruned is None:
            pruned = set()
        if covered is None:
            covered = set()
        else:  # Ensure that covered is a local copy. Next predicates to be covered should not propagate back
            covered = covered.copy()
        self._stats.increase_created()
        return DFSState(search=self, selector=selector,
                        depth=depth, pruned=pruned, covered=covered)

    def _run(self, root_selector: Selector=None) -> bool:
        # Initialisations
        self._objective_attainable = -math.inf
        self._reached_max_depth = False
        self._stats.reset()
        
        if root_selector is None:
            root_selector = self.language.root
        
        root_state = self._make_state(selector=root_selector, depth=0)
        self._visitor.start(self, root_state)
        self.try_add_state(root_state)
        if self.max_depth == 0:
            self._reached_max_depth = True
        else:
            self._states = [root_state]
            self._reached_max_depth = self.resume()
        self._visitor.stop(self)
        return self._reached_max_depth
    
    def try_add_state(self, state: DFSState, quiet=False) -> 'LanguageTopKBranchAndBound.AddResultOutcome':
        outcome = self.try_add_selector(selector=state.selector,
                                        optimistic_estimate=state.optimistic_estimate,
                                        objective_value=state.objective_value,
                                        quiet=quiet)
        if outcome.was_added:
            self._stats.increase_added()
            result_old = outcome.entry_out.data if outcome.entry_out is not None else None
            self._visitor.result_added(state, result_old=result_old)
        return outcome
    
    def resume(self) -> bool:
        
        # function pre-resolution
        stats = self._stats
        states = self._states
        results = self._results
        refine_selector = self.language.refine
        results_threshold: float = results.threshold
        log_ison = log.ison
        make_state = self._make_state
        approximation_factor = self.approximation_factor
        max_depth = self.max_depth
        visitor: DFSVisitor = self._visitor
        
        measure_property_name = self.measure.selector_property_name()
        oest_property_name = self.optimistic_estimator.selector_property_name()
        def update_evaluations(state):
            if measure_property_name in state.selector.cache:
                stats.increase_measures()
            if oest_property_name in state.selector.cache:
                stats.increase_oests()
            
                
        def create_refinement_states(state):
            nonlocal results_threshold
            refinements = refine_selector(state.selector, blacklist=state.blacklist)
            refinements = tuple(refinements)  # dbg-remove
            pruned:Set[int] = state.blacklist.copy()
            covered:Set[int] = set()
            
            new_states = (make_state(selector=refinement, depth=state.depth+1,pruned=pruned) for refinement in refinements)
            new_states = tuple(new_states) # dbg-remove
            
            new_states_srt_all = sorted(new_states, key=self.refinement_scoring.product)[::-1]
            
            new_states_srt: List[DFSState] = []
            for new_state in new_states_srt_all:
                selector:ConjunctionSelector = new_state.selector
                last_predicate: int = selector.index_last_extension
                new_state = new_state._replace(covered=covered.copy())
                covered.add(last_predicate)  # blacklist only for future siblings
                if set(new_state.blacklist) & set(new_state.selector.indices):
                    stats.increase_covered()
                    update_evaluations(new_state)
                else:
                    new_states_srt.append(new_state)
            
            valid_states: List[DFSState] = []
            for new_state in new_states_srt:
                selector:ConjunctionSelector = new_state.selector
                oest:float = new_state.optimistic_estimate
                last_predicate: int = selector.index_last_extension
                if oest * approximation_factor <= results_threshold:  # discard and blacklist
                    pruned.add(last_predicate)  # blacklist this predicate from all siblings
                    stats.increase_pruned()
                else:
                    stats.increase_queued()
                    outcome = self.try_add_state(new_state)
                    if outcome.was_added:
                        results_threshold = results.threshold
                    valid_states.append(new_state)
                update_evaluations(new_state)
                    
            if log_ison.progress:
                state_cnt = Counter(s.depth for s in states)
                state_txt = ' '.join(f'{i}:{v:4}' for i, v in state_cnt.items())
                log.progress(log.rlim.progress and f'Queue size: {len(states):6d}[{state_txt}]: Kept {len(valid_states)}/{len(new_states_srt)}/{len(new_states_srt_all)} while refining {state.selector!s}[d={state.depth}]. Full state: {state!s}')
            #if log.ison.debug:
                #valid_selectors = set(s.selector for s in valid_states)
                #state_txt = ','.join(map(lambda s:f'{"V" if s.selector in valid_selectors else "P"}{s}', new_states_srt))
                #log.debug(f'Current states: {len(states):3d}. Kept {len(valid_states)}/{len(new_states)} while refining {state!s}. These are: {state_txt}')
            visitor.state_expanded(self, state, valid_states, new_states_srt)
            return valid_states
        
        
        stop = False  # dbg-remove
        while states:
            state: DFSState = states.pop()
            visitor.state_popped(self, state)
            stats.increase_popped()
            
            if state.optimistic_estimate * approximation_factor <= results_threshold:  # discard
                stats.increase_ignored()
                continue
            else:  # refine
                if stop:  # dbg-remove
                    raise KeyboardInterrupt()  # dbg-remove
                
                new_states = create_refinement_states(state)
                if new_states and state.depth + 1 < max_depth:
                    new_states_srt = sorted(new_states, key=self.state_scoring.product)
                    states += new_states_srt[::-1]
                else:
                    if new_states:
                        self._reached_max_depth = True
                        stats.increase_deep(len(new_states))
                        max_attainable = max(state.optimistic_estimate for state in new_states)
                        self._objective_attainable = max(self._objective_attainable, max_attainable)
        return self._reached_max_depth

    def __summary_dict__(self, options: SummaryOptions):
        fields = ('statistics', 'time_elapsed', 'time_started', 'status', 'reached_max_depth', 'max_depth', 'state_scoring', 'refinement_scoring', 'objective_attainable','k')
        dct = summary_from_fields(self,fields)
        dct['subgroups'] = SummarisableList(self.subgroups())
        dct['status'] = str(dct['status'])
        return dct
    
    summary_name = 'depth-first-search'


DepthSpec = Union[float, int, Iterable[int]]


class IterativeDeepening(LanguageTopKBranchAndBound, SummarisableAsDict):
    
    def __init__(self, language: ConjunctionLanguage,
                 measure: Measure, optimistic_estimator: OptimisticEstimator, k:int=1, max_best:bool=True,
                 approximation_factor:float=1.,depths:DepthSpec=math.inf,
                 state_scoring=None, refinement_scoring='optimistic_estimate',
                 dfs=DepthFirstSearch) -> None:
        
        super().__init__(language=language, measure=measure, optimistic_estimator=optimistic_estimator, k=k, max_best=max_best, approximation_factor=approximation_factor)
        
        self._depths = depths
        self._dfs = dfs
        
        self._dfs_runs:List[DepthFirstSearch] = []
        self._state_scoring: ProductBundle = ScoringFunctions.get(state_scoring)
        self._refinement_scoring: ProductBundle = ScoringFunctions.get(refinement_scoring)
        
    @classmethod
    def _get_depths(cls, depths: DepthSpec) -> Iterable[int]:
        '''Parse depth parameter to an integer iterable'''
        depth_iter: Iterable[int]
        if isinstance(depths, float):
            if math.isinf(depths) and depths > 0:
                depth_iter = itertools.count()
            elif int(depths) == depths:
                depth_iter = cls._get_depths(int(depths))
            else:
                raise TypeError(f'Only integral values (or inf) are allowed for scalar depths, and {depths} is not one.')
        elif isinstance(depths, int):
            depth_iter = range(depths + 1)
        elif isinstance(depths, Iterable):
            depth_iter = depths
        else:
            raise TypeError(f'Specified depths {depths} must be a sequence, int or inf')
        return depth_iter

    @property
    def statistics(self):
        if self._dfs_runs is None:
            return DepthFirstSearch.Statistics()
        else:
            stats = functools.reduce(operator.add, (dfs.statistics for dfs in self._dfs_runs))
            return stats
    
    @property
    def states(self):
        if self._dfs_runs is not None and self._dfs_runs:
            return self._dfs_runs[-1].states
        return None

    @property
    def depths(self):
        return self._depths
     
    @property
    def state_scoring(self):
        '''the function used to sort the states. Higher values designate earlier popping.'''
        return self._state_scoring
    
    @property
    def refinement_scoring(self):
        '''the function used to sort the refinements. Lower values designate heavier pruning.'''
        return self._refinement_scoring
    
    def _run(self) -> Tuple[Entry, ...]:
        depth_iter = self._get_depths(self.depths)
        for depth in depth_iter:
            dfs = self._dfs(language=self.language, measure=self.measure, optimistic_estimator=self.optimistic_estimator,
                            k=self.k, approximation_factor=self.approximation_factor, max_depth=depth,
                            state_scoring=self._state_scoring, refinement_scoring=self._refinement_scoring)
            self._dfs_runs.append(dfs)
            
            dfs.try_add_results(self.results.elements(sort=False), quiet=True)
            dfs.run()

            self.try_add_results(dfs.results.elements(sort=False), quiet=True)
            log.info(f'ID: finished depth {depth} with stats: {dfs.statistics}')
            if not dfs.reached_max_depth:
                break
            depth += 1
        return self._results.entries()
        
    def __summary_dict__(self, options: SummaryOptions):
        fields = ('statistics', 'state_scoring', 'refinement_scoring', 'approximation_factor', 'time_elapsed', 'time_started', 'status')
        dct = summary_from_fields(self, fields)
        dct['steps'] = SummarisableList(self._dfs_runs)
        dct['subgroups'] = SummarisableList(self.subgroups())
        dct['status'] = str(dct['status'])
        dct['depths'] = str(self.depths)
        return dct
