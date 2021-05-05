'''
Created on May 3, 2021

@author: janis
'''

from typing import NamedTuple, Iterable, Tuple, Set, List, Union

import math
import itertools
import operator
import functools
import numpy as np

from sergio.language import Selector, ConjunctionLanguage, ConjunctionSelector
from sergio.scores import Measure, OptimisticEstimator
from .base import SubgroupSearch, SearchVisitor, SearchState, AddResultOutcome
from .utils import ScoringFunctions

from colito.summaries import summary_from_fields, SummaryOptions, SummarisableList, SummarisableAsDict,\
    SummarisableFromFields, SummaryFieldsAppend
from colito.queues import Entry
from colito.factory import ProductBundle
from colito.statistics import StatisticsBase, StatisticsUpdater
from collections import Counter

from colito.logging import getModuleLogger

log = getModuleLogger(__name__)

class DepthFirstSearch(SubgroupSearch, SummarisableFromFields):
    __collection_tag__ = 'depth-first'

    __summary_fields__ = SummaryFieldsAppend(('statistics', 'reached_max_depth', 'max_depth', 'state_scoring', 'objective_attainable', 'k'))
    
    def __init__(self, language: ConjunctionLanguage,
                 measure: Measure, optimistic_estimator: OptimisticEstimator,
                 k:int=1, max_best:bool=True,
                 approximation_factor:float=1., max_depth:float=math.inf,
                 state_scoring='optimistic_estimate', visitor = SearchVisitor()) -> None:
        
        super().__init__(language=language, measure=measure, optimistic_estimator=optimistic_estimator, k=k, max_best=max_best, approximation_factor=approximation_factor)
        self._stats = DepthFirstSearch.Statistics()
        self._states = None
        self._state_scoring: ProductBundle = ScoringFunctions.get(state_scoring)
        self._objective_attainable:float = None
        self._max_depth = max_depth
        self._reached_max_depth = None
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
    def states(self):
        return self._states
    
    @property
    def reached_max_depth(self):
        '''Whether the maximum depth left some states unvisited'''
        return self._reached_max_depth
    
    @property
    def objective_attainable(self):
        '''The best possible attainable by the unvisited states'''
        if self._states is None:
            att = None
        else:
            oest_vals = [state.optimistic_estimate for state in self._states]
            if self._results.max_best:
                att = np.max(oest_vals, initial='-inf')
            else:
                att = np.min(oest_vals, initial='inf')
        return att
    
    @property
    def max_depth(self):
        return self._max_depth
    
    def _make_state(self, selector:Selector, depth: int, pruned:Set[int]=None, covered: Set[int]=None) -> SearchState:
        if pruned is None:
            pruned = set()
        if covered is None:
            covered = set()
        else:  # Ensure that covered is a local copy. Next predicates to be covered should not propagate back
            covered = covered.copy()
        self._stats.increase_created()
        return SearchState(search=self, selector=selector,
                           depth=depth, pruned=pruned, covered=covered)

    def _run(self, root_selector: Selector=None) -> bool:
        # Initialisations
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
    
    def try_add_state(self, state: SearchState, quiet=False) -> AddResultOutcome:
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
        visitor: SearchState = self._visitor
        
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
            #covered:Set[int] = set()
            
            new_states = tuple(make_state(selector=refinement, depth=state.depth+1,pruned=pruned) for refinement in refinements)
            
            valid_states: List[SearchState] = []
            for new_state in new_states:
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
                log.progress(log.rlim.progress and f'Queue size: {len(states):6d}[{state_txt}]: Kept {len(valid_states)}/{len(new_states)} while refining {state.selector!s}[d={state.depth}]. Full state: {state!s}')
            #if log.ison.debug:
                #valid_selectors = set(s.selector for s in valid_states)
                #state_txt = ','.join(map(lambda s:f'{"V" if s.selector in valid_selectors else "P"}{s}', new_states_srt))
                #log.debug(f'Current states: {len(states):3d}. Kept {len(valid_states)}/{len(new_states)} while refining {state!s}. These are: {state_txt}')
            visitor.state_expanded(self, state, valid_states, new_states)
            return valid_states
        
        
        stop = False  # dbg-remove
        while states:
            state: SearchState = states.pop()
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
        return self._reached_max_depth

    summary_name = 'depth-first-search'


DepthSpec = Union[float, int, Iterable[int]]


class IterativeDeepening(SubgroupSearch, SummarisableFromFields):
    __collection_tag__ = 'iterative-deepening'
    __summary_fields__ = SummaryFieldsAppend(('statistics','state_scoring','steps','depths'))
    __summary_conversions__ = {'depths':str,'steps':SummarisableList}
    
    def __init__(self, language: ConjunctionLanguage,
                 measure: Measure, optimistic_estimator: OptimisticEstimator, k:int=1, max_best:bool=True,
                 approximation_factor:float=1.,depths:DepthSpec=math.inf,
                 state_scoring='optimistic_estimate',
                 dfs=DepthFirstSearch) -> None:
        
        super().__init__(language=language, measure=measure, optimistic_estimator=optimistic_estimator, k=k, max_best=max_best, approximation_factor=approximation_factor)
        
        self._depths = depths
        self._dfs = dfs
        
        self._dfs_runs:List[DepthFirstSearch] = []
        self._state_scoring: ProductBundle = ScoringFunctions.get(state_scoring)
        
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
    def steps(self):
        '''The DFS object from each run performed'''
        return self._dfs_runs
    @property
    def depths(self):
        return self._depths
     
    @property
    def state_scoring(self):
        '''the function used to sort the states. Higher values designate earlier popping.'''
        return self._state_scoring
    
    def _run(self) -> Tuple[Entry, ...]:
        depth_iter = self._get_depths(self.depths)
        for depth in depth_iter:
            dfs = self._dfs(language=self.language, measure=self.measure, optimistic_estimator=self.optimistic_estimator,
                            k=self.k, approximation_factor=self.approximation_factor, max_depth=depth,
                            state_scoring=self._state_scoring)
            self._dfs_runs.append(dfs)
            
            dfs.try_add_results(self.results.elements(sort=False), quiet=True)
            dfs.run()

            self.try_add_results(dfs.results.elements(sort=False), quiet=True)
            log.info(f'ID: finished depth {depth} with stats: {dfs.statistics}')
            if not dfs.reached_max_depth:
                break
            depth += 1
        return self._results.entries()
        
    def ___summary_dict__(self, options: SummaryOptions):
        dct_base = super().__summary_dict__(options)
        fields = ('statistics', 'state_scoring')
        dct = summary_from_fields(self, fields)
        dct['steps'] = SummarisableList(self._dfs_runs)
        dct['subgroups'] = SummarisableList(self.subgroups())
        dct['status'] = str(dct['status'])
        dct['depths'] = str(self.depths)
        return {**dct_base, **dct}
