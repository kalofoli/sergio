'''
Created on May 3, 2021

@author: janis
'''

if False:
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
    
