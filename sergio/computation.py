'''
Created on May 13, 2018

@author: janis
'''

from datetime import datetime
from typing import Dict, Any, Iterable
from pandas import DataFrame

from colito.logging import getModuleLogger, to_stuple
from colito.summaries import Summarisable, SummaryOptions, SummarisableList
from colito.runtime import RuntimeEnvironment
from colito.factory import DEFAULT_TYPE_RESOLVER

from sergio.language import LANGUAGES, Language
from sergio.scores import Measure, OptimisticEstimator
from sergio.searches import IterativeDeepening, DFSResultLogger, DepthFirstSearch

from sergio.discretisers import FrequencyDiscretiser
from sergio.predicates import Prediciser
from sergio import FileManager
from sergio.data.factory import DatasetFactory
from colito.cache import FileCache, DEFAULT_FILE_CACHE

#from .control import Controller


log = getModuleLogger(__name__) #pylint: disable=invalid-name

def rename_columns(df, reg, txt,regexp=False, inplace=False):
    if regexp:
        import re
        rex = re.compile(reg)
    else:
        rex = reg
    columns = df.columns.str.replace(rex, txt)
    if inplace:
        df.columns = columns
    else:
        df_new = df.copy()
        df_new.columns = columns
        return df_new
DataFrame.rename_columns = rename_columns
#TODO: Monkey patching is UGLY. Move to a better place.

class Computation(Summarisable):
    SCORE_CONFIGS = ['AverageCoreness']

    def __init__(self, tag=None, log=log, file_manager = None, cache = DEFAULT_FILE_CACHE):
        self._file_manager: FileManager = file_manager if file_manager is not None else FileManager()
        self._cache: FileCache = cache
        self._runtime_environment = RuntimeEnvironment()
        self._dataset = None
        self._prediciser = None
        self._language = None
        self._search = None
        self._oest = None
        self._measure = None
        self._tag = tag if tag is not None else str(datetime.timestamp(datetime.now()))
        self._subgroups = []
        self._log = log
        self._resume = None
        self._result_history = None
#        self._controller = Controller(self)

    @property
    def cache(self):
        '''Caches computaion intermediaries to files'''
        return self._cache
    @property
    def file_manager(self):
        '''Offers file path abstraction'''
        return self._file_manager
    @property
    def runtime_environment(self):
        return self._runtime_environment
    
    @property
    def dataset_name(self):
        return None if self.dataset is None else self.dataset.name
    
    @property
    def dataset(self):
        return self._dataset
    @dataset.setter
    def dataset(self, value): self.load_dataset(value)

    @property
    def prediciser(self) -> Prediciser:
        return self._prediciser
    
    @property
    def language(self) -> Language:
        return self._language
    @language.setter
    def language(self, name):
        self.load_language(name)

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, value):
        self._tag = value

    @property
    def search(self):
        '''The search algorithm used'''
        return self._search
    
    @property
    def subgroups(self):
        return self._subgroups
    
    @property
    def measure(self):
        return self._measure
    
    @property
    def optimistic_estimator(self):
        return self._oest
    
    @property
    def resume(self):
        return self._resume
    
    controller = property(lambda self:self._controller, None, 'The controller this experiment is using, if any.')
    
    result_history = property(lambda self:self._result_history,None,
                              'Get the result history. Available if optimise is invoked with track_results set to True.')

    def load_dataset(self, name, *args, **kwargs):
        """
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> c.dataset = 'moondot'
        >>> c.dataset
        <EntityAttributesWithArrayTarget[moondot](73x4/4) target: position(2d float64)>
        >>> c.dataset = 'moondot:CATEGORICAL'
        <EntityAttributesWithTarget[moondot](73x4/4) target: location(2d float)>
        """
        df = DatasetFactory(cache = self.cache, file_manager = self.file_manager)
        dataset = df.load_dataset(name, *args, **kwargs)
        self._dataset = dataset
        
    def load_prediciser(self, cuts=5, ranges='SLABS_POSITIVE', negate='BOOLEAN|CATEGORICAL|RANGED'):
        """
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> _ = c.load_prediciser(4)
        """
        discretiser = FrequencyDiscretiser(cut_count=cuts, ranges=ranges)
        self._prediciser = Prediciser(discretiser = discretiser, negate=negate)
        self._log.info(f'Loaded prediciser {self._prediciser}.')
        return self
        
    def load_language(self, name='closure-conjunctions-restricted'):
        """
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> c.dataset = 'moondot'
        >>> _ = c.load_prediciser()
        >>> c.language = 'conjunctions'
        >>> tuple(map(str,c.language.refine(c.language.root)))
        ('{a}', '{!a}', '{b}', '{!b}', '{c}', '{!c}', '{main}', '{!main}')
        """
        language_cls = LANGUAGES.get_class_from_tag(name)
        if self.dataset is None:
            raise ValueError('No dataset is set; languages need a dataset in order to be loaded.')
        predicates = tuple(self.dataset.make_predicates(self.prediciser))
        self._language = language_cls(self.dataset, predicates)
        self._log.info(f'Loaded language {name} {self.language} ({len(self.language.predicates)} predicates)')
        return self
    
    def load_measure(self, name, **kwargs):
        """
        >>> import sergio.scores.scalars
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> c.dataset = 'moondot'
        >>> _ = c.load_prediciser().load_language().load_measure('coverage')
        >>> c.measure
        <MeasureCoverage()>
        """
        
        self._log.info(f'Loading measure {name}.')
        if self.dataset is None:
            raise ValueError('No dataset is set; scores need a dataset in order to be loaded.')
        meas = Measure.make_from_strings(name, self.dataset, **kwargs)
        #oest = self.optimistic_estimator
        #if isinstance(meas, CallbackSubgraphEvaluator) and oest is not None and isinstance(oest, CallbackSubgraphEvaluator):  # Couple the two
        #    self._log.info(f'Coupling measure {meas} with optimistic estimator {oest}.')
        #    meas.couple_evaluator(oest)
        self._measure = meas
        self._log.info(f'Set measure to {meas}.')
        return self

    def load_optimistic_estimator(self, name, **kwargs):
        """
        >>> import sergio.scores.scalars
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> c.dataset = 'moondot'
        >>> _ = c.load_prediciser().load_language().load_measure('jaccard').load_optimistic_estimator('jaccard')
        >>> c.measure
        <MeasureCoverage()>
        """
        self._log.info(f'Loading optimistic estimator {name}.')
        if self.dataset is None:
            raise ValueError('No dataset is set; scores need a dataset in order to be loaded.')
        oest = OptimisticEstimator.make_from_strings(name, self.dataset, **kwargs)
        meas = self.measure
        #if isinstance(oest, CallbackSubgraphEvaluator) and meas is not None and isinstance(meas, CallbackSubgraphEvaluator):  # Couple the two
        #    self._log.info(f'Coupling optimistic estimator {oest} with measure {meas}.')
        #    oest.couple_evaluator(meas)
        self._oest = oest
        self._log.info(f'Set optimistic estimator to {oest}.')
        return self
    
    def load_subgroups(self, strings, append=False):
        parser = self.language.make_parser()
        subgroups = tuple(map(parser.parse, strings))
        if append:
            self._subgroups += subgroups
        else:
            self._subgroups = subgroups
        self._log.info(f'Loaded {len(subgroups)} subgroups. Current list: {to_stuple(self.subgroups,join=",")}.')
        return self
    
    def load_attributes(self, attributes:DataFrame, kinds='auto', selected=True):
        """ Append attributes to the dataset.
        
        @param kinds: The kind of each added attribute. 
            Can be either a single string or AttributeKind, in which case it applies to all, or a list of them.
            The string 'auto' can also be provided.  
        @param selected: Whether the added attributes should be used for predicisation.
            A single string or bool, which applies to all of them, or a list of them. 
        """
        def parse_bool(what):
            if isinstance(selected, str):
                value = DEFAULT_TYPE_RESOLVER.resolve(bool, what)
            elif isinstance(what, bool):
                value = selected
            else:
                value = bool(what)
            return value
        if isinstance(selected, str):
            parsed_selected = parse_bool(selected)
        elif isinstance(selected, Iterable):
            parsed_selected = tuple(map(parse_bool, selected))
        else:
            raise TypeError(f'Could not parse a boolean or list of bools from {selected}.')
        attributes, kinds, selected = self.dataset.add_attributes(attributes=attributes, kinds=kinds, selected=parsed_selected)
        log.info(f'Added {attributes.shape[0]}x{attributes.shape[1]}, kinds: {kinds} and selected: {selected}.')
        log.info(f'Updated dataset: {self.dataset}')
        
        return self
        
    
    def optimise(self, k, depths=6,approximation_factor=1, state_scoring=None,refinement_scoring=None, track_results=False):
        if track_results:
            log.info('Attaching results tracker visitor.')
            dfs_visitor = DFSResultLogger(self._result_history)
            self._result_history = dfs_visitor.result_history
            def dfs(*args, **kwargs):
                kwargs['visitor'] = dfs_visitor
                return DepthFirstSearch(*args,**kwargs)
        else:
            dfs = DepthFirstSearch 
        
        iddfs = IterativeDeepening(self._language, self._measure, self._oest,
                                   k=k, depths=depths,
                                   approximation_factor=approximation_factor,
                                   state_scoring=state_scoring, refinement_scoring=refinement_scoring,
                                   dfs=dfs
                                   )
        self._log.info(f'Starting Optimisation algorithm for k={k}, depths={depths}, language={self._language}, measure={self._measure}, oest={self._oest}.')
        iddfs.try_add_selectors(self._subgroups)
        aborted = False
        try:
            iddfs.run() 
        except KeyboardInterrupt as _:
            self._resume = iddfs.states
            aborted = True
        subgroups = iddfs.subgroups()
        self._log.info(f'Optimisation {"aborted" if aborted else "completed"} with results {to_stuple(subgroups,join=",")}.')
        self._search = iddfs
        self._subgroups = subgroups
        return self

    def summary_dict(self, options:SummaryOptions):
        summary: Dict[str, Any] = {
            'tag':self.tag,
            'runtime':self._runtime_environment,
            'dataset':self._dataset,
            'language':self._language,
            'measure':self.measure,
            'optimistic_estimator':self.optimistic_estimator,
            'search':self._search,
            'subgroups': SummarisableList(self._subgroups)
            }
        if self.result_history is not None:
            summary['result_history'] = self.result_history
        return summary
    
#     def summarise_to_json(self, indent=4, separators=(',', ': '), parts=SummaryParts.BASIC):
#         from colito.summaries import Summariser
#         if log.ison.progress:
#             results_text = 'results' if self.result_history is None else f'actual and {len(self.result_history)} historical results'
#             log.progress(f'Computing experiment summary with {len(self.subgroups)} {results_text}.')
#         summariser = Summariser()
#         #summariser.add_filter(SelectorSummariser())
#         #summariser.add_filter(ScoringfunctionSummariser())
#         summary_parts = SummaryParts.get_parts(parts)
#         summary = summariser.summarise_asdict(self, parts = summary_parts)
#         jsonstr = Summariser.to_json(summary, indent=indent, separators=separators)
#         return jsonstr
#     
#     def subgroups_to_json(self, indent=4, separators=(',', ': '), parts=SummaryParts.BASIC):
#         import json
#         summary_dict = self.subgroups_to_dict(parts=parts)
#         jsonstr = json.dumps(summary_dict, indent=indent, separators=separators)
#         return jsonstr

    def subgroups_to_dataframe(self, parts) -> DataFrame:
        from pandas.io.json import json_normalize
        subgroups = self.subgroups_to_dict(parts=parts)
        df = json_normalize(subgroups)
        df = df.drop('name',axis=1)\
            .rename_columns('records\.','')\
            .rename_columns('cached\.(?:Measure|OptimisticEstimator)(.*)_(measure|optimistic_estimator)',r'\2_\1_cached',regexp=True)\
            .rename_columns('(measure|optimistic_estimator)s',r'\1',regexp=True)\
            .rename_columns('([A-Z])([a-z0-9]+)',lambda m:f'_{m[1].lower()}{m[2]}',regexp=True)
        return df

#     def subgroups_to_csv(self, separator=None, float_format='%.5f', header=True, index=False, parts=SummaryParts.BASIC, **kwargs):
#         df = self.subgroups_to_dataframe(parts=parts)
#         result = df.to_csv(sep=separator, float_format=float_format, index=index, header=header, **kwargs)
#         return result
        
    def __repr__(self):
        return f'<{self.__class__.__name__}[{self.tag}] D:{self.dataset}, L:{self.language}, SG:{len(self.subgroups)}, M:{self.measure}, O:{self.optimistic_estimator}>'
    
if __name__ == "__main__":
    import doctest
    doctest.testmod() 
    