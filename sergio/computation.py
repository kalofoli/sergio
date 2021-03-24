'''
Created on May 13, 2018

@author: janis
'''

#import warnings
#warnings.filterwarnings("error")

from datetime import datetime
from typing import Dict, Any, Iterable
from pandas import DataFrame

from colito.logging import getModuleLogger, to_stuple
from colito.summaries import Summarisable, SummaryOptions, SummarisableList,\
    SummarisableFromFields
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
from sergio.kernels import Kernel
from sergio.kernels.gramian import GramianFromDataset

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

class Computation(SummarisableFromFields):
    SCORE_CONFIGS = ['AverageCoreness']
    __summary_fileds__ = ('tag', 'runtime', 'dataset', 'language', 'measure',
                          'optimistic_estimator', 'search', 'subgroups')
    
    __summary_convert__ = {'subgroups': SummarisableList}

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
        self._gramian = None
        self._kernel = None
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
    def kernel(self):
        return self._kernel
    
    @property
    def gramian(self):
        return self._gramian

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
        >>> c.load_dataset('moondot:CATEGORICAL').dataset
        <EntityAttributesWithArrayTarget[moondot](73x1/1) target: position(2d float64)>
        >>> c.load_dataset('toy-array:circledots').dataset
        <EntityAttributesWithArrayTarget[circledots](147x3/3) target: position(2d float64)>
        >>> c.load_dataset('toy-scalar:circledots').dataset
        <EntityAttributesWithAttributeTarget[circledots](147x3/4) target: position_norm2(float64@3)>
        >>> c.load_dataset('toy-scalar:circledots,NORM,param=inf').dataset
        <EntityAttributesWithAttributeTarget[circledots](147x3/4) target: position_norminf(float64@3)>
        >>> c.load_dataset('toy-scalar:circledots,param=outer,scalarify=ATTR').dataset
        <EntityAttributesWithAttributeTarget[circledots](147x2/3) target: outer(bool@2)>
        """
        df = DatasetFactory(cache = self.cache, file_manager = self.file_manager)
        dataset = df.load_dataset(name, *args, **kwargs)
        self._dataset = dataset
        return self
        
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
        ('{main}', '{!main}', '{a}', '{!a}', '{b}', '{!b}', '{c}', '{!c}')
        """
        language_cls = LANGUAGES.get_class_from_tag(name)
        if self.dataset is None:
            raise ValueError('No dataset is set; languages need a dataset in order to be loaded.')
        predicates = tuple(self.dataset.make_predicates(self.prediciser))
        self._language = language_cls(self.dataset, predicates)
        self._log.info(f'Loaded language {name} {self.language} ({len(self.language.predicates)} predicates)')
        return self
    
    def _require(self, attribute, loadable):
        '''Complain if not set'''
        try:
            val = getattr(self, attribute)
        except AttributeError:
            raise ValueError(f'Loading a {loadable} requires an attribute {attribute} which is not available in {self.__class__}.')
        if val is None:
            raise ValueError(f'Loading a {loadable} requires {attribute} to be set.')
        return val
    
    def load_measure(self, name, **kwargs):
        """
        >>> import sergio.scores.scalars
        >>> c = Computation(cache=None, file_manager=FileManager('datasets')).load_dataset('toy-scalar:circledots,ATTR,outer').load_prediciser().load_language()
        >>> c.load_measure('coverage').measure
        <MeasureCoverage()>
        >>> c.load_measure('jaccard').measure
        <MeasureJaccard(target_name='outer')>
        """
        
        self._log.info(f'Loading measure {name}.')
        if self.dataset is None:
            raise ValueError('No dataset is set; scores need a dataset in order to be loaded.')
        
        cls = Measure.__collection_factory__.tags[name]
        if hasattr(cls, 'from_dataset'):
            meas = cls.from_dataset(self.dataset, **kwargs)
        else:
            resolver = lambda cls, kwarg, info: self._require(kwarg, 'measure')
            meas = Measure.make_from_string_parts(name, kwargs=kwargs, kwarg_resolver=resolver)
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
        >>> c = Computation(cache=None, file_manager=FileManager('datasets')).load_dataset('toy-scalar:circledots,ATTR,outer').load_prediciser().load_language()
        >>> c.load_optimistic_estimator('jaccard').optimistic_estimator
        <OptimisticEstimatorJaccard(target_name='outer')>
        """
        self._log.info(f'Loading optimistic estimator {name}.')
        if self.dataset is None:
            raise ValueError('No dataset is set; scores need a dataset in order to be loaded.')
        cls = OptimisticEstimator.__collection_factory__.tags[name]
        if hasattr(cls, 'from_dataset'):
            oest = cls.from_dataset(self.dataset, **kwargs)
        else:
            resolver = lambda cls, kwarg, info: self._require(kwarg, 'optimistic estimator')
            oest = Measure.make_from_string_parts(name, kwargs=kwargs, kwarg_resolver=resolver)
        #meas = self.measure
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
    
    def load_kernel(self, name, **kwargs):
        """ Append attributes to the dataset.
        >>> import sergio.kernels.euclidean
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> c.load_kernel('rbf').kernel
        <RadialBasisFunctionKernel(sigma=1,kind=<Kind.EXP_QUAD: 1>)>
        """
        log.info(f'Loading kernel {name}.')
        kern = Kernel.make_from_strings(name, **kwargs)
        self._kernel = kern 
        log.info(f'Set kernel to {kern}')
        return self
        
    def compute_kernel(self, store:bool = True):
        '''
        >>> import sergio.scores.scalars
        >>> import sergio.kernels.euclidean
        >>> c = Computation(cache=None, tag='doctest', file_manager=FileManager('datasets'))\\
        ...     .load_dataset('toy-array:circledots').load_kernel('rbf')
        >>> G = c.compute_kernel().gramian
        >>> G.eigenvals.shape[0], G.dimension, G.rank
        (147, 147, 147)
        >>> 
        '''
        self._gramian = GramianFromDataset(self.dataset, self.kernel)
        return self

    
    def optimise(self, k=1, depths=6,approximation_factor=1, state_scoring=None,refinement_scoring=None, track_results=False):
        r'''
        >>> import sergio.scores.scalars
        >>> c = Computation(cache=None, tag='doctest', file_manager=FileManager('datasets'))\
        ...     .load_dataset('toy-scalar:circledots,ATTR,outer').load_prediciser().load_language()\
        ...     .load_measure('jaccard').load_optimistic_estimator('jaccard')
        >>> c
        <Computation[doctest] D:<EntityAttributesWithAttributeTarget[circledots](147x2/3) target: outer(bool@2)>, L:<ClosureConjunctionLanguageRestricted: of 10 predicates>, SG:0, M:<MeasureJaccard(target_name='outer')>, O:<OptimisticEstimatorJaccard(target_name='outer')>>
        >>> _ = c.optimise()
        >>> ','.join(map(str,c.subgroups))
        '{[label!=dot]^[label!=spread]}'
        '''
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

    def __summary_dict__(self, options:SummaryOptions):
        summary = super().__summary_dict__(options)
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
    