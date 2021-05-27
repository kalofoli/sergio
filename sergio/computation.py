'''
Created on May 13, 2018

@author: janis
'''

#import warnings
#warnings.filterwarnings("error")

from datetime import datetime
from typing import Dict, Any, Iterable
from pandas import DataFrame
import numpy as np

from colito.logging import getModuleLogger, to_stuple
from colito.summaries import SummaryOptions, SummarisableList, SummarisableFromFields,\
    JSONConvertingVisitor
from colito.runtime import RuntimeEnvironment
from colito.factory import DEFAULT_TYPE_RESOLVER

from sergio.language import Language
from sergio.scores import Measure, OptimisticEstimator
from sergio.searches import SearchResultLogger
from sergio.searches.subgroup.dfs import IterativeDeepening, DepthFirstSearch

from sergio.discretisers import FrequencyDiscretiser
from sergio.predicates import Prediciser
from sergio import FileManager
from sergio.data.factory import DatasetFactory
from colito.cache import DEFAULT_FILE_CACHE, PersistentCache
from sergio.kernels import Kernel
from sergio.kernels.gramian import GramianFromDataset, Gramian
from sergio.data.bundles.entities import EntityAttributes
from colito import NamedUniqueConstant
from collections import namedtuple
from sergio.summaries import SelectorSummariser
from colito.cloning import Cloneable
from sergio.language.base import Selector

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

'## Decorators'
def default_setter(prop, cls, name=None, allow_none=True):
    name = f'{prop.fget.__name__}' if name is None else name
    @prop.setter
    def prop_setter(self, what):
        if (allow_none and what is None) or isinstance(what, cls):
            setattr(self, '_'+name, what)
        else:
            raise TypeError(f'Could not set property {name!r} from {what} of type {type(what).__name__} which is not of type {cls.__name__}.')
    return prop_setter

def class_setter(prop, cls, accepted_classes, name=None, store_retval=True, allow_none=True):
    name = f'{prop.fget.__name__}' if name is None else name
    accepted_classes = tuple(accepted_classes)
    def decorator(fn):
        def prop_setter(self, what):
            if isinstance(what, accepted_classes):
                val = fn(self, what)
                if store_retval:
                    setattr(self, '_'+name, val)
            elif (allow_none and what is None) or isinstance(what, cls):
                setattr(self, '_'+name, what)
            else:
                scls = ','.join([c.__name__ for c in accepted_classes])
                raise TypeError((f'Could not set property {name!r} from {what} of type {type(what).__name__}. '
                                 f'Either specify an object of type {cls.__name__} or any of [{scls}] which will be converted into the former.'))
        prop_setter.__name__ = name
        return prop.setter(prop_setter)
    return decorator


OptimisationResult = namedtuple('OptimisationResult' ,('result_history', 'subgroups', 'search'))
class Computation(SummarisableFromFields, Cloneable):
    SCORE_CONFIGS = ['AverageCoreness']
    __summary_fields__ = ('tag', 'runtime_environment', 'dataset', 'language', 'measure',
                          'optimistic_estimator', 'search', 'subgroups','removed_subgroups')
    
    __summary_conversions__ = {'subgroups': SummarisableList, 'removed_subgroups': SummarisableList}

    def __init__(self, tag=None, log=log, file_manager = None, cache = DEFAULT_FILE_CACHE,
                 prediciser = None, dataset = None, language = None, oest = None, measure = None, gramian = None, removed_subgroups=[]):
        self._file_manager: FileManager = file_manager if file_manager is not None else FileManager()
        self._cache: PersistentCache = cache
        self._runtime_environment = RuntimeEnvironment()
        self._dataset = None
        self.dataset = dataset
        self._prediciser = None
        self.prediciser = prediciser
        self._language = None
        self.language = language
        self._oest = None
        self.oest = oest
        self._measure = None
        self.measure = measure
        self._gramian = None
        self.gramian = gramian
        self._removed_subgroups = []
        self.removed_subgroups = removed_subgroups
        self._kernel = None
        self._tag = tag if tag is not None else str(datetime.timestamp(datetime.now()))
        self._result = OptimisationResult(None, None, None)
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
    @class_setter(dataset, cls=EntityAttributes, accepted_classes=[str], store_retval=False)
    def dataset(self, value):  # pylint: disable=function-redefined
        self.load_dataset(value)

    @property
    def prediciser(self) -> Prediciser:
        return self._prediciser
    @class_setter(prediciser, cls=Prediciser, accepted_classes=[str], store_retval=False)
    def prediciser(self, what): # pylint: disable=function-redefined
        if not what: 
            self.load_prediciser()
        else:
            cuts, ranges, negate = what.split(':')
            cuts = int(cuts)
            self.load_prediciser(cuts, ranges, negate)
    
    
    @property
    def language(self) -> Language:
        return self._language
    @class_setter(language, cls=Language, accepted_classes=[str], store_retval=False)
    def language(self, what): # pylint: disable=function-redefined
        self.load_language(what)

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, value):
        self._tag = value

    @property
    def result(self):
        '''The result object, if exists'''
        return self._result

    @property
    def search(self):
        '''The search algorithm used'''
        return self._result.search if self._result is not None else None
    
    @property
    def subgroups(self):
        return self._result.subgroups if self._result is not None else None
    
    @property
    def removed_subgroups(self):
        '''The result object, if exists'''
        return self._removed_subgroups
    
    @removed_subgroups.setter
    def removed_subgroups(self, which):
        removed_subgroups = []
        for sg in which:
            removed_subgroups.append(self.parse_subgroup(sg))
        self._removed_subgroups = removed_subgroups
    
    @property
    def measure(self) -> Measure:
        return self._measure
    measure = default_setter(measure, cls=Measure)
    
    @property
    def optimistic_estimator(self) -> OptimisticEstimator:
        return self._oest
    optimistic_estimator = default_setter(optimistic_estimator, cls=OptimisticEstimator)
    
    @property
    def kernel(self) -> Kernel:
        return self._kernel
    kernel = default_setter(kernel, cls=Kernel)
    
    @property
    def gramian(self) -> Gramian:
        return self._gramian
    @class_setter(gramian, cls=Gramian, accepted_classes=[np.ndarray], store_retval=True)
    def gramian(self, what): # pylint: disable=function-redefined
        from sergio.kernels.gramian import GramianFromArray
        return GramianFromArray(what)

    @property
    def resume(self):
        return self._resume
    
    controller = property(lambda self:self._controller, None, 'The controller this experiment is using, if any.')
    
    result_history = property(lambda self:self._result_history,None,
                              'Get the result history. Available if optimise is invoked with track_results set to True.')

    def parse_dataset(self, name, *args, **kwargs):
        df = DatasetFactory(cache = self.cache, file_manager = self.file_manager)
        return df.load_dataset(name, *args, **kwargs)
    
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
        self.dataset = self.parse_dataset(name,*args, **kwargs)
        return self
        
    def parse_prediciser(self, cuts=5, ranges='SLABS_POSITIVE', negate='BOOLEAN|CATEGORICAL|RANGED'):
        discretiser = FrequencyDiscretiser(cut_count=cuts, ranges=ranges)
        return Prediciser(discretiser = discretiser, negate=negate)
        
    def load_prediciser(self, cuts=5, ranges='SLABS_POSITIVE', negate='BOOLEAN|CATEGORICAL|RANGED'):
        """
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> _ = c.load_prediciser(4)
        """
        self._prediciser = self.parse_prediciser(cuts, ranges, negate)
        self._log.info(f'Loaded prediciser {self._prediciser}.')
        return self

    def parse_language(self, name='conjunctions'):
        language_cls = Language.__collection_factory__.tags[name]
        if self.dataset is None:
            raise ValueError('No dataset is set; languages need a dataset in order to be loaded.')
        dataset = self._require('dataset', 'a language')
        prediciser = self._require('prediciser', 'a language')
        predicates = tuple(dataset.make_predicates(prediciser))
        return language_cls(self.dataset, predicates)
        
    def load_language(self, name='conjunctions'):
        """
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> c.dataset = 'moondot'
        >>> _ = c.load_prediciser()
        >>> c.language = 'conjunctions'
        >>> tuple(map(str,c.language.refine(c.language.root)))
        ('{main}', '{!main}', '{a}', '{!a}', '{b}', '{!b}', '{c}', '{!c}')
        """
        self._language = self.parse_language(name)
        self._log.info(f'Loaded language {name} {self.language} ({len(self.language.predicates)} predicates)')
        return self

    def parse_subgroup(self, what):
        """
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> c.dataset = 'moondot'
        >>> _ = c.load_prediciser()
        >>> c.language = 'conjunctions'
        >>> tuple(map(str,c.language.refine(c.language.root)))
        ('{main}', '{!main}', '{a}', '{!a}', '{b}', '{!b}', '{c}', '{!c}')
        """
        if isinstance(what, Selector):
            sg = what
        else:
            language = self._require('language','subgroups')
            if isinstance(what, str):
                parser = language.make_parser()
                sg = parser.parse(what)
            else:
                raise TypeError(f'Cannot parse subgroup from {what} which is of type {type(what).__name__}.')
        return sg
    
    def remove_subgroup(self, what):
        """
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> _ = c.load_dataset('moondot').load_prediciser().load_language()
        >>> c.removed_subgroups = []
        >>> _ = c.remove_subgroup('{main}')
        >>> c.removed_subgroups
        [<ConjunctionSelector: <PredicateBoolean:["main"]>>]
        >>> c.removed_subgroups = ['{main}','{c}']
        >>> c.removed_subgroups
        [<ConjunctionSelector: <PredicateBoolean:["main"]>>, <ConjunctionSelector: <PredicateBoolean:["c"]>>]
        """
        sg = self.parse_subgroup(what)
        self._removed_subgroups.append(sg)
        return self
    
    def _require(self, attribute, loadable):
        '''Complain if not set'''
        try:
            val = getattr(self, attribute)
        except AttributeError:
            raise ValueError(f'Parsing of {loadable} requires the attribute "{attribute}" which is not available in {self.__class__.__name__}.')
        if val is None:
            raise ValueError(f'Parsing of {loadable} requires "{attribute}" to be set.')
        return val
    
    MISSING = NamedUniqueConstant('Missing')
    UNSET = NamedUniqueConstant('Unset')
    def _make_resolver(self, loadable, **kwargs):
        '''Create a helper function which optionally looks up attributes during construction'''
        def resolve(kwarg, info=None):
            val = kwargs.get(kwarg, self.MISSING)
            if val not in {self.MISSING, self.UNSET}:
                return val
            else:
                return self._require(kwarg, loadable)
        return resolve
    
    
    def parse_measure(self, name, dataset=UNSET, **kwargs):
        cls = Measure.__collection_factory__.tags[name]
        resolver = self._make_resolver(f'measure {cls.__name__}', dataset=dataset)
        meas = Measure.make_from_string_parts(name, kwargs=kwargs, kwarg_resolver=resolver)
        return meas
    
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
        meas = self._measure = self.parse_measure(name, **kwargs)
        self._log.info(f'Set measure to {meas}.')
        return self
    
    def parse_optimistic_estimator(self, name, dataset=UNSET, **kwargs):
        cls = OptimisticEstimator.__collection_factory__.tags[name]
        resolver = self._make_resolver(f'optimistic estimator {cls.__name__}', dataset=dataset)
        oest = OptimisticEstimator.make_from_string_parts(name, kwargs=kwargs, kwarg_resolver=resolver)
        return oest

    def load_optimistic_estimator(self, name, **kwargs):
        """
        >>> import sergio.scores.scalars
        >>> c = Computation(cache=None, file_manager=FileManager('datasets')).load_dataset('toy-scalar:circledots,ATTR,outer').load_prediciser().load_language()
        >>> c.load_optimistic_estimator('jaccard').optimistic_estimator
        <OptimisticEstimatorJaccard(target_name='outer')>
        """
        self._log.info(f'Loading optimistic estimator {name}.')
        oest = self._oest = self.parse_optimistic_estimator(name, **kwargs)
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
    
    def parse_kernel(self, name, **kwargs):
        return Kernel.make_from_strings(name, **kwargs)
        
    def load_kernel(self, name, **kwargs):
        """ Append attributes to the dataset.
        >>> import sergio.kernels.euclidean
        >>> c = Computation(cache=None, file_manager=FileManager('datasets'))
        >>> c.load_kernel('rbf').kernel
        <RadialBasisFunctionKernel(sigma=1,kind=<Kind.GAUSSIAN: 1>)>
        """
        log.info(f'Loading kernel {name}.')
        self._kernel = kern = self.parse_kernel(name, **kwargs)
        log.info(f'Set kernel to {kern}')
        return self
    
    
    def compute_gramian(self, dataset=UNSET, kernel=UNSET):
        resolver = self._make_resolver('Gramian', dataset=dataset, kernel=kernel)
        return GramianFromDataset(resolver('dataset'), resolver('kernel'))

    def load_gramian(self):
        '''
        >>> import sergio.scores.scalars
        >>> import sergio.kernels.euclidean
        >>> c = Computation(cache=None, tag='doctest', file_manager=FileManager('datasets'))\\
        ...     .load_dataset('toy-array:circledots').load_kernel('rbf')
        >>> G = c.compute_gramian()
        >>> G.eigenvals.shape[0], G.dimension, G.rank
        (147, 147, 147)
        >>> 
        '''
        log.info(f'Computing Gramian for {self.kernel}')
        self._gramian = self.compute_gramian()
        log.info(f'Set Gramian to {self._gramian!r}')
        return self

    
    def optimise(self, k=1, depths=6,approximation_factor=1, state_scoring=None, track_results=False, initial_subgroups=None):
        r'''
        >>> import sergio.scores.scalars
        >>> c = Computation(cache=None, tag='doctest', file_manager=FileManager('datasets'))\
        ...     .load_dataset('toy-scalar:circledots,ATTR,outer').load_prediciser().load_language()\
        ...     .load_measure('jaccard').load_optimistic_estimator('jaccard')
        >>> c
        <Computation[doctest] D:<EntityAttributesWithAttributeTarget[circledots](147x2/3) target: outer(bool@2)>, L:<ConjunctionLanguage: of 10 predicates>, SG:-, M:<MeasureJaccard(target_name='outer')>, O:<OptimisticEstimatorJaccard(target_name='outer')>>
        >>> res = c.optimise()
        >>> ','.join(map(str,res.subgroups))
        '{[label=circle]}'
        '''
        if track_results:
            log.info('Attaching results tracker visitor.')
            dfs_visitor = SearchResultLogger(self._result_history)
            result_history = dfs_visitor.result_history
            def dfs(*args, **kwargs):
                kwargs['visitor'] = dfs_visitor
                return DepthFirstSearch(*args,**kwargs)
        else:
            dfs = DepthFirstSearch
            result_history = None
        
        iddfs = IterativeDeepening(self._language, self._measure, self._oest,
                                   k=k, depths=depths,
                                   approximation_factor=approximation_factor,
                                   state_scoring=state_scoring,
                                   dfs=dfs
                                   )
        self._log.info(f'Starting Optimisation algorithm for k={k}, depths={depths}, language={self._language}, measure={self._measure}, oest={self._oest}.')
        if initial_subgroups:
            iddfs.try_add_selectors(initial_subgroups)
        aborted = False
        try:
            iddfs.run()
        except KeyboardInterrupt as _:
            self._resume = iddfs.states
            aborted = True
        subgroups = iddfs.subgroups
        self._log.info(f'Optimisation {"aborted" if aborted else "completed"} with results {to_stuple(subgroups,join=",")}.')
        return OptimisationResult(result_history=result_history, search=iddfs, subgroups=subgroups)
    
    def optimise_inplace(self, k=1, depths=6,approximation_factor=1, state_scoring=None, track_results=False, initial_subgroups=None):
        if initial_subgroups is None:
            initial_subgroups = self._result.subgroups
        self._result = self.optimise(k=k, depths=depths, approximation_factor=approximation_factor, state_scoring=state_scoring, track_results=track_results, initial_subgroups=initial_subgroups)
        return self
    
    def __summary_dict__(self, options:SummaryOptions):
        summary = super().__summary_dict__(options)
        if self.result_history is not None:
            summary['result_history'] = self.result_history
        return summary
    
    def summarise_to_json(self, indent=4, separators=(',', ': '), scores=[]):
        r'''
        :param scores: extra scores to evaluate the selectors at.
        
        >>> import sergio.scores.scalars
        >>> c = Computation(cache=None, tag='doctest', file_manager=FileManager('datasets'))\
        ...     .load_dataset('toy-scalar:circledots,ATTR,outer').load_prediciser().load_language()\
        ...     .load_measure('coverage-mean-shift').load_optimistic_estimator('coverage-mean-shift')
        >>> c
        <Computation[doctest] D:<EntityAttributesWithAttributeTarget[circledots](147x2/3) target: outer(bool@2)>, L:<ConjunctionLanguage: of 10 predicates>, SG:-, M:<MeasureCoverageMeanShift(coverage_exponent=1.0)>, O:<OptimisticEstimatorCoverageMeanShift(coverage_exponent=1.0)>>
        >>> res = c.optimise_inplace(depths=1)
        >>> ','.join(map(str,res.subgroups))
        '{[label=circle]}'
        >>> c.summarise_to_json()[0]
        '{'
        '''
        from colito.summaries import NamedSummariser
        if log.ison.progress:
            results_text = 'results' if self.result_history is None else f'actual and {len(self.result_history)} historical results'
            log.progress(f'Computing experiment summary with {len(self.subgroups)} {results_text}.')
        summariser = NamedSummariser()
        summariser.add_visitor(SelectorSummariser(scores))
        summariser.add_visitor(JSONConvertingVisitor())
        summary = summariser(self)
        import json
        jsonstr = json.dumps(summary, indent=indent, separators=separators)
        return jsonstr

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
        return f'<{self.__class__.__name__}[{self.tag}] D:{self.dataset}, L:{self.language}, SG:{len(self.subgroups) if self.subgroups is not None else "-"}, M:{self.measure}, O:{self.optimistic_estimator}>'
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    
    