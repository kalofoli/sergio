'''
Created on Dec 10, 2017

@author: janis
'''

# pylint: disable=C0103
# invalid constant name

import logging
from datetime import datetime
import os.path
from collections import UserList, OrderedDict
import json
import enum
import re
from math import inf
from typing import Dict

from colito.resolvers import make_enum_resolver
from colito.config import ActionParser, ActionBase, ActionParameter
from colito.summarisable import SummaryParts
from colito.runtime import RuntimeEnvironment
from colito.logging import SergioLogger, getModuleLogger, formatLevelNames

from sergio.language import ConjunctionLanguage, LANGUAGES
from sergio.measures import MEASURES, OPTIMISTIC_ESTIMATORS

from sergio.searches import ScoringFunctions
from sergio.experiment import Experiment
from sergio.datasets.factory import DatasetFactory
from sergio.predicates import PREDICATE_KINDS_RESOLVER
from sergio.discretisers import DISCRETISER_RANGE_RESOLVER
from sergio.attributes import AttributeKind

from errno import EEXIST

import argparse
import traceback
import signal
import sys

log = getModuleLogger(__name__)

class FileManager:
    class Kinds(enum.Enum):
        LOG = enum.auto()
        WORK = enum.auto()
        DATA = enum.auto()
        SOCKET = enum.auto()
        DEFAULT = enum.auto()

    FILE_KINDS_TEXT = ', '.join(Kinds.__members__.keys())
        
    def __init__(self, paths: Dict[str,str]={}, default_path=None) -> None:
        self._paths:Dict[str,str] = {}
        for kind,path in paths.items():
            self.set_kind_path(kind, path)
        if default_path is None:
            default_path = os.path.curdir
        self.set_kind_path(None, default_path)

    def _kind_enum(self, kind):
        if isinstance(kind, str):
            kind_enum = FileManager.Kinds[kind]
        elif isinstance(kind, FileManager.Kinds):
            kind_enum = kind
        elif kind is None:
            kind_enum =FileManager.Kinds.DEFAULT
        else:
            raise TypeError(f'Only kinds {self.FILE_KINDS_TEXT} are allowed.')
        return kind_enum
            
    def set_kind_path(self, kind, path):
        kind_enum = self._kind_enum(kind)
        self._paths[kind_enum] = path
        
    def get_kind_path(self, kind):
        kind_enum = self._kind_enum(kind)
        return self._paths.get(kind_enum, self._paths[FileManager.Kinds.DEFAULT])
    
    def get(self, file, kind=None):
        base = self.get_kind_path(kind)
        path = os.path.join(base, file)
        return path
    
    def __repr__(self):
        txt = ",".join(f'{k.name}:"{v}"' for k,v in self._paths.items())
        return f'<{self.__class__.__name__}:{txt}>'


class ExperimentActions(UserList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parameter_dict = OrderedDict()
        self._parameter_dict_values = OrderedDict()
        self._parameters = []
        for action in self.data:
            kind = action.action
            action_parameters = self._parameter_dict[kind] = self._parameter_dict.get(kind,{})
            self._parameter_dict_values[kind] = action.values
            action_parameters.update(action.values.as_dict())
            action_dict = action.as_dict()
            self._parameters.append(action_dict)
        for index,action in enumerate(self.data):
            action.set_actions(self, index)
            
    def update(self, action, name, value):
        self._parameters[action.index]['values'][name]= value
        self._parameter_dict[action.action][name] = value
                    
        
    def get_parameter(self, action_name, parameter, default=None):
        return self._parameter_dict.get(action_name,{}).get(parameter,default)
        
    def format(self, what, action=None):
        if isinstance(what, str):
            try:
                kwargs = OrderedDict()
                if action is not None:
                    kwargs.update(action.values.as_dict())
                kwargs['runtime'] = RuntimeEnvironment()
                kwargs.update(self._parameter_dict_values)
                args = self._parameters
                formatted = what.format(*args,**kwargs)
            except:
                raise ValueError(f'Could not format value: "{what}" for action {action}. Available data: {args!s}, {kwargs!s}')
            return formatted
        elif isinstance(what, list):
            return list(self.format(w, action) for w in what)
        elif what is None:
            return what
        else:
            raise TypeError('Can only format strings and lists of strings')
    
    @property
    def parameter_dict(self):
        return self._parameter_dict
    
    @property
    def parameters(self):
        return self._parameters
    
    def to_json(self, indent=4, separators=None):
        txt = json.dumps(self._parameter_dict, indent=indent, separators=separators)
        return txt
        
class FmtActionParameter(ActionParameter):
    def __init__(self, name:str, *args, **kwargs) -> None:
        if 'format' in kwargs:
            self._formattable = bool(kwargs['format'])
            del kwargs['format']
        else:
            self._formattable = False
        super().__init__(name, *args, **kwargs)
    
    @property
    def formattable(self):
        return self._formattable
    
    def __repr__(self):
        return f'<{self.__class__.__name__} {self._name}={self.default} Formattable:{"NY"[int(self.formattable)]}>'
        
    
class ExperimentAction(ActionBase):
    action: str = None

    file_manager = FileManager()
    
    def __init__(self, experiment):
        super(ExperimentAction, self).__init__()
        self._experiment = experiment
        self._actions = None
        self._index = None
        
    @property
    def index(self):
        return self._index

    @property
    def actions(self) -> ExperimentActions:
        return self._actions
    
    def set_actions(self, actions:ExperimentActions, index):
        self._actions = actions
        self._index = index
        for action_parameter in self.action_parameters:
            if isinstance(action_parameter, FmtActionParameter) and action_parameter.formattable:
                value_name = action_parameter.name
                value = getattr(self.values, value_name)
                value_formatted = actions.format(value, self)
                value_name_fmt = f'{value_name}_formatted'
                setattr(self.values,value_name_fmt,value_formatted)
                actions.update(self,value_name_fmt,value_formatted)

    @property
    def experiment(self):
        return self._experiment
    
    def perform(self):
        raise NotImplementedError()

    @classmethod
    def get_default(cls, parameter):
        for action_parameter in cls.action_parameters:
            if action_parameter.name == parameter:
                return action_parameter.default
        return None
    
    def __repr__(self):
        return f'<{self.__class__.__name__} with values: {self.values}>'
    
class ConfigAction(ExperimentAction):
    action = 'config'
    action_parameters = (FmtActionParameter('tag', '-t', help='Tag of this experiment.', format=True, default='run-{runtime.pid}-{runtime.date}'),
                         ActionParameter('log_delay', '-D', '--log-delay', default=[], action='append', metavar=('LEVEL','DELAY'),nargs=2,help=f'Delay (in seconds) between successive prints of the given message level.'),
                         ActionParameter('log_level', '-l', '--log-level', default='INFO', help=f'Log level to use. Available: {formatLevelNames()}.'),
                         ActionParameter('path', '-p', '--path', metavar=('KIND','PATH'), default=[], action='append',nargs=2, help=f'Set the paths for different uses. Available: {FileManager.FILE_KINDS_TEXT}. Python formatted.'),
                         FmtActionParameter('log_file', '-f', '--log-file', default='{config.tag_formatted}.log', format=True, help='File to write log to. It will be python-formatted.'),
#                         FmtActionParameter('socket_file', '-s', '--socket-file', default='{config.tag_formatted}.sock', format=True, help='Socket for the control interface.'),
                         ActionParameter('log_fmt', '-F', '--log-format', default=SergioLogger.default_format(),metavar='FMT', help='Format to use when writing.'),
                         ActionParameter('reuse', '-R', '--reuse', default=False, action='store_true', help='Whether the log file should be reused if it exist.'),
                         )

    @property
    def log_file(self):
        if self.values.log_file is None:
            return None
        else:
            return self.file_manager.get(self.values.log_file_formatted,FileManager.Kinds.LOG)
    
    def validate(self):
        log.setLevel(self.values.log_level)
        for kind,path in self.values.path:
            path_formatted = self.actions.format(path, self)
            self.file_manager.set_kind_path(kind, path_formatted)
        if self.log_file is not None and not self.values.reuse and os.path.exists(self.log_file):
            raise OSError(EEXIST, f'Refusing to reuse existing log file "{self.log_file}".')

    def perform(self):
        # Setup logging
        self.validate()
        if self.log_file is not None:
            file = self.file_manager.get(self.values.log_file_formatted,FileManager.Kinds.LOG)
            ch = logging.FileHandler(file)
            ch.setLevel(log.level)
            ch.setFormatter(logging.Formatter(self.values.log_fmt))
            log.addHandler(ch)
            log.info(f'Added logging handler to "{file}".')
        for (lvl,delay) in self.values.log_delay:
            lvl_enum = SergioLogger.get_level(lvl)
            log.rlim.set_delay(delay=float(delay),level=lvl_enum)
        log.info(f'Set logger level to {log.level} and set delays to {GLOBAL_RATE_TRACKER}.')
        self.experiment.tag = self.values.tag
        if log.ison.info:
            argv = self.actions.argv
            rex = re.compile('^[a-zA-Z/_0-9.-]+$')
            arg_txt= " ".join(("{}" if rex.match(arg) else "'{}'").format(arg.replace("'","\'")) for arg in argv)
            log.info(log.ison.info and f'Invocation with arguments: {arg_txt}')
            log.info(log.ison.info and f'Using config:\n{actions.to_json()}')
        log.info(f'Redirecting standard output/error.')
        log.attach_standard()
        from sdcore.summaries import Summariser
        fields = Summariser().summarise_asfields(self.experiment.runtime_environment)
        text = fields.flatten().dump(use_repr=False,pad=1,align_keys='<', prefix='RuntimeEnvironment.')
        log.info(f'Configured run with runtime environment:\n{text}')
#        if self.values.sock_file:
#            self.experiment.add_cli_server(self.values.sock_file, 'UNIX')

class SignalHandlers(enum.Enum):
    INTERRUPT = enum.auto()
    SUMMARISE_JSON = enum.auto()
    SIG_ASIS = enum.auto()
    SIG_IGN = enum.auto()
    SIG_DFL = enum.auto()
    
SIGNAL_HANDLERS_RESOLVER = make_enum_resolver(SignalHandlers)
SIGNAL_RESOLVER = make_enum_resolver(signal.Signals)
class ControlAction(ExperimentAction):
    action = 'control'
    action_parameters = (FmtActionParameter('file', '-o', '--output', default=None, format=True, metavar='FILE', help='The file to write the triggered result to. If not specified, output is written to the standard output. Python formatted.'),
                         ActionParameter('signals', '-s', '--signal', default=[['SIGINT', 'INTERRUPT'],], action='append', metavar=('SIGNAL','EVENT'),nargs=2,help=f'Set handlers for signals. A signal of ALL affects all so-far specified signals. Setting a handler to SIG_ASIS takes no action (to overwrite a previously given setting), SIG_IGN sets to SIG_IGN and SIG_DFL uses the default handler. In total, all handlers are: {SIGNAL_HANDLERS_RESOLVER.options()}.'),
                         ActionParameter('list', '-l', '--list', default=False, action='store_true', help=f'List all known signals and exit.'),
                         ActionParameter('fork', '-f', '--fork', default=False, action='store_true', help=f'Fork before computing the summary.'),
                         )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._handlers = {}

    def handle_interrupt(self, signum, frame):
        raise KeyboardInterrupt()

    def handle_summarise_json(self, signum, frame):
        try:
            indent = self.actions.get_parameter('json-summarise','indent',SummariseJSONAction.get_default('indent'))
            separators = self.actions.get_parameter('json-summarise','separators',SummariseJSONAction.get_default('separators'))
            parts = self.actions.get_parameter('json-summarise','parts',SummariseJSONAction.get_default('parts'))
            append = self.actions.get_parameter('json-summarise','append',SummariseJSONAction.get_default('append'))
            output_file = self.values.file_formatted
            def do_write():
                text = self.experiment.summarise_to_json(indent=indent, separators=separators, parts=parts)
                write_data(data=text, title='JSON summaries', file=output_file,append=append)
            if self.values.fork:
                pid = os.fork()
                if pid:
                    log.info(f'Forked child process {pid} for writting on {"stdout" if output_file is None else output_file}')
                else:
                    do_write()
            else:
                do_write()
        except Exception:
            msg = traceback.format_exc()
            log.error(f'While outputing JSON:\n{msg}')
    
        
    def _store_handler(self, sigval, handler):
        '''Stores in each signal enum specified the requested handle enum'''
        signum = SIGNAL_RESOLVER.resolve(sigval)
        if handler == SignalHandlers.SIG_ASIS:
            if sigval in self._handlers:
                del self._handlers[signum]
        else:
            self._handlers[signum] = handler
    
    def _set_handler(self, sigval, handler):
        '''Sets the actual signal, depending on the SignalHandlers type'''
        if handler == SignalHandlers.SIG_IGN:
            handler_val = signal.SIG_IGN
        elif handler == SignalHandlers.SIG_DFL:
            handler_val = signal.SIG_DFL
        elif handler == SignalHandlers.SIG_ASIS:
            raise TypeError('Internal error: ASIS survived till signal setting. Contact developers.')
        elif isinstance(handler, SignalHandlers):
            handler_name = f'handle_{handler.name.lower()}'
            try:
                handler_val = getattr(self, handler_name)
            except AttributeError:
                raise TypeError(f'Internal error: No handler method {handler_name} defined for for handler {handler_val}. Contact developers.')
        else:
            raise TypeError(f'Object {handler} is not a valid SignalHandlers instance')
        log.info(f'Setting handler {handler.name} for signal {sigval.name}...')
        handle_old = signal.signal(sigval, handler_val)
        log.info(f'Setting handler {handler.name} for signal {sigval.name} replaced handler {handle_old}.')
            
    
    def validate(self):
        if not hasattr(signal,'signal'):
            raise ImportError('No signal method available in signal module.')
        if self.values.list:
            sigspec = SIGNAL_RESOLVER.options(sep="\n",equal=' ',pad_names=1)
            log.info(f'Known signals are:\n{sigspec}')
            print(sigspec)
            sys.exit()
        for signal_in, handler_in in self.values.signals:
            handler = SIGNAL_HANDLERS_RESOLVER.resolve(handler_in)
            if isinstance(signal_in, str):
                if signal_in.lower() == 'all':
                    for sigval in self._handlers.keys():
                        self._store_handler(sigval, handler)
                else:
                    self._store_handler(signal_in, handler)
            elif isinstance(signal_in, signal.Signals):
                sigval = signal_in
                self._store_handler(sigval, handler)
            
    
    def perform(self):
        for signum, handler in self._handlers.items():
            self._set_handler(signum, handler) 

class LoadDatasetAction(ExperimentAction):
    action = 'load-dataset'
    action_parameters = (# ActionParameter('mode','mode',default='load',choices=MODES),
                         ActionParameter('name', '-n', help=f'Dataset Name. Choices are: {DatasetFactory.description}', metavar='NAME[:ARGS]'),
                         ActionParameter('cut_count', '-c', '--cut-count', default=5, type=int,help=f'Number of cuts used for the numeric attributes.', metavar='CUTS'),
                         ActionParameter('ranges', '-r', '--ranges', default='SLABS_POSITIVE', help=f'Type of ranges to create based on the cuts. Valid options: {DISCRETISER_RANGE_RESOLVER.options()}', metavar='RANGE1|RANGE2'),
                         ActionParameter('negate', '-N', '--negate', default='ALL', help=f'Kind of predicates to also negate during predicisation. Valid options: {PREDICATE_KINDS_RESOLVER.options()}', metavar='KIND1|KIND2'),
                         )

    def __init__(self, experiment):
        super().__init__(experiment)
    
    def perform(self):
        self.experiment.load_dataset(self.values.name, cuts=self.values.cut_count, ranges=self.values.ranges,negate=self.values.negate)


class LoadLanguageAction(ExperimentAction):
    action = 'load-language'
    action_parameters = (ActionParameter('name', '-n', help='Language Name', choices=LANGUAGES.tags),)

    def perform(self):
        self.experiment.load_language(self.values.name)


class LoadScoresAction(ExperimentAction):
    action = 'load-scores'
    action_parameters = (ActionParameter('measure', '-m', metavar='MEASURE', help='The measure to use.', choices=MEASURES.tags),
                         ActionParameter('oest', '-e', metavar="ESTIMATOR", help='The optimistic estimator to use.', choices=OPTIMISTIC_ESTIMATORS.tags),
                         ActionParameter('meas_params', '--measure-parameter', '-M', default=[], nargs=2, action='append', metavar="KEY VALUE", help='Add named parameters for the provided measure.'),
                         ActionParameter('comm_params', '--common-parameter', '-C', default=[], nargs=2, action='append', metavar="KEY VALUE", help='Add named parameters for both provided measures.'),
                         ActionParameter('oest_params', '--optimistic_estimator-parameter', '-E', default=[], nargs=2, action='append', metavar="KEY VALUE", help='Add named parameters for the provided estimator.'),
                         ActionParameter('min_sup', '--min-support', '-s', default=None, type=float, metavar="MIN_SUP", help='The minimum support parameter used for the COIN estimators.'),
                         ActionParameter('gamma', '-g', default=None, type=float, metavar="GAMMA", help='The gamma parameter used for some estimators.'),
                         )

    def perform(self):
        kwargs_common = dict(self.values.comm_params)
        if self.values.gamma is not None:
            kwargs_common['gamma'] = self.values.gamma
        if self.values.min_sup is not None:
            kwargs_common['minimum_support'] = self.values.min_sup
        if self.values.measure:
            kwargs = kwargs_common.copy()
            kwargs.update(dict(self.values.meas_params))
            self.experiment.load_measure(name=self.values.measure, **kwargs)
        if self.values.oest:
            kwargs = kwargs_common.copy()
            kwargs.update(dict(self.values.oest_params))
            self.experiment.load_optimistic_estimator(name=self.values.oest, **kwargs)


class LoadSubgroupsAction(ExperimentAction):
    action = 'load-subgroups'
    action_parameters = (ActionParameter('strings', '-s', nargs='*', default=[], metavar='SUBGROUP', help='A list of string representations of subgroup descriptions to load.'),
                         FmtActionParameter('files', '-f', action='append', format=True, default=[], metavar='FILE', help='A file to read subgroup descriptions from. These must be specified one per line. Python formatted.'),
                         ActionParameter('append', '-a', type=bool, default=False, metavar='APPEND', help='Define whether to append or not the newly loaded list.'),
                         )

    def perform(self):
        for file in self.values.files_formatted:
            log.info(f'Reading subgroups from file "{file}"')
            with open(file, 'r') as fid:
                strings = fid.readlines()
            strings = tuple(map(lambda x:x[0], filter(bool, map(ConjunctionLanguage.SelectorParser.get_subgroups, strings))))
            log.info(f'Read {len(strings)} subgroups from file "{file}"')
            self.experiment.load_subgroups(strings, append=self.values.append)
        if self.values.strings:
            self.experiment.load_subgroups(self.values.strings, append=self.values.append)

ATTRBUTE_KIND_RESOLVER = make_enum_resolver(AttributeKind)
class LoadAttributesAction(ExperimentAction):
    action = 'load-attributes'
    action_parameters = (FmtActionParameter('files', '-f', action='append', format=True, default=[], metavar='FILE', help='A file to read attributes from. Python formatted.'),
                         ActionParameter('gzip', '-g', action='store_true', default=False, help='Whether gzip is to be expected.'),
                         ActionParameter('selected', '-s', metavar='SELECTED', default='false', help='Whether these columns should be selected for predicisation. Either a single value or a comma separated one of k values, one per added entry.'),
                         ActionParameter('dtypes', '-d', metavar='DTYPE', default='auto', help=f'The data type that the columns will be cast into. Single string value of {ATTRBUTE_KIND_RESOLVER.options()} or "auto" or a comma separated list of them, one per added entry.'),
                         ActionParameter('kinds', '-k', metavar='KIND', default='auto', help=f'The attribute kind of each column. Single value (string or index) of {ATTRBUTE_KIND_RESOLVER.options()} or "auto" or a comma separated list of them, one per added entry.'),
                         )

    def perform(self):
        import pandas as pd
        attribute_lst = []
        for file in self.values.files_formatted:
            log.info(f'Reading attributes from file "{file}"')
            if self.values.gzip:
                import gzip
                fnopen = gzip.open
            else:
                fnopen = open  
            with fnopen(file, 'r') as fid:
                data = pd.read_csv(fid)
            log.info(f'Read:  {data.shape[0]}x{data.shape[1]} attributes: {tuple(data.columns)}')
            attribute_lst.append(data)
        attrs = pd.concat(attribute_lst, axis=1) 
        log.info(f'Total: {attrs.shape[0]}x{attrs.shape[1]} attributes: {tuple(attrs.columns)}')
        
        selected = self.values.selected.split(',')
        if len(selected) == 1:
            selected = selected[0]
        kinds = self.values.kinds.split(',')
        if len(kinds)==1:
            kinds = kinds[0]
        dtypes = self.values.dtypes.split(',')
        if len(dtypes) == 1: dtypes = dtypes*attrs.shape[1]
        dtypes = dict((name,dtype) for name,dtype in zip(attrs.columns,dtypes) if dtype is not 'auto')
        attrs = attrs.astype(dtypes)
        self.experiment.load_attributes(attributes=attrs, kinds=kinds, selected=selected)

class OptimiseAction(ExperimentAction):
    from math import inf
    action = 'optimise'
    action_parameters = (ActionParameter('num_results', '-k', default=10, type=int, metavar='NUM_RESULTS', help='The number of best subgroups to keep track of.'),
                         ActionParameter('max_depth', '-d', default=inf, type=float, metavar='MAX_DEPTH', help='The maximum number of patterns allowed in the subgroup descriptor.'),
                         ActionParameter('state_scoring', '--state-scoring','-S', default=None, type=str, metavar='NAME[:K=V[,K=V]]', help=f'The state scoring description. Higher values lead to earlier pruning. Available: {ScoringFunctions.description}.'),
                         ActionParameter('refinement_scoring', '--refinement-scoring','-R', default=None, type=str, metavar='NAME[:K=V[,K=V]]', help=f'The refinement scoring description. Lower values lead to longer blacklists. Available: {ScoringFunctions.description}.'),
                         ActionParameter('approximation_factor', '-a', '--approximation-factor', default=1.0, type=float, metavar='ALPHA', help='The approximation factor. Should be smaller than 1. Guarantees the found solution is no worse than this factor times the exact best.'),
                         ActionParameter('track_results', '-t', '--track-results', default=False, action='store_true', help='Whether the addition of a result should be tracked in the output.'),
                         FmtActionParameter('resume_file', '-r', '--resume-file', format=True, default='{config.tag_formatted}-resume.states', metavar='STATES', help='File to use to save status for resume data in case of termination. Python formatted.'),
                         )

    def perform(self):
        if self.values.approximation_factor>1:
            raise ValueError(f'It makes no sense for the approximation factor to be larger than 1 (specified value: {self.values.approximation_factor}).')
        file_resume = self.file_manager.get(self.values.resume_file_formatted, FileManager.Kinds.WORK)
        try:
            with open(file_resume, 'wb') as fid:
                resume = json.load(self.experiment.resume,fid)
        except:
            log.info(f'Could not load json data from resume file "{file_resume}"')
            resume = None
        self.experiment.optimise(k=self.values.num_results,
                                 depths=self.values.max_depth,
                                 approximation_factor = self.values.approximation_factor,
                                 state_scoring=self.values.state_scoring,
                                 refinement_scoring=self.values.refinement_scoring,
                                 track_results=self.values.track_results,
                                 )
        if self.experiment.resume is not None:
            log.info(f'Writting resume data of len {len(self.experiment.resume)} to file: "{file_resume}"')
            try:
                with open(file_resume, 'w') as fid:
                    digests = list(r.serialise(json=True) for r in self.experiment.resume)
                    json.dump(digests,fid)
            except Exception as exc:
                log.error(f'Exception: {exc} while writing resume data to "{file_resume}"')
            
        
class CommunitiesAction(ExperimentAction):
    action = 'communities'
    action_parameters = (ActionParameter('num_results', '-k', default=10, type=int, metavar='NUM_RESULTS', help='The number of communities to keep track of.'),
                         )

    def perform(self):
        self.experiment.communities(k=self.values.num_results)


def write_data(data, file=None, title='data', append=False):
    file_name = f'"{file}"' if file is not None else '<stdout>'
    if log.isEnabledFor('DATA'):
        log.data(f'Writing {title} to {file_name}:\n{data}')
    else:
        log.info(f'Writing {title} to {file_name}')
    if file is None:
        print(data)
    else:
        mode = 'a+' if append else 'w'
        with open(file, mode) as fid:
            fid.write(data)

class SummariseJSONAction(ExperimentAction):
    action = 'summarise-json'
    action_parameters = (ActionParameter('indent', '-i', '--indent', default=2, type=int, metavar='INDENT', help='The indent scheme to use in the JSON formatter.'),
                         ActionParameter('separators', '-s', '--separators', default=None, nargs=2, metavar='COMMA COLON', help='The separators to use in the JSON formatter.'),
                         FmtActionParameter('file', '-o', '--output', default=None, format=True, metavar='FILE', help='The file to write the result to. If not specified, output is written to the standard output. Python formatted.'),
                         ActionParameter('append', '-a', '--append', default=False, action='store_true', help='Whether the results should be appended to the file.'),
                         ActionParameter('depth', '-d', '--depth', default=inf, metavar='DEPTH', type=float, help='How deep should the JSON summary tree be.'),
                         ActionParameter('parts', '-p', '--parts', default='BASIC', metavar='PRT1[|PRT2]', help=f'The parts of the summary. Available are {SummaryParts.members_desc()}. One can also use a numeric value or "ALL" for all.'),
                         ActionParameter('overwrite', '-O', '--overwrite', default=False, action='store_true', help='Whether the results should be overwritten if they exist.'),
                         )

    def validate(self):
        output_file = self.values.file_formatted
        if not self.values.overwrite and os.path.exists(output_file):
            raise OSError(EEXIST, f'Refusing to overwrite existing output file "{output_file}".')
    
    def perform(self):
        text = self.experiment.summarise_to_json(indent=self.values.indent, separators=self.values.separators, parts=self.values.parts)
        output_file = self.values.file_formatted
        write_data(data=text, title='JSON summaries', file=output_file,append=self.values.append)

class OutputJSONAction(ExperimentAction): 
    action = 'output-json'
    action_parameters = (ActionParameter('indent', '-i', '--indent', default=2, type=int, metavar='INDENT', help='The indent scheme to use in the JSON formatter.'),
                         ActionParameter('separators', '-s', '--separators', default=None, nargs=2, metavar='COMMA COLON', help='The separators to use in the JSON formatter.'),
                         FmtActionParameter('file', '-o', '--output', default=None, format=True, metavar='FILE', help='The file to write the result to. If not specified, output is written to the standard output. Python formatted.'),
                         )

    def perform(self):
        text = self.experiment.subgroups_to_json(indent=self.values.indent, separators=self.values.separators)
        write_data(data=text, title='JSON output', file=self.values.file_formatted,append=self.values.append)

class OutputCSVAction(ExperimentAction):
    action = 'output-csv'
    action_parameters = (ActionParameter('separator', '-s', '--separator', default=',', metavar='SEP', help='The separator to use in the CSV format.'),
                         ActionParameter('format', '-f', '--format', default='%.5f', metavar='FORMAT', help='Float format to use.'),
                         ActionParameter('index', '-i', '--index', default=False, action='store_true', help='Also write index.'),
                         ActionParameter('header', '-H', '--no-header', default=True, action='store_false', help='Suppress printing header.'),
                         FmtActionParameter('file', '-o', '--output', default=None, format=True, metavar='FILE', help='The file to write the result to. If not specified, output is written to the standard output. Python formatted.'),
                         ActionParameter('append', '-a', '--append', default=False, action='store_true', help='Whether the results should be appended to the file.'),
                         )

    def perform(self):
        text = self.experiment.subgroups_to_csv(separator=self.values.separator,
                                                float_format=self.values.format,
                                                header=self.values.header,
                                                index=self.values.index)
        write_data(data=text, title='CSV of subgroups', file=self.values.file_formatted,append=self.values.append)

class ExportDatasetAction(ExperimentAction):
    action = 'export-dataset'
    action_parameters = (FmtActionParameter('file', '-o', '--output', format=True, default='{load-dataset.name}-{load-dataset.cut_count}-{load-dataset.negate}-{load-dataset.ranges}.mat', metavar='FILE', help='A file to save the dataset to. Python formatted.'),
                         ActionParameter('overwrite', '-O', '--overwrite', default=False, action='store_true', help='Whether the output should be overwritten if it exists.'),
                         )

    def validate(self):
        output_file = self.file_manager.get(self.values.file_formatted, FileManager.Kinds.DATA)
        if not self.values.overwrite and os.path.exists(output_file):
            raise OSError(EEXIST, f'Refusing to overwrite existing output file "{output_file}".')
    
    def perform(self):
        output_file = self.file_manager.get(self.values.file_formatted, FileManager.Kinds.DATA)
        log.info(f'Writing dataset {self.experiment.dataset_name} to {output_file}')
        self.experiment.export_dataset(output_file)
    

