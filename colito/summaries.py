'''
Created on May 3, 2018

@author: janis
'''

import json
import enum

from collections import OrderedDict, namedtuple
from .summarisable import Summarisable, SummaryParts, SummaryOptions,\
    SummaryCompressible, DEFAULT_SUMMARY_OPTIONS, SummarisableDict,\
    SummarisableList, SummarisableAsDict, SummaryGroupable,\
    SummarisableAsList, CompressionToString, CompressionScheme
# from cofi.factory import ProductBundle
from .logging import getModuleLogger
from .resolvers import make_enum_resolver

log = getModuleLogger(__name__)

from typing import Dict, Any, Sequence, Union, List, cast
from abc import ABC, abstractmethod
from math import inf, nan

class SummaryState:

    def __init__(self, instance, depth, value=None, name=None, parent=None, key=None, must_summarise=None, priority:int=None) -> None:
        self.instance = instance
        self.value = value if value is not None else instance
        self.name = name if name is not None else type(self.value).__name__
        self.depth = depth
        self.parent:SummaryState = parent
        self.key = key
        self.must_summarise = must_summarise if must_summarise is not None else isinstance(self.value, Summarisable) 
        self.priority:int = priority if priority is not None else self.value.summary_sibling_priority if hasattr(self.value,'summary_sibling_priority') else 0
    @property
    def cls(self):
        return type(self.instance).__name__
    
    def __repr__(self):
        s = str(self.value)
        sval = f'"{s:.17}"...' if len(s)>20 else f'"{s}"'
        sname = f'{self.key}=' if self.key is not None else f'{self.name}=' if self.name is not None else ""
        return f'<State[{"S" if self.must_summarise else "s"}@{self.depth}]{sname}{sval}>'

class FilterStage(enum.Enum):
    IMMEDIATE = enum.auto()
    SUMMARISED = enum.auto()
    ASSEMBLE = enum.auto()

class SummaryVisitor:
    def on_start(self, state:SummaryState, actions): pass
    def on_encounter(self, state:SummaryState, actions): pass
    def on_children(self, state:SummaryState, actions, new_actions): pass
    def on_summarised(self, state:SummaryState, actions): pass
    def on_assemble(self, state:SummaryState, actions):pass


def ifapplicable(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(self, state, *args, **kwargs):
        if self.isapplicable(state):
            fn(self, state, *args, **kwargs)
    return wrapper

class ClassSummaryVisitor(SummaryVisitor,ABC):
    __summary_class__ = None
    
    def isapplicable(self, state:SummaryState):
        summary_class = self.__summary_class__ 
        if isinstance(summary_class, type):
            isapplicable = isinstance(state.value, summary_class)
        elif isinstance(summary_class, str):
            state_name = state.name
            isapplicable = state.cls.__name__ == summary_class or state_name == summary_class
        else:
            raise TypeError(f'This class has a wrong type of __summary_class__ member: {summary_class}.')
        return isapplicable
    
class StopFiltering(Exception): pass

class OnError(enum.Enum):
    IGNORE = enum.auto()
    SUMMARISE = enum.auto()
    RAISE = enum.auto()
    
ON_ERROR_RESOLVER = make_enum_resolver(OnError)

class Fields(OrderedDict):
    def __init__(self, name, pairs):
        self.pairs = list(pairs)
        self.name = name
        super().__init__(pairs)
    
    def copy(self):
        return Fields(str(self.name), list(self.items()))
    
    def __iadd__(self, fields:'Fields'):
        if not isinstance(fields, Fields):
            raise TypeError(f'Cannot combine class {self.__class__.__name__} with {fields.__class__.__name__}.')
        self.pairs.append((fields.name,fields))
        return self
        
    def __add__(self, fields:'Fields'):
        fields_new = self.copy()
        fields_new += fields
        return fields_new
    
    def __repr__(self):
        return f'<{self.__class__.__name__} with {len(self)} fields>'
    
    def dump(self, sep='=', newline='\n', align_keys=False, pad=0, use_repr=True, prefix=''):
        if align_keys:
            key_size = max(map(len, self.keys()))
            key_mod = f':{align_keys if isinstance(align_keys,str) else ""}{key_size}'
        else:
            key_mod = ''
        fmt = f'{prefix}{{key!s{key_mod}}}{" "*pad}{sep}{" "*pad}{{value{"!r" if use_repr else "!s"}}}'
        text = newline.join(fmt.format(key=key, value=value) for key,value in self.items())
        return text
    
    def flatten(self, sep='.'):
        """Flatten the fields of this SummaryFields into a new one.
        
        The names in the flat one are the concatenation of all ancestral key names joined with the provided separator."""
        def flatten_pairs(pairs):
            flat_pairs = []
            for key, value in pairs:
                if isinstance(value, Fields):
                    fields = cast(Fields, value)
                    sub_pairs = flatten_pairs(fields.items())
                    flat_pairs += list((f'{key}{sep}{sub_key}',sub_value) for sub_key,sub_value in sub_pairs)
                else:
                    flat_pairs.append((key,value))
            return flat_pairs
        pairs = flatten_pairs(self.items())
        fields = Fields(str(self.name), pairs)
        return fields
    
class SummaryException(Exception): pass

class Summariser:
    def __init__(self, visitors:Sequence[SummaryVisitor]=[], options:SummaryOptions=DEFAULT_SUMMARY_OPTIONS, onerror=OnError.RAISE):
        self.visitors = []
        for v in visitors:
            self.add_visitor(v)
        self.options = options
        self.onerror = ON_ERROR_RESOLVER.resolve(onerror)

    def add_visitor(self, state_visitor:SummaryVisitor):
        self.visitors.append(state_visitor)
        return self
        
    class SummaryAction:
        def __init__(self, summariser, state):
            self.summariser = summariser
            self.state = state
        @property
        def priority(self): return self.state.priority
        @property
        def options(self): return self.summariser.options
        def __call__(self, actions):
            raise NotImplementedError()
        def __repr__(self):
            return f'<{self.__class__.__name__} ({self.state})>'
    
    class SummariseAction(SummaryAction):
        def __init__(self, summariser, state):
            super().__init__(summariser=summariser,state=state)
        @property
        def isroot(self): return self.parent is None
        
        def __call__(self, actions):
            state = self.state
            self.summariser.notify('encounter', state, actions)
            AssembleAction = self.summariser.AssembleAction
            value = state.value
            options = self.options
            if isinstance(value, Summarisable) and state.must_summarise:
                state.value = value.summary(options)
                state.name = value.summary_name
                state.must_summarise = False
            self.summariser.notify('summarised', state, actions)
            if isinstance(state.value, (SummarisableDict, SummarisableList)):
                new_actions_children = []
                if isinstance(state.value, SummarisableDict):
                    state.value = dict(state.value)
                    for key,entry in state.value.items():
                        new_actions_children.append(self.process_key(state, actions, key, entry))
                elif isinstance(state.value, SummarisableList):
                    state.value = list(state.value)
                    for key,entry in enumerate(state.value):
                        new_actions_children.append(self.process_key(state, actions, key, entry))
                action_assemble = AssembleAction(summariser=self.summariser, state=state)
                new_actions = [action_assemble] + new_actions_children
                new_actions_srt = sorted(new_actions, key = lambda a:a.priority)
                self.summariser.notify('children',state, actions, new_actions=new_actions_srt)
                actions += new_actions_srt
            else:
                if state.parent is not None:
                    state.parent.value[state.key] = state.value
            return state.value
        def process_key(self, state, actions, key, entry):
            SummariseAction = self.summariser.SummariseAction
            sub_state = SummaryState(entry, depth=state.depth+1, parent=state, key = key)
            action = SummariseAction(self.summariser, sub_state)
            return action

    class AssembleAction(SummaryAction):
        priority = -1000 # overrides state priority. If the child state of the root has a lower priority than this, it overrides the final value of the summariser.
        def __call__(self,actions):
            state = self.state
            self.summariser.notify('assemble', state=state, actions=actions)
            if state.parent is not None:
                state.parent.value[state.key] = state.value
            return state.value
    
    
    def notify(self, event, state, actions, *args, **kwargs):
        try:
            for v in self.visitors[::-1]:
                fn = getattr(v,f'on_{event}')
                fn(state=state, actions=actions, **kwargs)
        except StopFiltering:
            pass
        except Exception as e:
            if self.onerror == OnError.SUMMARISE:
                log.exception(f'Error during event {event} state {state}: {e}')
                state.value = {'summary_error':str(e),'state':str(state),'error_type':type(e).__name__}
            else:
                raise
    def _initialise(self, instance, depth):
        state = SummaryState(instance, depth=depth)
        actions = [self.SummariseAction(self, state)]
        self.notify('start', state, actions)
        return actions
    def __call__(self, instance, depth=0):
        actions = self._initialise(instance, depth)
        while actions:
            action = actions.pop()
            try:
                res = action(actions)
            except Exception as e:
                if self.onerror == OnError.IGNORE:
                    log.warn(f'Summary error during handling of action {action}.')
                else:
                    raise SummaryException(f'While performing action {action}.') from e
                    
        return res

class NamingVisitor(SummaryVisitor):
    def __init__(self,flatten_dicts:bool=True, flatten_lists:bool=True):
        self.flatten_dicts = flatten_dicts
        self.flatten_lists = flatten_lists
        
    def on_assemble(self, state, actions):
        if isinstance(state.value, dict):
            if self.flatten_dicts:
                state.value = {'class':state.name, **state.value}
            else:
                state.value = {'name':state.name, 'records': state.value}
        elif isinstance(state.value, list):
            if self.flatten_lists:
                pass
            else:
                state.value = {'class':state.name, 'entries': state.value}
class NamedSummariser(Summariser):
    def __init__(self, *args, flatten_dicts:bool=True, flatten_lists:bool=True,**kwargs):
        super().__init__(*args, **kwargs)
        self.add_visitor(NamingVisitor(flatten_dicts=flatten_dicts, flatten_lists=flatten_lists))


class FieldVisitor(SummaryVisitor):
    def on_assemble(self, state, actions):
        if isinstance(state.value, dict):
            state.value = Fields(state.name, state.value.items())
class FieldSummariser(Summariser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_visitor(FieldVisitor())
 
try:
    import skopt
    class SearchSpaceSummaryVisitor(ClassSummaryVisitor):
        __summary_class__ = skopt.space.Dimension
        
        @ifapplicable
        def on_encounter(self, state:SummaryState):
            fields = ('args','name','digest')
            state.value = {field:state.value[field] for field in fields}
except ImportError: pass

COMPRESSION_SCHEME_RESOLVER = make_enum_resolver(CompressionScheme)
COMPRESSION_STRINGIFY_RESOLVER = make_enum_resolver(CompressionToString)

class CompressedEntry(dict):
    def __init__(self, value, scheme:CompressionScheme='b64gzip', tostring:CompressionToString='auto', encoding = 'utf8'):
        txt, tostring = self._stringify(value, COMPRESSION_STRINGIFY_RESOLVER.resolve(tostring))
        txt_bin = txt.encode(encoding)
        data, scheme = self._compress(txt_bin, COMPRESSION_SCHEME_RESOLVER.resolve(scheme))
        txt_data = data.decode('latin')
        super().__init__(value=txt_data, scheme=scheme, tostring=tostring, encoding=encoding)
    
    @classmethod
    def _stringify(cls, what, tostring):
        if tostring == CompressionToString.AUTO:
            if isinstance(what, (str, bytes)):
                return cls._stringify(what, CompressionToString.RAW)
            else:
                try:
                    return cls._stringify(what, CompressionToString.JSON)
                except TypeError:
                    return cls._stringify(what, CompressionToString.STR)
        else:
            if tostring == CompressionToString.JSON:
                s = cls._to_json(what)
            elif tostring == CompressionToString.STR or tostring == CompressionToString.RAW:
                s = str(what)
            elif tostring == CompressionToString.REPR:
                s = repr(what)
            else:
                raise RuntimeError(f'Implementation error. Contact developers.')
        return s, tostring.name
    @classmethod
    def _compress(cls, data, scheme:CompressionScheme):
        import base64, gzip
        if scheme == CompressionScheme.GZIP or scheme == CompressionScheme.B64GZIP:
            data = gzip.compress(data)
        if scheme == CompressionScheme.BASE64 or scheme == CompressionScheme.B64GZIP:
            data = base64.b64encode(data)
        return data, scheme.name
        
    @classmethod
    def _to_json(cls, what):
        je = json.JSONEncoder(indent=None, separators=',:')
        return je.encode(what)

class CompressingSummaryVisitor(ClassSummaryVisitor):
    __summary_class__ = SummaryCompressible
    encoding = 'utf8'
    def __init__(self, plain=False, scheme=CompressionScheme.B64GZIP, tostring=CompressionToString.AUTO, allow_override:bool=True):
        '''@param reorder_expansion: if the expansion of a data instance is requested, reorder it to be the last among its siblings. Can be True, False, once'''
        self.plain = plain
        self.tostring = COMPRESSION_STRINGIFY_RESOLVER.resolve(tostring)
        self.scheme = COMPRESSION_SCHEME_RESOLVER.resolve(scheme)
        self.allow_override: bool = allow_override
    
    @ifapplicable
    def on_encounter(self, state:SummaryState, actions):
        self._compress(state)
        
    @ifapplicable
    def on_summarised(self, state:SummaryState, actions):
        self._compress(state)

    def _compress(self, state):
        compressible:SummaryCompressible = state.value
        if compressible.summary_compressible_enable:
            value = compressible.summary_value
            tostring = compressible.summary_compressible_tostring if self.allow_override and hasattr(compressible,'summary_compressible_tostring') and compressible.summary_compressible_tostring is not None else self.tostring
            scheme = compressible.summary_compressible_scheme if self.allow_override and hasattr(compressible,'summary_compressible_scheme') and compressible.summary_compressible_scheme is not None else self.scheme
            data = CompressedEntry(value=value, scheme=scheme, tostring=tostring)
            state.value = data['value'] if self.plain else data
        
    def __repr__(self):
        return f'<{self.__class__.__name__} {"PLAIN" if self.plain else "OBJECT"} {self.tostring.name} {self.scheme.name}>'
        
class GrouppedEntry(SummarisableDict):
    def __init__(self, group, key):
        super().__init__(group=group, key=key)

class GrouppingSummaryVisitor(ClassSummaryVisitor):
    __summary_class__ = SummaryGroupable
    encoding = 'utf8'
    summary_compressible = SummaryCompressible
    
    class Data(Summarisable):
        summary_sibling_priority = -10
        class SummaryGroupData(SummarisableDict): pass
        def __init__(self, visitor):
            self.visitor = visitor
            self.groups = {}
            self.compressors = {}
            self.finalised = False
        def append(self, state:SummaryState):
            groupable = state.value
            group_name = groupable.summary_group_name
            if self.finalised:
                raise RuntimeError(f'Summary filter is finalised, so no more groups can be added.')
            if group_name not in self.groups:
                self.groups[group_name] = []
                if hasattr(groupable,'summary_group_compressible') and groupable.summary_group_compressible is not None:
                    self.compressors[group_name] = groupable.summary_group_compressible
            group = self.groups[group_name] = self.groups.get(group_name, [])
            key = len(group)
            group.append(state.value.summary_value)
            state.value = GrouppedEntry(group=group_name, key=key)
        
        @property
        def compress(self): return self.visitor.compress
        def _compressible_class(self, data, group):
            cls_cmp = self.compressors.get(group,self.visitor.summary_compressible)
            return cls_cmp(data)
            
        def summary(self, summary_options:SummaryOptions):
            self.finalised = True
            if self.compress:
                cls_lst,cls_dct,cls_cmp = SummarisableList, self.SummaryGroupData, self._compressible_class
            else:
                cls_lst,cls_dct,cls_cmp = list, dict, lambda data, group:data
            data = cls_lst(cls_dct(group=group,len=len(data),data=cls_cmp(data, group)) for group,data in self.groups.items())
            return data
        
        def __str__(self):
            sgrp = '['+','.join(f'{name}:{len(group)}' for name,group in self.groups.items())+']'
            return f'{"[FIN]" if self.finalised else "[OK]"} groups:{sgrp}>'
        def __repr__(self):
            return f'<{self.__class__.__name__} {self!s}>'
    
    def __init__(self, compress=False, allow_override:bool = True):
        self.data = None
        self.reset()
        self.compress = compress
        self.allow_override: bool = allow_override
    def reset(self):
        self.data = self.Data(self)

    @ifapplicable
    def on_encounter(self, state:SummaryState, actions):
        if state.value.summary_group_enable:
            self.data.append(state)
    
    def __repr__(self):
        return f'<{self.__class__.__name__} {self.data!s}>'

class JSONDefaultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return str(obj)
    
'''
import gzip
import base64
import numpy as np 

class ValidityCodec:
    
    @staticmethod
    def encode(validity):
        return base64.b64encode(gzip.compress(np.array(validity,bool))).decode('latin')
    
    @staticmethod
    def decode(txt):
        buf = gzip.decompress(base64.decodestring(txt))
        return np.frombuffer(buf, bool)

'''
'''
class SelectorSummariser(ClassSummaryFilter):
    summarise_class = 'Selector'
    
    def apply(self, state:SummaryState):
        selector = state.instance
        records = state.records
        data = selector.language.data
        is_cache_enabled = selector.cache.enabled
        selector.cache.enabled = False
        def evaluate(evaluators, part_check):
            eval_sum = OrderedDict()
            if state.parts & part_check:
                for eval_cls in evaluators.classes:
                    eval_tag = evaluators.get_class_tag(eval_cls)
                    try:
                        eval_inst = eval_cls(data)
                        eval_sum[eval_tag] = eval_inst.evaluate(selector)
                    except Exception as e:
                        log.error(f'While evaluating {eval_tag}: {e}')
                        eval_sum[eval_tag] = nan
            return eval_sum
        records['measures'] = evaluate(MEASURES_DEFAULT_CONSTRUCTIBLE, SummaryParts.SELECTOR_MEASURES)
        records['optimistic-estimators'] = evaluate(OPTIMISTIC_ESTIMATORS_DEFAULT_CONSTRUCTIBLE, SummaryParts.SELECTOR_OPTIMISTIC_ESTIMATES)
        records['cached'] = {k:v for k,v in selector.cache.items()
                             if isinstance(v,float)}
        selector.cache.enabled = is_cache_enabled
        if state.parts & SummaryParts.SELECTOR_VALIDITIES:
            records['validity'] = ValidityCodec.encode(selector.validity)
        
    def summarise(self, selectors) -> SummaryList:
        return SummaryList(map(self.summarise_selector, selectors))
'''

'''
class ScoringfunctionSummariser(ConditionalSummaryFilter):
    def isapplicable(self, state:SummaryState)->bool:
        if not isinstance(state.instance, ProductBundle):
            return False
        return issubclass(state.instance.factory_object, ScoringFunctions)
        
    def apply(self, state:SummaryState):
        fields = ('args','name','digest')
        state.records = OrderedDict(((field,state.records[field]) for field in fields))
'''
        
if __name__ == '__main__':
    
    s = Summariser()
    