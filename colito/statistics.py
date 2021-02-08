
import operator
from typing import Dict, Callable, Sequence # pylint: disable=unused-import
from .summarisable import SummarisableAsDict, SummaryOptions
from collections import OrderedDict


_template_counter = '''
def getter(self):
    return self.{name_private}

def setter(self, value):
    self.{name_private} = value
'''

_template_updater = '''
def updater(self, steps:int=1):
    value = getter(self)
    new_value = fun(value,steps*step)
    setter(self, new_value)
    return self
'''

class StatisticsUpdater:
    '''StatisticsUpdater'''

    def __init__(self, counter: 'StatisticsCounter', step=1, name=None, fun=operator.iadd, doc=None) -> None:
        self.counter: StatisticsCounter = counter
        self.step = step
        self.fun = fun
        self._updater = None
        self.name = name
        self.doc = doc

    def __repr__(self):
        return f'<StatisticsUpdater for {self.counter} with step {self.step} and function {self.fun}>'
    
    @property
    def updater(self):
        if self._updater is None:
            counter = self.counter
            base = self.name
            nm = {'setter':counter.setter, 'getter':counter.getter,'fun':self.fun,'step':self.step}
            exec(_template_updater, nm)
            self._updater = nm['updater']
            if isinstance(base, str):
                self._updater.__name__ = base
            if self.doc is not None:
                self._updater.__doc__ = self.doc
        return self._updater


class StatisticsCounter:

    def __init__(self, initial_value=0, name=None, doc=None):
        self.initial_value = initial_value
        self.name = name
        self.doc = doc
        name_private = '_' + self.name
        nm = {}
        exec(_template_counter.format(name_private=name_private),nm)
        self._getter = nm['getter']
        self._setter = nm['setter']


    def __repr__(self):
        return f'<StatisticsCounter for "{self.name if self.name is not None else "<unnamed>"}" with initial value {self.initial_value}>'
    
    @property
    def getter(self):
        return self._getter
        
    @property
    def setter(self):
        return self._setter

class _DecoratedUpdater:
    def __init__(self, fun):
        self.fun = fun

def updater(counter: 'StatisticsCounter', step=1, fun=operator.iadd):

    def decorate(decoratee):
        update_fn = _DecoratedUpdater(decoratee) if fun is None else fun
        return StatisticsUpdater(counter=counter, step=step, fun=update_fn, name=decoratee.__name__, doc=decoratee.__doc__)

    return decorate

class _StatisticsMeta(type):
    counters_attr_name = '__statistics_counters__'
    updaters_attr_name = '__statistics_updaters__'
        
    
    @staticmethod
    def _validate_names(objs, tag):
        for name, obj in objs.items():
            if obj.name is None:
                obj.name = name
            elif obj.name != name:
                raise ValueError(f'{tag} Entry for key {name} has mismatching name {obj.name}.')
        
    @classmethod
    def _update_user_objects(mcs, cls, attr_name, objs_memb):
        attr_val = getattr(cls, attr_name)
        if isinstance(attr_val, dict):
            mcs._validate_names(attr_val, f'Name mismatch for attribute {attr_name}.')
            objs_attr = attr_val.values()
        else:
            for obj in attr_val:
                if obj.name is None:
                    raise KeyError(f'{obj.__class__.__name__} specified in the {attr_name} field must have a name.')
            objs_attr = attr_val
        for obj in objs_attr:
            if obj.name in objs_memb:
                raise KeyError(f'{obj.__class__.__name__} specified in {attr_name} clashes with member of same name.')
            objs_memb[obj.name] = obj
    
    def __new__(cls, cls_name, bases, dct):
        cls_new = super(_StatisticsMeta, cls).__new__(cls, cls_name, bases, dct)
        counters = OrderedDict(filter(lambda x:isinstance(x[1], StatisticsCounter), dct.items()))
        updaters = OrderedDict(filter(lambda x:isinstance(x[1], StatisticsUpdater), dct.items()))
        # Make a property with just a getter
        cls._update_user_objects(cls_new, cls.counters_attr_name, counters)
        cls._update_user_objects(cls_new, cls.updaters_attr_name, updaters)
        cls._validate_names(counters, 'Name mismatch for specified counter.')
        cls._validate_names(updaters, 'Name mismatch for specified updater.')
        for name, updater in updaters.items():
            if isinstance(updater.counter,str):
                cname = updater.counter
                if cname not in counters:
                    counters[cname] = StatisticsCounter(name=cname)
                updater.counter = counters[cname]
            elif isinstance(updater.counter, StatisticsCounter):
                counter = updater.counter
                if counter.name not in counters:
                    counters[counter.name] = counter
                elif counters[counter.name] is not counter:    
                    raise ValueError(f'Counter {counter.name} already defined.')
            else:
                raise TypeError(f'Unknown counter type {updater.counter.__class__} for updater {updater}')
            if updater.name is None:
                updater.name = name
            setattr(cls_new, name, updater.updater)
        for name, counter in counters.items():
            prop = property(counter.getter)
            if counter.doc is not None:
                prop.__doc__ = counter.doc
            setattr(cls_new, name, prop)
        setattr(cls_new, cls.counters_attr_name, counters)
        setattr(cls_new, cls.updaters_attr_name, updaters)
        return cls_new

    def __call__(self, *args, **kwargs):
        inst = super().__call__(*args, **kwargs)
        for counter in getattr(inst, self.counters_attr_name).values():
            counter.setter(inst, counter.initial_value)
        return inst


class StatisticsBase(SummarisableAsDict, metaclass=_StatisticsMeta):
    __statistics_counters__: Dict[str,StatisticsCounter] = ()
    __statistics_updaters__: Dict[str, StatisticsUpdater] = ()
    
    def __repr__(self):
        text = ' '.join(map(lambda name:'{0}:{{0.{0}}}'.format(name).format(self), self.__statistics_counters__.keys()))
        return f'<{self.__class__.__name__}: {text}>'
    
    def reset(self):
        cls: Type[StatisticsBase] = self.__class__
        counters: Dict[str, StatisticsStatisticsCounter] = cls.__statistics_counters__
        for counter in counters.values():
            name_private = '_' + counter.name
            setattr(self, name_private, counter.initial_value)
        return self
    
    def _asdict(self):
        return {name:counter.getter(self) for name,counter in self.__statistics_counters__.items()}
    
    def summary_dict(self, options:SummaryOptions): # pylint: disable=unused-argument
        return self.summary_from_fields(self.__statistics_counters__)

    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f'Can only add same-class statistics and you are adding a {other.__class__.__name__} object.')
        for name, counter in self.__statistics_counters__.items():
            # StatisticsCounters are class-level objects. Therefore, they need to be provided the instance to function
            val_self = counter.getter(self)
            val_other = other.__statistics_counters__[name].getter(other)
            counter.setter(self, val_self + val_other)
        return self
    
    def __add__(self, other):
        instance = self.__class__()
        instance += self
        instance += other
        return instance

    def __isub__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f'Can only add same-class statistics and you are adding a {other.__class__.__name__} object.')
        for name, counter in self.__statistics_counters__.items():
            # StatisticsCounters are class-level objects. Therefore, they need to be provided the instance to function
            val_self = counter.getter(self)
            val_other = other.__statistics_counters__[name].getter(other)
            counter.setter(self, val_self - val_other)
        return self
    
    def __sub__(self, other):
        instance = self.__class__()
        instance += self
        instance -= other
        return instance

    def copy(self):
        return self.__class__() + self

    class _Context:
        def __init__(self, stats):
            self._stats = stats
            self._stats0 = stats.copy()
            self.stats = stats.__class__()
        def __enter__(self):
            return self.stats

        def __exit__(self, exc_type, exc_val, exc_tb):
            diff = self._stats - self._stats0
            self.stats += diff
    def block(self):
        return self._Context(self)
                
class MixinStatisticsBase(StatisticsBase):
    pass

_template_merge = '''
class {name}(StatisticsBase):
    __statistics_counters__ = counters
    __statistics_updaters__ = updaters
'''
def merge_statistics(statistics, name):
    '''Combine the MixinStatisticsBase inheriting classes of the same name along the base hierarchy.'''
    counters = {}
    updaters = {}
    for statistic in statistics:
        updaters.update(statistic.__statistics_updaters__)
        counters.update(statistic.__statistics_counters__)
    nm = {'counters':counters, 'updaters':updaters,'StatisticsBase':StatisticsBase}
    exec(_template_merge.format(name=name), nm)
    cls = nm[name]
    return cls

def merge_mixin_statistics(cls):
    '''Merge the MixinStatisticsBase inheriting classes of the same name along the base hierarchy.'''
    dct = cls.__dict__
    statistics = dict(filter(lambda x:isinstance(x[1], type) and issubclass(x[1],MixinStatisticsBase), dct.items()))
    merge_classes = []
    for name in statistics:
        for base in cls.__bases__:
            if hasattr(base, name):
                merge_classes.append(getattr(base,name))
        stats_cls = merge_statistics(merge_classes, name)
        stats_cls.__qualname__ = f'{cls.__qualname__}.{name}'
        setattr(cls, name, stats_cls)

class _MixinStatisticsMeta(type):
    def __new__(cls, cls_name, bases, dct):
        cls_new = super().__new__(cls, cls_name, bases, dct)
        merge_mixin_statistics(cls_new)
        return cls_new

class MixinStatisticsContainer(metaclass=_MixinStatisticsMeta):
    pass


    

if __name__ == '__main__':

    class A:
        class Statistics(MixinStatisticsBase):
            __statistics_updaters__ = (StatisticsUpdater('push',1,'increase_push'), StatisticsUpdater('pop',-1,'decrease_pop'))
            pop = StatisticsCounter('pop')
    class B:
        class Statistics(MixinStatisticsBase):
            increase_create = StatisticsUpdater('create', 1, doc='creating stuff')
    
    class M(A,B, MixinStatisticsContainer):
        class Statistics(MixinStatisticsBase):pass

    s = M.Statistics()
    print(s)
    
