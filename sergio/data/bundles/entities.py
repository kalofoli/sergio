'''
Created on Feb 8, 2021

@author: janis
'''

import pandas as pd
import numpy as np
import itertools
from typing import List, Sequence

from colito.summaries import SummarisableFromFields, SummaryFieldsAppend,\
    SummaryConversionsAppend
from sergio.attributes import AttributeInfoFactory
from colito.decorators import with_field_printers

__all__ = ['EntityAttributes', 'EntityAttributesWithArrayTarget', 'EntityAttributesWithAttributeTarget']

class EntityAttributes(SummarisableFromFields):
    '''A collection of vector-entities and edges between them
        :member attribute_info: Either a list of one attribute info per column, or a dict with keys indices or names and values an attribute info specification for the corresponding column. A key of None sets the default attribute kind.
        The attribute info specification may either be an AttributeKind (or a string resolvable to it), or an attrbute info object.  
        :member attribute_selection: Either a list of one boolean per columns, or a dict with keys indices or names and values a boolean for the corresponding column. A key of None sets the default attribute kind. 
    ''' 
    # pylint: disable=too-many-instance-attributes
    __summary_fields__ = ('name', 'attribute_selection', 'attribute_info')
    __summary_conversions__ = {'attribute_selection':str, 'attribute_info':str}

    def __init__(self, attribute_data: pd.DataFrame, name: str,
                 attribute_info={None:None}, attribute_selection={None:True}) -> None:
        '''
        >>> df = pd.DataFrame({'a':np.r_[1,2,3,5],'b':['one','two','three','five'],'c':[3,2,3,2]})
        >>> EntityAttributes(attribute_data= df, name='test', attribute_selection=[1,0,1])
        <EntityAttributes[test](4x2/3)>
        >>> ea = EntityAttributes(attribute_data= df, name='test', attribute_selection={None:False, 'b':1})
        >>> ea
        <EntityAttributes[test](4x1/3)>
        >>> ea.attribute_selection
        array([False,  True, False])
        >>> ea.attribute_selection = {'c':1}
        >>> ea.attribute_selection
        array([False,  True,  True])
        >>> ea.attribute_info
        [<Attribute[NUMERICAL]:a>, <Attribute[CATEGORICAL]:b>, <Attribute[NUMERICAL]:c>]
        >>> ea = EntityAttributes(attribute_data=df, name='test', attribute_info={'a':'index','b':'name'})
        >>> ea.attribute_info
        [<Attribute[INDEX]:a>, <Attribute[NAME]:b>, <Attribute[NUMERICAL]:c>]
        >>> ea.attribute_selection
        array([False, False,  True])
        >>> lu = ea.lookup_attribute('c')
        >>> lu.name, lu.data.sum(), lu.index, lu.selected, lu.info
        ('c', 10, 2, True, <Attribute[NUMERICAL]:c>)
        >>> lu.selected = False
        >>> list(ea.attribute_selection)
        [False, False, False]
        >>> lu.info = 'CATEGORICAL'
        >>> ea.attribute_info
        [<Attribute[INDEX]:a>, <Attribute[NAME]:b>, <Attribute[CATEGORICAL]:c>]
        '''
        super().__init__()
        self._attribute_data: pd.DataFrame = attribute_data
        self._name: str = name
        
        self._attributes_selected = np.ones(self.num_attributes_total, bool)
        self.attribute_selection = attribute_selection
        self._attribute_info = [None]*self.num_attributes_total
        if isinstance(attribute_info, dict):
            defaulted_attribute_info = {None:None}
            defaulted_attribute_info.update(attribute_info)
            self.attribute_info = defaulted_attribute_info
        else:
            self.attribute_info = attribute_info

    def _updated_columns_from_spec(self, spec):
        '''Parse an input of specifiers for each column.
        This can either be a list of elements or a dict with keys indices or names.
        '''
        if isinstance(spec, dict):
            data = self._attribute_data
            if None in spec:
                default_value = spec[None]
                indices = None
                values = [default_value]* self.num_attributes_total
                def set_pair(idx, val):
                    values[idx] = val
            else:
                indices = []
                values = []
                def set_pair(idx, val):
                    indices.append(idx)
                    values.append(val)
            for key, spec in spec.items():
                if key is None:
                    continue
                if isinstance(key, int):
                    idx = key
                else:
                    idx = data.columns.get_loc(key)
                set_pair(idx, spec)
        elif isinstance(spec, (Sequence, np.ndarray)):
            indices = None
            values = tuple(spec)
            if len(values) != self.num_attributes_total:
                raise ValueError(f'Specified {len(values)} inputs instead of the expected {self.num_attributes_total}.')
        else:
            raise TypeError(f'Unsupported input of type {type(spec).__name__}. Must be a dict or sequence.')
        return indices, values
    @property
    def name(self): 
        '''Dataset name'''
        return self._name
    
    @property
    def attribute_data(self): return self._attribute_data
    @property
    def attribute_selection(self): return self._attributes_selected
    @attribute_selection.setter
    def attribute_selection(self, spec):
        indices, values = self._updated_columns_from_spec(spec)
        if indices is None:
            indices = slice(None,None)
        self._attributes_selected[indices] = values
    
    @property
    def attribute_info(self): return self._attribute_info
    @attribute_info.setter
    def attribute_info(self, spec):
        indices,specs = self._updated_columns_from_spec(spec)
        if indices is None:
            indices = range(len(specs))
        aif = AttributeInfoFactory(self.attribute_data)
        for idx,spec in zip(indices, specs):
            ai = aif(idx, spec)
            if ai.__selectable__ is False:
                self._attributes_selected[idx] = False
            self._attribute_info[idx] = ai
            
    @property
    def attribute_names(self):
        return tuple(itertools.compress(self.attribute_data.columns, self.attribute_selection))
        
    num_entities = property(lambda s: s._attribute_data.shape[0], None, 'Number of entities')
    num_attributes_total = property(lambda s: s._attribute_data.shape[1], None, 'Number of all attributes')
    num_attributes_selected = property(lambda s: s._attributes_selected.sum(), None, 'Number of selected attributes')
    
    
    def __repr__(self):
        return (f'<{type(self).__name__}[{self.name}]' 
                f'({self.num_entities}x{self.num_attributes_selected}/{self.num_attributes_total})>')

    def make_predicates(self, prediciser):
        '''Get an iterator of all predicates. Index specifies indices.'''
        attrs = itertools.compress(self.attribute_info, self.attribute_selection)
        predicates = itertools.chain(*map(prediciser.predicates_from_attribute, attrs))
        return predicates

    @with_field_printers(('index','name','data','selected','info'))
    class LookupResult:
        def __init__(self, data, index):
            self._data = data
            self._index = index
        @property
        def index(self): return self._index
        @property
        def name(self): return self._data._attribute_data.columns[self._index]
        @property
        def data(self): return self._data._attribute_data.iloc[:,self._index]
        @property
        def selected(self): return self._data._attributes_selected[self._index]
        @selected.setter
        def selected(self, value): self._data.attribute_selection = {self._index:value}
            
        @property
        def info(self): return self._data._attribute_info[self._index]
        @info.setter
        def info(self, value): self._data.attribute_info = {self._index:value}
        
    def lookup_attribute(self, what) -> LookupResult:
        """Return a set of information about an attribute by index or name"""
        idx = _resolve_dataframe_index(self._attribute_data, what)
        df = self._attribute_data
        return self.LookupResult(self, idx)

def _resolve_dataframe_index(df, what):
    if isinstance(what, (np.int, int)):
        return what
    else:
        return df.columns.get_loc(what)

class EntityAttributesWithTarget(EntityAttributes):
    """Entity attributes with a target property. This is an abstract class.
    """
    __summary_fields__ = SummaryFieldsAppend(('target_name',))
    @property
    def target_data(self):
        """Data for the target property""" 
        raise NotImplementedError('This must be overriden')
    @property
    def target_name(self):
        """Name of the target property""" 
        raise NotImplementedError('This must be overriden')
    @property
    def target_description(self):
        return f'{self.target_name}'
    
    def __repr__(self):
        return (f'<{type(self).__name__}[{self.name}]' 
                f'({self.num_entities}x{self.num_attributes_selected}/{self.num_attributes_total}) target: {self.target_description}>')


class EntityAttributesWithAttributeTarget(EntityAttributesWithTarget):
    """Entity data where the target is one of the entity attributes.
    
    >>> df = pd.DataFrame({'a':np.r_[1,2,3,5],'b':['one','two','three','five'],'c':[3,2,3,2]})
    >>> EntityAttributesWithAttributeTarget(attribute_data=df, target='a', name='test', attribute_selection=[1,0,1])
    <EntityAttributesWithAttributeTarget[test](4x1/3) target: a(int64@0)>
    >>> ea = EntityAttributesWithAttributeTarget(attribute_data=df, name='test', target='a', attribute_selection={None:False, 'b':1})
    >>> ea
    <EntityAttributesWithAttributeTarget[test](4x1/3) target: a(int64@0)>
    """
    __summary_fields__ = SummaryFieldsAppend(('target_index','target_dtype'))
    __summary_conversions__ = SummaryConversionsAppend({'target_dtype':str, 'target_index':int})

    def __init__(self, attribute_data: pd.DataFrame, target, name,
                 attribute_info={None:None}, attribute_selection={None:1}) -> None:
        super().__init__(attribute_data=attribute_data, name=name, attribute_info=attribute_info, 
                         attribute_selection=attribute_selection)
        self._target_index = _resolve_dataframe_index(self.attribute_data, target)
        self.attribute_selection = {self.target_index:False}
    
    @property
    def target_index(self) -> int:
        """Index of the target attribute within the attribute features""" 
        return self._target_index
    @property
    def target_data(self) -> pd.Series:
        """Data for the target property""" 
        return self._attribute_data.iloc[:,self._target_index]
    @property
    def target_dtype(self) -> np.dtype:
        """Data type of target attribute""" 
        return self.target_data.dtype
    @property
    def target_name(self) -> str:
        """Name of the target property""" 
        return self._attribute_data.columns[self._target_index]
    @property
    def target_description(self) -> str:
        return f'{self.target_name}({self.target_dtype}@{self.target_index})'
    
class EntityAttributesWithArrayTarget(EntityAttributesWithTarget):
    """Entity data where the target is a numpy array with one row per entry.
    
    >>> df = pd.DataFrame({'a':np.r_[1,2,3,5],'b':['one','two','three','five'],'c':[3,2,3,2]})
    >>> t = np.c_[[10,2,3,4.],[4,4,4,4],[12,6,7,8.]]
    >>> EntityAttributesWithArrayTarget(attribute_data=df, target=t, target_name='data', name='test', attribute_selection=[1,0,1])
    <EntityAttributesWithArrayTarget[test](4x2/3) target: data(3d float64)>
    >>> ea = EntityAttributesWithArrayTarget(attribute_data=df, name='test', target=t[:,:2].astype(int), attribute_selection={None:False, 'b':1})
    >>> ea
    <EntityAttributesWithArrayTarget[test](4x1/3) target: target(2d int64)>
    """
    __summary_fields__ = SummaryFieldsAppend(('target_ndim','target_dtype'))
    __summary_conversions__ = SummaryConversionsAppend({'target_dtype':str})
    
    def __init__(self, attribute_data: pd.DataFrame, name: str, target:np.ndarray, target_name:str = 'target',
                 attribute_info={None:None}, attribute_selection={None:1}) -> None:
        super().__init__(attribute_data=attribute_data, name=name, attribute_info=attribute_info, 
                         attribute_selection=attribute_selection)
        target = np.array(target)
        if target.ndim == 1:
            target = target[:,None]
        if target.shape[0] != self.num_entities:
            raise ValueError(f'The target array must hav one row per entity, but has only {target.shape[0]} rows instead of {self.num_entities}.') 
        self._target_data = target
        self._target_name = target_name

    @property
    def target_data(self) -> np.ndarray:
        """Data for the target property""" 
        return self._target_data
    @property
    def target_ncols(self) -> int:
        """Dimensions of the target array""" 
        return self.target_data.shape[1]
    @property
    def target_dtype(self) -> np.dtype:
        """Data type of the target array""" 
        return self.target_data.dtype
    @property
    def target_name(self) -> str:
        """Name of the target property""" 
        return self._target_name
    @property
    def target_description(self) -> str:
        return f'{self.target_name}({self.target_ncols}d {self.target_dtype})'



if __name__ == '__main__':
    import doctest
    doctest.testmod()
