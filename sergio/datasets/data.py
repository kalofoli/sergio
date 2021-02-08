'''
Created on Feb 8, 2021

@author: janis
'''

from types import SimpleNamespace
from pandas import DataFrame, Series
from numpy import ndarray, arange, zeros

from colito.summarisable import Summarisable
from sergio.predicates import Prediciser, DEFAULT_PREDICISER


class Relations(SimpleNamespace):
    pass
class Structures(SimpleNamespace):
    pass

class EntitiesWithStructures(Summarisable):
    '''A collection of vector-entities and edges between them'''

    # pylint: disable=too-many-instance-attributes

    def __init__(self, entities: DataFrame, structures,name: str='unknown',
                 attribute_kinds=None, attribute_selection=None,
                 prediciser:Prediciser=DEFAULT_PREDICISER) -> None:
        super().__init__()
        self._structures = structures
        self._entities: DataFrame = entities
        self._name: str = name
        self._prediciser = prediciser
        
        self._attribute_kinds: List[AttributeKind] = [None] * self.ncols
        self.attribute_kinds = ['auto'] * self.ncols
        self._attribute_selection: List[bool] = [True] * self.ncols
        if attribute_kinds is not None:
            self.attribute_kinds = attribute_kinds
        if attribute_selection is not None:
            self.attribute_selection = attribute_selection


    '''
    Private class helpers
    '''

    @classmethod
    def _infer_attribute_kind(cls, series: Series, hint: KindType='auto'):
        kind : AttributeKind
        if hint == 'auto':
            kind = AttributeFactory.infer_kind_from_series(series)
        else:
            kind = AttributeFactory.lookup_kind(hint)
        return kind

    '''
    Class helpers
    '''

    def get_indices(self, which=None, fun=None, collapse=True, selection=False):
        '''Normalise index and name collections into (a list of) indices'''
        if not isinstance(which, str) and isinstance(which, Iterable):
            indices = list(self.get_indices(w, fun=fun, collapse=True) for w in which)
        elif which is None:
            indices = range(self.ncols)
            if selection:
                indices = compress(indices, self.attribute_selection)
            if fun is not None:
                indices = map(fun, indices)
            indices = list(indices)
        else:
            if isinstance(which, str):
                index = self._attr_name2index[which]
            elif isinstance(which, int):
                index = which
            else:
                raise IndexError("Cannon get index of: {0}".format(which))
            indices = Utils.wrap(index, allow_single=collapse, container=list, fun=fun)
        return indices

    def get_names(self, which, collapse=True, selection=False):
        index = self.get_indices(which, collapse=collapse, selection=selection)
        res = self._entities.columns[index]
        if len(res) == 1:
            res = res[0]
        return res

    def get_series(self, which=None, collapse=True, selection=False):
        idx = self.get_indices(which, collapse=collapse, selection=selection)
        vals = self._entities.iloc[:, idx]
        return vals

    '''
    Properties
    '''

    @property
    def name(self):
        '''Data name'''
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def entities(self):
        '''Entities data'''
        return self._entities

    @property
    def edges(self) -> ndarray:
        '''Return the edges as a numpy adjacency list of dimensions Ex2'''
        return self._edges.values

    @property
    def ncols(self):
        '''Number of columns/attributes'''
        return len(self._entities.columns)

    @property
    def nrows(self):
        '''Number of rows/entities'''
        return len(self._entities)

    prediciser = property(lambda self:self._prediciser, None, 'The prediciser used to create the predicates from the attributes of this dataset.')
    
    @prediciser.setter
    def prediciser(self, value):
        self._prediciser = value
    
    @property
    def attribute_kinds(self):
        '''Kinds for each attribute. Assignable to a dict for selective update.'''
        return self._attribute_kinds.copy()

    @attribute_kinds.setter
    def attribute_kinds(self, kinds: KindCollection):
        attr_kinds_new: List[AttributeKind] = self._attribute_kinds.copy()
        indices: List[int]
        kind_hints: List[KindType]
        if isinstance(kinds, dict):
            indices = self.get_indices(kinds.keys(), collapse=False)
            kind_hints = list(kinds.values())
        else:
            indices = self.get_indices(None)
            kind_hints = cast(List[KindType], kinds)
        for index, kind_hint in zip(indices, kind_hints):
            series = self.get_series(index)
            attr_kind = self._infer_attribute_kind(series, kind_hint)
            attr_kinds_new[index] = attr_kind
        self._attribute_kinds = attr_kinds_new

    @property
    def attribute_selection(self):
        '''Select which attributes should be included for predicate creation.'''
        return self._attribute_selection.copy()

    @attribute_selection.setter
    def attribute_selection(self, selection: Dict[IndexType, bool]):
        attr_sel_new: List[bool] = self._attribute_selection.copy()
        indices: List[int]
        indices_selected: List[bool]
        if isinstance(selection, dict):
            indices = self.get_indices(selection.keys(), collapse=False)
            indices_selected = list(selection.values())
        else:
            indices = self.get_indices(None)
            indices_selected = cast(List[bool], selection)
        for index, sel in zip(indices, indices_selected):
            attr_sel_new[index] = sel
        self._attribute_selection = attr_sel_new

    '''
    methods
    '''
    
    def add_attributes(self, attributes:DataFrame, kinds:Union[KindType, Iterable[KindType]]=None, selected:ndarray=True):
        import pandas as pd
        if self.nrows != attributes.shape[0]:
            raise ValueError(f'Added attributes must have {self.nrows} rows, one per entity.')
        
        num_attrs = attributes.shape[1]
        if isinstance(kinds, (str,AttributeKind)):
            kinds = [kinds]*num_attrs
        attribute_series = (attributes.iloc[:,i] for i in range(num_attrs))
        if kinds is None:
            kinds = list(map(self._infer_attribute_kind, attribute_series))
        elif isinstance(kinds, Iterable):
            if len(kinds) != num_attrs:
                raise ValueError(f'Kinds must be provided for {num_attrs} attributes but {len(kinds)} given.')
            kinds = list(self._infer_attribute_kind(series=s, hint=k) for s,k in zip(attribute_series,kinds))
        else:
            raise TypeError(f'Type {type(kinds)} s not a valid kind collection. Use a string/AttributeKind or an iterable of them.')
        if not isinstance(selected, Iterable):
            selected = [selected]*num_attrs
        if len(selected)!= num_attrs:
            raise ValueError(f'Selection must be provided for {num_attrs} attributes but {len(selected)} given.')
        selected = list(selected)
        def test_bool(entry):
            i, s = entry
            if not isinstance(s, bool):
                raise TypeError(f'Selection {i} is of type {type(s).__name__} instead of bool.')
        tuple(map(test_bool, enumerate(selected)))
        entities_all = pd.concat((self.entities,attributes),1)
        selected_all = self.attribute_selection + selected
        kinds_all = self.attribute_kinds + kinds
        self._entities = entities_all
        self._attribute_selection = selected_all
        self._attribute_kinds = kinds_all
        self._reset_name_index_map()
        return attributes,kinds,selected
    
    def get_attributes(self, which=None, collapse=False, selection: bool=False) -> List[Attribute]:
        '''Create Attribute instances describing the given indices'''

        def mk_attribute(index: int) -> Attribute:
            '''Make an attribute from a series index'''
            kind = self.attribute_kinds[index]
            attribute = AttributeFactory.make(self, index, kind)
            return attribute

        attrs = self.get_indices(which, collapse=collapse, selection=selection, fun=mk_attribute)
        return attrs

    def get_predicates(self, which: IndexType=None, selection=True) -> Iterator['Predicate']:
        '''Get an iterator of all predicates. Index specifies indices.'''
        attrs = self.get_attributes(which, selection=selection)
        predicates = chain(*map(self.prediciser.predicates_from_attribute, attrs))
        return predicates

    def select_entities(self, entity_loc, name: Optional[str]=None) -> 'GraphData':
        '''Select a subset of entities (rows) into a new GraphData instance'''
        sub_entities = self._entities.loc[entity_loc]
        edges_locb = self._edges.isin(sub_entities.index).all(axis=1)
        sub_edges = self._edges.loc[edges_locb]
        GraphData.Tools.remap_edges(id_map=sub_entities.index, edges=sub_edges, inplace=True)
        if name is None:
            name = self.name + "_selection"
        gdata = GraphData(sub_entities, sub_edges, name=name)
        gdata.attribute_kinds = self.attribute_kinds
        gdata.attribute_selection = self.attribute_selection
        return gdata
    
    def asgraph(self, attributes: IndexType=None, selection: bool=False) -> Graph:
        graph = Graph(directed=False)
        graph.add_edge_list(self.edges)
        graph.add_vertex(self.nrows-graph.num_vertices())
        indices = self.get_indices(which=attributes, collapse=False, selection=selection)
        for index in indices:
            series = self._entities.iloc[:,index]
            gttype = GraphData.lookup_gt_property_dtype(series)
            graph.vp[series.name] = graph.new_vertex_property(gttype, vals=series)
        return graph
        
    def drop_identity_columns(self,selection=None, columns=None, ignore_na=False):
        import numpy as np
        def isequal(s):
            idl = s.isna()
            idl_rate = idl.mean()
            if idl_rate == 1:
                return True
            elif idl_rate != 0:
                if ignore_na:
                    s = s[~idl]
                else:
                    return False
            return np.all(s == next(iter(s)))
        
        if columns is not None and selection is not None:
            raise ValueError('At most one column specifier must be provided.')
        if selection is not None:
            columns = self.entities.columns[selection]
        if columns is None:
            columns = self.entities.columns
        idl_equal = self.entities.loc[:,columns].apply(isequal, axis=0, reduce=True)
        bad_cols = columns[idl_equal]
        entities = self.entities.drop(bad_cols,axis=1)
        attribute_kinds = [kind for index,kind in enumerate(self.attribute_kinds) if not idl_equal[index]]
        attribute_selection = [sel for index,sel in enumerate(self.attribute_selection) if not idl_equal[index]]
        data = GraphData(entities=entities, edges=self._edges, name=self.name,
                         attribute_kinds=attribute_kinds, attribute_selection=attribute_selection)
        return data
    
    def __repr__(self):
        return (f'<{self.__class__.__name__}[{self.name}]' 
                f'({self.entities.shape[0]}x{self.entities.shape[1]})'
                f' with prediciser: {self.prediciser!r}>')

    def __str__(self):
        return (f'["{self.name}"({self.entities.shape[0]}x{self.entities.shape[1]}) PRED:{self.prediciser!s}]')

    def summary_dict(self, options):
        fields = ('name','nrows','ncols')
        dct = self.summary_from_fields(fields)
        dct['prediciser'] = self.prediciser
        return dct
    
class Tools:

    @classmethod
    def remap_edges_array(cls, id_map: ndarray, edges: ndarray, inplace: Optional[bool]=False) -> ndarray:
        '''Apply an index remapping to an edges DataFrame'''
        id_lut = zeros(max(id_map) + 1)
        id_lut[id_map] = arange(len(id_map))
        
        if inplace:
            edges_out = edges
        else:
            edges_out = edges.copy()
        edges_out[:] = id_lut[edges]
        return edges_out

    @classmethod
    def remap_edges(cls, id_map: ndarray, edges: DataFrame, inplace: Optional[bool]=False) -> DataFrame:
        '''Apply an index remapping to an edges DataFrame'''
        edges_out = cls.remap_edges_array(id_map=id_map, edges=edges.values, inplace=inplace)
        return edges_out
    
    @classmethod
    def filter_edges_array(cls, entities_selected: Union[ndarray, DataFrame, pandas.Index, Series],
                           edges: ndarray, edge_base:int=0, remap=False) -> DataFrame:
        from numpy import isin
        np_entities: ndarray
        if isinstance(entities_selected, (DataFrame, Series, pandas.Index)):
            np_entities = entities_selected.values
        else:
            np_entities = entities_selected
        if np_entities.dtype == bool:
            assert edges.max() < len(np_entities), 'Boolean selector shorted than maximal edge index'
            entities_idx = arange(edge_base,len(np_entities)+edge_base)[np_entities]
        elif np_entities.dtype == int:
            entities_idx = np_entities
        else:
            raise TypeError('Array dtype {0} not supported.'.format(np_entities.dtype))
        
        edges_selected = isin(edges, entities_idx).all(axis=1)
        edges_out = edges[edges_selected, :]
        if remap:
            cls.remap_edges_array(id_map=entities_idx, edges=edges_out, inplace=True)
        return edges_out

    @classmethod
    def filter_edges(cls, entities_selected: Union[ndarray, DataFrame, pandas.Index, Series],
                     edges: DataFrame, edge_base:int=0, remap=False) -> DataFrame:
        edge_values = cls.filter_edges_array(entities_selected=entities_selected,
                                             edges=edges.values, edge_base=edge_base,remap=remap)
        edges_out = DataFrame(edge_values, columns=edges.columns)
        return edges_out
