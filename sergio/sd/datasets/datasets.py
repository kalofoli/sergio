'''
Created on Dec 18, 2017

@author: janis
'''

from enum import Enum, auto
import re

from pandas import read_csv
from numpy import arange

from cache import DEFAULT_CACHE, Cache
from data import GraphData
from typing import Union, cast
from predicates import Prediciser, DEFAULT_PREDICISER
from discretisers import FrequencyDiscretiser, DiscretiserRanges,\
    Discretiser
from discretisers import DiscretiserUniform
from factory import factorymethod, factorygetter, FactoryGetter,\
    FactoryBase, FactoryDescription
import math
import os.path

class DatasetFactory(FactoryBase):
    '''Loader for various datasets'''

    def __init__(self, prediciser:Prediciser = DEFAULT_PREDICISER, cache: Cache=DEFAULT_CACHE, data_path='data') -> None:
        self._cache = cache
        self._prediciser = prediciser
        self._data_path = data_path
        
    prediciser = property(lambda self:self._prediciser, None, 'Prediciser to use on the created graph-data.')
    data_path = property(lambda self:self._data_path, None, 'Path on which the datasets are searched.')
    
    def get_path(self, what):
        if os.path.isabs(what):
            return what
        else:
            return os.path.join(self.data_path, what)
    
    @factorymethod('petsters')
    def load_petsters(self,mode:str='full') -> GraphData:
        '''Load petster dataset'''
        entities = read_csv(self.get_path('Petsters/entities-petsters.csv'))
        edges = read_csv(self.get_path('Petsters/edges-petsters.csv'))
        
        
        # entities.dropna(inplace=True)
        
        dtypes = {'weight': 'float', 'sex': 'category',
                  'species': 'category'}
        idl_race = entities.columns.str.match('^(rac|stt|cnt):.*')
        dtypes.update({column:'bool' for column in entities.columns[idl_race]})
        entities.loc[:,idl_race] = entities.loc[:,idl_race].fillna(0)
        entities = entities.astype(dtypes, inplace=True)

        kinds = {'weight': 'numerical', 'sex': 'categorical', 'species': 'categorical'}
        
        selection = {'species': mode=='full'}
        # edges_dense = GraphData.Tools.filter_edges(entities_selected=entities.index, edges=edges, remap=True)
        data = GraphData(entities=entities, edges=edges, name='petsters',prediciser=self.prediciser)
        data.attribute_kinds = kinds
        data.attribute_selection = selection

        if mode == 'full':
            data_name = 'petsters:full'
        else:
            if mode == 'cats':
                entities_sel = data.entities.species == 'Cat'
                data_name = 'petsters:cats'
            elif mode == 'dogs':
                entities_sel = data.entities.species == 'Dog'
                data_name = 'petsters:dogs'
            else:
                raise ValueError('mode must be one of: full, dogs, cats')
            data = data.select_entities(entities_sel)
            data = data.drop_identity_columns()
            data.name = data_name
        return data

    @factorymethod('dblp')
    def load_dblp(self, mode:str='small') -> GraphData:
        '''Load dblp from a folder'''
        edges = read_csv(self.get_path(f'DBLP/dblp_{mode}_auth2auth_edges'),names=['source','target'],skiprows=1)
        kinds = {'author': 'index'}
        selection = {'author': False}
        entities = read_csv(self.get_path(f'DBLP/dblp_{mode}_auth_entities'))
        entities.iloc[:,1:] = entities.iloc[:,1:].astype(bool)
        edges_dense = GraphData.Tools.filter_edges(entities_selected=entities.index, edges=edges, remap=True)
        data = GraphData(entities=entities, edges=edges_dense, name=f'DBLP:auth2auth:{mode}',prediciser=self.prediciser)
        data.attribute_kinds = kinds
        data.attribute_selection = selection
        return data

    @factorymethod('imdb')
    def load_imdb(self, mode:str='bool',cuts:int=3) -> GraphData:
        '''Load imdb from a folder'''
        get_file = lambda x:self.get_path(f'IMDB/{x}.csv')
        edges = read_csv(get_file('name2name_edges'),names=['source','target'],skiprows=1)
        kinds = {'person': 'index'}
        selection = {'person': False}
        entities = read_csv(get_file(f'name_feats-{mode}'))
        edges_dense = GraphData.Tools.filter_edges(entities_selected=entities.index, edges=edges, remap=True)
        if isinstance(cuts, str):
            cuts = int(cuts)
        data = GraphData(entities=entities, edges=edges_dense, name=f'IMDB:name2name:{mode}',prediciser=self.prediciser)
        data.attribute_kinds = kinds
        data.attribute_selection = selection
        return data

    @factorymethod('gattwto')
    def load_gattwto(self, years=1980) -> GraphData:
        '''Load GATT/WTO data from a folder'''
        get_file = lambda x:self.get_path(f'GATTWTO/{x}-{years}.csv')
        edges = read_csv(get_file(f'edges'),names=['source','target'],skiprows=1)
        kinds = {'cty': 'index'}
        selection = {'cty': False}
        entities = read_csv(get_file(f'entities'))
        edges_dense = GraphData.Tools.filter_edges(entities_selected=entities.index, edges=edges, remap=True)
        data = GraphData(entities=entities, edges=edges_dense, name=f'GATTWTO:{years}',prediciser=self.prediciser)
        data.attribute_kinds = kinds
        data.attribute_selection = selection
        return data

    @factorymethod('amazon')
    def load_amazon(self, category,mode) -> GraphData:
        '''Load Amazon customer meta data from a folder'''
        get_file = lambda x:self.get_path(f'Amazon/{x}-{category}.csv')
        edges = read_csv(get_file(f'edges'),names=['source','target'],skiprows=1)
        kinds = {'customer': 'index'}
        selection = {'customer': False}
        entities = read_csv(get_file(f'entities'))
        if mode == 'num':
            pass
        elif mode=='bool':
            numeric = {'customer','votes','rating','helpful'}
            columns = list(c for c in entities.columns if c not in numeric)
            entities.loc[:,columns] = entities.loc[:,columns].astype(bool)
        else:
            raise NotImplementedError('Choose one of: [num, bool]')
        # edges_dense = GraphData.Tools.filter_edges(entities_selected=entities.index, edges=edges, remap=True)
        entities = entities.iloc[:edges.max().max()+1,:]
        data = GraphData(entities=entities, edges=edges, name=f'Amazon:{category}:{mode}',prediciser=self.prediciser)
        data.attribute_kinds = kinds
        data.attribute_selection = selection
        return data

    @factorymethod('delicious')
    def load_delicious(self, tags:float=math.inf, mode:str='num') -> GraphData:
        '''Load the user-user Delicious data with a given tag count'''
        import gzip
        get_file = lambda x:gzip.open(self.get_path(f'Delicious/{x}.csv.gz'),'rb')
        edges = read_csv(get_file(f'edges'),names=['source','target'],skiprows=1)
        kinds = {}
        selection = {}
        entities = read_csv(get_file(f'entities'))
        if math.isinf(tags):
            tags = entities.shape[1]
        else:
            tags = int(tags)
        entities = entities.iloc[:,0:tags]
        if mode == 'bool':
            entities = entities.astype(bool)
        elif mode != 'num':
            raise ValueError(f'Mode can only be num or bool')

        edges_dense = GraphData.Tools.filter_edges(entities_selected=entities.index, edges=edges, remap=True)
        data = GraphData(entities=entities, edges=edges_dense, name=f'Delicious:{tags}:{mode}',prediciser=self.prediciser)
        data.attribute_kinds = kinds
        data.attribute_selection = selection
        return data

    @factorymethod('lastfm')
    def load_lastfm(self, kind:str, mode:str='num') -> GraphData:
        '''Load the user-Last.fm data with a given tag count'''
        import gzip
        get_file = lambda x:gzip.open(self.get_path(f'Lastfm/{x}-{kind}.csv.gz'),'rb')
        edges = read_csv(get_file(f'edges'),names=['source','target'],skiprows=1)
        kinds = {}
        selection = {}
        entities = read_csv(get_file(f'entities'))
        if mode == 'bool':
            entities = entities.astype(bool)
        elif mode != 'num':
            raise ValueError(f'Mode can only be num or bool')

        edges_dense = GraphData.Tools.filter_edges(entities_selected=entities.index, edges=edges, remap=True)
        data = GraphData(entities=entities, edges=edges_dense, name=f'Lastfm:{kind}:{mode}',prediciser=self.prediciser)
        data.attribute_kinds = kinds
        data.attribute_selection = selection
        return data
    
    @factorymethod('lastfm-hetrec')
    def load_lastfm_hetrec(self, tags:float=math.inf, mode:str='num') -> GraphData:
        '''Load the user-user Lastfm data with a given tag count'''
        import gzip
        get_file = lambda x:gzip.open(self.get_path(f'Lastfm/{x}-hetrec.csv.gz'),'rb')
        edges = read_csv(get_file(f'edges'),names=['source','target'],skiprows=1)
        kinds = {}
        selection = {}
        entities = read_csv(get_file(f'entities'))
        if math.isinf(tags):
            tags = entities.shape[1]
        else:
            tags = int(tags)
        entities = entities.iloc[:,0:tags]
        if mode == 'bool':
            entities = entities.astype(bool)
        elif mode != 'num':
            raise ValueError(f'Mode can only be num or bool')

        edges_dense = GraphData.Tools.filter_edges(entities_selected=entities.index, edges=edges, remap=True)
        data = GraphData(entities=entities, edges=edges_dense, name=f'Lastfm-HR:{tags}:{mode}',prediciser=self.prediciser)
        data.attribute_kinds = kinds
        data.attribute_selection = selection
        return data
    
    @factorymethod('ego')
    def load_ego(self, kind:str='facebook',tags:float=math.inf) -> GraphData:
        '''Load the user-user Lastfm data with a given tag count'''
        import gzip
        get_file = lambda x:gzip.open(self.get_path(f'Ego/{x}-{kind}.csv.gz'),'rb')
        edges = read_csv(get_file(f'edges'),names=['source','target'],skiprows=1)
        kinds = {}
        selection = {}
        entities = read_csv(get_file(f'entities'))
        if math.isinf(tags):
            tags = entities.shape[1]
        else:
            tags = int(tags)
        entities = entities.iloc[:,0:tags]
        entities = entities.astype(bool)

        edges_dense = GraphData.Tools.filter_edges(entities_selected=entities.index, edges=edges, remap=True)
        data = GraphData(entities=entities, edges=edges_dense, name=f'Ego:{kind}:{tags}',prediciser=self.prediciser)
        data.attribute_kinds = kinds
        data.attribute_selection = selection
        return data
    
    _get_dataset = FactoryGetter(classmethod=False)
    
    def __call__(self, text) -> GraphData:
        '''Load a dataset by name, optionally using cache'''
        bundle = self._get_dataset(text,produce=False)
        hint = f'dataset-{bundle.digest}'.replace(':','_')
        loader = lambda : bundle.produce(self)
        data = self._cache.load(tag=hint, loader=loader)
        data.prediciser = self.prediciser
        return data
    
    load_dataset = __call__
    
    description = FactoryDescription()



if __name__ == '__main__':
    
    df = DatasetFactory()
    
    df.load_dataset('amazon:movies,num')
    

