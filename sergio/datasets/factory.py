'''
Created on Dec 18, 2017

@author: janis
'''

from enum import Enum, auto
import re

from pandas import read_csv
from numpy import arange
import math
import os.path

from colito.factory import factorymethod, factorygetter, FactoryGetter,\
    FactoryBase, FactoryDescription

from colito.cache import DEFAULT_FILE_CACHE, FileCache
from typing import Union, cast
from sergio.predicates import Prediciser, DEFAULT_PREDICISER
from sergio.discretisers import FrequencyDiscretiser, DiscretiserRanges,\
    Discretiser, DiscretiserUniform

from types import SimpleNamespace


class DataBundle:
    class Relations(SimpleNamespace): pass
    class Structures(SimpleNamespace): pass
    def __init__(self, relations, structures, name):
        self._name = name
        self.rels = self.Relations(**relations)
        self.structs = self.Structures(**structures)

class DatasetFactory(FactoryBase):
    '''Loader for various datasets'''

    def __init__(self, prediciser:Prediciser = DEFAULT_PREDICISER, cache: FileCache=DEFAULT_FILE_CACHE, data_path='data') -> None:
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
    
    @factorymethod('lastfm-hetrec')
    def load_lastfm_hetrec(self, tags:float=math.inf, mode:str='num'):
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
    
    @factorymethod('twitter')
    def load_twitter(self, tags:float=math.inf):
        '''Load the user-user Lastfm data with a given tag count'''
        import pickle
        with self.get_path('twitter.pickle','rb') as fid:
            ds = pickle.load(fid)
        
        db = DataBundle(relations=ds[0], structures=ds[1], name='Twitter')
        return db
    
    
    
    _get_dataset = FactoryGetter(classmethod=False)
    
    def __call__(self, text):
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
    

