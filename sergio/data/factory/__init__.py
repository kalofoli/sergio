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

from types import SimpleNamespace
from sergio import FileManager, FileKinds



from urllib import request
import pandas
from sergio.data.bundles.entities import EntityAttributes,\
    EntityAttributesWithTarget, EntityAttributesWithArrayTarget
import enum
from sergio.data.bundles.structures import EntityAttributesWithStructures



def read_url(url):
    with request.urlopen(url) as req:
        data = req.read()
    return data

def save_url(url, file):
    with request.urlopen(url) as req:
        with open(file, 'wb') as fid:
            while True:
                data = req.read(100000)
                if not data:
                    return
                fid.write(data)

_DATASET_NAMES = {'moondot': 'dataset-moondot.csv'}


class DataBundle:
    class Relations(SimpleNamespace): pass
    class Structures(SimpleNamespace): pass
    def __init__(self, relations, structures, name):
        self._name = name
        self.rels = self.Relations(**relations)
        self.structs = self.Structures(**structures)

class DatasetFactory(FactoryBase):
    '''Loader for various datasets'''

    def __init__(self, cache: FileCache=None, file_manager:FileManager=None) -> None:
        self._cache = cache
        self._fm = file_manager if file_manager is not None else FileManager()
        
    prediciser = property(lambda self:self._prediciser, None, 'Prediciser to use on the created graph-data.')
    data_path = property(lambda self:self._data_path, None, 'Path on which the datasets are searched.')
    
    def open(self, file, *args, gzipped=False, **kwargs):
        if gzipped:
            import gzip
            path = self._fm(file, kind = FileKinds.DATA)
            return gzip.open(path, *args, **kwargs)
        else:
            return self._fm.open(file, *args, **kwargs, kind = FileKinds.DATA)
    
   
    class MoondotKind(enum.Enum):
        BOOLEAN = enum.auto()
        CATEGORICAL = enum.auto()
        
    @factorymethod('moondot')
    def load_moondot(self, kind: MoondotKind = MoondotKind.BOOLEAN):
        """
        >>> df = DatasetFactory(file_manager = FileManager(paths={FileKinds.DATA:'datasets/'}), cache=None)
        >>> df.load_moondot()
        <EntityAttributesWithArrayTarget[moondot](73x4/4) target: position(2d float64)>
        """
        name = _DATASET_NAMES['moondot']
        with self.open(name, 'r') as fid:
            df_struct = pandas.read_csv(fid, sep=' ')
        label = df_struct.pop('label').astype(pandas.CategoricalDtype())
        if not isinstance(kind, self.MoondotKind):
            raise TypeError(f'Provided kind {kind} is of type {type(kind).__name__} which is not MoondotKind. Did you mean to resolve the enum?') 
        if kind == self.MoondotKind.BOOLEAN:
            df_attr = pandas.DataFrame({l:label==l for l in pandas.unique(label)})
        elif kind == self.MoondotKind.CATEGORICAL:
            df_attr = pandas.DataFrame({'label':label})
        else:
            raise NotImplementedError(f'Unsupported kind: {kind} of type {type(kind).__name__}.')
        target = df_struct.values
        return EntityAttributesWithArrayTarget(name='moondot', attribute_data=df_attr, target=target, target_name='position')
    
    @factorymethod('twitter')
    def load_twitter(self, tags:float=math.inf):
        '''Load the user-user Lastfm data with a given tag count'''
        import pickle
        with self.open('ego-twitter.pickle.gz','rb', gzipped=True) as fid:
            db = pickle.load(fid)
        
        return db
    
    
    
    _get_dataset = FactoryGetter(classmethod=False)
    
    def __call__(self, text):
        '''Load a dataset by name, optionally using cache'''
        bundle = self._get_dataset(text,produce=False)
        hint = f'dataset-{bundle.digest}'
        loader = lambda : bundle.produce(self)
        if self._cache is not None:
            data = self._cache.load(tag=hint, loader=loader)
        else:
            data = loader()
        return data
    
    load_dataset = __call__
    
    description = FactoryDescription()



if __name__ == '__main__':
    
    import doctest
    doctest.testmod()

