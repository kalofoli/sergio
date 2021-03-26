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
    EntityAttributesWithTarget, EntityAttributesWithArrayTarget,\
    EntityAttributesWithAttributeTarget
import enum
from sergio.data.bundles.structures import EntityAttributesWithStructures,\
    EntityAttributesWithGraphs



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
    
    def file_path(self, file):
        return self._fm(file, kind = FileKinds.DATA)
            
    def open(self, file, *args, gzipped=False, **kwargs):
        if gzipped:
            import gzip
            path = self.file_path(file)
            return gzip.open(path, *args, **kwargs)
        else:
            return self._fm.open(file, *args, **kwargs, kind = FileKinds.DATA)
    
   
    @factorymethod('toy-array')
    def load_toy_array(self, name:str):
        """
        >>> df = DatasetFactory(file_manager = FileManager(paths={FileKinds.DATA:'datasets/'}), cache=None)
        >>> df.load_toy_array('circledots')
        <EntityAttributesWithArrayTarget[circledots](147x3/3) target: position(2d float64)>
        """
        filename = f'toy-{name}.hd5'
        path = self.file_path(filename)
        try:
            with pandas.HDFStore(path, 'r') as hs:
                df_attr = hs['attributes']
                df_target = hs['target']
        except OSError as e:
            from pathlib import Path
            rex = re.compile('toy-(?P<name>.*)\.hdf?5')
            matches = (rex.match(p.name) for p in Path(path).parent.iterdir())
            files = ','.join(repr(m.group('name')) for m in matches if m is not None)
            raise OSError(f'Error while loading dataset {name}. Available toy datasets: {files}.') from e
        target = df_target.values.astype(float)
        ds = EntityAttributesWithArrayTarget(name=name, attribute_data=df_attr, target=target, target_name='position')
        return ds
         
    
    class ToyScalarify(enum.Enum):
        TARGET = enum.auto()
        NORM = enum.auto()
        CNORM = enum.auto()
        ATTR = enum.auto()
    @factorymethod('toy-scalar')
    def load_toy_scalar(self, name:str, scalarify:ToyScalarify = ToyScalarify.NORM, param:str = None):
        """Load a toy dataset and scalarify it.
        .. param: scalarify: Define how the conversion to scalar works.
            If a ToyScalarify.TARGET: Get the target column at position param
            If a ToyScalarify.ANORM: Compute the target araray p-norm where p is equal to param (default: 2).
            If a ToyScalarify.CNORM: As above, but first center each target dimension around its mean.
            If a ToyScalarify.ATTR: Use the attribute as specified by param (index/name).
        
        >>> df = DatasetFactory(file_manager = FileManager(paths={FileKinds.DATA:'datasets/'}), cache=None)
        >>> df.load_toy_scalar('circledots')
        <EntityAttributesWithAttributeTarget[circledots](147x3/4) target: position_norm2(float64@3)>
        >>> df.load_toy_scalar('circledots', DatasetFactory.ToyScalarify.NORM,1)
        <EntityAttributesWithAttributeTarget[circledots](147x3/4) target: position_norm1(float64@3)>
        >>> df.load_toy_scalar('circledots', DatasetFactory.ToyScalarify.NORM, math.inf)
        <EntityAttributesWithAttributeTarget[circledots](147x3/4) target: position_norminf(float64@3)>
        >>> df.load_toy_scalar('circledots', DatasetFactory.ToyScalarify.CNORM, math.inf)
        <EntityAttributesWithAttributeTarget[circledots](147x3/4) target: position_norminf(float64@3)>
        >>> df.load_toy_scalar('circledots', DatasetFactory.ToyScalarify.ATTR, 0)
        <EntityAttributesWithAttributeTarget[circledots](147x2/3) target: label(category@2)>
        >>> df.load_toy_scalar('circledots', DatasetFactory.ToyScalarify.ATTR, 'east')
        <EntityAttributesWithAttributeTarget[circledots](147x2/3) target: east(bool@2)>
        """
        import numpy as np
        ds_arr = self.load_toy_array(name)
        target_data_arr = ds_arr.target_data
        df_attr = ds_arr.attribute_data
        if scalarify == self.ToyScalarify.TARGET:
            if param is None:
                param = 0
            else:
                param = int(param)
            target_data = target_data_arr[:,param]
            target_name = f'{ds_arr.target_name}{param}'
        elif scalarify == self.ToyScalarify.NORM or scalarify == self.ToyScalarify.CNORM:
            if param is None:
                param = 2
            else:
                param = float(param)
            if scalarify == self.ToyScalarify.CNORM:
                target_data_arr = target_data_arr-target_data_arr.mean(axis=0)
            target_data_arr = np.abs(target_data_arr)
            if param == 1:
                target_data = target_data_arr@np.r_[1,1]
            elif param == np.inf:
                target_data = np.max(target_data_arr, axis=1)
            elif param > 1:
                target_data = (target_data_arr**param@np.r_[1,1])**(1/param)
            else:
                raise ValueError(f'The parameter must indicate a valid p-norm.')
            target_name = f'{ds_arr.target_name}_norm{param:g}'
        elif scalarify == self.ToyScalarify.ATTR:
            if param is None:
                param = 0
            else:
                try: param = int(param)
                except ValueError: pass
            alu = ds_arr.lookup_attribute(param)
            target_name = alu.name
            target_data = df_attr.pop(alu.name)
        
        
        df_attr.loc[:, target_name] = target_data
        ds = EntityAttributesWithAttributeTarget(name=ds_arr.name, attribute_data=df_attr, target=target_name)
        return ds
        
        
        
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
    
    @factorymethod('morris')
    def load_morris(self, name:str, with_oidx:bool=True, directed:bool=None):
        '''Load the Morris database dataset.
        
        >>> df = DatasetFactory(file_manager = FileManager(paths={FileKinds.DATA:'datasets/'}), cache=None)
        >>> df.load_morris('MUTAG', True)
        <EntityAttributesWithGraphs[MUTAG](188x1/1)>
        '''
        from sergio.data.factory.morris import MorrisLoader
        data_home = self._fm.get_kind_path(FileKinds.DATA_MORRIS)
        ml = MorrisLoader(data_home=data_home)
        dsm = ml.load_dataset(name)
        ea = EntityAttributesWithGraphs.from_morris(**dsm.__dict__, with_oidx=with_oidx, directed=directed)
        return ea
    
    
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

