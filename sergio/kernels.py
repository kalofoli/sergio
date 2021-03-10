'''
Measures

Created on Oct 3, 2019

@author: janis
'''
# pylint: disable=bad-whitespace
from .utils import ClassCollection
from cofi.kernel_detail.random_walks import RandomWalkKernel, CoreRandomWalkKernel, WLRandomWalkKernel,\
    DegreeRandomWalkKernel, NLRandomWalkKernel, MIRandomWalkKernel
from cofi.kernel_detail.euclidean import RadialBasisFunctionKernel, LinearKernel, IncrementalSIKernel, IndicatorSIKernel,\
    AllEquivalentSIKernel, TruncatedRadialBasisFunctionSIKernel,\
    RadialBasisFunctionSIKernel
from cofi.kernel_detail import Kernel, ShiftInvariantKernel
from cofi.factory import make_string_constructor
from grakel.kernels.core_framework import CoreFramework
from cofi.summarisable import SummaryOptions, SummarisableDict,\
    COMPACT_SUMMARY_OPTIONS
from cofi.utils.locking import TrackingFileLock
try:
    from cofi.kernel_detail.wwl import WassersteinWeissfeilerLehmanKernel
except ImportError as e:
    exc = e
    class WassersteinWeissfeilerLehmanKernel(Kernel):
        name = 'Wasserstein Weissfeiler-Lehman Kernel'
        tag = 'wasserstein-weissfeiler-lehman'
        def __init__(self,*args, **kwargs):
            raise ImportError(f'Could not import necessary module wwl, which contains the original source code. Please install it from: https://github.com/BorgwardtLab/WWL') from exc
from cofi.utils.statistics import StatisticsBase
from zipfile import ZipFile
from cofi.utils import SliceableList

from cofi.logging import getLogger
import time
import re
from tempfile import TemporaryDirectory

log = getLogger(__name__.split('.')[-1])

class CoreFrameworkKernel(Kernel, CoreFramework):
    name = 'Core-Framework'
    tag = 'core-framework'
    
    def __init__(self, n_jobs=None, verbose=False,
                 normalize=False, min_core=-1, base_graph_kernel=None):
        self._stats = None
        super().__init__(n_jobs=n_jobs, verbose=verbose, normalize=normalize, min_core=-1, base_graph_kernel=base_graph_kernel)
        self._initialized.update(stats=False) 
    
    def summary_dict(self, summary_options:SummaryOptions):
        pars = self.get_params(deep=False)
        return SummarisableDict(pars)
    
    def get_params(self, deep=True, merge=True):
        params = super().get_params(deep=deep)
        if merge:
            params_kernel = self.base_graph_kernel.get_params()
            params.update(params_kernel)
        return params
    
    def set_params(self, **params):
        pars = self.get_params(deep=True,merge=False)
        pars_local = {k:v for k,v in params.items() if k in pars}
        pars_nested = {f'base_graph_kernel__{k}':v for k,v in params.items() if k not in pars}
        super().set_params(**pars_local, **pars_nested)
    
    def make_base_graph_kernel_instance(self, **params):
        inst = super().make_base_graph_kernel_instance(**params)
        if self._stats is not None:
            inst._stats = self._stats
        return inst
    
    def initialise_base_graph_kernel_factory(self, base_graph_kernel, params):
        if isinstance(base_graph_kernel, Kernel):
            factory = base_graph_kernel.__class__
            if hasattr(base_graph_kernel, '_stats') and isinstance(base_graph_kernel._stats, StatisticsBase):
                self._stats = base_graph_kernel._stats
            pars_inst = base_graph_kernel.get_params()
            params.update(**pars_inst)
            params.update(n_jobs=None, verbose=self.verbose,normalize=False)
        else:
            self._stats = None
            factory = super().initialise_base_graph_kernel_factory(base_graph_kernel, params)
        self._initialized.update(stats=True)
        return factory
    
    @property
    def stats_block(self):
        '''this is a context. Use with "with"'''
        if not self._initialized['stats']:
            self.initialize()
        ctx = self._stats.block if self._stats is not None else None
        return ctx

KERNELS = ClassCollection('Kernels', (RandomWalkKernel, CoreRandomWalkKernel, DegreeRandomWalkKernel,
                                      WLRandomWalkKernel, MIRandomWalkKernel, RadialBasisFunctionKernel,
                                      LinearKernel,CoreFrameworkKernel, NLRandomWalkKernel,
                                      WassersteinWeissfeilerLehmanKernel))
SI_KERNELS = ClassCollection('ShiftInvariantKernels', (IncrementalSIKernel, IndicatorSIKernel, AllEquivalentSIKernel,
                                                       TruncatedRadialBasisFunctionSIKernel,RadialBasisFunctionSIKernel))


Kernel.make_from_strings = make_string_constructor(KERNELS)
ShiftInvariantKernel.make_from_strings = make_string_constructor(SI_KERNELS)


import numpy as np
import os
class KernelCacheAdaptor(Kernel):
    __max_tag_size__ = 250
    class TaggedList(list): 
        def __init__(self, tag, *args, **kwargs):
            self.tag = tag
            super().__init__(*args, **kwargs)

    def __init__(self, kernel, file_manager, lock_cls = TrackingFileLock):
        self.file_manager = file_manager
        self.kernel = kernel
        self.updated = False
        self.valid = False
        self.tag = f'cached-{self.kernel.tag}'
        self.tag = f'Cached {self.kernel.name}'
        self.K = None
        self.lock = None
        self.lock_cls = lock_cls
        
        
    def patch_dataset(self, dataset):
        '''Adds indices to the dataset entries
        
        This is necessary so that partial views of the dataset (to compute the kernel on)
        can be mapped to the cached entries.
        ''' 
        dataset.data = SliceableList(self.TaggedList(*e) for e in enumerate(dataset.data))
        self._dataset = dataset
        self.lock = self.lock_cls(self.lock_file_name)
        return dataset
    
    rex_keys_1 = re.compile('[_]')
    rex_keys_2 = re.compile('[aoueiAOUEI_]')
    def _make_tag(self, dct, pressure):
        if pressure == 0:
            spars = ','.join(f'{k}:{v!r}' for k,v in dct.items())
        else:
            pars = []
            for k,v in dct.items():
                if v is None:
                    continue
                if isinstance(v, float):
                    sv = f'{v:.6g}'
                elif isinstance(v,bool):
                    sv = 'T' if v else 'F'
                else:
                    sv = f'{v!s}'
                if pressure >= 2:
                    rex = self.rex_keys_1 if pressure<3 else self.rex_keys_2
                    k = rex.sub('',k)
                pars.append(f'{k}:{sv}')
            spars = ','.join(pars)
            if pressure >= 4:
                spars = spars[:self.__max_tag_size__]
        return spars
    @property
    def file_tag(self):
        dct = self.kernel.summary_dict(COMPACT_SUMMARY_OPTIONS)
        for pressure in range(4):
            spars = self._make_tag(dct, pressure)
            stag = f'{self.dataset.digest}-{self.kernel.tag}-{spars}'
            if len(stag) < self.__max_tag_size__:
                break
        return stag
    
    @property
    def file_name(self): return self.file_manager.get(f'{self.file_tag}.npz', 'CACHE')
    @property
    def lock_file_name(self): return self.file_manager.get(f'{self.file_tag}.lck', 'CACHE')
    
    def _get_index(self, X):
        return np.array([x.tag for x in X])

    def _do_save(self, filename, **kwargs):
        from numpy import savez_compressed
        fn_cache = self.file_manager.get('.','CACHE')
        # need a new dir to avoid collisions and fle name length extension.
        # To avoid cross device links, the tmpdir is local.
        with TemporaryDirectory(dir=fn_cache) as tmp_dir:
            fn_base = os.path.split(filename)[-1]
            fn_tmp = os.path.join(tmp_dir, fn_base)
            with open(fn_tmp,'wb') as fid:
                savez_compressed(fid, **kwargs)
            os.rename(fn_tmp, filename)

    def _do_load(self, filename):
        from numpy import load
        return load(filename)

    def save(self):
        try:
            fn = self.file_name
            self._do_save(fn, K=self.K)
            log.progress(f'Stored kernel matrix for kernel {self.kernel} applied on dataset {self.dataset} to file "{fn}"')
        except Exception as e:
            log.error(f'While storing kernel matrix for kernel {self.kernel} applied on dataset {self.dataset} to file "{fn}": {e}')
            
    def load(self):
        fn = self.file_name
        try:
            data = self._do_load(fn)
            self.K = data['K']
            self.valid = True
            log.progress(f'Loaded kernel matrix for kernel {self.kernel} applied on dataset {self.dataset} from file "{fn}"')
        except FileNotFoundError: pass
        except Exception as e:
            log.error(f'While loading kernel matrix for kernel {self.kernel} applied on dataset {self.dataset} to file "{fn}": {e}')
    
    def invalidate_cache(self):
        self.valid = False
        
    def set_params(self, *args,**kwargs):
        self.invalidate_cache()
        self.kernel.set_params(*args, **kwargs)
        
    def get_params(self):
        return self.kernel.get_params()
    
    def fit(self, X):
        self._idx_fitted = self._get_index(X)
    
    @property
    def dataset(self): return self._dataset
    
    def populate(self):
        if not self.valid:
            
            self.load()
        if not self.valid:
            log.progress(f'Computing kernel {self.kernel} on dataset {self.dataset}')
            with self.lock:
                self.K = self.kernel.fit_transform(self.dataset.data)
                self.valid = True
                self.save()
        
        
    def transform(self, Y):
        self.populate()
        idx_trn = self._get_index(Y)
        K = self.K[np.ix_(idx_trn, self._idx_fitted)]
        return K
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def summary_dict(self, options:SummaryOptions):
        dct = self.kernel.summary_dict(options)
        dct.update(cached=True)
        return dct

    @property
    def summary_name(self):
        return f'kern-{self.tag}'
    
if __name__ == '__main__':
    from cofi.experiment import Experiment
    e = Experiment()
    e.load_dataset('MUTAG')
    import grakel
    
    k = grakel.RandomWalk()
    k.fit(e.dataset.data)
    
    krn = Kernel.make_from_strings('random-walk',n_jobs='4')
    pass
    
    