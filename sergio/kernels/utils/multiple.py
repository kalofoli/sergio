'''
Created on Sep 7, 2021

@author: janis
'''

import joblib
import pickle
import numpy as np

from sergio.kernels.gramian import GramianNystroemFromArray
from .operations import compute_gramian_frobenius,\
    compute_mkl_alignment_coeffs_orthogonal_heuristic

from colito.parallel import ProgressParallel
from colito.cache import cache_to_disk

class SubkernelOptimiserBase:
    '''For each feature, optimise a subkernel'''
    def __init__(self, indices, cache_prefix=None):
        self._cache_prefix = cache_prefix
        self._indices = np.arange(indices) if np.isscalar(indices) else indices
        self._evaluations = {}
    @property
    def cache_prefix(self): return self._cache_prefix
    @property
    def evaluations(self): return self._evaluations
    
    def cache_file(self, index):
        raise NotImplementedError()
    
    def invoke_kernel(self, idx, pars):
        X = self._features[:,[idx]]
        args, kwargs = [], {}
        it_par = iter(pars)
        for s in self._spaces:
            if s.name is None:
                args.append(next(it_par))
            else:
                kwargs[s.name] = next(it_par)
        return self._kern(X, *args, **kwargs)

    def feature_optimiser(self, index):
        '''Instantiates a feature optimiser for a given feature index'''
        raise NotImplementedError()
    def optimise_index(self, index, recompute=False, compute=True):
        if self._cache_prefix is not None:
            fname = self.cache_file(index)
            res = None
            if not recompute:
                try:
                    with open(fname, 'rb') as fid:
                        res = pickle.load(fid)
                except: pass
            if res is None:
                if compute:
                    opt = self.feature_optimiser(index)
                    res = opt()
                    with open(fname, 'wb') as fid:
                        pickle.dump(res, fid)
                else:
                    return {}
        else:
            opt = self.feature_optimiser(index)
            res = opt()
        return {index:res}

            
    def optimise_subkernels(self, indices=None, n_jobs=1, recompute=False, randomise=True, compute=True, progress=True):
        if indices is None:
            indices = self._indices[np.random.permutation(len(self._indices))]
        if recompute is False:
            indices_computed = np.r_[list(self.evaluations.keys())]
            res_known = {i:self.evaluations[i] for i in indices_computed}
            indices = np.setdiff1d(indices, indices_computed)
        else:
            res_known = {}
        if len(indices):
            if randomise:
                indices = indices[np.random.permutation(len(indices))]
            par = ProgressParallel(n_jobs=n_jobs, tqdm_opts=dict(total=len(indices), desc='Optimise subkernels')) \
                if progress else joblib.Parallel(n_jobs=n_jobs)
            packs = par([joblib.delayed(self.optimise_index)(idx, recompute=recompute, compute=compute) for idx in indices])
            from itertools import chain
            res_new = dict(chain(*[p.items() for p in packs]))
        else:
            res_new = {}
        res = {**res_known, **res_new}
        self._evaluations = res_srt = dict(sorted(res.items(), key=lambda x:x[0]))
        return res_srt

    def optimal_params(self, idx):
        evals = self._evaluations.get(idx)
        pars = tuple(evals.keys())
        vals = np.r_[list(evals.values())]
        idx_max = np.argmax(vals)
        pars_opt = pars[idx_max]
        return pars_opt, vals[idx_max]
        
    def optimal_subkernel(self, idx):
        '''The subkernel computed at the optimal parameters'''
        pars = self.optimal_params(idx)[0]
        of = self.feature_optimiser(idx)
        return of.get_kernel(pars)

    def optimal_alignment_candidate(self, idx):
        '''The alignment candidate derived from the optimal subkernel'''
        of = self.feature_optimiser(idx)
        K = self.optimal_subkernel(idx)
        return of.alignment_candidate(K)
    

class FeatureSubkernelOptimiser(SubkernelOptimiserBase):
    __cache_suffix__ = 'sko-{tag}-{index}.pickle'
    def __init__(self, *args, spaces, features, validities, feature_optimiser_class, gp_opts={}, **kwargs):
        self._features = features
        super().__init__(*args, indices=self.num_features, **kwargs)
        self._validities = validities
        self._foc = feature_optimiser_class
        self._gp_opts = gp_opts
        self._spaces = spaces
        self._fos = {}
    @property
    def validities(self): return self._validities
    def cache_file(self, index):
        if self._cache_prefix is None:
            return None
        suffix = self.__cache_suffix__.format(tag=self.__tag__, index=index)
        return self._cache_prefix + suffix
    @property
    def num_features(self): return self._features.shape[1]
    def feature_optimiser(self, index):
        fo = self._fos.get(index)
        if fo is None:
            f = self._features[:,[index]]
            gp_opts = {'random_state':index, **self._gp_opts}
            pars = {'spaces':self._spaces, 'gp_opts':gp_opts}
            fo = self._foc(feature=f, validities=self.validities, **pars)
            self._fos = {index:fo}
        return fo
    def __repr__(self):
        return f'<{type(self).__name__}[{self._foc.__tag__}] evals: {len(self.evaluations)}/{self.num_features}>'
    @property
    def __tag__(self): return self._foc.__tag__

    def _compute_optimal_gramians(self, indices=None, rank=25, n_jobs=1, recompute=False, progress=True,randomise=True, batch_size=10):
        def _provider(idx):
            K_cnd = self.optimal_alignment_candidate(idx)
            G = GramianNystroemFromArray(K_cnd, rank=rank)
            return {idx:G}

        if self._cache_prefix is not None:
            @cache_to_disk(cache_dir=self._cache_prefix, fmt=f'ogm-{self.__tag__}-r{rank}-{{idx}}.pickle', recompute=recompute)
            def provider(idx):
                return _provider(idx)
        else:
            provider = _provider
        if indices is None:
            indices = list(self.evaluations.keys())
        if randomise:
            indices = np.r_[indices][np.random.permutation(len(indices))]
        if n_jobs == -1:
            n_jobs = joblib.cpu_count()
        if batch_size is None:
            batch_size = np.ceil(len(indices)/n_jobs)
        n_parts = np.ceil(len(indices)/batch_size)
        parts = np.array_split(indices, n_parts)
        def process_batch(indices):
            res = {}
            for idx in indices:
                res.update(provider(idx))
            return res
        if progress:
            par = ProgressParallel(n_jobs=n_jobs, tqdm_opts=dict(total=n_parts, desc='Compute Gramians'))
        else:
            par = joblib.Parallel(n_jobs=n_jobs)
        gramians = par(joblib.delayed(process_batch)(part) for part in parts)
        from itertools import chain
        joined = dict(chain(*[g.items() for g in gramians]))
        srt = dict(sorted(joined.items(),key=lambda x:x[0]))
        return srt
    
    def aggregate_subkernels(self, coeffs, k=None, eps=1e-6, dtype=np.float64):
        idl_sel = coeffs>eps
        idx_sel = np.where(idl_sel)[0]
        K = self.optimal_subkernel(idx_sel[0]) * coeffs[idx_sel[0]]
        for idx in idx_sel[1:]:
            K += coeffs[idx]*self.optimal_subkernel(idx)
        return K
    
    def compute_subkernel_gramians(self, rank=25, n_jobs=1, batch_size=None, progress=True):
        gramians = {}
        evals = self.optimise_subkernels(n_jobs=n_jobs, progress=progress)
        indices = list(evals.keys())
        @cache_to_disk(cache_dir=self.cache_prefix, fmt=f'ogm-{self.__tag__}-r{rank}-all.pickle')
        def do():
            return self._compute_optimal_gramians(rank=rank, n_jobs=n_jobs, batch_size=batch_size, progress=progress)
        return do()

    def compute_frobenius_components(self, rank=25, normalise=True, n_jobs=1, progress=True):
        K_tar = self.feature_optimiser(0).alignment_target
        gram_dct = self.compute_subkernel_gramians(rank=rank, n_jobs=n_jobs)
        gramians = [gram_dct[i] for i in range(self.num_features)]
        @cache_to_disk(cache_dir=self.cache_prefix, fmt=f'frob-{self.__tag__}-r{rank}.pickle')
        def do():
            return compute_gramian_frobenius(gramians, K_tar, normalise=normalise, progress=progress)
        return do()
    
    def compute_alignment_coefficients(self, rank=25, n_jobs=1, with_info=False):
        W,v = self.compute_frobenius_components(rank=rank, n_jobs=n_jobs)
        res = compute_mkl_alignment_coeffs_orthogonal_heuristic(W,v)
        return res if with_info else res.a_best

    def compute_kernel(self, rank=25, normalise=True, n_jobs=1, progress=True):
        @cache_to_disk(cache_dir=self.cache_prefix, fmt=f'K-{self.__tag__}-r{rank}.pickle')
        def do():
            self.compute_frobenius_components(rank=rank, normalise=normalise, progress=progress, n_jobs=n_jobs)
            a = self.compute_alignment_coefficients(rank=rank, n_jobs=n_jobs)
            K = self.aggregate_subkernels(a)
            return K
        return do()
    __call__ = compute_kernel
    
    