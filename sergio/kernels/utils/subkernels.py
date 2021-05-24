'''
Created on May 20, 2021

@author: janis
'''

import typing

import numpy as np

from tqdm import tqdm
import joblib

import warnings

from sergio.kernels.gramian import GramianFromArray
from colito.cache import cache_to_disk

class RankDeficientMatrixWarning(Warning): pass

class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, total=None, **kwargs):
        self._total = total
        with tqdm(total=total) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks if not self._total is not None else self._total
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
        

def jaccard_kernel(X, Y=None):
    '''Computes the Jaccard kernel between all (column) features in the V set
    
    '''
    if Y is None:
        Y = X
    X = X.astype(int)
    Y = Y.astype(int)
    caps = X.T@Y
    sum_x = X.sum(0)
    sum_y = Y.sum(0)
    cups = (sum_x[:,None] + sum_y[None,:]) - caps 
    jaccard = np.empty(caps.shape, float)
    idl_nz = cups != 0
    jaccard[idl_nz] = caps[idl_nz]/cups[idl_nz]
    jaccard[~idl_nz] = 0
    return jaccard

def compute_kernel_alignment(K1, K2):
    '''Computes the kernel alignment between two Gram matrices over the same entities'''
    frob_12 = np.sum(K1*K2)
    frob_11 = np.sum(K1**2)
    frob_22 = np.sum(K2**2)
    return frob_12/np.sqrt(frob_11*frob_22)

def compute_mean_map_kernel(V, K, empty=np.nan, progress=False):
    '''Computes the mean map kernel from set characteristics and a kernel, both over the same entities.
    :param V: The set characteristics for each set (rows: entities, column: set)
        The entries of V must be memberships and convertible to logicals.
    :param K: The Gramian of a kernel over the entities over which V indicates set membership.
    
    :result: The Gramian of the mean map kernel. Its size is m x m, where m is the number of sets (columns of V)
    '''
    supports = V.sum(axis=0)
    idl_ok = supports > 0
    
    n_elems,n_sets = V.shape
    
    K_mm = np.empty((n_sets,n_sets), float)
    if progress:
        it = tqdm(range(n_sets))
    else:
        it = range(n_sets)
    idx_ok = np.where(idl_ok)[0]
    for oi, fi in enumerate(idx_ok):
        v = V[:,fi].astype(bool)
        idx_ok_rest = idx_ok[oi:]
        k = K[v,:].sum(axis=0)/supports[fi]
        k_mm = V[:,idx_ok_rest].T@k/supports[idx_ok_rest]
        K_mm[fi,idx_ok_rest] = k_mm
        K_mm[idx_ok_rest,fi] = k_mm
    K_mm[:,~idl_ok] = empty
    K_mm[~idl_ok,:] = empty
    return K_mm

def compute_mkl_alignment_coeffs(W, v, k=10, mu_w=1e-5, eps=1e-5):
    from numpy.linalg import norm
    from scipy.sparse.linalg import eigsh 
    n = W.shape[0]
    l,S = eigsh(W, k=k)
    idl_z = l<eps
    if np.any(idl_z):
        warn = RankDeficientMatrixWarning(f'During pseudo inverse computation of W: {idl_z.sum()} eigenvalues below tolerance of {eps:g} discarded. Remaining: {len(idl_z)-idl_z.sum()}.')
        warnings.warn(warn)
        l=l[~idl_z]
        S = S[:,~idl_z]
    N = S*(l+mu_w)**-.5@S.T
    a = N@v[:,None]
    a_nrm = a/norm(a)
    # Solve ||z-a_nrm|| s.t.: Nz>0
    from cvxopt import solvers, matrix
    solvers.options['show_progress'] = False
    # qp: 1/2 xTPx + qTp st: Gx<h
    P = matrix(np.eye(n))
    q = matrix(-a_nrm)
    G = matrix(-N)
    h = matrix(np.zeros(n))
    res = solvers.qp(P=P, q=q, G=G, h=h)
    z = np.array(res['x']).flatten()
    x = N@z
    return x.flatten()/np.linalg.norm(x)

class SubkernelOptimiser:
    def __init__(self, features, validities, spaces, kern: typing.Callable, cache_dir, progress=False, tag='',
                 gp_opts={}):
        self._features = features
        self._validities = validities
        self._kern = kern
        self._spaces = spaces
        self._idl_ok = self._validities.any(axis=0)
        self._K_jac = None
        self._progress = progress
        self._cache_dir = cache_dir
        self._tag = tag
        self._gp_opts = {'n_calls':50, 'noise':1e-10,**gp_opts}
        self._backend = 'joblib'
        self._evaluations = None

    @property
    def evaluations(self): return self._evaluations
    @property
    def K_jaccard(self):
        if self._K_jac is None:
            self._K_jac = jaccard_kernel(self.validities)
        return self._K_jac
    @property
    def K_jac_ok(self): return self.K_jaccard[np.ix_(self._idl_ok, self._idl_ok)]
    
    @property
    def validities(self): return self._validities
    @property
    def val_nz(self): return self._validities[:,self._idl_ok]
    @property
    def n_feats(self): return self._features.shape[1]
    @property
    def n_entities(self): return self._features.shape[0]
    @property
    def n_sets(self): return self._validities.shape[1]
    
    def compute_mean_map_kernel(self, K, ok_only=True):
        K_mm = compute_mean_map_kernel(self.val_nz if ok_only else self.validities, K)
        return K_mm
    def jaccard_alignment_from_feature_kernel(self, K):
        K_mm = self.compute_mean_map_kernel(K)
        return compute_kernel_alignment(K_mm, self.K_jac_ok)

    def _load_cache(self, idx, tag=None):
        import os
        import pickle
        if tag is None:
            tag = self._tag
        fname = os.path.join(self._cache_dir, f'{self._tag}{idx}.pickle')
        with open(fname, 'rb') as fid:
            return pickle.load(fid)
    
    def _store_cache(self, idx, data):
        import os
        import pickle
        fname = os.path.join(self._cache_dir, f'{self._tag}{idx}.pickle')
        with open(fname, 'wb') as fid:
            return pickle.dump(data, fid)
    
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


    def optimise_single_feature(self, idx_feat, spaces=None, features=None, gp_opts = {}, with_info=False):
        from skopt.optimizer import gp_minimize
        if features is None:
            features = self._features
        if spaces is None:
            spaces = self._spaces
        par_evals = {}
        
        def evaluate(x):
            x = tuple(x)
            
            if x not in par_evals:
                K_feat = self.invoke_kernel(idx_feat, x)
                algn = self.jaccard_alignment_from_feature_kernel(K_feat)
                par_evals[x] = algn
            return par_evals[x]

        res = gp_minimize(evaluate, spaces, **{**self._gp_opts, **gp_opts})
        if with_info:
            return par_evals, res
        else:
            return par_evals

    def compute_optimal_mm_gramians(self, rank_mm, keys=None, n_jobs=1, use_cache=True):
        def _provider(idx):
            K_mm = self.get_subkernel_mm(idx)
            G = GramianFromArray(K_mm, rank=rank_mm)
            return G
    
        if use_cache:
            @cache_to_disk(cache_dir=self._cache_dir, fmt=f'{self._tag}mmgram-r{rank_mm}-{{idx}}.pickle')
            def provider(idx):
                return _provider(idx)
        else:
            provider = _provider
        if keys is None:
            keys = list(self.evaluations.keys())
        keys = np.array(keys)[np.random.permutation(len(keys))]
        def process_keys_batch(keys):
            gramians = {}
            for key in keys:
                G = provider(key)
                gramians[key] = G
            return gramians
        
        if n_jobs is None:
            n_jobs = joblib.cpu_count()
        parts = np.array_split(keys, n_jobs)
        par = ProgressParallel(n_jobs=n_jobs)
        gramians = par([joblib.delayed(process_keys_batch)(part) for part in parts] , total=len(parts))
        from itertools import chain
        joined = dict(chain(*[g.items() for g in gramians]))
        return joined

    def optimise_subkernel_params(self, indices, n_jobs=1):
        parts = np.array_split(indices, n_jobs)
        def test_feats(parts):
            res_parts = {}
            for fi in parts:
                res = self.optimise_single_feature(fi)
                self._store_cache(fi, res)
                res_parts[fi] = res
            return res_parts
        if self._backend == 'joblib':
            par = ProgressParallel(n_jobs=n_jobs)
            packs = par(joblib.delayed(test_feats)(work) for work in parts)
            from itertools import chain
            res = dict(chain(*[p.items() for p in packs]))
        return res
    
    def get_subkernel_params(self, train=False, use_cache=True, tag=None,  **kwargs):
        res = {}
        missing = []
        if use_cache:
            for fi in range(self.n_feats):
                try:
                    res[fi] = self._load_cache(fi, tag=tag)
                except FileNotFoundError:
                    missing.append(fi)
        else:
            missing = np.arange(self.n_feats)
        if train:
            res_missing = self.optimise_subkernel_params(missing, **kwargs)
            res = {**res, **res_missing}
        return res
    def load_subkernel_params(self, train=False, use_cache=True, tag=None, **kwargs):
        self._evaluations = self.get_subkernel_params(train=train, use_cache=use_cache, tag=tag, **kwargs)
    def get_subkernel(self, idx):
        evals = self.evaluations[idx]
        vals = list(evals.values())
        pars = list(evals.keys())
        opt_pars = pars[np.argmax(vals)]
        return self.invoke_kernel(idx, opt_pars)
    
    def get_subkernel_mm(self, idx):
        K = self.get_subkernel(idx)
        return self.compute_mean_map_kernel(K)

    def get_subkernel_gramian(self, idx, k=10):
        '''Returns a low-rank Gramian for the kernel'''
        from sergio.kernels.gramian import GramianFromArray
        K = self.get_subkernel(idx)
        return GramianFromArray(K, rank=k)
