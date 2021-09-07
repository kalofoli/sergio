'''
Created on May 20, 2021

@author: janis
'''

import typing

import numpy as np

import joblib

from colito.parallel import ProgressParallel

import warnings
import typing

from sergio.kernels.gramian import GramianFromArray, Gramian
from colito.cache import cache_to_disk
from colito.parallel import tqdm
from types import SimpleNamespace

class RankDeficientMatrixWarning(Warning): pass


def compute_matrix_frobenius(gramians: typing.List[np.ndarray], K_tar):
    n = len(gramians)
    W = np.zeros((n,n))
    v = np.zeros(n)
    f_tar = np.linalg.norm(K_tar, 'fro')
    K_tar_nrm = K_tar/f_tar
    for i,G_i in enumerate(gramians):
        for j,G_j in enumerate(gramians[i:],i):
            W[i,j] = np.sum(G_i*G_j)
            W[j,i] = W[i,j]
    for i,G_i in enumerate(gramians):
        v[i] = np.sum(G_i*K_tar_nrm)
    return W, v

def compute_gramian_frobenius(gramians:typing.Sequence[Gramian], K_tar, normalise=False, progress=False):
    '''Compute the Frobenius inner products for MKL optimisation.
    
    @param gramians A sequence of Gramian objects.
    @param K_tar The numpy array representing the target kernel.
    @return Tuple[np.ndarray, np.ndarray] W,v where W is the matrix of inner products
        between the Gramians and v is the inner product between the Gramians and the 
        target kernel. 
     
    '''
    n = len(gramians)
    W = np.zeros((n,n))
    v = np.zeros(n)
    
    f_tar = np.linalg.norm(K_tar, 'fro')
    K_tar_nrm = K_tar/f_tar
    outer_seq = tqdm(gramians) if progress else gramians
    for i,g_i in enumerate(outer_seq):
        l_i,S_i = g_i.eigenvals, g_i.eigenvecs
        l_i = np.maximum(0,l_i)
        V_i = S_i*l_i**.5
        for j,g_j in enumerate(gramians[i:],i):
            l_j,S_j = g_j.eigenvals, g_j.eigenvecs
            l_j = np.maximum(0,l_j)
            V_j = S_j*l_j**.5
            
            W[i,j] = np.sum((V_i.T@V_j)**2)
            W[j,i] = W[i,j]
        v[i] = np.trace(V_i.T@K_tar_nrm@V_i)
    if normalise:
        s = W.max()
        W = W/s
        v = v/np.sqrt(s)
    return W, v


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

def normalise_kernel(K):
    m = 1/np.sqrt(np.diag(K))
    m[np.isnan(m)] = 1
    return K*m[:,None]*m[None,:]

def compute_extended_tanimoto_kernel(V, K, empty=np.nan, normalise=False):
    '''Computes the extended Tanimoto kernel from set characteristics and a kernel, both over the same entities.
    :param V: The set characteristics for each set (rows: entities, column: set)
        The entries of V must be memberships and convertible to logicals.
    :param K: The Gramian of a kernel over the entities over which V indicates set membership.
    
    :result: The Gramian of the mean map kernel. Its size is m x m, where m is the number of sets (columns of V)
    '''
    supports = V.sum(axis=0)
    idl_ok = supports > 0
    
    M = V.T@K
    N = M@V # nominator
    m = M.sum(axis=1)
    D = m[:,None] + m[None,:] - N
    D[:,~idl_ok] = D[~idl_ok,:] = 1
    K_et = N/D
    K_et[:,~idl_ok] = empty
    K_et[~idl_ok,:] = empty
    if normalise:
        K_et = normalise_kernel(K_et)
    return K_et

def compute_mean_map_kernel(V, K, empty=np.nan):
    '''Computes the mean map kernel from set characteristics and a kernel, both over the same entities.
    :param V: The set characteristics for each set (rows: entities, column: set)
        The entries of V must be memberships and convertible to logicals.
    :param K: The Gramian of a kernel over the entities over which V indicates set membership.
    
    :result: The Gramian of the mean map kernel. Its size is m x m, where m is the number of sets (columns of V)
    '''
    supports = V.sum(axis=0)
    idl_ok = supports > 0
    
    N = V.T@K@V
    sup = V.sum(axis=0)
    D = sup[:,None]*sup[None,:]
    K_mm = N/D
    K_mm[:,~idl_ok] = empty
    K_mm[~idl_ok,:] = empty
    return K_mm

def compute_mkl_alignment_coeffs_orthogonal_heuristic(W, v):
    alg_single = v/np.diag(W)**.5
    n = len(v)
    alg_top = np.zeros(n-1)
    p = np.argsort(alg_single)[::-1]
    for i in range(1,n):
        a_top = np.array(alg_single)
        a_top[p[i:]] = 0
        alg_top[i-1] = compute_mkl_alignment_value(a_top, W, v)
    idx_max = np.argmax(alg_top)
    a_best = np.array(alg_single)
    a_best[p[idx_max+1:]] = 0
    a_best /= np.linalg.norm(a_best)
    return SimpleNamespace(a_best=a_best, alignment_best=alg_top[idx_max], alignments=alg_top, idx_max=idx_max, alignments_single=alg_single, p_sort=p)

def compute_mkl_alignment_coeffs(W, v, k=10, mu_w=1e-5, eps=1e-5):
    raise NotImplementedError('Do not use. Prefer the heuristic above')
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

def compute_mkl_alignment_value(a,W,v):
    a = a/np.linalg.norm(a)
    denom2 = np.linalg.multi_dot([a,W,a])
    return a@v/denom2**.5

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
        srt = dict(sorted(joined.items(),key=lambda x:x[0]))
        return srt

    def aggregate_subkernels(self, coeffs, k=None, eps=1e-6, dtype=np.float64):
        idl_sel = coeffs>eps
        K = np.zeros((self.n_entities, self.n_entities), dtype=dtype)
        if k is not None:
            p = np.argsort(coeffs)[::-1]
            idl_sel[p[k:]] = 0
        idx_sel = np.where(idl_sel)[0]
        for idx in idx_sel:
            a = coeffs[idx]
            K += a*self.get_subkernel(idx)
        return K
    
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
