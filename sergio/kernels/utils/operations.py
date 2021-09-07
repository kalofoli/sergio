'''
Created on Sep 7, 2021

@author: janis
'''

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

from sergio.kernels.gramian import Gramian
from colito.parallel import tqdm
from types import SimpleNamespace

__all__ = [
    'compute_matrix_frobenius','compute_gramian_frobenius','jaccard_kernel',
    'compute_kernel_alignment','normalise_kernel','compute_extended_tanimoto_kernel',
    'compute_mean_map_kernel','compute_mkl_alignment_coeffs_orthogonal_heuristic',
    'compute_mkl_alignment_coeffs','compute_mkl_alignment_value','compute_entity_predicate_similarity'
    ]

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
    outer_seq = tqdm(gramians, desc='Frobenius components') if progress else gramians
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

def compute_entity_predicate_similarity(V):
    X = V.astype(float)
    m = X.sum(axis=1)**-.5
    X = X*m[:,None]
    return X@X.T

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

