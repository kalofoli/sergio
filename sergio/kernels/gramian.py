'''
Created on Mar 16, 2021

:author: janis
'''
import pandas as pd
import numpy as np

from types import SimpleNamespace
from typing import Sequence
from colito.summaries import SummarisableFromFields, SummaryFieldsAppend


class EigenValueAccessor:
    '''A helper to access eigenvalues.'''
    def __init__(self, K, eigs):
        self._K = K
        self._eigs = eigs
    @property
    def available(self): return self._eigs.S.shape[1]
    @property
    def nrows(self): return self._eigs.S.shape[0]
    @property
    def K(self): return self._comp._K
    
    def __getitem__(self, what):
        if isinstance(what, slice):
            max_idx = what.indices(self.nrows)[1]
        elif isinstance(what, np.ndarray):
            if what.dtype == bool:
                max_idx = self.nrows
            else:
                max_idx = np.max(what)
        elif isinstance(what, Sequence):
            max_idx = max(what)
        else:
            max_idx = what
            
        if self.available < max_idx+1:
            self._populate(max_idx+1)
        return self._eigs.l[what], self._eigs.S[:, what]
    def __repr__(self): return f'<{self.__class__.__name__} ({self.available}/{self.nrows})>'
class EigenValueAccessorFull(EigenValueAccessor):
    '''Full rank eigenvalue accessor.
    
    If any eigenvalue is requested, a full spectral decomposition is computed.
    ''' 
    def _populate(self, n):
        l, S = np.linalg.eig(self.K)
        p = np.argsort(l)[::-1]
        self._eigs.l = l[p]
        self._eigs.S = S[:,p]
    
class EigenValueAccessorTop(EigenValueAccessor):
    '''Top rank eigenvalue accessor.
    
    Only the needed eigenvalues are computed.
    
    .. note::
    
        Currently, the pre-existing eigenvalues are discarded if more are requested later, and 
        a new eigs invocation is issued. 
    ''' 
    def _populate(self, n):
        from scipy.sparse.linalg import eigsh
        l, S = eigsh(self.K, n)
        p = np.argsort(l)[::-1]
        self._eigs.l = l[p]
        self._eigs.S = S[:,p]

class Gramian(SummarisableFromFields):
    
    __summary_fields__ = ('rank','dimension') 
    
    def dot(self, X):
        raise NotImplementedError('Override')
    
    @property
    def dimension(self):
        raise NotImplementedError('Override')
    @property
    def rank(self):
        raise NotImplementedError('Override')
    @property
    def eigenvecs(self):
        raise NotImplementedError('Override')
    @property
    def eigenvals(self):
        raise NotImplementedError('Override')
    

class GramianFromArray(Gramian):
    '''Provides a Gramian from an array of its values
    
    >>> n = 10
    >>> K = np.random.RandomState(0).random((n,n)); K = K@K.T
    >>> G = GramianFromArray(K)
    >>> x = np.random.RandomState(0).random(n)
    >>> np.max(G.dot(x) - K@x)
    0.0
    >>> np.max(G.eigenvecs@np.diag(G.eigenvals)@G.eigenvecs.T-G.K) < 1e-10
    True
    >>> G.rank
    10
    '''
    def __init__(self, K, eigs=None, rank=None):
        self._K = K
        if rank is None:
            rank = self.dimension
        self._rank = rank
        if eigs is None:
            eigs = self._populate_eigenvecs(self.rank)
        else:
            self._eigs = eigs 
        super().__init__()
    def dot(self, x):
        return self._K@x
    @property
    def K(self): return self._K
    @property
    def dimension(self): return self._K.shape[0]
    @property
    def rank(self): return self._rank
    @property
    def eigenvecs(self):
        if self._eigs is None:
            self._populate_eigenvecs(n=self.rank)
        return self._eigs[1]
    @property
    def eigenvals(self):
        if self._eigs is None:
            self._populate_eigenvecs(n=self.rank)
        return self._eigs[0]
    def __str__(self):
        return f'{self.dimension}x{self.rank}'
    def __repr__(self):
        return f'<{self.__class__.__name__} {self!s}>'
    def _populate_eigenvecs(self, n):
        if self.rank == self.dimension:
            from scipy.linalg import eigh
            l, S = eigh(self._K)
        else:
            from scipy.sparse.linalg import eigsh
            l, S = eigsh(self._K, self.rank)
        self._rank = len(l)
        p = np.argsort(l[::-1])
        self._eigs = (l[p],S[:,p])

class GramianNystroemFromArray(Gramian):
    '''Provides a low-rank Nystroem approximation of a Gramian
    
    >>> n,k = 10, 5
    >>> K = np.random.RandomState(0).random((n,k)); K = K@K.T
    >>> G = GramianNystroemFromArray(K, k)
    >>> np.linalg.norm(G.K - K) < 1e-10
    True
    >>> l, S = np.linalg.eigh(K)
    >>> p = np.argsort(l)[::-1]; S, l = S[:,p[:k]], l[p[:k]]
    >>> np.linalg.norm(np.abs(G.eigenvecs/S)-1) < 1e-10
    True
    >>> np.linalg.norm(G.eigenvals - l) < 1e-10
    True
    >>> x = np.random.RandomState(1).random(n)
    >>> np.linalg.norm(G.dot(x) - K@x) < 1e-10
    True
    >>> G.rank == k
    True
    >>> G = GramianNystroemFromArray((l,S))
    >>> G
    <GramianNystroemFromArray 10x5>
    
    >>> G = GramianNystroemFromArray((l,S), 7)
    Traceback (most recent call last):
      ...
    ValueError: Requested rank 7 which is greater than available eigenvectors (5)
    >>> G = GramianNystroemFromArray((l,S), 3)
    >>> G
    <GramianNystroemFromArray 10x3>
    >>> np.linalg.norm(G.K - S[:,:3]*l[:3]@S[:,:3].T) < 1e-10
    True
    '''
    def __init__(self, data, rank=None):
        if isinstance(data, np.ndarray):
            if rank is None:
                raise TypeError('Argument "rank" is required by an array initialisation.')
            l,S = self._compute_eigs(data, rank)
        else:
            l,S = data
            if np.max(l.shape) != np.prod(l.shape):
                raise ValueError(f'Spectral data: first argument must be a vector')
            l = l.flatten()
            if S.shape[1] != len(l):
                raise ValueError(f'Spectral data: second argument must be an matrix with {len(l)} column eigenvectors')
            l,S = self._normalise_eigs(l, S)
            if rank is None:
                rank = S.shape[1]
            else:
                if rank > S.shape[1]:
                    raise ValueError(f'Requested rank {rank} which is greater than available eigenvectors ({S.shape[1]})')
                S = S[:,:rank]
                l = l[:rank]
        self._eigs = l,S
        super().__init__()
    def dot(self, x):
        l,S = self._eigs
        return S.dot(l*(x.dot(S)))
    @property
    def K(self):
        l,S = self._eigs
        return S*l@S.T
    @property
    def dimension(self): return self.eigenvecs.shape[0]
    @property
    def rank(self): return self.eigenvecs.shape[1]
    @property
    def eigenvecs(self):
        return self._eigs[1]
    @property
    def eigenvals(self):
        return self._eigs[0]
    def __str__(self):
        return f'{self.dimension}x{self.rank}'
    def __repr__(self):
        return f'<{self.__class__.__name__} {self!s}>'
    
    @classmethod
    def _normalise_eigs(cls, l, S):
        p = np.argsort(l[::-1])
        S = S*np.sign(S[0,:])[None,:]
        return l[p],S[:,p]
        
    @classmethod
    def _compute_eigs(cls, K, rank):
        from scipy.sparse.linalg import eigsh
        l, S = eigsh(K, rank)
        l, S = cls._normalise_eigs(l, S)
        return l, S



class GramianWithNullspace(GramianFromArray):
    '''Provides a Gramian from an array of its values
    
    >>> n = 10
    >>> K = np.random.RandomState(0).random((n,n)); K = K@K.T
    >>> from sergio.kernels.euclidean import RadialBasisFunctionKernel
    >>> V = np.random.RandomState(1).random((n,2))
    >>> G = GramianWithNullspace(K, nullspace=V)
    >>> x = np.random.RandomState(0).random(n)
    >>> np.max(G.dot(x) - K@(x-V@np.linalg.solve(V.T@V,V.T@x)))<1e-14
    True
    >>> np.max(np.abs(G.K@V)) < 1e-14
    True
    >>> G.nullspace_dimension
    2
    >>> G.rank
    10
    '''
    __summary_fields__ = SummaryFieldsAppend(('nullspace_dimension'))
    def __init__(self, K, eigs=None, rank=None, nullspace=None):
        # TODO: make faster
        if nullspace is None or nullspace.shape[1] == 0:
            n = K.shape[0]
            N = np.zeros((n,0))
            super().__init__(K=K, eigs=eigs, rank=rank)
            K_out = K
        else:
            N = np.array(nullspace).astype(float)
            D = (K@N)@np.linalg.solve(N.T@N, N.T)
            K_out = K-D
            super().__init__(K=K_out, rank=rank)
        self._nullspace = N
    @property
    def nullspace(self):
        return self._nullspace
    @property
    def nullspace_dimension(self):
        return self._nullspace.shape[1]

class GramianFromDataset(GramianFromArray):
    '''Computes a kernel on a dataset.
    
    >>> from sergio.data import EntityAttributesWithArrayTarget
    >>> df = pd.DataFrame({'a':np.r_[1,2,3,5],'b':['one','two','three','five'],'c':[3,2,3,2]})
    >>> t = np.c_[[10,2,3,4.],[4,4,4,4],[12,6,7,8.]]
    >>> EntityAttributesWithArrayTarget(attribute_data=df, target=t, target_name='data', name='test', attribute_selection=[1,0,1])
    <EntityAttributesWithArrayTarget[test](4x2/3) target: data(3d float64)>
    >>> ea = EntityAttributesWithArrayTarget(attribute_data=df, name='test', target=t[:,:2].astype(int), attribute_selection={None:False, 'b':1})
    >>> ea
    <EntityAttributesWithArrayTarget[test](4x1/3) target: target(2d int64)>
    >>> from sergio.kernels.euclidean import RadialBasisFunctionKernel
    >>> kern = RadialBasisFunctionKernel()
    >>> dkc = GramianFromDataset(ea, kern)
    >>> dkc.K.shape
    (4, 4)
    '''
    def __init__(self, dataset, kernel):
        self._dataset = dataset
        self._kernel = kernel
        self._params = self.kernel.get_params()
        K = self.kernel.fit_transform(self.dataset.target_data)
        super().__init__(K)
        
    @property
    def dataset(self): return self._dataset
    @property
    def kernel(self): return self._kernel
    
    def __str__(self):
        return f'{super().__str__()} K: {self.kernel} E: {self.dataset} P: {self._params}>' 


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    