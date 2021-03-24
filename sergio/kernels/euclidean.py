'''
Created on Mar 15, 2021

@author: janis
'''

import enum
import numpy as np

from colito.summaries import SummarisableFromFields

from sergio.kernels import Kernel
from colito.resolvers import make_enum_resolver


class DimensionArray():
    '''Accessor that mesh-expands a given array one column at a time.
    
    >>> X = np.c_[[1,2,3,1],[2,3,2,2]]
    >>> da = DimensionArray(X, 3, 0)
    >>> da 
    <DimensionArray [2d int64 4x3]>
    >>> da[0]
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3],
           [1, 1, 1]])
    >>> da = DimensionArray(X[:3,:], 4, 1)
    >>> da
    <DimensionArray [2d int64 4x3]>
    >>> da[0]
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    '''
    def __init__(self, arr, reps, axis):
        self._arr = arr
        self._reps = reps
        self._axis = axis
    def __getitem__(self, d):
        M = np.tile(self._arr[:, d], (self._reps,1))
        if self._axis == 0:
            M = M.T
        return M
    @property
    def shape(self):
        shape = self._arr.shape[0],self._reps
        if self._axis == 1:
            shape = shape[::-1]
        return tuple(shape)
    @property
    def dimension(self): return self._arr.shape[1]
    @property
    def dtype(self): return self._arr.dtype
    def __str__(self):
        return f'[{self.dimension}d {self.dtype} {self.shape[0]}x{self.shape[1]}]'
    def __repr__(self):
        return f'<{self.__class__.__name__} {self!s}>'
    
class EuclideanKernel(Kernel):
    """A kernel over Euclidean vectors
    """
    __collection_title__ = 'Euclidean Kernel'
    __collection_tag__ = None
    
    def transform(self, Y):
        X = self._X
        nx, ny = len(X), len(Y)
        daX, daY = DimensionArray(X, ny, 0), DimensionArray(Y, nx, 1)
        return self.from_dimensions(daX, daY)

class LinearKernel(EuclideanKernel):
    """
    >>> kern = LinearKernel()
    >>> X = np.c_[[1,2,3,1,2],[2,3,2,2,2]]
    >>> kern
    <LinearKernel()>
    >>> kern.fit_transform(X)
    array([[ 5.,  8.,  7.,  5.,  6.],
           [ 8., 13., 12.,  8., 10.],
           [ 7., 12., 13.,  7., 10.],
           [ 5.,  8.,  7.,  5.,  6.],
           [ 6., 10., 10.,  6.,  8.]])
    """
    __collection_title__ = 'Linear Kernel'
    __collection_tag__ = 'linear'
    
    __kernel_params__ = ()

    def from_dimensions(self, daX:DimensionArray, daY:DimensionArray):
        M = np.zeros(daX.shape, float)
        for d in range(daX.dimension):
            M += daX[d] * daY[d]
        return M


class ShiftInvariantKernel(EuclideanKernel):
    '''Shift Invariant kernels'''
    __collection_title__ = 'Shift Invariant Kernel'
    __collection_tag__ = None
    
    def from_dimensions(self, daX:DimensionArray, daY:DimensionArray):
        D = self.distance(daX, daY)
        M = self.from_scalar_distance(D)
        return M
    
    def distance(self, daX, daY):
        raise NotImplementedError('Override to use')
    
    def from_scalar_distance(self, diff:float) -> float:
        raise NotImplementedError('Override to use')


class RadialBasisFunctionKernel(ShiftInvariantKernel):
    '''Radial basis function Kernel. Includes the Gaussian.
    
    >>> X = np.c_[[1,2,3,1,2],[2,3,2,0,1]]
    >>> Y = np.c_[[4,2,-3,-3],[2,1,2,-1]]
    >>> kern = RadialBasisFunctionKernel()
    >>> kern
    <RadialBasisFunctionKernel(sigma=1,kind=<Kind.EXP_QUAD: 1>)>
    >>> K = kern.fit_transform(X, Y)
    >>> Ksol = np.exp(-((X[:,0][:,None]-Y[:,0][None,:])**2+ (X[:,1][:,None]-Y[:,1][None,:])**2)/2)
    >>> np.max(K-Ksol)
    0.0
    '''
    __collection_title__ = 'Radial Basis Function Kernel'
    __collection_tag__ = 'rbf'

    __kernel_params__ = ['sigma','kind']
    
    class Kind(enum.Enum):
        EXP_QUAD = enum.auto()
        EXP = enum.auto()
        INV_QUADRATIC = enum.auto()
        GAUSSIAN = EXP_QUAD
        
    KIND_RESOLVER = make_enum_resolver(Kind)

    def __init__(self, sigma:float=1, kind:Kind="exp_quad"):
        self.sigma = sigma
        self.kind = self.KIND_RESOLVER.resolve(kind)
        
    def distance(self, daX, daY):
        D = np.zeros(daX.shape, float)
        for d in range(daX.dimension):
            dX, dY = daX[d], daY[d]
            XX = dX[:,[0]]**2
            YY = dY[[0],:]**2
            XY = dX*dY
            D += XX - 2*XY + YY
        return D
        
    def from_scalar_distance(self, D):
        cls = self.__class__
        kind = cls.KIND_RESOLVER.resolve(self.kind)
        if kind == cls.Kind.EXP_QUAD:
            K = np.exp(D/(-2*self.sigma**2))
        elif kind == cls.Kind.EXP:
            dist = np.sqrt(D)
            K = np.exp(dist/(-2*self.sigma))
        elif kind == cls.Kind.INV_QUADRATIC:
            K = D**-self.sigma 
        else:
            raise NotImplementedError()
        return K

if __name__ == '__main__':
    import doctest
    doctest.testmod()

