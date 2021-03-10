'''
Created on Oct 7, 2019

@author: janis
'''

import enum
import numpy as np

from cofi.kernel_detail import Kernel, ShiftInvariantKernel
from cofi.utils.resolvers import EnumResolver
from typing import Sequence, Iterable
from warnings import warn
from cofi.kernel_detail.low_pass import truncate_si_kernel, sinc_interp


class IncrementalSIKernel(ShiftInvariantKernel):
    tag ='incremental'
    name = 'Truncated Incremental Shift Invariant Kernel'
    _summary_compact_fields = ['bandwidth','sidewidth']
    
    def __init__(self, bandwidth:int = None, sidewidth:float=None):
        from math import ceil
        if sidewidth is None and bandwidth is None:
            sidewidth = 0.
            bandwidth = 0
        else:
            if bandwidth is None:
                bandwidth = ceil(sidewidth)
            else:
                bandwidth_in = bandwidth
                bandwidth = int(bandwidth)
                if not bandwidth_in == bandwidth:
                    warn(f'Bandwidth specified as {bandwidth_in} of type {type(bandwidth_in)} which is not an int. The value of {bandwidth} will be used.')
                if sidewidth is None:
                    sidewidth = float(bandwidth)
        self.bandwidth:int = bandwidth
        self.sidewidth:float = sidewidth
    
    def _from_array(self, arr):
        w = np.maximum(-np.minimum(abs(arr),self.bandwidth+1)/(self.sidewidth+1)+1,0.)
        return w
    
class IndicatorSIKernel(ShiftInvariantKernel):
    tag = 'indicator'
    name = 'Indicator Shift Invariant Kernel'
    def _from_array(self, value):
        return np.array(value == 0, float)
    bandwidth:int = 0


class AllEquivalentSIKernel(ShiftInvariantKernel):
    tag = 'all-equivalent'
    name = 'Truncated Incremental Shift Invariant Kernel'
    def _from_array(self, value):
        return np.ones(value.shape)

class TruncatedRadialBasisFunctionSIKernel(ShiftInvariantKernel):
    name = 'Truncated Radial Basis Function Kernel'
    tag = 'trbf'
    
    _summary_compact_fields = ['bandwidth','sidewidth','sigma','max_iters','tol','samples']
    
    def __init__(self, sidewidth:float, sigma:float=1., samples:int=None, max_iters:int = 1000, tol:float = 1e-3):
        
        self.sidewidth = float(sidewidth)
        self.bandwidth = int(np.ceil(sidewidth))
        self.samples = 4*2*self.bandwidth+1 if samples is None else samples
        self.sigma = sigma
        self.max_iters = max_iters
        self.tol = tol
        
        self.i = i = np.linspace(0,4*self.bandwidth,self.samples)
        self.t = t = np.exp(-i**2/2/self.sigma**2)
        f = truncate_si_kernel(i,t,d=self.sidewidth, max_iters = self.max_iters, tol=tol)
        self.f = np.real(f)
        
    def _from_array(self, value):
        y = np.abs(value)
        v = np.zeros(len(y))
        idl_pass = y<=self.bandwidth
        y_pass = y[idl_pass]
        if len(np.setdiff1d(y_pass, self.i)) == 0:
            v_pass = np.interp(y_pass,self.i,self.f)
        else:
            i = np.concatenate((-self.i,self.i[1:]))
            t = np.concatenate((self.f,self.f[1:]))
            v_pass = sinc_interp(t, i, u=y_pass)
        v[idl_pass] = v_pass
        return v

class RadialBasisFunctionSIKernel(ShiftInvariantKernel):
    name = 'Radial Basis Function Kernel'
    tag = 'rbf'
    
    _summary_compact_fields = ['sigma']
    
    def __init__(self, sigma:float=1.):
        self.sigma = sigma
        
    def _from_array(self, value):
        y = np.abs(value)
        v = np.exp(-y**2/2/self.sigma**2)
        return v

    
class RadialBasisFunctionKernel(Kernel):
    name = 'Radial Basis Function Kernel'
    tag = 'rbf'

    class Kind(enum.Enum):
        EXP_QUAD = enum.auto()
        EXP = enum.auto()
        QUADRATIC = enum.auto()
        
    KIND_RESOLVER = EnumResolver(Kind)

    _summary_compact_fields = ['sigma','kind']
    
    def __init__(self, sigma:float=1, kind:str="exp_quad"):
        self.sigma = sigma
        self.kind = kind
        
    def set_params(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)
    
    def get_params(self, deep=False):
        return {'sigma':self.sigma, 'kind':self.kind}
    
    def fit(self, X):
        self._X = X
        self._X2 = np.sum(X**2,1)[None,:]

    def _rbf(self, dist2):
        cls = self.__class__
        kind = cls.KIND_RESOLVER.resolve(self.kind)
        if kind == cls.Kind.EXP_QUAD:
            K = np.exp(dist2/(-2*self.sigma**2))
        elif kind == cls.Kind.EXP:
            dist = np.sqrt(dist2)
            K = np.exp(dist/(-2*self.sigma))
        elif kind == cls.Kind.QUADRATIC: # not really an RBF, but, well, whatever
            K = dist2**self.sigma 
        else:
            raise NotImplementedError()
        return K

    def transform(self, X):
        A = X
        A2 = np.sum(A**2,1)[:,None]
        B = self._X
        B2 = self._X2
        
        dist2 = A2 - 2*A.dot(B.T) + B2
        K = self._rbf(dist2)
        return K
    
    def fit_transform(self, X):
        self.fit(X)
        K = self.transform(X)
        # this can be sped up twice, but a) we need the quadratic and b) I'm not going to code this in c.
        return K



class LinearKernel(Kernel):
    name = 'Linear Kernel'
    tag = 'linear'

    _summary_compact_fields = []
    
    def set_params(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)
    
    def get_params(self):
        return {}
    
    def fit(self, X):
        self._X = X

    def transform(self, X):
        return X.dot(self._X.T)
    
    def fit_transform(self, X):
        self.fit(X)
        K = self.transform(X)
        return K
