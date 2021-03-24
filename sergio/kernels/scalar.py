'''
Created on Mar 15, 2021

@author: janis
'''



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
    
    def from_scalar_distance(self, arr):
        w = np.maximum(-np.minimum(abs(arr),self.bandwidth+1)/(self.sidewidth+1)+1,0.)
        return w
    
class IndicatorSIKernel(ShiftInvariantKernel):
    tag = 'indicator'
    name = 'Indicator Shift Invariant Kernel'
    def from_scalar_distance(self, value):
        return np.array(value == 0, float)


class AllEquivalentSIKernel(ShiftInvariantKernel):
    tag = 'all-equivalent'
    name = 'Truncated Incremental Shift Invariant Kernel'
    def from_scalar_distance(self, value):
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
