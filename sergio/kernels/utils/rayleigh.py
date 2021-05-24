'''
Created on May 20, 2021

@author: janis
'''
import numpy as np


def make_posdef(n, k=None, seed=0, full=False):
    '''Make a positive definite matrix
    
    :param k: rank
    :param full: Return a complete eigenvector basis'''
    if k is None:
        k = n
    rs = np.random.RandomState(seed)
    A = rs.random((n,n if full else k))
    S = np.linalg.qr(A)[0]
    l = np.r_[np.sort(rs.random(k))[::-1]]
    return l,S

class Rayleigh:
    '''
    >>> rs = np.random.RandomState(1)
    >>> l,S = make_posdef(9,7)
    >>> r = Rayleigh(l=l, S=S)
    >>> r
    <Rayleigh 9x7>
    >>> A = r._S*r._l@r._S.T
    >>> np.abs(r(r.S[:,0]) - r.l[0]).max()<1e-16
    True
    >>> p = np.r_[np.pi/16,np.pi/6,np.pi/4,np.pi/3,0,0]
    >>> z = Rayleigh._rotate(S, p)
    >>> np.abs(r(z)-r.value_phis(p))<1e-15
    True
    
    >>> x = rs.random(r.n)
    >>> h = 1e-6
    >>> p_fw = p + np.eye(len(p),1)[:,0]*h
    >>> np.abs(r.inner_phis(p, x) - x.dot(r.rotate(p))).max()<1e-16
    True
    >>> np.abs(r.inner_ndiff(p,x, eps=h)[0] - x@(r.rotate(p_fw)-r.rotate(p))/h)<h
    True
    >>> np.abs(r.inner_ndiff(p) - r.inner_sdiff(p)).max()<1e-6
    True

    >>> np.abs(r.value_ndiff(p, eps=h)[0] - (r.value_phis(p_fw)-r.value_phis(p))/h) < h
    True
    >>> np.abs(r.value_ndiff(p, eps=h) - r.value_sdiff(p)).max()<h
    True
    '''
    def __init__(self, S, l):
        self._l, self._S = np.array(l), np.array(S)
        assert self._S.shape[1] == self._l.shape[0],f'Dimensions of eigenvalues/eigenvectors do not match'
        self._S = self._S*((self._S.sum(axis=0)>0)*2-1)
    @property
    def A(self): return self._S*self._l@self._S
    @property
    def S(self): return self._S.copy()
    @property
    def l(self): return self._l.copy()
    @property
    def n(self): return self._S.shape[0]
    @property
    def k(self): return len(self._l)
    def __call__(self, x):
        return x@self._S*self._l@self._S.T.dot(x)/np.linalg.norm(x)
    def value_phis(self, phis):
        return self._value_phis(self._l, phis)
    def value_sdiff(self, phis):
        '''Symbolically compute the gradient of the Rayleigh quotient w.r.t. the phis'''
        return self._value_sdiff(self._l, phis)
    def value_ndiff(self, phis, eps=1e-6):
        '''Numerically compute the gradient of the Rayleigh quotient w.r.t. the phis'''
        return self._value_ndiff(self._l, phis, eps=eps)
    def rotate(self, phis, full=False):
        '''Apply the phi rotations on the eigenvectors'''
        return self._rotate(self._S, phis, full)
    def inner_phis(self, phis, x=None):
        '''Compute the inner product of the rotated aigenvector and a vector'''
        if x is None:
            x = np.ones(self.n)
        return x.dot(self.rotate(phis))
    def inner_ndiff(self, phis, x=None, eps=1e-9):
        '''Numerically compute the gradient of the inner product with a rotated vector w.r.t. the phis'''
        from scipy.optimize import approx_fprime
        g = approx_fprime(phis, lambda p: self.inner_phis(p, x), eps)
        return g
    def inner_sdiff(self, phis, x=None):
        '''Symbolically compute the gradient of the inner product with a rotated vector w.r.t. the phis'''
        k = self.k
        if x is None:
            x = np.ones(self.n)
        g = np.zeros(k-1)
        eta = x@self._S
        for i in range(k-1):
            x = self._rotate(eta[None,:], phis[:i], full=True)
            x = self._apply_rotation(x, phis[i]+np.pi/2)
            g[i] = x.flat[0] * np.cos(phis[i+1:]).prod()
        return g
    @classmethod
    def _value_phis(cls, l, phis):
        '''Compute the Rayleigh quotient given the vector rotations of the eigenvectors'''
        z = cls._rotate_square(l[None,:], phis)
        return z[0]
    @classmethod
    def _value_ndiff(cls, l, phis, eps=1e-6):
        '''Numerically compute the gradient of the Rayleigh quotient w.r.t. the phis'''
        from scipy.optimize import approx_fprime
        return approx_fprime(phis, lambda x: cls._value_phis(l, x), eps)
    @classmethod
    def _value_sdiff(cls, l, phis):
        '''Symbolically compute the gradient of the Rayleigh quotient w.r.t. the phis'''
        l_t = l[None, :]
        n = len(phis)
        g = np.zeros(n)
        def apply_diff(x,t): 
            return cls._apply_scaling(x, -np.sin(2*t), np.sin(2*t))

        for i in range(n-1):
            z = cls._rotate_square(l_t, phis[:i], full=True)
            x = apply_diff(z, phis[i])
            g[i] = x.flat[0] * np.cos(phis[i+1:]).prod()**2

        return g
    @classmethod
    def _apply_scaling(cls,S,a,b):
        '''Sum the first two columns (scaled by a,b) and return the result concatenated to the rest'''
        return np.c_[S[:,0]*a+S[:,1]*b,S[:,2:]]
    @classmethod
    def _apply_rotation(cls,S,t):
        '''Rotate the first two columns and return the result concatenated to the rest'''
        return cls._apply_scaling(S,np.cos(t),np.sin(t))

    @classmethod
    def _rotate(cls, V, phis, full=False):
        z = V[:,0]
        for i,phi in enumerate(phis, 1):
            z = z*np.cos(phi) + V[:,i]*np.sin(phi)
        if full:
            n = len(phis)
            return np.c_[z,V[:,n+1:]]
        else:
            return z
    @classmethod
    def _rotate_square(cls, V, phis, full=False):
        z = V[:,0]
        n = phis.shape[0]
        for i,phi in enumerate(phis, 1):
            z = z*np.cos(phi)**2 + V[:,i]*np.sin(phi)**2
        if full:
            n = len(phis)
            return np.c_[z,V[:,n+1:]]
        else:
            return z
    
    def __repr__(self): return f'<{self.__class__.__name__} {self.n}x{self.k}>'


class GD:
    def __init__(self, r):
        self._r = r
    
    
    def __call__(self, p=None, mu_0=.1, maxit=1000, a=.01, tol=1e-5, eta=1e-3):
        r = self._r
        if p is None:
            p = np.zeros(r.k-1)
        mu = mu_0*np.ones(r.n) if np.ndim(mu_0)==0 else mu_0
        iters = 0
        x = r.rotate(p)
        steps = 0
        while iters < maxit:
            g_v = r.value_sdiff(p)
            y = x[x<0].sum()
            g_c = r.inner_sdiff(p, mu*(x<0))
            
            g = g_v + g_c
            p_nxt = p + eta*g/(steps+1)**.5
            x_nxt = r.rotate(p_nxt)
            mu_nxt = mu
            if np.linalg.norm(x-x_nxt)<tol:
                print('Converged')
                idl_neg = x<0
                if np.any(idl_neg):
                    mu_nxt[idl_neg] = mu[idl_neg]*(1+a)
                    print(f'{iters:4d}: Inc y: {y} and new mu: {mu_nxt}')
                    steps = 0
                else:
                    break
            self.updated(x,x_nxt,p,p_nxt, mu, mu_nxt)
            x = x_nxt
            p = p_nxt
            mu = mu_nxt
            iters += 1
            steps += 1
        return x
    def updated(self, x, x_nxt, p, p_nxt, mu, mu_nxt):
        pass
import functools
from collections import namedtuple
Step = namedtuple('Step',('x','phis','mu'))
class TrackingGD(GD):
    def __init__(self, r):
        super().__init__(r=r)
        self._results = []
    @functools.wraps(GD.__call__)
    def __call__(self, *args, **kwargs):
        self._results.clear()
        return super().__call__(*args, **kwargs)
    def updated(self, x, x_nxt, p, p_nxt, mu, mu_nxt):
        s = Step(x=self._r(x_nxt), phis=p_nxt, mu = np.array(mu_nxt))
        self._results.append(s)

    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
