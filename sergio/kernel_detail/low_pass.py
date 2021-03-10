'''
Created on Jun 10, 2020

@author: janis
'''

import numpy as np
from numpy.fft import fft,ifft
from scipy.linalg import toeplitz, eig


def cpad(x):
    return np.concatenate((x,x[:0:-1]))
unpad = lambda x:x[:int(np.ceil(len(x)/2))]

pfft = lambda x:fft(cpad(x))
uifft = lambda f:unpad(ifft(f))
#flt = lambda x,fl:filtfilt(fl,1,x)


def dykstra(projections,x,args=[],tol=1e-3, max_iters=1000,cb=None):
    n_p = len(projections)
    n = len(x)
    D = np.zeros([n,n_p])
    
    for ii in range(max_iters):
        x_0 = x
        for pi,p in enumerate(projections):
            d = D[:,pi]
            x_p = p(x + d,*args)
            d = D[:,pi] = x - x_p
            x = x_p
        norms = np.sqrt(D**2).sum(axis=0)
        # print(f'Iteration {ii+1}/{max_iters}: norms {norms}')
        if cb is not None:
            cb(x_0,x,D)
        if max(norms) < tol:
            break
    return x


def truncate_si_kernel(i,x,d,max_iters=1000, tol=1e-3):
    def proj_bl(x,i):
        x_p = np.array(x)
        x_p[np.abs(i)>d] = 0
        return x_p
    def proj_pd(x,i):
        n2p1 = len(x)
        n = int(np.floor(n2p1/2))
        T = toeplitz(x[n:])
        l,S = eig(T)
        is_p = np.real(l)>0
        T_p = (S[:,is_p]*l[None,is_p]@S[:,is_p].T)
        h = np.real(T_p[0,:])
        return np.concatenate([h[::-1],h[1:]])
    def proj_po(x,i):
        yf = unpad(pfft(x))
        idx_neg = np.where(yf<0)[0]
        if len(idx_neg)>0:
            idx_first_neg = idx_neg[0]
            yf[idx_first_neg:] = 0
            y = uifft(cpad(yf))
        else:
            y = x
        return y
    y = dykstra([proj_po,proj_bl],x,[i],tol=tol,max_iters=max_iters)
    return y


def sinc_interp(x, s, u):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")
    
    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html        
    """
    
    if len(x) != len(s):
        raise TypeError(f'x and s must be the same length')
    
    # Find the period
    T = s[1] - s[0]
    
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, None], (1, len(u)))
    y = np.dot(x, np.sinc(sincM/T))
    return y
