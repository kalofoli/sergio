'''
Created on Apr 1, 2020

@author: janis

forked from scipy
'''

from __future__ import division, print_function, absolute_import

from typing import Callable
from collections import namedtuple
import functools
from scipy.sparse.linalg import onenormest


"""Compute the action of the matrix exponential.
"""


import numpy as np

import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg import aslinearoperator,LinearOperator
from scipy.sparse.sputils import is_pydata_spmatrix
from scipy import sparse

class cached_callable(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, wrapped):
        self._name= wrapped.__name__
        self._wrapped = wrapped
        functools.update_wrapper(self, wrapped) 

    def __get__(self, instance, owner):
        # Compute the value
        value = instance._values.get(self._name, None)
        if isinstance(value, Callable):
            value = value()
        if value is None:
            value = self._wrapped(instance)
        
        setattr(instance, self._name, value)
        return value
    
class ConstantOperations():
    def __init__(self, A, norm1=None, norminf=None, trace=None):
        self._values = {'norm1':norm1, 'norminf':norminf,'trace':trace}
        self.A = A
    
    @cached_callable
    def norm1(self):
        '''Compute an estimate of the norm if no other solution exists.'''
        raise NotImplementedError()
    @cached_callable
    def norminf(self):
        raise NotImplementedError()
    @cached_callable
    def trace(self):
        raise NotImplementedError()

ExpmmParameters = namedtuple('ExpmvmParameters',('mu','m_star','s','tol'))

def compute_expmm_parameters(A, t, n0=1, mu=None, tol=None, ell=2, m_max=55, co=None):
    if co is None:
        co = ConstantOperations(A)
    
    if tol is None:
        # TODO: replace defaults
        u_d = 2**-53
        tol = u_d
    
    A = aslinearoperator(A)
    n = A.shape[0]
    if mu is None:
        mu = co.trace / float(n)
    if mu > 0:
        ident = LinearOperator((n,n), mv=lambda x:x)
        A = A - mu * ident
        co.trace -= n*mu
    if t*co.norm1 == 0:
        m_star, s = 0, 1
    else:
        norm_info = LazyOperatorNormInfo(t*A, ell=ell, co = co)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell, m_max=m_max)
    return ExpmmParameters(mu=mu,m_star=m_star,s=s,tol=tol)
    

def expm_multiply(A, B, t=1.0, balance=False, tol=None, mu=None, co = None, m_max=55):
    """
    Compute the action of the matrix exponential at a single time point.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    t : float
        A time point.
    parameters: a namedtuple with fields defining the algorithm parameters:
        ('mu','m_star','s','tol')
    balance : bool
        Indicates whether or not to apply balancing.

    Returns
    -------
    F : ndarray
        :math:`e^{t A} B`

    Notes
    -----
    This is algorithm (3.2) in Al-Mohy and Higham (2011).

    """
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('the matrices A and B have incompatible shapes')
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    
    parameters = compute_expmm_parameters(A, t, n0, tol=tol, mu=mu, co=co, m_max=m_max)
    
    return expm_multiply_parameters(A, B, t, **parameters._asdict(), balance=balance)


def expm_multiply_parameters(A, B, t, mu, m_star, s, tol, balance=False):
    """
    A helper function. Do not call directly.
    """
    norminf = lambda X: np.linalg.norm(X, np.inf)
    if balance:
        raise NotImplementedError
    if tol is None:
        u_d = 2 ** -53
        tol = u_d
    F = B
    eta = np.exp(t*mu / float(s))
    for i in range(s):
        c1 = norminf(B)
        for j in range(m_star):
            coeff = t / float(s*(j+1))
            B = coeff * A.dot(B)
            c2 = norminf(B)
            F = F + B
            if c1 + c2 <= tol * norminf(F):
                break
            c1 = c2
        F = eta * F
        # print(f'Iter {i:5} m*:{m_star:5} s:{s:5} |.|:{np.linalg.norm(B-F):5}')
        B = F
    return F


# This table helps to compute bounds.
# They seem to have been difficult to calculate, involving symbolic
# manipulation of equations, followed by numerical root finding.
_theta = {
        # The first 30 values are from table A.3 of Computing Matrix Functions.
        1: 2.29e-16,
        2: 2.58e-8,
        3: 1.39e-5,
        4: 3.40e-4,
        5: 2.40e-3,
        6: 9.07e-3,
        7: 2.38e-2,
        8: 5.00e-2,
        9: 8.96e-2,
        10: 1.44e-1,
        # 11
        11: 2.14e-1,
        12: 3.00e-1,
        13: 4.00e-1,
        14: 5.14e-1,
        15: 6.41e-1,
        16: 7.81e-1,
        17: 9.31e-1,
        18: 1.09,
        19: 1.26,
        20: 1.44,
        # 21
        21: 1.62,
        22: 1.82,
        23: 2.01,
        24: 2.22,
        25: 2.43,
        26: 2.64,
        27: 2.86,
        28: 3.08,
        29: 3.31,
        30: 3.54,
        # The rest are from table 3.1 of
        # Computing the Action of the Matrix Exponential.
        35: 4.7,
        40: 6.0,
        45: 7.2,
        50: 8.5,
        55: 9.9,
        }


class LazyOperatorNormInfo:
    """
    Information about an operator is lazily computed.

    The information includes the exact 1-norm of the operator,
    in addition to estimates of 1-norms of powers of the operator.
    This uses the notation of Computing the Action (2011).
    This class is specialized enough to probably not be of general interest
    outside of this module.

    """
    def __init__(self, A, ell=2, scale=1, normest_max_iters = 5, normest_num_directions=2, co=None):
        """
        Provide the operator and some norm-related information.

        Parameters
        ----------
        A : linear operator
            The operator of interest.
        normest_max_iters : int, optional
            The maximum iterations allowed in one estimation of Lp-norm.
        normest_num_directions : int, optional
            The number of candidate directions (vectors) used in one estimation of Lp-norm.
        scale : int, optional
            If specified, return the norms of scale*A instead of A.

        """
        self._A = aslinearoperator(A)
        self._ell = ell
        self._normest_max_iters = normest_max_iters
        self._normest_num_directions = normest_num_directions
        self._d = {}
        self._scale = scale
        self._co = ConstantOperations(A) if co is None else co

    @property
    def normest_num_directions(self):
        '''Aka the l parameter of Eq. 3.13'''
        return self._normest_num_directions
    def set_scale(self,scale):
        """
        Set the scale parameter.
        """
        self._scale = scale

    def onenorm(self):
        """
        Compute the exact 1-norm.
        """
        return self._scale*self._co.norm1

    def d(self, p):
        """
        Lazily estimate d_p(A) ~= || A^p ||^(1/p) where ||.|| is the 1-norm.
        """
        if p not in self._d:
            est = onenormest(self._A ** p, itmax=self._normest_max_iters, t=self._normest_num_directions)
            self._d[p] = est ** (1.0 / p)
        return self._scale*self._d[p]

    def alpha(self, p):
        """
        Lazily compute max(d(p), d(p+1)).
        """
        return max(self.d(p), self.d(p+1))

def _compute_cost_div_m(m, p, norm_info):
    """
    A helper function for computing bounds.

    This is equation (3.10).
    It measures cost in terms of the number of required matrix products.

    Parameters
    ----------
    m : int
        A valid key of _theta.
    p : int
        A matrix power.
    norm_info : LazyOperatorNormInfo
        Information about 1-norms of related operators.

    Returns
    -------
    cost_div_m : int
        Required number of matrix products divided by m.

    """
    return int(np.ceil(norm_info.alpha(p) / _theta[m]))


def _compute_p_max(m_max):
    """
    Compute the largest positive integer p such that p*(p-1) <= m_max + 1.

    Do this in a slightly dumb way, but safe and not too slow.

    Parameters
    ----------
    m_max : int
        A count related to bounds.

    """
    sqrt_m_max = np.sqrt(m_max)
    p_low = int(np.floor(sqrt_m_max))
    p_high = int(np.ceil(sqrt_m_max + 1))
    return max(p for p in range(p_low, p_high+1) if p*(p-1) <= m_max + 1)


def _fragment_3_1(norm_info, n0, tol, m_max=55, ell=2):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    norm_info : LazyOperatorNormInfo
        Information about norms of certain linear operators of interest.
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    tol : float
        Expected to be
        :math:`2^{-24}` for single precision or
        :math:`2^{-53}` for double precision.
    m_max : int
        A value related to a bound.
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.

    Returns
    -------
    best_m : int
        Related to bounds for error control.
    best_s : int
        Amount of scaling.

    Notes
    -----
    This is code fragment (3.1) in Al-Mohy and Higham (2011).
    The discussion of default values for m_max and ell
    is given between the definitions of equation (3.11)
    and the definition of equation (3.12).

    """
    if ell < 1:
        raise ValueError('expected ell to be a positive integer')
    best_m = None
    best_s = None
    if _condition_3_13(norm_info, n0, m_max):
        for m, theta in _theta.items():
            s = int(np.ceil(norm_info.onenorm() / theta))
            if best_m is None or m * s < best_m * best_s:
                best_m = m
                best_s = s
    else:
        # Equation (3.11).
        for p in range(2, _compute_p_max(m_max) + 1):
            for m in range(p*(p-1)-1, m_max+1):
                if m in _theta:
                    s = _compute_cost_div_m(m, p, norm_info)
                    if best_m is None or m * s < best_m * best_s:
                        best_m = m
                        best_s = s
        best_s = max(best_s, 1)
    return best_m, best_s


def _condition_3_13(norm_info, n0, m_max):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    m_max : int
        A value related to a bound.
    
    Returns
    -------
    value : bool
        Indicates whether or not the condition has been met.

    Notes
    -----
    This is condition (3.13) in Al-Mohy and Higham (2011).

    """
    ell = norm_info.normest_num_directions
    # This is the rhs of equation (3.12).
    p_max = _compute_p_max(m_max)
    a = 2 * ell * p_max * (p_max + 3)

    # Evaluate the condition (3.13).
    b = _theta[m_max] / float(n0 * m_max)
    return norm_info.onenorm() <= a * b


def expm_multiply_interval(A, B, start=None, stop=None,
        num=None, endpoint=None, balance=False):
    """
    Compute the action of the matrix exponential at multiple time points.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix or vector to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    endpoint : bool, optional
        If True, `stop` is the last time point.  Otherwise, it is not included.

    Returns
    -------
    expm_A_B : ndarray
         The result of the action :math:`e^{t_k A} B`.

    Notes
    -----
    The optional arguments defining the sequence of evenly spaced time points
    are compatible with the arguments of `numpy.linspace`.

    The output ndarray shape is somewhat complicated so I explain it here.
    The ndim of the output could be either 1, 2, or 3.
    It would be 1 if you are computing the expm action on a single vector
    at a single time point.
    It would be 2 if you are computing the expm action on a vector
    at multiple time points, or if you are computing the expm action
    on a matrix at a single time point.
    It would be 3 if you want the action on a matrix with multiple
    columns at multiple time points.
    If multiple time points are requested, expm_A_B[0] will always
    be the action of the expm at the first time point,
    regardless of whether the action is on a vector or a matrix.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
           "Computing the Action of the Matrix Exponential,
           with an Application to Exponential Integrators."
           SIAM Journal on Scientific Computing,
           33 (2). pp. 488-511. ISSN 1064-8275
           http://eprints.ma.man.ac.uk/1591/

    .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
           "Computing Matrix Functions."
           Acta Numerica,
           19. 159-208. ISSN 0962-4929
           http://eprints.ma.man.ac.uk/1451/

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import expm, expm_multiply_ts
    >>> A = csc_matrix([[1, 0], [0, 1]])
    >>> A.todense()
    matrix([[1, 0],
            [0, 1]], dtype=int64)
    >>> B = np.array([np.exp(-1.), np.exp(-2.)])
    >>> B
    array([ 0.36787944,  0.13533528])
    >>> expm_multiply_ts(A, B, start=1, stop=2, num=3, endpoint=True)
    array([[ 1.        ,  0.36787944],
           [ 1.64872127,  0.60653066],
           [ 2.71828183,  1.        ]])
    >>> expm(A).dot(B)                  # Verify 1st timestep
    array([ 1.        ,  0.36787944])
    >>> expm(1.5*A).dot(B)              # Verify 2nd timestep
    array([ 1.64872127,  0.60653066])
    >>> expm(2*A).dot(B)                # Verify 3rd timestep
    array([ 2.71828183,  1.        ])

    Notes
    -----
    This is algorithm (5.2) in Al-Mohy and Higham (2011).

    There seems to be a typo, where line 15 of the algorithm should be
    moved to line 6.5 (between lines 6 and 7).

    """
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('the matrices A and B have incompatible shapes')
    ident = _ident_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    mu = _trace(A) / float(n)

    # Get the linspace samples, attempting to preserve the linspace defaults.
    linspace_kwargs = {'retstep': True}
    if num is not None:
        linspace_kwargs['num'] = num
    if endpoint is not None:
        linspace_kwargs['endpoint'] = endpoint
    samples, step = np.linspace(start, stop, **linspace_kwargs)

    # Convert the linspace output to the notation used by the publication.
    nsamples = len(samples)
    if nsamples < 2:
        raise ValueError('at least two time points are required')
    q = nsamples - 1
    h = step
    t_0 = samples[0]
    t_q = samples[q]

    # Define the output ndarray.
    # Use an ndim=3 shape, such that the last two indices
    # are the ones that may be involved in level 3 BLAS operations.
    X_shape = (nsamples,) + B.shape
    X = np.empty(X_shape, dtype=np.result_type(A.dtype, B.dtype, float))
    t = t_q - t_0
    A = A - mu * ident
    A_1_norm = _exact_1_norm(A)
    ell = 2
    norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)

    # Compute the expm action up to the initial time point.
    X[0] = expm_multiply_parameters(A, B, t_0, mu, m_star, s)

    # Compute the expm action at the rest of the time points.
    if q <= s:
        return _expm_multiply_interval_core_0(A, X,
                h, mu, q, norm_info, tol, ell,n0)
    elif not (q % s):
        return _expm_multiply_interval_core_1(A, X,
                h, mu, m_star, s, q, tol)
    elif (q % s):
        return _expm_multiply_interval_core_2(A, X,
                h, mu, m_star, s, q, tol)
    else:
        raise Exception('internal error')


def _expm_multiply_interval_core_0(A, X, h, mu, q, norm_info, tol, ell, n0):
    """
    A helper function, for the case q <= s.
    """

    # Compute the new values of m_star and s which should be applied
    # over intervals of size t/q
    if norm_info.onenorm() == 0:
        m_star, s = 0, 1
    else:
        norm_info.set_scale(1./q)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
        norm_info.set_scale(1)

    for k in range(q):
        X[k+1] = expm_multiply_parameters(A, X[k], h, mu, m_star, s)
    return X, 0


def _expm_multiply_interval_core_1(A, X, h, mu, m_star, s, q, tol):
    """
    A helper function, for the case q > s and q % s == 0.
    """
    d = q // s
    input_shape = X.shape[1:]
    K_shape = (m_star + 1, ) + input_shape
    K = np.empty(K_shape, dtype=X.dtype)
    for i in range(s):
        Z = X[i*d]
        K[0] = Z
        high_p = 0
        for k in range(1, d+1):
            F = K[0]
            c1 = _exact_inf_norm(F)
            for p in range(1, m_star+1):
                if p > high_p:
                    K[p] = h * A.dot(K[p-1]) / float(p)
                coeff = float(pow(k, p))
                F = F + coeff * K[p]
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i*d] = np.exp(k*h*mu) * F
    return X, 1


def _expm_multiply_interval_core_2(A, X, h, mu, m_star, s, q, tol):
    """
    A helper function, for the case q > s and q % s > 0.
    """
    d = q // s
    j = q // d
    r = q - d * j
    input_shape = X.shape[1:]
    K_shape = (m_star + 1, ) + input_shape
    K = np.empty(K_shape, dtype=X.dtype)
    for i in range(j + 1):
        Z = X[i*d]
        K[0] = Z
        high_p = 0
        if i < j:
            effective_d = d
        else:
            effective_d = r
        for k in range(1, effective_d+1):
            F = K[0]
            c1 = _exact_inf_norm(F)
            for p in range(1, m_star+1):
                if p == high_p + 1:
                    K[p] = h * A.dot(K[p-1]) / float(p)
                    high_p = p
                coeff = float(pow(k, p))
                F = F + coeff * K[p]
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i*d] = np.exp(k*h*mu) * F
    return X, 2
