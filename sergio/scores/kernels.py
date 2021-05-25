'''
Created on Mar 16, 2021

@author: janis
'''

import numpy as np
import pandas as pd

from sergio.scores import Measure, OptimisticEstimator, CachingScoreMixin
from sergio.kernels import Kernel
from sergio.data import EntityAttributesWithAttributeTarget
from colito.summaries import SummaryFieldsAppend, SummarisableFromFields
import enum
from colito.resolvers import make_enum_resolver
from sergio.kernels.gramian import Gramian
import warnings

class ScoreWarning(RuntimeWarning): pass

class GramianEvaluatorMixin:
    __summary_fields__ = SummaryFieldsAppend(('target_name', 'kernel'))
    def __init__(self, *args, target=None, **kwargs):
        super().__init__(*args, **kwargs)
        if target is None:
            if isinstance(self.data, EntityAttributesWithAttributeTarget):
                target_data = self.data.target_data
            else:
                raise TypeError(f'No target specified and data has no default.')
        else:
            target_data = self.data.lookup_attribute(target).data
        self._target_data = target_data
    @property
    def target_data(self) -> pd.Series: return self._target_data
    @property
    def target_name(self) -> str: return self._target_data.name
    @property
    def kernel_value(self) -> Kernel: return self._kernel
    

class KernelMeasure(GramianEvaluatorMixin, Measure):
    __collection_tag__ = None
class KernelOptimisticEstimator(GramianEvaluatorMixin, OptimisticEstimator):
    __collection_tag__ = None

class ComparisonMode(enum.Enum):
    #: Compares the selected subset to its complement (under the population)
    CONTRASTIVE = enum.auto()
    #: Compares the selected subset to the whole population
    ANOMALY = enum.auto()
COMPARISON_RESOLVER = make_enum_resolver(ComparisonMode)

class MaximumMeanDeviationScoreMixin(CachingScoreMixin, SummarisableFromFields):
    __collection_title__ = 'Maximum Mean Deviation'
    __summary_fields__ = SummaryFieldsAppend(('comparison',))
    @property
    def comparison(self): return self._comparison
    @property
    def dimension(self): return self.gramian.dimension
    @property
    def rank(self): return self.gramian.rank
    @property
    def gramian(self): return self._gramian
    @property
    def contrastive(self): return self.comparison == ComparisonMode.CONTRASTIVE
    
    def __init__(self, gramian:Gramian, comparison:ComparisonMode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gramian = gramian
        self._comparison = COMPARISON_RESOLVER.resolve(comparison)
        
class MeasureMaximumMeanDeviation(MaximumMeanDeviationScoreMixin, Measure):
    __collection_title__ = 'Maximum Mean Deviation'
    __collection_tag__ = "mmd"
    __summary_fields__ = SummaryFieldsAppend(('rank',))
    __summary_conversions__ = {'comparison':str}
    
    def __init__(self, gramian, comparison:ComparisonMode, rank=0):
        super().__init__(gramian, comparison)
        if rank == 0:
            rank = self._gramian.rank
        self._rank = rank
        l = self._gramian.eigenvals[:rank]
        S = self._gramian.eigenvecs[:,:rank]
        self._eigs = (l,S)
        
        n = gramian.dimension
        self._c_i = self._eigs[1].sum(0)/n
    
    def evaluate_uncached(self, selector)->float:
        r'''Evaluate the measure.
        
        The computation follows the following formula.
        
        .. math::
        
               f_\mu^{-1}(\mathbf x)\sum_i\lambda_i\left(\mathbf x^\top v_i-mc_i\right)^2
               
        
        where :math:`c_i = \tfrac1n\mathbf e^\top v_i` and :math:`m=\mathbf x^\top\mathbf e`
        '''
        x: np.ndarray = selector.validity
        l, S = self._eigs
        m,n = x.sum(), S.shape[0]
        if m>0 and m!=n:
            fm = m*(n-m)/n if self._comparison == ComparisonMode.CONTRASTIVE else m
            xtv = S[x,:].sum(0)
            quant_paren = xtv-m*self._c_i
            quant_sum = l.dot(quant_paren*quant_paren)
            J = quant_sum/fm
        else:
            J = 0
        return J
    
class OptimisticEstimatorMaximumMeanDeviationSingleDirection(MaximumMeanDeviationScoreMixin, OptimisticEstimator):
    ''' 

    .. note:: This value results by assigning the (squared) lengths of a unit vector parallel to each eigenvector (starting with the largest eigenvalue one),
     so that each component does not exceed a bound on the (squared) length along each eigenvector.
     These bounds are computed on the single direction of an eigenvector as the
     (squares of) the cosine between the eigenvector and its projection on the 0-1 vectors that satisfy the constraints.
       
    '''
    __collection_title__ = 'Maximum Mean Deviation using a single direction bound'
    __collection_tag__ = "mmd-sd"
    __summary_conversions__ = {'comparison':str}
    __summary_fields__ = ('comparison','max_rank')
    
    def __init__(self, gramian, comparison:ComparisonMode, max_rank:int=None):
        super().__init__(gramian, comparison)
        self._eigenvals,S = self._gramian.eigenvals, self._gramian.eigenvecs
        
        if max_rank is None:
            max_rank = S.shape[1]
        else:
            if max_rank>S.shape[1]:
                warnings.warn(f'Requested max_rank {max_rank} which exceeds the available of {S.shape[1]}. Trimming.', RuntimeWarning)
                max_rank = S.shape[1]
        self._max_rank:int = max_rank
        
        idx_sort = np.argsort(S, axis=0) # column-wise sorting permutation
        S_srt = np.take_along_axis(S, idx_sort, axis=0)
        self._eigenvec_sums = S_srt.sum(axis=0)
        self._eigenvec_srt = S_srt
        self._eigenvec_idx_sort = idx_sort
        self._debug = False

    @property
    def max_rank(self): return self._max_rank
    
    def cos_sq_max(self, k, idl_sel, m_run, ztz_run):
        '''Compute the squared cosine of the closest 0-1 vector to th k-th eigenvector using units only from idl_sel'''
        m,n = m_run[-1],len(idl_sel)
        p = self._eigenvec_idx_sort[:,k]  # sorting permutation
        v = self._eigenvec_srt[idl_sel[p],k]
        vte = self._eigenvec_sums[k]
        
        vtx_min = np.cumsum(v)[:m]
        vtx_max = np.cumsum(v[::-1])[:m]
        if self._debug:
            if m == n-1:
                np.testing.assert_almost_equal(vtx_min[-1]+v[-1],v.sum(0),10,'vtx_min is missing something')
                np.testing.assert_almost_equal(v.sum(),v[0]+vtx_max[-1],10,'vtx_max is missing something')
                np.testing.assert_almost_equal(v.sum(),vte,10,'vte is missing something')
            else:
                np.testing.assert_almost_equal(vtx_min[-1],v.sum(0),10,'vtx_min is missing something')
                np.testing.assert_almost_equal(v.sum(),vtx_max[-1],10,'vtx_max is missing something')
        
        vtz_min = vtx_min - vte*m_run/n
        vtz_max = vtx_max - vte*m_run/n
        vtz_sq = np.maximum(vtz_max**2, vtz_min**2)
        cos_sq_max = vtz_sq/ztz_run # the best possible cosine (square) of theta for each eigenvector
        return cos_sq_max

    def evaluate_uncached(self, selector)->float:
        idl_sel: np.ndarray = selector.validity
        n = self.dimension
        rank = self.max_rank
        m = idl_sel.sum()
        if m == 0:
            return 0
        elif m == n:
            m = n-1
        debug = self._debug
        contrastive = self.contrastive
        if debug: cos_sq = np.empty((m,rank))
        m_run = np.arange(1,m+1)
        ztz_run = m_run*(n-m_run)/n
        get_cos_sq_max = lambda k: self.cos_sq_max(k, idl_sel=idl_sel, m_run=m_run, ztz_run=ztz_run)
        
        cos_sq_cur = np.empty(m)
        cos_sq_rem = np.empty(m)
        rayleigh = np.zeros(m)
        
        cos_sq_cur[:] = get_cos_sq_max(0)
        cos_sq_rem[:] = 1 - cos_sq_cur # The remaining angle to assign to an eigenvalue
        rayleigh[:] = cos_sq_cur*self._eigenvals[0]  # The result of the rayleigh quotient (for each size) so far
        max_ev_idx = 0 # maximum eigenvector index needed (because of non-empty cosines)
        if debug: cos_sq[:, max_ev_idx] = cos_sq_cur
        while True:
            max_ev_idx += 1
            if cos_sq_rem.sum() == 0:
                if max_ev_idx < rank:
                    warnings.warn(f'Ending prematurely at rank: {max_ev_idx}/{rank}', ScoreWarning)
                break
            if max_ev_idx == rank:
                break
            cos_sq_max = get_cos_sq_max(max_ev_idx)
            cos_sq_cur[:] = np.minimum(cos_sq_max, cos_sq_rem)
            if debug: cos_sq[:, max_ev_idx] = cos_sq_cur
            cos_sq_rem -= cos_sq_cur
            rayleigh += cos_sq_cur*self._eigenvals[max_ev_idx]
        fm = m_run*(n-m_run)/n if contrastive else m_run
        f_all = ztz_run/fm*rayleigh
        if debug: return pd.DataFrame({'m':m_run,'f':f_all,'cost_sq':list(cos_sq),'rayleigh':rayleigh, 'cos_sq_rem':cos_sq_rem})
        return float(f_all.max())
        
