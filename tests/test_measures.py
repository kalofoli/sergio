'''
Created on Jan 14, 2018

@author: janis
'''
import numpy as np
import pandas as pd

import unittest

from sergio.language import StaticSelector
from sergio.scores.kernels import ComparisonMode
from types import SimpleNamespace

class ObjectiveSpectral():
    
    def __init__(self, G, contrastive):
        l,S = self._eigs = G.eigenvals,G.eigenvecs
        self._contrastive = contrastive
        self._vte = S.sum(0)
    
    def __call__(self, x):
        n,p = x.shape
        m = x.sum(0)
        l,S = self._eigs
        rank = S.shape[1]
        vtx = np.zeros((p,rank))
        for k,v in enumerate(S.T):
            vtx[:,k] = np.sum(v[:,None]*x,axis=0)
        vtz = vtx - m[:,None]/n*self._vte[None,:]
        par_sq = vtz**2
        fm = m*(n-m)/n if self._contrastive else m
        return par_sq@l/fm
    
    def components(self, x):
        n = x.shape[0]
        x = x == 1
        m = x.sum(axis=0)
        ztz = m*(n-m)/n
        l, S = self._eigs
        vtz = (S[:,None,:]*x[:,:,None]).sum(axis=0)-m[:,None]/n*self._vte[None,:]
        cost_sq = vtz**2/ztz[:,None]
        fm = m*(n-m)/n if self._contrastive else m
        f = cost_sq@l*ztz/fm
        return pd.DataFrame({'m':m, 'ztz':ztz, 'cost_sq':list(cost_sq), 'fm':fm, 'f':f})


def to_binary(a,k = None):
    if k is None:
        k = int(np.floor(np.log2(np.max(a)))) + 1

    def vectorised(a):
        b = int(np.ceil(k/8))
        n = len(a)
        I = np.empty((n,b),np.uint8)
        for i in range(b):
            I[:,i] = np.bitwise_and(a, 0xff)
            a = np.right_shift(a, 8)
        A = np.unpackbits(I,1,bitorder='little',count=k).T
        return A
    if np.isscalar(a):
        x = vectorised([a])[:,0]
    else:
        x = vectorised(a)
    return x
def binary_combinations(k,skip=0):
    n = 2**k
    a = np.arange(skip,n)
    A = to_binary(a,k)
    return A

def mk_random(seed=None):
    if seed is None:
        rs = np.random
    else:
        if isinstance(seed, int):
            rs = np.random.RandomState(seed)
        else:
            rs = seed
    return rs

def mk_feas(n, seed=None):
    rs = mk_random(seed)
    return rs.randint(0,2,n)

def process_all(X,l,S, contrastive, idl = None):
    n = S.shape[0]
    if idl is None:
        idl = np.ones(n, bool)
    obj = ObjectiveSpectral(l, S, contrastive)
    vte = S.sum(0)
    def row(x):
        x = x == 1
        m = x.sum()
        ztz = m*(n-m)/n
        vtx = S[x,:].sum(0)
        vtz = vtx-m/n*vte
        cost_sq = vtz**2/ztz
        fm = m*(n-m)/n if contrastive else m
        return {'m':m, 'ztz':ztz, 'vtx':vtx, 'cost_sq':cost_sq, 'fm':fm, 'f':obj(x)}
    
    if np.sum(~idl):
        rows = [row(x) for x in X.T if np.all(x[~idl]) == False]
    else:
        rows = [row(x) for x in X.T]
    return pd.DataFrame(rows)
    

def exact_best(grammian, contrastive = True):
    l, S = grammian.eigenvals, grammian.eigenvecs 
    n = S.shape[0]
    X = binary_combinations(n)
    df_all = process_all(X,l,S, contrastive)
    def getmaxf(df):
        df = df.loc[df.f.max()==df.f,:]
        df.index.name='opt'
        return df.reset_index()
    df_opt = df_all.groupby('m').apply(getmaxf).reset_index(drop=True)
    return df_opt

class TestMeasures(unittest.TestCase):
    def make_kernel_data(self, n, rank=None, seed = 0):
        rs = mk_random(seed)
        if rank is None:
            rank = n
        A = rs.random((n,rank))
        K = A@A.T
        return K
    
    def make_gaussian_data(self, n, seed = 0):
        rs = mk_random(seed)
        y = rs.randn(n)
        return y
    
    def get_selector_combinations(self, n, n_var, m_fix=None, seed=0):
        rs = mk_random(seed)
        n_fix = n - n_var
        if m_fix is None:
            m_fix = int((n-n_var)/2)
        idx_fix = rs.choice(n, n_fix, replace=False)
        idl_fix = np.zeros(n, bool)
        idl_fix[idx_fix] = True
        
        idx_val_fix = rs.choice(n_fix, m_fix)
        val_fix = np.zeros(n_fix, bool)
        val_fix[idx_val_fix] = True
        
        val_var = binary_combinations(n_var)
        
        val_all = np.empty((n,val_var.shape[1]), bool)
        val_all[idl_fix,:] = val_fix[:,None]
        val_all[~idl_fix,:] = val_var
        return val_all, ~idl_fix

    def eval_all(self, meas, n, m, seed=0):
        rs = mk_random(seed)
        idx_sels_all, idl_sel = self.get_selector_combinations(n, m, 0, seed=rs)
        def evaluate_score(meas, v):
            selector = StaticSelector(v)
            v = meas.evaluate_uncached(selector)
            return v
        meas_vals = np.r_[[evaluate_score(meas, idx_sels_all[:,i]) for i in range(idx_sels_all.shape[1])]]
        return SimpleNamespace(meas_vals = meas_vals, idl_sel = idl_sel, idx_sels_all= idx_sels_all)

    def test_meanshift(self, seed=0):
        from sergio.scores.scalars import MeasureCoverageMeanShift, OptimisticEstimatorCoverageMeanShift
        
        n = 100
        rs = mk_random(seed)
        y = self.make_gaussian_data(n, rs)
        meas = MeasureCoverageMeanShift(target_data = y, target_name='test')
        res = self.eval_all(meas, n, 13, rs)
        meas_vals, idl_sel = res.meas_vals, res.idl_sel
        
        oest = OptimisticEstimatorCoverageMeanShift(target_data = y, target_name='test')
        oest_val = oest.evaluate_uncached(StaticSelector(idl_sel))
        self.assertAlmostEqual(max(meas_vals), oest_val, 6,'Mean shift oest fail')
        
        
    def test_mmd(self, seed=0):
        from sergio.kernels.gramian import GramianFromArray
        from sergio.scores.kernels import MeasureMaximumMeanDeviation, OptimisticEstimatorMaximumMeanDeviationSingleDirection
        
        n = 100
        rs = mk_random(seed)
        
        def compare(rank, comparison):
            K = self.make_kernel_data(n, rank=rank)
            G = GramianFromArray(K, rank=rank)
            
            meas = MeasureMaximumMeanDeviation(gramian=G, comparison=comparison)
            res = self.eval_all(meas, n, 13, rs)
            obj = ObjectiveSpectral(G, meas.comparison==ComparisonMode.CONTRASTIVE)
            f_all_sol = obj(res.idx_sels_all)
            f_all_sol[0] = 0
            m_sel = 13
            np.testing.assert_almost_equal(f_all_sol, res.meas_vals, m_sel, 'Fails for measure comparison: all')
            df_opt = pd.DataFrame({'f':f_all_sol,'m':res.idx_sels_all.sum(0)})\
                .groupby('m').apply(lambda x:x.iloc[[np.where(x.f==x.f.max())[0][0]],:])\
                .droplevel(0).reset_index().rename({'index':'idx_opt'}, axis=1)
            
            
            def get_full_index(x):
                idl_x_sel = to_binary(x, m_sel)
                idl_x = np.zeros((n,m_sel), bool)
                idl_x[res.idl_sel,:] = idl_x_sel[:,1:]
                return idl_x
            idx_full = get_full_index(df_opt.idx_opt)
            df_comp = obj.components(idx_full)
            pd.set_option('max_colwidth',0,'max_columns',7,'display.width',120)
            
            oest = OptimisticEstimatorMaximumMeanDeviationSingleDirection(gramian=G, comparison=comparison)
            oest._debug = True
            df_oest = oest.evaluate_uncached(StaticSelector(res.idl_sel))
            cost_sq_oest = np.c_[np.vstack(df_oest.cost_sq),df_oest.cos_sq_rem]
            cost_sq_sol = np.vstack(df_comp.cost_sq)
            np.testing.assert_almost_equal(cost_sq_oest.sum(1), np.ones(m_sel),15,'Sum of cost_sq to 1')
            np.testing.assert_array_less(
                np.cumsum(cost_sq_sol,axis=1)-1e-15,
                np.cumsum(cost_sq_oest[:,:rank],axis=1),
                f'Cumulative cosine square fails for rank {rank}'
            )
            oest._debug = False
            oest_val = oest.evaluate_uncached(StaticSelector(res.idl_sel))
            full_val = df_comp.f.max()
            np.testing.assert_almost_equal(oest_val, df_oest.f.max(), 15, 'Non-debug fails')
            np.testing.assert_array_less(full_val, oest_val, 'Oest value fails')
        
        for rank,comparison in [(2,'contrastive'),(3,'contrastive'),(5,'contrastive'),
                                (2,'anomaly'),(3,'anomaly'),(5,'anomaly'),]:
            compare(rank, comparison)
        
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    
