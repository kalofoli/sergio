'''
Created on Jan 14, 2018

@author: janis
'''
import numpy as np
import pandas as pd

import unittest
from sergio.scores.scalars import OptimisticEstimatorCoverageMeanShiftGroupped
from sergio.language import StaticSelector
from sergio.computation import Computation
from sergio import FileManager

debug=True
def get_groups(B_full, idl_sel):
    n = B_full.shape[0]
    idl_nonred = np.where(B_full[idl_sel,:].sum(0) < idl_sel.sum())[0]
    map_sel2full = np.arange(n)[idl_sel]
    C = np.packbits(B_full[np.ix_(idl_sel,idl_nonred)],axis=1)
    
    pb = lambda x:np.packbits(x[:,idl_nonred],axis=1)
    
    C_unq, map_sel2grp, grp_cnt = np.unique(C,axis=0, return_inverse=True, return_counts=True)
    map_seg2sel = np.argsort(map_sel2grp)
    np.testing.assert_array_equal(C_unq[map_sel2grp,:], C, 'Mapping unique to selection')
    # Since sorted, the segment ids will be consequtive
    map_seg2full = map_sel2full[map_seg2sel]
    seg_groups = np.split(map_seg2full, np.cumsum(grp_cnt)[:-1])
    if debug:
        pb = lambda x:np.packbits(x[:,idl_nonred],axis=1)
        for idx_grp, seg_group in enumerate(seg_groups):
            np.testing.assert_array_equal(pb(B_full)[seg_group,:], np.repeat(C_unq[[idx_grp],:],grp_cnt[idx_grp],0))
    return seg_groups

def mk_random(seed=None):
    if seed is None:
        rs = np.random
    else:
        if isinstance(seed, int):
            rs = np.random.RandomState(seed)
        else:
            rs = seed
    return rs

def oest_groups(groups, target, gamma=1., full=False):
    R = np.r_[[(target[g].sum(), len(g)) for g in groups]]
    p = np.argsort(R[:,0]/R[:,1])[::-1]
    R_csum = np.cumsum(R[p,:], axis=0)
    y_run, m_run = R_csum[:,0], R_csum[:,1]
    n = len(target)
    c = target.sum()/n
    f = ((m_run/n)**gamma*(y_run/m_run-c))**1/gamma
    if full:
        return f
    else:
        return float(f.max())

class TestSuperTight(unittest.TestCase):
    def make_data(self, shape, seed=0):
        rs = mk_random(seed)
        B = rs.randint(0,2,shape)>0
        B = np.c_[B,B[:,-1]]
        idl = B[:,-1]
        B = B[:,:-1]
        return B, idl
    def make_fixed(self):
        B = np.vstack([[1,0,0,0,0,0,0,1,0,0,0],
                       [1,0,1,1,0,0,1,0,0,0,1],
                       [1,0,1,1,0,0,1,1,1,0,0],
                       [0,1,0,1,0,1,1,1,0,1,1],
                       [0,1,0,1,0,1,1,1,0,1,1],
                       [0,1,0,1,0,1,1,1,0,1,1],
                       [1,1,1,1,0,0,1,1,1,0,0],
                       [1,0,1,0,1,1,1,1,0,1,1],
                       [1,0,1,1,0,0,1,1,0,1,1],
                       [1,0,1,1,0,0,1,1,0,1,1],
                       [1,0,1,1,0,0,1,1,0,1,1],
                       [1,0,1,1,0,0,1,1,0,1,1],
                       [1,0,1,1,0,0,0,1,1,0,0],
                       [1,0,0,0,1,1,1,0,1,0,0]])
        idl = B[:,-1]>0
        B = B[:,:-1]
        return B, idl
    
    def test_base(self, seed=22):
        rs = mk_random(seed)
        B,idl = self.make_fixed()
        groups = get_groups(B, idl)
        n = B.shape[0]
        y = rs.random(n)
        f = oest_groups(groups, y)
        print(groups)
    
    def test_scores(self, seed=22):
        rs = mk_random(seed)
        B,idl = self.make_fixed()
        groups = get_groups(B, idl)
        n = B.shape[0]
        y = rs.random(n)
        oest = OptimisticEstimatorCoverageMeanShiftGroupped(
            validities=B, target_data=y, target_name='target'
        )
        f_sol = oest_groups(groups, y, full=True)
        f_cmp = oest._evaluate_raw(idl, full=True)
        np.testing.assert_equal(f_cmp, f_sol, 'Oest fails')
        print(groups)

    def test_data(self):
        #df = pd.read_csv('../datasets/tabular-titanic.csv')
        c = Computation(file_manager=FileManager('../datasets/'), cache=None)
        
        c.dataset = 'titanic:fare'
        c.load_prediciser()
        c.language = 'closure-conjunctions-restricted'
        c.load_measure('coverage-mean-shift')
        c.load_optimistic_estimator('coverage-mean-shift-groupped')
        res_co = c.optimise(depths=1)
        c.load_optimistic_estimator('coverage-mean-shift')
        res_uc = c.optimise(depths=1)
        print(res)
        
        #EntityAttributesWithAttributeTarget(attribute_data=df)
        
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    
