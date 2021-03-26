'''
Created on Jan 14, 2018

@author: janis
'''

import unittest

from colito.logging import getRootLogger

log = getRootLogger()

class TestSearchMMD(unittest.TestCase):
    
    def test_mmd(self):
        import sergio.kernels.euclidean
        import sergio.scores.kernels
        
        from sergio.computation import Computation
        from sergio import FileManager
        
        log.add_stderr()
        log.setLevel(0)
        log.rlim.delay['progress'] = 5
        c = Computation(file_manager=FileManager('../datasets'), cache=None)
        
        #c.dataset = 'moondot:CATEGORICAL'
        c.dataset = 'toy-array:circledots'
        c.load_kernel('rbf',sigma=1.,kind='gaussian')
        c.compute_gramian()
        c.load_measure('mmd',comparison='contrastive')
        c.load_optimistic_estimator('mmd-sd', comparison='contrastive')
        c.load_prediciser()
        c.load_language('conjunctions')
        
        c.optimise(track_results=True)
        c.optimise(track_results=True)
        c.result_history
        c.subgroups        
        
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    
