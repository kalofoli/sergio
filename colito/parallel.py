'''
Created on Sep 6, 2021

@author: janis
'''

try:
    import joblib
    from tqdm import tqdm as _tqdm
    from functools import wraps
    
    def _switch_tqdm_backend(tqdm_new):
        global _tqdm
        _tqdm = tqdm_new
    
    @wraps(_tqdm)
    def tqdm(*args, **kwargs):
        return _tqdm(*args, **kwargs)
    
    class ProgressParallel(joblib.Parallel):
        def __init__(self, *args, tqdm_opts={}, total=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._tqdm_opts = {'total':total, **tqdm_opts}
            
        def __call__(self, *args, total=None, tqdm_opts={}, **kwargs):
            _tqdm_opts = {**self._tqdm_opts, **tqdm_opts}
            if total is not None:
                _tqdm_opts['total'] = total
            with tqdm(**_tqdm_opts) as self._pbar:
                return joblib.Parallel.__call__(self, *args, **kwargs)
    
        def print_progress(self):
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()

except ImportError: pass