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

    class PersistentProgressParallel(joblib.Parallel):
        def __init__(self, *args, tqdm_opts={}, total=None, progress=True, **kwargs):
            super().__init__(*args, **kwargs)
            self._progress = progress
            self._tqdm_opts = {'total':total, **tqdm_opts}
            self._tqdm = None
            self.__entered = False
            self.__n_completed_tasks = 0
        @property
        def progressbar(self): return self._tqdm
        @property
        def progress(self): return self._progress
        @progress.setter
        def progress(self, value):
            if not value and self._progress:
                self._progress_stop()
            if value and self.__entered:
                self._progress_start()
            self._progress = value
        def __call__(self, *args, **kwargs):
            if hasattr(self, 'n_completed_tasks'):
                self.__n_completed_tasks += self.n_completed_tasks
            return super().__call__(*args, **kwargs)
        def _progress_stop(self):
            if self._tqdm is not None:
                self._tqdm.__exit__(None,None,None)
                self._tqdm = None
        def _progress_start(self):
            if self._progress:
                self._tqdm = tqdm(**self._tqdm_opts).__enter__()
                self._tqdm.n = self.__n_completed_tasks
                    
        def print_progress(self):
            if self._tqdm is not None:
                self._tqdm.n = self.__n_completed_tasks + self.n_completed_tasks
                self._tqdm.refresh()
        def __enter__(self):
            self.__entered = False
            self._progress_start()
            return super().__enter__()
        def __exit__(self, exc_type, exc_val, exc_tb):
            self._progress_stop()
            self.__entered = False
            return super().__exit__(exc_type, exc_val, exc_tb)

except ImportError: pass