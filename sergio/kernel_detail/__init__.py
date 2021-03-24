'''
Measures

Created on Oct 3, 2019

@author: janis
'''
# pylint: disable=bad-whitespace

import grakel

from cofi.summarisable import SummarisableAsDict, SummaryOptions
from cofi.factory import resolve_arguments
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Sequence


class Kernel(grakel.Kernel, SummarisableAsDict):
    '''SummarisableAsDict Kernel class for scikit'''
    _summary_convert = {np.integer:int, float:float}
    def summary_dict(self, options:SummaryOptions):
        return self.get_params()
    
    @property
    def summary_name(self):
        return f'kern-{self.__class__.tag}'
    
    @classmethod
    def parse_string_argument(cls, name, value, parameter):
        '''Returns the parsed value if successful, or NoConversionError to use the default parser.
        
        Override this method if special parsing is required'''
        raise NoConversionError()

    @classmethod
    def make_from_strings(cls, name, args, kwargs): pass # filled later, once all kernels have been loaded

class ShiftInvariantKernel(SummarisableAsDict,metaclass=ABCMeta):
    '''Simple base class for Shift Invariant kernels'''
    _summary_compact_fields = []
    bandwidth:int = None
    def summary_dict(self, options):
        '''The parameters to be included in the summary as a dict'''
        dct = self.summary_from_fields(self._summary_compact_fields)
        if not options.is_compact:
            dct['description'] = str(self)
        return dct
    
    @abstractmethod
    def _from_array(self, diff:float) -> float:
        pass

    def __call__(self, value:float) -> float:
        if isinstance(value, int):
            arr = np.array([value])
            r = self._from_array(arr)[0]
        else:
            if isinstance(value, (np.ndarray, Sequence)):
                arr = np.array(value)
            else:
                raise TypeError(f'Cannot compute kernel value for {value} which is of non-array-convertible type {type(value)}')
            r = self._from_array(arr)
        return r

    @classmethod
    def make_from_strings(cls, name, args, kwargs): pass # filled later, once all kernels have been loaded


if __name__ == '__main__':
    from cofi.experiment import Experiment
    e = Experiment()
    e.load_dataset('MUTAG')
    
    k = grakel.RandomWalk()
    k.fit(e.dataset.data)
    
    pass
    
    