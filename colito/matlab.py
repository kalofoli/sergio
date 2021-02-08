'''
Created on Aug 9, 2018

@author: janis
'''

import numpy as np
from collections import namedtuple
from typing import Callable

UniqueResult = namedtuple('UniqueResult',('unique_entries','unique_indices','entry_labels','sorter'))
def unique(values, sorter=np.argsort):
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    values = np.array(values)
    if isinstance(sorter, Callable):
        idx_srt = sorter(values)
    elif sorter is None:
        idx_srt = np.arange(len(values))
    elif isinstance(sorter, np.ndarray):
        idx_srt = sorter
    else:
        raise TypeError(f'Invalid sorter type {sorter}. Sorter must be either an argsort-like function, an index array or None to designate a sorted input.')
    values_srt = values[idx_srt]
    diff = np.diff(values_srt)
    is_new = np.concatenate(([True],diff!=0))
    entry_labels_srt = np.cumsum(is_new)-1
    entry_labels = np.zeros(len(entry_labels_srt),int)
    entry_labels[idx_srt] = entry_labels_srt
    return UniqueResult(
        unique_entries=values_srt[is_new],
        unique_indices=idx_srt[is_new],
        entry_labels=entry_labels,
        sorter=idx_srt,
    )
