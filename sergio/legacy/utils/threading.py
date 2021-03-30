'''
Created on Sep 13, 2018

@author: janis
'''

import threading
import time
from sdcore.summarisable import Summarisable, SummaryOptions

class Pauser(Summarisable):
    timeout = None
    def __init__(self, paused = False):
        self._time_created:float = 0
        self._duration_paused:float = 0
        self._duration_waited:float = 0
        self._times_paused:int = 0
        self._times_waited:int = 0
        
        self._is_paused = False
        self._is_waited = False
        
        self._pause_start = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        if paused:
            self.pause()
        self._time_created = self._now()
    
    def _now(self):
        return time.time()
    
    @property
    def is_paused(self):
        return self._is_paused
    
    @is_paused.setter
    def is_paused(self, val):
        if val:
            self.pause()
        else:
            self.unpause()
    
    is_waited = property(lambda self:self._is_waited, None, 'Tell whether this Pauser is currently paused')
    time_created = property(lambda self:self._time_created, None, 'The time at which this pauser was created.')
    times_paused = property(lambda self:self._times_paused, None, 'The number of times this pauser was requested to pause.')
    times_waited = property(lambda self:self._times_waited, None, 'The number of times a thread had to wait for this pauser.')
    duration_alive = property(lambda self:self._now() - self._time_created, None, 'The duration this pauser has been alive.')
    duration_paused = property(lambda self:self._duration_paused, None, 'The duration this pauser spent in the paused state.')
    duration_waited = property(lambda self:self._duration_waited, None, 'The duration some thread has spent waiting for this pauser.')
    
    def pause(self):
        with self._cond:
            if self.is_paused:
                return False
            else:
                self._times_paused += 1
                self._pause_start = self._now()
                self._is_paused = True
    
    def wait(self, timeout=None):
        with self._cond:
            if self.is_paused:
                if self._is_waited:
                    raise ValueError(f'Only one thread can wait on this Pauser.')
                self._is_waited = True
                self._times_waited += 1
                time_wait_start = self._now()
                
                deadline = time_wait_start + timeout if timeout is not None else None
                syscall_timeout = self.timeout
                while self.is_paused:
                    if deadline is not None:
                        time_now = self._now()
                        syscall_timeout = min(syscall_timeout, deadline - time_now)
                    self._cond.wait(syscall_timeout)
                
                time_wait_stop = self._now()
                self._duration_waited += time_wait_stop - time_wait_start
                self._is_waited = False
            return self.is_paused
        
    
    def unpause(self):
        with self._cond:
            self._is_paused = False
            self._cond.notify_all()
            paused_stop = self._now()
            self._duration_paused += paused_stop - self._pause_start
    
    def __repr__(self):
        tags = []
        if self.is_paused:
            tags.append('PAUSED')
        if self.is_waited:
            tags.append('WAITED')
        duration_alive = self.duration_alive
        return (f'<{self.__class__.__name__} [{"P" if self.is_paused else "p"}{"W" if self.is_waited else "w"}] '
                f'time: '
                f'P={self.duration_paused:5g}({self.duration_paused/duration_alive*100:.3g}%) '
                f'W={self.duration_waited:5g}({self.duration_waited/duration_alive*100:.3g}%) '
                f'count: P={self.times_paused} W={self.times_waited}>')
    
    def __summary_dict__(self, options:SummaryOptions):
        fields = ['times_paused','times_waited','duration_paused','duration_waited','duration_alive','time_created','is_paused','is_waited']
        return summary_from_fields(self,fields)

