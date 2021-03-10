'''
Created on Feb 1, 2021

@author: janis
'''
class NoMessage():
    def __repr__(self):
        return 'SuppressLog'
    def __bool__(self):
        return False
    
SUPPRESS = NoMessage()
class LoggingRateTracker:
    class LevelEntry:
        def __init__(self):
            self._delay = 0
            self._last = 0
        
    def __init__(self):
        self._delays = defaultdict(self.LevelEntry)

    def install_handlers(self, logger):
        now_in_seconds = LoggerRateLimiter.now_in_seconds
        last = self._last
        delays = self._delays
        passes_tracker = self._passes

        # Speedup: avoid lookups
        def passes_both(self, level, now):
            return self.passes(level) and passes_tracker(last, delays, now, level.value) or SUPPRESS
        
        def add_handlers(level):

            def is_on(self):
                now = now_in_seconds()
                return passes_both(self, level, now)

            def mark(self):
                now = now_in_seconds()
                if passes_both(self, level, now):
                    last[level.value] = now
                    return True
                return SUPPRESS


            setattr(logger, f'is_on_{level.name.lower()}', property(is_on))
            setattr(logger, level.name.lower(), property(mark))
        
        for lvl in LogLevels:
            add_handlers(lvl) 
    
    def set_delay(self, delay, level=None):
        if level is None:
            for lvl in LogLevels:
                self.set_delay(delay, lvl)
        else:
            lvl_val = FinerLogger.get_level_value(level)
            try:
                rate = float(delay)
                assert rate >= 0, 'Rates must be positive floats'
                self._delays[lvl_val] = rate
            except:
                raise ValueError(f'Could not set rate to {rate}.')
    
    @classmethod
    def _passes(cls, last, delays, now, lvl_val):
        return now - last[lvl_val] > delays[lvl_val]
    
    def is_on(self, lvl):
        now = self.now_in_seconds()
        lvl_val = get_log_level_value(lvl)
        return self._passes(self._last, self._delays, now, lvl_val) or SUPPRESS
    
    def __repr__(self):
        now = self.now_in_seconds()
        level_rem = lambda lvl: (now - self._last[lvl.value]) / self._delays[lvl.value] if self._delays[lvl.value] else math.inf
        txt = ', '.join(f'{lvl.name}:{"Y" if self.is_on(lvl.name) else "N"}/{self._delays[lvl.value]}s({level_rem(lvl)*100:.1f}%)' for lvl in LogLevels if self._delays[lvl.value]!=0)
        return f'<{self.__class__.__name__}:{txt}>'
    
    @classmethod
    def now_in_seconds(cls) -> float:
        '''Return the current timestamp in seconds'''
        return time.time()


GLOBAL_RATE_TRACKER = LoggingRateTracker()


class LoggerRateLimiter():

    def __init__(self, logger) -> None:
        self._logger = logger
        is_enabled_for = logger.isEnabledFor
        
        self.passes = lambda lvl_name: is_enabled_for(lvl_name.value)

    @classmethod
    def set_delay(cls, delay, level=None):
        return GLOBAL_RATE_TRACKER.set_delay(delay, level)
    
    def is_on(self, lvl):
        lvl_enum = FinerLogger.get_level(lvl)
        return getattr(self, f'is_on_{lvl_enum.name.lower()}')
    
    now_in_seconds = LoggingRateTracker.now_in_seconds
    
    def __repr__(self):
        txt = ', '.join(f'{lvl.name}:{"Y" if self.is_on(lvl.name) else "N"}' for lvl in LogLevels)
        return f'<{self.__class__.__name__}:{txt}>'
    

GLOBAL_RATE_TRACKER.install_handlers(LoggerRateLimiter)



def add_level_helper_handlers():
    for lvl in LogLevels:
        def make_property(level):
            def is_on(self):
                return self._logger.isEnabledFor(level) # pylint: disable = protected-access
            return property(is_on)
        setattr(LevelHelper,lvl.name.lower(), make_property(lvl.value))

add_level_helper_handlers()
