'''
Created on Jan 14, 2018

@author: janis
'''

from functools import total_ordering
from typing import List, Type, Tuple, Sized, Optional, Sequence, Any, Union, Iterable, Iterator, TypeVar, Generic, Set, cast
from heapq import heappop, heappushpop, heappush, heapreplace, heapify
from math import inf


DataType = TypeVar('DataType')

@total_ordering
class Entry(Iterable, Generic[DataType]):
    '''An entry to be used in priority queues'''

    def __init__(self, data: DataType, priority: float) -> None:
        super(Entry, self).__init__()
        self._data = data
        if priority is None:
            raise TypeError('Priority must be float')
        self._priority = priority
        
    @property
    def priority(self) -> float:
        '''The priority to be used in a priority queue'''
        return self._priority

    @property
    def data(self) -> DataType:
        '''An arbitrary data payload'''
        return self._data
    
    @classmethod
    def worst(cls, data=None) -> 'Entry':
        '''Create the worse possible priority entry'''
        return cls(data=data, priority=cls.PRIORITY_WORST)
    
    @classmethod
    def best(cls, data=None) -> 'Entry':
        '''Create the best possible priority entry'''
        return cls(data=data, priority=cls.PRIORITY_BEST)

    def copy(self) -> 'Entry[DataType]':
        '''Create a (shallow) copy otf this entry'''
        return self.__class__(self.data, self.priority)
    
    def equals(self, entry: 'Entry[DataType]') -> bool:
        '''Test for full equality, including data'''
        return (self._priority == entry.priority and
                self._data == entry.data)
    
    def __lt__(self, entry: 'Entry[DataType]') -> bool:
        raise NotImplementedError()
    
    def __eq__(self, entry) -> bool:
        return self._priority == entry.priority and self._data == entry.data

    def __iter__(self):
        yield self.data
        yield self.priority
    
    def __repr__(self):
        return '<{0.__class__.__name__}:{0.data!r}@{0.priority}>'.format(self)

    def __str__(self):
        return '{0.data!s}@{0.priority}'.format(self)


class MinEntry(Entry[DataType], Generic[DataType]):
    '''Entry sorting in decreasing value order'''

    def __lt__(self, entry) -> bool:
        return self.priority < entry.priority

    PRIORITY_WORST = inf
    PRIORITY_BEST = -inf
    priority_best = min
    priority_worst = max

    
class MaxEntry(Entry[DataType], Generic[DataType]):
    '''Entry sorting in increasing value order'''

    def __lt__(self, entry) -> bool:
        return self.priority > entry.priority

    PRIORITY_WORST = -inf
    PRIORITY_BEST = inf
    priority_best = max
    priority_worst = min


class PriorityQueue(Sized, Iterable, Generic[DataType]):
    '''Implements a basic heap priority queue'''

    def __init__(self, entry_type: Type=MinEntry) -> None:
        super(PriorityQueue, self).__init__()
        self._heap: List[Entry[DataType]] = []
        self._entry_type: Type = entry_type

    def make_entry(self, data: DataType, priority: float, entry: Optional[Entry]=None) -> Entry[DataType]:
        '''Make an entry of the specified type'''
        entry_out: Entry[DataType] 
        if entry is not None:
            if entry.__class__ == self._entry_type:
                entry_out = entry
            else:
                entry_out = self._entry_type(data=entry.data, priority=entry.priority)
        else:
            entry_out = self._entry_type(data=data, priority=priority)
        return entry_out

    def is_valid_entry(self, entry: Entry[DataType]) -> bool:
        '''Decide whether an entry is aligned with the one used internally'''
        return isinstance(entry, self._entry_type)
        
    def make_entries(self, data: Optional[Union[Sequence[DataType], Iterator[DataType]]]=None,
                     priorities: Optional[Union[Sequence[float], Iterator[float]]]=None,
                     entries: Optional[Union[Sequence[Entry[DataType]], Iterator[Entry[DataType]]]]=None) -> Iterator[Entry[DataType]]:
        entries_out: Iterator[Entry[DataType]]
        if entries is None:

            def make_entry(pair: Tuple[DataType, float]) -> Entry[DataType]:
                data, value = pair
                return self._entry_type(data=data, priority=value)

            entries_out = map(make_entry, zip(data, priorities))
        else:
            entries_out = entries_out
        return entries_out
    
    @property
    def empty(self) -> bool:
        '''Return if the queue is empty'''
        return not self
    
    @property
    def front(self) -> Optional[Entry[DataType]]:
        '''The next element in the queue. (copied)'''
        return self.peek().copy() if self else None

    @property
    def entry_type(self) -> Type[Entry[DataType]]:
        '''The type of entries this queue uses'''
        return self._entry_type
    
    def __len__(self):
        return len(self._heap)
    
    def __nonzero__(self) -> bool:
        return bool(self._heap)
    
    def copy(self) -> 'PriorityQueue[DataType]':
        '''Create a (shallow) copy of this queue'''
        queue_copy = self.__class__(entry_type=self.entry_type)
        queue_copy._heap = self._heap.copy() # pylint: disable=protected-access
        return queue_copy
    
    def clear(self):
        '''Empty the queue'''
        self._heap.clear()
    
    def push(self, data:DataType=None, priority: Optional[float]=None, entry: Optional[Entry[DataType]]=None) -> None:
        '''Push a single element into the queue'''
        entry = self.make_entry(data=data, priority=priority, entry=entry)
        heappush(self._heap, entry)
        
    def push_entries(self, entries: Union[Sequence[Entry], Iterable[Entry]]=None) -> None:
        '''Add multiple entries to the queue.'''
        entries_new: List[Entry] = list(entries)
        if not all(map(self.is_valid_entry, entries)):
            raise ValueError('Entry classes must be of class {0._entry_type._name__}.')
        self._heap += entries_new
        heapify(self._heap)
        
    def __repr__(self):
        return '<{0.__class__.__name__}{0._heap!s}>'.format(self)
    
    def __str__(self):
        contents = ','.join(map(str, self))
        return '{{{0}}}'.format(contents)
        
    def pop(self):
        '''Pop the min (resp. max) item'''
        entry_out = heappop(self._heap)
        return entry_out
    
    def peek(self):
        '''Return the next element, without popping it.'''
        return self._heap[0].copy()
    
    def pushpop(self, data=None, priority: Optional[float]=None, entry: Optional[Entry]=None):
        '''Push and then Pop. Faster than individual calls'''
        entry = self.make_entry(data=data, priority=priority, entry=entry)
        entry_out = heappushpop(self._heap, entry)
        return entry_out
    
    def poppush(self, data=None, priority: Optional[float]=None, entry: Optional[Entry]=None):
        '''Pop and then Push. Faster than individual calls'''
        entry = self.make_entry(data=data, priority=priority, entry=entry)
        entry_out = heapreplace(self._heap, entry)
        return entry_out
    
    def entries(self, sort=True) -> Tuple[Entry, ...]:
        '''Return all current entries in the queue
        
        @param sort bool: If True, the entries are returned sorted'''
        if sort:
            return tuple(iter(self))
        else:
            return tuple(self._heap)
    
    def heap(self) -> Tuple[Entry, ...]:
        '''Return raw entries in the heap, unsorted'''
        return tuple(self._heap)
    
    def __iter__(self):
        entries = self._heap.copy()
        while entries:
            yield heappop(entries)


class TopKQueue(Sized, Iterable, Generic[DataType]):
    '''Preserve the top k entries, based on their value'''

    def __init__(self, k: int, max_best=True) -> None:
        super(TopKQueue, self).__init__()
        self._k: int = k
        self._entry_type: Type = MinEntry if max_best else MaxEntry
        self._queue: PriorityQueue[DataType] = PriorityQueue(entry_type=self._entry_type)
        self._data: Set[DataType] = set()
    
    def __len__(self) -> int:
        return len(self._queue)
    
    def __nonzero__(self) -> bool:
        return bool(self._queue)
    
    def copy(self):
        '''Make a shallow copy of the current item. The same data is used.'''
        q_new = self.__class__(k=self.k, max_best=self.max_best)
        q_new._queue = self._queue.copy()
        q_new._data = self._data.copy()
        return q_new
    
    @property
    def k(self) -> int:
        '''Maximum size allowed in the queue. Beyond that, entries are dropped.'''
        return self._k
    
    @property
    def max_best(self):
        '''Whether the biggest values are to be kept'''
        return self._entry_type == MinEntry
    
    @k.setter
    def k(self, value: int):
        if value <= 0:
            raise ValueError('Parameter k must be positive')
        else:
            if len(self) > value:
                # TODO: prune queue and add here
                raise NotImplementedError()
        self._k = value
                
    @property
    def threshold(self) -> float:
        '''The priority value of the worst element'''
        if len(self) >= self.k:
            priority = self._queue.peek().priority
        else:
            priority = self._queue.entry_type.best().priority
        return priority

    @property
    def threshold_element(self) -> Entry[DataType]:
        return self._queue.peek()
        
    @property
    def entry_type(self) -> Type[Entry[DataType]]:
        return self._entry_type
    
    def add(self, data:DataType=None, value: Optional[float]=None, entry: Optional[Entry[DataType]]=None) -> Tuple[Optional[Entry[DataType]], bool]:
        '''Try to add an element, and then keep the best k.
        
        @return: (dropped, was_added)
            was_added is true if the new element is kept.
            dropped is the element which was possibly dropped, if needed.
        '''
        entry_in: Entry[DataType] = self.make_entry(data=data, value=value, entry=entry)
        entry_out: Entry[DataType] = None
        was_added: bool = False
        if entry_in.data in self._data:
            pass
        elif len(self) >= self.k:
            entry_out = self._queue.pushpop(entry=entry_in)
            was_added = entry_in is not entry_out
            if was_added:
                self._data.remove(entry_out.data)
                self._data.add(entry_in.data)
        else:
            self._queue.push(entry=entry_in)
            self._data.add(entry_in.data)
            was_added = True
            if was_added:
                self._data.add(entry_in.data)
        return entry_out, was_added
    
    def entries(self, sort=True) -> Tuple[Entry[DataType], ...]:
        '''Return all current entries.
        
        @param sort bool: If true, entries are returned sorted'''
        return self._queue.entries(sort=sort)[::-1]

    def elements(self, sort=True) -> Tuple[Entry[DataType], ...]:
        '''Return all current data elements.
        
        @param sort bool: If true, entries are returned sorted'''
        return tuple(entry.data for entry in self._queue.entries(sort=sort))
    
    def make_entry(self, data:DataType, value: float, entry: Optional[Entry[DataType]]=None) -> Entry:
        '''Make an entry of the appropriate kind with the provided values'''
        entry_out: Entry = self._queue.make_entry(data=data, priority=value, entry=entry)
        return entry_out
    
    def __in__(self, what):
        isin: bool = False
        if isinstance(what, Entry):
            entry: Entry = cast(Entry, what)
            isin = entry.data in self._data
        else:
            data: DataType = cast(DataType, what)
            isin = data in self._data
        return isin
        
    def __iter__(self):
        return iter(self._queue)
    
    def __repr__(self) -> str:
        direction = 'min' if issubclass(self.entry_type, MinEntry) else 'max'
        return '<Top{0.k}Queue of {1} ({2}={0.threshold})>'.format(self, len(self), direction)
