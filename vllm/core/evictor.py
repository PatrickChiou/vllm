import enum
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    LFU = enum.auto()
    FIFO = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    def on_insert(self, block_id: int) -> None:
        """A new immutable block starts a cache residency."""
        return

    def on_access(self, block_id: int) -> None:
        """An immutable block is reused, possibly while still active."""
        return

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class OrderedEvictor(Evictor):
    """FIFO or LFU over eligible blocks, retaining history while referenced.

    A block leaving the free table is being pinned, not evicted. Its insertion
    order and frequency survive this transition. LFU counts allocator reuse
    operations and breaks ties by last access, then insertion order.
    """

    def __init__(self, policy: EvictionPolicy):
        self.policy = policy
        self.free_table: Dict[int, BlockMetaData] = {}
        self._history: Dict[int, Tuple[int, int]] = {}
        self._serial = 0
        self._versions: Dict[int, int] = {}
        self.priority_queue: List[Tuple[float, float, int, int, int]] = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def on_insert(self, block_id: int) -> None:
        self._serial += 1
        self._history[block_id] = (self._serial, 1)

    def on_access(self, block_id: int) -> None:
        if block_id not in self._history:
            self.on_insert(block_id)
        order, frequency = self._history[block_id]
        self._history[block_id] = (order, frequency + 1)
        if block_id in self.free_table:
            self._push(block_id)

    def _entry(self, block_id: int) -> Tuple[float, float, int, int, int]:
        order, frequency = self._history[block_id]
        block = self.free_table[block_id]
        if self.policy == EvictionPolicy.FIFO:
            priority, recency = order, 0.0
        else:
            priority, recency = frequency, block.last_accessed
        return (priority, recency, order, self._versions[block_id], block_id)

    def _push(self, block_id: int) -> None:
        self._serial += 1
        self._versions[block_id] = self._serial
        heapq.heappush(self.priority_queue, self._entry(block_id))
        if len(self.priority_queue) > 50 * max(len(self.free_table), 1):
            self.priority_queue = [self._entry(b) for b in self.free_table]
            heapq.heapify(self.priority_queue)

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        if block_id not in self._history:
            self.on_insert(block_id)
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                 num_hashed_tokens,
                                                 last_accessed)
        self._push(block_id)

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed
        self._push(block_id)

    def remove(self, block_id: int):
        self.free_table.pop(block_id)
        self._versions.pop(block_id, None)

    def evict(self) -> Tuple[int, int]:
        while self.priority_queue:
            _, _, _, version, block_id = heapq.heappop(self.priority_queue)
            if self._versions.get(block_id) != version:
                continue
            block = self.free_table[block_id]
            self.remove(block_id)
            self._history.pop(block_id)
            return block_id, block.content_hash
        raise ValueError("No usable cache memory left")

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy in (EvictionPolicy.LFU, EvictionPolicy.FIFO):
        return OrderedEvictor(eviction_policy)
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
