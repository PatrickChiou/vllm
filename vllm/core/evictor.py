import enum
import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List, Tuple, Set, Optional


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    LFU = enum.auto()
    FIFO = enum.auto()
    TLRU = enum.auto()
    TFIFO = enum.auto()
    HFIFO = enum.auto()
    HLRU = enum.auto()
    RFIFO = enum.auto()


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
             last_accessed: float, workloads: Optional[Set[str]] = None):
            self.content_hash = content_hash
            self.num_hashed_tokens = num_hashed_tokens
            self.last_accessed = last_accessed
            self.workloads = set(workloads or ())


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
        self._workloads: Dict[int, Set[str]] = {}

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
                self._workloads.pop(content_hash, None)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  self._workloads.get(content_hash))
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

    def note(self, content_hash: int, workload: str) -> None:
        self._workloads.setdefault(content_hash, set()).add(workload)


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
        self._workloads: Dict[int, Set[str]] = {}

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
                                                 last_accessed,
                                                 self._workloads.get(content_hash))
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
            self._workloads.pop(block.content_hash, None)
            return block_id, block.content_hash
        raise ValueError("No usable cache memory left")

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def note(self, content_hash: int, workload: str) -> None:
        self._workloads.setdefault(content_hash, set()).add(workload)

class TLRUEvictor(Evictor):
    """Evict one-shot blocks first, then unprotected workloads, then shared prefixes.

    Shared blocks have been used by at least two workload types. A block whose
    only workload is a multi-turn type stays protected until hold_s after
    last_accessed, and only up to hold_budget of the free table. The four
    protected types share that one budget. Inside one tier, the oldest
    last_accessed is evicted; a tie evicts the larger num_hashed_tokens,
    then the smaller block_id.

    Each group lives in a min-heap ordered by last_accessed. Hold expiry and
    the budget are applied when choosing a victim, so a refreshed block stays
    in the protected heap instead of being reclassified early.
    """

    # The five types share that one budget. Inside one tier, the oldest
    # last_accessed is evicted.
    # PROTECTED = {"chat", "agent", "coding-agent", "bailian-chat",
    #              "bailian-coder"}
    PROTECTED = {"agent", "coding-agent", "bailian-chat",
                 "bailian-coder"}
    CLEANUP_THRESHOLD = 50
    _FIXED = "fixed"
    _PROTECTED = "protected"
    _SHARED = "shared"

    def __init__(self, hold_s: float, hold_budget: float):
        self.free_table: Dict[int, BlockMetaData] = {}
        self._workloads: Dict[int, Set[str]] = {}
        self.hold_s = hold_s
        self.hold_budget = hold_budget
        # self._heaps: Dict[str, List[Tuple[float, int, int]]] = {
        self._heaps: Dict[str, List[Tuple[float, int, int, int]]] = {
            self._FIXED: [], self._PROTECTED: [], self._SHARED: []}
        self._version: Dict[int, int] = {}
        self._kind: Dict[int, str] = {}
        self._counts: Dict[str, int] = {
            self._FIXED: 0, self._PROTECTED: 0, self._SHARED: 0}

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def note(self, content_hash: int, workload: str) -> None:
        self._workloads.setdefault(content_hash, set()).add(workload)

    def _classify(self, workloads: Set[str]) -> str:
        if len(workloads) >= 2:
            return self._SHARED
        if len(workloads) == 1 and workloads <= self.PROTECTED:
            return self._PROTECTED
        return self._FIXED

    def _push(self, block_id: int, kind: str) -> None:
        self._version[block_id] = self._version.get(block_id, 0) + 1
        self._kind[block_id] = kind
        self._counts[kind] += 1
        block = self.free_table[block_id]
        # heapq.heappush(self._heaps[kind],
        #                (block.last_accessed, block_id,
        #                 self._version[block_id]))
        heapq.heappush(self._heaps[kind],
                       (block.last_accessed, -block.num_hashed_tokens,
                        block_id, self._version[block_id]))
        self._cleanup_if_necessary(kind)

    def _cleanup_if_necessary(self, kind: str) -> None:
        live = max(self._counts[kind], 1)
        if len(self._heaps[kind]) > self.CLEANUP_THRESHOLD * live:
            self._rebuild(kind)

    def _rebuild(self, kind: str) -> None:
        # rebuilt = [(self.free_table[block_id].last_accessed, block_id,
        #             self._version[block_id])
        rebuilt = [(self.free_table[block_id].last_accessed,
                    -self.free_table[block_id].num_hashed_tokens, block_id,
                    self._version[block_id])
                   for block_id, block_kind in self._kind.items()
                   if block_kind == kind and block_id in self.free_table]
        heapq.heapify(rebuilt)
        self._heaps[kind] = rebuilt

    def _peek(self, kind: str) -> Optional[Tuple[float, int, int, int]]:
        """Drop stale entries and return the oldest live block in this heap."""
        heap = self._heaps[kind]
        while heap:
            # last_accessed, block_id, version = heap[0]
            last_accessed, num_hashed_tokens, block_id, version = heap[0]
            if (block_id in self.free_table
                    and self._version.get(block_id) == version
                    and self._kind.get(block_id) == kind):
                return last_accessed, num_hashed_tokens, block_id, version
            heapq.heappop(heap)
        return None

    def _take(self, kind: str) -> Tuple[int, int]:
        if self._peek(kind) is None:
            raise ValueError("No usable cache memory left")
        # _, block_id, _ = heapq.heappop(self._heaps[kind])
        _, _, block_id, _ = heapq.heappop(self._heaps[kind])
        block = self.free_table.pop(block_id)
        self._kind.pop(block_id, None)
        self._counts[kind] -= 1
        self._version[block_id] = self._version.get(block_id, 0) + 1
        self._workloads.pop(block.content_hash, None)
        return block_id, block.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        workloads = set(self._workloads.get(content_hash) or ())
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed, workloads)
        self._push(block_id, self._classify(workloads))

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed
        kind = self._kind[block_id]
        self._counts[kind] -= 1
        self._push(block_id, kind)

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)
        kind = self._kind.pop(block_id)
        self._counts[kind] -= 1
        self._version[block_id] = self._version.get(block_id, 0) + 1

    def evict(self) -> Tuple[int, int]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")
        now = time.time()
        limit = int(self.hold_budget * len(self.free_table))
        # tier0: List[Tuple[float, int, str]] = []
        tier0: List[Tuple[float, int, int, str]] = []
        fixed = self._peek(self._FIXED)
        if fixed is not None:
            # tier0.append((fixed[0], fixed[1], self._FIXED))
            tier0.append((fixed[0], fixed[1], fixed[2], self._FIXED))
        protected = self._peek(self._PROTECTED)
        if protected is not None:
            last_accessed = protected[0]
            expired = now - last_accessed > self.hold_s
            # Once the oldest protected block is fresh, every newer one is too.
            outside_budget = self._counts[self._PROTECTED] > limit
            if expired or outside_budget:
                # tier0.append((protected[0], protected[1], self._PROTECTED))
                tier0.append((protected[0], protected[1], protected[2],
                              self._PROTECTED))
        if tier0:
            # _, _, kind = min(tier0)
            _, _, _, kind = min(tier0)
            return self._take(kind)
        if protected is not None:
            return self._take(self._PROTECTED)
        if self._peek(self._SHARED) is not None:
            return self._take(self._SHARED)
        raise ValueError("No usable cache memory left")

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class TFIFOEvictor(Evictor):
    """Evict one-shot blocks first, then multi-turn blocks, then shared prefixes.

    Order inside a tier is the serial assigned when the block first became
    cached. Pinning and later access do not change it, and a multi-turn block
    is never demoted because of age or a hold budget. A smaller serial is
    evicted first.
    """

    PROTECTED = {"agent", "coding-agent", "bailian-chat", "bailian-coder"}
    CLEANUP_THRESHOLD = 50
    _FIXED = "fixed"
    _PROTECTED = "protected"
    _SHARED = "shared"

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self._workloads: Dict[int, Set[str]] = {}
        self._heaps: Dict[str, List[Tuple[int, int, int]]] = {
            self._FIXED: [], self._PROTECTED: [], self._SHARED: []}
        self._version: Dict[int, int] = {}
        self._order: Dict[int, int] = {}
        self._kind: Dict[int, str] = {}
        self._counts: Dict[str, int] = {
            self._FIXED: 0, self._PROTECTED: 0, self._SHARED: 0}
        self._serial = 0

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def note(self, content_hash: int, workload: str) -> None:
        self._workloads.setdefault(content_hash, set()).add(workload)

    def on_insert(self, block_id: int) -> None:
        self._serial += 1
        self._order[block_id] = self._serial

    def _classify(self, workloads: Set[str]) -> str:
        if len(workloads) >= 2:
            return self._SHARED
        if len(workloads) == 1 and workloads <= self.PROTECTED:
            return self._PROTECTED
        return self._FIXED

    def _remember_order(self, block_id: int) -> None:
        if block_id not in self._order:
            self.on_insert(block_id)

    def _push(self, block_id: int, kind: str) -> None:
        self._version[block_id] = self._version.get(block_id, 0) + 1
        self._kind[block_id] = kind
        self._counts[kind] += 1
        heapq.heappush(self._heaps[kind],
                       (self._order[block_id], block_id,
                        self._version[block_id]))
        self._cleanup_if_necessary(kind)

    def _cleanup_if_necessary(self, kind: str) -> None:
        live = max(self._counts[kind], 1)
        if len(self._heaps[kind]) > self.CLEANUP_THRESHOLD * live:
            self._rebuild(kind)

    def _rebuild(self, kind: str) -> None:
        rebuilt = [(self._order[block_id], block_id, self._version[block_id])
                   for block_id, block_kind in self._kind.items()
                   if (block_kind == kind and block_id in self.free_table
                       and block_id in self._order)]
        heapq.heapify(rebuilt)
        self._heaps[kind] = rebuilt

    def _peek(self, kind: str) -> Optional[Tuple[int, int, int]]:
        heap = self._heaps[kind]
        while heap:
            serial, block_id, version = heap[0]
            if (block_id in self.free_table
                    and self._version.get(block_id) == version
                    and self._kind.get(block_id) == kind
                    and self._order.get(block_id) == serial):
                return serial, block_id, version
            heapq.heappop(heap)
        return None

    def _take(self, kind: str) -> Tuple[int, int]:
        if self._peek(kind) is None:
            raise ValueError("No usable cache memory left")
        _, block_id, _ = heapq.heappop(self._heaps[kind])
        block = self.free_table.pop(block_id)
        self._kind.pop(block_id, None)
        self._counts[kind] -= 1
        self._version[block_id] = self._version.get(block_id, 0) + 1
        self._order.pop(block_id, None)
        self._workloads.pop(block.content_hash, None)
        return block_id, block.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        workloads = set(self._workloads.get(content_hash) or ())
        self._remember_order(block_id)
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed, workloads)
        self._push(block_id, self._classify(workloads))

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)
        kind = self._kind.pop(block_id)
        self._counts[kind] -= 1
        self._version[block_id] = self._version.get(block_id, 0) + 1

    def evict(self) -> Tuple[int, int]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")
        for kind in (self._FIXED, self._PROTECTED, self._SHARED):
            if self._peek(kind) is not None:
                return self._take(kind)
        raise ValueError("No usable cache memory left")

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class _HitGateEvictor(Evictor):
    """Two tiers from observed reuse. Workload labels are ignored.

    A block starts in the new tier. The first ``on_access`` moves that block,
    and only that block, to the reused tier. ``on_insert`` starts a new
    residency and clears the hit. Eviction removes the hit as well, so a
    recycled physical id starts in the new tier again.
    """

    CLEANUP_THRESHOLD = 50
    _NEW = "new"
    _REUSED = "reused"

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self._version: Dict[int, int] = {}
        self._kind: Dict[int, str] = {}
        self._reused: Set[int] = set()
        self._counts: Dict[str, int] = {self._NEW: 0, self._REUSED: 0}
        self._heaps: Dict[str, list] = {self._NEW: [], self._REUSED: []}

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def note(self, content_hash: int, workload: str) -> None:
        return

    def _tier(self, block_id: int) -> str:
        if block_id in self._reused:
            return self._REUSED
        return self._NEW

    def _invalidate(self, block_id: int) -> None:
        if block_id not in self.free_table:
            return
        kind = self._kind.pop(block_id)
        self._counts[kind] -= 1
        self._version[block_id] = self._version.get(block_id, 0) + 1

    def _cleanup_if_necessary(self, kind: str) -> None:
        live = max(self._counts[kind], 1)
        if len(self._heaps[kind]) > self.CLEANUP_THRESHOLD * live:
            self._rebuild(kind)

    def _rebuild(self, kind: str) -> None:
        raise NotImplementedError

    def _push(self, block_id: int, kind: str) -> None:
        raise NotImplementedError

    def _peek(self, kind: str):
        raise NotImplementedError

    def on_insert(self, block_id: int) -> None:
        self._reused.discard(block_id)
        self._invalidate(block_id)

    def on_access(self, block_id: int) -> None:
        if block_id in self._reused:
            return
        self._reused.add(block_id)
        self._invalidate(block_id)
        if block_id in self.free_table:
            self._push(block_id, self._REUSED)

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self._invalidate(block_id)
        self.free_table.pop(block_id)

    def _take(self, kind: str) -> Tuple[int, int]:
        if self._peek(kind) is None:
            raise ValueError("No usable cache memory left")
        block_id = self._pop_live(kind)
        block = self.free_table.pop(block_id)
        self._kind.pop(block_id, None)
        self._counts[kind] -= 1
        self._version[block_id] = self._version.get(block_id, 0) + 1
        self._reused.discard(block_id)
        self._forget(block_id)
        return block_id, block.content_hash

    def _pop_live(self, kind: str) -> int:
        raise NotImplementedError

    def _forget(self, block_id: int) -> None:
        return

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class HFIFOEvictor(_HitGateEvictor):
    """Hit-gated FIFO. Insertion order inside each tier; no hold."""

    def __init__(self):
        super().__init__()
        self._order: Dict[int, int] = {}
        self._serial = 0

    def on_insert(self, block_id: int) -> None:
        super().on_insert(block_id)
        self._serial += 1
        self._order[block_id] = self._serial
        if block_id in self.free_table:
            self._push(block_id, self._NEW)

    def _forget(self, block_id: int) -> None:
        self._order.pop(block_id, None)

    def _ensure_order(self, block_id: int) -> None:
        if block_id not in self._order:
            self._serial += 1
            self._order[block_id] = self._serial

    def _push(self, block_id: int, kind: str) -> None:
        self._version[block_id] = self._version.get(block_id, 0) + 1
        self._kind[block_id] = kind
        self._counts[kind] += 1
        heapq.heappush(self._heaps[kind],
                       (self._order[block_id], block_id,
                        self._version[block_id]))
        self._cleanup_if_necessary(kind)

    def _rebuild(self, kind: str) -> None:
        rebuilt = [(self._order[block_id], block_id, self._version[block_id])
                   for block_id, block_kind in self._kind.items()
                   if (block_kind == kind and block_id in self.free_table
                       and block_id in self._order)]
        heapq.heapify(rebuilt)
        self._heaps[kind] = rebuilt

    def _peek(self, kind: str) -> Optional[Tuple[int, int, int]]:
        heap = self._heaps[kind]
        while heap:
            serial, block_id, version = heap[0]
            if (block_id in self.free_table
                    and self._version.get(block_id) == version
                    and self._kind.get(block_id) == kind
                    and self._order.get(block_id) == serial):
                return serial, block_id, version
            heapq.heappop(heap)
        return None

    def _pop_live(self, kind: str) -> int:
        _, block_id, _ = heapq.heappop(self._heaps[kind])
        return block_id

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self._ensure_order(block_id)
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed, set())
        self._push(block_id, self._tier(block_id))

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def evict(self) -> Tuple[int, int]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")
        for kind in (self._NEW, self._REUSED):
            if self._peek(kind) is not None:
                return self._take(kind)
        raise ValueError("No usable cache memory left")


class HLRUEvictor(_HitGateEvictor):
    """Hit-gated recency. The reused tier uses TLRU's hold and budget."""

    def __init__(self, hold_s: float, hold_budget: float):
        super().__init__()
        self.hold_s = hold_s
        self.hold_budget = hold_budget

    def on_insert(self, block_id: int) -> None:
        super().on_insert(block_id)
        if block_id in self.free_table:
            self._push(block_id, self._NEW)

    def _push(self, block_id: int, kind: str) -> None:
        self._version[block_id] = self._version.get(block_id, 0) + 1
        self._kind[block_id] = kind
        self._counts[kind] += 1
        block = self.free_table[block_id]
        heapq.heappush(self._heaps[kind],
                       (block.last_accessed, -block.num_hashed_tokens,
                        block_id, self._version[block_id]))
        self._cleanup_if_necessary(kind)

    def _rebuild(self, kind: str) -> None:
        rebuilt = [(self.free_table[block_id].last_accessed,
                    -self.free_table[block_id].num_hashed_tokens, block_id,
                    self._version[block_id])
                   for block_id, block_kind in self._kind.items()
                   if block_kind == kind and block_id in self.free_table]
        heapq.heapify(rebuilt)
        self._heaps[kind] = rebuilt

    def _peek(self, kind: str) -> Optional[Tuple[float, int, int, int]]:
        heap = self._heaps[kind]
        while heap:
            last_accessed, num_hashed_tokens, block_id, version = heap[0]
            if (block_id in self.free_table
                    and self._version.get(block_id) == version
                    and self._kind.get(block_id) == kind):
                return last_accessed, num_hashed_tokens, block_id, version
            heapq.heappop(heap)
        return None

    def _pop_live(self, kind: str) -> int:
        _, _, block_id, _ = heapq.heappop(self._heaps[kind])
        return block_id

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed, set())
        self._push(block_id, self._tier(block_id))

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed
        kind = self._kind[block_id]
        self._counts[kind] -= 1
        self._push(block_id, kind)

    def evict(self) -> Tuple[int, int]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")
        now = time.time()
        limit = int(self.hold_budget * len(self.free_table))
        tier0: List[Tuple[float, int, int, str]] = []
        fresh = self._peek(self._NEW)
        if fresh is not None:
            tier0.append((fresh[0], fresh[1], fresh[2], self._NEW))
        reused = self._peek(self._REUSED)
        if reused is not None:
            expired = now - reused[0] > self.hold_s
            outside_budget = self._counts[self._REUSED] > limit
            if expired or outside_budget:
                tier0.append((reused[0], reused[1], reused[2], self._REUSED))
        if tier0:
            _, _, _, kind = min(tier0)
            return self._take(kind)
        if reused is not None:
            return self._take(self._REUSED)
        raise ValueError("No usable cache memory left")


class RFIFOEvictor(Evictor):
    """FIFO with admission from each label's recent block reuse.

    A label keeps the last ``WINDOW`` blocks whose outcome is known: a hit, or
    an eviction with no hit. Once that window is full, a rate of at least
    ``THRESHOLD`` admits the label's new blocks straight into tier 1. Otherwise
    they start in tier 0, which is emptied first. Order inside a tier is the
    insertion serial. A pin or a later access does not change it.
    """

    WINDOW = 100
    THRESHOLD = 0.05
    CLEANUP_THRESHOLD = 50
    _LOW = "low"
    _HIGH = "high"

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self._workloads: Dict[int, Set[str]] = {}
        self._version: Dict[int, int] = {}
        self._kind: Dict[int, str] = {}
        self._counts: Dict[str, int] = {self._LOW: 0, self._HIGH: 0}
        self._heaps: Dict[str, List[Tuple[int, int, int]]] = {
            self._LOW: [], self._HIGH: []}
        self._order: Dict[int, int] = {}
        self._serial = 0
        self._labels: Dict[int, Set[str]] = {}
        self._tier_of: Dict[int, str] = {}
        self._resolved: Set[int] = set()
        self._rates: Dict[str, Deque[bool]] = {}

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def note(self, content_hash: int, workload: str) -> None:
        self._workloads.setdefault(content_hash, set()).add(workload)

    def _admits(self, labels: Set[str]) -> bool:
        for label in labels:
            window = self._rates.get(label)
            if (window is not None and len(window) >= self.WINDOW
                    and sum(window) / len(window) >= self.THRESHOLD):
                return True
        return False

    def _record(self, block_id: int, hit: bool) -> None:
        if block_id in self._resolved or block_id not in self._labels:
            return
        self._resolved.add(block_id)
        for label in self._labels[block_id]:
            window = self._rates.setdefault(label, deque(maxlen=self.WINDOW))
            window.append(hit)

    def _clear_residency(self, block_id: int) -> None:
        self._labels.pop(block_id, None)
        self._tier_of.pop(block_id, None)
        self._resolved.discard(block_id)
        self._order.pop(block_id, None)

    def _ensure_order(self, block_id: int) -> None:
        if block_id not in self._order:
            self._serial += 1
            self._order[block_id] = self._serial

    def _cleanup_if_necessary(self, kind: str) -> None:
        live = max(self._counts[kind], 1)
        if len(self._heaps[kind]) > self.CLEANUP_THRESHOLD * live:
            self._rebuild(kind)

    def _push(self, block_id: int, kind: str) -> None:
        self._version[block_id] = self._version.get(block_id, 0) + 1
        self._kind[block_id] = kind
        self._counts[kind] += 1
        heapq.heappush(self._heaps[kind],
                       (self._order[block_id], block_id,
                        self._version[block_id]))
        self._cleanup_if_necessary(kind)

    def _rebuild(self, kind: str) -> None:
        rebuilt = [(self._order[block_id], block_id, self._version[block_id])
                   for block_id, block_kind in self._kind.items()
                   if (block_kind == kind and block_id in self.free_table
                       and block_id in self._order)]
        heapq.heapify(rebuilt)
        self._heaps[kind] = rebuilt

    def _peek(self, kind: str) -> Optional[Tuple[int, int, int]]:
        heap = self._heaps[kind]
        while heap:
            serial, block_id, version = heap[0]
            if (block_id in self.free_table
                    and self._version.get(block_id) == version
                    and self._kind.get(block_id) == kind
                    and self._order.get(block_id) == serial):
                return serial, block_id, version
            heapq.heappop(heap)
        return None

    def _take(self, kind: str) -> Tuple[int, int]:
        if self._peek(kind) is None:
            raise ValueError("No usable cache memory left")
        _, block_id, _ = heapq.heappop(self._heaps[kind])
        block = self.free_table.pop(block_id)
        self._kind.pop(block_id, None)
        self._counts[kind] -= 1
        self._version[block_id] = self._version.get(block_id, 0) + 1
        self._record(block_id, False)
        self._clear_residency(block_id)
        return block_id, block.content_hash

    def on_insert(self, block_id: int) -> None:
        if block_id in self.free_table:
            kind = self._kind.pop(block_id)
            self._counts[kind] -= 1
            self._version[block_id] = self._version.get(block_id, 0) + 1
            self.free_table.pop(block_id, None)
        self._record(block_id, False)
        self._clear_residency(block_id)
        self._serial += 1
        self._order[block_id] = self._serial

    def on_access(self, block_id: int) -> None:
        self._record(block_id, True)

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self._ensure_order(block_id)
        if block_id not in self._tier_of:
            labels = set(self._workloads.get(content_hash) or ())
            self._labels[block_id] = labels
            self._tier_of[block_id] = (self._HIGH if self._admits(labels)
                                       else self._LOW)
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed, set())
        self._push(block_id, self._tier_of[block_id])

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        kind = self._kind.pop(block_id)
        self._counts[kind] -= 1
        self._version[block_id] = self._version.get(block_id, 0) + 1
        self.free_table.pop(block_id)

    def evict(self) -> Tuple[int, int]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")
        for kind in (self._LOW, self._HIGH):
            if self._peek(kind) is not None:
                return self._take(kind)
        raise ValueError("No usable cache memory left")

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy, 
                 hold_s: Optional[float] = None, 
                 hold_budget: Optional[float] = None) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy in (EvictionPolicy.LFU, EvictionPolicy.FIFO):
        return OrderedEvictor(eviction_policy)
    elif eviction_policy == EvictionPolicy.TLRU:
        if hold_s is None or hold_budget is None:
            raise ValueError("TLRU requires hold_s and hold_budget")
        return TLRUEvictor(hold_s, hold_budget)
    elif eviction_policy == EvictionPolicy.TFIFO:
        return TFIFOEvictor()
    elif eviction_policy == EvictionPolicy.HFIFO:
        return HFIFOEvictor()
    elif eviction_policy == EvictionPolicy.HLRU:
        if hold_s is None or hold_budget is None:
            raise ValueError("HLRU requires hold_s and hold_budget")
        return HLRUEvictor(hold_s, hold_budget)
    elif eviction_policy == EvictionPolicy.RFIFO:
        return RFIFOEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
