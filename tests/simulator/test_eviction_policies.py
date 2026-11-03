import pytest

from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.core.evictor import EvictionPolicy, make_evictor


@pytest.mark.parametrize("policy,expected", [
    (EvictionPolicy.FIFO, 1), (EvictionPolicy.LFU, 2), (EvictionPolicy.LRU, 2)])
def test_policy_history_survives_pin_and_release(policy, expected):
    evictor = make_evictor(policy)
    evictor.on_insert(1)
    evictor.add(1, 101, 4, 1)
    evictor.on_insert(2)
    evictor.add(2, 102, 4, 2)
    # Block 1 is reused and pinned, then released at a later time.
    evictor.on_access(1)
    evictor.remove(1)
    evictor.add(1, 101, 4, 3)
    assert evictor.evict()[0] == expected
    assert evictor.num_blocks == 1


def test_lfu_frequency_survives_pin_and_resets_on_eviction():
    evictor = make_evictor(EvictionPolicy.LFU)
    for block_id in [1, 2]:
        evictor.on_insert(block_id)
        evictor.add(block_id, block_id + 100, 4, block_id)
    evictor.on_access(1)
    evictor.on_access(1)
    evictor.update(1, 3)
    evictor.on_access(2)
    evictor.update(2, 4)
    assert evictor.evict() == (2, 102)
    evictor.on_insert(2)
    evictor.add(2, 999, 4, 5)
    assert evictor.evict() == (2, 999)


@pytest.mark.parametrize("policy,retained", [
    (EvictionPolicy.FIFO, "b"), (EvictionPolicy.LFU, "a"),
    (EvictionPolicy.LRU, "a")])
def test_real_allocator_evicts_according_to_policy(policy, retained):
    allocator = PrefixCachingBlockAllocator(num_blocks=2, block_size=4,
                                             eviction_policy=policy)
    hashes = {}
    for time, name in enumerate(["a", "b", "a", "c"], 1):
        block = allocator.allocate_immutable_block(None, [ord(name)] * 4)
        hashes[name] = block.content_hash
        allocator.mark_blocks_as_accessed([block.block_id], time)
        allocator.mark_blocks_as_computed([])
        allocator.free(block)
    assert hashes[retained] in allocator._cached_blocks
    evicted = "b" if retained == "a" else "a"
    assert hashes[evicted] not in allocator._cached_blocks
