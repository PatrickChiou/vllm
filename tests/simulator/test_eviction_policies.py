import pytest
import time

from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.core.evictor import EvictionPolicy, make_evictor
from vllm.config import CacheConfig


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


def _add(evictor, block_id, content_hash, workload, last_accessed,
         num_hashed_tokens=4):
    labels = [workload] if isinstance(workload, str) else workload
    for label in labels:
        evictor.note(content_hash, label)
    # evictor.add(block_id, content_hash, 4, last_accessed)
    evictor.add(block_id, content_hash, num_hashed_tokens, last_accessed)


def test_tlru_evicts_one_shot_before_a_fresh_multiturn_block():
    evictor = make_evictor(EvictionPolicy.TLRU, 1.0, 0.5)
    now = time.time()
    _add(evictor, 1, 101, "code", now)
    _add(evictor, 2, 102, "quality", now)
    _add(evictor, 3, 103, "mooncake-agent", now)
    # _add(evictor, 4, 104, "chat", now)
    _add(evictor, 4, 104, "agent", now)
    assert [evictor.evict()[0] for _ in range(3)] == [1, 2, 3]


def test_tlru_evicts_expired_multiturn_before_a_fresh_one():
    hold_s = 1.0
    evictor = make_evictor(EvictionPolicy.TLRU, hold_s, 0.5)
    now = time.time()
    _add(evictor, 1, 101, "agent", now - hold_s - 1)
    # _add(evictor, 2, 102, "chat", now)
    _add(evictor, 2, 102, "coding-agent", now)
    assert evictor.evict()[0] == 1


def test_tlru_keeps_a_shared_prefix_ahead_of_one_shot():
    evictor = make_evictor(EvictionPolicy.TLRU, 1.0, 0.5)
    now = time.time()
    _add(evictor, 1, 101, ["chat", "code"], now - 10)
    _add(evictor, 2, 102, "quality", now)
    assert evictor.evict()[0] == 2


def test_tlru_budget_keeps_the_newest_protected_blocks():
    evictor = make_evictor(EvictionPolicy.TLRU, 1.0, 0.5)
    now = time.time()
    _add(evictor, 1, 101, "agent", now - 0.4)
    _add(evictor, 2, 102, "bailian-chat", now - 0.3)
    _add(evictor, 3, 103, "coding-agent", now - 0.2)
    _add(evictor, 4, 104, "bailian-coder", now - 0.1)
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2


def test_tlru_update_reorders_protected_blocks():
    evictor = make_evictor(EvictionPolicy.TLRU, 1.0, 0.5)
    now = time.time()
    _add(evictor, 1, 101, "agent", now - 0.4)
    # _add(evictor, 2, 102, "chat", now - 0.3)
    _add(evictor, 2, 102, "bailian-chat", now - 0.3)
    _add(evictor, 3, 103, "coding-agent", now - 0.2)
    _add(evictor, 4, 104, "bailian-coder", now - 0.1)
    evictor.update(1, now)
    assert evictor.evict()[0] == 2
    assert evictor.evict()[0] == 3


def test_tlru_update_restores_an_expired_block():
    hold_s = 1.0
    evictor = make_evictor(EvictionPolicy.TLRU, hold_s, 0.5)
    now = time.time()
    _add(evictor, 1, 101, "agent", now - hold_s - 1)
    _add(evictor, 2, 102, "code", now)
    evictor.update(1, now)
    assert evictor.evict()[0] == 2


def test_tlru_remove_drops_a_pinned_block():
    evictor = make_evictor(EvictionPolicy.TLRU, 1.0, 0.5)
    now = time.time()
    _add(evictor, 1, 101, "code", now)
    # _add(evictor, 2, 102, "chat", now)
    _add(evictor, 2, 102, "agent", now)
    evictor.remove(1)
    assert evictor.num_blocks == 1
    assert evictor.evict()[0] == 2


def test_tlru_zero_budget_protects_nothing():
    evictor = make_evictor(EvictionPolicy.TLRU, 1.0, 0.0)
    now = time.time()
    # _add(evictor, 1, 101, "chat", now - 0.2)
    _add(evictor, 1, 101, "bailian-chat", now - 0.2)
    _add(evictor, 2, 102, "agent", now - 0.1)
    _add(evictor, 3, 103, ["chat", "code"], now - 10)
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2
    assert evictor.evict()[0] == 3


def test_tlru_tie_evicts_the_longer_prefix_first():
    evictor = make_evictor(EvictionPolicy.TLRU, 1.0, 1.0)
    now = time.time()
    _add(evictor, 1, 101, "agent", now, num_hashed_tokens=16)
    _add(evictor, 2, 102, "coding-agent", now, num_hashed_tokens=64)
    assert evictor.evict()[0] == 2


def test_tfifo_evicts_a_later_one_shot_before_an_older_protected_block():
    evictor = make_evictor(EvictionPolicy.TFIFO)
    _add(evictor, 1, 101, "agent", 0)
    _add(evictor, 2, 102, "chat", 1000)
    _add(evictor, 3, 103, "code", 1001)
    assert [evictor.evict()[0] for _ in range(2)] == [2, 3]


def test_tfifo_orders_protected_blocks_by_insertion():
    evictor = make_evictor(EvictionPolicy.TFIFO)
    _add(evictor, 1, 101, "coding-agent", 50)
    _add(evictor, 2, 102, "bailian-chat", 0)
    evictor.update(1, 0)
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2


def test_tfifo_keeps_insertion_order_across_a_pin():
    evictor = make_evictor(EvictionPolicy.TFIFO)
    _add(evictor, 1, 101, "agent", 1)
    _add(evictor, 2, 102, "coding-agent", 2)
    evictor.remove(1)
    _add(evictor, 1, 101, "agent", 3)
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2


def test_tfifo_evict_assigns_a_new_serial_when_the_id_returns():
    evictor = make_evictor(EvictionPolicy.TFIFO)
    _add(evictor, 1, 101, "agent", 1)
    _add(evictor, 2, 102, "coding-agent", 2)
    assert evictor.evict()[0] == 1
    _add(evictor, 1, 103, "agent", 3)
    assert evictor.evict()[0] == 2
    assert evictor.evict()[0] == 1


def test_tfifo_keeps_a_shared_prefix_after_both_other_tiers():
    evictor = make_evictor(EvictionPolicy.TFIFO)
    _add(evictor, 1, 101, ["chat", "code"], 0)
    _add(evictor, 2, 102, "quality", 1)
    _add(evictor, 3, 103, "agent", 2)
    assert [evictor.evict()[0] for _ in range(3)] == [2, 3, 1]


def test_cache_config_accepts_tlru():
    config = CacheConfig(16, 0.9, 0, "auto",
                         prefix_cache_eviction_policy="tlru")
    assert config.prefix_cache_eviction_policy == "tlru"
    tfifo = CacheConfig(16, 0.9, 0, "auto",
                        prefix_cache_eviction_policy="tfifo")
    assert tfifo.prefix_cache_eviction_policy == "tfifo"
    hfifo = CacheConfig(16, 0.9, 0, "auto",
                        prefix_cache_eviction_policy="hfifo")
    assert hfifo.prefix_cache_eviction_policy == "hfifo"
    hlru = CacheConfig(16, 0.9, 0, "auto",
                       prefix_cache_eviction_policy="hlru")
    assert hlru.prefix_cache_eviction_policy == "hlru"
    rfifo = CacheConfig(16, 0.9, 0, "auto",
                        prefix_cache_eviction_policy="rfifo")
    assert rfifo.prefix_cache_eviction_policy == "rfifo"
    with pytest.raises(ValueError, match="Unknown"):
        CacheConfig(16, 0.9, 0, "auto",
                    prefix_cache_eviction_policy="reuse")


def test_hfifo_evicts_an_unhit_block_before_a_hit_of_any_label():
    evictor = make_evictor(EvictionPolicy.HFIFO)
    _add(evictor, 1, 101, "agent", 1)
    _add(evictor, 2, 102, "chat", 2)
    evictor.on_access(2)
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2


def test_hfifo_keeps_insertion_order_after_a_hit_and_a_pin():
    evictor = make_evictor(EvictionPolicy.HFIFO)
    _add(evictor, 1, 101, "code", 1)
    _add(evictor, 2, 102, "quality", 2)
    evictor.on_access(1)
    evictor.on_access(2)
    evictor.update(1, 100)
    evictor.remove(1)
    _add(evictor, 1, 101, "code", 3)
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2


def test_hfifo_evict_clears_the_hit_when_the_id_returns():
    evictor = make_evictor(EvictionPolicy.HFIFO)
    _add(evictor, 1, 101, "chat", 1)
    evictor.on_access(1)
    _add(evictor, 2, 102, "agent", 2)
    evictor.on_access(2)
    assert evictor.evict()[0] == 1
    _add(evictor, 1, 103, "chat", 3)
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2


def test_hfifo_ignores_a_shared_label_until_the_block_is_hit():
    evictor = make_evictor(EvictionPolicy.HFIFO)
    _add(evictor, 1, 101, ["chat", "code"], 0)
    _add(evictor, 2, 102, "quality", 1)
    evictor.on_access(2)
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2


def test_hlru_evicts_an_unhit_block_before_a_fresh_hit():
    evictor = make_evictor(EvictionPolicy.HLRU, 1.0, 1.0)
    now = time.time()
    _add(evictor, 1, 101, "agent", now)
    _add(evictor, 2, 102, "code", now - 10)
    evictor.on_access(2)
    evictor.update(2, now)
    assert evictor.evict()[0] == 1


def test_hlru_evicts_an_expired_hit_before_a_newer_unhit_block():
    hold_s = 1.0
    evictor = make_evictor(EvictionPolicy.HLRU, hold_s, 1.0)
    now = time.time()
    _add(evictor, 1, 101, "coding-agent", now - hold_s - 1)
    evictor.on_access(1)
    _add(evictor, 2, 102, "chat", now)
    assert evictor.evict()[0] == 1


def test_hlru_update_refreshes_a_hit_block():
    evictor = make_evictor(EvictionPolicy.HLRU, 1.0, 1.0)
    now = time.time()
    _add(evictor, 1, 101, "chat", now - 0.4)
    _add(evictor, 2, 102, "agent", now - 0.1)
    evictor.on_access(1)
    evictor.on_access(2)
    evictor.update(1, now)
    assert evictor.evict()[0] == 2


def test_hlru_tie_evicts_the_longer_prefix_first():
    evictor = make_evictor(EvictionPolicy.HLRU, 1.0, 1.0)
    now = time.time()
    _add(evictor, 1, 101, "code", now, num_hashed_tokens=16)
    _add(evictor, 2, 102, "quality", now, num_hashed_tokens=64)
    assert evictor.evict()[0] == 2


def _resolve_rfifo(evictor, label, hits, misses, start_id):
    for offset in range(hits + misses):
        block_id = start_id + offset
        _add(evictor, block_id, 100000 + block_id, label, offset)
        if offset < hits:
            evictor.on_access(block_id)
        assert evictor.evict()[0] == block_id


def test_rfifo_window_argument_admits_after_fewer_blocks():
    evictor = make_evictor(EvictionPolicy.RFIFO, rfifo_window=4,
                           rfifo_threshold=0.5)
    _resolve_rfifo(evictor, "agent", hits=2, misses=2, start_id=1000)
    _add(evictor, 1, 101, "agent", 1)
    _add(evictor, 2, 102, "chat", 2)
    assert evictor.evict()[0] == 2
    assert evictor.evict()[0] == 1


def test_rfifo_waits_for_a_full_window_before_protecting():
    evictor = make_evictor(EvictionPolicy.RFIFO)
    _resolve_rfifo(evictor, "agent", hits=50, misses=0, start_id=1000)
    _add(evictor, 1, 101, "agent", 1)
    _add(evictor, 2, 102, "chat", 2)
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2


def test_rfifo_protects_a_label_at_the_five_percent_line():
    evictor = make_evictor(EvictionPolicy.RFIFO)
    _resolve_rfifo(evictor, "agent", hits=5, misses=95, start_id=1000)
    _resolve_rfifo(evictor, "chat", hits=4, misses=96, start_id=2000)
    _add(evictor, 1, 101, "agent", 1)
    _add(evictor, 2, 102, "chat", 2)
    assert evictor.evict()[0] == 2
    assert evictor.evict()[0] == 1


def test_rfifo_keeps_insertion_order_inside_the_protected_tier():
    evictor = make_evictor(EvictionPolicy.RFIFO)
    _resolve_rfifo(evictor, "agent", hits=5, misses=95, start_id=1000)
    _resolve_rfifo(evictor, "coding-agent", hits=5, misses=95, start_id=2000)
    _add(evictor, 1, 101, "coding-agent", 1)
    _add(evictor, 2, 102, "agent", 2)
    evictor.update(1, 50)
    evictor.remove(1)
    _add(evictor, 1, 101, "coding-agent", 3)
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2


def test_rfifo_forgets_a_label_after_the_window_fills_with_misses():
    evictor = make_evictor(EvictionPolicy.RFIFO)
    _resolve_rfifo(evictor, "chat", hits=100, misses=0, start_id=1000)
    _resolve_rfifo(evictor, "chat", hits=0, misses=100, start_id=2000)
    _add(evictor, 1, 101, "chat", 1)
    _add(evictor, 2, 102, "code", 2)
    assert evictor.evict()[0] == 1