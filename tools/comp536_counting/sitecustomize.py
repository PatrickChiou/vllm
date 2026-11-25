"""Opt-in allocator counter for isolated COMP536 experiment subprocesses.

Activated only by VLLM_SIM_EVICTION_COUNTS_PATH and this directory in
PYTHONPATH. It counts successful unused-block eviction allocations without
changing the allocator return value, victim order or reference counts.
No model weights are loaded by this hook. The normal API shutdown flushes
an atomic JSON counter; a missing file means unavailable, never zero.
"""
import atexit
import json
import os
from pathlib import Path

path = os.getenv('VLLM_SIM_EVICTION_COUNTS_PATH')
if path:
    from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator

    original = PrefixCachingBlockAllocator._maybe_allocate_evicted_block_id
    counts = {'cache_evictions': 0,
              'definition': 'successful evictor allocations of unused cached blocks',
              'measurement': 'opt-in allocator wrapper; not inferred from misses'}

    def counted(self):
        result = original(self)
        if result is not None:
            counts['cache_evictions'] += 1
        return result

    PrefixCachingBlockAllocator._maybe_allocate_evicted_block_id = counted

    def flush():
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(counts, indent=2) + '\n')
        temporary.replace(destination)

    atexit.register(flush)
