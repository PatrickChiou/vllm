"""Opt-in measurements of actual prefix skips in the simulator.

Block identity is a stable digest of the entire token prefix ending at a full
block. It is independent of physical block ID reuse. One access is recorded
per logical block per request, including blocks computed during decode; cache
hits are the blocks the scheduler supplied for skipping initial prefill.
"""
import hashlib
import json
import struct
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional

from vllm.sequence import SequenceGroupMetadata


class PrefixMetrics:

    def __init__(self, path: str, block_size: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._resources = ExitStack()
        self._file = self._resources.enter_context(
            open(path, "w", encoding="utf-8"))
        self.block_size = block_size
        self._start = time.monotonic()
        self._requests: Dict[str, List[bytes]] = {}
        self._last_access: Dict[bytes, float] = {}
        self.write({"event": "metadata", "version": 1,
                    "block_size": block_size,
                    "clock": "monotonic wall time since metrics start",
                    "block_identity": "sha256 of chained token prefixes"})

    def write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")

    def observe(self, metadata: SequenceGroupMetadata,
                now: Optional[float] = None) -> None:
        now = time.monotonic() - self._start if now is None else now
        request_id = metadata.request_id
        data = next(iter(metadata.seq_data.values()))
        cached_blocks = (len(metadata.computed_block_nums or [])
                         if metadata.is_prompt else 0)
        if request_id not in self._requests:
            self._requests[request_id] = []
            self.write({"event": "request", "request_id": request_id,
                        "time_s": now, "prompt_tokens": data.get_prompt_len(),
                        "cached_tokens": min(cached_blocks * self.block_size,
                                             data.get_prompt_len())})

        hashes = self._requests[request_id]
        end = min(data.get_len(), data.get_num_computed_tokens() +
                  (metadata.token_chunk_size or 0))
        if len(hashes) >= end // self.block_size:
            return
        tokens = data.get_token_ids()
        for index in range(len(hashes), end // self.block_size):
            chunk = tokens[index * self.block_size:
                           (index + 1) * self.block_size]
            previous = hashes[-1] if hashes else b""
            digest = hashlib.sha256(
                previous + struct.pack(f"!{len(chunk)}q", *chunk)).digest()
            hashes.append(digest)
            hit = (metadata.is_prompt and data.get_output_len() == 0
                   and index < cached_blocks)
            last = self._last_access.get(digest)
            self.write({"event": "block", "request_id": request_id,
                        "block_key": digest.hex(), "block_index": index,
                        "time_s": now, "hit": hit,
                        "reuse_gap_s": (now - last
                                        if hit and last is not None else None)})
            self._last_access[digest] = now

    def finish(self, request_ids: List[str]) -> None:
        for request_id in request_ids:
            self._requests.pop(request_id, None)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._resources.close()
