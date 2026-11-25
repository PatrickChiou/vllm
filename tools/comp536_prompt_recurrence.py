"""Policy-independent *planned prompt* recurrence, not cache-hit availability.

Prompt blocks only: ignores decode-produced blocks, actual completion time,
capacity and causality delays. It is not an infinite-cache serving bound.
"""
import hashlib
import json
import struct
from pathlib import Path


def analyze(prepared: Path, block_size: int = 16):
    workload = json.loads((prepared / 'workload.json').read_text())
    traces = {e['request_id']: e for e in
              (json.loads(line) for line in
               (prepared / 'trace.jsonl').read_text().splitlines())}
    turns = sorted((t for c in workload['conversations'] for t in c['turns']),
                   key=lambda t: (t['arrival_time_s'], t['request_id']))
    last, gaps, repeated, total, prompt = {}, [], 0, 0, 0
    for turn in turns:
        key = 'chatcmpl-' + turn['request_id']
        tokens = traces[key]['prompt_token_ids']
        prompt += len(tokens)
        digest = b''
        for offset in range(0, len(tokens) - block_size + 1, block_size):
            chunk = tokens[offset:offset + block_size]
            digest = hashlib.sha256(digest + struct.pack(
                f'!{len(chunk)}q', *chunk)).digest()
            total += 1
            if digest in last:
                repeated += 1
                gaps.append(turn['arrival_time_s'] - last[digest])
            last[digest] = turn['arrival_time_s']
    return {'block_size': block_size, 'requests': len(turns),
            'prompt_tokens': prompt, 'full_prompt_block_accesses': total,
            'repeated_prompt_block_accesses': repeated,
            'unique_prompt_blocks': len(last),
            'planned_prompt_recurrence_fraction': repeated * block_size / prompt,
            'planned_recurrence_gaps_s': gaps,
            'definition': 'prior full prompt block seen at a preceding planned arrival',
            'limitations': 'prompt-only; no decode blocks, capacity, availability, actual dispatch or completion; not a serving-cache upper bound'}
