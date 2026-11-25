"""Resumable required COMP536 experiments. Never overwrite a prior run.

M2 reuses documented 200-conversation, .05/.10 GiB, block8/16/32 settings.
Mixed sources reuse existing context32768 prepared data; new equal-token
cohorts retain the first causal prefix reaching 260k prompt tokens per label.
Only completed attempts with matching input/control hashes can be resumed.
"""
import argparse
import asyncio
import copy
import hashlib
import itertools
import os
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from vllm.simulator.analysis import analyze
from vllm.simulator.eviction_study import cache_blocks, working_set
from vllm.simulator.mix_workload import mix
from vllm.simulator.replay import replay
from vllm.simulator.study import server
from vllm.simulator.trace_builder import TraceBuilder
from vllm.simulator.workload import prepare

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
POLICIES = ['lru', 'lfu', 'fifo', 'tlru', 'tfifo', 'hlru', 'hfifo', 'rfifo']
LABELS = ['chat', 'agent', 'coding-agent', 'quality', 'bailian-chat',
          'bailian-coder', 'mooncake-agent', 'code']
REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def source_state():
    tracked = subprocess.check_output(['git', 'ls-files', 'vllm'], cwd=REPO,
                                      text=True).splitlines()
    hashes = {p: sha(REPO / p) for p in tracked}
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                 cwd=REPO, text=True).strip(),
            'driver_sha256': sha(Path(__file__)),
            'dirty_status': subprocess.check_output(['git', 'status', '--short'],
                                                    cwd=REPO, text=True),
            'implementation_sha256': hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def run_one(root, name, prepared, config, source):
    """A completed attempt is immutable. Interrupted attempts are retained."""
    fingerprint = {'workload_sha256': sha(prepared / 'workload.json'),
                   'trace_sha256': sha(prepared / 'trace.jsonl'),
                   'implementation_sha256': source['implementation_sha256'],
                   **config}
    if config.get('mixture'):
        fingerprint['allocator_counter_sha256'] = sha(REPO / 'tools/comp536_counting/sitecustomize.py')
    folder = root / 'runs' / name
    folder.mkdir(parents=True, exist_ok=True)
    for attempt in sorted(folder.glob('attempt-*')):
        marker = attempt / 'complete.json'
        if marker.exists():
            done = json.loads(marker.read_text())
            if done['fingerprint'] != fingerprint:
                raise ValueError(f'Incompatible completed attempt: {attempt}')
            print('RESUME completed', name, flush=True)
            return json.loads((attempt / 'summary.json').read_text())
    attempt = folder / f'attempt-{len(list(folder.glob("attempt-*"))) + 1:03d}'
    manifest = json.loads((prepared / 'workload.json').read_text())
    block = config['block_size']
    sizing = {}
    if config.get('cache_ratio') is not None:
        footprint = working_set(prepared / 'trace.jsonl', block)
        sizing = cache_blocks(config['cache_ratio'], footprint, block)
        sizing['working_set_blocks'] = footprint['unique_blocks']
    if config.get('mixture'):
        hook = REPO / 'tools/comp536_counting'
        os.environ['VLLM_SIM_EVICTION_COUNTS_PATH'] = str(
            (attempt / 'eviction_counts.json').resolve())
        previous = os.environ.get('PYTHONPATH', '')
        if str(hook) not in previous.split(os.pathsep):
            os.environ['PYTHONPATH'] = str(hook) + os.pathsep + previous
        fingerprint['allocator_counter_sha256'] = sha(hook / 'sitecustomize.py')
    start = time.time()
    print('START', name, flush=True)
    with server(manifest, prepared / 'trace.jsonl', attempt,
                block_size=block, cache_gb=config.get('cache_gb') or .02,
                policy=config['policy'], prefill_ms=.01, decode_ms=.1,
                max_num_seqs=config['concurrency'], startup_timeout=120,
                num_gpu_blocks=sizing.get('num_gpu_blocks'),
                hold_s=8 if config.get('mixture') else None,
                hold_budget=1 if config.get('mixture') else None) as (url, cache):
        metadata = {**source, **fingerprint, **sizing, 'name': name,
                    'model': MODEL, 'revision': REVISION,
                    'tokenizer_revision': REVISION,
                    'num_gpu_blocks': cache['num_gpu_blocks'],
                    'virtual_cache_bytes': cache['num_gpu_blocks'] * cache['block_bytes'],
                    'conversations': len(manifest['conversations']),
                    'prefill_ms_per_token': .01, 'decode_ms_per_seq': .1,
                    'arrival': manifest['arrival'],
                    'request_rate': manifest['request_rate'],
                    'think_time_s': manifest['think_time_s'],
                    'max_model_len': manifest['max_model_len'],
                    'chat_template_kwargs': manifest['chat_template_kwargs'],
                    'command': sys.argv,
                    'prepared_directory': str(prepared.resolve()),
                    'started_epoch': start}
        write(attempt / 'run.json', metadata)
        rows = asyncio.run(replay(manifest, url, attempt / 'requests.jsonl',
                                 concurrency=config['concurrency'], timeout=600))
    expected = sum(len(c['turns']) for c in manifest['conversations'])
    if len(rows) != expected or not all(r['success'] for r in rows):
        write(attempt / 'failed.json', {'expected': expected, 'received': len(rows)})
        raise RuntimeError(f'Failed replay: {attempt}')
    analyze(attempt, plots=False)
    summary = json.loads((attempt / 'summary.json').read_text())
    if config.get('mixture'):
        counter = attempt / 'eviction_counts.json'
        if not counter.exists():
            raise RuntimeError(f'Missing allocator counter after shutdown: {attempt}')
        summary.update(json.loads(counter.read_text()))
        write(attempt / 'summary.json', summary)
    write(attempt / 'complete.json', {'fingerprint': fingerprint,
                                    'finished_epoch': time.time(),
                                    'runtime_s': time.time() - start})
    print('END', name, f'{time.time()-start:.1f}s',
          f"S={summary['token_weighted_shared_fraction']:.5f}", flush=True)
    return summary


def m2(args, state):
    root = args.output / 'm2'
    builder = TraceBuilder(MODEL, revision=REVISION,
                           chat_template_kwargs={'date_string': '26 Jul 2024'})
    configs = []
    for seed in args.seeds:
        for mode in ['single', 'multi']:
            dest = root / 'prepared' / f'seed-{seed}' / mode
            if not (dest / 'workload.json').exists():
                prepare(args.data / 'sharegpt1000.jsonl', dest, builder, mode=mode,
                        limit=200, max_model_len=4096, arrival='poisson',
                        request_rate=100, seed=seed, think_time=0)
            for capacity, block, policy in itertools.product([.05, .10], [8, 16, 32], POLICIES[:3]):
                configs.append((dest, {'seed': seed, 'mode': mode, 'cache_gb': capacity,
                                       'block_size': block, 'policy': policy,
                                       'concurrency': 64, 'cache_ratio': None}))
    # Complete all configurations of seed 0 before later seeds.
    selected = []
    for seed in args.seeds:
        batch = [x for x in configs if x[1]['seed'] == seed]
        random.Random(seed).shuffle(batch)
        selected.extend(batch)
    if args.pilot:
        selected = [x for x in configs if x[1]['seed'] == args.seeds[0] and
                    x[1]['mode'] == 'multi' and x[1]['policy'] == 'lru' and
                    x[1]['block_size'] == 16 and x[1]['cache_gb'] == .05]
    summaries = []
    for dest, cfg in selected:
        name = (f"seed-{cfg['seed']}-{cfg['mode']}-{cfg['cache_gb']:g}GiB-"
                f"b{cfg['block_size']}-{cfg['policy']}")
        summaries.append(run_one(root, name, dest, cfg, state))
        write(root / ('pilot-comparison.json' if args.pilot else 'comparison.json'), summaries)


def token_total(workload):
    return sum(t['prompt_tokens'] for c in workload['conversations'] for t in c['turns'])


def select_balanced(source, dest):
    if (dest / 'workload.json').exists():
        return
    workload = json.loads((source / 'workload.json').read_text())
    chosen, count = [], 0
    for original in workload['conversations']:
        conv = copy.deepcopy(original)
        conv['turns'] = []
        for turn in original['turns']:
            conv['turns'].append(turn)
            count += turn['prompt_tokens']
            if count >= 260000:
                break
        chosen.append(conv)
        if count >= 260000:
            break
    if count < 260000:
        raise ValueError(f'Insufficient tokens: {source}: {count}')
    workload['conversations'] = chosen
    workload['balance_selection'] = {'rule': 'first causal prefix reaching token budget',
                                     'target_prompt_tokens': 260000,
                                     'actual_prompt_tokens': count,
                                     'source_workload_sha256': sha(source / 'workload.json')}
    ids = {(f"cmpl-{t['request_id']}-0" if 'prompt_token_ids' in t
            else f"chatcmpl-{t['request_id']}") for c in chosen for t in c['turns']}
    dest.mkdir(parents=True, exist_ok=False)
    write(dest / 'workload.json', workload)
    with (source / 'trace.jsonl').open() as fin, (dest / 'trace.jsonl').open('w') as fout:
        for line in fin:
            if json.loads(line)['request_id'] in ids:
                fout.write(line)


def mixed(args, state):
    root = args.output / 'mixed'
    for mixture in ['skewed', 'equal-token']:
        inputs = []
        for label in LABELS:
            src = args.mixed_prepared / label
            if mixture == 'equal-token':
                if token_total(json.loads((src / 'workload.json').read_text())) < 260000:
                    src = args.expanded_prepared / label
                dest = root / 'balanced-prepared' / label
                select_balanced(src, dest)
                src = dest
            inputs.append((label, src))
        for seed in args.seeds:
            prepared = root / mixture / f'seed-{seed}' / 'prepared'
            if not (prepared / 'workload.json').exists():
                mix(inputs, prepared, request_rate=100, think_time=1, seed=seed)
            order = list(POLICIES)
            random.Random(seed + 1709).shuffle(order)
            write(root / mixture / f'seed-{seed}' / 'policy-order.json', order)
            summaries = []
            for index, policy in enumerate(order):
                cfg = {'mixture': mixture, 'seed': seed, 'policy': policy,
                       'policy_order': index, 'block_size': 16,
                       'cache_gb': None, 'cache_ratio': .1, 'concurrency': 8,
                       'hold_s': 8, 'hold_budget': 1}
                name = f'{mixture}-seed-{seed}-{policy}'
                summaries.append(run_one(root, name, prepared, cfg, state))
                write(root / mixture / f'seed-{seed}' / 'comparison.json', summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['m2', 'mixed'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mixed-prepared', type=Path)
    p.add_argument('--expanded-prepared', type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    state = source_state()
    if args.phase == 'm2':
        m2(args, state)
    else:
        mixed(args, state)


if __name__ == '__main__':
    main()
