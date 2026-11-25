# Final implementation validation

This record supersedes the older audit's unresolved multiprocessing finding. It concerns the simulator on `comp536-sharing`, base `3d501ab40fb9a6b1bed5688d1db338b21f4a185f` plus the accompanying working-tree changes. Original reports and logs remain historical evidence.

## Architecture and scope

The online path is OpenAI-compatible HTTP preprocessing → engine client → v0 engine/sequence groups → iteration-level scheduler/continuous batching → block allocation and prefix lookup → `SimExecutor`/`SimExecutorAsync` → normal output processing. The simulator substitutes recorded output tokens and synthetic delays for model computation. It does not load model weights or allocate real GPU KV tensors. Model configuration and tokenizer files remain dependencies. The scheduler/control flow remains active; simulator selection and metadata plumbing are modified, so the entire upstream system is not described as unchanged.

Virtual block bytes use `2 × attention layers × KV heads per rank × head dimension × tokens/block × bytes/element`. The virtual block budget is a cache model, not measured GPU memory. Strict request-ID and prompt-token matching disambiguates duplicate prompts. Current output length determines the next replay token after recomputation. Output processing retains EOS/stop and cleanup behavior. The causal client waits for a previous reply before submitting its next turn and records actual dispatch lag.

## Minimal fixes

| Finding | Final change | Evidence |
|---|---|---|
| API handlers supplied `workload`, but `MQLLMEngineClient.generate` neither accepted nor serialized it | Added an optional argument through generate/process request, `RPCProcessRequest` constructors, pickling and backend `add_request`; missing labels remain `None` | New RPC round-trip/backend-submission regression for `None`, coding-agent and quality; HTTP label checks on both endpoints and both frontend modes |
| Unequal source rows could be silently truncated by ordinary `zip` (Ruff B905) | Explicit `zip(..., strict=True)` in the heterogeneous study | Targeted Ruff check and simulator suite pass |

The transport change does not alter direct-mode cache/executor policy decisions. The historical Task 1 direct-mode results remain applicable. Multiprocessing was repaired rather than globally disabled. The experiment helper continues using its historical direct frontend for consistent controls.

## Executed checks

```bash
python -m pytest tests/simulator -q -p no:cacheprovider
```

**146 passed, 0 failed in 68.10 seconds.** See `evidence/full-tests.log`. The suite covers the following behaviors; it is not the entire GPU-dependent upstream test suite.

| Behavior | Representative test module(s) |
|---|---|
| Exact prompt order, request IDs, duplicate replies, strict misses | `test_trace_store.py`, `test_trace_builder.py`, `test_sim_engine.py` |
| Prefill/decode, chunking, concurrent replay, EOS/limits | `test_sim_engine.py`, `test_sim_api_server.py` |
| Preemption/recomputation and final-token sampling | `test_sim_engine.py` |
| Finished/aborted request cleanup | `test_sim_engine.py` |
| Partial/full prefix hits and virtual working-set sizing | `test_sim_engine.py`, `test_eviction_study.py` |
| FIFO/LFU persistence across pin/release and reset on eviction | `test_eviction_policies.py` |
| Static, hit-gated and adaptive custom-policy decisions | `test_eviction_policies.py` |
| Causal replay, seeded arrivals, failures blocking descendants | `test_timing.py`, `test_study.py`, `test_workload.py` |
| Metrics, logical identities, zero-hit population and recomputation | `test_prefix_metrics.py` |
| Optional workload serialization and real API/block labels | `test_rpc_workload.py`, `test_sim_api_server.py` |
| Mixed preparation, IDs and causal order | `test_mix_workload.py` |

Targeted lint passes for the modified study and regression tests (`evidence/ruff-targeted.log`). Unrelated pre-existing pyupgrade warnings in the large legacy RPC modules were not mass-refactored; no repository-wide lint pass is claimed. The suite emits the existing pytest-asyncio fixture-loop-scope deprecation warning.

An isolated opt-in mixed-study counter check verifies that a successful eviction allocation increments once and a `None` result does not. A real server startup/health/graceful-shutdown check flushes an atomic zero-count file without inference. These are additional measurement checks, not added to the 146-test count. The hook preserves return values and victim decisions; its Python-call overhead remains part of the new mixed measurement environment. Promotion/protection counters are unavailable.

## Failed and interrupted attempts

Initial test attempts failed while model metadata/tokenizer caches were unavailable; assets were cached and final tests were run offline. An intermediate new completion-label assertion used a prompt too short to create a full block; the regression was corrected to use a full-block prompt before the successful full run. These setup/test-development logs are preserved separately and are not hidden by the final result.

An initial mixed LFU attempt was terminated with exit143 before completion. No completed marker exists for it; its data are retained and excluded. The reason for termination is undetermined. Resumed attempts use new directories and input/implementation fingerprints. Completed experimental cohorts, rather than partially successful requests, enter the final comparisons.

## Remaining limits

No known simulator correctness failure remains in the tested scope. LoRA, prompt adapters, speculative/multistep execution, GPU numerical output, GPU kernels/calibrated device timing and distributed KV transfer are outside this executor's supported study. Fresh operating-system installation and every upstream vLLM feature were not tested. Human review and presentation rehearsal remain team responsibilities; automated checks do not establish every member's understanding.
