# Final experiment protocol and evidence boundaries

## Resources and authorization

The 2026-11-24 request explicitly authorizes minimal bug repair, full applicable validation, missing M2 studies and missing mixed studies after checking existing artifacts. Repository/work/outputs were searched; old Task 1 raw results are available, complete M2 and mixed comparison trees were not. Historical reports and partial TLRU data are retained, not treated as completed comparisons.

Observed resources:2 CPU cores,7.7 GiB RAM,4 GiB swap, initially29 GiB disk available. No GPU or model weights are required. One benchmark server runs at a time. The full simulator suite is completed before the main study batch; brief document-tool/analysis work may still share the host. Synthetic delays, HTTP and normal host scheduling affect wall-clock retention, so same-seed results need not be identical.

M2 pilot: documented200 eligible conversations/context 4096/concurrency 64,0.05 GiB/block 16/LRU/multi/seed 0.608 requests passed in 26.7s including server lifecycle/analysis. Its completed cell is reused in the final sweep. This supports an estimated tens-of-minutes M2 batch; mixed full cohorts were previously documented at several minutes per policy. Required comparisons take priority over optional ablations.

## Source and test state

Base branch:comp536-sharing, head3d501ab40fb9a6b1bed5688d1db338b21f4a185f plus final working-tree changes. Full history is recovered; assignment baseline94d545a1a18e20ea8763a6760194589b8a3c9065 is an ancestor. Remote main contains unrelated upstream development. There is no reset, rebase, force push or new remote push.

Optional workload labels now pass through MQLLMEngineClient.generate, its request processor, RPCProcessRequest pickle fields and MQLLMEngine's engine.add_request call. No-label defaults remain None.146 simulator tests pass; the isolated allocator-counter check additionally passes. Targeted Ruff checks resolve B905. Legacy module-wide pyupgrade warnings are not mass-refactored.

The old Task 1 study used the direct frontend and its executor/policy decisions are unchanged by this transport fix or strict-zip validation. All 51 old runs are independently rechecked and reused. New M2/mixed study helpers also select the direct frontend for historical control consistency; both frontend modes are separately integration-tested.

## Required experiments

- M2:36 configurations per seed, seeds 0/1/2, totaling 108. Modes single/multi; cache0.05/0.10 GiB; blocks 8/16/32; policiesLRU/LFU/FIFO.200 matched eligible conversations, context 4096, target 100/s, concurrency/max-sequences 64, think 0, time_scale1, prefill0.01ms/uncached token, decode0.1ms/sequence-step. This recovers the original documented recipe rather than mixing it with the fallback ratio sweep. The M2-A cells overlap/reuse M2-B LRU/block 16 configurations.
- Mixed: original skewed cohort and newly constructed equal-token cohort, eight policies each, seed 0, totaling 16. Context 32768, block 16, cache ratio 0.10 of distinct full prompt+generated blocks excluding final EOS, target 100/s, concurrency 8, think 1s, time_scale1. Hold 8s/budget 1 for TLRU/HLRU; other policies ignore those controls. Fresh server/cache per policy; order randomized with a recorded seed.
- Balanced input: retain each label's first causal turn prefix until at least 260k prompt tokens, including final-turn overshoot. Expand only deficient source preparations. Equal-token composition is reported alongside request composition; it is not exact equal tokens or equal requests.
- No optional window/threshold/hold ablations are claimed. Mixed n=1 has unknown variability. Additional seeds are supported by the driver but are not marked complete unless executed.

## Provenance and immutable attempts

New experiments live under `work/final-2026-11-24/experiments/`; old `work/rerun-2026-11-23/`, `work/m3task1-results/`, previous outputs and original documents remain unchanged. Each configuration has numbered attempt directories; only an input/control/implementation-matched complete.json can be resumed. An interrupted/failed attempt is retained and retried in a new directory. No completed attempt is silently rerun or overwritten.

Per-run run.json preserves source commit/dirty status, implementation fingerprint, exact argv, model/tokenizer revision, prepared-directory and workload/trace hashes, context, template kwargs, arrival/load, capacity, policy and synthetic delay. The prepared manifest/global dataset-provenance file supplies raw source identifiers and SHA256. Time_scale1 is the unchanged replay default; it is documented explicitly here even where old per-run schema omits that key. The final driver adds optional mixed instrumentation; its M2 execution branch is unchanged from the executed M2 batch.

Atomic aggregate comparison files are working indexes updated as completed attempts accumulate; immutable raw attempts remain the evidence. A stale/interrupted old status file is not a completed-run marker. Large datasets/model caches are outside Git and final compact documents.

## Metrics and independent validation

Primary score S=sum cached prompt tokens/sum prompt tokens. Mean/median per-request fractions, hit counts including zero-hit logical blocks, gap CDFs, actual send rates/dispatch lag, successful/failed counts, prompt/cached tokens and allocated capacity are also exported. Request rates are CPU replay observations, not GPU throughput.

Logical full-prefix identity is a chained SHA256 token hash independent of physical block ID. Successful-hit-associated gaps use monotonic wall time since the previous distinct request access and exclude missed reuse. An additional policy-independent diagnostic computes recurrence between nominally planned full prompt blocks (seed 0, block 16):4.65% single-turn and57.73% multi-turn prompt-token recurrence. It excludes decode-produced blocks, capacity, actual dispatch/completion and availability; it is not a serving-cache upper bound or measured hit interval. The diagnostic reads prepared traces only, with no additional benchmark.

New mixed runs opt into an allocator wrapper via an isolated PYTHONPATH/sitecustomize hook. It counts a successful `_maybe_allocate_evicted_block_id` return, preserves the return/ordering/refcount behavior and flushes an atomic count on normal shutdown. A missing count prevents the new mixed attempt from being marked complete. Its unit check exercises a real eviction return and an empty-table None return. This introduces small Python-call overhead and is disclosed; it does not change victim-selection logic. Old M2/Task 1 eviction counts and promotion/protection counters are unavailable and explicitly null, not zero or miss-derived estimates.

Final figure generation reads only existing raw attempts and validates unique request IDs, fractions/arithmetic, success/completeness, block/gap tables, label scores, actual prepared-file hashes, common within-comparison request cohorts and capacity. It writes new CSV/JSON/figures to a separate deliverable path and never calls a server. Sample SD requires repeated seeds; no CI, significance claim or GPU speedup is invented.

The final validator also streams raw block events independently: request IDs/labels/prompt and cached counts must match API-level rows; per-block access/hit counts and every ordered reuse-gap record must match the exported CSVs. Recorded cache geometry/policy and mean gap statistics are checked, and complete raw block streams are SHA256-hashed. The preliminary one-run-per-family check passed before the final all-run pass.
