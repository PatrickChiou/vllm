# Experiment versions and populations

## Historical raw-validated studies

The 108 M2, 51 Task1 and 16 mixed runs are retained unchanged from the November 24 artifact set: 140,450 replayed requests, zero failures. Their counts, seeds, raw hashes, controls and CPU-only restrictions remain in the copied numerical JSON and inherited evidence. The original new-run lifecycles totaled 109.44 minutes; this is not runtime of the new sweep. Model/tokenizer Qwen/Qwen2.5-0.5B-Instruct revision 7ae557604adf67be50417f59c2c2f167def9a775.

## Repository-reported sensitivity sweep

Thomas Lin's ad1d826542 documents m3-hold-grid3: five mixtures × 25 settings, seed0, block16, cache ratio .10, concurrency8, 100 target requests/s, think1s, context32768, prefill .01ms/token and decode .1ms/sequence-step. Baselines LRU/LFU/FIFO/TFIFO run once per mixture; RFIFO uses 3 windows × 3 thresholds; TLRU and HLRU each use 3 holds × 2 budgets. HFIFO is not swept. Execution order is fixed, not randomized. Each mixture uses its own working-set-sized capacity; weights and absolute bytes differ across mixtures.

Larger cohort sizes and all rounded scores are extracted directly from docs/milestone3-task2.md with a source hash. Zero failures/no clamping are upstream statements, not independently observed here. Raw logs, prepared cohorts, run manifests, exact score numerators and runtime are absent. No new sweep was run in this update. Mean/SD/CI cannot be inferred from multiple settings; there is only one seed per cell.

The smaller m3-hold-grid2 table is a separate scale: coding-agent remains capped at12 while other source limits increase, changing token shares and working-set capacity. It is not pooled with grid3 or the historical two-mixture benchmark. Macro means equally weight the five mixtures; regret subtracts each setting from the best tested rounded score per mixture. Both are in-sample descriptive summaries, not token-pooled or held-out estimates. Margins near0.1pp are limited by rounding before considering run variability.

The new stop-token preparation changes potential traces. A rerun must use a fresh output directory and verify complete request IDs/counts because the grid resume predicate alone does not establish completeness. No GPU speedup or calibrated device latency is inferred.
