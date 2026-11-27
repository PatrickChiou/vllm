# Current implementation validation

Source reviewed: `ad1d8265420c2515882300710d1cdf5d375515ba`. No simulator source was edited by this documentation update.

The new commit exposes `--sim-rfifo-window` and `--sim-rfifo-threshold` (defaults 100 and .05), validates configuration ranges, and propagates the settings through the cache configuration, scheduler, block manager and CPU/GPU allocators to RFIFO. The trace builder now removes configured EOS/stop IDs from recorded replies to avoid replay ending before the expected completion length. The new regression covers Qwen end-of-text filtering. Preparation may change recorded outputs and later working sets; old data are not silently regenerated.

The full applicable suite ran on this source: **153 passed, 0 failed, 73.93 seconds**. Seven tests were added since the 146-test snapshot: five grid/progress tests, one RFIFO admission test, one stop-token test. The grid dimension test establishes 5 × 25 settings; it does not prove 125 replays finished.

Review finding: `hold_grid.run_succeeded` accepts a nonempty successful request prefix, without checking expected request IDs/counts. Consequently an interrupted prefix can be skipped on resume. The upstream guide reports complete runs, but raw logs are needed to verify that claim. This limitation is disclosed rather than patched as part of a report/PPT update.

The driver uses fixed settings order, does not record the same source/cohort fingerprints as the historical final driver, and has no independently accessible raw evidence here. The committed zero-byte `prepared/agent.jsonl` supplies no usable workload. Historical exact-token/API/counter results remain in `evidence/inherited-2026-11-24` and describe their recorded source version.
