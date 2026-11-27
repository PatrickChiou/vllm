# November 27 update changelog

- Fast-forwarded `comp536-sharing` from `f5d84b66cd` to Thomas Lin's `ad1d826542` without overwriting old deliverables.
- Reviewed RFIFO parameter transport through CLI/cache config/scheduler/allocator, the new five-mixture sweep/progress driver, and stop-token filtering in trace preparation.
- Reran the applicable simulator suite: 153 passed, zero failed, 73.93 seconds. No fresh benchmarks executed.
- Added all 125 rounded upstream sweep cells with source commit/hash, larger-cohort metadata, selected-setting heatmap and objective-dependent regret comparison; retained historical two-mixture results separately.
- Updated report, slides, English notes/Q&A, contribution evidence and reproducibility guidance. Corrected an old spoken-note claim: single-turn/block8 baselines nearly tie, rather than asserting FIFO materially exceeds LRU.
- Preserved old raw-validated 175-run counts; did not assert raw validation of new upstream results or combine smaller/larger scales as repeats.
- Remaining: missing new raw logs/manifests/runtime, repeated arrival seeds, held-out tuning, fresh OS/GPU validation and team review/rehearsal. The grid resume predicate can accept a partial successful prefix; no simulator fix was added in this document update.
- This revision is local; no new commit or push has been performed during this update.
