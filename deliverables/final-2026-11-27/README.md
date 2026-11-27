# November 27 report and presentation update

Rice University — COMP536. Thomas Lin, Yu-Ting Chung, Patrick Chiou, Yu-Xin Lai (confirmed order).

This revision integrates Thomas Lin's commit `ad1d826542` (November 25), following final-deliverables commit `f5d84b66cd`. The November 24 files are preserved. It updates the English paper, editable slides, native presentation PDF, notes, Q&A and figures without rerunning benchmarks or changing simulator source.

## Evidence and findings

- Current full simulator suite: **153 passed, 0 failed, 73.93 seconds** on `ad1d826542`; the earlier 146-test run is historical.
- Historical raw-validated evidence remains **175 runs, 140,450 requests, zero failures** (M2 108, Task1 51, mixed 16), executed under the earlier implementation. It is not a rerun of the updated reply builder.
- New upstream report: **125 cells**, five mixtures × 25 settings, one seed each. The guide reports zero failed requests/no clamping. Only rounded tables and plots are in Git; no raw replay files, full cohorts, runtime or manifests were available for independent verification here. Do not add these cells to the historical raw-validated count.
- The new sweep covers HLRU/TLRU holds 3/8/30 seconds, budgets .25/1; RFIFO windows 20/100/500, thresholds .02/.05/.20; and LRU/LFU/FIFO/TFIFO baselines. HFIFO is absent. Execution order is fixed. Selected settings are evaluated in-sample; no confidence intervals or universal winner are claimed.
- Reported larger-sweep leaders: RFIFO w500/t.05 on chat-heavy (23.6%), HLRU h8/b1 on protected-heavy (39.3%), LFU on template-heavy (47.6%), HLRU h30/b1 on trace-heavy/balanced (27.8%/40.1%). The 0.1pp trace margin is not reliable evidence of superiority.
- HLRU h30/b1 has the highest five-mixture macro mean (32.4%); RFIFO w100/t.02 has the smallest worst regret (6.9pp). These objectives differ and use rounded in-sample scores, not pooled prompt-token weights.

## Main files

- `report/Prefix_Sharing_Final_Report.pdf` and `report/main.tex`: official MLSys 2025 source and updated paper; six-page main-text limit, references/appendix separate.
- `presentation/Prefix_Sharing_Final_Presentation.pptx` and `.pdf`: 15 main slides plus 13 backup slides; native editable diagrams/tables/text and native Impress PDF export.
- `presentation/Presentation_Speaker_Notes.md`, `Presentation_QA.md`, `timing-summary.json`: English notes/Q&A, 810-second estimate. No human rehearsal or presenter assignment is claimed.
- `figures/hold-sweep-reported.json`/`.csv`: source-attributed extraction of all 25 × 5 rounded scores, cohort metadata and descriptive regret calculations.
- `evidence/full-tests.log`, `test-summary.json`, `update-provenance.json`, `upstream-commit.txt`: fresh tests and source attribution. `evidence/inherited-2026-11-24/` contains historical evidence snapshots; their source hashes, pre-publication Git state and 24-slide counts do not describe the new revision.

## Rebuild documents (repository root)

Use the documented Python 3.11 simulator environment. Document-only dependencies are in `requirements-document-tools.txt`; this file does not replace simulator requirements.

```bash
python -m pytest tests/simulator -q -p no:cacheprovider
python deliverables/final-2026-11-27/figures/build_hold_figures.py
python deliverables/final-2026-11-27/report/build_report.py
(cd deliverables/final-2026-11-27/report && tectonic --keep-logs main.tex)
python deliverables/final-2026-11-27/presentation/build_presentation.py
libreoffice --headless --convert-to pdf \
  --outdir deliverables/final-2026-11-27/presentation \
  deliverables/final-2026-11-27/presentation/Prefix_Sharing_Final_Presentation.pptx
```

The report builder reads both historical validated tables and the explicitly attributed reported sweep table. It never runs experiments. `main.tex`, supplied styles/bibliography and vector figures compile without the simulator. The upstream guide and original November 24 README retain simulator setup, data sourcing and large historical evidence locations.

To inspect or deliberately reproduce the new grid, configure the pinned model revision and source files specified in `vllm/simulator/hold_grid.py`, then use a fresh output directory:

```bash
python -m vllm.simulator.hold_grid --dry-run
python -m vllm.simulator.hold_grid --revision "$MODEL_REVISION" \
  --output experiments/m3-hold-grid-review-new
```

This update did not execute that expensive grid. The driver uses repository-relative dataset paths (including `data/sharegpt.json`). Its resume check accepts any nonempty set of successful rows; use a fresh output and check expected IDs/counts before treating a run as complete. Source and cohort changes can alter scores and working-set capacity. A zero-byte tracked prepared `agent.jsonl` is not a usable experiment dataset.
