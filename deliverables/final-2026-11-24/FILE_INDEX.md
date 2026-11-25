# Final artifact index

Primary standalone files in `outputs/final-2026-11-24`:

- `Prefix_Sharing_Final_Report.pdf`: final official-style paper.
- `Prefix_Sharing_Final_Presentation.pptx` and `.pdf`: editable slide deck and native Impress export.
- `Presentation_Speaker_Notes.md`, `Presentation_QA.md`, `README.md`: English script, Q&A and complete TA guide.
- `Final_Source_Changes.patch`: implementation/tests/tools/README patch against base3d501ab; final dated documents are supplied separately. The code archive already includes the patch's changes: do not apply twice.
- `COMP536_Final_Deliverables.zip`: compact report, slides, sources, figures/tables and validation records.
- `Prefix_Sharing_Final_LaTeX.zip`: complete source/styles/bibliography/vector figures to compile `report/main.tex`.
- `COMP536_Code_with_Git.zip`: current source and original `.git` history; no virtual environment, model cache or large dataset.
- `COMP536_Raw_Experiment_Evidence.zip`: all three study families, prepared cohorts/source inputs and retained incomplete attempt. Only completed markers count in the new studies.
- `ARTIFACTS.json`: archive/standalone SHA256, exact paths, sizes and raw-source locations.

Within the dated deliverable tree, `figures` contains validated CSV/JSON, vector PDFs/high-resolution PNGs, editable SVG diagrams and the all-eight-source CDF supplement. `evidence/validation.json` is not the data validator: use `figures/validation.json`; document/test/source checks live separately in `evidence`. Historical artifacts outside the dated tree remain preserved snapshots.

After extracting the raw archive, use `raw/m2`, `raw/mixed` and `raw/task1` as the figure generator's three inputs. Adjacent prepared cohorts resolve portably without altering absolute provenance paths. Source data are included for exact input reconstruction; model/tokenizer metadata must be downloaded separately from the recorded public revisions. Raw archive entries/hashes are listed in `evidence/raw-archive-index.json`.
