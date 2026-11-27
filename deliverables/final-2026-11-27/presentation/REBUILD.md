# Rebuilding the editable presentation

Run after the complete experiment tables/figures have been validated. The presentation builder consumes the same numerical JSON/PNG files as the paper and does not run benchmarks.

```bash
python -m pip install -r deliverables/final-2026-11-27/requirements-document-tools.txt
python deliverables/final-2026-11-27/presentation/build_presentation.py
```

The final build used Python3.11.17, python-pptx1.0.2, ReportLab5.1.0 and Pillow13.0.0 in an isolated document-tool directory. DejaVu Sans/DejaVu Sans Bold fonts are required by the shared-layout builder. Editable PowerPoint text, shapes, tables and notes are retained; plots are high-resolution PNGs with vector PDF and generating source supplied separately.

The builder produces the PPTX, speaker script, timing JSON, layout-element JSON and a separately named `Prefix_Sharing_Layout_Companion.pdf`. That companion is not an Office rendering. Export the PPTX itself to the required PDF using LibreOffice Impress (the final export used24.2.7.2):

```bash
libreoffice --headless --convert-to pdf --outdir /path/to/new/export \
  deliverables/final-2026-11-27/presentation/Prefix_Sharing_Final_Presentation.pptx
```

The actual final export used an isolated private LibreOffice extraction and bootstrap paths, without modifying system packages or simulator dependencies. Its command/version/export log accompany the final evidence. Temporary renderer bootstrap debugging is not a simulator change. `VALIDATION.md` records final page counts, timing and visual checks. Rehearse the file in the actual presentation environment because PowerPoint/font substitution may wrap text differently. No presenter split or completed human rehearsal is inferred.

New sweep inputs are rounded upstream aggregates, explicitly source-attributed; historical data retain their original validation scope. Rebuild the reported sweep figures first with `figures/build_hold_figures.py`. No document builder launches a benchmark.
