# MLSys 2025 report source

`main.tex` is the final generated English paper. The supplied official `mlsys2025.sty`, bibliography style and auxiliary style files are unchanged. The course notice is set in the paper; this project is not represented as an accepted MLSys paper. The four confirmed authors, emails and Rice affiliation are used without a corresponding-author designation.

Compile the included source and vector figures from this directory:

```bash
tectonic --keep-logs main.tex
```

The first build may download TeX support files. The final build used Tectonic0.17.0 and the supplied MLSys2025 style. The standalone required PDF is `Prefix_Sharing_Final_Report.pdf`; `main.pdf` is its compiled source copy. References and appendices begin after an explicit page break and do not count toward the six-page main-text limit.

To regenerate numerical prose/tables rather than simply compile the supplied source, first regenerate `../figures` with the repository's `tools/comp536_final_figures.py`, then run:

```bash
python build_report.py
tectonic --keep-logs main.tex
```

`build_report.py` requires the complete validated108-run M2 and16-run mixed comparison tables; it does not launch experiments. Source-specific Task1 CDF supplements remain in the parent figure directory. The full source/code bundle contains those analysis scripts; the LaTeX-only archive is sufficient to compile the supplied final `main.tex`.
