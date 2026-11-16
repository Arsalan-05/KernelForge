# docs/paper — arXiv paper skeleton

| File | Purpose |
|---|---|
| `paper.tex` | The paper. Method and setup are written and checked against the code. All results are red `[TODO: …]` placeholders. |
| `references.bib` | Only entries verified against a primary source. |
| `SOURCES.md` | For each citation: where it was checked, and the only claims the paper makes about it. |
| `UNVERIFIED.md` | Things not cited because they couldn't be verified, plus repo-doc claims the sources contradict. |
| `CHECKLIST.md` | Decisions to make before the GPU run, fill-ins after it, and claims that depend on results. |
| `RESULTS_INTERFACE.md` | Exact row/column format of the five table fragments (`docs/results/table_*.tex`) that `paper.tex` reads when they exist. |
| `DOC_CORRECTIONS.md` | Exact text replacements for README / MASTER_PLAN / DOCUMENTATION / write-up, to apply after the merge. |

Build (TeX Live or Overleaf):

```bash
cd docs/paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

`% src:` comments in `paper.tex` name the file each repo claim comes from. If that code changes,
re-check the sentence.
