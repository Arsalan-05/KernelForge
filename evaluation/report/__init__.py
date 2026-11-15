"""Results report: turns a GPU session's kernelforge_outputs(.zip) into tables, figures and write-up fragments.

    python -m evaluation.report kernelforge_outputs.zip --out docs/results/

artifacts.py  find + parse the session artifacts (formats documented there)
stats.py      Wilson intervals, means
render.py     table rows, Markdown, README snippet, limitations, summary.json
latex.py      one LaTeX fragment per table, written only when its data exists
figures.py    matplotlib PNGs (optional: pip install -r evaluation/report/requirements.txt)

Usage, outputs and the eval-report interface: evaluation/report/README.md
"""
