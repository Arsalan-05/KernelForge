"""Results report: turns a GPU session's kernelforge_outputs(.zip) into tables, figures and write-up fragments.

    python -m evaluation.report kernelforge_outputs.zip --out docs/results/

artifacts.py  find + parse the session artifacts (formats documented there)
stats.py      Wilson intervals, means
render.py     Markdown / LaTeX / README snippet / limitations / summary.json
figures.py    matplotlib PNGs (optional dependency)
"""
