"""CLI: python -m evaluation.report <outputs_dir_or_zip> --out docs/results/

Reads the artifacts of a GPU session (scripts/gpu_session.py) and writes:

    results.md                 full report: stages, tables, figures, limitations, sources
    readme_results_snippet.md  paste-ready "Results" section for README.md
    tables.tex                 booktabs table fragments for a paper
    summary.json               every computed number, for checking
    figures/*.png              figures the data supports (needs matplotlib)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.report.artifacts import load_artifacts
from evaluation.report.figures import make_figures
from evaluation.report.render import latex_fragments, readme_snippet, results_markdown, stage_status, summary_json


def build_report(src: Path, out: Path, figures: bool = True) -> list[Path]:
    """Write every report file into `out`; returns the paths written."""
    art = load_artifacts(src)
    out.mkdir(parents=True, exist_ok=True)
    made, skipped = make_figures(art, out) if figures else ([], [("all figures", "--no-figures")])
    files = {
        "results.md": results_markdown(art, made, skipped),
        "readme_results_snippet.md": readme_snippet(art),
        "tables.tex": latex_fragments(art),
        "summary.json": json.dumps(summary_json(art), indent=2) + "\n",
    }
    written = []
    for name, text in files.items():
        (out / name).write_text(text)
        written.append(out / name)
    written += [out / rel for rel, _ in made]

    # Console summary: what was found, what wasn't.
    for stage, status in stage_status(art):
        print(f"  {stage:<40} {status}")
    for name, why in skipped:
        print(f"  figure {name}: skipped ({why})")
    for path, why in art.excluded:
        print(f"  excluded {path}: {why}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.report", description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="kernelforge_outputs/ directory or kernelforge_outputs.zip")
    parser.add_argument("--out", type=Path, default=Path("docs/results"), help="output directory")
    parser.add_argument("--no-figures", action="store_true", help="skip the matplotlib figures")
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"error: {args.source} does not exist", file=sys.stderr)
        return 2
    try:
        written = build_report(args.source, args.out, figures=not args.no_figures)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\nWrote {len(written)} files to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
