"""CLI: python -m evaluation.report <outputs_dir_or_zip> --out docs/results/

Reads the artifacts of a GPU session (scripts/gpu_session.py) and writes:

    results.md                 full report: stages, tables, figures, limitations, sources
    readme_results_snippet.md  paste-ready "Results" section for README.md
    table_*.tex                one LaTeX fragment per table, ONLY for tables whose data exists
    tables.tex                 all fragments that exist, concatenated (absent if none)
    summary.json               every computed number, for checking
    figures/*.png              figures the data supports (needs matplotlib)

See evaluation/report/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.report.artifacts import load_artifacts
from evaluation.report.figures import make_figures
from evaluation.report.latex import FRAGMENT_NAMES, build_fragments, combined
from evaluation.report.render import readme_snippet, results_markdown, stage_status, summary_json


def build_report(src: Path, out: Path, figures: bool = True) -> list[Path]:
    """Write every report file into `out`; returns the paths written."""
    art = load_artifacts(src)
    out.mkdir(parents=True, exist_ok=True)
    made, skipped = make_figures(art, out) if figures else ([], [("all figures", "--no-figures")])
    frags, refused = build_fragments(art)
    latex_notes = ([f"Written: {', '.join(f'{n}.tex' for n in FRAGMENT_NAMES if n in frags)}."] if frags else [])
    latex_notes += [f"Not written (no data): {', '.join(f'{n}.tex' for n in FRAGMENT_NAMES if n not in frags)}."]
    latex_notes += [f"REFUSED: {why}" for why in refused]
    files = {
        "results.md": results_markdown(art, made, skipped, latex_notes),
        "readme_results_snippet.md": readme_snippet(art),
        "summary.json": json.dumps(summary_json(art), indent=2) + "\n",
    }
    files.update({f"{name}.tex": tex for name, tex in frags.items()})
    if frags:
        files["tables.tex"] = combined(frags)

    # A fragment from an earlier run whose data is now absent must not survive: the paper
    # would show it as a current result. Delete it so the paper falls back to its TODO box.
    for name in (*FRAGMENT_NAMES, "tables"):
        stale = out / f"{name}.tex"
        if f"{name}.tex" not in files and stale.exists():
            stale.unlink()
            print(f"  removed stale {stale.name} (its data is not in this source)")
    for name in FRAGMENT_NAMES:
        if name not in frags:
            print(f"  {name}.tex: not written")
    for why in refused:
        print(f"  {why}")

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
