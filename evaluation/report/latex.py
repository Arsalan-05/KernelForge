"""LaTeX table fragments for the paper, conforming to docs/paper/RESULTS_INTERFACE.md (agent/d).

    table_dataset.tex     Table 1: 6 cells; 5 category rows, \\midrule, Total
    table_categories.tex  Table 2: 7 cells; 5 category rows
    table_search.tex      Table 3: 4 cells; rows Base, Fine-tuned
    table_training.tex    Table 4: 5 cells; one row per training run, reported run last
    table_rubric.tex      Table 5: 5 cells; rows Base, Fine-tuned

The paper owns header, column spec, caption and rules; a fragment is ONLY the body
rows, and its first line is a comment naming the source artifacts and the time.
Cells: percentages with one decimal and no % sign, speedups as 1.37$\\times$, integer
counts, "--" for an undefined value, "n/r" for a run that was not done.

A fragment is written only when its data exists (the paper shows red TODO rows for
a missing file). A fragment whose checks fail (dataset columns not adding up, base
and fine-tuned on different prompt counts) is NOT written; the reason is returned
so the report can show it. The richer metrics (intervals, geomean, fast_1, few-shot
runs) stay in results.md and summary.json.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from evaluation.report.artifacts import CATEGORIES, CATEGORY_LABELS, Artifacts, CategoryResult, EvalRun, merge
from evaluation.report.render import hours_minutes
from evaluation.report.stats import Proportion

FRAGMENT_NAMES = ("table_dataset", "table_categories", "table_search", "table_training", "table_rubric")
CELLS = {"table_dataset": 6, "table_categories": 7, "table_search": 4, "table_training": 5, "table_rubric": 5}
NOT_RUN = "n/r"
UNDEFINED = "--"
# The paper's "Base" and "FT" are zero-shot runs; a run that doesn't record its prompt mode
# is accepted in their place. Few-shot base results appear only in results.md.
PAPER_PROMPT_MODES = ("zero_shot", "unspecified")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _row(cells: list[str]) -> str:
    return " & ".join(cells) + " \\\\"


def _fragment(sources: list[str], rows: list[str], notes: list[str] = ()) -> str:
    """First line: provenance comment. Extra notes are comments too (they don't affect the table)."""
    files = sorted({src.replace(" (search)", "") for src in sources})  # a search run lives in its report
    head = [f"% generated {now()} from {', '.join(files) or 'no artifact'}"]
    head += [f"% {note}" for note in notes]
    return "\n".join(head + rows) + "\n"


def pct(p: Proportion) -> str:
    """One decimal, no % sign; undefined when there were no samples."""
    return UNDEFINED if p.n == 0 else f"{p.fraction * 100:.1f}"


def speedup(run: EvalRun, c: CategoryResult) -> str:
    """Mean speedup over passing kernels, two decimals; undefined if none passed or untimed."""
    value = None if run.interpreted else c.avg_speedup
    return UNDEFINED if value is None or c.speedup_n == 0 else f"{value:.2f}$\\times$"


def paper_run(art: Artifacts, model: str, num_candidates: int = 1, repair_rounds: int = 0) -> Optional[EvalRun]:
    for mode in PAPER_PROMPT_MODES:
        run = art.run(model, num_candidates, repair_rounds, mode)
        if run is not None:
            return run
    return None


def _partial_notes(runs: list[Optional[EvalRun]]) -> list[str]:
    return [f"WARNING: {r.label} is PARTIAL ({'incremental log only' if r.incremental else 'report says so'})"
            for r in runs if r is not None and r.partial]


# --------------------------------------------------------------------------- Table 1


def dataset_fragment(art: Artifacts) -> tuple[Optional[str], Optional[str]]:
    report = art.dataset
    if report is None or not report.per_category:
        return None, None
    rows, totals, complete = [], [0, 0, 0, 0, 0], True
    for cat in CATEGORIES:
        c = report.per_category.get(cat)
        if c is None:
            rows.append(_row([CATEGORY_LABELS[cat]] + [NOT_RUN] * 5))
            complete = False
            continue
        generated = c.reported_generated if c.reported_generated is not None else c.total
        values = [generated, c.passed, c.incorrect, c.oom, c.timeout + c.other]
        if sum(values[1:]) != generated:
            return None, (f"table_dataset.tex not written: for {CATEGORY_LABELS[cat]} GPU-verified + incorrect + "
                          f"OOM + other = {sum(values[1:])}, but generated = {generated}")
        totals = [t + v for t, v in zip(totals, values)]
        rows.append(_row([CATEGORY_LABELS[cat]] + [str(v) for v in values]))
    rows.append("\\midrule")
    rows.append(_row(["Total"] + ([str(v) for v in totals] if complete else [NOT_RUN] * 5)))
    notes = ([f"WARNING: PARTIAL build, {report.not_run if report.not_run is not None else 'some'} candidates "
              "not verified"] if report.partial else [])
    notes += ["smoke run: one instance per template, not the full grid"] if report.smoke else []
    return _fragment([report.source], rows, notes), None


# --------------------------------------------------------------------------- Table 2


def categories_fragment(art: Artifacts) -> tuple[Optional[str], Optional[str]]:
    base, ft, search = paper_run(art, "base"), paper_run(art, "ft"), paper_run(art, "ft", 4, 1)
    if base is None and ft is None and search is None:
        return None, None
    rows = []
    for cat in CATEGORIES:
        b = base.categories.get(cat, CategoryResult()) if base else None
        f = ft.categories.get(cat, CategoryResult()) if ft else None
        if b is not None and f is not None and b.n != f.n:
            return None, (f"table_categories.tex not written: base and fine-tuned were evaluated on different "
                          f"prompt counts ({CATEGORY_LABELS[cat]}: {b.n} vs {f.n}); the paper's table assumes "
                          "the same prompts for both")
        n = b.n if b is not None else f.n if f is not None else None
        s = search.categories.get(cat, CategoryResult()) if search else None
        rows.append(_row([
            CATEGORY_LABELS[cat],
            pct(b.proportion) if b is not None else NOT_RUN,
            speedup(base, b) if b is not None else NOT_RUN,
            pct(f.proportion) if f is not None else NOT_RUN,
            speedup(ft, f) if f is not None else NOT_RUN,
            pct(s.proportion) if s is not None else NOT_RUN,
            str(n) if n is not None else NOT_RUN,
        ]))
    notes = _partial_notes([base, ft, search])
    if search is not None:
        notes.append(f"column 6 (best-of-4 + repair) is on the search subset: {search.overall().n} prompts, "
                     "not the n in column 7")
    sources = [r.source for r in (base, ft, search) if r is not None]
    return _fragment(sources, rows, notes), None


# --------------------------------------------------------------------------- Table 3


def search_fragment(art: Artifacts) -> tuple[Optional[str], Optional[str]]:
    """pass@1 is computed on the SAME prompts the row's search scored (never pass@1 on all prompts
    next to search on a subset). A row with no search run shows pass@1 on all its prompts."""
    rows, notes, sources, any_search = [], [], [], False
    for model, label in (("base", "Base"), ("ft", "Fine-tuned")):
        pass1, s4, s41 = paper_run(art, model), paper_run(art, model, 4, 0), paper_run(art, model, 4, 1)
        searches = [r for r in (s4, s41) if r is not None]
        any_search = any_search or bool(searches)
        sources += [r.source for r in (pass1, *searches) if r is not None]
        ids = set(searches[-1].entry_ids) if searches and not searches[-1].from_summary else None
        if ids is not None and pass1 is not None and not pass1.from_summary:
            ids &= set(pass1.entry_ids)  # only prompts both scored are like-for-like

        def cell(run: Optional[EvalRun], restrict: bool) -> str:
            if run is None:
                return NOT_RUN
            if not restrict:
                return pct(run.overall().proportion)
            if ids is None or run.from_summary:
                return UNDEFINED  # can't be put on the same prompts
            return pct(merge(run.subset(ids).values()).proportion)

        rows.append(_row([label, cell(pass1, bool(searches)), cell(s4, True), cell(s41, True)]))
        if searches:
            notes.append(f"{label}: all cells on the same {len(ids) if ids is not None else '?'} prompts "
                         "(the search subset that pass@1 also scored)")
            if s4 is not None and s4.derived_from:
                notes.append(f"{label}: best-of-4 = round 0 of the best-of-4 + repair run")
        elif pass1 is not None:
            notes.append(f"{label}: no search run; pass@1 is over all {pass1.overall().n} prompts")
        notes += _partial_notes([pass1, *searches])
    if not any_search:
        return None, None
    return _fragment(sources, rows, notes), None


# --------------------------------------------------------------------------- Table 4


def training_fragment(art: Artifacts) -> tuple[Optional[str], Optional[str]]:
    runs = art.training_runs
    if not runs:
        return None, None
    rows, notes, sources = [], [], []
    for i, t in enumerate(runs, 1):
        label = "QLoRA run" if len(runs) == 1 else f"QLoRA run {i}" + (" (reported)" if i == len(runs) else "")
        if t.partial:
            label += " (partial)"
            notes.append(f"{t.name}: PARTIAL, stopped at step {t.stopped_at_step} of {t.steps_planned}")
        val_set = (t.split_info or {}).get("val")
        if t.best_eval_loss is not None:
            val = f"{t.best_eval_loss:.3f}"
        else:
            val = UNDEFINED if val_set == 0 else NOT_RUN  # no val set -> undefined; not logged -> n/r
        seconds, origin = t.wall_clock
        if origin:
            notes.append(f"{t.name}: wall clock from the {origin}")
        rows.append(_row([label, f"{t.final_step_loss:.3f}" if t.final_step_loss is not None else NOT_RUN, val,
                          hours_minutes(seconds) if seconds is not None else NOT_RUN, NOT_RUN]))
        sources.append(f"training/{t.name}")
    notes.append("peak GPU memory: n/r, no script records it")
    return _fragment(sources, rows, notes), None


# --------------------------------------------------------------------------- Table 5


def _paper_review(art: Artifacts, model: str):
    """The paper row's review: zero-shot (or unspecified) runs of that model; with several review
    files, the one with the most scored items."""
    candidates = [r for r in art.reviews if r.model == model and r.prompt_mode in PAPER_PROMPT_MODES and r.scores]
    return max(candidates, key=lambda r: len(r.scores)) if candidates else None


def rubric_fragment(art: Artifacts) -> tuple[Optional[str], Optional[str]]:
    base, ft = _paper_review(art, "base"), _paper_review(art, "ft")
    if base is None and ft is None:
        return None, None
    rows = []
    for label, review, copied in (("Base", base, UNDEFINED), ("Fine-tuned", ft, NOT_RUN)):
        if review is None:
            rows.append(_row([label] + [NOT_RUN] * 4))
        else:
            rows.append(_row([label] + [str(review.scores.count(s)) for s in (2, 1, 0)] + [copied]))
    notes = [f"{r.label}: {len(r.scores)} scored of {r.n_items} items" for r in (base, ft) if r is not None]
    notes.append("copied-from-template count: n/r, the review format has no such field")
    return _fragment([r.source for r in (base, ft) if r is not None], rows, notes), None


# --------------------------------------------------------------------------- all


BUILDERS = {"table_dataset": dataset_fragment, "table_categories": categories_fragment,
            "table_search": search_fragment, "table_training": training_fragment,
            "table_rubric": rubric_fragment}


def build_fragments(art: Artifacts) -> tuple[dict[str, str], list[str]]:
    """({name: LaTeX rows} for every table with data that passed its checks, [why a fragment was refused])."""
    out, problems = {}, []
    for name, build in BUILDERS.items():
        text, problem = build(art)
        if text is not None:
            out[name] = text
        if problem:
            problems.append(problem)
    return out, problems


def fragments(art: Artifacts) -> dict[str, str]:
    return build_fragments(art)[0]


def combined(frags: dict[str, str]) -> str:
    """tables.tex: every fragment that exists, for reading; the paper \\input{}s the separate files."""
    return "\n".join(f"% ---- {name}.tex ----\n{frags[name]}" for name in FRAGMENT_NAMES if name in frags)
