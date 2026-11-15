"""Turn parsed artifacts into Markdown, LaTeX, a README snippet and a JSON summary.

Cell conventions (same in every format):
    "not run"      the stage that produces this cell has no artifact
    "no samples"   the run exists but evaluated nothing in this category
    40% (2/5) [12-77]   % correct, k/n, Wilson 95% interval in percent
    1.23x (n=2)    mean speedup over the n correct kernels
    dagger         n < 20: too few samples to compare on
"""

from __future__ import annotations

from typing import Optional

import re

from evaluation.report.artifacts import (CATEGORIES, CATEGORY_LABELS, Artifacts, CategoryResult, EvalRun,
                                         mode_label)
from evaluation.report.stats import SMALL_N, Proportion, median

NOT_RUN = "not run"
NO_SAMPLES = "no samples"
DAGGER = "†"

# The Phase 5 table in MASTER_PLAN.md: (header, model, num_candidates, repair_rounds, metric)
PHASE5_COLUMNS = [
    ("Base % correct", "base", 1, 0, "correct"),
    ("Base avg speedup", "base", 1, 0, "speedup"),
    ("FT % correct", "ft", 1, 0, "correct"),
    ("FT avg speedup", "ft", 1, 0, "speedup"),
    ("FT best-of-4 + repair % correct", "ft", 4, 1, "correct"),
]

# The search comparison: pass@1 vs best-of-4 vs best-of-4 + 1 repair.
SEARCH_MODES = [(1, 0), (4, 0), (4, 1)]


# --------------------------------------------------------------------------- cells


def _category_rows(run: Optional[EvalRun]) -> dict[str, Optional[CategoryResult]]:
    """category -> result (None if the run is missing), plus an "All" row."""
    if run is None:
        return {c: None for c in (*CATEGORIES, "all")}
    rows = {c: run.categories.get(c, CategoryResult()) for c in CATEGORIES}
    rows["all"] = run.overall()
    return rows


def prop_cell(p: Proportion, latex: bool = False) -> str:
    if p.n == 0:
        return NO_SAMPLES
    lo, hi = p.interval
    pct = "\\%" if latex else "%"
    dash = "--" if latex else "–"
    mark = ("$^\\dagger$" if latex else DAGGER) if p.small else ""
    return f"{p.fraction * 100:.0f}{pct} ({p.successes}/{p.n}) [{lo * 100:.0f}{dash}{hi * 100:.0f}]{mark}"


def speedup_cell(run: EvalRun, result: CategoryResult, latex: bool = False) -> str:
    if result.n == 0:
        return NO_SAMPLES
    if run.interpreted:
        return "n/a (interpreter)"
    avg = result.avg_speedup
    if result.speedup_n == 0 or avg is None:
        return "-- (0 correct)" if latex else "– (0 correct)"
    times = "$\\times$" if latex else "×"
    mark = ("$^\\dagger$" if latex else DAGGER) if result.speedup_n < SMALL_N else ""
    return f"{avg:.2f}{times} (n={result.speedup_n}){mark}"


def _cell(run: Optional[EvalRun], result: Optional[CategoryResult], metric: str, latex: bool) -> str:
    if run is None or result is None:
        return NOT_RUN
    return prop_cell(result.proportion, latex) if metric == "correct" else speedup_cell(run, result, latex)


def _row_label(key: str) -> str:
    return "All" if key == "all" else CATEGORY_LABELS.get(key, key)


def phase5_table(art: Artifacts, latex: bool = False) -> list[list[str]]:
    """Rows of the Phase 5 category table, header first."""
    rows = [["Category"] + [h for h, *_ in PHASE5_COLUMNS]]
    per_column = [(_category_rows(art.run(m, nc, rr)), art.run(m, nc, rr), metric)
                  for _, m, nc, rr, metric in PHASE5_COLUMNS]
    for key in (*CATEGORIES, "all"):
        rows.append([_row_label(key)] + [_cell(run, cats[key], metric, latex) for cats, run, metric in per_column])
    return rows


def search_table(art: Artifacts, latex: bool = False) -> list[list[str]]:
    """% correct for each search mode, base and fine-tuned, overall and per category."""
    rows = [["Model", "Mode"] + [_row_label(k) for k in (*CATEGORIES, "all")]]
    for model in ("base", "ft"):
        for nc, rr in SEARCH_MODES:
            run = art.run(model, nc, rr)
            cats = _category_rows(run)
            label = "Fine-tuned" if model == "ft" else "Base"
            rows.append([label, mode_label(nc, rr)] + [_cell(run, cats[k], "correct", latex)
                                                         for k in (*CATEGORIES, "all")])
    return rows


def dataset_table(art: Artifacts, by: str = "category") -> list[list[str]]:
    """Outcome counts of the dataset build, per category or per template."""
    report = art.dataset or art.interpret
    if report is None:
        return []
    groups = report.per_category if by == "category" else report.per_template
    rows = [[by.capitalize(), "Generated", "Pass", "Incorrect", "OOM", "Timeout", "Other", "Median speedup"]]
    for name in sorted(groups, key=lambda k: (CATEGORIES.index(k.split("__")[0]) if k.split("__")[0] in CATEGORIES
                                               else len(CATEGORIES), k)):
        c = groups[name]
        med = "n/a (interpreter)" if report.interpreted else (f"{c.median_speedup:.2f}×"
                                                              if c.median_speedup is not None else "–")
        label = CATEGORY_LABELS.get(name, name) if by == "category" else name
        rows.append([label, str(c.total), str(c.passed), str(c.incorrect), str(c.oom), str(c.timeout),
                     str(c.other), med])
    return rows


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


TABLE_NOTES = (
    f"Percentages are % correct with k/n and a Wilson 95% interval in brackets. "
    f"{DAGGER} = fewer than {SMALL_N} samples: treat as indicative only. "
    "Speedup is the arithmetic mean of (PyTorch reference time / kernel time) over the correct kernels "
    "only, so base and fine-tuned speedups can average over different kernels. "
    "\"not run\" = no artifact for that stage."
)


# --------------------------------------------------------------------------- stage status & limitations


def stage_status(art: Artifacts) -> list[tuple[str, str]]:
    """(stage, what we found) for every stage the report can use, in pipeline order."""
    session_stages = (art.session or {}).get("stages", {})

    def why_missing(session_key: Optional[str]) -> str:
        if session_key and session_key in session_stages:
            return f"absent (session: {session_stages[session_key].get('status', '?')})"
        return "absent"

    out = []
    if art.dataset:
        out.append(("Dataset build (GPU)", f"present: {art.dataset.source}"))
    elif art.interpret:
        out.append(("Dataset build (GPU)", f"absent; interpreter-only check present: {art.interpret.source}"))
    else:
        out.append(("Dataset build (GPU)", why_missing("dataset")))
    for model, session_key, title in (("base", "baseline", "Base model"), ("ft", "eval", "Fine-tuned model")):
        if model == "ft":
            out.append(("Fine-tune (split_info.json)", "present" if art.split_info else why_missing("finetune")))
        for nc, rr in SEARCH_MODES:
            run = art.run(model, nc, rr)
            name = f"{title}, {mode_label(nc, rr)}"
            if run:
                out.append((name, f"present: {run.source}"))
            else:
                # Only pass@1 is produced by the session scripts today.
                out.append((name, why_missing(session_key) if (nc, rr) == (1, 0) else "absent"))
    scored = [r for r in art.reviews if r.scores]
    if scored:
        out.append(("Explanation review", "present: " + ", ".join(r.source for r in scored)))
    elif art.reviews:
        out.append(("Explanation review", "template present but no item has a rubric_score yet"))
    else:
        out.append(("Explanation review", "absent"))
    return out


def limitations(art: Artifacts) -> list[str]:
    """Limitation statements that follow from what actually ran (and the fixed ones from MASTER_PLAN)."""
    out = []
    runs = list(art.runs.values())
    missing = [name for name, status in stage_status(art) if status.startswith("absent")]
    if missing:
        out.append("Not run / no artifact: " + "; ".join(missing) + ". Those cells say \"not run\"; "
                   "no number in this report is estimated or carried over from elsewhere.")
    if not runs:
        out.append("No model evaluation results exist yet, so there is no correctness or speedup claim to make.")

    interpreted = [r.label for r in runs if r.interpreted]
    if interpreted or (art.interpret and not art.dataset):
        what = ", ".join(interpreted) if interpreted else "the dataset check"
        out.append(f"Interpreter-only ({what}): the Triton CPU interpreter checks correctness only; "
                   "it says nothing about speed, so no speedup is reported for these.")

    sized = [(r.label, [c.n for c in r.categories.values() if c.n]) for r in runs]
    sized = [(label, ns) for label, ns in sized if ns]
    if any(min(ns) < SMALL_N for _, ns in sized):
        per_cat = ", ".join(f"{label}: {min(ns)}–{max(ns)}" for label, ns in sized)
        out.append(f"Small samples: per-category n ({per_cat}) is below {SMALL_N} in places, so the 95% intervals "
                   "are wide and most category-to-category or base-vs-fine-tuned differences are not significant.")

    splits = {r.split for r in runs if r.split} | ({art.split_info.get("split")} if art.split_info else set())
    held_out = sorted({t for r in runs for t in r.held_out_templates}
                      | set((art.split_info or {}).get("held_out_templates") or []))
    if "random" in splits:
        out.append("A random (shape-level) split was used for at least one run: the same kernel with different "
                   "shape constants can appear in train and test, so fine-tuned scores may reflect memorisation.")
    if "template" in splits or held_out:
        out.append("Template split: test ops are whole templates held out from training"
                   + (f" ({', '.join(held_out)})" if held_out else "")
                   + ". That is one op per category, so results describe generalisation to a sibling op "
                     "in a seen category, not to new categories.")

    base, ft = art.run("base"), art.run("ft")
    if base and ft and set(base.entry_ids) != set(ft.entry_ids) and not (base.from_summary or ft.from_summary):
        out.append(f"Base and fine-tuned pass@1 were evaluated on different entries "
                   f"({len(set(base.entry_ids) & set(ft.entry_ids))} shared of {len(set(base.entry_ids))} / "
                   f"{len(set(ft.entry_ids))}); compare them with care.")
    if any(r.from_summary for r in runs):
        out.append("Some reports had no per-entry results; their numbers come from the report's own per-category "
                   "summary and the speedup distribution figure omits them.")
    inf = sum(c.infinite_speedups for r in runs for c in r.categories.values())
    if inf:
        out.append(f"{inf} correct kernel(s) were timed at 0 s (speedup = inf) and are left out of speedup means.")

    if art.dataset:
        oom = sum(c.oom for c in art.dataset.per_category.values())
        if oom:
            out.append(f"{oom} dataset shape(s) ran out of GPU memory and are absent from the dataset; "
                       "that is a memory limit of the test GPU, not a wrong kernel.")
        if art.dataset.smoke:
            out.append("The dataset report is a smoke run (one small instance per template), not the full grid.")
    gpu = ((art.session or {}).get("env") or {}).get("gpu")
    if any(not r.interpreted for r in runs) or art.dataset:
        out.append(f"Timings come from a single GPU ({gpu or 'model not recorded'}); speedups do not transfer "
                   "to other GPUs and are averaged over one timing configuration per kernel.")

    scored = [r for r in art.reviews if r.scores]
    if scored:
        out.append("Explanations were scored by hand with a 0/1/2 rubric by one reviewer on "
                   + ", ".join(f"{len(r.scores)} {r.model} samples" for r in scored) + "; not blind, not inter-rated.")

    out.append("Scope (fixed): a small model and dataset compared with 8B–32B RL systems; 5 narrow categories; "
               "no RL (search is inference-time only); demo-grade sandbox.")
    return out


# --------------------------------------------------------------------------- documents


def _review_lines(art: Artifacts) -> list[str]:
    lines = []
    for r in art.reviews:
        if not r.scores:
            lines.append(f"- `{r.source}`: {r.n_items} items, none scored yet.")
            continue
        dist = {s: r.scores.count(s) for s in (0, 1, 2)}
        lines.append(f"- {r.model} (`{r.source}`): {len(r.scores)}/{r.n_items} scored; mean {r.mean_score:.2f}/2; "
                     f"scores 2/1/0 = {dist[2]}/{dist[1]}/{dist[0]}; share scoring 2: {prop_cell(r.top_score)}")
    return lines or ["No explanation review found."]


def _provenance_lines(art: Artifacts) -> list[str]:
    lines = []
    env = (art.session or {}).get("env") or {}
    if env:
        lines.append("- Session env: " + ", ".join(f"{k}={env[k]}" for k in
                                                   ("gpu", "cuda", "torch", "triton", "git_commit") if k in env))
    for run in art.runs.values():
        lines.append(f"- {run.label}: `{run.source}` (model {run.model_name or '?'}, "
                     f"adapter {run.adapter_path or 'none'}, device {run.device or '?'}, "
                     f"split {run.split or 'not recorded'}, generated {run.generated_at or '?'})")
    for path in art.superseded:
        lines.append(f"- Superseded by a newer report of the same kind, not used: `{path}`")
    for path, why in art.excluded:
        lines.append(f"- Excluded `{path}`: {why}")
    for p in art.problems:
        lines.append(f"- Could not read {p}")
    return lines or ["- No artifacts found."]


def results_markdown(art: Artifacts, figures: list[tuple[str, str]], skipped_figures: list[tuple[str, str]]) -> str:
    parts = [
        "# KernelForge results",
        "",
        f"Generated by `python -m evaluation.report` from `{art.source}`. Every number below is computed from "
        "the artifacts listed under *Sources*; anything without an artifact is marked \"not run\".",
        "",
        "## Stages",
        "",
        markdown_table([["Stage", "Status"]] + [[s, v] for s, v in stage_status(art)]),
        "",
        "## Category results (Phase 5)",
        "",
        markdown_table(phase5_table(art)),
        "",
        TABLE_NOTES,
        "",
        "## Search: pass@1 vs best-of-4 vs best-of-4 + repair",
        "",
        markdown_table(search_table(art)),
        "",
        "## Dataset build",
        "",
    ]
    report = art.dataset or art.interpret
    if report:
        if report.interpreted:
            parts += ["*Interpreter-only report: correctness on CPU, no timings.*", ""]
        if report.smoke:
            parts += ["*Smoke run: one small instance per template, not the full shape grid.*", ""]
        parts += [markdown_table(dataset_table(art, "category")), "", markdown_table(dataset_table(art, "template")), ""]
    else:
        parts += ["No dataset report found.", ""]
    parts += ["## Explanation review", "", *_review_lines(art), "", "## Figures", ""]
    parts += [f"![{caption}]({path})\n\n*{caption}*\n" for path, caption in figures]
    parts += [f"- {name}: not drawn ({why})" for name, why in skipped_figures]
    parts += ["", "## Limitations", "", *[f"- {line}" for line in limitations(art)], "",
              "## Sources", "", *_provenance_lines(art), ""]
    return "\n".join(parts)


def readme_snippet(art: Artifacts) -> str:
    """A short 'Results' section to paste into README.md."""
    if not art.runs:
        lines = ["## Results", "", "No evaluation results yet: the baseline and fine-tuned evaluations have not "
                 "produced artifacts."]
        if art.dataset:
            total = sum(c.total for c in art.dataset.per_category.values())
            passed = sum(c.passed for c in art.dataset.per_category.values())
            lines += ["", f"Dataset: {passed}/{total} generated candidates passed GPU verification "
                      f"(see [docs/results/results.md](docs/results/results.md))."]
        return "\n".join(lines) + "\n"

    env = (art.session or {}).get("env") or {}
    held_out = sorted({t for r in art.runs.values() for t in r.held_out_templates})
    lines = ["## Results", ""]
    intro = f"Held-out evaluation on {env.get('gpu', 'a GPU')}"
    if held_out:
        intro += f" (template split; held-out ops: {', '.join(held_out)})"
    lines += [intro + ". Full report, figures and sources: [docs/results/results.md](docs/results/results.md).", "",
              markdown_table(phase5_table(art)), "", TABLE_NOTES, ""]
    lines += ["**Limitations.** " + " ".join(limitations(art)[:3]), ""]
    return "\n".join(lines)


def _tex(cell: str) -> str:
    """Escape the characters our cells can contain; cells built with latex=True are already escaped."""
    return re.sub(r"(?<!\\)%", r"\\%", cell).replace("×", "$\\times$").replace("–", "--")


def _latex_tabular(rows: list[list[str]], caption: str, label: str, label_cols: int = 1) -> str:
    rows = [[_tex(c) for c in row] for row in rows]
    cols = "l" * label_cols + "r" * (len(rows[0]) - label_cols)
    body = [" & ".join(rows[0]) + " \\\\", "\\midrule"] + [" & ".join(r) + " \\\\" for r in rows[1:]]
    return "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
        *body, "\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}",
    ])


def latex_fragments(art: Artifacts) -> str:
    """booktabs table fragments for a paper (needs \\usepackage{booktabs})."""
    note = ("Cells: \\% correct (k/n) [Wilson 95\\% interval]; $^\\dagger$ $n<20$. Speedup: mean over correct "
            "kernels vs.\\ the PyTorch reference. ``not run'': no artifact.")
    return "\n\n".join([
        "% Generated by python -m evaluation.report -- do not edit by hand; regenerate instead.",
        _latex_tabular(phase5_table(art, latex=True), "Per-category results on held-out templates. " + note, "tab:category-results"),
        _latex_tabular(search_table(art, latex=True), "Inference-time search: pass@1 vs.\\ best-of-4 vs.\\ best-of-4 + 1 repair. " + note,
                       "tab:search", label_cols=2),
    ]) + "\n"


def summary_json(art: Artifacts) -> dict:
    """Machine-readable version of every computed number (for checking by hand or other tools)."""
    def cat(run: EvalRun, c: CategoryResult) -> dict:
        p = c.proportion
        return {"n": c.n, "correct": c.correct, "pct_correct": None if p.fraction is None else p.fraction * 100,
                "wilson95_pct": None if p.interval is None else [p.interval[0] * 100, p.interval[1] * 100],
                "small_n": p.small, "speedup_n": 0 if run.interpreted else c.speedup_n,
                "avg_speedup": None if run.interpreted else c.avg_speedup,
                "median_speedup": None if run.interpreted else median(c.speedups)}
    return {
        "source": art.source,
        "stages": dict(stage_status(art)),
        "runs": {f"{r.model}/{r.mode_label}": {
            "source": r.source, "model_name": r.model_name, "interpreted": r.interpreted, "split": r.split,
            "held_out_templates": r.held_out_templates,
            "per_category": {k: cat(r, v) for k, v in r.categories.items()},
            "all": cat(r, r.overall())} for r in art.runs.values()},
        "explanation_reviews": [{"source": r.source, "model": r.model, "items": r.n_items, "scored": len(r.scores),
                                 "mean_score": r.mean_score} for r in art.reviews],
        "limitations": limitations(art),
    }
