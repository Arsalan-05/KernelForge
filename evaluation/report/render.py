"""Turn parsed artifacts into table rows, Markdown, a README snippet and a JSON summary.
(LaTeX lives in latex.py and reuses the row builders here.)

Cell conventions (same in every format):
    "not run"      the stage that produces this cell has no artifact
    "no samples"   the run exists but evaluated nothing in this category
    40% (2/5) [12-77]   % correct, k/n, Wilson 95% interval in percent
    1.23x (n=2)    mean (or geometric mean) speedup over the n correct, finitely timed kernels
    dagger         n < 20: too few samples to compare on

Three speedup views, because each hides something:
    mean     arithmetic mean over correct kernels; matches evaluation/common.py
    geomean  geometric mean over the same kernels; the right average for ratios
    fast_1   correct AND faster than the reference, over ALL attempts (KernelBench);
             the only one that also penalises failed attempts
"""

from __future__ import annotations

from typing import Optional

from evaluation.report.artifacts import (CATEGORIES, CATEGORY_LABELS, Artifacts, CategoryResult, EvalRun,
                                         mode_label)
from evaluation.report.stats import SMALL_N, Proportion, median

NOT_RUN = "not run"
NO_SAMPLES = "no samples"
DAGGER = "†"

# The Phase 5 table in MASTER_PLAN.md: (header, model, num_candidates, repair_rounds, metric)
# plus the geometric-mean and fast_1 speedup views next to each arithmetic mean.
PHASE5_COLUMNS = [
    ("Base % correct", "base", 1, 0, "correct"),
    ("Base mean speedup", "base", 1, 0, "mean"),
    ("Base geomean speedup", "base", 1, 0, "geomean"),
    ("Base fast_1", "base", 1, 0, "fast_1"),
    ("FT % correct", "ft", 1, 0, "correct"),
    ("FT mean speedup", "ft", 1, 0, "mean"),
    ("FT geomean speedup", "ft", 1, 0, "geomean"),
    ("FT fast_1", "ft", 1, 0, "fast_1"),
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


def speedup_cell(run: EvalRun, result: CategoryResult, latex: bool = False, kind: str = "mean") -> str:
    """kind = "mean" (arithmetic) or "geomean"; both over correct kernels with a finite speedup."""
    if result.n == 0:
        return NO_SAMPLES
    if run.interpreted:
        return "n/a (interpreter)"
    if kind == "geomean" and run.from_summary:
        return "n/a (summary-only report)"
    value = result.avg_speedup if kind == "mean" else result.geomean_speedup
    if result.speedup_n == 0 or value is None:
        return "-- (none timed)" if latex else "– (none timed)"
    times = "$\\times$" if latex else "×"
    mark = ("$^\\dagger$" if latex else DAGGER) if result.speedup_n < SMALL_N else ""
    return f"{value:.2f}{times} (n={result.speedup_n}){mark}"


def fast1_cell(run: EvalRun, result: CategoryResult, latex: bool = False) -> str:
    if result.n == 0:
        return NO_SAMPLES
    if run.interpreted:
        return "n/a (interpreter)"
    if result.fast_1 is None:
        return "n/a (summary-only report)"
    return prop_cell(result.fast_1, latex)


def _cell(run: Optional[EvalRun], result: Optional[CategoryResult], metric: str, latex: bool) -> str:
    if run is None or result is None:
        return NOT_RUN
    if metric == "correct":
        return prop_cell(result.proportion, latex)
    if metric == "fast_1":
        return fast1_cell(run, result, latex)
    return speedup_cell(run, result, latex, kind=metric)


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


def training_table(art: Artifacts) -> list[list[str]]:
    """(Metric, Value) rows for the newest training run, only for values that exist. [] if none."""
    t = art.training
    if t is None:
        return []
    rows = []
    split = t.split_info or {}
    cfg = split.get("config") or {}
    state = t.trainer_state or {}
    if cfg.get("model_name"):
        rows.append(["Base model", str(cfg["model_name"])])
    if split.get("split"):
        rows.append(["Split", f"{split['split']}"])
    if all(k in split for k in ("train", "val", "test")):
        rows.append(["Train / val / test entries", f"{split['train']} / {split['val']} / {split['test']}"])
    if split.get("held_out_templates"):
        rows.append(["Held-out templates", ", ".join(split["held_out_templates"])])
    if "lora_r" in cfg and "lora_alpha" in cfg:
        rows.append(["LoRA r / alpha (configured)", f"{cfg['lora_r']} / {cfg['lora_alpha']}"])
    if "learning_rate" in cfg:
        rows.append(["Learning rate (configured)", f"{cfg['learning_rate']:g}"])
    if "per_device_train_batch_size" in cfg and "gradient_accumulation_steps" in cfg:
        rows.append(["Effective batch (configured)",
                     str(cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"])])
    # Epochs/steps actually run come from the trainer, not the config (the CPU fallback runs 1 epoch).
    if state.get("epoch") is not None:
        rows.append(["Epochs completed", f"{float(state['epoch']):g}"])
    elif "num_epochs" in cfg:
        rows.append(["Epochs (configured)", str(cfg["num_epochs"])])
    if state.get("global_step") is not None:
        rows.append(["Optimizer steps", str(state["global_step"])])
    if t.final_train_loss is not None:
        rows.append(["Final train loss", f"{t.final_train_loss:.4f}"])
    if t.final_eval_loss is not None:
        rows.append(["Final validation loss", f"{t.final_eval_loss:.4f}"])
    if t.wall_clock_s is not None:
        rows.append(["Training wall-clock", f"{t.wall_clock_s / 3600:.2f} h"])
    if t.peak_memory_gb is not None:
        rows.append(["Peak GPU memory", f"{t.peak_memory_gb:.1f} GB"])
    return [["Metric", "Value"]] + rows if rows else []


def rubric_table(art: Artifacts, latex: bool = False) -> list[list[str]]:
    """One row per explanation review that has at least one scored item. [] if none."""
    scored = [r for r in art.reviews if r.scores]
    if not scored:
        return []
    rows = [["Model", "Scored", "Mean score (0-2)", "Scores 2 / 1 / 0", "Share scoring 2"]]
    names = {"base": "Base", "ft": "Fine-tuned", "unknown": "Unknown model"}
    for r in scored:
        rows.append([names[r.model], f"{len(r.scores)}/{r.n_items}", f"{r.mean_score:.2f}",
                     " / ".join(str(r.scores.count(s)) for s in (2, 1, 0)), prop_cell(r.top_score, latex)])
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
    "Speedup = PyTorch reference time / kernel time. *Mean* is the arithmetic mean and *geomean* the geometric "
    "mean, both over correct kernels with a finite speedup only (n in brackets), so base and fine-tuned can "
    "average over different kernels. *fast_1* = attempts that are correct AND faster than the reference, "
    "out of all attempts, with a Wilson 95% interval. \"not run\" = no artifact for that stage."
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
            out.append(("Fine-tune (training/run_*)", f"present: {art.training.name}" if art.training
                        else "present (split_info.json)" if art.split_info else why_missing("finetune")))
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
    if any(not r.interpreted for r in runs):
        out.append("Speedup is reported three ways. The arithmetic mean (matches evaluation/common.py) and the "
                   "geometric mean are over correct kernels only, so they say nothing about how often the model "
                   "fails; fast_1 (correct and faster than the PyTorch reference, over all attempts) does.")
    inf = sum(c.infinite_speedups for r in runs for c in r.categories.values())
    if inf:
        out.append(f"{inf} correct kernel(s) were timed at 0 s (speedup = inf). They are left out of the mean, "
                   "the geometric mean and fast_1, and counted separately in summary.json.")
    if art.run("ft") and not (art.training and art.training.trainer_state):
        out.append("Training loss and wall-clock are not in the artifacts (scripts/gpu_session.py collects "
                   "split_info.json and the adapter, not trainer_state.json), so they are not reported.")

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
    parts += ["## Training", ""]
    parts += [markdown_table(training_table(art)) if training_table(art) else "No training artifacts found.", ""]
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


def summary_json(art: Artifacts) -> dict:
    """Machine-readable version of every computed number (for checking by hand or other tools)."""
    def cat(run: EvalRun, c: CategoryResult) -> dict:
        p = c.proportion
        return {"n": c.n, "correct": c.correct, "pct_correct": None if p.fraction is None else p.fraction * 100,
                "wilson95_pct": None if p.interval is None else [p.interval[0] * 100, p.interval[1] * 100],
                "small_n": p.small, "speedup_n": 0 if run.interpreted else c.speedup_n,
                "avg_speedup": None if run.interpreted else c.avg_speedup,  # arithmetic mean, correct only
                "geomean_speedup": None if run.interpreted else c.geomean_speedup,  # geometric mean, correct only
                "median_speedup": None if run.interpreted else median(c.speedups),
                "fast_1": prop(c.fast_1) if not run.interpreted else None,  # correct & >1x over all attempts
                "infinite_speedups_excluded": c.infinite_speedups}

    def prop(p: Optional[Proportion]) -> Optional[dict]:
        if p is None or p.n == 0:
            return None
        return {"k": p.successes, "n": p.n, "pct": p.fraction * 100,
                "wilson95_pct": [p.interval[0] * 100, p.interval[1] * 100], "small_n": p.small}

    t = art.training
    return {
        "source": art.source,
        "stages": dict(stage_status(art)),
        "runs": {f"{r.model}/{r.mode_label}": {
            "source": r.source, "model_name": r.model_name, "interpreted": r.interpreted, "split": r.split,
            "held_out_templates": r.held_out_templates,
            "per_category": {k: cat(r, v) for k, v in r.categories.items()},
            "all": cat(r, r.overall())} for r in art.runs.values()},
        "training": None if t is None else {
            "run": t.name, "trainer_state_source": t.trainer_state_source or None,
            "final_train_loss": t.final_train_loss, "final_eval_loss": t.final_eval_loss,
            "wall_clock_s": t.wall_clock_s, "peak_memory_gb": t.peak_memory_gb, "split_info": t.split_info},
        "explanation_reviews": [{"source": r.source, "model": r.model, "items": r.n_items, "scored": len(r.scores),
                                 "mean_score": r.mean_score} for r in art.reviews],
        "limitations": limitations(art),
    }
