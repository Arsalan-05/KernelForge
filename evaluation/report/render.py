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

Two rules this module never breaks:
    * zero-shot and few-shot base runs are separate columns/rows, never pooled;
    * a search run (scored on a search_limit subset) is only ever compared with
      pass@1 on the SAME prompts, recomputed from the pass@1 run's per-entry rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evaluation.report.artifacts import (CATEGORIES, CATEGORY_LABELS, Artifacts, CategoryResult, EvalRun,
                                         group_label, merge, mode_label)
from evaluation.report.stats import SMALL_N, Proportion, median

NOT_RUN = "not run"
NO_SAMPLES = "no samples"
DAGGER = "†"
SUBSET_MARK = "‡"

# The search comparison: pass@1 vs best-of-4 vs best-of-4 + 1 repair.
SEARCH_MODES = [(1, 0), (4, 0), (4, 1)]
ROW_KEYS = (*CATEGORIES, "all")
OUTCOME_ORDER = ("correct", "incorrect", "no_kernel", "error", "timeout", "crash", "harness_exception",
                 "not_verified")
PROMPT_ORDER = {"zero_shot": 0, "unspecified": 1, "few_shot": 2}


# --------------------------------------------------------------------------- cells


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


def _by_row(categories: Optional[dict[str, CategoryResult]]) -> dict[str, Optional[CategoryResult]]:
    """category -> result plus an "All" row; all None when the run is missing."""
    if categories is None:
        return {k: None for k in ROW_KEYS}
    rows = {c: categories.get(c, CategoryResult()) for c in CATEGORIES}
    rows["all"] = merge(categories.values())
    return rows


def _row_label(key: str) -> str:
    return "All" if key == "all" else CATEGORY_LABELS.get(key, key)


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def sorted_runs(art: Artifacts) -> list[EvalRun]:
    """Base before fine-tuned, zero-shot before few-shot, pass@1 before search."""
    return sorted(art.runs.values(), key=lambda r: (r.model != "base", PROMPT_ORDER.get(r.prompt_mode, 3),
                                                    r.prompt_mode, r.num_candidates, r.repair_rounds))


def n_scored(run: EvalRun) -> int:
    return len(run.rows) if not run.from_summary else run.overall().n


# --------------------------------------------------------------------------- Phase 5 category table


def model_groups(art: Artifacts) -> list[tuple[str, Optional[str]]]:
    """(model, prompt mode) column groups, base first. A model with no run gets one
    (model, None) group so its columns still appear, as "not run"."""
    groups = []
    for model in ("base", "ft"):
        modes = art.prompt_modes(model)
        groups += [(model, m) for m in modes] if modes else [(model, None)]
    return groups


def short_label(model: str, prompt_mode: Optional[str]) -> str:
    return group_label(model, prompt_mode or "unspecified").replace("Fine-tuned", "FT")


def phase5_table(art: Artifacts, latex: bool = False) -> list[list[str]]:
    """The MASTER_PLAN Phase 5 table, one column block per (model, prompt mode):
    % correct, mean / geomean speedup, fast_1; fine-tuned blocks add best-of-4 + repair."""
    header, columns = ["Category"], []
    for model, pm in model_groups(art):
        name = short_label(model, pm)
        run = art.run(model, 1, 0, pm) if pm else None
        for metric, title in (("correct", "% correct"), ("mean", "mean speedup"), ("geomean", "geomean speedup"),
                              ("fast_1", "fast_1")):
            header.append(f"{name} {title}")
            columns.append((run, _by_row(run.categories if run else None), metric))
        if model == "ft":
            search = art.run(model, 4, 1, pm) if pm else None
            header.append(f"{name} best-of-4 + repair % correct {SUBSET_MARK}")
            columns.append((search, _by_row(search.categories if search else None), "correct"))
    rows = [header]
    for key in ROW_KEYS:
        rows.append([_row_label(key)] + [_cell(run, cats[key], metric, latex) for run, cats, metric in columns])
    return rows


# --------------------------------------------------------------------------- search on the same prompts


@dataclass
class SearchGroup:
    """Search runs of one model + prompt mode that scored the same prompts, with the
    pass@1 run of that model + prompt mode to compare against on exactly those prompts."""

    model: str
    prompt_mode: str
    ids: Optional[frozenset]  # None when the search report has no per-entry results
    runs: list[EvalRun]
    pass1: Optional[EvalRun]

    @property
    def label(self) -> str:
        return group_label(self.model, self.prompt_mode)

    @property
    def search_limit(self) -> Optional[int]:
        return next((r.search_limit for r in self.runs if r.search_limit is not None), None)

    @property
    def pass1_covered(self) -> int:
        """How many of the search prompts the pass@1 run also scored."""
        if self.pass1 is None or self.ids is None:
            return 0
        return len(self.ids & set(self.pass1.entry_ids))

    @property
    def shared_ids(self) -> Optional[set]:
        """Search prompts that pass@1 also scored; the only like-for-like population."""
        if self.pass1 is None or self.ids is None or self.pass1.from_summary:
            return None
        return set(self.ids) & set(self.pass1.entry_ids)

    @property
    def incomplete_pass1(self) -> bool:
        return self.shared_ids is not None and len(self.shared_ids) < len(self.ids)

    def pass1_same_prompts(self) -> Optional[dict[str, CategoryResult]]:
        """pass@1 restricted to the search prompts; None if it can't be computed."""
        if self.pass1 is None or self.ids is None or self.pass1.from_summary:
            return None
        return self.pass1.subset(set(self.ids))


def search_groups(art: Artifacts) -> list[SearchGroup]:
    grouped: dict[tuple, list[EvalRun]] = {}
    for run in art.runs.values():
        if run.is_search:
            ids = None if run.from_summary else frozenset(run.entry_ids)
            grouped.setdefault((run.model, run.prompt_mode, ids), []).append(run)
    groups = [SearchGroup(model, pm, ids, sorted(runs, key=lambda r: (r.num_candidates, r.repair_rounds)),
                          art.run(model, 1, 0, pm))
              for (model, pm, ids), runs in grouped.items()]
    return sorted(groups, key=lambda g: (g.model != "base", PROMPT_ORDER.get(g.prompt_mode, 3), g.label,
                                         -(len(g.ids) if g.ids else 0)))


def search_table(art: Artifacts, latex: bool = False) -> list[list[str]]:
    """% correct for pass@1 on the search prompts, then each search mode, per group. [] if no search ran."""
    groups = search_groups(art)
    if not groups:
        return []
    rows = [["Model", "Prompts", "Mode"] + [_row_label(k) for k in ROW_KEYS]]
    for g in groups:
        if g.ids is None:
            prompts = "unknown (no per-entry results)"
        else:
            prompts = f"same {len(g.ids)}" + (f" (search_limit {g.search_limit})" if g.search_limit else "")
        subset = g.pass1_same_prompts()
        if g.pass1 is None:
            pass1_cells = [NOT_RUN] * len(ROW_KEYS)
        elif subset is None:
            pass1_cells = ["n/a (no per-entry results)"] * len(ROW_KEYS)
        else:
            cats = _by_row(subset)
            pass1_cells = [prop_cell(cats[k].proportion, latex) for k in ROW_KEYS]
        mode = "pass@1, same prompts"
        if subset is not None and g.pass1_covered < len(g.ids):
            mode += f" ({g.pass1_covered} of {len(g.ids)} scored)"
        rows.append([g.label, prompts, mode] + pass1_cells)
        for run in g.runs:
            cats = _by_row(run.categories)
            rows.append([g.label, prompts, run.mode_label] + [prop_cell(cats[k].proportion, latex) for k in ROW_KEYS])
        if g.incomplete_pass1:
            # pass@1 missed some search prompts: repeat the search rows on the shared prompts only,
            # so there is one strictly like-for-like comparison.
            shared = f"shared {len(g.shared_ids)}"
            for run in g.runs:
                cats = _by_row(run.subset(g.shared_ids))
                rows.append([g.label, shared, f"{run.mode_label}, prompts pass@1 also scored"]
                            + [prop_cell(cats[k].proportion, latex) for k in ROW_KEYS])
    return rows


# --------------------------------------------------------------------------- session, coverage, outcomes, tokens


def session_stage_table(art: Artifacts) -> list[list[str]]:
    """Stage | Status | Seconds | Detail, from stage_status.json (or session.json). [] if neither."""
    if not art.stage_records:
        return []
    rows = [["Stage", "Status", "Seconds", "Error / detail"]]
    for rec in art.stage_records:
        status = str(rec.get("status") or "unknown")
        seconds = rec.get("seconds")
        detail = [str(rec["error"]).replace("\n", " ").replace("|", "\\|")[:300]] if rec.get("error") else []
        if rec.get("prompts_evaluated") is not None or rec.get("held_out_total") is not None:
            detail.append(f"prompts scored {rec.get('prompts_evaluated', '?')} of {rec.get('held_out_total', '?')}")
        rows.append([str(rec.get("stage", "?")), status if status == "ok" else status.upper(),
                     f"{seconds:g}" if isinstance(seconds, (int, float)) else "–", "; ".join(detail) or "–"])
    return rows


def coverage_table(art: Artifacts) -> list[list[str]]:
    if not art.runs:
        return []
    rows = [["Run", "Prompts scored", "Prompts planned", "Held-out prompts", "Full held-out set", "Status"]]
    for run in sorted_runs(art):
        if run.is_search:
            full = f"subset by design (search_limit {run.search_limit})" if run.search_limit else "subset (search)"
        else:
            full = {True: "yes", False: "NO", None: "unknown"}[run.full_held_out_set]
        status = ("PARTIAL: killed, incremental log only" if run.incremental
                  else "PARTIAL (report says so)" if run.partial else "complete")
        rows.append([run.label, str(n_scored(run)),
                     str(run.prompts_planned) if run.prompts_planned is not None else "not recorded",
                     str(run.held_out_total) if run.held_out_total is not None else "unknown", full, status])
    return rows


def coverage_warnings(art: Artifacts) -> list[str]:
    """Loud warnings: partial runs, evals that skipped held-out prompts, unfinished dataset builds."""
    out = []
    for run in sorted_runs(art):
        n = n_scored(run)
        if run.incremental:
            out.append(f"{run.label}: the run was killed before writing its report; only the incremental log "
                       f"(`{run.source}`, {n} prompts) exists, so these numbers are PARTIAL.")
        elif run.partial:
            out.append(f"{run.label}: the eval report is marked partial ({n} prompts scored).")
        if not run.is_search and run.full_held_out_set is False:
            total = f" of {run.held_out_total}" if run.held_out_total is not None else ""
            out.append(f"{run.label}: scored {n}{total} held-out prompts, NOT the full held-out set.")
        if run.prompts_planned is not None and not run.from_summary and not run.derived_from \
                and len(run.rows) < run.prompts_planned and not run.incremental:
            out.append(f"{run.label}: scored {len(run.rows)} of the {run.prompts_planned} prompts planned.")
    penalties = repetition_penalties(art)
    if len(set(penalties.values())) > 1:
        out.append("Runs compared in this report used DIFFERENT repetition_penalty values ("
                   + "; ".join(f"{label}: {value}" for label, value in penalties.items())
                   + "); differences between them may come from decoding, not the model.")
    for t in art.training_runs:
        if t.partial:
            stop = f" at step {t.stopped_at_step}" if t.stopped_at_step is not None else ""
            planned = f" of {t.steps_planned}" if t.steps_planned is not None else ""
            out.append(f"Training run {t.name} is PARTIAL: it stopped early{stop}{planned} (time limit).")
    for g in search_groups(art):
        if g.incomplete_pass1:
            out.append(f"{g.label}: pass@1 scored only {len(g.shared_ids)} of the {len(g.ids)} search prompts; "
                       "the like-for-like comparison is the rows marked 'prompts pass@1 also scored'.")
    for report, label in ((art.dataset, "GPU dataset build"), (art.interpret, "Interpreter dataset check")):
        if report and report.partial:
            missing = f": {report.not_run} candidates were not verified" if report.not_run is not None else ""
            out.append(f"{label} is PARTIAL{missing}.")
    for log in art.progress:
        report = art.interpret if log.mode == "interpret" else art.dataset
        if report is None:
            out.append(f"Dataset build ({log.mode}) did not finish: `{log.source}` records {log.records} "
                       f"verifications ({log.passed} passed) but no dataset report was written.")
    return out


SETTING_COLUMNS = ("decoding", "do_sample", "temperature", "top_k", "top_p", "repetition_penalty",
                   "max_new_tokens", "seed", "round0_temperature", "repair_temperature")


def repetition_penalties(art: Artifacts) -> dict[str, object]:
    """run label -> recorded repetition_penalty, for runs (not derived ones) that record it."""
    found = {}
    for r in sorted_runs(art):
        if r.derived_from:
            continue
        for suffix, settings in (("", r.settings), (" (repair rounds)", r.repair_settings)):
            if settings.get("repetition_penalty") is not None:
                found[r.label + suffix] = settings["repetition_penalty"]
    return found


def settings_table(art: Artifacts) -> list[list[str]]:
    """Generation settings each run recorded; "not recorded" where the report has no value."""
    runs = [r for r in sorted_runs(art) if not r.derived_from]
    if not runs:
        return []
    rows = [["Run", "Prompt mode", "Few-shot source"] + list(SETTING_COLUMNS) + ["repair-round settings"]]
    for r in runs:
        values = {**r.settings}
        if "max_new_tokens" not in values and r.max_new_tokens is not None:
            values["max_new_tokens"] = r.max_new_tokens
        few_shot = r.few_shot_source or ("n/a (zero-shot)" if r.prompt_mode == "zero_shot" else "not recorded")
        repair = (", ".join(f"{k}={v}" for k, v in r.repair_settings.items() if v is not None) or "not recorded"
                  if r.repair_rounds else "n/a")
        mode = "few-shot" if r.prompt_mode.startswith("few_shot") else r.prompt_mode.replace("_", "-")
        rows.append([r.label, mode, few_shot]
                    + [str(values[k]) if values.get(k) is not None else "not recorded" for k in SETTING_COLUMNS]
                    + [repair])
    return rows


def outcomes_table(art: Artifacts) -> list[list[str]]:
    runs = [r for r in sorted_runs(art) if r.outcomes]
    if not runs:
        return []
    seen = {o for r in runs for o in r.outcomes}
    labels = [o for o in OUTCOME_ORDER if o in seen] + sorted(seen - set(OUTCOME_ORDER))
    rows = [["Run", "Total"] + labels]
    for r in runs:
        rows.append([r.label, str(sum(r.outcomes.values()))] + [str(r.outcomes.get(o, 0)) for o in labels])
    return rows


def token_limit_table(art: Artifacts, latex: bool = False) -> list[list[str]]:
    """Outputs cut off at max_new_tokens, and the share of failures they explain. [] if nothing recorded."""
    rows = [["Run", "max_new_tokens", "Outputs", "Hit token limit", "Failures", "Failures at limit (share)"]]
    for r in sorted_runs(art):
        cap = str(r.max_new_tokens) if r.max_new_tokens is not None else "not recorded"
        stats = r.token_limit()
        if stats is not None:
            hits = f"{stats.hits}" + (f" (recorded for {stats.recorded})" if stats.recorded < stats.outputs else "")
            share = prop_cell(stats.failures_explained, latex) if stats.failures else "no failures"
            rows.append([r.label, cap, str(stats.outputs), hits, str(stats.failures),
                         f"{stats.failures_at_limit}: {share}"])
        elif r.reported_hit_token_limit is not None:
            rows.append([r.label, cap, str(n_scored(r)), str(r.reported_hit_token_limit), "–",
                         "not recorded per output"])
    return rows if len(rows) > 1 else []


# --------------------------------------------------------------------------- dataset, training, rubric


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


def hours_minutes(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}:{minutes % 60:02d}"


def training_table(art: Artifacts) -> list[list[str]]:
    """(Metric, Value) rows for the newest training run, only for values that exist. [] if none."""
    t = art.training
    if t is None:
        return []
    rows = []
    split = t.split_info or {}
    cfg = split.get("config") if isinstance(split.get("config"), dict) else {}
    rows.append(["Run", t.name])
    if t.partial:
        rows.append(["Status", f"PARTIAL: stopped at step {t.stopped_at_step if t.stopped_at_step is not None else '?'}"
                               f" of {t.steps_planned if t.steps_planned is not None else '?'} (time limit)"])
    elif "partial" in split:
        rows.append(["Status", "complete"])
    if cfg.get("model_name"):
        rows.append(["Base model", str(cfg["model_name"])])
    if split.get("split"):
        rows.append(["Split", f"{split['split']}"])
    if all(k in split for k in ("train", "val", "test")):
        rows.append(["Train / val / test entries", f"{split['train']} / {split['val']} / {split['test']}"])
    if split.get("held_out_templates"):
        rows.append(["Held-out templates", ", ".join(map(str, split["held_out_templates"]))])
    if "lora_r" in cfg and "lora_alpha" in cfg:
        rows.append(["LoRA r / alpha (configured)", f"{cfg['lora_r']} / {cfg['lora_alpha']}"])
    if isinstance(cfg.get("learning_rate"), (int, float)):
        rows.append(["Learning rate (configured)", f"{cfg['learning_rate']:g}"])
    if isinstance(cfg.get("per_device_train_batch_size"), int) and isinstance(
            cfg.get("gradient_accumulation_steps"), int):
        rows.append(["Effective batch (configured)",
                     str(cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"])])
    # Epochs/steps actually run come from the trainer, not the config.
    if t.epochs is not None:
        rows.append(["Epochs completed", f"{t.epochs:g}"])
    elif "num_epochs" in cfg:
        rows.append(["Epochs (configured)", str(cfg["num_epochs"])])
    if t.steps is not None:
        rows.append(["Optimizer steps", str(t.steps) + (f" of {t.steps_planned}" if t.steps_planned else "")])
    if t.final_step_loss is not None:
        rows.append(["Final logged train loss", f"{t.final_step_loss:.4f}"])
    if t.mean_train_loss is not None:
        rows.append(["Mean train loss over the run (Trainer summary)", f"{t.mean_train_loss:.4f}"])
    if t.best_eval_loss is not None:
        rows.append(["Best validation loss", f"{t.best_eval_loss:.4f}"])
    if t.final_eval_loss is not None:
        rows.append(["Final validation loss", f"{t.final_eval_loss:.4f}"])
    seconds, origin = t.wall_clock
    if seconds is not None:
        rows.append(["Training wall-clock", f"{hours_minutes(seconds)} (h:mm, {origin})"])
    rows.append(["Peak GPU memory", f"{t.peak_memory_gb:.1f} GB" if t.peak_memory_gb is not None
                 else "n/r (split_info.json has no peak_memory_gb)"])
    return [["Metric", "Value"]] + rows


def rubric_table(art: Artifacts, latex: bool = False) -> list[list[str]]:
    """One row per explanation review that has at least one scored item. [] if none."""
    scored = [r for r in art.reviews if r.scores]
    if not scored:
        return []
    rows = [["Model", "Scored", "Mean score (0-2)", "Scores 2 / 1 / 0", "Share scoring 2"]]
    for r in scored:
        rows.append([r.label, f"{len(r.scores)}/{r.n_items}", f"{r.mean_score:.2f}",
                     " / ".join(str(r.scores.count(s)) for s in (2, 1, 0)), prop_cell(r.top_score, latex)])
    return rows


TABLE_NOTES = (
    f"Percentages are % correct with k/n and a Wilson 95% interval in brackets. "
    f"{DAGGER} = fewer than {SMALL_N} samples: treat as indicative only. "
    "Speedup = PyTorch reference time / kernel time. *Mean* is the arithmetic mean and *geomean* the geometric "
    "mean, both over correct kernels with a finite speedup only (n in brackets), so base and fine-tuned can "
    "average over different kernels. *fast_1* = attempts that are correct AND faster than the reference, "
    f"out of all attempts, with a Wilson 95% interval. {SUBSET_MARK} = scored on the search subset only "
    "(search_limit prompts): compare it with pass@1 on the same prompts in the search table, not with the "
    "% correct column. Zero-shot and few-shot base runs are separate columns and are never pooled. "
    "\"not run\" = no artifact for that stage."
)


# --------------------------------------------------------------------------- artifact status & limitations


def stage_status(art: Artifacts) -> list[tuple[str, str]]:
    """(artifact, what we found) for everything the report can use, in pipeline order."""
    session_stages = {str(r.get("stage")): r for r in art.stage_records}

    def why_missing(session_key: Optional[str]) -> str:
        if session_key and session_key in session_stages:
            rec = session_stages[session_key]
            detail = f"{rec.get('status', '?')}" + (f": {rec['error']}" if rec.get("error") else "")
            return f"absent (session: {detail})"
        return "absent"

    out = []
    if art.dataset:
        out.append(("Dataset build (GPU)",
                    f"present{' (PARTIAL)' if art.dataset.partial else ''}: {art.dataset.source}"))
    elif art.interpret:
        out.append(("Dataset build (GPU)", f"absent; interpreter-only check present: {art.interpret.source}"))
    else:
        out.append(("Dataset build (GPU)", why_missing("dataset")))
    for model, session_key in (("base", "baseline"), ("ft", "eval")):
        if model == "ft":
            out.append(("Fine-tune (training/run_*)", f"present: {art.training.name}" if art.training
                        else "present (split_info.json)" if art.split_info else why_missing("finetune")))
        for pm in art.prompt_modes(model) or [None]:
            for nc, rr in SEARCH_MODES:
                run = art.run(model, nc, rr, pm) if pm else None
                name = f"{short_label(model, pm).replace('FT', 'Fine-tuned')}, {mode_label(nc, rr)}"
                if run:
                    note = " (PARTIAL, incremental log)" if run.incremental else " (PARTIAL)" if run.partial else ""
                    out.append((name, f"present{note}: {run.source}"))
                else:
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
    out = [f"WARNING: {w}" for w in coverage_warnings(art)]
    runs = sorted_runs(art)
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

    groups = search_groups(art)
    if groups:
        sizes = ", ".join(f"{g.label}: {len(g.ids) if g.ids is not None else 'unknown'}" for g in groups)
        out.append(f"Search runs scored only a subset of the held-out prompts (search_limit; {sizes} prompts). "
                   "They are compared only with pass@1 recomputed on those same prompts, never with pass@1 on all "
                   "prompts, which is a different population. The best-of-4 + repair column of the category "
                   f"table ({SUBSET_MARK}) is on that subset.")
    if len(art.prompt_modes("base")) > 1:
        out.append("The base model was evaluated with more than one prompt mode ("
                   + ", ".join(m.replace("_", "-") for m in art.prompt_modes("base"))
                   + "); each is reported separately and they are never averaged together.")

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

    for base_pm in art.prompt_modes("base"):
        for ft_pm in art.prompt_modes("ft"):
            base, ft = art.run("base", 1, 0, base_pm), art.run("ft", 1, 0, ft_pm)
            if base and ft and not (base.from_summary or ft.from_summary) and set(base.entry_ids) != set(ft.entry_ids):
                out.append(f"{base.label} and {ft.label} were evaluated on different entries "
                           f"({len(set(base.entry_ids) & set(ft.entry_ids))} shared of {len(set(base.entry_ids))} / "
                           f"{len(set(ft.entry_ids))}); compare them with care.")
    if any(r.from_summary for r in runs):
        out.append("Some reports had no per-entry results; their numbers come from the report's own per-category "
                   "summary, and geomean, fast_1, same-prompt comparisons and the speedup figure omit them.")
    if any(not r.interpreted for r in runs):
        out.append("Speedup is reported three ways. The arithmetic mean (matches evaluation/common.py) and the "
                   "geometric mean are over correct kernels only, so they say nothing about how often the model "
                   "fails; fast_1 (correct and faster than the PyTorch reference, over all attempts) does.")
    inf = sum(c.infinite_speedups for r in runs for c in r.categories.values())
    if inf:
        out.append(f"{inf} correct kernel(s) were timed at 0 s (speedup = inf). They are left out of the mean, "
                   "the geometric mean and fast_1, and counted separately in summary.json.")
    for r in runs:
        stats = r.token_limit()
        if stats and stats.hits:
            cap = f"max_new_tokens={r.max_new_tokens}" if r.max_new_tokens is not None else "the token limit"
            out.append(f"{r.label}: {stats.hits} of {stats.outputs} outputs were cut off at {cap}; they account for "
                       f"{stats.failures_at_limit} of {stats.failures} failures, which say more about the "
                       "generation budget than about the model's kernels.")
    if art.prompt_modes("ft") and not (art.training and (art.training.trainer_state or art.training.train_log)):
        out.append("Training loss is not in the artifacts (no trainer_state.json or train_log.json), "
                   "so it is not reported.")
    if art.training and art.training.peak_memory_gb is None:
        out.append("Peak GPU memory during training is not in the artifacts (no peak_memory_gb in "
                   "split_info.json), so it is reported as n/r.")

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
                   + ", ".join(f"{len(r.scores)} {r.label} samples" for r in scored) + "; not inter-rated.")
    if any(r.model == "unknown" for r in art.reviews):
        out.append("Some review items could not be attributed to a model (a blind review without its "
                   "<review>.key.json answer key); they are reported as 'Unknown model'.")

    out.append("Scope (fixed): a small model and dataset compared with 8B–32B RL systems; 5 narrow categories; "
               "no RL (search is inference-time only); demo-grade sandbox.")
    return out


# --------------------------------------------------------------------------- documents


def _review_lines(art: Artifacts) -> list[str]:
    lines = []
    for r in art.reviews:
        if not r.scores:
            lines.append(f"- {r.label} (`{r.source}`): {r.n_items} items, none scored yet.")
            continue
        dist = {s: r.scores.count(s) for s in (0, 1, 2)}
        lines.append(f"- {r.label} (`{r.source}`): {len(r.scores)}/{r.n_items} scored; mean {r.mean_score:.2f}/2; "
                     f"scores 2/1/0 = {dist[2]}/{dist[1]}/{dist[0]}; share scoring 2: {prop_cell(r.top_score)}")
    return lines or ["No explanation review found."]


def _provenance_lines(art: Artifacts) -> list[str]:
    lines = []
    env = (art.session or {}).get("env") or {}
    if env:
        lines.append("- Session env: " + ", ".join(f"{k}={env[k]}" for k in
                                                   ("gpu", "cuda", "torch", "triton", "git_commit") if k in env))
    zip_files = (art.session or {}).get("zip_files")
    if isinstance(zip_files, list):
        lines.append(f"- session.json lists {len(zip_files)} files in the zip.")
    for run in sorted_runs(art):
        extras = [f"{k} {v}" for k, v in (("data source", run.data_source), ("backend", run.backend),
                                          ("max_new_tokens", run.max_new_tokens),
                                          ("compared with", run.baseline_report)) if v is not None]
        lines.append(f"- {run.label}: `{run.source}` (model {run.model_name or '?'}, "
                     f"adapter {run.adapter_path or 'none'}, device {run.device or '?'}, "
                     f"split {run.split or 'not recorded'}, generated {run.generated_at or '?'}"
                     + "".join(f", {e}" for e in extras) + ")")
    for log in art.progress:
        lines.append(f"- Dataset progress log `{log.source}`: {log.records} records, {log.passed} passed.")
    for path in art.superseded:
        lines.append(f"- Superseded (a newer or final report covers it), not used: `{path}`")
    for path, why in art.excluded:
        lines.append(f"- Excluded `{path}`: {why}")
    for p in art.problems:
        lines.append(f"- Could not read {p}")
    return lines or ["- No artifacts found."]


def _section(title: str, rows: list[list[str]], empty: str) -> list[str]:
    return [f"## {title}", "", markdown_table(rows) if rows else empty, ""]


def results_markdown(art: Artifacts, figures: list[tuple[str, str]], skipped_figures: list[tuple[str, str]],
                     latex_notes: list[str] = ()) -> str:
    warnings = coverage_warnings(art)
    parts = [
        "# KernelForge results",
        "",
        f"Generated by `python -m evaluation.report` from `{art.source}`. Every number below is computed from "
        "the artifacts listed under *Sources*; anything without an artifact is marked \"not run\".",
        "",
    ]
    if warnings:
        parts += ["> **WARNING: these results are incomplete.**", ">"] + [f"> - {w}" for w in warnings] + [""]
    parts += _section("Session stages", session_stage_table(art),
                      "No stage_status.json or session stage records found.")
    parts += _section("Eval coverage", coverage_table(art), "No evaluation runs found.")
    parts += _section("Run settings", settings_table(art), "No evaluation runs found.")
    parts += _section("Artifacts found", [["Artifact", "Status"]] + [[s, v] for s, v in stage_status(art)], "")
    parts += ["## Category results (Phase 5)", "", markdown_table(phase5_table(art)), "", TABLE_NOTES, ""]
    parts += _section("Search: pass@1 vs best-of-N vs best-of-N + repair, on the same prompts", search_table(art),
                      "No search runs found (only pass@1, or nothing).")
    parts += _section("Outcomes", outcomes_table(art), "No outcome labels in the eval reports.")
    parts += _section("Token-limit truncations", token_limit_table(art), "No eval report records hit_token_limit.")
    parts += ["## Dataset build", ""]
    report = art.dataset or art.interpret
    if report:
        if report.interpreted:
            parts += ["*Interpreter-only report: correctness on CPU, no timings.*", ""]
        if report.smoke:
            parts += ["*Smoke run: one small instance per template, not the full shape grid.*", ""]
        if report.partial:
            parts += ["**PARTIAL build**" + (f": {report.not_run} candidates not verified."
                                             if report.not_run is not None else "."), ""]
        stats = {k: v for k, v in report.run_stats.items() if not isinstance(v, (list, dict))}
        if stats:
            parts += ["Run stats: " + ", ".join(f"{k}={v}" for k, v in stats.items()), ""]
        parts += [markdown_table(dataset_table(art, "category")), "",
                  markdown_table(dataset_table(art, "template")), ""]
    else:
        parts += ["No dataset report found.", ""]
    parts += _section("Training", training_table(art), "No training artifacts found.")
    parts += ["## Explanation review", "", *_review_lines(art), "", "## Figures", ""]
    parts += [f"![{caption}]({path})\n\n*{caption}*\n" for path, caption in figures]
    parts += [f"- {name}: not drawn ({why})" for name, why in skipped_figures]
    if latex_notes:
        parts += ["", "## LaTeX fragments (docs/paper/RESULTS_INTERFACE.md)", "", *[f"- {n}" for n in latex_notes]]
    parts += ["", "## Limitations", "", *[f"- {line}" for line in limitations(art)], "",
              "## Sources", "", *_provenance_lines(art), ""]
    return "\n".join(parts)


def readme_snippet(art: Artifacts) -> str:
    """A short 'Results' section to paste into README.md."""
    warnings = coverage_warnings(art)
    warn = ["> **Warning: incomplete results.** " + " ".join(warnings), ""] if warnings else []
    if not art.runs:
        lines = ["## Results", "", *warn, "No evaluation results yet: the baseline and fine-tuned evaluations "
                 "have not produced artifacts."]
        if art.dataset:
            total = sum(c.total for c in art.dataset.per_category.values())
            passed = sum(c.passed for c in art.dataset.per_category.values())
            lines += ["", f"Dataset: {passed}/{total} generated candidates passed GPU verification "
                      f"(see [docs/results/results.md](docs/results/results.md))."]
        return "\n".join(lines) + "\n"

    env = (art.session or {}).get("env") or {}
    held_out = sorted({t for r in art.runs.values() for t in r.held_out_templates})
    intro = f"Held-out evaluation on {env.get('gpu', 'a GPU')}"
    if held_out:
        intro += f" (template split; held-out ops: {', '.join(held_out)})"
    lines = ["## Results", "", *warn,
             intro + ". Full report, figures and sources: [docs/results/results.md](docs/results/results.md).", "",
             markdown_table(phase5_table(art)), "", TABLE_NOTES, ""]
    rest = [line for line in limitations(art) if not line.startswith("WARNING")]
    lines += ["**Limitations.** " + " ".join(rest[:3]), ""]
    return "\n".join(lines)


def summary_json(art: Artifacts) -> dict:
    """Machine-readable version of every computed number (for checking by hand or other tools)."""
    def prop(p: Optional[Proportion]) -> Optional[dict]:
        if p is None or p.n == 0:
            return None
        return {"k": p.successes, "n": p.n, "pct": p.fraction * 100,
                "wilson95_pct": [p.interval[0] * 100, p.interval[1] * 100], "small_n": p.small}

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

    def token(r: EvalRun) -> Optional[dict]:
        stats = r.token_limit()
        if stats is None:
            return None if r.reported_hit_token_limit is None else {"hits": r.reported_hit_token_limit,
                                                                    "per_output": False}
        return {"outputs": stats.outputs, "recorded": stats.recorded, "hits": stats.hits,
                "failures": stats.failures, "failures_at_limit": stats.failures_at_limit,
                "share_of_failures": prop(stats.failures_explained), "per_output": True}

    def search(g: SearchGroup) -> dict:
        subset = g.pass1_same_prompts()
        return {"model": g.model, "prompt_mode": g.prompt_mode, "prompts": None if g.ids is None else len(g.ids),
                "search_limit": g.search_limit,
                "pass1_same_prompts": None if subset is None else prop(merge(subset.values()).proportion),
                "pass1_covered": g.pass1_covered,
                "search_runs": {r.mode_label: prop(r.overall().proportion) for r in g.runs},
                "search_runs_on_shared_prompts": None if not g.incomplete_pass1 else
                {r.mode_label: prop(merge(r.subset(g.shared_ids).values()).proportion) for r in g.runs}}

    def training(t) -> dict:
        seconds, origin = t.wall_clock
        return {"run": t.name, "partial": t.partial, "stopped_at_step": t.stopped_at_step,
                "steps_planned": t.steps_planned, "steps": t.steps, "epochs": t.epochs,
                "final_step_loss": t.final_step_loss, "mean_train_loss": t.mean_train_loss,
                "best_eval_loss": t.best_eval_loss, "final_eval_loss": t.final_eval_loss,
                "wall_clock_s": seconds, "wall_clock_source": origin or None,
                "peak_memory_gb": t.peak_memory_gb, "split_info": t.split_info}

    return {
        "source": art.source,
        "warnings": coverage_warnings(art),
        "session_stages": art.stage_records,
        "artifacts": dict(stage_status(art)),
        "runs": {f"{r.model}/{r.prompt_mode}/{r.mode_label}": {
            "source": r.source, "model_name": r.model_name, "prompt_mode": r.prompt_mode,
            "interpreted": r.interpreted, "split": r.split, "held_out_templates": r.held_out_templates,
            "partial": r.partial, "incremental_log_only": r.incremental,
            "prompts_scored": n_scored(r), "held_out_total": r.held_out_total,
            "full_held_out_set": r.full_held_out_set, "search_limit": r.search_limit,
            "search_prompts_scored": r.prompts_scored, "max_new_tokens": r.max_new_tokens,
            "data_source": r.data_source, "backend": r.backend, "outcomes": r.outcomes,
            "settings": r.settings, "repair_settings": r.repair_settings, "few_shot_source": r.few_shot_source, "derived_from": r.derived_from,
            "prompts_planned": r.prompts_planned,
            "token_limit": token(r),
            "per_category": {k: cat(r, v) for k, v in r.categories.items()},
            "all": cat(r, r.overall())} for r in sorted_runs(art)},
        "search_comparisons": [search(g) for g in search_groups(art)],
        "training_runs": [training(t) for t in art.training_runs],
        "explanation_reviews": [{"source": r.source, "model": r.model, "prompt_mode": r.prompt_mode,
                                 "items": r.n_items, "scored": len(r.scores),
                                 "score_2": r.scores.count(2), "score_1": r.scores.count(1),
                                 "score_0": r.scores.count(0), "mean_score": r.mean_score} for r in art.reviews],
        "limitations": limitations(art),
    }
