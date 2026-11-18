"""Find and parse the artifacts a GPU session writes into kernelforge_outputs(.zip).

Paths are matched by suffix, so the zip may or may not have a top-level folder.
Formats follow what agent/a (scripts/gpu_session.py, data/build_dataset.py) and
agent/b (evaluation/common.py, training/finetune.py, evaluation/explanation_review.py)
write; every key is optional and a missing key never becomes a number.

    stage_status.json                     [{stage, status ok/failed/skipped, seconds, error,
                                            eval stages: eval_report, prompts_evaluated,
                                            held_out_total, full_held_out_set}]
    session.json                          env, the same stage records as a dict, zip_files
    data/raw/generation_report.json       GPU dataset build ("partial", "run_stats")
    data/raw/interpret_report.json        interpreter-only dataset check
    data/raw/progress_<mode>.jsonl        one line per finished dataset verification
    evaluation/results/<run>_<ts>.json    eval report: pass@1 rows in "results", plus an
                                          embedded "search" block with its own "results"
    evaluation/results/baseline.jsonl, baseline_few_shot_<source>.jsonl, finetuned.jsonl
                                          incremental log: {"type": "meta", ...} then
                                          {"type": "row", "phase": "pass@1"|"search", "id", ...}
    training/run_*/split_info.json        split, config, partial, stopped_at_step, steps_planned
    training/run_*/train_log.json         Trainer log_history (list)
    training/run_*/trainer_state.json, checkpoint-*/trainer_state.json
    <review>.json + <review>.key.json     evaluation/explanation_review.py (key only when blind)

Peak GPU memory is read from an optional split_info.json "peak_memory_gb"; without it, "n/r".
Anything named dryrun_* is skipped.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional

from evaluation.report.stats import Proportion, finite, geomean, mean

# Same order and names as kernelforge.dataset.CATEGORIES (a test checks they
# match). Copied rather than imported so the report never needs torch.
CATEGORIES = ("quantized_matmul", "attention", "kv_cache", "norm", "rope")
CATEGORY_LABELS = {
    "quantized_matmul": "Quantized matmul",
    "attention": "Attention",
    "kv_cache": "KV-cache",
    "norm": "Norm",
    "rope": "RoPE",
}
PROMPT_LABELS = {"zero_shot": "zero-shot", "few_shot": "few-shot", "unspecified": "",
                 "few_shot_other_category": "few-shot (other_category)",
                 "few_shot_same_category": "few-shot (same_category)"}
MODEL_LABELS = {"base": "Base", "ft": "Fine-tuned", "unknown": "Unknown model"}

# evaluation/common.py writes its per-prompt log to evaluation/results/<run name>.jsonl:
# baseline.jsonl, baseline_few_shot_<source>.jsonl, finetuned.jsonl. A rerun moves an old log
# aside as <run name>.<timestamp>.jsonl, which this pattern does not match.
INCREMENTAL_LOG = re.compile(r"^(baseline|finetuned|baseline_few_shot(_(other|same)_category)?)\.jsonl$")

# Generation settings shown in the run-settings table. The explicit "generation_settings"
# dict wins over top-level keys of the same name.
SETTING_KEYS = ("decoding", "do_sample", "temperature", "top_k", "top_p", "repetition_penalty",
                "max_new_tokens", "seed", "round0_temperature", "repair_temperature", "few_shot_source")

MAX_FILE_BYTES = 50 * 1024 * 1024  # skip anything larger; no report is close to this


def _int(value) -> Optional[int]:
    """int(value), or None for anything missing or non-numeric (never a made-up 0)."""
    try:
        return None if value is None or isinstance(value, bool) else int(value)
    except (TypeError, ValueError):
        return None


def _float(value) -> Optional[float]:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def _category_of_id(entry_id: str) -> str:
    """Category from a dataset id like "attention__causal_prefill__b2_s512"."""
    return str(entry_id).split("__")[0]


# --------------------------------------------------------------------------- evaluation runs


@dataclass
class CategoryResult:
    """Counts for one category (or "all") of one eval run, or of a subset of its prompts."""

    n: int = 0
    correct: int = 0
    speedups: list[float] = field(default_factory=list)  # correct kernels with a finite speedup
    infinite_speedups: int = 0  # correct kernels the harness timed at 0 s (speedup = inf)
    # Only set when the report had no per-entry "results" list and we fell back
    # to its pre-aggregated per_category numbers.
    summary_avg_speedup: Optional[float] = None
    summary_passed: Optional[int] = None

    @property
    def proportion(self) -> Proportion:
        return Proportion(self.correct, self.n)

    @property
    def avg_speedup(self) -> Optional[float]:
        return mean(self.speedups) if self.summary_passed is None else self.summary_avg_speedup

    @property
    def speedup_n(self) -> int:
        return len(self.speedups) if self.summary_passed is None else self.summary_passed

    @property
    def geomean_speedup(self) -> Optional[float]:
        """None for summary-only reports: their per-kernel speedups are unknown."""
        return geomean(self.speedups) if self.summary_passed is None else None

    @property
    def fast_1(self) -> Optional[Proportion]:
        """fast_1 (KernelBench): attempts that are correct AND faster than the reference
        (finite speedup > 1), out of ALL attempts n. Infinite speedups are not counted
        as fast; they are reported separately. None for summary-only reports."""
        if self.summary_passed is not None:
            return None
        return Proportion(sum(1 for s in self.speedups if s > 1.0), self.n)


@dataclass
class EntryRow:
    """One scored prompt of an eval run."""

    id: str
    category: str
    correct: bool
    speedup: Optional[float] = None  # finite speedup of a correct kernel that ran cleanly
    infinite: bool = False  # correct, ran cleanly, timed at 0 s
    outcome: Optional[str] = None  # evaluation/common.py outcome label, if the report has it
    hit_token_limit: Optional[bool] = None  # None = not recorded
    example_id: Optional[str] = None  # few-shot example shown with this prompt
    selected_round: Optional[int] = None  # search rows: round of the selected candidate (0 = no repair)


def aggregate(rows: Iterable[EntryRow]) -> dict[str, CategoryResult]:
    cats: dict[str, CategoryResult] = {}
    for r in rows:
        c = cats.setdefault(r.category, CategoryResult())
        c.n += 1
        c.correct += r.correct
        if r.speedup is not None:
            c.speedups.append(r.speedup)
        c.infinite_speedups += r.infinite
    return cats


def merge(results: Iterable[CategoryResult]) -> CategoryResult:
    """Pool categories into one "all" result."""
    results = list(results)
    total = CategoryResult()
    for c in results:
        total.n += c.n
        total.correct += c.correct
        total.speedups.extend(c.speedups)
        total.infinite_speedups += c.infinite_speedups
    if any(c.summary_passed is not None for c in results):
        # Pre-aggregated reports only give per-category means; a pooled mean
        # weighted by the number of passing kernels is the honest overall.
        weighted = [(c.summary_avg_speedup, c.summary_passed or 0) for c in results
                    if c.summary_avg_speedup is not None]
        passed = sum(w for _, w in weighted)
        total.summary_passed = sum(c.summary_passed or 0 for c in results)
        total.summary_avg_speedup = (sum(a * w for a, w in weighted) / passed) if passed else None
    return total


@dataclass
class TokenLimitStats:
    outputs: int  # rows in the run
    recorded: int  # rows that say whether they hit the limit
    hits: int  # rows that hit max_new_tokens
    failures: int  # rows that are not correct
    failures_at_limit: int  # failures that also hit the limit

    @property
    def failures_explained(self) -> Proportion:
        """Share of failures whose output was cut off by the token limit."""
        return Proportion(self.failures_at_limit, self.failures)


@dataclass
class EvalRun:
    source: str
    model: str  # "base" or "ft"
    num_candidates: int
    repair_rounds: int
    prompt_mode: str = "unspecified"  # "zero_shot", "few_shot", or as given
    run_name: str = ""
    model_name: str = ""
    adapter_path: Optional[str] = None
    device: str = ""
    generated_at: str = ""
    split: Optional[str] = None
    held_out_templates: list[str] = field(default_factory=list)
    interpreted: bool = False  # correctness only, no timings
    from_summary: bool = False  # per-entry results were missing
    partial: bool = False  # the report says so, or only an incremental log exists
    incremental: bool = False  # built from an incremental .jsonl, no final report
    derived_from: Optional[str] = None  # best-of-N read from round 0 of a best-of-N + repair run
    search_limit: Optional[int] = None
    prompts_planned: Optional[int] = None
    prompts_scored: Optional[int] = None
    data_source: Optional[str] = None
    backend: Optional[str] = None
    max_new_tokens: Optional[int] = None
    baseline_report: Optional[str] = None
    settings: dict = field(default_factory=dict)  # generation settings recorded by the report (round 0 for search)
    repair_settings: dict = field(default_factory=dict)  # search only: settings of the repair rounds
    reported_outcomes: dict[str, int] = field(default_factory=dict)  # top-level "outcomes"
    reported_hit_token_limit: Optional[int] = None  # top-level count, when rows don't say
    # Filled in from session records / split_info when available.
    held_out_total: Optional[int] = None
    full_held_out_set: Optional[bool] = None
    rows: list[EntryRow] = field(default_factory=list)
    categories: dict[str, CategoryResult] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (self.model, self.num_candidates, self.repair_rounds, self.prompt_mode)

    @property
    def is_search(self) -> bool:
        return (self.num_candidates, self.repair_rounds) != (1, 0)

    @property
    def mode_label(self) -> str:
        label = mode_label(self.num_candidates, self.repair_rounds)
        return label + " (round 0 of the repair run)" if self.derived_from else label

    @property
    def group_label(self) -> str:
        """Model + prompt mode, e.g. "Base few-shot"."""
        return group_label(self.model, self.prompt_mode)

    @property
    def label(self) -> str:
        return f"{self.group_label} {self.mode_label}"

    @property
    def entry_ids(self) -> list[str]:
        return [r.id for r in self.rows]

    def overall(self) -> CategoryResult:
        return merge(self.categories.values())

    def subset(self, ids: set[str]) -> dict[str, CategoryResult]:
        """Per-category results on just these prompts (for same-prompt search comparisons)."""
        return aggregate(r for r in self.rows if r.id in ids)

    @property
    def outcomes(self) -> dict[str, int]:
        """Outcome counts from the rows when they carry labels, else the report's own counts."""
        labelled = [r.outcome for r in self.rows if r.outcome]
        if labelled:
            counts: dict[str, int] = {}
            for o in labelled:
                counts[o] = counts.get(o, 0) + 1
            return counts
        return dict(self.reported_outcomes)

    def token_limit(self) -> Optional[TokenLimitStats]:
        """None when no row records hit_token_limit (unknown, not zero)."""
        recorded = [r for r in self.rows if r.hit_token_limit is not None]
        if not recorded:
            return None
        return TokenLimitStats(
            outputs=len(self.rows), recorded=len(recorded),
            hits=sum(1 for r in recorded if r.hit_token_limit),
            failures=sum(1 for r in self.rows if not r.correct),
            failures_at_limit=sum(1 for r in recorded if r.hit_token_limit and not r.correct))

    @property
    def few_shot_source(self) -> Optional[str]:
        """"same_category" / "other_category" as recorded, else derived from each row's
        example_id (its category prefix vs the row's category). None if neither exists."""
        recorded = self.settings.get("few_shot_source")
        if isinstance(recorded, str) and recorded:
            return recorded  # the explicit key wins; example_id is only a fallback
        shown = [r for r in self.rows if r.example_id]
        if not shown:
            return None
        same = sum(1 for r in shown if _category_of_id(r.example_id) == r.category)
        if same == len(shown):
            return "same_category (from example_id)"
        if same == 0:
            return "other_category (from example_id)"
        return f"mixed: {same} same_category, {len(shown) - same} other_category (from example_id)"


def mode_label(num_candidates: int, repair_rounds: int) -> str:
    if num_candidates <= 1 and repair_rounds == 0:
        return "pass@1"
    label = f"best-of-{num_candidates}" if num_candidates > 1 else "single sample"
    if repair_rounds == 1:
        label += " + repair"
    elif repair_rounds > 1:
        label += f" + {repair_rounds} repairs"
    return label


def group_label(model: str, prompt_mode: str) -> str:
    prompt = PROMPT_LABELS.get(prompt_mode, prompt_mode)
    return f"{MODEL_LABELS.get(model, model)} {prompt}".strip()


def _settings(block: dict, nested: Optional[dict] = None) -> dict:
    """Generation settings recorded at this level of a report: top-level keys, overridden by the
    explicit settings dict (block["generation_settings"] unless `nested` is given)."""
    found = {k: block[k] for k in SETTING_KEYS if k in block and not isinstance(block[k], dict)}
    nested = block.get("generation_settings") if nested is None else nested
    if isinstance(nested, dict):
        found.update({k: v for k, v in nested.items() if k in SETTING_KEYS and not isinstance(v, dict)})
    return found


def _prompt_mode(meta: dict) -> str:
    """The run's prompt variant. A few-shot run that records its source is keyed by it, so
    other_category and same_category runs are separate runs, never one superseding the other."""
    mode = str(meta.get("prompt_mode") or "unspecified")
    source = meta.get("few_shot_source")
    if mode == "few_shot" and isinstance(source, str) and source:
        return f"few_shot_{source}"
    return mode


def _row(r: dict) -> EntryRow:
    v = r.get("verification") if isinstance(r.get("verification"), dict) else {}
    # Same definition as evaluation/common.py::aggregate_rows:
    # "correct" = the harness compared outputs and they matched;
    # speedups only for kernels that also ran cleanly (status ok).
    correct = v.get("correct") is True or (not v and r.get("outcome") == "correct")
    speedup, infinite = None, False
    if correct and v.get("status") == "ok" and v.get("speedup") is not None:
        value = _float(v["speedup"])
        if value is not None and finite([value]):
            speedup = value
        elif value is not None:
            infinite = True
    hit = r.get("hit_token_limit")
    return EntryRow(id=str(r.get("id")), category=str(r.get("category") or "unknown"), correct=correct,
                    speedup=speedup, infinite=infinite, outcome=r.get("outcome"),
                    hit_token_limit=hit if isinstance(hit, bool) else None,
                    example_id=r.get("example_id") if isinstance(r.get("example_id"), str) else None,
                    selected_round=_int(r.get("selected_round")))


def _new_run(meta: dict, source: str, num_candidates: int, repair_rounds: int) -> EvalRun:
    run_name = str(meta.get("run_name") or Path(source).stem)
    is_ft = run_name.startswith("finetuned") or bool(meta.get("adapter_path"))
    hit = meta.get("hit_token_limit")
    outcomes = meta.get("outcomes")
    return EvalRun(
        source=source, model="ft" if is_ft else "base", num_candidates=num_candidates,
        repair_rounds=repair_rounds, prompt_mode=_prompt_mode(meta),
        run_name=run_name, model_name=str(meta.get("model_name") or ""), adapter_path=meta.get("adapter_path"),
        device=str(meta.get("device") or ""), generated_at=str(meta.get("generated_at") or ""),
        split=meta.get("split"), held_out_templates=list(meta.get("held_out_templates") or []),
        partial=bool(meta.get("partial")), interpreted=bool(meta.get("interpret")),
        prompts_planned=_int(meta.get("prompts_planned")), prompts_scored=_int(meta.get("prompts_scored")),
        data_source=meta.get("data_source"), backend=meta.get("backend"),
        max_new_tokens=_int(meta.get("max_new_tokens")), baseline_report=meta.get("baseline_report"),
        # Search-only settings (seed, sampling temperatures) don't apply to greedy pass@1.
        settings={k: v for k, v in _settings(meta).items()
                  if (num_candidates, repair_rounds) != (1, 0)
                  or k not in ("seed", "round0_temperature", "repair_temperature")},
        reported_outcomes={k: _int(v) for k, v in outcomes.items() if _int(v) is not None}
        if isinstance(outcomes, dict) else {},
        reported_hit_token_limit=_int(hit),
    )


def _finish(run: EvalRun, rows: list) -> EvalRun:
    run.rows = [_row(r) for r in rows if isinstance(r, dict)]
    run.interpreted = run.interpreted or any(
        isinstance(r, dict) and isinstance(r.get("verification"), dict) and r["verification"].get("interpreted")
        for r in rows)
    run.categories = aggregate(run.rows)
    if run.interpreted:
        # Interpreter timings are meaningless; never average them.
        for c in run.categories.values():
            c.speedups, c.summary_avg_speedup = [], None
    return run


def _excluded_reason(meta: dict, source: str) -> Optional[str]:
    if Path(source).name.startswith("dryrun_") or str(meta.get("run_name", "")).startswith("dryrun"):
        return "dry run (dryrun_*): not model output"
    if meta.get("dry_run"):
        return "dry run: scored the dataset's own kernels, not model output"
    if meta.get("skip_verification"):
        return "parse-only run (--skip-verification): no correctness measured"
    return None


def _legacy_search_settings(report: dict) -> tuple[int, int]:
    """(num_candidates, repair_rounds) of a report that IS a search run (the old separate-report
    format: a "search" dict without its own results, or a run_name like baseline_best4_repair1)."""
    search = report.get("search") if isinstance(report.get("search"), dict) else {}
    nc = _int(search.get("num_candidates", report.get("num_candidates")))
    rr = _int(search.get("repair_rounds", report.get("repair_rounds")))
    name = str(report.get("run_name", ""))
    if nc is None:
        m = re.search(r"best(?:_?of)?_?(\d+)", name)
        nc = int(m.group(1)) if m else 1
    if rr is None:
        m = re.search(r"repair_?(\d+)", name)
        rr = int(m.group(1)) if m else (1 if "repair" in name else 0)
    return nc, rr


def _search_run(meta: dict, block: dict, rows: list, source: str) -> EvalRun:
    """The search phase: rows are the selected candidate per prompt."""
    nc = _int(block.get("num_candidates")) or 1
    rr = _int(block.get("repair_rounds")) or 0
    run = _new_run(meta, source, nc, rr)
    run.search_limit = _int(block.get("search_limit"))
    run.prompts_planned = _int(block.get("prompts_planned"))
    run.prompts_scored = _int(block.get("prompts_scored"))
    # The search samples, so pass@1's settings don't apply: use the block's own round-0 / repair
    # settings (search.generation_settings.round0 / .repair) plus its seed and temperatures.
    gen = block.get("generation_settings") if isinstance(block.get("generation_settings"), dict) else {}
    run.settings = _settings(block, gen.get("round0") if isinstance(gen.get("round0"), dict) else {})
    run.repair_settings = _settings({}, gen.get("repair")) if isinstance(gen.get("repair"), dict) else {}
    if meta.get("few_shot_source") is not None:
        run.settings["few_shot_source"] = meta["few_shot_source"]  # the search uses the same examples
    outcomes = block.get("outcomes")
    run.reported_outcomes = ({k: _int(v) for k, v in outcomes.items() if _int(v) is not None}
                             if isinstance(outcomes, dict) else {})
    run.reported_hit_token_limit = None
    _finish(run, rows)
    if run.prompts_planned is not None and len(run.rows) < run.prompts_planned:
        run.partial = True
    return run


def round0_run(search: EvalRun) -> Optional[EvalRun]:
    """best-of-N without repair, read off a best-of-N + repair run: the search stops at
    round 0 when any candidate passes (kernelforge/search.py), so "correct with the
    selected candidate from round 0" is exactly best-of-N. Needs selected_round on every row."""
    if search.repair_rounds < 1 or search.num_candidates < 2 or not search.rows \
            or any(r.selected_round is None for r in search.rows):
        return None
    rows = [replace(r, correct=r.correct and r.selected_round == 0,
                    speedup=r.speedup if r.selected_round == 0 else None,
                    infinite=r.infinite and r.selected_round == 0) for r in search.rows]
    run = replace(search, repair_rounds=0, derived_from=search.source, rows=rows, settings=dict(search.settings),
                  repair_settings={}, reported_outcomes={})
    run.categories = aggregate(rows)
    return run


def parse_eval_report_runs(report: dict, source: str) -> tuple[list[EvalRun], Optional[str]]:
    """Every run in one eval report: pass@1, the embedded search, and best-of-N derived from
    the search's round 0. Returns ([], why) when the report is excluded."""
    why = _excluded_reason(report, source)
    if why:
        return [], why
    search = report.get("search") if isinstance(report.get("search"), dict) else None
    embedded = search is not None and isinstance(search.get("results"), list)
    runs = []
    # With an embedded search block the report's own rows are pass@1; otherwise the report
    # may itself be a search run (old separate-report format, or a run_name like *_best4_repair1).
    nc_rr = (1, 0) if embedded else _legacy_search_settings(report)
    main = _new_run(report, source, *nc_rr)
    if not embedded and search is not None:
        main.search_limit = _int(search.get("search_limit"))
        main.prompts_scored = _int(search.get("prompts_scored"))
    results = report.get("results")
    if isinstance(results, list) and results:
        runs.append(_finish(main, results))
    elif isinstance(report.get("per_category"), dict):
        main.from_summary = True
        for name, m in report["per_category"].items():
            if isinstance(m, dict):
                main.categories[name] = CategoryResult(
                    n=_int(m.get("total")) or 0, correct=_int(m.get("correct")) or 0,
                    summary_avg_speedup=_float(m.get("avg_speedup")), summary_passed=_int(m.get("passed")) or 0)
        runs.append(main)
    if embedded and search["results"]:
        s = _search_run(report, search, search["results"], f"{source} (search)")
        runs.append(s)
        derived = round0_run(s)
        if derived is not None:
            runs.append(derived)
    if not runs:
        return [], "no per-entry results and no per_category summary"
    return runs, None


def parse_eval_report(report: dict, source: str) -> tuple[Optional[EvalRun], Optional[str]]:
    """The report's main run (pass@1, or the search run of an old separate search report)."""
    runs, why = parse_eval_report_runs(report, source)
    return (runs[0], None) if runs else (None, why)


def parse_incremental_log(lines: list[dict], source: str) -> tuple[list[EvalRun], Optional[str]]:
    """evaluation/common.py ResultsLog: a {"type": "meta"} line, then {"type": "row",
    "phase": "pass@1"|"search"} rows. Lines without "id" are treated as metadata."""
    meta: dict = {}
    for line in lines:
        if line.get("type") == "meta" or "id" not in line:
            meta.update({k: v for k, v in line.items() if k != "type"})
    why = _excluded_reason(meta, source)
    if why:
        return [], why
    rows = [line for line in lines if line.get("type") == "row" or ("id" in line and line.get("type") != "meta")]
    if not rows:
        return [], "incremental log with no scored prompts"
    runs = []
    pass1 = [r for r in rows if r.get("phase", "pass@1") == "pass@1"]
    search = [r for r in rows if r.get("phase") == "search"]
    if pass1:
        run = _finish(_new_run(meta, source, 1, 0), pass1)
        runs.append(run)
    if search:
        block = {"num_candidates": meta.get("num_candidates"), "repair_rounds": meta.get("repair_rounds"),
                 "search_limit": meta.get("search_limit"), "seed": meta.get("seed"),
                 "repetition_penalty": meta.get("repetition_penalty")}
        run = _search_run(meta, block, search, f"{source} (search)")
        runs.append(run)
        derived = round0_run(run)
        if derived is not None:
            runs.append(derived)
    for run in runs:
        run.partial = run.incremental = True
    return runs, None


# --------------------------------------------------------------------------- dataset reports


@dataclass
class OutcomeCounts:
    """Outcome of verifying generated dataset candidates (one category or template)."""

    passed: int = 0
    incorrect: int = 0
    oom: int = 0
    timeout: int = 0
    other: int = 0  # harness error / crash / anything else
    median_speedup: Optional[float] = None
    reported_generated: Optional[int] = None  # the report's own "generated" count, for the sum check

    @property
    def total(self) -> int:
        return self.passed + self.incorrect + self.oom + self.timeout + self.other

    def add_reason(self, reason: str, count: int = 1) -> None:
        if reason in ("incorrect", "oom", "timeout"):
            setattr(self, reason, getattr(self, reason) + count)
        else:
            self.other += count


@dataclass
class DatasetReport:
    source: str
    device: str
    interpreted: bool
    smoke: bool
    per_category: dict[str, OutcomeCounts]
    per_template: dict[str, OutcomeCounts]
    partial: bool = False
    run_stats: dict = field(default_factory=dict)  # resumed, oom_retried, oom_recovered_on_retry, not_run
    generated_at: str = ""

    @property
    def not_run(self) -> Optional[int]:
        value = self.run_stats.get("not_run")
        return len(value) if isinstance(value, list) else _int(value)


@dataclass
class ProgressLog:
    """data/raw/progress_<mode>.jsonl: verifications finished so far (survives a killed build)."""

    source: str
    mode: str  # "cuda", "interpret", ...
    records: int
    passed: int


def _rejection_reason(status: str, passed: bool, error: Optional[str]) -> str:
    """Mirror of data/build_dataset.py::_rejection_reason, for interpreter reports
    (which store raw status + error instead of a reason)."""
    err = (error or "").lower()
    if "out of memory" in err or "outofmemoryerror" in err or "exit code -9" in err:
        return "oom"
    if status == "ok" and not passed:
        return "incorrect"
    return status or "unknown"


def _run_stats(report: dict) -> dict:
    return report["run_stats"] if isinstance(report.get("run_stats"), dict) else {}


def parse_generation_report(report: dict, source: str) -> DatasetReport:
    """data/raw/generation_report.json (GPU build)."""
    per_category = {}
    for c, m in (report.get("per_category") or {}).items():
        counts = OutcomeCounts(passed=_int(m.get("verified")) or 0, median_speedup=_float(m.get("median_speedup")),
                               reported_generated=_int(m.get("generated")))
        for reason, n in (m.get("rejection_reasons") or {}).items():
            counts.add_reason(reason, _int(n) or 0)
        per_category[c] = counts
    per_template = {}
    for name, t in (report.get("per_template") or {}).items():
        counts = OutcomeCounts(passed=_int(t.get("verified")) or 0, median_speedup=_float(t.get("median_speedup")))
        for reason, n in (t.get("rejected") or {}).items():
            counts.add_reason(reason, _int(n) or 0)
        per_template[name] = counts
    return DatasetReport(source, str(report.get("device") or ""), False, bool(report.get("smoke")),
                         per_category, per_template, bool(report.get("partial")), _run_stats(report),
                         str(report.get("generated_at") or ""))


def parse_interpret_report(report: dict, source: str) -> DatasetReport:
    """data/raw/interpret_report.json (CPU interpreter; correctness only)."""
    per_category: dict[str, OutcomeCounts] = {}
    per_template: dict[str, OutcomeCounts] = {}
    for key, rows in (report.get("templates") or {}).items():
        category, _, template = key.partition("/")
        name = f"{category}__{template}"  # same slug as kernelforge.dataset.template_of
        for counts in (per_category.setdefault(category, OutcomeCounts()),
                       per_template.setdefault(name, OutcomeCounts())):
            for row in rows:
                if row.get("passed"):
                    counts.passed += 1
                else:
                    counts.add_reason(_rejection_reason(row.get("status", ""), False, row.get("error")))
    return DatasetReport(source, "cpu (Triton interpreter)", True, bool(report.get("smoke")),
                         per_category, per_template, bool(report.get("partial")), _run_stats(report),
                         str(report.get("generated_at") or ""))


def parse_progress_log(lines: list[dict], source: str) -> ProgressLog:
    mode = Path(source).stem.replace("progress_", "", 1)
    passed = 0
    for line in lines:
        result = line.get("result") if isinstance(line.get("result"), dict) else {}
        passed += result.get("status") == "ok" and result.get("correct") is True
    return ProgressLog(source, mode, len(lines), passed)


# --------------------------------------------------------------------------- explanation review


@dataclass
class ExplanationReview:
    """Scores for one model + prompt mode from one review file."""

    source: str
    model: str  # "base", "ft" or "unknown" (blind review without its answer key)
    prompt_mode: str
    n_items: int
    scores: list[int]  # only items a human actually scored (0/1/2)
    source_report: str = ""

    @property
    def label(self) -> str:
        return group_label(self.model, self.prompt_mode)

    @property
    def mean_score(self) -> Optional[float]:
        return mean(self.scores)

    @property
    def top_score(self) -> Proportion:
        """Share of scored explanations that got a 2 (specific and correct)."""
        return Proportion(sum(1 for s in self.scores if s == 2), len(self.scores))


def _model_of_source(src: dict) -> tuple[str, str]:
    name = str(src.get("run_name") or src.get("report") or "")
    if name.startswith("finetuned") or src.get("adapter_path"):
        model = "ft"
    elif name.startswith("baseline"):
        model = "base"
    else:
        model = "unknown"
    return model, str(src.get("prompt_mode") or "unspecified")


def parse_reviews(report: dict, source: str, key: Optional[dict]) -> list[ExplanationReview]:
    """One ExplanationReview per (model, prompt mode) in a review file.

    Current format (agent/b): items from several reports, each with a "source" dict, or,
    for a blind review, a sample_id -> source answer key in <review>.key.json.
    Older format: a single top-level "source_report"."""
    items = [i for i in (report.get("items") or []) if isinstance(i, dict)]
    legacy = str(report.get("source_report") or "")
    groups: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        src = item.get("source") if isinstance(item.get("source"), dict) else (key or {}).get(item.get("sample_id"))
        if isinstance(src, dict):
            group = _model_of_source(src)
        elif legacy:
            name = Path(legacy).name
            group = ("ft" if name.startswith("finetuned") else "base" if name.startswith("baseline") else "unknown",
                     "unspecified")
        else:
            group = ("unknown", "unspecified")
        groups.setdefault(group, []).append(item)
    return [ExplanationReview(source, model, pm, len(group_items),
                              [int(i["rubric_score"]) for i in group_items if i.get("rubric_score") in (0, 1, 2)],
                              legacy)
            for (model, pm), group_items in sorted(groups.items())]


# --------------------------------------------------------------------------- training runs


@dataclass
class TrainingRun:
    """What one training/run_*/ folder left behind (training/finetune.py). Every field is optional."""

    name: str
    split_info: Optional[dict] = None
    trainer_state: Optional[dict] = None  # HF Trainer trainer_state.json (newest)
    trainer_state_source: str = ""
    trainer_state_step: float = -1  # checkpoint number of trainer_state (inf = run-level file)
    train_log: Optional[list] = None  # train_log.json: Trainer.state.log_history
    stage_seconds: Optional[float] = None  # the session's finetune stage time (fallback wall-clock)

    def _log_history(self) -> list:
        """The longer of train_log.json and trainer_state.json's log_history (same data when both exist)."""
        state_log = (self.trainer_state or {}).get("log_history")
        logs = [log for log in (self.train_log, state_log) if isinstance(log, list)]
        return max(logs, key=len) if logs else []

    def _values(self, key: str) -> list[float]:
        return [_float(e[key]) for e in self._log_history()
                if isinstance(e, dict) and _float(e.get(key)) is not None]

    @property
    def final_step_loss(self) -> Optional[float]:
        """Last logged training loss (a per-logging-step "loss")."""
        values = self._values("loss")
        return values[-1] if values else None

    @property
    def mean_train_loss(self) -> Optional[float]:
        """The Trainer's end-of-run summary "train_loss" (mean over the whole run)."""
        values = self._values("train_loss")
        return values[-1] if values else None

    @property
    def best_eval_loss(self) -> Optional[float]:
        values = self._values("eval_loss")
        return min(values) if values else None

    @property
    def final_eval_loss(self) -> Optional[float]:
        values = self._values("eval_loss")
        return values[-1] if values else None

    @property
    def train_runtime_s(self) -> Optional[float]:
        values = self._values("train_runtime")
        return values[-1] if values else None

    @property
    def wall_clock(self) -> tuple[Optional[float], str]:
        """(seconds, where it came from). The Trainer's train_runtime first; else the session's
        finetune stage time, which also includes loading the model."""
        if self.train_runtime_s is not None:
            return self.train_runtime_s, "Trainer train_runtime"
        if self.stage_seconds is not None:
            return self.stage_seconds, "session finetune stage (includes model loading)"
        return None, ""

    @property
    def steps(self) -> Optional[int]:
        state_step = _int((self.trainer_state or {}).get("global_step"))
        if state_step is not None:
            return state_step
        values = self._values("step")
        return int(values[-1]) if values else None

    @property
    def epochs(self) -> Optional[float]:
        value = _float((self.trainer_state or {}).get("epoch"))
        if value is not None:
            return value
        values = self._values("epoch")
        return values[-1] if values else None

    @property
    def partial(self) -> bool:
        return bool((self.split_info or {}).get("partial"))

    @property
    def stopped_at_step(self) -> Optional[int]:
        return _int((self.split_info or {}).get("stopped_at_step"))

    @property
    def steps_planned(self) -> Optional[int]:
        return _int((self.split_info or {}).get("steps_planned"))

    @property
    def peak_memory_gb(self) -> Optional[float]:
        """Optional split_info.json "peak_memory_gb" (a planned training/finetune.py change); None if absent."""
        return _float((self.split_info or {}).get("peak_memory_gb"))

    @property
    def is_dry_run(self) -> bool:
        return bool((self.split_info or {}).get("dry_run"))

    @property
    def has_data(self) -> bool:
        return any(x is not None for x in (self.split_info, self.trainer_state, self.train_log))


_TRAINING_PATH = re.compile(r"(?:^|/)training/(?:[^/]+/)*?(run_[^/]+)/(.+)$")
_CHECKPOINT_STEP = re.compile(r"checkpoint-(\d+)/trainer_state\.json$")


def _add_training_file(runs: dict[str, TrainingRun], path: str, data) -> bool:
    """File it under its training run if `path` is inside training/.../run_*/; True if used."""
    m = _TRAINING_PATH.search(path)
    if not m:
        return False
    run = runs.setdefault(m.group(1), TrainingRun(m.group(1)))
    rel = m.group(2)
    if rel == "split_info.json" and isinstance(data, dict):
        run.split_info = data
    elif rel == "train_log.json" and isinstance(data, list):
        run.train_log = data
    elif rel.endswith("trainer_state.json") and isinstance(data, dict):
        # Keep the newest: the highest checkpoint-N, and a run-level trainer_state.json beats any checkpoint.
        m_step = _CHECKPOINT_STEP.search(rel)
        step = int(m_step.group(1)) if m_step else float("inf")
        if step >= run.trainer_state_step:
            run.trainer_state, run.trainer_state_source, run.trainer_state_step = data, path, step
    else:
        return False
    return True


# --------------------------------------------------------------------------- loading


@dataclass
class Artifacts:
    source: str
    runs: dict[tuple[str, int, int, str], EvalRun] = field(default_factory=dict)  # newest per key
    superseded: list[str] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    dataset: Optional[DatasetReport] = None
    interpret: Optional[DatasetReport] = None
    progress: list[ProgressLog] = field(default_factory=list)
    reviews: list[ExplanationReview] = field(default_factory=list)
    session: Optional[dict] = None
    stage_records: list[dict] = field(default_factory=list)  # stage_status.json, else session["stages"]
    split_info: Optional[dict] = None
    training_runs: list[TrainingRun] = field(default_factory=list)  # oldest first
    problems: list[str] = field(default_factory=list)  # unreadable files

    @property
    def training(self) -> Optional[TrainingRun]:
        """The newest training run (the reported one)."""
        return self.training_runs[-1] if self.training_runs else None

    def run(self, model: str, num_candidates: int = 1, repair_rounds: int = 0,
            prompt_mode: Optional[str] = None) -> Optional[EvalRun]:
        """Exact match. Without prompt_mode, only an unambiguous match is returned:
        zero-shot and few-shot runs are never silently substituted for each other."""
        if prompt_mode is not None:
            return self.runs.get((model, num_candidates, repair_rounds, prompt_mode))
        matches = [r for k, r in self.runs.items() if k[:3] == (model, num_candidates, repair_rounds)]
        return matches[0] if len(matches) == 1 else None

    def prompt_modes(self, model: str) -> list[str]:
        """Prompt modes with any run for this model, zero-shot first."""
        order = {"zero_shot": 0, "unspecified": 1, "few_shot": 2}
        return sorted({k[3] for k in self.runs if k[0] == model}, key=lambda m: (order.get(m, 3), m))


def _parse_bytes(name: str, raw: bytes, found: dict, problems: list) -> None:
    try:
        if name.endswith(".jsonl"):
            lines, bad = [], 0
            for line in raw.decode().splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    bad += 1  # a run killed mid-write can leave a truncated last line
                    continue
                if isinstance(value, dict):
                    lines.append(value)
            if bad:
                problems.append(f"{name}: skipped {bad} unreadable line(s)")
            found[name] = lines
        else:
            found[name] = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        problems.append(f"{name}: not valid JSON ({exc})")


def _wanted(name: str) -> bool:
    """All .json; .jsonl only for eval logs and dataset progress (not the dataset itself)."""
    if name.endswith(".json"):
        return True
    return name.endswith(".jsonl") and ("evaluation/results/" in name or Path(name).name.startswith("progress_"))


def _read_files(src: Path) -> tuple[dict[str, object], list[str]]:
    """{relative path: parsed JSON (or list of JSONL objects)} from a directory or a zip.

    The zip is read in memory and never extracted, so odd member paths can't
    write anywhere. Adapter weights and other files are ignored.
    """
    found: dict[str, object] = {}
    problems: list[str] = []
    if src.is_dir():
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = path.relative_to(src).as_posix()
            if not _wanted(rel):
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                problems.append(f"{rel}: skipped, larger than {MAX_FILE_BYTES} bytes")
                continue
            _parse_bytes(rel, path.read_bytes(), found, problems)
    elif zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as zf:
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                if info.is_dir() or not _wanted(info.filename):
                    continue
                if info.file_size > MAX_FILE_BYTES:
                    problems.append(f"{info.filename}: skipped, larger than {MAX_FILE_BYTES} bytes")
                    continue
                _parse_bytes(info.filename, zf.read(info), found, problems)
    else:
        raise ValueError(f"{src} is neither a directory nor a zip file")
    return found, problems


def _stage_records(art: Artifacts, stage_status) -> list[dict]:
    if isinstance(stage_status, list):
        return [r for r in stage_status if isinstance(r, dict)]
    stages = (art.session or {}).get("stages")
    if isinstance(stages, dict):  # session.json dict form: {stage name: record}
        return [{"stage": name, **rec} for name, rec in stages.items() if isinstance(rec, dict)]
    return []


def _attach_coverage(art: Artifacts) -> None:
    """Held-out totals for each run: from the session's eval stage record that names the
    run's report, else (fine-tuned) the training run's own held-out test count."""
    for run in art.runs.values():
        name = Path(run.source.replace(" (search)", "")).name
        for rec in art.stage_records:
            report = rec.get("eval_report")
            if report and (report == name or Path(str(report)).stem == Path(name).stem):
                run.held_out_total = _int(rec.get("held_out_total"))
                full = rec.get("full_held_out_set")
                run.full_held_out_set = full if isinstance(full, bool) and not run.is_search else None
        if run.held_out_total is None and run.model == "ft" and art.split_info:
            run.held_out_total = _int(art.split_info.get("test"))
        if run.full_held_out_set is None and run.held_out_total is not None and not run.from_summary \
                and not run.is_search:
            run.full_held_out_set = len(run.rows) == run.held_out_total


def _place_runs(art: Artifacts, finals: list[EvalRun], logs: list[EvalRun]) -> None:
    """One run per (model, search mode, prompt mode). The newest final report wins.
    An incremental log replaces it only if it holds every prompt the report has and more
    (a resumed run that was killed again); otherwise the log is listed as superseded."""
    for run in sorted(finals, key=lambda r: (r.generated_at, r.source)):
        if run.key in art.runs:
            art.superseded.append(art.runs[run.key].source)
        art.runs[run.key] = run
    for log in logs:
        final = art.runs.get(log.key)
        if final is None:
            art.runs[log.key] = log
        elif set(log.entry_ids) > set(final.entry_ids) and not final.from_summary:
            art.superseded.append(final.source)
            art.runs[log.key] = log
        else:
            art.superseded.append(log.source)


def load_artifacts(src: Path | str) -> Artifacts:
    src = Path(src)
    files, problems = _read_files(src)
    art = Artifacts(source=str(src), problems=problems)
    finals: list[EvalRun] = []
    logs: list[EvalRun] = []
    training_runs: dict[str, TrainingRun] = {}
    stage_status = None
    review_files: dict[str, dict] = {}
    key_files: dict[str, dict] = {}

    for name, data in files.items():
        base = Path(name).name
        if _add_training_file(training_runs, name, data):
            continue
        if isinstance(data, list):  # stage_status.json or a .jsonl
            if base == "stage_status.json":
                stage_status = data
            elif base.startswith("progress_") and name.endswith(".jsonl"):
                art.progress.append(parse_progress_log(data, name))
            elif INCREMENTAL_LOG.match(base):
                runs, why = parse_incremental_log(data, name)
                logs.extend(runs) if runs else art.excluded.append((name, why))
            elif name.endswith(".jsonl"):
                reason = ("dry run (dryrun_*): not model output" if base.startswith("dryrun_") else
                          "not one of the current incremental logs (e.g. an older log moved aside)")
                art.excluded.append((name, reason))
            continue
        if not isinstance(data, dict):
            continue
        if base.endswith(".key.json"):
            key_files[name[: -len(".key.json")]] = data
        elif base == "session.json":
            art.session = data
        elif base == "generation_report.json":
            art.dataset = parse_generation_report(data, name)
        elif base == "interpret_report.json":
            art.interpret = parse_interpret_report(data, name)
        elif base == "split_info.json":
            if art.split_info is None or str(data.get("created_at", "")) > str(art.split_info.get("created_at", "")):
                art.split_info = data
        elif "rubric" in data and "items" in data:
            review_files[name] = data
        elif "run_name" in data and ("results" in data or "per_category" in data):
            runs, why = parse_eval_report_runs(data, name)
            finals.extend(runs) if runs else art.excluded.append((name, why))
        elif base.startswith("dryrun_"):
            art.excluded.append((name, "dry run (dryrun_*): not model output"))

    for name, review in review_files.items():
        key = key_files.get(name[: -len(".json")])
        art.reviews.extend(parse_reviews(review, name, key))

    art.stage_records = _stage_records(art, stage_status)

    # Run folders are named run_<UTC timestamp>, so name order is time order.
    for run in sorted(training_runs.values(), key=lambda r: r.name):
        if not run.has_data:
            continue
        if run.is_dry_run:
            art.excluded.append((f"training/{run.name}", "dry-run training (tiny random model)"))
            continue
        art.training_runs.append(run)
    if art.training:
        art.split_info = art.training.split_info or art.split_info
        finetune = next((r for r in art.stage_records if r.get("stage") == "finetune"), None)
        if finetune and finetune.get("status") == "ok":
            art.training.stage_seconds = _float(finetune.get("seconds"))

    # If generation_report.json wasn't collected but session.json has a copy, use it.
    if art.dataset is None and art.session and isinstance(art.session.get("dataset_report"), dict):
        art.dataset = parse_generation_report(art.session["dataset_report"], "session.json (dataset_report)")

    _place_runs(art, finals, logs)
    _attach_coverage(art)
    return art
