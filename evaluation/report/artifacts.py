"""Find and parse the artifacts a GPU session writes into kernelforge_outputs(.zip).

Paths are matched by suffix, so the zip may or may not have a top-level folder:

    stage_status.json                     one record per stage: stage, status (ok/failed/skipped),
                                          seconds, error; eval stages add eval_report,
                                          prompts_evaluated, held_out_total, full_held_out_set
    session.json                          env (GPU, versions), the same stage records as a dict,
                                          dataset_report copy, zip_files
    data/raw/generation_report.json       GPU dataset build (may be "partial": true + "run_stats")
    data/raw/interpret_report.json        interpreter-only dataset check (same optional keys)
    data/raw/progress_<mode>.jsonl        one line per finished dataset verification
    evaluation/results/<run>_*.json       eval report (evaluation/common.py::save_eval_report)
    evaluation/results/<run>_*.jsonl      incremental eval log left by a run that was killed
    training/**/run_*/split_info.json, train_log.json, trainer_state.json,
                      checkpoint-*/trainer_state.json, training_metrics.json
    any JSON with "rubric" + "items"      evaluation/explanation_review.py --output

Eval report keys used (ALL optional; a missing key never becomes a number):
    run_name, model_name, adapter_path, device, generated_at, split, held_out_templates,
    dry_run, skip_verification, partial, prompt_mode ("zero_shot" | "few_shot"),
    data_source, backend, max_new_tokens, baseline_report, outcomes,
    search {num_candidates, repair_rounds, search_limit, prompts_scored},
    results[]: id, category, outcome, verification {status, correct, speedup, interpreted},
               hit_token_limit, generated_explanation, generated_text
    per_category (fallback when results[] is missing)

A search run is its own report: its results[] are the winning candidates for the
search_limit prompts it scored. Anything named dryrun_* is skipped.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
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
PROMPT_LABELS = {"zero_shot": "zero-shot", "few_shot": "few-shot", "unspecified": ""}
MODEL_LABELS = {"base": "Base", "ft": "Fine-tuned"}

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
    search_limit: Optional[int] = None
    prompts_scored: Optional[int] = None
    data_source: Optional[str] = None
    backend: Optional[str] = None
    max_new_tokens: Optional[int] = None
    baseline_report: Optional[str] = None
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
        return mode_label(self.num_candidates, self.repair_rounds)

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


def _search_settings(report: dict) -> tuple[int, int]:
    """(num_candidates, repair_rounds) for a report; (1, 0) means plain pass@1."""
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


def _row(r: dict) -> EntryRow:
    v = r.get("verification") if isinstance(r.get("verification"), dict) else {}
    # Same definition as evaluation/common.py::aggregate_metrics:
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
                    hit_token_limit=hit if isinstance(hit, bool) else None)


def _new_run(meta: dict, source: str) -> EvalRun:
    run_name = str(meta.get("run_name") or Path(source).stem)
    is_ft = run_name.startswith("finetuned") or bool(meta.get("adapter_path"))
    nc, rr = _search_settings({**meta, "run_name": run_name})
    search = meta.get("search") if isinstance(meta.get("search"), dict) else {}
    hit = meta.get("hit_token_limit")
    return EvalRun(
        source=source, model="ft" if is_ft else "base", num_candidates=nc, repair_rounds=rr,
        prompt_mode=str(meta.get("prompt_mode") or "unspecified"),
        run_name=run_name, model_name=str(meta.get("model_name") or ""), adapter_path=meta.get("adapter_path"),
        device=str(meta.get("device") or ""), generated_at=str(meta.get("generated_at") or ""),
        split=meta.get("split"), held_out_templates=list(meta.get("held_out_templates") or []),
        partial=bool(meta.get("partial")),
        search_limit=_int(search.get("search_limit")), prompts_scored=_int(search.get("prompts_scored")),
        data_source=meta.get("data_source"), backend=meta.get("backend"),
        max_new_tokens=_int(meta.get("max_new_tokens")), baseline_report=meta.get("baseline_report"),
        reported_outcomes={k: v for k, v in (meta.get("outcomes") or {}).items() if _int(v) is not None}
        if isinstance(meta.get("outcomes"), dict) else {},
        reported_hit_token_limit=_int(hit),
    )


def _finish(run: EvalRun) -> EvalRun:
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


def parse_eval_report(report: dict, source: str) -> tuple[Optional[EvalRun], Optional[str]]:
    """Turn one eval report JSON into an EvalRun, or return (None, why it was excluded)."""
    why = _excluded_reason(report, source)
    if why:
        return None, why
    run = _new_run(report, source)
    results = report.get("results")
    if isinstance(results, list) and results:
        run.rows = [_row(r) for r in results if isinstance(r, dict)]
        run.interpreted = any((r.get("verification") or {}).get("interpreted")
                              for r in results if isinstance(r, dict) and isinstance(r.get("verification"), dict))
        return _finish(run), None
    if isinstance(report.get("per_category"), dict):
        run.from_summary = True
        for name, m in report["per_category"].items():
            run.categories[name] = CategoryResult(
                n=_int(m.get("total")) or 0, correct=_int(m.get("correct")) or 0,
                summary_avg_speedup=_float(m.get("avg_speedup")), summary_passed=_int(m.get("passed")) or 0)
        return run, None
    return None, "no per-entry results and no per_category summary"


def parse_incremental_log(lines: list[dict], source: str) -> tuple[Optional[EvalRun], Optional[str]]:
    """An eval's incremental .jsonl (no final report: the run was killed). Lines with an "id"
    are result rows; any other line is run metadata (run_name, prompt_mode, search, ...)."""
    meta: dict = {}
    for line in lines:
        if "id" not in line:
            meta.update(line)
    why = _excluded_reason(meta, source)
    if why:
        return None, why
    rows = [line for line in lines if "id" in line]
    if not rows:
        return None, "incremental log with no scored prompts"
    run = _new_run(meta, source)
    run.partial = run.incremental = True
    run.rows = [_row(r) for r in rows]
    run.interpreted = any((r.get("verification") or {}).get("interpreted")
                          for r in rows if isinstance(r.get("verification"), dict))
    return _finish(run), None


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
        counts = OutcomeCounts(passed=_int(m.get("verified")) or 0, median_speedup=_float(m.get("median_speedup")))
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
                         per_category, per_template, bool(report.get("partial")), _run_stats(report))


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
                         per_category, per_template, bool(report.get("partial")), _run_stats(report))


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
    source: str
    source_report: str
    model: str  # "base", "ft" or "unknown"
    n_items: int
    scores: list[int]  # only items a human actually scored (0/1/2)

    @property
    def mean_score(self) -> Optional[float]:
        return mean(self.scores)

    @property
    def top_score(self) -> Proportion:
        """Share of scored explanations that got a 2 (specific and correct)."""
        return Proportion(sum(1 for s in self.scores if s == 2), len(self.scores))


def parse_review(report: dict, source: str) -> ExplanationReview:
    src = str(report.get("source_report") or "")
    name = Path(src).name
    model = "ft" if name.startswith("finetuned") else "base" if name.startswith("baseline") else "unknown"
    items = report.get("items") or []
    scores = [int(i["rubric_score"]) for i in items
              if isinstance(i, dict) and i.get("rubric_score") in (0, 1, 2)]
    return ExplanationReview(source, src, model, len(items), scores)


# --------------------------------------------------------------------------- training runs

# training_metrics.json key spellings we accept, and the factor to the unit we report.
_WALL_CLOCK_KEYS = (("wall_clock_s", 1.0), ("wall_clock_seconds", 1.0), ("train_runtime_s", 1.0),
                    ("train_runtime", 1.0), ("seconds", 1.0))
_MEMORY_KEYS = (("peak_memory_gb", 1.0), ("peak_gpu_memory_gb", 1.0), ("peak_memory_mb", 1e-3),
                ("peak_gpu_memory_mb", 1e-3), ("peak_memory_bytes", 1e-9), ("peak_gpu_memory_bytes", 1e-9),
                ("max_memory_allocated", 1e-9))  # torch.cuda.max_memory_allocated() is in bytes
_STEP_KEYS = ("steps", "global_step", "optimizer_steps")


@dataclass
class TrainingRun:
    """What one training/run_*/ folder left behind. Every field is optional."""

    name: str
    split_info: Optional[dict] = None
    trainer_state: Optional[dict] = None  # HF Trainer trainer_state.json (newest checkpoint)
    trainer_state_source: str = ""
    trainer_state_step: float = -1  # checkpoint number of trainer_state (inf = run-level file)
    train_log: Optional[list] = None  # train_log.json: Trainer.state.log_history
    metrics: Optional[dict] = None  # training_metrics.json: wall-clock, steps, peak GPU memory

    def _log_history(self) -> list:
        state_log = (self.trainer_state or {}).get("log_history")
        return state_log if isinstance(state_log, list) else (self.train_log or [])

    def _last_log(self, key: str) -> Optional[float]:
        for entry in reversed(self._log_history()):
            if isinstance(entry, dict) and _float(entry.get(key)) is not None:
                return _float(entry[key])
        return None

    def _metric(self, keys) -> Optional[float]:
        for key, factor in keys:
            value = _float((self.metrics or {}).get(key))
            if value is not None:
                return value * factor
        return None

    @property
    def final_train_loss(self) -> Optional[float]:
        # The summary entry Trainer appends at the end has "train_loss" (mean over the run);
        # fall back to the last logged step "loss".
        value = self._last_log("train_loss")
        return value if value is not None else self._last_log("loss")

    @property
    def final_eval_loss(self) -> Optional[float]:
        return self._last_log("eval_loss")

    @property
    def wall_clock_s(self) -> Optional[float]:
        explicit = self._metric(_WALL_CLOCK_KEYS)
        return explicit if explicit is not None else self._last_log("train_runtime")

    @property
    def peak_memory_gb(self) -> Optional[float]:
        return self._metric(_MEMORY_KEYS)

    @property
    def steps(self) -> Optional[int]:
        for key in _STEP_KEYS:
            if _int((self.metrics or {}).get(key)) is not None:
                return _int(self.metrics[key])
        state_step = _int((self.trainer_state or {}).get("global_step"))
        if state_step is not None:
            return state_step
        step = self._last_log("step")
        return int(step) if step is not None else None

    @property
    def epochs(self) -> Optional[float]:
        value = _float((self.trainer_state or {}).get("epoch"))
        return value if value is not None else self._last_log("epoch")

    @property
    def has_data(self) -> bool:
        return any(x is not None for x in (self.split_info, self.trainer_state, self.train_log, self.metrics))


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
    elif rel == "training_metrics.json" and isinstance(data, dict):
        run.metrics = data
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
    training: Optional[TrainingRun] = None  # newest training/run_*/ with any artifact
    problems: list[str] = field(default_factory=list)  # unreadable files

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
        name = Path(run.source).name
        for rec in art.stage_records:
            report = rec.get("eval_report")
            if report and (report == name or Path(str(report)).stem == Path(name).stem):
                run.held_out_total = _int(rec.get("held_out_total"))
                full = rec.get("full_held_out_set")
                run.full_held_out_set = full if isinstance(full, bool) else None
        if run.held_out_total is None and run.model == "ft" and art.split_info:
            run.held_out_total = _int(art.split_info.get("test"))
        if run.full_held_out_set is None and run.held_out_total is not None and not run.from_summary:
            run.full_held_out_set = len(run.rows) == run.held_out_total


def load_artifacts(src: Path | str) -> Artifacts:
    src = Path(src)
    files, problems = _read_files(src)
    art = Artifacts(source=str(src), problems=problems)
    finals: list[EvalRun] = []
    incremental: list[EvalRun] = []
    training_runs: dict[str, TrainingRun] = {}
    stage_status = None

    for name, data in files.items():
        base = Path(name).name
        if _add_training_file(training_runs, name, data):
            continue
        if isinstance(data, list):  # stage_status.json or a .jsonl
            if base == "stage_status.json":
                stage_status = data
            elif base.startswith("progress_") and name.endswith(".jsonl"):
                art.progress.append(parse_progress_log(data, name))
            elif name.endswith(".jsonl"):
                run, why = parse_incremental_log(data, name)
                (incremental.append(run) if run else art.excluded.append((name, why)))
            continue
        if not isinstance(data, dict):
            continue
        if base == "session.json":
            art.session = data
        elif base == "generation_report.json":
            art.dataset = parse_generation_report(data, name)
        elif base == "interpret_report.json":
            art.interpret = parse_interpret_report(data, name)
        elif base == "split_info.json":
            if art.split_info is None or str(data.get("created_at", "")) > str(art.split_info.get("created_at", "")):
                art.split_info = data
        elif "rubric" in data and "items" in data:
            art.reviews.append(parse_review(data, name))
        elif "run_name" in data and ("results" in data or "per_category" in data):
            run, why = parse_eval_report(data, name)
            (finals.append(run) if run else art.excluded.append((name, why)))
        elif base.startswith("dryrun_"):
            art.excluded.append((name, "dry run (dryrun_*): not model output"))

    art.stage_records = _stage_records(art, stage_status)

    # Run folders are named run_<UTC timestamp>, so the lexically largest is the newest.
    with_data = [r for r in training_runs.values() if r.has_data]
    if with_data:
        art.training = max(with_data, key=lambda r: r.name)
        art.split_info = art.training.split_info or art.split_info

    # If generation_report.json wasn't collected but session.json has a copy, use it.
    if art.dataset is None and art.session and isinstance(art.session.get("dataset_report"), dict):
        art.dataset = parse_generation_report(art.session["dataset_report"], "session.json (dataset_report)")

    # One run per (model, search mode, prompt mode): the newest final report wins.
    for run in sorted(finals, key=lambda r: (r.generated_at, r.source)):
        if run.key in art.runs:
            art.superseded.append(art.runs[run.key].source)
        art.runs[run.key] = run
    # An incremental log is used only when no final report covers it: same key, same
    # file stem, or a final report of the same model and search mode that scored all its prompts.
    for run in incremental:
        covered = any(
            f.key == run.key or Path(f.source).stem == Path(run.source).stem
            or (f.model == run.model and (f.num_candidates, f.repair_rounds) == (run.num_candidates, run.repair_rounds)
                and set(run.entry_ids) <= set(f.entry_ids))
            for f in finals)
        if covered or run.key in art.runs:
            art.superseded.append(run.source)
        else:
            art.runs[run.key] = run
    _attach_coverage(art)
    return art
