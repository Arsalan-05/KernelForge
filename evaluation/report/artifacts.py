"""Find and parse the artifacts a GPU session writes into kernelforge_outputs(.zip).

Layout produced by scripts/gpu_session.py (paths relative to the outputs root;
the zip has no top-level folder, but we match on path suffixes so either works):

    session.json                          stage statuses, env (GPU, versions), dataset_report copy
    data/raw/generation_report.json       dataset build report (data/build_dataset.py --device cuda)
    data/raw/interpret_report.json        interpreter-only dataset check (if someone ran --interpret)
    evaluation/results/baseline_*.json    evaluation/common.py::save_eval_report, run_name="baseline"
    evaluation/results/finetuned_*.json   same format, run_name="finetuned"
    training/run_*/split_info.json        training/finetune.py split description
    **/*review*.json                      evaluation/explanation_review.py --output (scored by hand)

Search results (best-of-N / repair): no script writes these yet. We read them
from the same eval-report format, with the search settings in either a
top-level `"search": {"num_candidates": 4, "repair_rounds": 1}` object, top-level
`num_candidates` / `repair_rounds` keys, or a run_name such as
"baseline_best4_repair1". A report with none of these is treated as pass@1.

Nothing here invents a value: a stage that is missing stays missing, and a
report that cannot stand for model output (dry run, parse-only) is excluded
with the reason recorded.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

MAX_JSON_BYTES = 50 * 1024 * 1024  # skip anything larger; no report is close to this


# --------------------------------------------------------------------------- evaluation runs


@dataclass
class CategoryResult:
    """Counts for one category of one eval run."""

    n: int = 0
    correct: int = 0
    speedups: list[float] = field(default_factory=list)  # passing kernels only
    infinite_speedups: int = 0  # passing kernels the harness timed at 0 s (speedup = inf)
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
class EvalRun:
    source: str
    model: str  # "base" or "ft"
    num_candidates: int
    repair_rounds: int
    run_name: str = ""
    model_name: str = ""
    adapter_path: Optional[str] = None
    device: str = ""
    generated_at: str = ""
    split: Optional[str] = None
    held_out_templates: list[str] = field(default_factory=list)
    interpreted: bool = False  # correctness only, no timings
    from_summary: bool = False  # per-entry results were missing
    entry_ids: list[str] = field(default_factory=list)
    categories: dict[str, CategoryResult] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.model, self.num_candidates, self.repair_rounds)

    @property
    def mode_label(self) -> str:
        return mode_label(self.num_candidates, self.repair_rounds)

    @property
    def label(self) -> str:
        return f"{'Fine-tuned' if self.model == 'ft' else 'Base'} {self.mode_label}"

    def overall(self) -> CategoryResult:
        total = CategoryResult()
        for c in self.categories.values():
            total.n += c.n
            total.correct += c.correct
            total.speedups.extend(c.speedups)
            total.infinite_speedups += c.infinite_speedups
        if self.from_summary:
            # Pre-aggregated reports only give per-category means; a pooled mean
            # weighted by the number of passing kernels is the honest overall.
            weighted = [(c.summary_avg_speedup, c.summary_passed or 0) for c in self.categories.values()
                        if c.summary_avg_speedup is not None]
            passed = sum(w for _, w in weighted)
            total.summary_passed = sum(c.summary_passed or 0 for c in self.categories.values())
            total.summary_avg_speedup = (sum(a * w for a, w in weighted) / passed) if passed else None
        return total


def mode_label(num_candidates: int, repair_rounds: int) -> str:
    if num_candidates <= 1 and repair_rounds == 0:
        return "pass@1"
    label = f"best-of-{num_candidates}" if num_candidates > 1 else "single sample"
    if repair_rounds == 1:
        label += " + repair"
    elif repair_rounds > 1:
        label += f" + {repair_rounds} repairs"
    return label


def _search_settings(report: dict) -> tuple[int, int]:
    """(num_candidates, repair_rounds) for a report; (1, 0) means plain pass@1."""
    search = report.get("search") if isinstance(report.get("search"), dict) else {}
    nc = search.get("num_candidates", report.get("num_candidates"))
    rr = search.get("repair_rounds", report.get("repair_rounds"))
    name = str(report.get("run_name", ""))
    if nc is None:
        m = re.search(r"best(?:_?of)?_?(\d+)", name)
        nc = int(m.group(1)) if m else 1
    if rr is None:
        m = re.search(r"repair_?(\d+)", name)
        rr = int(m.group(1)) if m else (1 if "repair" in name else 0)
    return int(nc), int(rr)


def parse_eval_report(report: dict, source: str) -> tuple[Optional[EvalRun], Optional[str]]:
    """Turn one save_eval_report() JSON into an EvalRun, or return (None, why it was excluded)."""
    if report.get("dry_run"):
        return None, "dry run: scored the dataset's own kernels, not model output"
    if report.get("skip_verification"):
        return None, "parse-only run (--skip-verification): no correctness measured"

    run_name = str(report.get("run_name", ""))
    is_ft = run_name.startswith("finetuned") or bool(report.get("adapter_path"))
    nc, rr = _search_settings(report)
    run = EvalRun(
        source=source,
        model="ft" if is_ft else "base",
        num_candidates=nc,
        repair_rounds=rr,
        run_name=run_name,
        model_name=str(report.get("model_name") or ""),
        adapter_path=report.get("adapter_path"),
        device=str(report.get("device") or ""),
        generated_at=str(report.get("generated_at") or ""),
        split=report.get("split"),
        held_out_templates=list(report.get("held_out_templates") or []),
    )

    results = report.get("results")
    if isinstance(results, list) and results:
        for r in results:
            cat = run.categories.setdefault(r.get("category", "unknown"), CategoryResult())
            cat.n += 1
            run.entry_ids.append(str(r.get("id")))
            v = r.get("verification") or {}
            if v.get("interpreted"):
                run.interpreted = True
            # Same definition as evaluation/common.py::aggregate_metrics:
            # "correct" = the harness compared outputs and they matched;
            # speedups only for kernels that also ran cleanly (status ok).
            if v.get("correct") is True:
                cat.correct += 1
                if v.get("status") == "ok" and v.get("speedup") is not None:
                    if finite([v["speedup"]]):
                        cat.speedups.append(float(v["speedup"]))
                    else:
                        cat.infinite_speedups += 1
    elif isinstance(report.get("per_category"), dict):
        run.from_summary = True
        for name, m in report["per_category"].items():
            run.categories[name] = CategoryResult(
                n=int(m.get("total") or 0),
                correct=int(m.get("correct") or 0),
                summary_avg_speedup=m.get("avg_speedup"),
                summary_passed=int(m.get("passed") or 0),
            )
    else:
        return None, "no per-entry results and no per_category summary"
    if run.interpreted:
        # Interpreter timings are meaningless; never average them.
        for c in run.categories.values():
            c.speedups, c.summary_avg_speedup = [], None
    return run, None


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


def _rejection_reason(status: str, passed: bool, error: Optional[str]) -> str:
    """Mirror of data/build_dataset.py::_rejection_reason, for interpreter reports
    (which store raw status + error instead of a reason)."""
    err = (error or "").lower()
    if "out of memory" in err or "outofmemoryerror" in err or "exit code -9" in err:
        return "oom"
    if status == "ok" and not passed:
        return "incorrect"
    return status or "unknown"


def parse_generation_report(report: dict, source: str) -> DatasetReport:
    """data/raw/generation_report.json (GPU build)."""
    per_category = {}
    for c, m in (report.get("per_category") or {}).items():
        counts = OutcomeCounts(passed=int(m.get("verified") or 0), median_speedup=m.get("median_speedup"))
        for reason, n in (m.get("rejection_reasons") or {}).items():
            counts.add_reason(reason, int(n))
        per_category[c] = counts
    per_template = {}
    for name, t in (report.get("per_template") or {}).items():
        counts = OutcomeCounts(passed=int(t.get("verified") or 0), median_speedup=t.get("median_speedup"))
        for reason, n in (t.get("rejected") or {}).items():
            counts.add_reason(reason, int(n))
        per_template[name] = counts
    return DatasetReport(source, str(report.get("device") or ""), False, bool(report.get("smoke")),
                         per_category, per_template)


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
                         per_category, per_template)


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


@dataclass
class TrainingRun:
    """What one training/run_*/ folder left behind. Every field is optional because
    scripts/gpu_session.py currently collects only split_info.json (and the adapter)."""

    name: str
    split_info: Optional[dict] = None
    trainer_state: Optional[dict] = None  # HF Trainer checkpoint-*/trainer_state.json (newest checkpoint)
    trainer_state_source: str = ""
    trainer_state_step: float = -1  # checkpoint number of trainer_state (inf = run-level file)
    metrics: Optional[dict] = None  # training_metrics.json, e.g. {"peak_memory_gb": 9.8, "wall_clock_s": 3600}

    def _last_log(self, key: str) -> Optional[float]:
        for entry in reversed((self.trainer_state or {}).get("log_history") or []):
            if isinstance(entry, dict) and entry.get(key) is not None:
                return float(entry[key])
        return None

    @property
    def final_train_loss(self) -> Optional[float]:
        # The summary entry Trainer appends at the end has "train_loss" (mean over the run);
        # fall back to the last logged step "loss".
        return self._last_log("train_loss") if self._last_log("train_loss") is not None else self._last_log("loss")

    @property
    def final_eval_loss(self) -> Optional[float]:
        return self._last_log("eval_loss")

    @property
    def wall_clock_s(self) -> Optional[float]:
        explicit = (self.metrics or {}).get("wall_clock_s")
        return float(explicit) if explicit is not None else self._last_log("train_runtime")

    @property
    def peak_memory_gb(self) -> Optional[float]:
        value = (self.metrics or {}).get("peak_memory_gb")
        return float(value) if value is not None else None

    @property
    def has_data(self) -> bool:
        return any(x is not None for x in (self.split_info, self.trainer_state, self.metrics))


_TRAINING_PATH = re.compile(r"(?:^|/)training/(run_[^/]+)/(.+)$")
_CHECKPOINT_STEP = re.compile(r"checkpoint-(\d+)/trainer_state\.json$")


def _add_training_file(runs: dict[str, TrainingRun], path: str, data: dict) -> bool:
    """File it under its training run if `path` is inside training/run_*/; True if used."""
    m = _TRAINING_PATH.search(path)
    if not m:
        return False
    run = runs.setdefault(m.group(1), TrainingRun(m.group(1)))
    rel = m.group(2)
    if rel == "split_info.json":
        run.split_info = data
    elif rel == "training_metrics.json":
        run.metrics = data
    elif rel.endswith("trainer_state.json"):
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
    runs: dict[tuple[str, int, int], EvalRun] = field(default_factory=dict)  # latest per (model, mode)
    superseded: list[str] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    dataset: Optional[DatasetReport] = None
    interpret: Optional[DatasetReport] = None
    reviews: list[ExplanationReview] = field(default_factory=list)
    session: Optional[dict] = None
    split_info: Optional[dict] = None
    training: Optional[TrainingRun] = None  # newest training/run_*/ with any artifact
    problems: list[str] = field(default_factory=list)  # unreadable files

    def run(self, model: str, num_candidates: int = 1, repair_rounds: int = 0) -> Optional[EvalRun]:
        return self.runs.get((model, num_candidates, repair_rounds))


def _read_jsons(src: Path) -> tuple[dict[str, object], list[str]]:
    """{relative path: parsed JSON} for every .json file under a directory or in a zip.

    The zip is read in memory and never extracted, so odd member paths can't
    write anywhere. Adapter weights and other non-JSON files are ignored.
    """
    found: dict[str, object] = {}
    problems: list[str] = []

    def _parse(name: str, raw: bytes) -> None:
        try:
            found[name] = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            problems.append(f"{name}: not valid JSON ({exc})")

    if src.is_dir():
        for path in sorted(src.rglob("*.json")):
            rel = path.relative_to(src).as_posix()
            if path.stat().st_size > MAX_JSON_BYTES:
                problems.append(f"{rel}: skipped, larger than {MAX_JSON_BYTES} bytes")
                continue
            _parse(rel, path.read_bytes())
    elif zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as zf:
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                if info.is_dir() or not info.filename.endswith(".json"):
                    continue
                if info.file_size > MAX_JSON_BYTES:
                    problems.append(f"{info.filename}: skipped, larger than {MAX_JSON_BYTES} bytes")
                    continue
                _parse(info.filename, zf.read(info))
    else:
        raise ValueError(f"{src} is neither a directory nor a zip file")
    return found, problems


def load_artifacts(src: Path | str) -> Artifacts:
    src = Path(src)
    files, problems = _read_jsons(src)
    art = Artifacts(source=str(src), problems=problems)
    candidates: list[EvalRun] = []
    training_runs: dict[str, TrainingRun] = {}

    for name, data in files.items():
        if not isinstance(data, dict):
            continue
        base = Path(name).name
        if _add_training_file(training_runs, name, data):
            continue
        if base == "session.json":
            art.session = data
        elif base == "generation_report.json":
            art.dataset = parse_generation_report(data, name)
        elif base == "interpret_report.json":
            art.interpret = parse_interpret_report(data, name)
        elif base == "split_info.json":
            # Several training runs could be present; keep the newest.
            if art.split_info is None or str(data.get("created_at", "")) > str(art.split_info.get("created_at", "")):
                art.split_info = data
        elif "rubric" in data and "items" in data:
            art.reviews.append(parse_review(data, name))
        elif "run_name" in data and ("results" in data or "per_category" in data):
            run, why = parse_eval_report(data, name)
            if run is None:
                art.excluded.append((name, why))
            else:
                candidates.append(run)

    # Run folders are named run_<UTC timestamp>, so the lexically largest is the newest.
    with_data = [r for r in training_runs.values() if r.has_data]
    if with_data:
        art.training = max(with_data, key=lambda r: r.name)
        art.split_info = art.training.split_info or art.split_info

    # If generation_report.json wasn't collected but session.json has a copy, use it.
    if art.dataset is None and art.session and isinstance(art.session.get("dataset_report"), dict):
        art.dataset = parse_generation_report(art.session["dataset_report"], "session.json (dataset_report)")

    # One run per (model, mode): the newest wins; older ones are listed, not mixed in.
    for run in sorted(candidates, key=lambda r: (r.generated_at, r.source)):
        if run.key in art.runs:
            art.superseded.append(art.runs[run.key].source)
        art.runs[run.key] = run
    return art
