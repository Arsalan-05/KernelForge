"""Shared evaluation utilities: metrics aggregation and result persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kernelforge.dataset import CATEGORIES
from kernelforge.parsing import ParsedOutput, parse_model_output
from verification.verify_kernel import VerifyResult, verify_model_kernel

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class EntryEvalResult:
    id: str
    category: str
    generated_text: str
    parsed: ParsedOutput
    verification: VerifyResult | None
    has_explanation: bool
    explanation_matches_reference: bool | None = None
    error: str | None = None


@dataclass
class CategoryMetrics:
    category: str
    total: int = 0
    parsed_kernel: int = 0
    parsed_explanation: int = 0
    correct: int = 0
    passed: int = 0
    speedups: list[float] = field(default_factory=list)

    @property
    def pct_correct(self) -> float | None:
        return (self.correct / self.total * 100) if self.total else None

    @property
    def pct_passed(self) -> float | None:
        return (self.passed / self.total * 100) if self.total else None

    @property
    def avg_speedup(self) -> float | None:
        return sum(self.speedups) / len(self.speedups) if self.speedups else None

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "total": self.total,
            "parsed_kernel": self.parsed_kernel,
            "parsed_explanation": self.parsed_explanation,
            "correct": self.correct,
            "passed": self.passed,
            "pct_correct": self.pct_correct,
            "pct_passed": self.pct_passed,
            "avg_speedup": self.avg_speedup,
        }


def evaluate_generated_kernel(
    entry: dict,
    generated_text: str,
    *,
    device: str = "cuda",
    warmup: int = 5,
    iters: int = 20,
    timeout_s: float = 60.0,
    skip_verification: bool = False,
) -> EntryEvalResult:
    """Parse model output and optionally run the verification harness."""
    parsed = parse_model_output(generated_text)
    verification = None
    error = None

    if not parsed.has_kernel:
        error = "no parseable kernel in model output"
    elif not skip_verification:
        try:
            tol = entry.get("tolerance", {"atol": 1e-2, "rtol": 1e-2})
            verification = verify_model_kernel(
                entry["pytorch_reference"],
                parsed.triton_kernel,
                device=device,
                atol=tol["atol"],
                rtol=tol["rtol"],
                warmup=warmup,
                iters=iters,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            error = str(exc)

    return EntryEvalResult(
        id=entry["id"],
        category=entry["category"],
        generated_text=generated_text,
        parsed=parsed,
        verification=verification,
        has_explanation=parsed.has_explanation,
        error=error,
    )


def aggregate_metrics(results: list[EntryEvalResult]) -> dict[str, CategoryMetrics]:
    """Aggregate per-entry results into per-category metrics."""
    metrics = {c: CategoryMetrics(category=c) for c in CATEGORIES}

    for result in results:
        m = metrics.setdefault(result.category, CategoryMetrics(category=result.category))
        m.total += 1
        if result.parsed.has_kernel:
            m.parsed_kernel += 1
        if result.has_explanation:
            m.parsed_explanation += 1
        if result.verification and result.verification.correct:
            m.correct += 1
        if result.verification and result.verification.passed:
            m.passed += 1
            if result.verification.speedup is not None:
                m.speedups.append(result.verification.speedup)

    return metrics


def save_eval_report(
    *,
    run_name: str,
    model_name: str,
    adapter_path: str | None,
    device: str,
    results: list[EntryEvalResult],
    metrics: dict[str, CategoryMetrics],
    extra: dict | None = None,
) -> Path:
    """Write a JSON evaluation report to evaluation/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{run_name}_{timestamp}.json"

    serializable_results = []
    for r in results:
        serializable_results.append(
            {
                "id": r.id,
                "category": r.category,
                "has_explanation": r.has_explanation,
                "parsed_kernel": r.parsed.has_kernel,
                "verification": asdict(r.verification) if r.verification else None,
                "error": r.error,
                "generated_text_preview": r.generated_text[:500],
            }
        )

    report = {
        "run_name": run_name,
        "model_name": model_name,
        "adapter_path": adapter_path,
        "device": device,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "per_category": {c: m.to_dict() for c, m in metrics.items() if m.total > 0},
        "results": serializable_results,
        **(extra or {}),
    }
    out_path.write_text(json.dumps(report, indent=2))
    return out_path


def print_metrics_table(metrics: dict[str, CategoryMetrics], title: str = "Results") -> None:
    """Print a category-level summary table to stdout."""
    print(f"\n=== {title} ===")
    print(f"{'Category':<20} {'Total':>6} {'Kernel':>7} {'Expl':>6} {'Correct':>8} {'Passed':>7} {'Avg speedup':>12}")
    for category in CATEGORIES:
        m = metrics.get(category)
        if not m or m.total == 0:
            continue
        speedup = f"{m.avg_speedup:.2f}x" if m.avg_speedup is not None else "n/a"
        print(
            f"{category:<20} {m.total:>6} {m.parsed_kernel:>7} {m.parsed_explanation:>6} "
            f"{m.pct_correct:>7.1f}% {m.pct_passed:>6.1f}% {speedup:>12}"
        )
