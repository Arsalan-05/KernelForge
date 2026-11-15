"""Shared evaluation utilities: metrics aggregation and result persistence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kernelforge.dataset import CATEGORIES, interleave_by_category
from kernelforge.parsing import ParsedOutput, parse_model_output
from kernelforge.prompts import format_response
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
    hit_token_limit: bool | None = None  # generation stopped at max_new_tokens

    @property
    def outcome(self) -> str:
        """One label per entry, so the report separates parse failures from wrong kernels."""
        if not self.parsed.has_kernel:
            return "no_kernel"
        if self.verification is None:
            return "harness_exception" if self.error else "not_verified"
        if self.verification.status != "ok":
            return self.verification.status  # "error", "timeout", ...
        return "correct" if self.verification.correct else "incorrect"


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
    interpret: bool = False,
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
                interpret=interpret,
            )
        except Exception as exc:
            error = str(exc)

    return EntryEvalResult(
        id=entry["id"],
        category=entry["category"],
        generated_text=generated_text,
        parsed=parsed,
        verification=verification,
        # Without a kernel the parser returns the raw text as the "explanation";
        # don't count that (same rule as kernelforge/search.py).
        has_explanation=parsed.has_explanation and parsed.has_kernel,
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


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags shared by baseline_eval.py and finetuned_eval.py."""
    parser.add_argument("--config", type=Path, default=Path("training/configs/qwen2.5-coder-1.5b.json"))
    parser.add_argument("--dataset", type=Path, default=None, help="Path to verified dataset.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Cap test examples (round-robin across categories)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--skip-verification", action="store_true", help="Parse only, no harness run")
    parser.add_argument("--interpret", action="store_true",
                        help="Verify with Triton's CPU interpreter (correctness only, no speedups)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the model; score the ground-truth kernels to check the eval pipeline")
    parser.add_argument("--split", choices=["template", "random"], default="template",
                        help="template = held-out ops (no leakage); random = shape-level split")


def run_eval(test_entries: list[dict], args: argparse.Namespace, generator=None) -> list[EntryEvalResult]:
    """Generate (or, on --dry-run, use the reference answer) and verify each entry."""
    if args.limit:
        test_entries = interleave_by_category(test_entries)[: args.limit]
    results = []
    for i, entry in enumerate(test_entries, 1):
        print(f"[{i}/{len(test_entries)}] {entry['id']} ({entry['category']})")
        if args.dry_run:
            generated, hit_limit = format_response(entry), None  # exactly the training target
        else:
            generated = generator.generate(entry["description"], entry["pytorch_reference"])
            hit_limit = generator.last_hit_token_limit

        result = evaluate_generated_kernel(
            entry, generated, device=args.device, warmup=args.warmup, iters=args.iters,
            timeout_s=args.timeout, skip_verification=args.skip_verification, interpret=args.interpret,
        )
        result.hit_token_limit = hit_limit
        results.append(result)

        note = "  (hit max_new_tokens)" if hit_limit else ""
        if result.verification:
            v = result.verification
            print(f"  {result.outcome}: status={v.status} correct={v.correct} speedup={v.speedup}{note}")
            if v.status != "ok" and v.error:
                print(f"    {v.error.strip().splitlines()[-1][:200]}")
        else:
            print(f"  {result.outcome}: kernel={result.parsed.has_kernel} "
                  f"explanation={result.has_explanation}{note}" + (f" ({result.error})" if result.error else ""))
    return results


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
                "outcome": r.outcome,
                "has_explanation": r.has_explanation,
                "parsed_kernel": r.parsed.has_kernel,
                "hit_token_limit": r.hit_token_limit,
                "verification": asdict(r.verification) if r.verification else None,
                "error": r.error,
                # The explanation sits after the kernel, so a 500-char preview
                # almost never contains it; keep it and the full text.
                "generated_explanation": r.parsed.optimization_explanation if r.has_explanation else None,
                "generated_text": r.generated_text,
                "generated_text_preview": r.generated_text[:500],
            }
        )

    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1

    report = {
        "run_name": run_name,
        "model_name": model_name,
        "adapter_path": adapter_path,
        "device": device,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "per_category": {c: m.to_dict() for c, m in metrics.items() if m.total > 0},
        "outcomes": outcomes,
        "hit_token_limit": sum(1 for r in results if r.hit_token_limit),
        "results": serializable_results,
        **(extra or {}),
    }
    out_path.write_text(json.dumps(report, indent=2))
    return out_path


def print_metrics_table(metrics: dict[str, CategoryMetrics], title: str = "Results") -> None:
    """Print a category-level summary table to stdout."""
    print(f"\n=== {title} ===")
    print(f"{'Category':<20} {'Total':>6} {'Kernel':>7} {'Expl':>6} {'Correct':>8} {'Passed':>7} {'Avg speedup':>12}")
    # Known categories first, then anything else present (e.g. schema examples).
    for category in list(CATEGORIES) + sorted(set(metrics) - set(CATEGORIES)):
        m = metrics.get(category)
        if not m or m.total == 0:
            continue
        speedup = f"{m.avg_speedup:.2f}x" if m.avg_speedup is not None else "n/a"
        print(
            f"{category:<20} {m.total:>6} {m.parsed_kernel:>7} {m.parsed_explanation:>6} "
            f"{m.pct_correct:>7.1f}% {m.pct_passed:>6.1f}% {speedup:>12}"
        )


def print_outcomes(results: list[EntryEvalResult]) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    print("Outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    cut = sum(1 for r in results if r.hit_token_limit)
    if cut:
        print(f"WARNING: {cut}/{len(results)} generations hit max_new_tokens; raise it if these failed to parse.")

