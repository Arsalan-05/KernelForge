"""Shared evaluation code for baseline_eval.py and finetuned_eval.py.

One eval run has up to two phases:
  pass@1  greedy decoding, one sample per prompt, on ALL held-out prompts;
  search  best-of-N + harness-feedback repair (kernelforge/search.py) on a
          stratified subset of `--search-limit` prompts, with a fixed seed.

Every finished prompt is appended to a JSONL log, so a killed run keeps its
progress; `--resume` skips prompts already in the log, and `--max-minutes`
stops cleanly between prompts. The JSON report is always rebuilt from the log.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from kernelforge.dataset import CATEGORIES, interleave_by_category
from kernelforge.model import effective_settings, generation_kwargs
from kernelforge.parsing import ParsedOutput, parse_model_output
from kernelforge.prompts import format_response
from kernelforge.search import SearchConfig, run_search
from verification.verify_kernel import VerifyResult, verify_model_kernel

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_MAX_NEW_TOKENS = 2048  # longest reference answer is 1,302 Qwen tokens
# Overrides Qwen2.5's bundled 1.05 for every eval generation (pass@1, few-shot,
# search, fine-tuned), so no hidden penalty shapes the measured outputs.
EVAL_REPETITION_PENALTY = 1.0


def outcome_of(parsed_kernel: bool, verification: dict | None, error: str | None = None) -> str:
    """One label per entry, so reports separate parse failures from wrong kernels."""
    if not parsed_kernel:
        return "no_kernel"
    if not verification or verification.get("ran") is False:
        return "harness_exception" if error else "not_verified"
    if verification.get("status") != "ok":
        return verification.get("status") or "error"  # "error", "timeout", "crash"
    return "correct" if verification.get("correct") else "incorrect"


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
        return outcome_of(self.parsed.has_kernel, asdict(self.verification) if self.verification else None,
                          self.error)


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


def result_row(r: EntryEvalResult, **extra) -> dict:
    """JSON-ready row for one pass@1 result."""
    return {
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
        **extra,
    }


def aggregate_rows(rows: list[dict]) -> dict[str, CategoryMetrics]:
    """Per-category metrics from result rows (pass@1 or search rows; same fields)."""
    metrics = {c: CategoryMetrics(category=c) for c in CATEGORIES}
    for row in rows:
        m = metrics.setdefault(row["category"], CategoryMetrics(category=row["category"]))
        m.total += 1
        m.parsed_kernel += bool(row["parsed_kernel"])
        m.parsed_explanation += bool(row["has_explanation"])
        v = row.get("verification") or {}
        if v.get("correct"):
            m.correct += 1
        if v.get("status") == "ok" and v.get("correct"):
            m.passed += 1
            if v.get("speedup") is not None:
                m.speedups.append(v["speedup"])
    return metrics


def aggregate_metrics(results: list[EntryEvalResult]) -> dict[str, CategoryMetrics]:
    return aggregate_rows([result_row(r) for r in results])


# --- incremental results -------------------------------------------------------

class ResultsLog:
    """Append-only JSONL: one meta line, then one line per finished prompt.

    Without `resume`, an existing non-empty log is renamed aside (never deleted).
    With `resume`, rows are reloaded and the run settings must match the meta line,
    so rows from different settings can't end up in one report.
    """

    def __init__(self, path: Path, meta: dict, resume: bool):
        self.path = path
        self.rows: list[dict] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume and path.exists():
            lines = self._read(path)
            old_meta = next((line for line in lines if line.get("type") == "meta"), {})
            diffs = {k: (old_meta.get(k), v) for k, v in meta.items() if old_meta.get(k) != v}
            if diffs:
                raise SystemExit(f"--resume: settings differ from {path} (old, new): {diffs}")
            self.rows = [line for line in lines if line.get("type") == "row"]
            print(f"Resuming from {path}: {len(self.rows)} rows already done.")
            return
        if path.exists() and path.stat().st_size:
            aside = path.with_name(f"{path.stem}.{_timestamp()}{path.suffix}")
            path.rename(aside)
            print(f"Moved previous results log to {aside} (pass --resume to continue a run instead).")
        path.write_text(json.dumps({"type": "meta", **meta}) + "\n")

    @staticmethod
    def _read(path: Path) -> list[dict]:
        lines = []
        for n, text in enumerate(path.read_text().splitlines(), 1):
            if not text.strip():
                continue
            try:
                lines.append(json.loads(text))
            except json.JSONDecodeError:  # a line cut off when the process was killed
                print(f"Ignoring unreadable line {n} of {path}.")
        return lines

    def done(self, phase: str) -> set[str]:
        return {row["id"] for row in self.rows if row["phase"] == phase}

    def phase_rows(self, phase: str) -> list[dict]:
        return [row for row in self.rows if row["phase"] == phase]

    def append(self, phase: str, row: dict) -> None:
        row = {"type": "row", "phase": phase, **row}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)


class Deadline:
    """--max-minutes, checked between prompts (a prompt already running finishes)."""

    def __init__(self, max_minutes: float | None, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._end = None if max_minutes is None else clock() + max_minutes * 60

    def expired(self) -> bool:
        return self._end is not None and self._clock() >= self._end


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --- CLI -------------------------------------------------------------------------

def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags shared by baseline_eval.py and finetuned_eval.py."""
    parser.add_argument("--config", type=Path, default=Path("training/configs/qwen2.5-coder-1.5b.json"))
    parser.add_argument("--dataset", type=Path, default=None, help="Path to verified dataset.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None,
                        help="Dev only: cap pass@1 prompts (round-robin across categories). Default: all held-out")
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
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help=f"Generation cap (default {DEFAULT_MAX_NEW_TOKENS}; overrides the config file)")
    # Incremental results and time limits.
    parser.add_argument("--results-jsonl", type=Path, default=None,
                        help="Per-prompt log (default: evaluation/results/<run name>.jsonl)")
    parser.add_argument("--resume", action="store_true", help="Skip prompts already in the results log")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop cleanly between prompts after this long and write a partial report")
    # Best-of-N + repair search (off unless --num-candidates > 1 or --repair-rounds > 0).
    parser.add_argument("--num-candidates", type=int, default=1, help="Search: samples per prompt in round 0")
    parser.add_argument("--repair-rounds", type=int, default=0, help="Search: harness-feedback repair rounds")
    parser.add_argument("--search-limit", type=int, default=25,
                        help="Search: prompts to run it on, stratified (25 = 5 per category)")
    parser.add_argument("--seed", type=int, default=1234, help="Search: torch seed set before each prompt's search")


def search_enabled(args: argparse.Namespace) -> bool:
    return args.num_candidates > 1 or args.repair_rounds > 0


def run_meta(args: argparse.Namespace, **fields) -> dict:
    """Settings that must not change between a run and its --resume."""
    return {
        "dry_run": args.dry_run, "split": args.split, "device": args.device, "interpret": args.interpret,
        "skip_verification": args.skip_verification, "max_new_tokens": args.max_new_tokens,
        "num_candidates": args.num_candidates, "repair_rounds": args.repair_rounds,
        "search_limit": args.search_limit, "seed": args.seed, **fields,
    }


# --- the two phases --------------------------------------------------------------

def _print_row(n: int, total: int, row: dict) -> None:
    v = row.get("verification") or {}
    note = "  (hit max_new_tokens)" if row.get("hit_token_limit") else ""
    extra = f" speedup={v.get('speedup')}" if v.get("speedup") is not None else ""
    print(f"[{n}/{total}] {row['id']}: {row['outcome']}{extra}{note}")
    if v.get("status") not in (None, "ok") and v.get("error"):
        print(f"    {v['error'].strip().splitlines()[-1][:200]}")


def run_pass_at_1(entries: list[dict], args, generator, log: ResultsLog, deadline: Deadline,
                  example_for: Callable[[dict], dict | None] | None = None) -> bool:
    """Greedy, one sample per prompt. Returns False if stopped by the deadline."""
    done = log.done("pass@1")
    for n, entry in enumerate(entries, 1):
        if entry["id"] in done:
            continue
        if deadline.expired():
            return False
        example = example_for(entry) if example_for else None
        if args.dry_run:
            generated, hit_limit = format_response(entry), None  # exactly the training target
        else:
            generated = generator.generate(entry["description"], entry["pytorch_reference"],
                                           temperature=0.0, example=example)
            hit_limit = generator.last_hit_token_limit
        result = evaluate_generated_kernel(
            entry, generated, device=args.device, warmup=args.warmup, iters=args.iters,
            timeout_s=args.timeout, skip_verification=args.skip_verification, interpret=args.interpret,
        )
        result.hit_token_limit = hit_limit
        row = result_row(result, example_id=example["id"] if example else None)
        log.append("pass@1", row)
        _print_row(n, len(entries), row)
    return True


def _harness_verify(entry: dict, args) -> Callable[[str], dict]:
    def verify(kernel: str) -> dict:
        tol = entry.get("tolerance", {"atol": 1e-2, "rtol": 1e-2})
        try:
            r = verify_model_kernel(entry["pytorch_reference"], kernel, device=args.device, atol=tol["atol"],
                                    rtol=tol["rtol"], warmup=args.warmup, iters=args.iters,
                                    timeout_s=args.timeout, interpret=args.interpret)
        except Exception as exc:
            return {"ran": True, "status": "error", "correct": None, "passed": False, "error": str(exc)}
        return {**asdict(r), "ran": True, "passed": r.passed}

    return verify


def _ground_truth_generate(entry: dict):
    """Dry-run stand-in for KernelGenerator.stream_chat: every sample is the reference answer."""
    def generate(messages, num_sequences, temperature):
        return iter([(i, format_response(entry)) for i in range(num_sequences)])

    return generate


def _skip_reason(args) -> Callable[[str | None], str | None]:
    def skip_reason(kernel: str | None) -> str | None:
        if kernel is None:  # run_search also asks about unparseable candidates
            return "no kernel parsed"
        return "verification skipped (--skip-verification)" if args.skip_verification else None

    return skip_reason


def run_search_phase(entries: list[dict], args, generator, log: ResultsLog, deadline: Deadline,
                     example_for: Callable[[dict], dict | None] | None = None,
                     base_temperature: float | None = None) -> bool:
    """Best-of-N + repair on each entry. Returns False if stopped by the deadline."""
    done = log.done("search")
    for n, entry in enumerate(entries, 1):
        if entry["id"] in done:
            continue
        if deadline.expired():
            return False
        example = example_for(entry) if example_for else None
        if generator is not None:
            import torch

            torch.manual_seed(args.seed)  # same seed per prompt, so a resumed run samples the same way
        config = SearchConfig(args.num_candidates, args.repair_rounds, base_temperature, example=example)
        selected = None
        for event in run_search(
            entry["description"], entry["pytorch_reference"], config,
            generate=generator.stream_chat if generator is not None else _ground_truth_generate(entry),
            verify=_harness_verify(entry, args),
            skip_reason=_skip_reason(args),
            count_tokens=generator.count_tokens if generator is not None else (lambda text: len(text.split())),
        ):
            if event["event"] == "selected":
                selected = event
        parsed_kernel = selected["status"] == "ok"
        verification = selected["verification"]
        row = {
            "id": entry["id"],
            "category": entry["category"],
            "outcome": outcome_of(parsed_kernel, verification),
            "parsed_kernel": parsed_kernel,
            "has_explanation": bool(selected["optimization_explanation"]),
            "verification": verification,
            "selected_candidate": selected["candidate"],
            "selected_round": selected["round"],
            "reason": selected["reason"],
            "num_candidates": selected["num_candidates"],
            "num_passed": selected["num_passed"],
            "candidates": selected["candidates"],
            "generated_explanation": selected["optimization_explanation"],
            "generated_text": selected["raw_output"],
            "example_id": example["id"] if example else None,
        }
        log.append("search", row)
        _print_row(n, len(entries), row)
    return True


# --- orchestration + report ------------------------------------------------------

def _generation_settings(gen_config, generator, temperature: float | None, num_sequences: int = 1) -> dict:
    """Effective generate() settings to record; from the loaded model when there is one."""
    if generator is not None:
        return generator.effective_generation_settings(temperature, num_sequences)
    return effective_settings(generation_kwargs(gen_config, temperature, num_sequences))


def evaluate(args, test_entries: list[dict], *, run_name: str, gen_config, generator, prompt_mode: str,
             few_shot_source: str | None = None, example_for=None,
             extra: dict | None = None) -> tuple[Path, dict]:
    """Run pass@1 (and search, if enabled), then write the report from the results log.

    The caller must have set gen_config.repetition_penalty = EVAL_REPETITION_PENALTY
    before building the generator (checked here).
    """
    if gen_config.repetition_penalty != EVAL_REPETITION_PENALTY:
        raise ValueError(f"eval runs must pin repetition_penalty={EVAL_REPETITION_PENALTY}")
    ordered = interleave_by_category(test_entries)  # round-robin: any prefix covers every category
    pass1_entries = ordered[: args.limit] if args.limit else ordered
    search_entries = pass1_entries[: args.search_limit] if search_enabled(args) else []

    meta = run_meta(args, run_name=run_name, model_name=gen_config.model_name,
                    adapter_path=gen_config.adapter_path, prompt_mode=prompt_mode,
                    few_shot_source=few_shot_source, repetition_penalty=gen_config.repetition_penalty)
    log = ResultsLog(args.results_jsonl or RESULTS_DIR / f"{run_name}.jsonl", meta, args.resume)
    deadline = Deadline(args.max_minutes)

    print(f"\npass@1 (greedy) on {len(pass1_entries)} prompts")
    completed = run_pass_at_1(pass1_entries, args, generator, log, deadline, example_for)
    if search_entries and completed:
        print(f"\nsearch (best-of-{args.num_candidates} + {args.repair_rounds} repair) "
              f"on {len(search_entries)} prompts, seed {args.seed}")
        completed = run_search_phase(search_entries, args, generator, log, deadline, example_for,
                                     base_temperature=gen_config.temperature)
    if not completed:
        print(f"\nStopped at --max-minutes={args.max_minutes}; writing a partial report. "
              "Rerun with --resume to continue.")

    pass1_ids = {e["id"] for e in pass1_entries}
    pass1_rows = [r for r in log.phase_rows("pass@1") if r["id"] in pass1_ids]
    report = {
        "run_name": run_name,
        "model_name": gen_config.model_name,
        "adapter_path": gen_config.adapter_path,
        "device": args.device,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_mode": prompt_mode,
        "few_shot_source": few_shot_source,  # None for zero-shot
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "repetition_penalty": gen_config.repetition_penalty,
        "generation_settings": _generation_settings(gen_config, generator, 0.0),
        "partial": len(pass1_rows) < len(pass1_entries) or not completed,
        "prompts_planned": len(pass1_entries),
        "prompts_scored": len(pass1_rows),
        "results_jsonl": str(log.path),
        "per_category": {c: m.to_dict() for c, m in aggregate_rows(pass1_rows).items() if m.total > 0},
        "outcomes": _count_outcomes(pass1_rows),
        "hit_token_limit": sum(1 for r in pass1_rows if r.get("hit_token_limit")),
        "explanation_rate_pct": (sum(r["has_explanation"] for r in pass1_rows) / len(pass1_rows) * 100
                                 if pass1_rows else None),
        "dry_run": args.dry_run,
        "split": args.split,
        "skip_verification": args.skip_verification,
        "interpret": args.interpret,
        "backend": generator.backend if generator is not None else None,
        **(extra or {}),
        "results": pass1_rows,
    }
    if search_entries:
        search_ids = {e["id"] for e in search_entries}
        search_rows = [r for r in log.phase_rows("search") if r["id"] in search_ids]
        temps = SearchConfig(args.num_candidates, args.repair_rounds, gen_config.temperature)
        report["search"] = {
            "num_candidates": args.num_candidates,
            "repair_rounds": args.repair_rounds,
            "search_limit": args.search_limit,
            "prompts_scored": len(search_rows),
            "prompts_planned": len(search_entries),
            "seed": args.seed,
            "round0_temperature": temps.temperature(args.num_candidates),
            "repair_temperature": temps.temperature(1),
            "generation_settings": {
                "round0": _generation_settings(gen_config, generator, temps.temperature(args.num_candidates),
                                               args.num_candidates),
                "repair": _generation_settings(gen_config, generator, temps.temperature(1)),
            },
            "per_category": {c: m.to_dict() for c, m in aggregate_rows(search_rows).items() if m.total > 0},
            "outcomes": _count_outcomes(search_rows),
            "results": search_rows,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{run_name}_{_timestamp()}.json"
    n = 1
    while path.exists():  # a partial and a resumed report can land in the same second
        n += 1
        path = RESULTS_DIR / f"{run_name}_{_timestamp()}_{n}.json"
    path.write_text(json.dumps(report, indent=2))

    print_metrics_table(aggregate_rows(pass1_rows), title=f"{run_name}: pass@1 ({prompt_mode}, greedy)")
    print(f"Outcomes: {report['outcomes']}")
    if report["hit_token_limit"]:
        print(f"WARNING: {report['hit_token_limit']} generations hit max_new_tokens={args.max_new_tokens}.")
    if "search" in report:
        print_metrics_table(aggregate_rows(report["search"]["results"]),
                            title=f"{run_name}: best-of-{args.num_candidates} + {args.repair_rounds} repair "
                                  f"({report['search']['prompts_scored']} prompts)")
    print(f"\n{'PARTIAL ' if report['partial'] else ''}Report saved to {path}")
    return path, report


def _count_outcomes(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    return counts


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
