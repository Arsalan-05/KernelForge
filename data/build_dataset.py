"""
Generates the full Op -> Kernel dataset (data/SCHEMA.md) from the template
library in data/templates/, runs every generated entry through
verify_model_kernel(), and keeps only the ones that pass in
data/verified/dataset.jsonl. Entries that fail (wrong output, error,
timeout, crash) go to data/raw/rejected.jsonl with the failure attached, so
they're inspectable rather than silently dropped.

Real verification (correctness + speedup) needs Linux + an NVIDIA GPU:

    python data/build_dataset.py --device cuda
    python data/build_dataset.py --device cuda --resume          # continue a killed run
    python data/build_dataset.py --device cuda --max-minutes 90  # stop cleanly, write a partial report

Correctness-only check of every template on any machine with Docker, via
Triton's CPU interpreter (no timings; writes only data/raw/interpret_report.json,
never the dataset files):

    docker run --rm -v "$PWD":/work kernelforge-verify \\
        python data/build_dataset.py --interpret --smoke --jobs 8

Memory and robustness (why the run loop looks the way it does):
  * Every verification runs in its own subprocess (verification/verify_kernel.py),
    so all of its GPU/CPU memory goes back to the OS when that process exits.
    Nothing accumulates between jobs, so there is no cache to clear in this process.
  * CUDA runs are strictly one job at a time: two kernels sharing the GPU would
    corrupt each other's timings, and serial runs leave the whole 16 GB to one job.
  * An out-of-memory result is kept separate from "incorrect" and retried once,
    alone, after the main pass (in interpreter mode the first try may have shared
    RAM with --jobs other workers). What "passed" means is unchanged.
  * Each result is appended to data/raw/progress_<mode>.jsonl the moment it
    finishes. --resume skips IDs already recorded there, so a dead session only
    loses the jobs that were in flight.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.templates import GENERATORS, SMOKE_GENERATORS
from kernelforge.dataset import template_of
from verification.verify_kernel import VerifyResult, verify_model_kernel

VERIFIED_PATH = Path(__file__).parent / "verified" / "dataset.jsonl"
REJECTED_PATH = Path(__file__).parent / "raw" / "rejected.jsonl"
REPORT_PATH = Path(__file__).parent / "raw" / "generation_report.json"
INTERPRET_REPORT_PATH = Path(__file__).parent / "raw" / "interpret_report.json"
RAW_DIR = Path(__file__).parent / "raw"


def _template_name(entry_id: str) -> str:
    return entry_id.split("__")[1]


# --------------------------------------------------------------------------- #
# Out-of-memory classification
# --------------------------------------------------------------------------- #

# Substrings (lower-cased) that mean "ran out of memory", not "computed the wrong thing":
#   "out of memory"       torch.cuda.OutOfMemoryError / "CUDA error: out of memory"
#   "outofmemoryerror"    the exception class name in a traceback
#   "memoryerror"         Python MemoryError (e.g. an RLIMIT_AS cap in the sandbox)
#   "exit code -9"        the sandbox process was SIGKILLed, which is what the
#                         Linux/Docker OOM killer does
_OOM_MARKERS = ("out of memory", "outofmemoryerror", "memoryerror", "exit code -9")


def is_oom(result: VerifyResult) -> bool:
    """True if the verification failed because it ran out of memory."""
    if result.passed:
        return False
    error = (result.error or "").lower()
    return any(marker in error for marker in _OOM_MARKERS)


def _rejection_reason(result) -> str:
    """Why an entry was rejected, so out-of-memory shapes aren't mistaken for broken templates."""
    if is_oom(result):
        return "oom"
    if result.status == "ok" and result.correct is False:
        return "incorrect"
    if result.status == "ok" and result.correct:
        return "not_passed"
    return result.status


# --------------------------------------------------------------------------- #
# Incremental progress log (one JSON line per finished verification)
# --------------------------------------------------------------------------- #

def progress_path(device: str, interpret: bool) -> Path:
    """One log per mode, so an interpreter run never resumes into a CUDA run (or vice versa)."""
    return RAW_DIR / f"progress_{'interpret' if interpret else device}.jsonl"


def load_progress(path: Path) -> dict[str, dict]:
    """Read the progress log into {entry_id: record}. A later record for the same ID
    wins (an OOM retry supersedes the first try). A half-written last line, left by
    a session killed mid-write, is skipped rather than crashing the resume."""
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            records[record["id"]] = record
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return records


def append_progress(handle, entry_id: str, result: VerifyResult, retried: bool) -> dict:
    """Write one finished verification and flush, so it survives the process dying next."""
    record = {
        "id": entry_id,
        "retried": retried,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "result": dataclasses.asdict(result),
    }
    handle.write(json.dumps(record) + "\n")
    handle.flush()
    return record


def result_from_record(record: dict) -> VerifyResult:
    return VerifyResult(**record["result"])


# --------------------------------------------------------------------------- #
# Run loop
# --------------------------------------------------------------------------- #

def _empty_stats() -> dict:
    return {"resumed": 0, "oom_retried": 0, "oom_recovered_on_retry": 0, "not_run": []}


def run_entries(
    entries: list[dict],
    verify: Callable[[dict], VerifyResult],
    progress_handle,
    jobs: int = 1,
    deadline: Optional[float] = None,
    retried: bool = False,
    on_result: Optional[Callable[[dict, VerifyResult], None]] = None,
) -> dict[str, dict]:
    """Verify `entries` with up to `jobs` workers and log each result as it finishes.

    Entries that haven't *started* by `deadline` (a time.monotonic() value) are
    left unrun; one already running is allowed to finish. Returns
    {entry_id: progress record} for the entries that ran.
    """
    def run(entry):
        if deadline is not None and time.monotonic() > deadline:
            return entry, None
        return entry, verify(entry)

    done: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        # as_completed (not pool.map) so a finished result is logged right away
        # instead of waiting behind a slower entry that was submitted earlier.
        for future in as_completed([pool.submit(run, e) for e in entries]):
            entry, result = future.result()
            if result is None:
                continue
            done[entry["id"]] = append_progress(progress_handle, entry["id"], result, retried)
            if on_result is not None:
                on_result(entry, result)
    return done


def _print_result(entry: dict, result: VerifyResult, interpret: bool) -> None:
    detail = "" if result.passed else f"  -> {(result.error or '').strip().splitlines()[-1][:200] if result.error else ''}"
    speed = "" if interpret else f" speedup={result.speedup}"
    print(f"[{entry['category']}] {entry['id']}: status={result.status} correct={result.correct}{speed}{detail}",
          flush=True)


def verify_all(
    entries: list[dict],
    verify: Callable[[dict], VerifyResult],
    progress_file: Path,
    resume: bool = False,
    jobs: int = 1,
    deadline: Optional[float] = None,
    on_result: Optional[Callable[[dict, VerifyResult], None]] = None,
) -> tuple[list[tuple[dict, VerifyResult]], dict]:
    """Main pass, then one solo retry of each OOM, all logged to `progress_file`.

    Returns ([(entry, result)] in entry order for every entry that has a result,
    stats) where stats counts resumed / OOM-retried / not-run entries.
    """
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    wanted = {e["id"] for e in entries}
    done = {k: v for k, v in load_progress(progress_file).items() if k in wanted} if resume else {}
    if not resume:
        progress_file.write_text("")  # fresh run: drop any stale log
    resumed = len(done)
    if resume:
        print(f"Resuming: {resumed}/{len(entries)} entries already recorded in {progress_file}", flush=True)

    with open(progress_file, "a") as handle:
        todo = [e for e in entries if e["id"] not in done]
        done.update(run_entries(todo, verify, handle, jobs=jobs, deadline=deadline, on_result=on_result))

        # Retry each OOM once, alone. `retried` is stored so a resumed run does
        # the retry if the first session died before reaching it, but never twice.
        oom = [e for e in entries
               if e["id"] in done and not done[e["id"]]["retried"] and is_oom(result_from_record(done[e["id"]]))]
        if oom:
            print(f"\nRetrying {len(oom)} out-of-memory entries one at a time", flush=True)
            done.update(run_entries(oom, verify, handle, jobs=1, deadline=deadline, retried=True,
                                    on_result=on_result))

    results = [(e, result_from_record(done[e["id"]])) for e in entries if e["id"] in done]
    stats = {
        "resumed": resumed,
        "oom_retried": len(oom),
        "oom_recovered_on_retry": sum(1 for e in oom if result_from_record(done[e["id"]]).passed),
        "not_run": [e["id"] for e in entries if e["id"] not in done],
    }
    return results, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and verify the KernelForge Op->Kernel dataset.")
    parser.add_argument("--device", default="cuda", help="cuda (required for a real verification run) or cpu")
    parser.add_argument("--categories", nargs="+", default=list(GENERATORS.keys()), choices=list(GENERATORS.keys()))
    parser.add_argument("--limit", type=int, default=None, help="cap entries per category, for quick smoke tests")
    parser.add_argument("--smoke", action="store_true", help="one small instance per template instead of the full grid")
    parser.add_argument("--interpret", action="store_true",
                        help="correctness-only via Triton's CPU interpreter; never writes the dataset files")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel verifications (interpreter only; GPU timing runs must stay serial)")
    parser.add_argument("--resume", action="store_true",
                        help="skip entries already recorded in data/raw/progress_<mode>.jsonl by an earlier run")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="stop starting new verifications after this long and write a partial report")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    if args.jobs > 1 and not args.interpret:
        parser.error("--jobs > 1 would corrupt GPU timings; use it with --interpret only")

    generators = SMOKE_GENERATORS if args.smoke else GENERATORS
    entries = []
    for category in args.categories:
        cat_entries = generators[category]()
        entries.extend(cat_entries[: args.limit] if args.limit else cat_entries)

    def verify(entry):
        return verify_model_kernel(
            entry["pytorch_reference"],
            entry["triton_kernel"],
            device=args.device,
            atol=entry["tolerance"]["atol"],
            rtol=entry["tolerance"]["rtol"],
            warmup=args.warmup,
            iters=args.iters,
            timeout_s=args.timeout if not args.interpret else max(args.timeout, 600.0),
            interpret=args.interpret,
        )

    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
    results, stats = verify_all(
        entries, verify, progress_path(args.device, args.interpret),
        resume=args.resume, jobs=args.jobs, deadline=deadline,
        on_result=lambda e, r: _print_result(e, r, args.interpret),
    )
    if stats["not_run"]:
        print(f"\nTime budget reached: {len(stats['not_run'])} entries not run "
              f"(rerun with --resume to finish them).", flush=True)

    if args.interpret:
        _write_interpret_report(results, args, stats)
    else:
        _write_dataset(results, args, stats)


def _write_interpret_report(results, args, stats: Optional[dict] = None) -> None:
    stats = stats or _empty_stats()
    per_template = {}
    for entry, result in results:
        per_template.setdefault(f"{entry['category']}/{_template_name(entry['id'])}", []).append({
            "id": entry["id"],
            "passed": result.passed,
            "status": result.status,
            "error": result.error,
        })
    passed = sum(1 for _, r in results if r.passed)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "triton-interpreter (correctness only, CPU; no timings)",
        "smoke": args.smoke,
        "total": len(results) + len(stats["not_run"]),
        "passed": passed,
        "partial": bool(stats["not_run"]),
        "run_stats": stats,
        "templates": per_template,
    }
    INTERPRET_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTERPRET_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n=== Interpreter check: {passed}/{report['total']} passed ===")
    for name, rows in per_template.items():
        ok = sum(r["passed"] for r in rows)
        print(f"  {'PASS' if ok == len(rows) else 'FAIL'}  {name:<42} {ok}/{len(rows)}")
    print(f"\nReport: {INTERPRET_REPORT_PATH}")
    sys.exit(0 if passed == report["total"] else 1)


def _write_dataset(results, args, stats: Optional[dict] = None) -> None:
    stats = stats or _empty_stats()
    VERIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REJECTED_PATH.parent.mkdir(parents=True, exist_ok=True)

    categories = list(dict.fromkeys(entry["category"] for entry, _ in results))
    verified_count = {c: 0 for c in categories}
    rejected_count = {c: 0 for c in categories}
    status_breakdown = {c: {} for c in categories}
    rejection_reasons = {c: {} for c in categories}
    per_template: dict[str, dict] = {}
    speedups = {c: [] for c in categories}

    with open(VERIFIED_PATH, "w") as verified_f, open(REJECTED_PATH, "w") as rejected_f:
        for entry, result in results:
            category = entry["category"]
            status_breakdown[category][result.status] = status_breakdown[category].get(result.status, 0) + 1
            entry["verification"] = {
                "verified": bool(result.passed),
                "correct": result.correct,
                "speedup": result.speedup,
                "device": args.device,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }
            template = per_template.setdefault(template_of(entry), {"verified": 0, "rejected": {}, "speedups": []})
            if result.passed:
                verified_count[category] += 1
                speedups[category].append(result.speedup)
                template["verified"] += 1
                template["speedups"].append(result.speedup)
                verified_f.write(json.dumps(entry) + "\n")
            else:
                rejected_count[category] += 1
                reason = _rejection_reason(result)
                rejection_reasons[category][reason] = rejection_reasons[category].get(reason, 0) + 1
                template["rejected"][reason] = template["rejected"].get(reason, 0) + 1
                entry["verification"]["status"] = result.status
                entry["verification"]["reason"] = reason
                entry["verification"]["error"] = result.error
                rejected_f.write(json.dumps(entry) + "\n")

    def _median(values):
        values = sorted(v for v in values if v is not None)
        return values[len(values) // 2] if values else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "smoke": args.smoke,
        "total_generated": len(results) + len(stats["not_run"]),
        # partial = the time budget ran out; not_run lists what --resume would still verify
        "partial": bool(stats["not_run"]),
        "run_stats": stats,
        "per_category": {
            c: {
                "generated": verified_count[c] + rejected_count[c],
                "verified": verified_count[c],
                "rejected": rejected_count[c],
                "median_speedup": _median(speedups[c]),
                "status_breakdown": status_breakdown[c],
                "rejection_reasons": rejection_reasons[c],
            }
            for c in categories
        },
        "per_template": {
            name: {"verified": t["verified"], "rejected": t["rejected"], "median_speedup": _median(t["speedups"])}
            for name, t in per_template.items()
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n=== Summary ===")
    print(f"{'category':<18} {'generated':>10} {'verified':>10} {'rejected':>10} {'(oom)':>8} {'median x':>10}")
    for c in categories:
        med = report["per_category"][c]["median_speedup"]
        print(f"{c:<18} {verified_count[c] + rejected_count[c]:>10} {verified_count[c]:>10} "
              f"{rejected_count[c]:>10} {rejection_reasons[c].get('oom', 0):>8} {(f'{med:.2f}' if med else '-'):>10}")
    print("\nPer template:")
    for name, t in report["per_template"].items():
        med = t["median_speedup"]
        reasons = ", ".join(f"{k}={v}" for k, v in t["rejected"].items()) or "-"
        print(f"  {name:<40} verified {t['verified']:>2}  median {(f'{med:.2f}x' if med else '-'):>7}  rejected: {reasons}")
    total_verified = sum(verified_count.values())
    print(f"{'TOTAL':<18} {len(results):>10} {total_verified:>10} {len(results) - total_verified:>10}")
    print(f"\nVerified dataset: {VERIFIED_PATH} ({total_verified} entries)")
    print(f"Rejected entries: {REJECTED_PATH} ({len(results) - total_verified} entries)")
    if stats["not_run"]:
        print(f"Not run (budget): {len(stats['not_run'])} entries; rerun with --resume")
    print(f"Full report:      {REPORT_PATH}")


if __name__ == "__main__":
    main()
