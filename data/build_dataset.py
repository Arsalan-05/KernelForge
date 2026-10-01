"""
Generates the full Op -> Kernel dataset (data/SCHEMA.md) from the template
library in data/templates/, runs every generated entry through
verify_model_kernel(), and keeps only the ones that pass in
data/verified/dataset.jsonl. Entries that fail (wrong output, error,
timeout, crash) go to data/raw/rejected.jsonl with the failure attached, so
they're inspectable rather than silently dropped.

Real verification (correctness + speedup) needs Linux + an NVIDIA GPU:

    python data/build_dataset.py --device cuda

Correctness-only check of every template on any machine with Docker, via
Triton's CPU interpreter (no timings; writes only data/raw/interpret_report.json,
never the dataset files):

    docker run --rm -v "$PWD":/work kernelforge-verify \\
        python data/build_dataset.py --interpret --smoke --jobs 8
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.templates import GENERATORS, SMOKE_GENERATORS
from kernelforge.dataset import template_of
from verification.verify_kernel import verify_model_kernel

VERIFIED_PATH = Path(__file__).parent / "verified" / "dataset.jsonl"
REJECTED_PATH = Path(__file__).parent / "raw" / "rejected.jsonl"
REPORT_PATH = Path(__file__).parent / "raw" / "generation_report.json"
INTERPRET_REPORT_PATH = Path(__file__).parent / "raw" / "interpret_report.json"


def _template_name(entry_id: str) -> str:
    return entry_id.split("__")[1]


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

    def run(entry):
        return entry, verify_model_kernel(
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

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = []
        for entry, result in pool.map(run, entries):
            results.append((entry, result))
            detail = "" if result.passed else f"  -> {(result.error or '').strip().splitlines()[-1][:200] if result.error else ''}"
            speed = "" if args.interpret else f" speedup={result.speedup}"
            print(f"[{entry['category']}] {entry['id']}: status={result.status} correct={result.correct}{speed}{detail}",
                  flush=True)

    if args.interpret:
        _write_interpret_report(results, args)
    else:
        _write_dataset(results, args)


def _write_interpret_report(results, args) -> None:
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
        "total": len(results),
        "passed": passed,
        "templates": per_template,
    }
    INTERPRET_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTERPRET_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n=== Interpreter check: {passed}/{len(results)} passed ===")
    for name, rows in per_template.items():
        ok = sum(r["passed"] for r in rows)
        print(f"  {'PASS' if ok == len(rows) else 'FAIL'}  {name:<42} {ok}/{len(rows)}")
    print(f"\nReport: {INTERPRET_REPORT_PATH}")
    sys.exit(0 if passed == len(results) else 1)


def _rejection_reason(result) -> str:
    """Why an entry was rejected, so out-of-memory shapes aren't mistaken for broken templates."""
    error = (result.error or "").lower()
    if "out of memory" in error or "outofmemoryerror" in error or "exit code -9" in error:
        return "oom"
    if result.status == "ok" and result.correct is False:
        return "incorrect"
    if result.status == "ok" and result.correct:
        return "not_passed"
    return result.status


def _write_dataset(results, args) -> None:
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
        "total_generated": len(results),
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
    print(f"Full report:      {REPORT_PATH}")


if __name__ == "__main__":
    main()
