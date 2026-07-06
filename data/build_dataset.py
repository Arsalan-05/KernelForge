"""
Generates the full Op -> Kernel dataset (data/SCHEMA.md) from the template
library in data/templates/, runs every generated entry through
verify_model_kernel(), and keeps only the ones that pass in
data/verified/dataset.jsonl. Entries that fail (wrong output, error,
timeout, crash) go to data/raw/rejected.jsonl with the failure attached, so
they're inspectable rather than silently dropped.

Requires a Linux + NVIDIA GPU (Triton has no macOS/Windows or CPU backend):

    python data/build_dataset.py --device cuda

Running with --device cpu (the only option on this machine) will report
every entry as `status: "error"` ("no CUDA device available") rather than a
real correctness/speed verdict -- useful only to confirm the pipeline runs
end-to-end, not to produce a real pass/fail count. See DOCUMENTATION.md.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.templates import GENERATORS
from verification.verify_kernel import verify_model_kernel

VERIFIED_PATH = Path(__file__).parent / "verified" / "dataset.jsonl"
REJECTED_PATH = Path(__file__).parent / "raw" / "rejected.jsonl"
REPORT_PATH = Path(__file__).parent / "raw" / "generation_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and verify the KernelForge Op->Kernel dataset.")
    parser.add_argument("--device", default="cuda", help="cuda (required for a real verification run) or cpu")
    parser.add_argument("--categories", nargs="+", default=list(GENERATORS.keys()), choices=list(GENERATORS.keys()))
    parser.add_argument("--limit", type=int, default=None, help="cap entries per category, for quick smoke tests")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    VERIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REJECTED_PATH.parent.mkdir(parents=True, exist_ok=True)

    verified_count = {c: 0 for c in args.categories}
    rejected_count = {c: 0 for c in args.categories}
    status_breakdown = {c: {} for c in args.categories}
    total_generated = 0

    with open(VERIFIED_PATH, "w") as verified_f, open(REJECTED_PATH, "w") as rejected_f:
        for category in args.categories:
            entries = GENERATORS[category]()
            if args.limit:
                entries = entries[: args.limit]
            total_generated += len(entries)

            for entry in entries:
                result = verify_model_kernel(
                    entry["pytorch_reference"],
                    entry["triton_kernel"],
                    device=args.device,
                    atol=entry["tolerance"]["atol"],
                    rtol=entry["tolerance"]["rtol"],
                    warmup=args.warmup,
                    iters=args.iters,
                    timeout_s=args.timeout,
                )

                status_breakdown[category][result.status] = status_breakdown[category].get(result.status, 0) + 1

                entry["verification"] = {
                    "verified": bool(result.passed),
                    "correct": result.correct,
                    "speedup": result.speedup,
                    "device": args.device,
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                }

                if result.passed:
                    verified_count[category] += 1
                    verified_f.write(json.dumps(entry) + "\n")
                else:
                    rejected_count[category] += 1
                    entry["verification"]["status"] = result.status
                    entry["verification"]["error"] = result.error
                    rejected_f.write(json.dumps(entry) + "\n")

                print(
                    f"[{category}] {entry['id']}: "
                    f"status={result.status} correct={result.correct} speedup={result.speedup}"
                )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "total_generated": total_generated,
        "per_category": {
            c: {
                "generated": verified_count[c] + rejected_count[c],
                "verified": verified_count[c],
                "rejected": rejected_count[c],
                "status_breakdown": status_breakdown[c],
            }
            for c in args.categories
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n=== Summary ===")
    print(f"{'category':<18} {'generated':>10} {'verified':>10} {'rejected':>10}")
    for c in args.categories:
        print(f"{c:<18} {verified_count[c] + rejected_count[c]:>10} {verified_count[c]:>10} {rejected_count[c]:>10}")
    total_verified = sum(verified_count.values())
    total_rejected = sum(rejected_count.values())
    print(f"{'TOTAL':<18} {total_generated:>10} {total_verified:>10} {total_rejected:>10}")
    print(f"\nVerified dataset: {VERIFIED_PATH} ({total_verified} entries)")
    print(f"Rejected entries: {REJECTED_PATH} ({total_rejected} entries)")
    print(f"Full report:      {REPORT_PATH}")


if __name__ == "__main__":
    main()
