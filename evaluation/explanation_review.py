"""Phase 5: manual rubric-based review of generated explanations.

Explanation quality has no clean automatic metric. This script samples N
outputs from an eval report, presents them alongside the reference
explanation, and collects rubric scores for honest reporting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernelforge.dataset import load_training_data
from kernelforge.parsing import parse_model_output


def _find_entry_by_id(entry_id: str, entries: list[dict]) -> dict | None:
    for entry in entries:
        if entry["id"] == entry_id:
            return entry
    return None


def build_review_items(report: dict, entries: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Sample eval results and pair with reference explanations."""
    results = report.get("results", [])
    rng = random.Random(seed)
    rng.shuffle(results)
    sampled = results[:sample_size]

    items = []
    for r in sampled:
        entry = _find_entry_by_id(r["id"], entries)
        if not entry:
            continue
        generated_explanation = None
        if "generated_text_preview" in r:
            parsed = parse_model_output(r["generated_text_preview"])
            generated_explanation = parsed.optimization_explanation

        items.append(
            {
                "id": r["id"],
                "category": r["category"],
                "reference_explanation": entry["optimization_explanation"],
                "generated_explanation": generated_explanation,
                "rubric_score": None,
                "reviewer_notes": "",
            }
        )
    return items


def print_review_template(items: list[dict]) -> None:
    """Print items for manual scoring (fill rubric_score in the output JSON)."""
    for i, item in enumerate(items, 1):
        print(f"\n--- [{i}/{len(items)}] {item['id']} ({item['category']}) ---")
        print("REFERENCE:")
        print(item["reference_explanation"])
        print("\nGENERATED:")
        print(item["generated_explanation"] or "(none parsed)")
        print("\nRubric: 2=specific+correct, 1=partial/generic, 0=wrong/missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explanation-quality rubric review from an eval report.")
    parser.add_argument("--report", type=Path, required=True, help="Path to baseline or finetuned eval report JSON")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Write review template JSON here")
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    entries = load_training_data(args.dataset)
    items = build_review_items(report, entries, args.sample, args.seed)

    print_review_template(items)

    if args.output:
        review = {
            "source_report": str(args.report),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sample_size": len(items),
            "rubric": {
                "2": "correctly identifies the specific optimization technique",
                "1": "partially correct or generic but directionally right",
                "0": "wrong, missing, or pure boilerplate",
            },
            "items": items,
            "summary": {
                "scored": 0,
                "score_2": 0,
                "score_1": 0,
                "score_0": 0,
                "notes": "Fill rubric_score for each item, then re-run with --summarize",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(review, indent=2))
        print(f"\nReview template saved to {args.output}")
        print("Edit rubric_score fields (0/1/2), then summarize manually or extend this script.")


if __name__ == "__main__":
    main()
