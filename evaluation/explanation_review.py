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

from kernelforge.dataset import load_eval_entries
from kernelforge.parsing import parse_model_output


def _find_entry_by_id(entry_id: str, entries: list[dict]) -> dict | None:
    for entry in entries:
        if entry["id"] == entry_id:
            return entry
    return None


def _generated_explanation(result: dict) -> str | None:
    """The model's explanation from one report row.

    Newer reports store it directly. Older ones only kept a 500-char preview,
    which usually ends inside the kernel, so the explanation is missing there.
    """
    if "generated_explanation" in result:
        return result["generated_explanation"]
    text = result.get("generated_text") or result.get("generated_text_preview")
    if not text:
        return None
    parsed = parse_model_output(text)
    return parsed.optimization_explanation if parsed.has_kernel else None


def summarize_review(review: dict) -> dict:
    """Count rubric scores in a filled-in review file."""
    scores = [item.get("rubric_score") for item in review["items"]]
    scored = [s for s in scores if s in (0, 1, 2)]
    by_category: dict[str, list[int]] = {}
    for item in review["items"]:
        if item.get("rubric_score") in (0, 1, 2):
            by_category.setdefault(item["category"], []).append(item["rubric_score"])
    return {
        "scored": len(scored),
        "unscored": len(scores) - len(scored),
        "score_2": scored.count(2),
        "score_1": scored.count(1),
        "score_0": scored.count(0),
        "mean_score": sum(scored) / len(scored) if scored else None,
        "mean_by_category": {c: sum(v) / len(v) for c, v in sorted(by_category.items())},
    }


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
        generated_explanation = _generated_explanation(r)

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
    parser.add_argument("--report", type=Path, help="Path to baseline or finetuned eval report JSON")
    parser.add_argument("--summarize", type=Path, default=None,
                        help="Filled-in review JSON: count the rubric scores and write them into its summary")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Write review template JSON here")
    args = parser.parse_args()

    if args.summarize:
        review = json.loads(args.summarize.read_text())
        review["summary"] = summarize_review(review)
        args.summarize.write_text(json.dumps(review, indent=2))
        print(json.dumps(review["summary"], indent=2))
        return
    if not args.report:
        parser.error("--report is required unless --summarize is given")

    report = json.loads(args.report.read_text())
    entries, _ = load_eval_entries(args.dataset, dry_run=bool(report.get("dry_run")))
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
        print(f"Edit rubric_score fields (0/1/2), then run: python evaluation/explanation_review.py --summarize {args.output}")


if __name__ == "__main__":
    main()
