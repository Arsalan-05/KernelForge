"""Phase 5: rubric review (0/1/2) of generated explanations, optionally blind.

Explanation quality has no clean automatic metric, so a person scores a sample.

    # 1. build: the same sampled entries from each report; --blind hides which model wrote each one
    python evaluation/explanation_review.py --report baseline_X.json finetuned_Y.json \
        --output review.json --blind            # also writes review.key.json; don't open it yet
    # 2. score: shows each sample next to the reference, asks for 0/1/2, saves after every answer
    python evaluation/explanation_review.py --score review.json
    # 3. summarize: counts scores, per model (joined from the key) and per category
    python evaluation/explanation_review.py --summarize review.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernelforge.dataset import load_eval_entries
from kernelforge.parsing import parse_model_output

RUBRIC = {
    "2": "correctly identifies the specific optimization technique",
    "1": "partially correct or generic but directionally right",
    "0": "wrong, missing, or pure boilerplate",
}


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


def _source(path: Path, report: dict) -> dict:
    """Which model produced a sample; hidden in the key file when blind."""
    return {
        "report": path.name,
        "run_name": report.get("run_name"),
        "model_name": report.get("model_name"),
        "adapter_path": report.get("adapter_path"),
        "prompt_mode": report.get("prompt_mode", "zero_shot"),
    }


def _label(source: dict) -> str:
    return f"{source['run_name']} ({source['prompt_mode']})"


def key_path_for(review_path: Path) -> Path:
    return review_path.with_name(f"{review_path.stem}.key.json")


def build_review(reports: list[tuple[Path, dict]], entries: list[dict], sample_size: int, seed: int,
                 blind: bool) -> tuple[dict, dict | None]:
    """Sample the same entry ids from every report (pass@1 rows) and pair with references.

    Returns (review, key). With `blind`, items are shuffled across reports and
    carry no model information; `key` maps sample_id -> source.
    """
    rows_by_report = [{row["id"]: row for row in report.get("results", [])} for _, report in reports]
    references = {e["id"]: e for e in entries}
    common_ids = sorted(set.intersection(*(set(rows) for rows in rows_by_report)) & set(references))
    if not common_ids:
        raise SystemExit("The reports share no scored entries with a known reference explanation.")
    rng = random.Random(seed)
    sampled = rng.sample(common_ids, min(sample_size, len(common_ids)))

    items = []
    for entry_id in sampled:
        for (path, report), rows in zip(reports, rows_by_report):
            items.append({
                "entry_id": entry_id,
                "category": rows[entry_id]["category"],
                "reference_explanation": references[entry_id]["optimization_explanation"],
                "generated_explanation": _generated_explanation(rows[entry_id]),
                "rubric_score": None,
                "reviewer_notes": "",
                "_source": _source(path, report),
            })
    if blind:
        rng.shuffle(items)  # separates the two models' answers for the same entry

    key = {} if blind else None
    for n, item in enumerate(items, 1):
        item["sample_id"] = f"s{n:03d}"
        source = item.pop("_source")
        if blind:
            key[item["sample_id"]] = source
        else:
            item["source"] = source
    review = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blind": blind,
        "num_reports": len(reports),
        "sample_entries": len(sampled),
        "rubric": RUBRIC,
        "items": items,
    }
    return review, key


def score_review(path: Path, ask: Callable[[str], str] = input) -> None:
    """Ask for a score per unscored item; saves after every answer, so it can be resumed."""
    review = json.loads(path.read_text())
    todo = [item for item in review["items"] if item.get("rubric_score") not in (0, 1, 2)]
    print(f"{len(todo)} of {len(review['items'])} items left. Rubric: "
          + "; ".join(f"{k} = {v}" for k, v in sorted(RUBRIC.items(), reverse=True)))
    for item in todo:
        print(f"\n--- {item['sample_id']} ({item['category']}) ---\nREFERENCE:\n{item['reference_explanation']}")
        print(f"\nGENERATED:\n{item['generated_explanation'] or '(none parsed)'}\n")
        while True:
            answer = ask("Score 0/1/2 (s = skip, q = save and quit): ").strip().lower()
            if answer in ("0", "1", "2", "s", "q"):
                break
        if answer == "q":
            break
        if answer in ("0", "1", "2"):
            item["rubric_score"] = int(answer)
            item["reviewer_notes"] = ask("Notes (optional, Enter to skip): ").strip()
            path.write_text(json.dumps(review, indent=2))
    path.write_text(json.dumps(review, indent=2))


def _counts(scores: list[int]) -> dict:
    return {
        "scored": len(scores),
        "score_2": scores.count(2),
        "score_1": scores.count(1),
        "score_0": scores.count(0),
        "mean_score": sum(scores) / len(scores) if scores else None,
    }


def summarize_review(review: dict, key: dict | None = None) -> dict:
    """Count rubric scores overall, per category, and per model (from item sources or the key)."""
    def source_of(item):
        return item.get("source") or (key or {}).get(item["sample_id"])

    scored = [item for item in review["items"] if item.get("rubric_score") in (0, 1, 2)]
    by_category: dict[str, list[int]] = {}
    by_model: dict[str, list[int]] = {}
    by_model_category: dict[str, dict[str, list[int]]] = {}
    for item in scored:
        by_category.setdefault(item["category"], []).append(item["rubric_score"])
        source = source_of(item)
        if source:
            label = _label(source)
            by_model.setdefault(label, []).append(item["rubric_score"])
            by_model_category.setdefault(label, {}).setdefault(item["category"], []).append(item["rubric_score"])
    per_model = {}
    for label, scores in sorted(by_model.items()):
        categories = by_model_category[label]
        per_model[label] = {**_counts(scores),
                            "mean_by_category": {c: sum(s) / len(s) for c, s in sorted(categories.items())}}
    return {
        **_counts([item["rubric_score"] for item in scored]),
        "unscored": len(review["items"]) - len(scored),
        "mean_by_category": {c: sum(v) / len(v) for c, v in sorted(by_category.items())},
        "by_model": per_model,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Explanation-quality rubric review of eval reports.")
    parser.add_argument("--report", type=Path, nargs="+", help="One or more baseline/finetuned eval report JSONs")
    parser.add_argument("--output", type=Path, default=None, help="Review file to write (build step)")
    parser.add_argument("--blind", action="store_true",
                        help="Shuffle samples and hide which model wrote each; the answer key goes to <output>.key.json")
    parser.add_argument("--score", type=Path, default=None, help="Review file to score interactively")
    parser.add_argument("--summarize", type=Path, default=None, help="Scored review file to summarize")
    parser.add_argument("--key", type=Path, default=None, help="Answer key for --summarize (default: <review>.key.json)")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=25, help="Entries to sample (each scored once per report)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.score:
        score_review(args.score)
        print(f"\nSaved. Next: python evaluation/explanation_review.py --summarize {args.score}")
        return
    if args.summarize:
        review = json.loads(args.summarize.read_text())
        key_file = args.key or key_path_for(args.summarize)
        key = json.loads(key_file.read_text()) if key_file.exists() else None
        if review.get("blind") and key is None:
            parser.error(f"blind review: answer key {key_file} not found (pass --key)")
        review["summary"] = summarize_review(review, key)
        args.summarize.write_text(json.dumps(review, indent=2))
        print(json.dumps(review["summary"], indent=2))
        return
    if not args.report or not args.output:
        parser.error("building a review needs --report and --output (or use --score / --summarize)")

    reports = [(path, json.loads(path.read_text())) for path in args.report]
    entries, _ = load_eval_entries(args.dataset, dry_run=any(r.get("dry_run") for _, r in reports))
    review, key = build_review(reports, entries, args.sample, args.seed, args.blind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, indent=2))
    print(f"Review with {len(review['items'])} samples saved to {args.output}")
    if key is not None:
        key_path_for(args.output).write_text(json.dumps(key, indent=2))
        print(f"Answer key saved to {key_path_for(args.output)}; don't open it until scoring is done.")
    print(f"Next: python evaluation/explanation_review.py --score {args.output}")


if __name__ == "__main__":
    main()
