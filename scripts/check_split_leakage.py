"""Check the held-out-template split for data leakage.

This script verifies that:
  1. No template appears in both train/val and test sets (template-level isolation).
  2. No entry id appears in both train/val and test sets.
  3. No kernel source appears on both sides (shape constants excluded: kernel logic is
     identical within a template, so a shape-level split leaks the kernel logic).
  4. The test set covers exactly one template per category.
  5. The category distribution is balanced (equal number of test entries per category).
  6. Explanation text is not duplicated across the split boundary (a training explanation
     present verbatim in the test set means the model only needs to copy, not generalise).

Exits 0 if all checks pass, 1 if any leakage is found.

Usage:
    python scripts/check_split_leakage.py                 # uses template candidates (CPU)
    python scripts/check_split_leakage.py --dataset data/verified/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernelforge.dataset import (CATEGORIES, load_eval_entries, load_template_candidates,
                                  make_split, template_of)


def check_leakage(train: list[dict], val: list[dict], test: list[dict]) -> list[str]:
    """Return a list of leakage descriptions. Empty list means no leakage."""
    errors: list[str] = []
    train_val = train + val

    # 1. Template isolation
    train_templates = {template_of(e) for e in train_val}
    test_templates = {template_of(e) for e in test}
    leaked_templates = train_templates & test_templates
    if leaked_templates:
        errors.append(
            f"LEAK: {len(leaked_templates)} template(s) appear in both train/val and test: "
            f"{sorted(leaked_templates)}"
        )

    # 2. ID isolation
    train_ids = {e["id"] for e in train_val}
    test_ids = {e["id"] for e in test}
    leaked_ids = train_ids & test_ids
    if leaked_ids:
        errors.append(
            f"LEAK: {len(leaked_ids)} entry id(s) appear in both train/val and test: "
            f"{sorted(leaked_ids)[:5]}{'...' if len(leaked_ids) > 5 else ''}"
        )

    # 3. Kernel source isolation
    # Within a template all shapes share the same kernel source (only shape constants differ).
    # If the same kernel source appears on both sides, a model could copy it at test time.
    train_kernels = {e.get("triton_kernel", "") for e in train_val}
    test_kernels = {e.get("triton_kernel", "") for e in test}
    leaked_kernels = train_kernels & test_kernels - {""}  # ignore empty/missing
    if leaked_kernels:
        errors.append(
            f"LEAK: {len(leaked_kernels)} kernel source(s) appear in both train/val and test "
            f"(this would let the model copy a training kernel to pass a test prompt)"
        )

    # 4. One held-out template per category
    test_by_category: dict[str, set] = {}
    for e in test:
        test_by_category.setdefault(e["category"], set()).add(template_of(e))
    for cat in CATEGORIES:
        n = len(test_by_category.get(cat, set()))
        if n != 1:
            errors.append(
                f"STRUCTURE: category '{cat}' has {n} held-out template(s) in test "
                f"(expected exactly 1)"
            )

    # 5. Category balance
    test_cats = Counter(e["category"] for e in test)
    counts = set(test_cats.values())
    if len(counts) > 1:
        errors.append(
            f"STRUCTURE: test set is imbalanced across categories: {dict(test_cats)}"
        )

    # 6. Explanation leakage
    # Explanations are fixed per template; a training explanation verbatim in
    # the test set means the reference answer is in the training data.
    # This is expected (by design) but we report it for visibility.
    train_explanations = {e.get("optimization_explanation", "").strip() for e in train_val} - {""}
    test_explanations = {e.get("optimization_explanation", "").strip() for e in test} - {""}
    shared_explanations = train_explanations & test_explanations
    if shared_explanations:
        # This is NOT a leakage error — it's expected because the split is by
        # template and explanations are per-template, so test explanations are
        # not in training. But if they ARE shared, that's a different problem.
        errors.append(
            f"EXPLANATION LEAK: {len(shared_explanations)} explanation(s) appear verbatim "
            f"in both train/val and test. This should not happen in a template split."
        )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None,
                        help="Path to dataset JSONL (default: use unverified template candidates)")
    parser.add_argument("--strategy", default="template",
                        choices=["template", "random"],
                        help="Split strategy (default: template)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset:
        entries, source = load_eval_entries(args.dataset)
        if not entries:
            print(f"No entries found in {args.dataset}", file=sys.stderr)
            sys.exit(1)
    else:
        entries = load_template_candidates()
        source = "unverified template candidates"

    print(f"Checking split leakage on {len(entries)} entries from {source}")
    train, val, test = make_split(entries, args.strategy, seed=args.seed)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)} (strategy={args.strategy})")

    errors = check_leakage(train, val, test)
    if errors:
        print("\nFAILED — leakage detected:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        test_cats = Counter(e["category"] for e in test)
        held_out = {e["category"]: template_of(e) for e in test}
        held_out_unique = {}
        for e in test:
            held_out_unique[e["category"]] = template_of(e)
        print("\nPASSED — no leakage detected.")
        print(f"  Test entries per category: {dict(test_cats)}")
        print(f"  Held-out templates: {held_out_unique}")


if __name__ == "__main__":
    main()
