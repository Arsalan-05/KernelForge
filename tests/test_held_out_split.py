"""
Tests that the canonical held-out split (data/held_out_split.json) is
consistent and that no held-out ID leaks into training data.

These tests run on CPU (no GPU, no Docker) and complete in < 1 s.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SPLIT_PATH = REPO_ROOT / "data" / "held_out_split.json"


@pytest.fixture(scope="module")
def split():
    assert SPLIT_PATH.exists(), f"Split file not found: {SPLIT_PATH}"
    return json.loads(SPLIT_PATH.read_text())


def test_split_file_exists():
    assert SPLIT_PATH.exists(), f"Split file missing: {SPLIT_PATH}"


def test_split_counts(split):
    assert split["counts"]["train"] == len(split["train_ids"])
    assert split["counts"]["val"]   == len(split["val_ids"])
    assert split["counts"]["test"]  == len(split["test_ids"])


def test_no_train_test_overlap(split):
    train_set = set(split["train_ids"])
    test_set  = set(split["test_ids"])
    overlap   = train_set & test_set
    assert not overlap, f"train/test overlap ({len(overlap)} ids): {sorted(overlap)[:5]}"


def test_no_val_test_overlap(split):
    val_set  = set(split["val_ids"])
    test_set = set(split["test_ids"])
    overlap  = val_set & test_set
    assert not overlap, f"val/test overlap ({len(overlap)} ids): {sorted(overlap)[:5]}"


def test_no_train_val_overlap(split):
    train_set = set(split["train_ids"])
    val_set   = set(split["val_ids"])
    overlap   = train_set & val_set
    assert not overlap, f"train/val overlap ({len(overlap)} ids): {sorted(overlap)[:5]}"


def test_all_ids_unique(split):
    all_ids = split["train_ids"] + split["val_ids"] + split["test_ids"]
    assert len(all_ids) == len(set(all_ids)), "Duplicate IDs across splits"


def test_total_count(split):
    total = split["counts"]["train"] + split["counts"]["val"] + split["counts"]["test"]
    assert total == 300, f"Expected 300 total, got {total}"


def test_held_out_templates_are_test_only(split):
    """Every entry from a held-out template must be in test, not in train/val."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from kernelforge.dataset import template_of

    train_set = set(split["train_ids"])
    val_set   = set(split["val_ids"])
    held_templates = set(split["held_out_templates"])

    for split_name, id_list in [("train", split["train_ids"]), ("val", split["val_ids"])]:
        for entry_id in id_list:
            tmpl = template_of({"id": entry_id})
            assert tmpl not in held_templates, (
                f"Held-out template {tmpl!r} entry {entry_id!r} found in {split_name} split"
            )


def test_each_category_has_one_held_out_template(split):
    """The five held-out templates cover all five categories (one per category)."""
    cats_covered = {t.split("__")[0] for t in split["held_out_templates"]}
    expected_cats = {"quantized_matmul", "attention", "kv_cache", "norm", "rope"}
    assert cats_covered == expected_cats, (
        f"Missing categories: {expected_cats - cats_covered}"
    )
    assert len(split["held_out_templates"]) == 5, (
        f"Expected 5 held-out templates (1 per category), got {len(split['held_out_templates'])}"
    )


def test_test_ids_match_held_out_templates(split):
    """Every test ID belongs to a held-out template, and every held-out-template entry is in test."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from kernelforge.dataset import template_of

    held_templates = set(split["held_out_templates"])
    test_set = set(split["test_ids"])

    for entry_id in split["test_ids"]:
        tmpl = template_of({"id": entry_id})
        assert tmpl in held_templates, (
            f"Test entry {entry_id!r} belongs to non-held-out template {tmpl!r}"
        )

    # Each held-out template should have exactly 20 test entries
    from collections import Counter
    tmpl_counts = Counter(
        template_of({"id": eid}) for eid in split["test_ids"]
    )
    for tmpl in held_templates:
        assert tmpl_counts[tmpl] == 20, (
            f"Held-out template {tmpl!r} has {tmpl_counts[tmpl]} test entries, expected 20"
        )


def test_split_is_reproducible(split):
    """Re-running split_by_template(seed=42) produces identical IDs."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from kernelforge.dataset import split_by_template, load_template_candidates

    all_entries = load_template_candidates()
    train, val, test = split_by_template(all_entries, seed=42)
    assert sorted(e["id"] for e in train) == split["train_ids"]
    assert sorted(e["id"] for e in val)   == split["val_ids"]
    assert sorted(e["id"] for e in test)  == split["test_ids"]


def test_fp32_baseline_leakage_documented():
    """Document (but do not fail on) the leakage in fp32_baseline_eval.py.

    The existing fp32_baseline_eval.py evaluates on 60 entries: 12 per category
    (3 smoke + 9 regular). Of those 60 entries, 30 are in the train/val split
    (leakage for fine-tuned evaluation). This test verifies the leakage count
    matches what was audited on 2026-11-17 and raises a descriptive error if
    the entry selection logic changes.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from data.templates import GENERATORS, SMOKE_GENERATORS

    cats = ["quantized_matmul", "attention", "kv_cache", "norm", "rope"]
    split_data = json.loads(SPLIT_PATH.read_text())
    train_set = set(split_data["train_ids"])
    val_set   = set(split_data["val_ids"])

    eval_subset = []
    for cat in cats:
        smoke = list(SMOKE_GENERATORS[cat]())
        regular = list(GENERATORS[cat]())
        cat_entries = smoke[:]
        smoke_ids = {e["id"] for e in smoke}
        for e in regular:
            if e["id"] not in smoke_ids and len(cat_entries) < 12:
                cat_entries.append(e)
        eval_subset.extend(cat_entries[:12])

    in_trainval = sum(1 for e in eval_subset if e["id"] in train_set or e["id"] in val_set)
    # Verified on 2026-11-17: 30 of 60 eval entries are in train/val (leakage).
    # If this count changes (e.g. template logic or smoke logic is updated), this
    # test fails to alert the developer to re-audit.
    assert in_trainval == 30, (
        f"fp32_baseline leakage count changed from 30 to {in_trainval}. "
        "Re-run the leakage audit and update this test + OVERNIGHT_LOG."
    )
