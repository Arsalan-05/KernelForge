"""Tests for scripts/check_split_leakage.py.

No GPU needed. No real results are produced here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from kernelforge.dataset import load_template_candidates, make_split
from scripts.check_split_leakage import check_leakage


@pytest.fixture(scope="module")
def candidates():
    return load_template_candidates()


def test_template_split_has_no_leakage(candidates):
    train, val, test = make_split(candidates, "template")
    errors = check_leakage(train, val, test)
    assert errors == [], f"Leakage detected: {errors}"


def test_check_leakage_detects_id_overlap():
    """check_leakage must catch an entry that appears in both train and test."""
    entry = {"id": "norm__rmsnorm__r128_h256", "category": "norm",
             "triton_kernel": "k1", "optimization_explanation": "e1",
             "template": "norm__rmsnorm"}
    # Poison: same entry in train and test
    errors = check_leakage(train=[entry], val=[], test=[entry])
    assert any("LEAK" in e and "id" in e.lower() for e in errors)


def test_check_leakage_detects_template_overlap():
    """check_leakage must catch a template that appears on both sides."""
    entry_train = {"id": "norm__rmsnorm__r128_h256", "category": "norm",
                   "triton_kernel": "k1", "optimization_explanation": "e1"}
    entry_test = {"id": "norm__rmsnorm__r256_h512", "category": "norm",
                  "triton_kernel": "k2", "optimization_explanation": "e1"}
    # Both have template "norm__rmsnorm"
    errors = check_leakage(train=[entry_train], val=[], test=[entry_test])
    assert any("LEAK" in e and "template" in e.lower() for e in errors)


def test_check_leakage_detects_kernel_source_overlap():
    """Same kernel source (same logic, different id) must be flagged."""
    entry_train = {"id": "norm__rmsnorm__r128_h256", "category": "norm",
                   "triton_kernel": "SAME_KERNEL", "optimization_explanation": "e1"}
    entry_test = {"id": "norm__layernorm__r128_h256", "category": "norm",
                  "triton_kernel": "SAME_KERNEL", "optimization_explanation": "e2"}
    errors = check_leakage(train=[entry_train], val=[], test=[entry_test])
    assert any("kernel" in e.lower() for e in errors)


def test_check_leakage_flags_explanation_verbatim_overlap():
    """If the same explanation appears verbatim on both sides, it must be flagged."""
    shared_explanation = "This is a shared explanation"
    entry_train = {"id": "norm__rmsnorm__r128_h256", "category": "norm",
                   "triton_kernel": "k1", "optimization_explanation": shared_explanation}
    entry_test = {"id": "norm__layernorm__r128_h256", "category": "norm",
                  "triton_kernel": "k2", "optimization_explanation": shared_explanation}
    errors = check_leakage(train=[entry_train], val=[], test=[entry_test])
    assert any("EXPLANATION" in e for e in errors)


def test_check_leakage_no_false_positives_on_clean_split(candidates):
    """The real template split must produce no errors."""
    train, val, test = make_split(candidates, "template")
    assert check_leakage(train, val, test) == []


def test_check_leakage_flags_missing_category_in_test():
    """If a category has no held-out template in test, it must be reported."""
    entry = {"id": "norm__rmsnorm__r128_h256", "category": "norm",
             "triton_kernel": "k1", "optimization_explanation": "e1"}
    # Only 'norm' in test, others missing
    errors = check_leakage(train=[], val=[], test=[entry])
    # Should complain about categories with 0 held-out templates
    category_errors = [e for e in errors if "STRUCTURE" in e and "template" in e.lower()]
    assert len(category_errors) > 0
