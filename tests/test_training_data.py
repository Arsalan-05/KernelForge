"""
Task 1 + Task 2: training-data quality and prompt consistency checks.

Fast path: smoke entries (15 total — one per template, CPU-interpreter shapes).
Slow path: all 300 entries (marked with pytest.mark.slow; run with -m slow).
"""
from __future__ import annotations

import inspect
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.templates import GENERATORS, SMOKE_GENERATORS
from kernelforge.prompts import build_messages, format_response

REQUIRED_IMPORTS = [
    "import torch",
    "import triton",
    "import triton.language as tl",
]


# ---------------------------------------------------------------------------
# Task 1: training-data audit
# ---------------------------------------------------------------------------

class TestSmokeEntryRequiredElements:
    """Every smoke-entry triton_kernel must include imports and ModelNew."""

    def test_smoke_entries_have_required_imports(self):
        for cat, gen in SMOKE_GENERATORS.items():
            for entry in gen():
                kernel = entry["triton_kernel"]
                for imp in REQUIRED_IMPORTS:
                    assert imp in kernel, (
                        f"{entry['id']}: missing '{imp}' in triton_kernel"
                    )

    def test_smoke_entries_have_model_new(self):
        for cat, gen in SMOKE_GENERATORS.items():
            for entry in gen():
                kernel = entry["triton_kernel"]
                assert "class ModelNew" in kernel, (
                    f"{entry['id']}: missing 'class ModelNew' in triton_kernel"
                )


def test_random_sample_20_across_all_categories():
    """20 randomly-sampled entries from all 300 pass both checks."""
    all_entries = []
    for gen in GENERATORS.values():
        all_entries.extend(gen())
    sample = random.Random(42).sample(all_entries, min(20, len(all_entries)))

    failures = []
    for entry in sample:
        kernel = entry["triton_kernel"]
        missing = [i for i in REQUIRED_IMPORTS if i not in kernel]
        if missing:
            failures.append(f"{entry['id']}: missing imports {missing}")
        if "class ModelNew" not in kernel:
            failures.append(f"{entry['id']}: missing 'class ModelNew'")

    assert not failures, "\n".join(failures)


@pytest.mark.slow
def test_all_300_entries_required_elements():
    """All 300 generated entries pass both checks (slow — skipped by default)."""
    failures = []
    for gen in GENERATORS.values():
        for entry in gen():
            kernel = entry["triton_kernel"]
            for imp in REQUIRED_IMPORTS:
                if imp not in kernel:
                    failures.append(f"{entry['id']}: missing '{imp}'")
            if "class ModelNew" not in kernel:
                failures.append(f"{entry['id']}: missing 'class ModelNew'")
    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])


def test_format_response_includes_kernel_and_explanation():
    """format_response output starts with 'kernel:' and contains 'explanation:'."""
    for cat, gen in SMOKE_GENERATORS.items():
        entry = next(iter(gen()))
        resp = format_response(entry)
        assert resp.startswith("kernel:\n"), (
            f"{entry['id']}: format_response doesn't start with 'kernel:\\n'"
        )
        assert "\nexplanation:\n" in resp, (
            f"{entry['id']}: format_response missing '\\nexplanation:\\n'"
        )
        assert entry["triton_kernel"].strip() in resp, (
            f"{entry['id']}: triton_kernel not preserved in format_response"
        )
        break  # one entry is enough


# ---------------------------------------------------------------------------
# Task 2: prompt consistency
# ---------------------------------------------------------------------------

def test_training_uses_notes_false():
    """build_sft_features must call build_messages without notes=True.

    Training prompt must match the Kaggle eval prompt (both notes=False).
    The original 60-entry local baseline used notes=True, but the Kaggle
    baseline_eval and finetune both use notes=False.
    """
    from training.finetune import build_sft_features
    source = inspect.getsource(build_sft_features)
    assert "notes=True" not in source, (
        "training/finetune.py build_sft_features calls build_messages with notes=True; "
        "training prompt must match the Kaggle eval prompt (notes=False)"
    )


def test_kernel_generator_uses_notes_false():
    """KernelGenerator.generate() must call build_messages without notes=True."""
    from kernelforge.model import KernelGenerator
    source = inspect.getsource(KernelGenerator.generate)
    assert "notes=True" not in source, (
        "kernelforge/model.py KernelGenerator.generate() uses notes=True; "
        "eval prompt must match training prompt (notes=False)"
    )


def test_local_baseline_eval_uses_notes_false():
    """scripts/local_baseline_eval.py must not add notes=True to build_messages.

    The existing 60-entry numbers were collected with notes=True (TRITON_NOTES
    appended to system prompt).  Those numbers are not directly comparable to
    the Kaggle baseline.  The script now uses notes=False to match.
    """
    source = (Path(__file__).parent.parent / "scripts" / "local_baseline_eval.py").read_text()
    assert "notes=True" not in source, (
        "scripts/local_baseline_eval.py still uses notes=True; "
        "must match Kaggle baseline (notes=False)"
    )
