"""
Task 3: tests for the degenerate-output (NaN/fp16) guard.

is_degenerate_output() lives in kernelforge/parsing.py.
The guard is wired into evaluation/common.py:evaluate_generated_kernel()
and scripts/local_baseline_eval.py:_verify_in_docker().
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernelforge.parsing import is_degenerate_output


# ---------------------------------------------------------------------------
# is_degenerate_output unit tests
# ---------------------------------------------------------------------------

class TestIsDegenerateOutput:

    def test_all_bang_chars_is_degenerate(self):
        assert is_degenerate_output("!" * 300)

    def test_majority_bang_tail_is_degenerate(self):
        # 200 normal chars followed by 300 bangs — tail is all bangs
        text = "import triton\n" * 15 + "!" * 300
        assert is_degenerate_output(text)

    def test_valid_kernel_is_not_degenerate(self):
        kernel = (
            "import torch\nimport triton\nimport triton.language as tl\n\n"
            "@triton.jit\ndef _kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):\n"
            "    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
            "    mask = idx < n\n"
            "    x = tl.load(x_ptr + idx, mask=mask)\n"
            "    tl.store(out_ptr + idx, x * 2, mask=mask)\n\n"
            "class ModelNew(torch.nn.Module):\n"
            "    def forward(self, x): return x\n"
        )
        assert not is_degenerate_output(kernel)

    def test_empty_string_is_not_degenerate(self):
        assert not is_degenerate_output("")

    def test_single_char_string_is_not_degenerate(self):
        assert not is_degenerate_output("x")

    def test_exactly_at_threshold_is_not_degenerate(self):
        # Exactly 50% — threshold is *strictly* greater than 0.5
        tail = "!" * 150 + "x" * 150
        assert not is_degenerate_output(tail)

    def test_just_above_threshold_is_degenerate(self):
        # 151 out of 300 = 50.3% — just above 50%
        tail = "!" * 151 + "x" * 149
        assert is_degenerate_output(tail)

    def test_prose_ending_in_period_is_not_degenerate(self):
        text = "This kernel uses shared memory. " * 10
        assert not is_degenerate_output(text)

    def test_newline_heavy_code_is_not_degenerate(self):
        # Code with lots of newlines is not degenerate
        text = "\n" * 160 + "    pass\n" * 20
        # 160 newlines out of 300 tail chars = 53% — would be flagged
        # Real code doesn't look like this, but test with actual code
        text = "    x = tl.load(ptr + offs, mask=mask)\n" * 8
        assert not is_degenerate_output(text)

    def test_custom_threshold(self):
        # 70% threshold — 60% single char should not trigger 70% threshold
        tail = "!" * 180 + "x" * 120  # 60% bangs
        assert not is_degenerate_output(tail, threshold=0.70)
        assert is_degenerate_output(tail, threshold=0.50)

    def test_custom_tail_chars(self):
        # Only look at last 10 chars; if those are clean, not degenerate
        text = "!" * 500 + "import triton\n"
        assert not is_degenerate_output(text, tail_chars=20)

    def test_other_repeated_characters(self):
        # Degeneration can appear as any single character, not only '!'
        assert is_degenerate_output("0" * 300)
        assert is_degenerate_output(" " * 300)


# ---------------------------------------------------------------------------
# Integration: outcome_of and evaluate_generated_kernel
# ---------------------------------------------------------------------------

def test_outcome_of_flags_degenerate_error():
    from evaluation.common import outcome_of
    result = outcome_of(
        parsed_kernel=False,
        verification=None,
        error="degenerate_output: repeated single character >50% of tail",
    )
    assert result == "degenerate_output"


def test_outcome_of_degenerate_takes_priority_over_no_kernel():
    """degenerate_output is reported even when parsed_kernel is False."""
    from evaluation.common import outcome_of
    result = outcome_of(
        parsed_kernel=False,
        verification=None,
        error="degenerate_output: ...",
    )
    assert result == "degenerate_output"


def test_evaluate_generated_kernel_degenerate_skips_verification():
    """evaluate_generated_kernel returns degenerate_output without calling Docker."""
    from evaluation.common import evaluate_generated_kernel

    degenerate_text = "!" * 400
    # Use a minimal entry dict (verification is skipped, so pytorch_reference is never used)
    entry = {
        "id": "test__degenerate",
        "category": "norm",
        "pytorch_reference": "import torch\nclass Model(torch.nn.Module): pass",
        "tolerance": {"atol": 1e-2, "rtol": 1e-2},
    }
    result = evaluate_generated_kernel(entry, degenerate_text, skip_verification=True)
    assert result.outcome == "degenerate_output"
    assert result.error is not None and "degenerate_output" in result.error
    assert result.verification is None


def test_evaluate_generated_kernel_valid_text_not_flagged():
    """Valid kernel text is not flagged as degenerate."""
    from evaluation.common import evaluate_generated_kernel

    valid_text = (
        "kernel:\nimport torch\nimport triton\nimport triton.language as tl\n"
        "@triton.jit\ndef k(x, BLOCK: tl.constexpr): pass\n"
        "class ModelNew(torch.nn.Module):\n    def forward(self, x): return x\n\n"
        "explanation:\nFuses memory access.\n"
    )
    entry = {
        "id": "test__valid",
        "category": "norm",
        "pytorch_reference": "import torch\nclass Model(torch.nn.Module): pass",
        "tolerance": {"atol": 1e-2, "rtol": 1e-2},
    }
    result = evaluate_generated_kernel(entry, valid_text, skip_verification=True)
    assert result.outcome != "degenerate_output"


# ---------------------------------------------------------------------------
# local_baseline_eval _build_report degenerate bucket
# ---------------------------------------------------------------------------

def test_local_baseline_report_counts_degenerate():
    from scripts.local_baseline_eval import _build_report

    records = [
        {
            "id": "a", "category": "norm",
            "parse_status": "degenerate", "verify_status": None,
            "correct": False, "error": "degenerate_output: ...",
            "tokens_per_second": 10.0,
        },
        {
            "id": "b", "category": "norm",
            "parse_status": "ok", "verify_status": "error",
            "correct": False, "error": "NameError: name 'triton' is not defined",
            "tokens_per_second": 11.0,
        },
    ]
    report = _build_report(records)
    assert report["failure_types"].get("degenerate_output") == 1
    assert report["failure_types"].get("name_error") == 1
