"""Tests for local_baseline_eval.py pipeline fixes.

Covers:
- _build_report classifies NameError as 'name_error' (not 'runtime_error')
- _build_report classifies SyntaxError variants as 'syntax_error'
- _build_report still classifies other runtime errors as 'runtime_error'
- _verify_in_docker returns raw_text and extracted_kernel in result dict
  (checked via parse_model_output + a fake verification path)
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.local_baseline_eval import _build_report


def _make_record(error, verify_status: str = "error") -> dict:
    return {
        "id": "test__smoke",
        "category": "norm",
        "tokens_per_second": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 400,
        "parse_status": "ok",
        "verify_status": verify_status,
        "correct": False,
        "error": error,
    }


def test_name_error_classified_as_name_error():
    err = (
        'Traceback (most recent call last):\n'
        '  File "candidate_module.py", line 1, in <module>\n'
        '    @triton.jit\n'
        "NameError: name 'triton' is not defined"
    )
    report = _build_report([_make_record(err)])
    assert report["failure_types"] == {"name_error": 1}, (
        f"Expected name_error, got {report['failure_types']}"
    )


def test_syntax_error_classified_as_syntax_error():
    err = (
        'Traceback (most recent call last):\n'
        '  File "candidate_module.py", line 5\n'
        "SyntaxError: '(' was never closed"
    )
    report = _build_report([_make_record(err)])
    assert report["failure_types"] == {"syntax_error": 1}


def test_syntax_error_eq_variant_classified_as_syntax_error():
    err = (
        'Traceback (most recent call last):\n'
        '  File "candidate_module.py", line 10\n'
        "SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '='?"
    )
    report = _build_report([_make_record(err)])
    assert report["failure_types"] == {"syntax_error": 1}


def test_other_runtime_error_classified_as_runtime_error():
    err = (
        'Traceback (most recent call last):\n'
        '  File "candidate_module.py", line 20\n'
        "RuntimeError: CUDA error: device-side assert triggered"
    )
    report = _build_report([_make_record(err)])
    assert report["failure_types"] == {"runtime_error": 1}


def test_name_error_not_miscounted_as_runtime_error():
    """Before the fix, NameError fell into the runtime_error bucket.
    Verify the fix: a mix of NameError and SyntaxError produces the right counts.
    """
    name_err = (
        'Traceback (most recent call last):\n'
        "NameError: name 'triton' is not defined"
    )
    syntax_err = (
        'Traceback (most recent call last):\n'
        "SyntaxError: '(' was never closed"
    )
    records = [_make_record(name_err)] * 3 + [_make_record(syntax_err)] * 2
    report = _build_report(records)
    assert report["failure_types"]["name_error"] == 3
    assert report["failure_types"]["syntax_error"] == 2
    assert "runtime_error" not in report["failure_types"]


def test_verify_result_includes_raw_text_and_extracted_kernel():
    """_verify_in_docker returns raw_text and extracted_kernel in the result.

    We test this without Docker by calling parse_model_output directly and
    verifying the dict keys are present in the result of a no-kernel input
    (the early-return path that skips Docker entirely).
    """
    from scripts.local_baseline_eval import _verify_in_docker

    # An entry dict with pytorch_reference (required by _verify_in_docker)
    entry = {"id": "test__smoke", "pytorch_reference": "class Model: pass"}

    # generated_text with no kernel section → early return before Docker
    result = _verify_in_docker(entry, "just some text with no kernel section")
    assert "raw_text" in result, "raw_text missing from _verify_in_docker result"
    assert "extracted_kernel" in result, "extracted_kernel missing from _verify_in_docker result"
    assert result["parse_status"] == "failed"
    assert result["raw_text"] == "just some text with no kernel section"
    assert result["extracted_kernel"] is None
