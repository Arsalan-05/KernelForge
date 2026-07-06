"""
Exercises the harness mechanics (subprocess sandboxing, correctness check,
benchmark timing, timeout/crash handling) entirely on CPU with plain
PyTorch functions standing in for kernels. This does not require Triton or
a GPU, so it runs anywhere — including this macOS dev machine.

Real Triton kernels (examples/kernels/) still need to be verified on a
Linux + NVIDIA box (Kaggle/Colab); see examples/kernels/README.md.
"""

from verification.verify_kernel import TensorSpec, verify_kernel

REFERENCE_ADD = """
import torch

def solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
"""

CANDIDATE_CORRECT = """
import torch

def solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.add(x, y)
"""

CANDIDATE_WRONG = """
import torch

def solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y
"""

CANDIDATE_RAISES = """
import torch

def solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    raise RuntimeError("boom")
"""

CANDIDATE_MISSING_ENTRY_POINT = """
import torch

def not_solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
"""

CANDIDATE_HANGS = """
import time
import torch

def solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    time.sleep(5)
    return x + y
"""

INPUT_SPEC = [TensorSpec(shape=[256]), TensorSpec(shape=[256])]


def test_correct_candidate_passes():
    result = verify_kernel(
        REFERENCE_ADD, CANDIDATE_CORRECT, INPUT_SPEC,
        device="cpu", warmup=1, iters=3,
    )
    assert result.status == "ok"
    assert result.correct is True
    assert result.passed is True
    assert result.speedup is not None and result.speedup > 0


def test_incorrect_candidate_fails_correctness():
    result = verify_kernel(
        REFERENCE_ADD, CANDIDATE_WRONG, INPUT_SPEC,
        device="cpu", warmup=1, iters=3,
    )
    assert result.status == "ok"
    assert result.correct is False
    assert result.passed is False


def test_candidate_that_raises_is_reported_as_error():
    result = verify_kernel(
        REFERENCE_ADD, CANDIDATE_RAISES, INPUT_SPEC,
        device="cpu", warmup=1, iters=3,
    )
    assert result.status == "error"
    assert "boom" in result.error
    assert result.passed is False


def test_candidate_missing_entry_point_is_reported_as_error():
    result = verify_kernel(
        REFERENCE_ADD, CANDIDATE_MISSING_ENTRY_POINT, INPUT_SPEC,
        device="cpu", warmup=1, iters=3,
    )
    assert result.status == "error"
    assert "solution" in result.error


def test_candidate_that_hangs_times_out():
    result = verify_kernel(
        REFERENCE_ADD, CANDIDATE_HANGS, INPUT_SPEC,
        device="cpu", warmup=0, iters=1, timeout_s=1.0,
    )
    assert result.status == "timeout"
    assert result.passed is False


def test_identical_seed_yields_identical_inputs_across_processes():
    # reference and candidate are executed in separate subprocess calls to
    # _build_inputs; same seed must reproduce the same tensors for the
    # correctness check to mean anything.
    result = verify_kernel(
        REFERENCE_ADD, CANDIDATE_CORRECT, INPUT_SPEC,
        device="cpu", seed=42, warmup=0, iters=1,
    )
    assert result.correct is True
