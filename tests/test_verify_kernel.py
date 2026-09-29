"""
Exercises the harness mechanics (subprocess sandboxing, correctness check,
benchmark timing, timeout/crash handling) entirely on CPU with plain
PyTorch functions standing in for kernels. This does not require Triton or
a GPU, so it runs anywhere — including this macOS dev machine.

Real Triton kernels (examples/kernels/) still need to be verified on a
Linux + NVIDIA box (Kaggle/Colab); see examples/kernels/README.md. The
interpreter-mode test runs only where Triton is installed (e.g. the
docker/verify.Dockerfile image).
"""

import pytest

from verification.verify_kernel import TensorSpec, verify_kernel, verify_model_kernel

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


# --- Model / ModelNew contract (the one dataset entries actually use) ------

MODEL_REFERENCE_ADD = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

def get_inputs():
    return [torch.rand(8), torch.rand(8)]

def get_init_inputs():
    return [8]
"""

MODEL_CANDIDATE_CORRECT = """
import torch

class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y)
"""

MODEL_CANDIDATE_WRONG = """
import torch

class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x - y
"""

MODEL_CANDIDATE_MISSING_CLASS = """
import torch

class NotModelNew(torch.nn.Module):
    def forward(self, x, y):
        return x + y
"""

MODEL_REFERENCE_RANDOM_WEIGHT = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight

def get_inputs():
    return [torch.rand(16)]

def get_init_inputs():
    return [16]
"""

MODEL_CANDIDATE_RANDOM_WEIGHT_SAME_CALL_ORDER = """
import torch

class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight
"""

MODEL_REFERENCE_TUPLE_OUTPUT = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y, x - y

def get_inputs():
    return [torch.rand(8), torch.rand(8)]

def get_init_inputs():
    return [8]
"""

MODEL_CANDIDATE_TUPLE_OUTPUT = """
import torch

class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.add(x, y), torch.sub(x, y)
"""


def test_model_contract_correct_candidate_passes():
    result = verify_model_kernel(
        MODEL_REFERENCE_ADD, MODEL_CANDIDATE_CORRECT,
        device="cpu", warmup=1, iters=3,
    )
    assert result.status == "ok"
    assert result.correct is True
    assert result.passed is True


def test_model_contract_incorrect_candidate_fails_correctness():
    result = verify_model_kernel(
        MODEL_REFERENCE_ADD, MODEL_CANDIDATE_WRONG,
        device="cpu", warmup=1, iters=3,
    )
    assert result.status == "ok"
    assert result.correct is False


def test_model_contract_missing_modelnew_is_reported_as_error():
    result = verify_model_kernel(
        MODEL_REFERENCE_ADD, MODEL_CANDIDATE_MISSING_CLASS,
        device="cpu", warmup=1, iters=3,
    )
    assert result.status == "error"
    assert "ModelNew" in result.error


def test_model_contract_seeds_random_init_identically():
    # Model and ModelNew each independently call torch.randn in __init__;
    # seeding immediately before each construction must make them match.
    result = verify_model_kernel(
        MODEL_REFERENCE_RANDOM_WEIGHT, MODEL_CANDIDATE_RANDOM_WEIGHT_SAME_CALL_ORDER,
        device="cpu", seed=123, warmup=1, iters=3,
    )
    assert result.status == "ok"
    assert result.correct is True


def test_model_contract_handles_tuple_outputs():
    result = verify_model_kernel(
        MODEL_REFERENCE_TUPLE_OUTPUT, MODEL_CANDIDATE_TUPLE_OUTPUT,
        device="cpu", warmup=1, iters=3,
    )
    assert result.status == "ok"
    assert result.correct is True


def test_model_contract_reports_where_outputs_differ():
    result = verify_model_kernel(
        MODEL_REFERENCE_ADD, MODEL_CANDIDATE_WRONG,
        device="cpu", warmup=1, iters=3,
    )
    assert result.correct is False
    assert "max abs error" in result.error


MODEL_CANDIDATE_TRITON_ADD = """
import torch
import triton
import triton.language as tl

@triton.jit
def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x, y):
        out = torch.empty_like(x)
        _add[(triton.cdiv(x.numel(), 4),)](x, y, out, x.numel(), BLOCK=4)
        return out
"""


def test_interpreter_mode_checks_a_triton_kernel_on_cpu():
    pytest.importorskip("triton")
    result = verify_model_kernel(
        MODEL_REFERENCE_ADD, MODEL_CANDIDATE_TRITON_ADD, interpret=True, timeout_s=120,
    )
    assert result.status == "ok", result.error
    assert result.passed and result.interpreted
    assert result.speedup is None  # interpreter timings would be meaningless
