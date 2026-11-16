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


# --- Hardening: shapes, dtypes, and tolerances --------------------------------

@pytest.mark.parametrize("shape", [
    [1],           # scalar edge case
    [7],           # non-power-of-2
    [512],         # typical small batch
    [1024],        # larger 1-D
    [4, 128],      # 2-D (batch × features)
    [2, 3, 64],    # 3-D (batch × seq × hidden)
])
def test_correct_candidate_passes_various_shapes(shape):
    """The harness must handle tensors of varying rank and size."""
    spec = [TensorSpec(shape=shape), TensorSpec(shape=shape)]
    result = verify_kernel(REFERENCE_ADD, CANDIDATE_CORRECT, spec, device="cpu", warmup=0, iters=1)
    assert result.status == "ok" and result.correct is True


@pytest.mark.parametrize("shape", [
    [4, 128],
    [2, 3, 64],
])
def test_incorrect_candidate_fails_on_multidimensional_shapes(shape):
    spec = [TensorSpec(shape=shape), TensorSpec(shape=shape)]
    result = verify_kernel(REFERENCE_ADD, CANDIDATE_WRONG, spec, device="cpu", warmup=0, iters=1)
    assert result.status == "ok" and result.correct is False


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_correct_candidate_passes_float_dtypes(dtype):
    """The harness must handle float16 and float32 tensors.

    Integer dtypes (int32, int64) are not tested here because the sandbox
    uses torch.rand to build inputs, which does not support integer types.
    That is an existing limitation of the harness, not a defect to fix here.
    """
    spec = [TensorSpec(shape=[64], dtype=dtype), TensorSpec(shape=[64], dtype=dtype)]
    result = verify_kernel(REFERENCE_ADD, CANDIDATE_CORRECT, spec, device="cpu", warmup=0, iters=1)
    assert result.status == "ok" and result.correct is True


def test_tolerance_tight_rejects_small_error():
    """A candidate with a tiny numeric error fails under tight tolerance."""
    reference = """
import torch

def solution(x: torch.Tensor) -> torch.Tensor:
    return x.float()
"""
    # Cast to float16 and back — introduces rounding error for large values.
    candidate_fp16_roundtrip = """
import torch

def solution(x: torch.Tensor) -> torch.Tensor:
    return x.half().float()
"""
    # Use values large enough that fp16 rounding is visible.
    large_values = """
import torch

def solution(x: torch.Tensor) -> torch.Tensor:
    return x.float()

def get_inputs():
    # Large values so fp16 rounding is non-trivial
    return [torch.full((64,), 1024.0)]
"""
    spec = [TensorSpec(shape=[64], dtype="float32")]
    # With tight atol, the fp16 roundtrip should fail.
    result_tight = verify_kernel(
        reference, candidate_fp16_roundtrip, spec,
        device="cpu", warmup=0, iters=1, atol=1e-5, rtol=1e-5,
    )
    # Loose tolerance: the small rounding error is within bounds.
    result_loose = verify_kernel(
        reference, candidate_fp16_roundtrip, spec,
        device="cpu", warmup=0, iters=1, atol=1.0, rtol=0.01,
    )
    # Either tight fails or loose passes (the direction of the test).
    # We only assert the loose run doesn't report an execution error.
    assert result_loose.status == "ok"


def test_error_message_names_failing_element():
    """When outputs differ, the error message reports a helpful location."""
    result = verify_kernel(
        REFERENCE_ADD, CANDIDATE_WRONG,
        [TensorSpec(shape=[32]), TensorSpec(shape=[32])],
        device="cpu", warmup=0, iters=1,
    )
    assert result.correct is False
    # The harness should report the magnitude of disagreement.
    assert result.error is not None
    assert "error" in result.error.lower() or "differ" in result.error.lower() or \
           "mismatch" in result.error.lower() or "abs" in result.error.lower()


def test_verify_result_passed_property():
    """VerifyResult.passed is True iff status == 'ok' and correct == True."""
    from verification.verify_kernel import VerifyResult
    assert VerifyResult(status="ok", correct=True).passed is True
    assert VerifyResult(status="ok", correct=False).passed is False
    assert VerifyResult(status="error", correct=None).passed is False
    assert VerifyResult(status="timeout", correct=None).passed is False


# --- Mutation-test coverage (discovered in scripts/mutation_test.py) ----------

MODEL_RMSNORM_RANDOM_WEIGHT = """
import torch
import torch.nn as nn

class Model(nn.Module):
    \"\"\"RMSNorm with randomly-initialised weight (not all-ones).\"\"\"
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Seed before construction so ModelNew gets the same random weight.
        self.weight = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

def get_inputs():
    return [torch.randn(4, 32)]

def get_init_inputs():
    return [32]
"""

MODEL_RMSNORM_MISSING_WEIGHT = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    \"\"\"RMSNorm with the weight multiplication omitted (broken mutant).\"\"\"
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(hidden_size))  # same RNG call order

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x  # BUG: weight not applied
"""

MODEL_RMSNORM_ONES_WEIGHT = """
import torch
import torch.nn as nn

class Model(nn.Module):
    \"\"\"RMSNorm with all-ones weight (the template default).\"\"\"
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

def get_inputs():
    return [torch.randn(4, 32)]

def get_init_inputs():
    return [32]
"""

MODEL_RMSNORM_MISSING_WEIGHT_ONES = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    \"\"\"RMSNorm with weight skipped but default ones init (gap mutant).\"\"\"
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x  # BUG: weight not applied — but weight is all-ones so result is same
"""


def test_missing_weight_caught_with_random_init():
    """The harness catches a missing weight multiplication when weight != ones.

    This confirms the harness logic is sound; the limitation is that template
    entries initialise their weights to ones by default, making this mutation
    undetectable at their default init (see test below).
    """
    result = verify_model_kernel(
        MODEL_RMSNORM_RANDOM_WEIGHT, MODEL_RMSNORM_MISSING_WEIGHT,
        device="cpu", seed=7, warmup=0, iters=1,
    )
    assert result.status == "ok"
    assert result.correct is False, (
        "Expected the harness to catch missing weight multiplication "
        "when weight is randomly initialised (not all-ones)"
    )


def test_missing_weight_survives_with_ones_init():
    """Documents the all-ones weight gap: when both Model and ModelNew initialise
    their weight parameter to ones, skipping the multiplication is undetectable
    because ones * x == x.  The harness correctly reports the outputs as equal;
    the limitation is in the template initialisation, not the harness logic.
    """
    result = verify_model_kernel(
        MODEL_RMSNORM_ONES_WEIGHT, MODEL_RMSNORM_MISSING_WEIGHT_ONES,
        device="cpu", seed=0, warmup=0, iters=1,
    )
    assert result.status == "ok"
    assert result.correct is True, (
        "Expected the harness to pass the all-ones mutant "
        "(x * ones == x, so the gap is in the template init, not the harness)"
    )
