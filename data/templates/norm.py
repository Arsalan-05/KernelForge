"""
norm templates: normalization ops sitting on the critical path of every
transformer block. Small individually, but run twice per layer per token,
so launch/memory overhead compounds heavily at serving scale.

3 templates x 20 (rows, hidden_size) shape combos = 60 entries.
"""

from .common import FP16_TOLERANCE, make_entry

ROWS = [1024, 4096, 16384, 65536]  # batch * seq_len
HIDDEN_SIZES = [2048, 4096, 8192, 11008, 12288]  # 4 x 5 = 20 combos per template


def _rmsnorm(rows: int, hidden_size: int) -> dict:
    reference = f'''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    RMSNorm (LLaMA/Mistral-style): normalizes by root-mean-square instead
    of mean+variance (no mean-centering, no bias), then scales by a
    learned per-channel gain.
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

rows = {rows}
hidden_size = {hidden_size}

def get_inputs():
    return [torch.randn(rows, hidden_size, dtype=torch.float16)]

def get_init_inputs():
    return [hidden_size]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    row_stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_ptr = x_ptr + row_idx * row_stride
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / n_cols
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * inv_rms * weight

    out_row_ptr = out_ptr + row_idx * row_stride
    tl.store(out_row_ptr + col_offsets, y.to(tl.float16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = min(max(BLOCK_SIZE // 256, 1), 16)  # ~8 fp32 values per thread, no spills
        _rmsnorm_kernel[(n_rows,)](
            x, self.weight, out,
            x.stride(0), n_cols, self.eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
        return out
'''

    return make_entry(
        id=f"norm__rmsnorm__r{rows}_h{hidden_size}",
        category="norm",
        op_name="RMSNorm",
        description=(
            f"Apply RMSNorm with a learned per-channel gain to a "
            f"({rows}, {hidden_size}) activation tensor."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager RMSNorm runs as several kernel launches (square, mean-reduce, "
            "rsqrt, multiply, multiply-by-weight), each reading/writing the full "
            "tensor. One Triton program per row loads the row once, computes the "
            "reduction and rsqrt on-chip, and writes the result once -- one read "
            "plus one write per row instead of roughly five. This sits on the "
            "critical path of every transformer block, twice per layer. num_warps scales "
            "with the row width so even 12K-wide rows stay in registers without spilling."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "rows": rows, "hidden_size": hidden_size}],
        provenance_notes="Hand-written. Gain initialized to all-ones in both Model and ModelNew, matching real RMSNorm's default init -- no random-init seeding concern.",
    )


def _layernorm(rows: int, hidden_size: int) -> dict:
    reference = f'''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Standard LayerNorm: mean-centers, normalizes by variance, then applies
    a learned per-channel scale and bias.
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        return x_norm * self.weight + self.bias

rows = {rows}
hidden_size = {hidden_size}

def get_inputs():
    return [torch.randn(rows, hidden_size, dtype=torch.float16)]

def get_init_inputs():
    return [hidden_size]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    row_stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_ptr = x_ptr + row_idx * row_stride
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv_std * weight + bias

    out_row_ptr = out_ptr + row_idx * row_stride
    tl.store(out_row_ptr + col_offsets, y.to(tl.float16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = min(max(BLOCK_SIZE // 256, 1), 16)  # ~8 fp32 values per thread, no spills
        _layernorm_kernel[(n_rows,)](
            x, self.weight, self.bias, out,
            x.stride(0), n_cols, self.eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
        return out
'''

    return make_entry(
        id=f"norm__layernorm__r{rows}_h{hidden_size}",
        category="norm",
        op_name="LayerNorm",
        description=(
            f"Apply LayerNorm (mean-centered, with learned scale and bias) to a "
            f"({rows}, {hidden_size}) activation tensor."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Same argument as RMSNorm, with one more reduction (mean) and an "
            "extra elementwise term (bias): eager runs mean-reduce, variance-"
            "reduce, subtract, rsqrt, multiply, and add-bias as separate passes "
            "over the full tensor. The Triton version computes both reductions "
            "from a single on-chip copy of the row and writes the fully "
            "normalized-scaled-biased result once."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "rows": rows, "hidden_size": hidden_size}],
        provenance_notes="Hand-written. Weight/bias initialized to ones/zeros in both Model and ModelNew -- no random-init seeding concern.",
    )


def _rmsnorm_residual(rows: int, hidden_size: int) -> dict:
    reference = f'''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused residual-add + RMSNorm: the pre-norm transformer-block pattern
    `residual = x + residual; out = RMSNorm(residual)`, returning both the
    normalized output (fed to the next sublayer) and the updated residual
    stream (carried forward to the next block).
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        residual = x + residual
        variance = residual.pow(2).mean(-1, keepdim=True)
        normed = residual * torch.rsqrt(variance + self.eps)
        return normed * self.weight, residual

rows = {rows}
hidden_size = {hidden_size}

def get_inputs():
    x = torch.randn(rows, hidden_size, dtype=torch.float16)
    residual = torch.randn(rows, hidden_size, dtype=torch.float16)
    return [x, residual]

def get_init_inputs():
    return [hidden_size]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_residual_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr, new_residual_ptr,
    row_stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_ptr = x_ptr + row_idx * row_stride
    res_ptr = residual_ptr + row_idx * row_stride
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(res_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    residual = x + residual

    mean_sq = tl.sum(residual * residual, axis=0) / n_cols
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = residual * inv_rms * weight

    out_row_ptr = out_ptr + row_idx * row_stride
    new_res_ptr = new_residual_ptr + row_idx * row_stride
    tl.store(out_row_ptr + col_offsets, y.to(tl.float16), mask=mask)
    tl.store(new_res_ptr + col_offsets, residual.to(tl.float16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        new_residual = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = min(max(BLOCK_SIZE // 256, 1), 16)  # ~8 fp32 values per thread, no spills
        _rmsnorm_residual_kernel[(n_rows,)](
            x, residual, self.weight, out, new_residual,
            x.stride(0), n_cols, self.eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
        return out, new_residual
'''

    return make_entry(
        id=f"norm__rmsnorm_residual__r{rows}_h{hidden_size}",
        category="norm",
        op_name="Fused residual-add + RMSNorm",
        description=(
            f"Apply the pre-norm residual-add + RMSNorm pattern to a "
            f"({rows}, {hidden_size}) activation and residual stream, returning "
            "both the normalized output and the updated residual."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager computes the residual add as its own pass, then runs the same "
            "multi-launch RMSNorm on top -- two full-tensor round-trips before "
            "normalization even starts. The Triton kernel loads x and the "
            "residual once, adds them on-chip, computes the RMSNorm reduction "
            "from that same on-chip value, and writes both required outputs "
            "(normalized activation and updated residual) from a single pass -- "
            "this is exactly the fused op most serving-optimized transformer "
            "implementations use at every pre-attention and pre-MLP norm site."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "rows": rows, "hidden_size": hidden_size}],
        provenance_notes="Hand-written. Gain initialized to all-ones -- no random-init seeding concern. Two-tensor output exercises the harness's tuple-output comparison path.",
    )


def generate() -> list:
    entries = []
    for rows in ROWS:
        for hidden_size in HIDDEN_SIZES:
            entries.append(_rmsnorm(rows, hidden_size))
            entries.append(_layernorm(rows, hidden_size))
            entries.append(_rmsnorm_residual(rows, hidden_size))
    return entries


def generate_smoke() -> list:
    """One small, non-power-of-2 instance per template, for interpreter checks."""
    return [
        {**_rmsnorm(7, 300), "id": "norm__rmsnorm__smoke"},
        {**_layernorm(7, 300), "id": "norm__layernorm__smoke"},
        {**_rmsnorm_residual(7, 300), "id": "norm__rmsnorm_residual__smoke"},
    ]
