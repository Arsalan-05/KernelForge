"""
quantized_matmul templates: weight/activation-quantized GEMMs, the core
cost lever in LLM inference serving (lower precision -> more tokens/sec/GPU
for the same weight-memory footprint).

3 templates x 20 (in_features, out_features, batch_size) shape combos = 60 entries.
"""

from .common import FP16_TOLERANCE, make_entry

LINEAR_SHAPES = [
    (4096, 4096, "attn_proj"),
    (4096, 11008, "mlp_up"),
    (11008, 4096, "mlp_down"),
    (4096, 12288, "qkv_fused"),
    (8192, 8192, "attn_proj_large"),
]
BATCH_SIZES = [1, 8, 32, 128]  # 1/8 = decode-like, 32/128 = prefill-like


def _w8a16_pertensor(in_features: int, out_features: int, batch_size: int, label: str) -> dict:
    reference = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Weight-only quantized linear layer (W8A16), per-tensor scale: int8
    weights with a single global fp16 scale, dequantized and multiplied
    against a fp16 activation.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        weight_fp = torch.randn(out_features, in_features)
        scale = weight_fp.abs().max() / 127.0
        self.register_buffer(
            "weight_int8",
            torch.round(weight_fp / scale).clamp(-127, 127).to(torch.int8),
        )
        self.register_buffer("scale", scale.to(torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_int8.to(torch.float16) * self.scale
        return F.linear(x, weight)

batch_size = {batch_size}
in_features = {in_features}
out_features = {out_features}

def get_inputs():
    return [torch.randn(batch_size, in_features, dtype=torch.float16)]

def get_init_inputs():
    return [in_features, out_features]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_pertensor_matmul_kernel(
    x_ptr, w_ptr, scale_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        w_mask = (rn[:, None] < N) & (rk[None, :] + k < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_int8 = tl.load(w_ptrs, mask=w_mask, other=0)
        acc += tl.dot(x.to(tl.float32), tl.trans(w_int8.to(tl.float32)))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    scale = tl.load(scale_ptr).to(tl.float32)
    acc = acc * scale

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        weight_fp = torch.randn(out_features, in_features)
        scale = weight_fp.abs().max() / 127.0
        self.register_buffer(
            "weight_int8",
            torch.round(weight_fp / scale).clamp(-127, 127).to(torch.int8),
        )
        self.register_buffer("scale", scale.view(1).to(torch.float16))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        M, K = x.shape
        N = self.out_features
        out = torch.empty((M, N), device=x.device, dtype=torch.float16)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _w8a16_pertensor_matmul_kernel[grid](
            x, self.weight_int8, self.scale, out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight_int8.stride(0), self.weight_int8.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out
'''

    return make_entry(
        id=f"quantized_matmul__w8a16_pertensor__{label}_b{batch_size}",
        category="quantized_matmul",
        op_name="W8A16 quantized linear (per-tensor scale)",
        description=(
            f"Apply a linear layer ({in_features}->{out_features}) whose weights are "
            "int8 with a single per-tensor fp16 scale, to a fp16 activation of "
            f"batch size {batch_size}, dequantizing and matmul-ing in a fused step."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager dequantizes the full int8 weight matrix to a materialized fp16 "
            "copy before handing it to cuBLAS. The Triton kernel dequantizes each "
            "BLOCK_N x BLOCK_K weight tile inline inside the accumulation loop, so "
            "the dequantized weight never exists as a full matrix in memory -- this "
            "keeps memory traffic close to reading the int8 weights alone (~half of "
            "reading them as fp16), which is the whole point of weight-only "
            "quantization at this GEMM size (memory-bandwidth-bound on the weights)."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": label, "batch_size": batch_size, "in_features": in_features, "out_features": out_features}],
        provenance_notes=(
            "Hand-written; per-tensor variant of the w8a16 quantized linear pattern. "
            "CAVEAT: Model.__init__/ModelNew.__init__ each call torch.randn "
            "independently for the weight -- verify_model_kernel seeds identically "
            "before each construction, so this is only correct if the RNG call "
            "sequence matches exactly between the two, which it does here."
        ),
    )


def _w8a16_perchannel(in_features: int, out_features: int, batch_size: int, label: str) -> dict:
    reference = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Weight-only quantized linear layer (W8A16), per-output-channel scale:
    int8 weights with a per-row fp16 scale, dequantized and multiplied
    against a fp16 activation.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        weight_fp = torch.randn(out_features, in_features)
        scale = weight_fp.abs().amax(dim=1, keepdim=True) / 127.0
        self.register_buffer(
            "weight_int8",
            torch.round(weight_fp / scale).clamp(-127, 127).to(torch.int8),
        )
        self.register_buffer("scale", scale.to(torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_int8.to(torch.float16) * self.scale
        return F.linear(x, weight)

batch_size = {batch_size}
in_features = {in_features}
out_features = {out_features}

def get_inputs():
    return [torch.randn(batch_size, in_features, dtype=torch.float16)]

def get_init_inputs():
    return [in_features, out_features]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_perchannel_matmul_kernel(
    x_ptr, w_ptr, scale_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        w_mask = (rn[:, None] < N) & (rk[None, :] + k < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_int8 = tl.load(w_ptrs, mask=w_mask, other=0)
        acc += tl.dot(x.to(tl.float32), tl.trans(w_int8.to(tl.float32)))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    scale = tl.load(scale_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc = acc * scale[None, :]

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        weight_fp = torch.randn(out_features, in_features)
        scale = weight_fp.abs().amax(dim=1, keepdim=True) / 127.0
        self.register_buffer(
            "weight_int8",
            torch.round(weight_fp / scale).clamp(-127, 127).to(torch.int8),
        )
        self.register_buffer("scale", scale.view(-1).to(torch.float16))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        M, K = x.shape
        N = self.out_features
        out = torch.empty((M, N), device=x.device, dtype=torch.float16)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _w8a16_perchannel_matmul_kernel[grid](
            x, self.weight_int8, self.scale, out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight_int8.stride(0), self.weight_int8.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out
'''

    return make_entry(
        id=f"quantized_matmul__w8a16_perchannel__{label}_b{batch_size}",
        category="quantized_matmul",
        op_name="W8A16 quantized linear (per-channel scale)",
        description=(
            f"Apply a linear layer ({in_features}->{out_features}) whose weights are "
            "int8 with a per-output-channel fp16 scale, to a fp16 activation of "
            f"batch size {batch_size}, dequantizing and matmul-ing in a fused step."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Same fused-dequant approach as the per-tensor variant, but with a "
            "per-output-channel scale (one fp16 value per row of the weight "
            "matrix) instead of one global scale. Per-channel scales are the more "
            "realistic choice in production weight-only quantization because they "
            "cut quantization error substantially for weight rows with outlier "
            "magnitudes, at the cost of one extra vector load per output tile -- "
            "negligible next to the weight-tile load it's fused alongside."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": label, "batch_size": batch_size, "in_features": in_features, "out_features": out_features}],
        provenance_notes=(
            "Hand-written; same random-init seeding caveat as the per-tensor "
            "variant applies (Model/ModelNew both call torch.randn for the weight "
            "independently, relying on identical pre-construction seeding)."
        ),
    )


def _w8a8_linear(in_features: int, out_features: int, batch_size: int, label: str) -> dict:
    reference = f'''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fully quantized linear layer (W8A8): int8 weights with a static
    per-tensor scale, and int8 activations quantized dynamically per call
    (per-tensor scale computed from the current input). Integer matmul
    accumulates in int32, dequantized once at the end with the combined
    scale -- the standard dynamic-activation / static-weight INT8 GEMM
    pattern used to serve models at INT8 throughput.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        weight_fp = torch.randn(out_features, in_features)
        w_scale = weight_fp.abs().max() / 127.0
        self.register_buffer(
            "weight_int8",
            torch.round(weight_fp / w_scale).clamp(-127, 127).to(torch.int8),
        )
        self.register_buffer("weight_scale", w_scale.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scale = x.abs().amax() / 127.0
        x_int8 = torch.round(x.float() / x_scale).clamp(-127, 127).to(torch.int8)
        acc = torch.matmul(x_int8.to(torch.int32), self.weight_int8.to(torch.int32).t())
        return (acc.to(torch.float32) * x_scale * self.weight_scale).to(torch.float16)

batch_size = {batch_size}
in_features = {in_features}
out_features = {out_features}

def get_inputs():
    return [torch.randn(batch_size, in_features, dtype=torch.float16)]

def get_init_inputs():
    return [in_features, out_features]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _w8a8_matmul_kernel(
    x_ptr, w_ptr, x_scale_ptr, w_scale_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        w_mask = (rn[:, None] < N) & (rk[None, :] + k < K)
        x_int8 = tl.load(x_ptrs, mask=x_mask, other=0)
        w_int8 = tl.load(w_ptrs, mask=w_mask, other=0)
        # Accumulated in fp32 here for portability across Triton/GPU versions;
        # a production kernel would issue this as a native int8 tl.dot with an
        # int32 accumulator to also win on tensor-core integer throughput.
        acc += tl.dot(x_int8.to(tl.float32), tl.trans(w_int8.to(tl.float32)))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    x_scale = tl.load(x_scale_ptr).to(tl.float32)
    w_scale = tl.load(w_scale_ptr).to(tl.float32)
    acc = acc * x_scale * w_scale

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        weight_fp = torch.randn(out_features, in_features)
        w_scale = weight_fp.abs().max() / 127.0
        self.register_buffer(
            "weight_int8",
            torch.round(weight_fp / w_scale).clamp(-127, 127).to(torch.int8),
        )
        self.register_buffer("weight_scale", w_scale.view(1).to(torch.float32))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        M, K = x.shape
        N = self.out_features

        x_scale = (x.abs().amax() / 127.0).view(1)
        x_int8 = torch.round(x.float() / x_scale).clamp(-127, 127).to(torch.int8)

        out = torch.empty((M, N), device=x.device, dtype=torch.float16)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _w8a8_matmul_kernel[grid](
            x_int8, self.weight_int8, x_scale, self.weight_scale, out,
            M, N, K,
            x_int8.stride(0), x_int8.stride(1),
            self.weight_int8.stride(0), self.weight_int8.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out
'''

    return make_entry(
        id=f"quantized_matmul__w8a8__{label}_b{batch_size}",
        category="quantized_matmul",
        op_name="W8A8 quantized linear (dynamic activation + static weight)",
        description=(
            f"Apply a fully int8 linear layer ({in_features}->{out_features}): "
            "statically quantized int8 weights and dynamically per-call "
            f"quantized int8 activations of batch size {batch_size}, integer "
            "matmul accumulated in int32/fp32, dequantized once at the end."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager quantizes the activation to int8 in one pass, then runs a "
            "separate (simulated) integer matmul, then a separate dequant pass -- "
            "three kernel launches and, worse, a materialized int8 activation "
            "written to and re-read from global memory. The Triton version still "
            "quantizes the activation up front (that reduction can't be fused into "
            "the GEMM itself), but fuses the integer matmul and the final combined "
            "dequant (x_scale * weight_scale) into one launch, removing the "
            "separate dequant pass and its full-tensor memory round-trip. NOTE: "
            "this accumulates in fp32 rather than native int32 tensor-core MMA for "
            "portability -- a production W8A8 kernel would use int8 tl.dot with an "
            "int32 accumulator to also capture the compute-side throughput win, "
            "not just the fusion win captured here."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": label, "batch_size": batch_size, "in_features": in_features, "out_features": out_features}],
        provenance_notes=(
            "Hand-written. Same per-construction-seeding caveat as the other two "
            "quantized_matmul templates."
        ),
    )


def generate() -> list:
    entries = []
    for in_f, out_f, label in LINEAR_SHAPES:
        for batch in BATCH_SIZES:
            entries.append(_w8a16_pertensor(in_f, out_f, batch, label))
            entries.append(_w8a16_perchannel(in_f, out_f, batch, label))
            entries.append(_w8a8_linear(in_f, out_f, batch, label))
    return entries
