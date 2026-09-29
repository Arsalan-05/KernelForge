"""
quantized_matmul templates: weight/activation-quantized GEMMs, the core
cost lever in LLM inference serving (lower precision -> more tokens/sec/GPU
for the same weight-memory footprint).

All three kernels dequantize int8 tiles to fp16 in registers and feed
tensor-core tl.dot with an fp32 accumulator. The W8A16 kernels apply the
scale to each weight tile before the dot -- the same fp16 rounding the
reference's `weight_int8.to(fp16) * scale` performs -- so the only numeric
difference from eager is accumulation order.

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

_LAUNCH = '''
def _block_m(M):
    # tl.dot needs >= 16 rows; decode-sized batches shouldn't pay for 64.
    return 16 if M <= 16 else (32 if M <= 32 else 64)
'''


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
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn  # W^T tile: (BLOCK_K, BLOCK_N)
    scale = tl.load(scale_ptr)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (rk + k0) < K
        x = tl.load(x_ptrs, mask=(rm[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & (rn[None, :] < N), other=0)
        w = w.to(tl.float16) * scale
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))
''' + _LAUNCH + '''

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
        M, K = x.shape
        N = self.out_features
        out = torch.empty((M, N), device=x.device, dtype=torch.float16)
        BLOCK_M, BLOCK_N, BLOCK_K = _block_m(M), 64, 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _w8a16_pertensor_matmul_kernel[grid](
            x, self.weight_int8, self.scale, out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight_int8.stride(0), self.weight_int8.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
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
            "Eager dequantizes the whole int8 weight matrix into a materialized fp16 "
            "copy (one full read of the int8 weights, one full fp16 write, then cuBLAS "
            "reads the fp16 copy again). The Triton kernel loads each int8 weight tile "
            "once, dequantizes it to fp16 in registers, and feeds it straight into a "
            "tensor-core tl.dot, so HBM traffic on the weights is ~1 byte per element "
            "instead of ~5 -- the win that matters because at serving batch sizes this "
            "GEMM is bound by weight bandwidth, not FLOPs. BLOCK_M shrinks to 16 for "
            "decode-sized batches so a batch of 1 doesn't pay for 64 padded rows."
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
    n_mask = rn < N

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn  # W^T tile: (BLOCK_K, BLOCK_N)
    scale = tl.load(scale_ptr + rn, mask=n_mask, other=0.0)  # one fp16 scale per output channel

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (rk + k0) < K
        x = tl.load(x_ptrs, mask=(rm[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        w = w.to(tl.float16) * scale[None, :]
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16), mask=(rm[:, None] < M) & n_mask[None, :])
''' + _LAUNCH + '''

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
        M, K = x.shape
        N = self.out_features
        out = torch.empty((M, N), device=x.device, dtype=torch.float16)
        BLOCK_M, BLOCK_N, BLOCK_K = _block_m(M), 64, 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _w8a16_perchannel_matmul_kernel[grid](
            x, self.weight_int8, self.scale, out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight_int8.stride(0), self.weight_int8.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
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
            "Same fused dequant-into-tensor-core-GEMM structure as the per-tensor "
            "variant, with one fp16 scale per output channel. The BLOCK_N scales for a "
            "program's output columns are loaded once before the K loop and broadcast "
            "over every weight tile, so per-channel quantization (much lower error "
            "for weight rows with outliers, which is why production uses it) costs "
            "one extra vector load per program rather than per tile."
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
    (per-tensor scale computed from the current input), dequantized once at
    the end with the combined scale -- the dynamic-activation / static-weight
    INT8 GEMM pattern used to serve models at INT8 throughput.
    torch.matmul has no int32 CUDA kernel, so the int8 products are
    accumulated in fp32 (each product is exact).
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
        acc = torch.matmul(x_int8.float(), self.weight_int8.float().t())
        return (acc * x_scale * self.weight_scale).to(torch.float16)

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
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn  # W^T tile: (BLOCK_K, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (rk + k0) < K
        x_q = tl.load(x_ptrs, mask=(rm[:, None] < M) & k_mask[None, :], other=0)
        w_q = tl.load(w_ptrs, mask=k_mask[:, None] & (rn[None, :] < N), other=0)
        # int8 values are exact in fp16, and fp16 products are exact in the fp32
        # accumulator, so this runs on fp16 tensor cores on every GPU generation.
        acc += tl.dot(x_q.to(tl.float16), w_q.to(tl.float16))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    x_scale = tl.load(x_scale_ptr).to(tl.float32)
    w_scale = tl.load(w_scale_ptr)
    acc = acc * x_scale * w_scale

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))
''' + _LAUNCH + '''

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
        M, K = x.shape
        N = self.out_features

        x_scale = (x.abs().amax() / 127.0).view(1)
        x_int8 = torch.round(x.float() / x_scale).clamp(-127, 127).to(torch.int8)

        out = torch.empty((M, N), device=x.device, dtype=torch.float16)
        BLOCK_M, BLOCK_N, BLOCK_K = _block_m(M), 64, 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _w8a8_matmul_kernel[grid](
            x_int8, self.weight_int8, x_scale, self.weight_scale, out,
            M, N, K,
            x_int8.stride(0), x_int8.stride(1),
            self.weight_int8.stride(0), self.weight_int8.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
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
            f"quantized int8 activations of batch size {batch_size}, "
            "accumulated in fp32 and dequantized once at the end."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager upcasts both int8 operands to full fp32 copies (4x the int8 bytes, "
            "written and re-read), runs an fp32 GEMM, then a separate dequant pass. "
            "The Triton kernel reads the int8 activation and weight tiles directly, "
            "widens them to fp16 in registers (exact for int8), accumulates on fp16 "
            "tensor cores into fp32, and applies the combined x_scale * w_scale in the "
            "epilogue, so the GEMM and the dequant are one launch and nothing wider "
            "than int8 is ever written. The activation amax + quantize still run "
            "before the GEMM, since a global reduction can't be fused into it. On "
            "sm80+ a native int8 tl.dot with an int32 accumulator would add the "
            "integer tensor-core throughput win on top."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": label, "batch_size": batch_size, "in_features": in_features, "out_features": out_features}],
        provenance_notes=(
            "Hand-written. Same per-construction-seeding caveat as the other two "
            "quantized_matmul templates. The reference accumulates in fp32 because "
            "torch.matmul on int32 tensors is not implemented for CUDA (the original "
            "template's integer matmul failed on GPU)."
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


def generate_smoke() -> list:
    """One small, non-multiple-of-block instance per template, for interpreter checks."""
    return [
        _w8a16_pertensor(96, 80, 5, "smoke"),
        _w8a16_perchannel(96, 80, 5, "smoke"),
        _w8a8_linear(96, 80, 5, "smoke"),
    ]
