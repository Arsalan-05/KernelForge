"""
rope templates: rotary positional embedding, applied to every Q/K vector at
every layer, every forward pass -- another small-but-frequent op in the
inference-serving critical path.

3 templates x 20 (batch, seq_len) shape combos = 60 entries.
"""

from .common import FP16_TOLERANCE, make_entry

BATCHES = [1, 2, 4, 8, 16]
SEQ_LENS = [256, 512, 1024, 2048]  # 5 x 4 = 20 combos per template
NUM_HEADS, HEAD_DIM = 32, 128


def _rope_rotate_half(batch: int, seq_len: int) -> dict:
    reference = f'''import torch
import torch.nn as nn

def _build_rope_cache(seq_len, head_dim, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)          # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)            # (seq_len, head_dim)
    return emb.cos(), emb.sin()

def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

class Model(nn.Module):
    """
    RoPE (rotate-half / LLaMA-style): rotates each (query or key) vector by
    a position-dependent angle, applied identically to every attention
    head at every layer, every forward pass.
    """
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: (batch, num_heads, seq_len, head_dim)
        seq_len = q.shape[2]
        cos, sin = _build_rope_cache(seq_len, self.head_dim, q.device)
        cos = cos.to(q.dtype)[None, None, :, :]
        sin = sin.to(q.dtype)[None, None, :, :]
        return q * cos + _rotate_half(q) * sin

batch_size = {batch}
num_heads = {NUM_HEADS}
seq_len = {seq_len}
head_dim = {HEAD_DIM}

def get_inputs():
    return [torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)]

def get_init_inputs():
    return [num_heads, head_dim]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _rope_rotate_half_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_xbh, stride_xm, stride_xd,
    stride_cm, stride_cd,
    half_dim,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < half_dim

    row_ptr = x_ptr + pid_bh * stride_xbh + pid_m * stride_xm
    x1 = tl.load(row_ptr + d_offsets * stride_xd, mask=d_mask, other=0.0).to(tl.float32)
    x2 = tl.load(row_ptr + (d_offsets + half_dim) * stride_xd, mask=d_mask, other=0.0).to(tl.float32)

    cos_row = cos_ptr + pid_m * stride_cm
    sin_row = sin_ptr + pid_m * stride_cm
    cos_v = tl.load(cos_row + d_offsets * stride_cd, mask=d_mask, other=0.0).to(tl.float32)
    sin_v = tl.load(sin_row + d_offsets * stride_cd, mask=d_mask, other=0.0).to(tl.float32)

    out1 = x1 * cos_v - x2 * sin_v
    out2 = x2 * cos_v + x1 * sin_v

    out_row_ptr = out_ptr + pid_bh * stride_xbh + pid_m * stride_xm
    tl.store(out_row_ptr + d_offsets * stride_xd, out1.to(tl.float16), mask=d_mask)
    tl.store(out_row_ptr + (d_offsets + half_dim) * stride_xd, out2.to(tl.float16), mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim, base=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base = base

    def _build_cache(self, seq_len, device, dtype):
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, inv_freq)  # (seq_len, half)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        assert q.is_cuda
        batch, num_heads, seq_len, head_dim = q.shape
        cos, sin = self._build_cache(seq_len, q.device, q.dtype)
        out = torch.empty_like(q)

        q_ = q.reshape(batch * num_heads, seq_len, head_dim)
        out_ = out.reshape(batch * num_heads, seq_len, head_dim)

        BLOCK_D = triton.next_power_of_2(head_dim // 2)
        grid = (batch * num_heads, seq_len)
        _rope_rotate_half_kernel[grid](
            q_, cos, sin, out_,
            q_.stride(0), q_.stride(1), q_.stride(2),
            cos.stride(0), cos.stride(1),
            head_dim // 2,
            BLOCK_D=BLOCK_D,
        )
        return out
'''

    return make_entry(
        id=f"rope__rotate_half__b{batch}_s{seq_len}",
        category="rope",
        op_name="RoPE (rotate-half / LLaMA-style)",
        description=(
            f"Apply rotate-half rotary positional embedding to a query tensor "
            f"of batch={batch}, seq_len={seq_len}, num_heads={NUM_HEADS}, "
            f"head_dim={HEAD_DIM}."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager builds rotate_half(x) as its own concatenated tensor (a full "
            "extra allocation and copy the same size as x), then runs two "
            "elementwise multiplies and an add as separate passes. The Triton "
            "kernel reads each half of the head_dim once, computes both rotated "
            "halves in registers, and writes the result once -- rotate_half's "
            "tensor never gets materialized. Applied to every Q/K vector, every "
            "head, every layer, every forward pass, so avoiding that extra "
            "allocation is a small win that repeats at very high frequency."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_heads": NUM_HEADS, "head_dim": HEAD_DIM}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def _rope_interleaved(batch: int, seq_len: int) -> dict:
    reference = f'''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    RoPE (interleaved-pair / GPT-NeoX-J-style): rotates adjacent element
    pairs (x[..., 0::2], x[..., 1::2]) rather than first-half/second-half
    pairs -- the other common RoPE layout used by several open model
    families.
    """
    def __init__(self, num_heads, head_dim, base=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base = base

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: (batch, num_heads, seq_len, head_dim)
        seq_len = q.shape[2]
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=q.device).float() / self.head_dim))
        positions = torch.arange(seq_len, device=q.device).float()
        freqs = torch.outer(positions, inv_freq)  # (seq_len, half)
        cos = freqs.cos().to(q.dtype)[None, None, :, :]
        sin = freqs.sin().to(q.dtype)[None, None, :, :]

        x_even = q[..., 0::2]
        x_odd = q[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_odd * cos + x_even * sin
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)

batch_size = {batch}
num_heads = {NUM_HEADS}
seq_len = {seq_len}
head_dim = {HEAD_DIM}

def get_inputs():
    return [torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)]

def get_init_inputs():
    return [num_heads, head_dim]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _rope_interleaved_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_xbh, stride_xm, stride_xd,
    stride_cm, stride_cd,
    half_dim,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < half_dim

    row_ptr = x_ptr + pid_bh * stride_xbh + pid_m * stride_xm
    even_ptrs = row_ptr + (2 * d_offsets) * stride_xd
    odd_ptrs = row_ptr + (2 * d_offsets + 1) * stride_xd
    x_even = tl.load(even_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    x_odd = tl.load(odd_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    cos_row = cos_ptr + pid_m * stride_cm
    sin_row = sin_ptr + pid_m * stride_cm
    cos_v = tl.load(cos_row + d_offsets * stride_cd, mask=d_mask, other=0.0).to(tl.float32)
    sin_v = tl.load(sin_row + d_offsets * stride_cd, mask=d_mask, other=0.0).to(tl.float32)

    out_even = x_even * cos_v - x_odd * sin_v
    out_odd = x_odd * cos_v + x_even * sin_v

    out_row_ptr = out_ptr + pid_bh * stride_xbh + pid_m * stride_xm
    tl.store(out_row_ptr + (2 * d_offsets) * stride_xd, out_even.to(tl.float16), mask=d_mask)
    tl.store(out_row_ptr + (2 * d_offsets + 1) * stride_xd, out_odd.to(tl.float16), mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim, base=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base = base

    def _build_cache(self, seq_len, device, dtype):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, inv_freq)  # (seq_len, half)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        assert q.is_cuda
        batch, num_heads, seq_len, head_dim = q.shape
        cos, sin = self._build_cache(seq_len, q.device, q.dtype)
        out = torch.empty_like(q)

        q_ = q.reshape(batch * num_heads, seq_len, head_dim)
        out_ = out.reshape(batch * num_heads, seq_len, head_dim)

        BLOCK_D = triton.next_power_of_2(head_dim // 2)
        grid = (batch * num_heads, seq_len)
        _rope_interleaved_kernel[grid](
            q_, cos, sin, out_,
            q_.stride(0), q_.stride(1), q_.stride(2),
            cos.stride(0), cos.stride(1),
            head_dim // 2,
            BLOCK_D=BLOCK_D,
        )
        return out
'''

    return make_entry(
        id=f"rope__interleaved__b{batch}_s{seq_len}",
        category="rope",
        op_name="RoPE (interleaved-pair / GPT-NeoX-J-style)",
        description=(
            f"Apply interleaved-pair rotary positional embedding to a query "
            f"tensor of batch={batch}, seq_len={seq_len}, num_heads={NUM_HEADS}, "
            f"head_dim={HEAD_DIM}."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager's strided slicing (`q[..., 0::2]`, `q[..., 1::2]`) followed by "
            "`stack` + `flatten` to re-interleave the result materializes three "
            "extra tensors around the actual rotation math. The Triton kernel "
            "reads even/odd elements directly via strided pointer arithmetic and "
            "writes the rotated pairs back to their original interleaved "
            "positions in one pass -- no intermediate even/odd/stacked tensors "
            "at all."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_heads": NUM_HEADS, "head_dim": HEAD_DIM}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def _rope_fused_qk(batch: int, seq_len: int) -> dict:
    reference = f'''import torch
import torch.nn as nn

def _build_rope_cache(seq_len, head_dim, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

class Model(nn.Module):
    """
    RoPE applied to both Q and K in one logical step (the standard
    `apply_rotary_pos_emb(q, k, cos, sin)` call in most transformer
    implementations), sharing one rotation table across both tensors.
    """
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: (batch, num_heads, seq_len, head_dim)
        seq_len = q.shape[2]
        cos, sin = _build_rope_cache(seq_len, self.head_dim, q.device)
        cos = cos.to(q.dtype)[None, None, :, :]
        sin = sin.to(q.dtype)[None, None, :, :]
        q_out = q * cos + _rotate_half(q) * sin
        k_out = k * cos + _rotate_half(k) * sin
        return q_out, k_out

batch_size = {batch}
num_heads = {NUM_HEADS}
seq_len = {seq_len}
head_dim = {HEAD_DIM}

def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    return [q, k]

def get_init_inputs():
    return [num_heads, head_dim]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fused_qk_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
    stride_bh, stride_m, stride_d,
    stride_cm, stride_cd,
    half_dim,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < half_dim

    cos_row = cos_ptr + pid_m * stride_cm
    sin_row = sin_ptr + pid_m * stride_cm
    cos_v = tl.load(cos_row + d_offsets * stride_cd, mask=d_mask, other=0.0).to(tl.float32)
    sin_v = tl.load(sin_row + d_offsets * stride_cd, mask=d_mask, other=0.0).to(tl.float32)

    base_offset = pid_bh * stride_bh + pid_m * stride_m

    q_row = q_ptr + base_offset
    q1 = tl.load(q_row + d_offsets * stride_d, mask=d_mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_row + (d_offsets + half_dim) * stride_d, mask=d_mask, other=0.0).to(tl.float32)
    q_out_row = q_out_ptr + base_offset
    tl.store(q_out_row + d_offsets * stride_d, (q1 * cos_v - q2 * sin_v).to(tl.float16), mask=d_mask)
    tl.store(q_out_row + (d_offsets + half_dim) * stride_d, (q2 * cos_v + q1 * sin_v).to(tl.float16), mask=d_mask)

    k_row = k_ptr + base_offset
    k1 = tl.load(k_row + d_offsets * stride_d, mask=d_mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_row + (d_offsets + half_dim) * stride_d, mask=d_mask, other=0.0).to(tl.float32)
    k_out_row = k_out_ptr + base_offset
    tl.store(k_out_row + d_offsets * stride_d, (k1 * cos_v - k2 * sin_v).to(tl.float16), mask=d_mask)
    tl.store(k_out_row + (d_offsets + half_dim) * stride_d, (k2 * cos_v + k1 * sin_v).to(tl.float16), mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim, base=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base = base

    def _build_cache(self, seq_len, device, dtype):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, inv_freq)  # (seq_len, half)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        assert q.is_cuda
        batch, num_heads, seq_len, head_dim = q.shape
        cos, sin = self._build_cache(seq_len, q.device, q.dtype)
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        q_ = q.reshape(batch * num_heads, seq_len, head_dim)
        k_ = k.reshape(batch * num_heads, seq_len, head_dim)
        q_out_ = q_out.reshape(batch * num_heads, seq_len, head_dim)
        k_out_ = k_out.reshape(batch * num_heads, seq_len, head_dim)

        BLOCK_D = triton.next_power_of_2(head_dim // 2)
        grid = (batch * num_heads, seq_len)
        _rope_fused_qk_kernel[grid](
            q_, k_, cos, sin, q_out_, k_out_,
            q_.stride(0), q_.stride(1), q_.stride(2),
            cos.stride(0), cos.stride(1),
            head_dim // 2,
            BLOCK_D=BLOCK_D,
        )
        return q_out, k_out
'''

    return make_entry(
        id=f"rope__fused_qk__b{batch}_s{seq_len}",
        category="rope",
        op_name="RoPE fused over Q and K",
        description=(
            f"Apply rotate-half RoPE to both query and key tensors in a single "
            f"fused call, for batch={batch}, seq_len={seq_len}, "
            f"num_heads={NUM_HEADS}, head_dim={HEAD_DIM}."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager's `apply_rotary_pos_emb(q, k, cos, sin)` is typically two "
            "independent rotate-half calls under the hood -- two kernel launches "
            "each materializing their own rotate_half tensor. This kernel handles "
            "both Q and K from a single launch, reusing the same loaded cos/sin "
            "values for both without a second grid dispatch. Since this call "
            "happens once per layer, per forward pass, halving the launch count "
            "for this specific op is a direct, easily-measurable win, distinct "
            "from (and additive with) the per-tensor rotate_half fusion in the "
            "single-tensor RoPE template."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_heads": NUM_HEADS, "head_dim": HEAD_DIM}],
        provenance_notes="Hand-written; no learnable random state. Two-tensor output exercises the harness's tuple-output comparison path.",
    )


def generate() -> list:
    entries = []
    for batch in BATCHES:
        for seq_len in SEQ_LENS:
            entries.append(_rope_rotate_half(batch, seq_len))
            entries.append(_rope_interleaved(batch, seq_len))
            entries.append(_rope_fused_qk(batch, seq_len))
    return entries
