"""
rope templates: rotary positional embedding, applied to every Q/K vector at
every layer, every forward pass -- another small-but-frequent op in the
inference-serving critical path.

Kernels view the (batch, heads, seq, head_dim) tensor as rows of head_dim
and process BLOCK_M rows per program (position = row % seq_len), instead of
one tiny program per row -- the largest shape here would otherwise launch
over a million programs of 64 elements each.

3 templates x 20 (batch, seq_len) shape combos = 60 entries.
"""

from .common import FP16_TOLERANCE, make_entry

BATCHES = [1, 2, 4, 8, 16]
SEQ_LENS = [256, 512, 1024, 2048]  # 5 x 4 = 20 combos per template
NUM_HEADS, HEAD_DIM = 32, 128

_CACHE = '''
def _rope_cache(seq_len, head_dim, base, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim / 2)
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()
'''


def _rope_rotate_half(batch: int, seq_len: int, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM,
                      suffix: str = "") -> dict:
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
num_heads = {num_heads}
seq_len = {seq_len}
head_dim = {head_dim}

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
    n_rows, seq_len, half_dim, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_D)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < half_dim)

    x_offs = rows[:, None] * head_dim + cols[None, :]
    cs_offs = (rows % seq_len)[:, None] * half_dim + cols[None, :]
    x1 = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + x_offs + half_dim, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + cs_offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs_offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + x_offs, (x1 * cos - x2 * sin).to(tl.float16), mask=mask)
    tl.store(out_ptr + x_offs + half_dim, (x2 * cos + x1 * sin).to(tl.float16), mask=mask)
''' + _CACHE + '''

class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim, base=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base = base

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = q.shape
        cos, sin = _rope_cache(seq_len, head_dim, self.base, q.device, q.dtype)
        q = q.contiguous()
        out = torch.empty_like(q)
        n_rows = batch * num_heads * seq_len

        BLOCK_M = 16
        BLOCK_D = triton.next_power_of_2(head_dim // 2)
        _rope_rotate_half_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
            q, cos, sin, out,
            n_rows, seq_len, head_dim // 2, head_dim,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        )
        return out
'''

    return make_entry(
        id=f"rope__rotate_half__b{batch}_s{seq_len}{suffix}",
        category="rope",
        op_name="RoPE (rotate-half / LLaMA-style)",
        description=(
            f"Apply rotate-half rotary positional embedding to a query tensor "
            f"of batch={batch}, seq_len={seq_len}, num_heads={num_heads}, "
            f"head_dim={head_dim}."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager builds a full-size cos/sin table by concatenating the frequencies "
            "twice, materializes rotate_half(x) as its own tensor (a full extra "
            "allocation and copy), then runs two multiplies and an add as separate "
            "passes. The kernel reads each half of every head vector once, rotates both "
            "halves in registers, and writes once; the cos/sin table is only "
            "(seq_len, head_dim/2) and is shared by every batch and head through "
            "row % seq_len, so it stays hot in L2. Each program handles a 16-row tile "
            "rather than a single 64-element row, keeping the launch grid small."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_heads": num_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def _rope_interleaved(batch: int, seq_len: int, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM,
                      suffix: str = "") -> dict:
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
num_heads = {num_heads}
seq_len = {seq_len}
head_dim = {head_dim}

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
    n_rows, seq_len, half_dim, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    pairs = tl.arange(0, BLOCK_D)
    mask = (rows[:, None] < n_rows) & (pairs[None, :] < half_dim)

    even_offs = rows[:, None] * head_dim + 2 * pairs[None, :]
    cs_offs = (rows % seq_len)[:, None] * half_dim + pairs[None, :]
    x_even = tl.load(x_ptr + even_offs, mask=mask, other=0.0).to(tl.float32)
    x_odd = tl.load(x_ptr + even_offs + 1, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + cs_offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs_offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + even_offs, (x_even * cos - x_odd * sin).to(tl.float16), mask=mask)
    tl.store(out_ptr + even_offs + 1, (x_odd * cos + x_even * sin).to(tl.float16), mask=mask)
''' + _CACHE + '''

class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim, base=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base = base

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = q.shape
        cos, sin = _rope_cache(seq_len, head_dim, self.base, q.device, q.dtype)
        q = q.contiguous()
        out = torch.empty_like(q)
        n_rows = batch * num_heads * seq_len

        BLOCK_M = 16
        BLOCK_D = triton.next_power_of_2(head_dim // 2)
        _rope_interleaved_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
            q, cos, sin, out,
            n_rows, seq_len, head_dim // 2, head_dim,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        )
        return out
'''

    return make_entry(
        id=f"rope__interleaved__b{batch}_s{seq_len}{suffix}",
        category="rope",
        op_name="RoPE (interleaved-pair / GPT-NeoX-J-style)",
        description=(
            f"Apply interleaved-pair rotary positional embedding to a query "
            f"tensor of batch={batch}, seq_len={seq_len}, num_heads={num_heads}, "
            f"head_dim={head_dim}."
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
            "positions in one pass, processing 16-row tiles per program with the "
            "(seq_len, head_dim/2) cos/sin table indexed by row % seq_len -- no "
            "intermediate even/odd/stacked tensors at all."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_heads": num_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def _rope_fused_qk(batch: int, seq_len: int, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM,
                   suffix: str = "") -> dict:
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
num_heads = {num_heads}
seq_len = {seq_len}
head_dim = {head_dim}

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
    n_rows, seq_len, half_dim, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_D)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < half_dim)

    x_offs = rows[:, None] * head_dim + cols[None, :]
    cs_offs = (rows % seq_len)[:, None] * half_dim + cols[None, :]
    cos = tl.load(cos_ptr + cs_offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs_offs, mask=mask, other=0.0).to(tl.float32)

    q1 = tl.load(q_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + x_offs + half_dim, mask=mask, other=0.0).to(tl.float32)
    tl.store(q_out_ptr + x_offs, (q1 * cos - q2 * sin).to(tl.float16), mask=mask)
    tl.store(q_out_ptr + x_offs + half_dim, (q2 * cos + q1 * sin).to(tl.float16), mask=mask)

    k1 = tl.load(k_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_ptr + x_offs + half_dim, mask=mask, other=0.0).to(tl.float32)
    tl.store(k_out_ptr + x_offs, (k1 * cos - k2 * sin).to(tl.float16), mask=mask)
    tl.store(k_out_ptr + x_offs + half_dim, (k2 * cos + k1 * sin).to(tl.float16), mask=mask)
''' + _CACHE + '''

class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim, base=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base = base

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        batch, num_heads, seq_len, head_dim = q.shape
        cos, sin = _rope_cache(seq_len, head_dim, self.base, q.device, q.dtype)
        q = q.contiguous()
        k = k.contiguous()
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        n_rows = batch * num_heads * seq_len

        BLOCK_M = 16
        BLOCK_D = triton.next_power_of_2(head_dim // 2)
        _rope_fused_qk_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
            q, k, cos, sin, q_out, k_out,
            n_rows, seq_len, head_dim // 2, head_dim,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        )
        return q_out, k_out
'''

    return make_entry(
        id=f"rope__fused_qk__b{batch}_s{seq_len}{suffix}",
        category="rope",
        op_name="RoPE fused over Q and K",
        description=(
            f"Apply rotate-half RoPE to both query and key tensors in a single "
            f"fused call, for batch={batch}, seq_len={seq_len}, "
            f"num_heads={num_heads}, head_dim={head_dim}."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager's `apply_rotary_pos_emb(q, k, cos, sin)` is two independent "
            "rotate-half chains -- each materializing its own rotate_half tensor and "
            "running separate multiply/add passes. This kernel rotates Q and K in the "
            "same program: each cos/sin tile is loaded once and applied to both "
            "tensors, so the table traffic is halved and the whole op is one launch "
            "over 16-row tiles instead of several per tensor. It runs once per layer "
            "per forward pass, so the launch and traffic savings repeat constantly."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_heads": num_heads, "head_dim": head_dim}],
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


def generate_smoke() -> list:
    """One small, non-power-of-2 instance per template, for interpreter checks."""
    return [
        _rope_rotate_half(2, 9, num_heads=3, head_dim=64, suffix="__smoke"),
        _rope_interleaved(2, 9, num_heads=3, head_dim=64, suffix="__smoke"),
        _rope_fused_qk(2, 9, num_heads=3, head_dim=64, suffix="__smoke"),
    ]
