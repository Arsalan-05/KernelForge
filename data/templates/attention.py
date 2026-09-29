"""
attention templates: full-sequence attention variants (the op dominating
prefill latency). All three kernels are FlashAttention-style: one program
per (query block, batch*head), looping over KV blocks with an online
(running-max) softmax and tensor-core tl.dot, so the (seq_len, seq_len)
score matrix never exists in memory and sequence length isn't bounded by
what fits in one block.

3 templates x 20 (batch, seq_len) shape combos = 60 entries.
"""

from __future__ import annotations

from .common import FP16_TOLERANCE, make_entry

BATCHES = [1, 2, 4, 8, 16]
SEQ_LENS = [256, 512, 1024, 2048]  # 5 x 4 = 20 combos per template

NUM_HEADS, HEAD_DIM = 32, 128
GQA_QUERY_HEADS, GQA_KV_HEADS = 32, 8


def _causal_attention(batch: int, seq_len: int, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM,
                      suffix: str = "") -> dict:
    reference = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Full-sequence causal self-attention (the prefill-step op): every query
    position attends to all key positions at or before it.
    """
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: (batch, num_heads, seq_len, head_dim)
        seq_len = q.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        future = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v)

batch_size = {batch}
num_heads = {num_heads}
seq_len = {seq_len}
head_dim = {head_dim}

def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    return [q, k, v]

def get_init_inputs():
    return [num_heads, head_dim]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _causal_flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    seq_len, head_dim, qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim

    q = tl.load(
        q_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=m_mask[:, None] & d_mask[None, :], other=0.0,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Causal: keys past this block's last query row are masked for every row, so stop there.
    hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        n_mask = cols < seq_len
        k = tl.load(
            k_ptr + pid_bh * stride_kh + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=n_mask[None, :] & d_mask[:, None], other=0.0,
        )
        s = tl.dot(q, k) * qk_scale
        s = tl.where((offs_m[:, None] >= cols[None, :]) & n_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(
            v_ptr + pid_bh * stride_vh + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(tl.float16), mask=m_mask[:, None] & d_mask[None, :],
    )


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, q, k, v):
        batch, num_heads, seq_len, head_dim = q.shape
        q_ = q.reshape(batch * num_heads, seq_len, head_dim)
        k_ = k.reshape(batch * num_heads, seq_len, head_dim)
        v_ = v.reshape(batch * num_heads, seq_len, head_dim)
        out = torch.empty_like(q_)

        BLOCK_M, BLOCK_N = 64, 64
        BLOCK_D = max(16, triton.next_power_of_2(head_dim))
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * num_heads)
        _causal_flash_attention_kernel[grid](
            q_, k_, v_, out,
            q_.stride(0), q_.stride(1), q_.stride(2),
            k_.stride(0), k_.stride(1), k_.stride(2),
            v_.stride(0), v_.stride(1), v_.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            seq_len, head_dim, self.scale * 1.4426950408889634,  # fold log2(e) in so the kernel can use exp2
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )
        return out.reshape(batch, num_heads, seq_len, head_dim)
'''

    return make_entry(
        id=f"attention__causal_prefill__b{batch}_s{seq_len}{suffix}",
        category="attention",
        op_name="Causal self-attention (prefill)",
        description=(
            f"Compute full-sequence causal self-attention for batch={batch}, "
            f"seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim} -- "
            "the attention op executed once per prefill forward pass."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager runs QK^T, the mask, softmax, and probs@V as separate launches and "
            "writes the full (seq_len, seq_len) score matrix to global memory between "
            "them -- O(seq_len^2) memory traffic per head. This is a FlashAttention-style "
            "kernel: each program owns a 64-row query block, streams K/V through in "
            "64-column tiles, and keeps a running max and running sum (online softmax) "
            "so partial results are rescaled instead of recomputed; scores live only in "
            "registers and both matmuls run on tensor cores via tl.dot. The causal "
            "structure is exploited by stopping each program's KV loop at its diagonal, "
            "skipping roughly half the tiles, and log2(e) is folded into the scale so the "
            "inner loop uses exp2."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_heads": num_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state, so no cross-construction seeding concern.",
    )


def _cross_attention(batch: int, seq_len_kv: int, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM,
                     seq_len_q: int | None = None, suffix: str = "") -> dict:
    seq_len_q = seq_len_q or max(64, seq_len_kv // 4)
    reference = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Non-causal cross-attention: decoder queries attend over the full
    encoder key/value sequence (e.g. encoder-decoder / retrieval-augmented
    generation cross-attention), no masking.
    """
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: (batch, num_heads, seq_len_q, head_dim)
        # k, v: (batch, num_heads, seq_len_kv, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v)

batch_size = {batch}
num_heads = {num_heads}
seq_len_q = {seq_len_q}
seq_len_kv = {seq_len_kv}
head_dim = {head_dim}

def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_len_q, head_dim, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, dtype=torch.float16)
    return [q, k, v]

def get_init_inputs():
    return [num_heads, head_dim]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _cross_flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    seq_len_q, seq_len_kv, head_dim, qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < seq_len_q
    d_mask = offs_d < head_dim

    q = tl.load(
        q_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=m_mask[:, None] & d_mask[None, :], other=0.0,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, seq_len_kv, BLOCK_N):
        cols = start_n + offs_n
        n_mask = cols < seq_len_kv
        k = tl.load(
            k_ptr + pid_bh * stride_kh + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=n_mask[None, :] & d_mask[:, None], other=0.0,
        )
        s = tl.dot(q, k) * qk_scale
        s = tl.where(n_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(
            v_ptr + pid_bh * stride_vh + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(tl.float16), mask=m_mask[:, None] & d_mask[None, :],
    )


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, q, k, v):
        batch, num_heads, seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[2]
        q_ = q.reshape(batch * num_heads, seq_len_q, head_dim)
        k_ = k.reshape(batch * num_heads, seq_len_kv, head_dim)
        v_ = v.reshape(batch * num_heads, seq_len_kv, head_dim)
        out = torch.empty_like(q_)

        BLOCK_M, BLOCK_N = 64, 64
        BLOCK_D = max(16, triton.next_power_of_2(head_dim))
        grid = (triton.cdiv(seq_len_q, BLOCK_M), batch * num_heads)
        _cross_flash_attention_kernel[grid](
            q_, k_, v_, out,
            q_.stride(0), q_.stride(1), q_.stride(2),
            k_.stride(0), k_.stride(1), k_.stride(2),
            v_.stride(0), v_.stride(1), v_.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            seq_len_q, seq_len_kv, head_dim, self.scale * 1.4426950408889634,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )
        return out.reshape(batch, num_heads, seq_len_q, head_dim)
'''

    return make_entry(
        id=f"attention__cross_noncausal__b{batch}_skv{seq_len_kv}{suffix}",
        category="attention",
        op_name="Non-causal cross-attention",
        description=(
            f"Compute non-causal cross-attention for batch={batch}, decoder "
            f"seq_len_q={seq_len_q} against encoder seq_len_kv={seq_len_kv}, "
            f"num_heads={num_heads}, head_dim={head_dim}."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Same FlashAttention structure as causal prefill (query-block programs, "
            "KV streamed in tiles, online softmax, tensor-core tl.dot for both "
            "matmuls, so the seq_len_q x seq_len_kv score matrix is never written to "
            "memory), minus the causal mask and with independent query/KV lengths -- "
            "the encoder-decoder / retrieval cross-attention pattern, where seq_len_q "
            "is usually much shorter than seq_len_kv. Because K/V are tiled, long "
            "encoder contexts cost more loop iterations, not more on-chip memory."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len_q": seq_len_q, "seq_len_kv": seq_len_kv, "num_heads": num_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def _gqa_attention(batch: int, seq_len: int, num_query_heads: int = GQA_QUERY_HEADS,
                   num_kv_heads: int = GQA_KV_HEADS, head_dim: int = HEAD_DIM, suffix: str = "") -> dict:
    group_size = num_query_heads // num_kv_heads
    reference = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Grouped-query causal self-attention (GQA): {num_query_heads} query heads
    share {num_kv_heads} key/value heads ({group_size} query heads per KV
    head) -- the attention variant used by most current serving-optimized
    LLMs to cut KV-cache size relative to full multi-head attention.
    """
    def __init__(self, num_query_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: (batch, num_query_heads, seq_len, head_dim)
        # k, v: (batch, num_kv_heads, seq_len, head_dim)
        seq_len = q.shape[2]
        k_rep = k.repeat_interleave(self.group_size, dim=1)
        v_rep = v.repeat_interleave(self.group_size, dim=1)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale
        future = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v_rep)

batch_size = {batch}
num_query_heads = {num_query_heads}
num_kv_heads = {num_kv_heads}
seq_len = {seq_len}
head_dim = {head_dim}

def get_inputs():
    q = torch.randn(batch_size, num_query_heads, seq_len, head_dim, dtype=torch.float16)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float16)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float16)
    return [q, k, v]

def get_init_inputs():
    return [num_query_heads, num_kv_heads, head_dim]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    group_size, seq_len, head_dim, qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_qh = tl.program_id(1)  # flattened (batch, query_head)
    pid_kvh = pid_qh // group_size  # == batch * num_kv_heads + query_head // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim

    q = tl.load(
        q_ptr + pid_qh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=m_mask[:, None] & d_mask[None, :], other=0.0,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        n_mask = cols < seq_len
        k = tl.load(
            k_ptr + pid_kvh * stride_kh + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=n_mask[None, :] & d_mask[:, None], other=0.0,
        )
        s = tl.dot(q, k) * qk_scale
        s = tl.where((offs_m[:, None] >= cols[None, :]) & n_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(
            v_ptr + pid_kvh * stride_vh + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr + pid_qh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(tl.float16), mask=m_mask[:, None] & d_mask[None, :],
    )


class ModelNew(torch.nn.Module):
    def __init__(self, num_query_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(self, q, k, v):
        batch, num_query_heads, seq_len, head_dim = q.shape
        num_kv_heads = k.shape[1]
        q_ = q.reshape(batch * num_query_heads, seq_len, head_dim)
        k_ = k.reshape(batch * num_kv_heads, seq_len, head_dim)
        v_ = v.reshape(batch * num_kv_heads, seq_len, head_dim)
        out = torch.empty_like(q_)

        BLOCK_M, BLOCK_N = 64, 64
        BLOCK_D = max(16, triton.next_power_of_2(head_dim))
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * num_query_heads)
        _gqa_flash_attention_kernel[grid](
            q_, k_, v_, out,
            q_.stride(0), q_.stride(1), q_.stride(2),
            k_.stride(0), k_.stride(1), k_.stride(2),
            v_.stride(0), v_.stride(1), v_.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            self.group_size, seq_len, head_dim, self.scale * 1.4426950408889634,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )
        return out.reshape(batch, num_query_heads, seq_len, head_dim)
'''

    return make_entry(
        id=f"attention__gqa_causal_prefill__b{batch}_s{seq_len}{suffix}",
        category="attention",
        op_name="Grouped-query causal attention (prefill)",
        description=(
            f"Compute causal grouped-query attention for batch={batch}, "
            f"seq_len={seq_len}, {num_query_heads} query heads sharing "
            f"{num_kv_heads} KV heads (group size {group_size})."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager materializes repeat_interleave'd copies of K and V "
            f"({group_size}x their size) purely to make shapes broadcast, then runs the "
            "same unfused QK^T/mask/softmax/PV chain as MHA. The kernel never builds "
            "the repeated tensors: each query-head program maps to its shared KV head "
            "with one integer division (pid_kvh = pid_qh // group_size) and streams "
            "that head's K/V tiles directly, on top of the FlashAttention structure "
            "(online softmax, tensor-core tl.dot, KV loop stopped at the causal "
            "diagonal). Query heads in the same group read the same K/V tiles close "
            "together in time, so most of those reads hit L2 instead of HBM."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_query_heads": num_query_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def generate() -> list:
    entries = []
    for batch in BATCHES:
        for seq_len in SEQ_LENS:
            entries.append(_causal_attention(batch, seq_len))
            entries.append(_cross_attention(batch, seq_len))
            entries.append(_gqa_attention(batch, seq_len))
    return entries


def generate_smoke() -> list:
    """One small, non-power-of-2 instance per template, for interpreter checks."""
    return [
        _causal_attention(2, 80, num_heads=2, head_dim=64, suffix="__smoke"),
        _cross_attention(2, 80, num_heads=2, head_dim=64, seq_len_q=24, suffix="__smoke"),
        _gqa_attention(2, 80, num_query_heads=4, num_kv_heads=2, head_dim=64, suffix="__smoke"),
    ]
