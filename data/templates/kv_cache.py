"""
kv_cache templates: the decode-step ops (executed once per generated token)
-- the highest-frequency ops in an LLM serving system, and the most direct
link to Cohere/NVIDIA serving-efficiency framing.

The decode-attention kernels stream the cache in BLOCK_N tiles with an
online softmax (flash-decoding style, without the split-KV second pass), so
cached context length costs loop iterations rather than registers.

3 templates x 20 (batch, seq_len) shape combos = 60 entries.
"""

from .common import FP16_TOLERANCE, make_entry

BATCHES = [1, 2, 4, 8, 16]
SEQ_LENS = [256, 512, 1024, 2048]  # cached context length at the decode step

NUM_HEADS, HEAD_DIM = 32, 128
GQA_QUERY_HEADS, GQA_KV_HEADS = 32, 8


def _decode_attention_mha(batch: int, seq_len: int, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM,
                          suffix: str = "") -> dict:
    reference = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Single-query "decode step" attention: one new query token attends over
    the full cached K/V history (the op executed once per generated token
    during autoregressive decoding). No causal mask is needed beyond what
    the cache already encodes -- the query is always the newest position.
    """
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        # q: (batch, num_heads, 1, head_dim)
        # k_cache, v_cache: (batch, num_heads, seq_len, head_dim)
        scores = torch.matmul(q, k_cache.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v_cache)

batch_size = {batch}
num_heads = {num_heads}
head_dim = {head_dim}
seq_len = {seq_len}

def get_inputs():
    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float16)
    k_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    v_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    return [q, k_cache, v_cache]

def get_init_inputs():
    return [num_heads, head_dim]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _decode_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh,
    seq_len, head_dim, qk_scale,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    q = tl.load(q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    k_base = k_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = v_ptr + pid_b * stride_vb + pid_h * stride_vh

    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        cols = start_n + offs_n
        n_mask = cols < seq_len
        kv_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :], mask=kv_mask, other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, axis=1) * qk_scale
        s = tl.where(n_mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :], mask=kv_mask, other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d, out.to(tl.float16), mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, q, k_cache, v_cache):
        batch, num_heads, _, head_dim = q.shape
        seq_len = k_cache.shape[2]
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        out = torch.empty((batch, num_heads, 1, head_dim), device=q.device, dtype=torch.float16)

        BLOCK_N = 64
        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (batch, num_heads)
        _decode_attention_kernel[grid](
            q, k_cache, v_cache, out,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            out.stride(0), out.stride(1),
            seq_len, head_dim, self.scale * 1.4426950408889634,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        return out
'''

    return make_entry(
        id=f"kv_cache__decode_attention_mha__b{batch}_s{seq_len}{suffix}",
        category="kv_cache",
        op_name="Single-query decode-step attention (MHA)",
        description=(
            f"Compute scaled dot-product attention for a single new query token "
            f"against a cached K/V history for batch={batch}, seq_len={seq_len}, "
            f"num_heads={num_heads}, head_dim={head_dim} -- the attention op "
            "executed once per generated token during decoding."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Decode attention is pure memory bandwidth: each step reads the whole K/V "
            "cache once to produce one output vector. Eager adds three launches and "
            "writes/re-reads the score and probability rows in between. This kernel is "
            "one launch per (batch, head) that streams the cache in 64-token tiles, "
            "keeps a running max and sum (online softmax) so each tile is read exactly "
            "once, and never writes scores to memory. It's the highest-frequency op in "
            "the decode loop (once per token per layer), so the saved launches and "
            "traffic compound over a generation. At batch*heads below the SM count, a "
            "split-KV (flash-decoding) second pass would add parallelism across the "
            "sequence -- the natural next step."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_heads": num_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def _decode_attention_gqa(batch: int, seq_len: int, num_query_heads: int = GQA_QUERY_HEADS,
                          num_kv_heads: int = GQA_KV_HEADS, head_dim: int = HEAD_DIM, suffix: str = "") -> dict:
    group_size = num_query_heads // num_kv_heads
    reference = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Single-query decode-step attention with grouped-query KV caching:
    {num_query_heads} query heads share {num_kv_heads} cached KV heads
    ({group_size} query heads per KV head) -- the decode-step op for
    GQA-serving models, where the cache itself is already smaller than
    full multi-head attention would require.
    """
    def __init__(self, num_query_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        # q: (batch, num_query_heads, 1, head_dim)
        # k_cache, v_cache: (batch, num_kv_heads, seq_len, head_dim)
        k_rep = k_cache.repeat_interleave(self.group_size, dim=1)
        v_rep = v_cache.repeat_interleave(self.group_size, dim=1)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v_rep)

batch_size = {batch}
num_query_heads = {num_query_heads}
num_kv_heads = {num_kv_heads}
head_dim = {head_dim}
seq_len = {seq_len}

def get_inputs():
    q = torch.randn(batch_size, num_query_heads, 1, head_dim, dtype=torch.float16)
    k_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float16)
    v_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float16)
    return [q, k_cache, v_cache]

def get_init_inputs():
    return [num_query_heads, num_kv_heads, head_dim]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _decode_attention_gqa_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh,
    group_size, seq_len, head_dim, qk_scale,
    BLOCK_G: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kvh = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_G)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    g_mask = offs_g < group_size
    d_mask = offs_d < head_dim
    q_heads = pid_kvh * group_size + offs_g

    # Every query head that shares this KV head, as the rows of one (BLOCK_G, D) tile.
    q = tl.load(
        q_ptr + pid_b * stride_qb + q_heads[:, None] * stride_qh + offs_d[None, :],
        mask=g_mask[:, None] & d_mask[None, :], other=0.0,
    )
    k_base = k_ptr + pid_b * stride_kb + pid_kvh * stride_kh
    v_base = v_ptr + pid_b * stride_vb + pid_kvh * stride_vh

    m_i = tl.full((BLOCK_G,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_G,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_G, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        cols = start_n + offs_n
        n_mask = cols < seq_len
        k = tl.load(k_base + cols[None, :] * stride_kn + offs_d[:, None],
                    mask=n_mask[None, :] & d_mask[:, None], other=0.0)  # (BLOCK_D, BLOCK_N)
        s = tl.dot(q, k) * qk_scale
        s = tl.where(n_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr + pid_b * stride_ob + q_heads[:, None] * stride_oh + offs_d[None, :],
        acc.to(tl.float16), mask=g_mask[:, None] & d_mask[None, :],
    )


class ModelNew(torch.nn.Module):
    def __init__(self, num_query_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(self, q, k_cache, v_cache):
        batch, num_query_heads, _, head_dim = q.shape
        num_kv_heads, seq_len = k_cache.shape[1], k_cache.shape[2]
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        out = torch.empty((batch, num_query_heads, 1, head_dim), device=q.device, dtype=torch.float16)

        BLOCK_G = max(16, triton.next_power_of_2(self.group_size))  # tl.dot needs >= 16 rows
        BLOCK_N = 64
        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (batch, num_kv_heads)
        _decode_attention_gqa_kernel[grid](
            q, k_cache, v_cache, out,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            out.stride(0), out.stride(1),
            self.group_size, seq_len, head_dim, self.scale * 1.4426950408889634,
            BLOCK_G=BLOCK_G, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        return out
'''

    return make_entry(
        id=f"kv_cache__decode_attention_gqa__b{batch}_s{seq_len}{suffix}",
        category="kv_cache",
        op_name="Single-query decode-step attention (GQA)",
        description=(
            f"Compute GQA decode-step attention for batch={batch}, seq_len={seq_len}, "
            f"{num_query_heads} query heads sharing {num_kv_heads} cached KV heads."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager's repeat_interleave rebuilds a full-size K/V cache -- exactly what the "
            "GQA layout exists to avoid -- before an unfused softmax chain. This kernel "
            "uses GQA packing: one program per (batch, KV head) loads all "
            f"{group_size} query heads that share that KV head as the rows of a single "
            "tile, so every cached K/V tile is read from HBM once per group instead of "
            "once per query head, and the group's scores come out of one tensor-core "
            "tl.dot (padded to 16 rows, the MMA minimum) instead of per-head dot "
            "products. The cache is streamed in 64-token tiles with an online softmax, "
            "so context length never has to fit in registers."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_query_heads": num_query_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def _kv_cache_write(batch: int, seq_len: int, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM,
                    suffix: str = "") -> dict:
    position = seq_len // 2
    reference = f'''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    KV-cache write: insert a newly computed key/value vector for one token
    into a pre-allocated cache buffer at a fixed position -- the op run
    once per generated token, per layer, to append to the running cache.
    """
    def __init__(self, num_heads, head_dim, position):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.position = position

    def forward(self, cache_k: torch.Tensor, cache_v: torch.Tensor, new_k: torch.Tensor, new_v: torch.Tensor):
        # cache_k, cache_v: (batch, num_heads, max_seq_len, head_dim)
        # new_k, new_v: (batch, num_heads, 1, head_dim)
        cache_k = cache_k.clone()
        cache_v = cache_v.clone()
        cache_k[:, :, self.position, :] = new_k.squeeze(2)
        cache_v[:, :, self.position, :] = new_v.squeeze(2)
        return cache_k, cache_v

batch_size = {batch}
num_heads = {num_heads}
head_dim = {head_dim}
max_seq_len = {seq_len}
position = {position}

def get_inputs():
    cache_k = torch.randn(batch_size, num_heads, max_seq_len, head_dim, dtype=torch.float16)
    cache_v = torch.randn(batch_size, num_heads, max_seq_len, head_dim, dtype=torch.float16)
    new_k = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float16)
    new_v = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float16)
    return [cache_k, cache_v, new_k, new_v]

def get_init_inputs():
    return [num_heads, head_dim, position]
'''

    triton_kernel = '''import torch
import triton
import triton.language as tl


@triton.jit
def _kv_cache_write_kernel(
    cache_k_ptr, cache_v_ptr, new_k_ptr, new_v_ptr,
    stride_ckb, stride_ckh, stride_ckn, stride_ckd,
    stride_cvb, stride_cvh, stride_cvn, stride_cvd,
    stride_nkb, stride_nkh, stride_nkd,
    stride_nvb, stride_nvh, stride_nvd,
    position, head_dim,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    new_k_ptrs = new_k_ptr + pid_b * stride_nkb + pid_h * stride_nkh + d_offsets * stride_nkd
    new_v_ptrs = new_v_ptr + pid_b * stride_nvb + pid_h * stride_nvh + d_offsets * stride_nvd
    k_val = tl.load(new_k_ptrs, mask=d_mask, other=0.0)
    v_val = tl.load(new_v_ptrs, mask=d_mask, other=0.0)

    cache_k_ptrs = (cache_k_ptr + pid_b * stride_ckb + pid_h * stride_ckh
                    + position * stride_ckn + d_offsets * stride_ckd)
    cache_v_ptrs = (cache_v_ptr + pid_b * stride_cvb + pid_h * stride_cvh
                    + position * stride_cvn + d_offsets * stride_cvd)
    tl.store(cache_k_ptrs, k_val, mask=d_mask)
    tl.store(cache_v_ptrs, v_val, mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim, position):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.position = position

    def forward(self, cache_k, cache_v, new_k, new_v):
        batch, num_heads, max_seq_len, head_dim = cache_k.shape
        new_k_ = new_k.squeeze(2)
        new_v_ = new_v.squeeze(2)

        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (batch, num_heads)
        _kv_cache_write_kernel[grid](
            cache_k, cache_v, new_k_, new_v_,
            cache_k.stride(0), cache_k.stride(1), cache_k.stride(2), cache_k.stride(3),
            cache_v.stride(0), cache_v.stride(1), cache_v.stride(2), cache_v.stride(3),
            new_k_.stride(0), new_k_.stride(1), new_k_.stride(2),
            new_v_.stride(0), new_v_.stride(1), new_v_.stride(2),
            self.position, head_dim,
            BLOCK_D=BLOCK_D,
        )
        # Written in place; return the same buffers so the shape/tuple contract
        # matches the reference's clone-then-assign implementation.
        return cache_k, cache_v
'''

    return make_entry(
        id=f"kv_cache__cache_write__b{batch}_s{seq_len}{suffix}",
        category="kv_cache",
        op_name="KV-cache write (single-token insert)",
        description=(
            f"Write a new key/value vector for one token into a "
            f"({batch}, {num_heads}, {seq_len}, {head_dim}) pre-allocated KV cache "
            f"at position {position} -- the op run once per generated token, per "
            "layer, to append to the running cache."
        ),
        dtype="float16",
        pytorch_reference=reference,
        triton_kernel=triton_kernel,
        optimization_explanation=(
            "Eager's `cache[:, :, position, :] = new_k` must clone the entire "
            f"(batch, num_heads, {seq_len}, head_dim) cache buffer first so the op "
            "can return a new tensor rather than mutate its input in place -- an "
            "O(max_seq_len x head_dim) copy to update a single O(head_dim) row. "
            "The Triton kernel launches one program per (batch, head) and writes "
            "only the new row directly into the pre-allocated cache buffer in "
            "place -- O(head_dim) work regardless of how long the cache has "
            "grown. This mirrors how real serving engines (vLLM, TensorRT-LLM) "
            "implement KV-cache writes as in-place scatter kernels rather than "
            "clone-and-assign, which would get more expensive every single step "
            "as the cache grows across a generation run."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "max_seq_len": seq_len, "position": position, "num_heads": num_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state. Candidate mutates its cache input in place by design -- see optimization_explanation.",
    )


def generate() -> list:
    entries = []
    for batch in BATCHES:
        for seq_len in SEQ_LENS:
            entries.append(_decode_attention_mha(batch, seq_len))
            entries.append(_decode_attention_gqa(batch, seq_len))
            entries.append(_kv_cache_write(batch, seq_len))
    return entries


def generate_smoke() -> list:
    """One small, non-power-of-2 instance per template, for interpreter checks."""
    return [
        _decode_attention_mha(2, 100, num_heads=2, head_dim=64, suffix="__smoke"),
        _decode_attention_gqa(2, 100, num_query_heads=8, num_kv_heads=2, head_dim=64, suffix="__smoke"),
        _kv_cache_write(2, 100, num_heads=2, head_dim=64, suffix="__smoke"),
    ]
