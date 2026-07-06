"""
kv_cache templates: the decode-step ops (executed once per generated token)
-- the highest-frequency ops in an LLM serving system, and the most direct
link to Cohere/NVIDIA serving-efficiency framing.

3 templates x 20 (batch, seq_len) shape combos = 60 entries.
"""

from .common import FP16_TOLERANCE, make_entry

BATCHES = [1, 2, 4, 8, 16]
SEQ_LENS = [256, 512, 1024, 2048]  # cached context length at the decode step


def _decode_attention_mha(batch: int, seq_len: int) -> dict:
    num_heads, head_dim = 32, 128
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
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_od,
    seq_len, head_dim, scale,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + d_offsets * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < seq_len

    k_ptrs = (k_ptr + pid_b * stride_kb + pid_h * stride_kh
              + n_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd)
    k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    scores = tl.sum(q[None, :] * k, axis=1) * scale
    scores = tl.where(n_mask, scores, -float("inf"))

    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)

    v_ptrs = (v_ptr + pid_b * stride_vb + pid_h * stride_vh
              + n_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd)
    v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    out = tl.sum(p[:, None] * v, axis=0) / denom

    out_ptrs = out_ptr + pid_b * stride_ob + pid_h * stride_oh + d_offsets * stride_od
    tl.store(out_ptrs, out.to(tl.float16), mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, q, k_cache, v_cache):
        assert q.is_cuda
        batch, num_heads, _, head_dim = q.shape
        seq_len = k_cache.shape[2]
        out = torch.empty((batch, num_heads, 1, head_dim), device=q.device, dtype=torch.float16)

        BLOCK_N = triton.next_power_of_2(seq_len)
        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (batch, num_heads)
        _decode_attention_kernel[grid](
            q.squeeze(2), k_cache, v_cache, out.squeeze(2),
            q.stride(0), q.stride(1), q.stride(3),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            out.stride(0), out.stride(1), out.stride(3),
            seq_len, head_dim, self.scale,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        return out
'''

    return make_entry(
        id=f"kv_cache__decode_attention_mha__b{batch}_s{seq_len}",
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
            "Fuses QK^T, softmax, and the weighted V-sum into one launch per "
            "(batch, head), so attention scores never leave on-chip memory. This "
            "is the single highest-frequency op in the decode loop -- executed "
            "once per token per layer -- so removing the extra launches and score "
            "materialization compounds heavily across a full generation run. "
            "CAVEAT: loads the entire cache into one block; a production kernel "
            "would tile over KV pages with an online softmax ('flash decoding') "
            "to scale past a few thousand tokens of context."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_heads": num_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def _decode_attention_gqa(batch: int, seq_len: int) -> dict:
    num_query_heads, num_kv_heads, head_dim = 32, 8, 128
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
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_od,
    group_size, seq_len, head_dim, scale,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_qh = tl.program_id(1)
    pid_kvh = pid_qh // group_size

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    q_ptrs = q_ptr + pid_b * stride_qb + pid_qh * stride_qh + d_offsets * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < seq_len

    k_ptrs = (k_ptr + pid_b * stride_kb + pid_kvh * stride_kh
              + n_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd)
    k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    scores = tl.sum(q[None, :] * k, axis=1) * scale
    scores = tl.where(n_mask, scores, -float("inf"))

    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)

    v_ptrs = (v_ptr + pid_b * stride_vb + pid_kvh * stride_vh
              + n_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd)
    v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    out = tl.sum(p[:, None] * v, axis=0) / denom

    out_ptrs = out_ptr + pid_b * stride_ob + pid_qh * stride_oh + d_offsets * stride_od
    tl.store(out_ptrs, out.to(tl.float16), mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_query_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(self, q, k_cache, v_cache):
        assert q.is_cuda
        batch, num_query_heads, _, head_dim = q.shape
        seq_len = k_cache.shape[2]
        out = torch.empty((batch, num_query_heads, 1, head_dim), device=q.device, dtype=torch.float16)

        BLOCK_N = triton.next_power_of_2(seq_len)
        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (batch, num_query_heads)
        _decode_attention_gqa_kernel[grid](
            q.squeeze(2), k_cache, v_cache, out.squeeze(2),
            q.stride(0), q.stride(1), q.stride(3),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            out.stride(0), out.stride(1), out.stride(3),
            self.group_size, seq_len, head_dim, self.scale,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        return out
'''

    return make_entry(
        id=f"kv_cache__decode_attention_gqa__b{batch}_s{seq_len}",
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
            "Same decode-step fusion as the MHA version, plus avoiding the eager "
            "repeat_interleave materialization of the cached K/V by mapping each "
            "query-head program to its KV head via integer division. Relevant "
            "specifically to GQA-serving models, where the whole point of the "
            "cache layout is to store fewer KV heads than query heads -- eager's "
            "repeat_interleave briefly reconstructs the larger tensor the cache "
            "was designed to avoid."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_query_heads": num_query_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def _kv_cache_write(batch: int, seq_len: int) -> dict:
    num_heads, head_dim = 32, 128
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
        assert cache_k.is_cuda
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
        id=f"kv_cache__cache_write__b{batch}_s{seq_len}",
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
