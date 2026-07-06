"""
attention templates: full-sequence attention variants (the op dominating
prefill latency). Single-block-per-query-row kernels (BLOCK_N covers the
full KV sequence) -- correct and fused, but not tiled the way production
FlashAttention is, so they're bounded to moderate sequence lengths. Each
optimization_explanation calls this out explicitly.

3 templates x 20 (batch, seq_len) shape combos = 60 entries.
"""

from .common import FP16_TOLERANCE, make_entry

BATCHES = [1, 2, 4, 8, 16]
SEQ_LENS = [256, 512, 1024, 2048]  # 5 x 4 = 20 combos per template


def _causal_attention(batch: int, seq_len: int) -> dict:
    num_heads, head_dim = 32, 128
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
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=q.device), diagonal=1
        )
        scores = scores + causal_mask
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
def _causal_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_len, head_dim, scale,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # flattened (batch, head)
    pid_m = tl.program_id(1)   # query row index

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    q_ptrs = q_ptr + pid_bh * stride_qh + pid_m * stride_qm + d_offsets * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < seq_len
    causal_mask = n_offsets <= pid_m

    k_ptrs = (k_ptr + pid_bh * stride_kh
              + n_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd)
    k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    scores = tl.sum(q[None, :] * k, axis=1) * scale
    scores = tl.where(n_mask & causal_mask, scores, -float("inf"))

    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)

    v_ptrs = (v_ptr + pid_bh * stride_vh
              + n_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd)
    v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    out = tl.sum(p[:, None] * v, axis=0) / denom

    out_ptrs = out_ptr + pid_bh * stride_oh + pid_m * stride_om + d_offsets * stride_od
    tl.store(out_ptrs, out.to(tl.float16), mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, q, k, v):
        assert q.is_cuda
        batch, num_heads, seq_len, head_dim = q.shape
        out = torch.empty_like(q)

        q_ = q.reshape(batch * num_heads, seq_len, head_dim)
        k_ = k.reshape(batch * num_heads, seq_len, head_dim)
        v_ = v.reshape(batch * num_heads, seq_len, head_dim)
        out_ = out.reshape(batch * num_heads, seq_len, head_dim)

        BLOCK_N = triton.next_power_of_2(seq_len)
        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (batch * num_heads, seq_len)
        _causal_attention_kernel[grid](
            q_, k_, v_, out_,
            0, q_.stride(0), q_.stride(1), q_.stride(2),
            0, k_.stride(0), k_.stride(1), k_.stride(2),
            0, v_.stride(0), v_.stride(1), v_.stride(2),
            0, out_.stride(0), out_.stride(1), out_.stride(2),
            seq_len, head_dim, self.scale,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        return out
'''

    return make_entry(
        id=f"attention__causal_prefill__b{batch}_s{seq_len}",
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
            "Eager issues a QK^T matmul, an elementwise mask-add, a softmax, and a "
            "probs@V matmul as separate launches, materializing the full "
            "(seq_len, seq_len) score matrix to global memory between them. This "
            "kernel assigns one program per (batch, head, query row) and fuses "
            "QK^T, the causal mask, softmax, and the weighted V-sum into one "
            "launch, so scores never leave on-chip memory. CAVEAT: each program "
            "loads the entire key/value sequence into one block "
            "(BLOCK_N = next_power_of_2(seq_len)), which is fine up to a few "
            "thousand tokens but is not tiled -- production FlashAttention breaks "
            "the KV sequence into blocks with a running (online) softmax "
            "specifically so this loop doesn't need the whole sequence resident "
            "at once."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len": seq_len, "num_heads": num_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state, so no cross-construction seeding concern.",
    )


def _cross_attention(batch: int, seq_len_kv: int) -> dict:
    num_heads, head_dim = 32, 128
    seq_len_q = max(64, seq_len_kv // 4)
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
def _cross_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    seq_len_kv, head_dim, scale,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    q_ptrs = q_ptr + pid_bh * stride_qh + pid_m * stride_qm + d_offsets * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < seq_len_kv

    k_ptrs = (k_ptr + pid_bh * stride_kh
              + n_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd)
    k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    scores = tl.sum(q[None, :] * k, axis=1) * scale
    scores = tl.where(n_mask, scores, -float("inf"))

    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)

    v_ptrs = (v_ptr + pid_bh * stride_vh
              + n_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd)
    v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    out = tl.sum(p[:, None] * v, axis=0) / denom

    out_ptrs = out_ptr + pid_bh * stride_oh + pid_m * stride_om + d_offsets * stride_od
    tl.store(out_ptrs, out.to(tl.float16), mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, q, k, v):
        assert q.is_cuda
        batch, num_heads, seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[2]
        out = torch.empty_like(q)

        q_ = q.reshape(batch * num_heads, seq_len_q, head_dim)
        k_ = k.reshape(batch * num_heads, seq_len_kv, head_dim)
        v_ = v.reshape(batch * num_heads, seq_len_kv, head_dim)
        out_ = out.reshape(batch * num_heads, seq_len_q, head_dim)

        BLOCK_N = triton.next_power_of_2(seq_len_kv)
        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (batch * num_heads, seq_len_q)
        _cross_attention_kernel[grid](
            q_, k_, v_, out_,
            q_.stride(0), q_.stride(1), q_.stride(2),
            k_.stride(0), k_.stride(1), k_.stride(2),
            v_.stride(0), v_.stride(1), v_.stride(2),
            out_.stride(0), out_.stride(1), out_.stride(2),
            seq_len_kv, head_dim, self.scale,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        return out
'''

    return make_entry(
        id=f"attention__cross_noncausal__b{batch}_skv{seq_len_kv}",
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
            "Same fusion argument as causal prefill attention (QK^T + softmax + "
            "PV collapsed into one launch, scores never materialized), minus the "
            "causal mask and with independent query/KV sequence lengths -- the "
            "pattern used for encoder-decoder or retrieval-augmented cross-"
            "attention, where seq_len_q is usually much shorter than seq_len_kv. "
            "Same single-block-per-KV-sequence caveat as the causal variant: not "
            "tiled, bounded to moderate seq_len_kv."
        ),
        tolerance=FP16_TOLERANCE,
        test_shapes=[{"name": "default", "batch_size": batch, "seq_len_q": seq_len_q, "seq_len_kv": seq_len_kv, "num_heads": num_heads, "head_dim": head_dim}],
        provenance_notes="Hand-written; no learnable random state.",
    )


def _gqa_attention(batch: int, seq_len: int) -> dict:
    num_query_heads, num_kv_heads, head_dim = 32, 8, 128
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
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=q.device), diagonal=1
        )
        scores = scores + causal_mask
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
def _gqa_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    group_size, seq_len, head_dim, scale,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_qh = tl.program_id(0)  # flattened (batch, query_head)
    pid_m = tl.program_id(1)
    pid_kvh = pid_qh // group_size

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    q_ptrs = q_ptr + pid_qh * stride_qh + pid_m * stride_qm + d_offsets * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < seq_len
    causal_mask = n_offsets <= pid_m

    k_ptrs = (k_ptr + pid_kvh * stride_kh
              + n_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd)
    k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    scores = tl.sum(q[None, :] * k, axis=1) * scale
    scores = tl.where(n_mask & causal_mask, scores, -float("inf"))

    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)

    v_ptrs = (v_ptr + pid_kvh * stride_vh
              + n_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd)
    v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    out = tl.sum(p[:, None] * v, axis=0) / denom

    out_ptrs = out_ptr + pid_qh * stride_oh + pid_m * stride_om + d_offsets * stride_od
    tl.store(out_ptrs, out.to(tl.float16), mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_query_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(self, q, k, v):
        assert q.is_cuda
        batch, num_query_heads, seq_len, head_dim = q.shape
        num_kv_heads = k.shape[1]
        out = torch.empty_like(q)

        q_ = q.reshape(batch * num_query_heads, seq_len, head_dim)
        k_ = k.reshape(batch * num_kv_heads, seq_len, head_dim)
        v_ = v.reshape(batch * num_kv_heads, seq_len, head_dim)
        out_ = out.reshape(batch * num_query_heads, seq_len, head_dim)

        BLOCK_N = triton.next_power_of_2(seq_len)
        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (batch * num_query_heads, seq_len)
        _gqa_attention_kernel[grid](
            q_, k_, v_, out_,
            q_.stride(0), q_.stride(1), q_.stride(2),
            k_.stride(0), k_.stride(1), k_.stride(2),
            v_.stride(0), v_.stride(1), v_.stride(2),
            out_.stride(0), out_.stride(1), out_.stride(2),
            self.group_size, seq_len, head_dim, self.scale,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        return out
'''

    return make_entry(
        id=f"attention__gqa_causal_prefill__b{batch}_s{seq_len}",
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
            "Eager materializes the repeated K/V tensors via repeat_interleave "
            "before the matmul -- an extra full-size memory allocation and copy "
            "purely to satisfy shape broadcasting. The Triton kernel instead maps "
            "each query-head program directly to its KV head via integer division "
            "(pid_kvh = pid_qh // group_size) and loads K/V once per query head "
            "without ever materializing the repeated tensor, on top of the same "
            "QK^T/softmax/PV fusion as the plain causal kernel. This matters "
            "specifically for GQA/MQA-serving models (most current open LLMs) "
            "where this repeat is pure overhead relative to true multi-head "
            "attention's memory pattern."
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
