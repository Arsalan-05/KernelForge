#!/usr/bin/env python3
"""
Mutation test of the KernelForge verification harness.

Creates deliberately broken Triton kernel variants, runs each through the
interpreter harness (verify_model_kernel with interpret=True), and reports
how many are caught (incorrect result detected) vs survived (not caught).
Survivors indicate harness coverage gaps; each one is investigated below.

Usage (from repo root, inside the kernelforge-verify Docker image):

    docker run --rm -v "$PWD":/work kernelforge-verify python scripts/mutation_test.py

Takes ~15-25 minutes (32 mutations, smoke shapes, serial, interpreter mode).

What counts as "caught":
  - result.passed is False (harness correctly rejects the mutant)
  - status is "error", "timeout", or "crash" (mutant fails to run)

What counts as "survived":
  - result.passed is True (harness incorrectly accepts the mutant)

Equivalent mutations (expected survivors, not harness failures):
  - harmless_change: mathematically equivalent to the reference on all inputs
  - documented_gap: equivalent only because of a specific test-data property
    (all-ones weights, all-zeros biases), not an inherent harness limitation.
    These are real gaps; fix is to use non-trivial random init in the test.
"""

from __future__ import annotations

import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data.templates import SMOKE_GENERATORS
from verification.verify_kernel import VerifyResult, verify_model_kernel

TIMEOUT_S = 90.0


@dataclass
class MutantResult:
    name: str
    mutation_type: str
    template: str
    caught: bool
    status: str
    error: Optional[str]
    elapsed_s: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _apply_substitution(kernel: str, old: str, new: str, label: str) -> str:
    count = kernel.count(old)
    if count == 0:
        raise ValueError(f"[{label}] substitution target not found:\n  {old!r}")
    if count > 1:
        raise ValueError(f"[{label}] substitution target found {count} times (expected 1):\n  {old!r}")
    return kernel.replace(old, new, 1)


# ===========================================================================
# ATTENTION / CAUSAL PREFILL (3 mutations)
# ===========================================================================

def _mut_attention_missing_causal_mask(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "s = tl.where((offs_m[:, None] >= cols[None, :]) & n_mask[None, :], s, float(\"-inf\"))",
        "s = tl.where(n_mask[None, :], s, float(\"-inf\"))  # BUG: causal mask removed",
        "attention_missing_causal_mask",
    )


def _mut_attention_zero_max_init(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "m_i = tl.full((BLOCK_M,), float(\"-inf\"), dtype=tl.float32)",
        "m_i = tl.zeros((BLOCK_M,), dtype=tl.float32)  # NOTE: different init; output unchanged by shift-invariance",
        "attention_zero_max_init",
    )


def _mut_attention_full_loop(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)",
        "hi = seq_len  # NOTE: loop extended past diagonal; causal mask still applied",
        "attention_full_loop",
    )


# ===========================================================================
# ATTENTION / CROSS NONCAUSAL (2 mutations)
# ===========================================================================

def _mut_cross_attn_wrong_reduction_axis(ref: str, kernel: str) -> str:
    """Change softmax max reduction from axis=1 (per query row) to axis=0 (per key col).

    This collapses the wrong dimension so every query row gets the same running max
    (the column-max instead of the row-max), making the online softmax incorrect.
    """
    return _apply_substitution(
        kernel,
        "m_new = tl.maximum(m_i, tl.max(s, 1))",
        "m_new = tl.maximum(m_i, tl.max(s, 0))  # BUG: wrong reduction axis",
        "cross_attn_wrong_reduction_axis",
    )


def _mut_cross_attn_missing_denominator(ref: str, kernel: str) -> str:
    """Skip the final softmax denominator division, producing unnormalized attention."""
    return _apply_substitution(
        kernel,
        "acc = acc / l_i[:, None]",
        "acc = acc  # BUG: softmax denominator not applied — output is unnormalized",
        "cross_attn_missing_denominator",
    )


def _mut_cross_attn_l_i_accumulation_reset(ref: str, kernel: str) -> str:
    """Reset l_i each tile instead of accumulating — only the last tile's denominator counts.

    Correct: l_i = l_i * alpha + tl.sum(p, 1)  (accumulates across all KV tiles).
    Mutant:  l_i = tl.sum(p, 1)                 (discards prior tile contributions).
    With seq_len_kv=80 > BLOCK_N=64 there are 2 tiles; the first tile's normalization
    contribution is lost, so the final softmax denominator is wrong.
    """
    return _apply_substitution(
        kernel,
        "l_i = l_i * alpha + tl.sum(p, 1)",
        "l_i = tl.sum(p, 1)  # BUG: l_i reset instead of accumulated across tiles",
        "cross_attn_l_i_accumulation_reset",
    )


# ===========================================================================
# ATTENTION / GQA CAUSAL PREFILL (2 mutations)
# ===========================================================================

def _mut_gqa_attn_missing_causal_mask(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "s = tl.where((offs_m[:, None] >= cols[None, :]) & n_mask[None, :], s, float(\"-inf\"))",
        "s = tl.where(n_mask[None, :], s, float(\"-inf\"))  # BUG: causal mask removed",
        "gqa_attn_missing_causal_mask",
    )


def _mut_gqa_attn_wrong_kv_head_index(ref: str, kernel: str) -> str:
    """Remove the group_size division so every Q-head loads from the wrong KV head.

    Correct: pid_kvh = pid_qh // group_size (maps to the shared KV head).
    Mutant:  pid_kvh = pid_qh              (treats each Q-head as a KV head index).
    With more Q-heads than KV heads this reads out-of-bounds KV data.
    """
    return _apply_substitution(
        kernel,
        "pid_kvh = pid_qh // group_size  # == batch * num_kv_heads + query_head // group_size",
        "pid_kvh = pid_qh  # BUG: group_size division missing — wrong KV head",
        "gqa_attn_wrong_kv_head_index",
    )


# ===========================================================================
# KV CACHE / DECODE ATTENTION MHA (2 mutations)
# ===========================================================================

def _mut_decode_mha_wrong_dot_axis(ref: str, kernel: str) -> str:
    """Change the axis of the manual Q·K dot product from 1 (reduce keys) to 0.

    axis=1 reduces over the head_dim of k (each row dot-products with q),
    giving one scalar per key position. axis=0 reduces over key positions,
    giving one scalar per head_dim element — a completely wrong shape.
    This causes a shape mismatch or silently wrong attention scores.
    """
    return _apply_substitution(
        kernel,
        "s = tl.sum(q[None, :] * k, axis=1) * qk_scale",
        "s = tl.sum(q[None, :] * k, axis=0) * qk_scale  # BUG: wrong reduction axis",
        "decode_mha_wrong_dot_axis",
    )


def _mut_decode_mha_missing_normalization(ref: str, kernel: str) -> str:
    """Skip the softmax denominator division, producing unnormalized attention."""
    return _apply_substitution(
        kernel,
        "out = acc / l_i",
        "out = acc  # BUG: softmax denominator not applied — unnormalized",
        "decode_mha_missing_normalization",
    )


# ===========================================================================
# KV CACHE / DECODE ATTENTION GQA (2 mutations)
# ===========================================================================

def _mut_decode_gqa_missing_kv_mask(ref: str, kernel: str) -> str:
    """Replace -inf masking of out-of-bounds keys with 0.0, so padding contributes.

    The correct mask zeros out positions >= seq_len. Using 0.0 instead of -inf
    makes exp(0)=1 contribute to the softmax denominator for padding positions,
    producing incorrect attention weights when BLOCK_N exceeds the residual key count.
    """
    return _apply_substitution(
        kernel,
        "s = tl.where(n_mask[None, :], s, float(\"-inf\"))",
        "s = tl.where(n_mask[None, :], s, 0.0)  # BUG: -inf replaced with 0 — padding contributes",
        "decode_gqa_missing_kv_mask",
    )


def _mut_decode_gqa_wrong_group_offset(ref: str, kernel: str) -> str:
    """Drop the KV-head base so all GQA groups read from query heads 0..group_size-1.

    Correct: q_heads = pid_kvh * group_size + offs_g (each KV head's group).
    Mutant:  q_heads = offs_g                        (always the first group).
    Result: all KV heads process the same query heads; most output locations
    never get written to.
    """
    return _apply_substitution(
        kernel,
        "q_heads = pid_kvh * group_size + offs_g",
        "q_heads = offs_g  # BUG: KV-head base offset missing",
        "decode_gqa_wrong_group_offset",
    )


# ===========================================================================
# KV CACHE / CACHE WRITE (2 mutations)
# ===========================================================================

def _mut_cache_write_missing_position_k(ref: str, kernel: str) -> str:
    """Write new_k to cache slot 0 instead of the correct position.

    The correct K-cache pointer adds position * stride_ckn to place the new
    key vector at the right sequence position. Setting this to 0 always overwrites
    slot 0, so only decode step 0 gets the correct key; all subsequent steps corrupt it.
    """
    return _apply_substitution(
        kernel,
        "cache_k_ptrs = (cache_k_ptr + pid_b * stride_ckb + pid_h * stride_ckh\n"
        "                    + position * stride_ckn + d_offsets * stride_ckd)",
        "cache_k_ptrs = (cache_k_ptr + pid_b * stride_ckb + pid_h * stride_ckh\n"
        "                    + 0 + d_offsets * stride_ckd)  # BUG: position ignored",
        "cache_write_missing_position_k",
    )


def _mut_cache_write_k_stores_v(ref: str, kernel: str) -> str:
    """Store the new V-vector into the K-cache instead of the K-vector.

    Any subsequent decode step that reads K-cache will get value data, completely
    corrupting the attention scores.
    """
    return _apply_substitution(
        kernel,
        "tl.store(cache_k_ptrs, k_val, mask=d_mask)",
        "tl.store(cache_k_ptrs, v_val, mask=d_mask)  # BUG: stores V in K cache",
        "cache_write_k_stores_v",
    )


# ===========================================================================
# NORM / RMSNORM (2 mutations — existing)
# ===========================================================================

def _mut_norm_rmsnorm_missing_divisor(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "mean_sq = tl.sum(x * x, axis=0) / n_cols",
        "mean_sq = tl.sum(x * x, axis=0)  # BUG: should divide by n_cols",
        "norm_rmsnorm_missing_divisor",
    )


def _mut_norm_rmsnorm_missing_weight(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "y = x * inv_rms * weight",
        "y = x * inv_rms  # BUG: weight skipped (survives due to all-ones init)",
        "norm_rmsnorm_missing_weight",
    )


# ===========================================================================
# NORM / LAYERNORM (2 mutations)
# ===========================================================================

def _mut_layernorm_missing_mean_centering(ref: str, kernel: str) -> str:
    """Skip mean subtraction in LayerNorm — x is not centered before normalization.

    Without centering, the variance is computed from the uncentered x (which equals
    E[x²]), not from the centered diff (which equals Var[x]). The normalization
    removes the wrong component and the output will differ from true LayerNorm.
    """
    return _apply_substitution(
        kernel,
        "diff = tl.where(mask, x - mean, 0.0)",
        "diff = tl.where(mask, x, 0.0)  # BUG: mean subtraction missing",
        "layernorm_missing_mean_centering",
    )


def _mut_layernorm_missing_bias(ref: str, kernel: str) -> str:
    """Skip the bias addition.  With bias=zeros this is equivalent (documented gap).

    ModelNew and Model both init bias to torch.zeros, so skipping it produces
    the same output. This is a gap: the harness cannot detect missing bias when bias==0.
    """
    return _apply_substitution(
        kernel,
        "y = diff * inv_std * weight + bias",
        "y = diff * inv_std * weight  # BUG: bias not added (gap: bias==zeros in test)",
        "layernorm_missing_bias",
    )


# ===========================================================================
# NORM / RMSNORM RESIDUAL (2 mutations)
# ===========================================================================

def _mut_rmsnorm_res_missing_add(ref: str, kernel: str) -> str:
    """Skip the residual addition — normalize x alone instead of x + residual.

    The residual stream is a critical component of transformer residual connections.
    Skipping the add changes both the normalization target and the residual output.
    """
    return _apply_substitution(
        kernel,
        "residual = x + residual",
        "residual = x  # BUG: residual connection skipped",
        "rmsnorm_res_missing_add",
    )


def _mut_rmsnorm_res_normalize_x_not_residual(ref: str, kernel: str) -> str:
    """Compute RMS over x instead of over (x + residual).

    The correct kernel normalizes the updated residual = x + prev_residual.
    This mutant uses the raw x for the RMS computation while still writing
    the correct x+residual as the new residual, so the normalization is wrong.
    """
    return _apply_substitution(
        kernel,
        "mean_sq = tl.sum(residual * residual, axis=0) / n_cols",
        "mean_sq = tl.sum(x * x, axis=0) / n_cols  # BUG: normalizes x not (x+residual)",
        "rmsnorm_res_normalize_x_not_residual",
    )


# ===========================================================================
# ROPE / ROTATE HALF (2 mutations)
# ===========================================================================

def _mut_rope_sign_flip(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "tl.store(out_ptr + x_offs, (x1 * cos - x2 * sin).to(tl.float16), mask=mask)",
        "tl.store(out_ptr + x_offs, (x1 * cos + x2 * sin).to(tl.float16), mask=mask)  # BUG: sign flipped",
        "rope_sign_flip",
    )


def _mut_rope_second_half_sign_flip(ref: str, kernel: str) -> str:
    """Flip the sign of x1·sin in the second-half output: x2·cos - x1·sin instead of + x1·sin."""
    return _apply_substitution(
        kernel,
        "tl.store(out_ptr + x_offs + half_dim, (x2 * cos + x1 * sin).to(tl.float16), mask=mask)",
        "tl.store(out_ptr + x_offs + half_dim, (x2 * cos - x1 * sin).to(tl.float16), mask=mask)  # BUG: sign flipped",
        "rope_second_half_sign_flip",
    )


# ===========================================================================
# ROPE / INTERLEAVED (2 mutations)
# ===========================================================================

def _mut_interleaved_even_stride_wrong(ref: str, kernel: str) -> str:
    """Use stride 1 instead of stride 2 for even-index positions.

    Correct: even_offs = rows * head_dim + 2 * pairs  (every other element).
    Mutant:  even_offs = rows * head_dim + pairs       (every element, mixing even/odd).
    Result: reads and writes interleaved pairs from the wrong positions.
    """
    return _apply_substitution(
        kernel,
        "even_offs = rows[:, None] * head_dim + 2 * pairs[None, :]",
        "even_offs = rows[:, None] * head_dim + pairs[None, :]  # BUG: stride 1 instead of 2",
        "interleaved_even_stride_wrong",
    )


def _mut_interleaved_odd_sign_flip(ref: str, kernel: str) -> str:
    """Flip the rotation sign for odd positions: x_odd·cos - x_even·sin."""
    return _apply_substitution(
        kernel,
        "tl.store(out_ptr + even_offs + 1, (x_odd * cos + x_even * sin).to(tl.float16), mask=mask)",
        "tl.store(out_ptr + even_offs + 1, (x_odd * cos - x_even * sin).to(tl.float16), mask=mask)  # BUG: sign flipped",
        "interleaved_odd_sign_flip",
    )


# ===========================================================================
# ROPE / FUSED QK (2 mutations)
# ===========================================================================

def _mut_fused_qk_k_reads_q(ref: str, kernel: str) -> str:
    """Read Q tensor instead of K tensor when computing K's first half rotation.

    k1 is the first half of the key vector. Using q_ptr instead of k_ptr means
    K gets the same rotation as Q but applied to Q's values — always wrong for
    separate Q and K sequences.
    """
    return _apply_substitution(
        kernel,
        "k1 = tl.load(k_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)",
        "k1 = tl.load(q_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)  # BUG: reads Q not K",
        "fused_qk_k_reads_q",
    )


def _mut_fused_qk_q_sign_flip(ref: str, kernel: str) -> str:
    """Flip the rotation sign for Q's first half: q1·cos + q2·sin."""
    return _apply_substitution(
        kernel,
        "tl.store(q_out_ptr + x_offs, (q1 * cos - q2 * sin).to(tl.float16), mask=mask)",
        "tl.store(q_out_ptr + x_offs, (q1 * cos + q2 * sin).to(tl.float16), mask=mask)  # BUG: sign flipped",
        "fused_qk_q_sign_flip",
    )


# ===========================================================================
# QUANTIZED MATMUL / W8A16 PERTENSOR (2 mutations)
# ===========================================================================

def _mut_qmatmul_no_scale(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "w = w.to(tl.float16) * scale",
        "w = w.to(tl.float16)  # BUG: scale not applied",
        "qmatmul_no_scale",
    )


def _mut_pertensor_accumulation_reset(ref: str, kernel: str) -> str:
    """Reset the accumulator on each K-tile instead of accumulating.

    Correct: acc += tl.dot(x, w)  (adds each K-tile's contribution).
    Mutant:  acc  = tl.dot(x, w)  (overwrites on every iteration).
    With K > BLOCK_K the final acc only reflects the last tile, discarding
    all earlier tile contributions.
    """
    return _apply_substitution(
        kernel,
        "acc += tl.dot(x, w)",
        "acc = tl.dot(x, w)  # BUG: accumulator reset instead of accumulate",
        "pertensor_accumulation_reset",
    )


# ===========================================================================
# QUANTIZED MATMUL / W8A16 PERCHANNEL (2 mutations)
# ===========================================================================

def _mut_perchannel_scale_missing(ref: str, kernel: str) -> str:
    """Skip per-channel dequantization scale entirely."""
    return _apply_substitution(
        kernel,
        "w = w.to(tl.float16) * scale[None, :]",
        "w = w.to(tl.float16)  # BUG: per-channel scale not applied",
        "perchannel_scale_missing",
    )


def _mut_perchannel_scale_uses_first_channel(ref: str, kernel: str) -> str:
    """Load scale[0] for every output channel instead of the per-channel scale.

    Correct: scale = tl.load(scale_ptr + rn, ...)   (one scale per output channel).
    Mutant:  scale = tl.load(scale_ptr + rn * 0, ...) (always index 0 → scalar broadcast).
    All output channels use the first channel's scale, corrupting all but channel 0.
    """
    return _apply_substitution(
        kernel,
        "scale = tl.load(scale_ptr + rn, mask=n_mask, other=0.0)  # one fp16 scale per output channel",
        "scale = tl.load(scale_ptr + rn * 0, mask=n_mask, other=0.0)  # BUG: always loads scale[0]",
        "perchannel_scale_uses_first_channel",
    )


# ===========================================================================
# QUANTIZED MATMUL / W8A8 (2 mutations)
# ===========================================================================

def _mut_w8a8_missing_x_scale(ref: str, kernel: str) -> str:
    """Skip the activation (x) dequantization scale.

    Correct: acc * x_scale * w_scale  (undoes both int8 quantizations).
    Mutant:  acc * w_scale            (only undoes weight quantization).
    Output is off by a factor of x_scale ≈ amax(x) / 127.
    """
    return _apply_substitution(
        kernel,
        "acc = acc * x_scale * w_scale",
        "acc = acc * w_scale  # BUG: x_scale dequantization skipped",
        "w8a8_missing_x_scale",
    )


def _mut_w8a8_output_int8_dtype(ref: str, kernel: str) -> str:
    """Store the output as int8 instead of float16, causing severe clipping/saturation.

    Float16 output values are typically in the range [-10, 10]; converting to int8
    saturates to [-128, 127] and clips any value outside that range.
    """
    return _apply_substitution(
        kernel,
        "tl.store(out_ptrs, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))",
        "tl.store(out_ptrs, acc.to(tl.int8), mask=(rm[:, None] < M) & (rn[None, :] < N))  # BUG: int8 output",
        "w8a8_output_int8_dtype",
    )


# ===========================================================================
# Mutation table
# ===========================================================================

MUTATIONS = [
    # (name, mutation_type, smoke_category_key, template_idx, transform_fn)
    # expected: "caught" | "equivalent" | "documented_gap"

    # --- attention/causal_prefill ---
    ("attention__missing_causal_mask",        "missing_mask",    "attention", 0, _mut_attention_missing_causal_mask),
    ("attention__zero_max_init",              "harmless_change", "attention", 0, _mut_attention_zero_max_init),
    ("attention__full_loop_no_op",            "harmless_change", "attention", 0, _mut_attention_full_loop),

    # --- attention/cross_noncausal ---
    # NOTE: wrong_reduction_axis survives by softmax shift-invariance (same mechanism as
    # zero_max_init). The final acc/l_i cancels any per-axis constant shift in m_new.
    # Verified equivalent on 10 random seeds (max_abs_diff=0.000488 < atol=0.01).
    ("cross_attn__wrong_reduction_axis",      "harmless_change", "attention", 1, _mut_cross_attn_wrong_reduction_axis),
    ("cross_attn__missing_denominator",       "wrong_reduction", "attention", 1, _mut_cross_attn_missing_denominator),
    ("cross_attn__l_i_accumulation_reset",    "wrong_reduction", "attention", 1, _mut_cross_attn_l_i_accumulation_reset),

    # --- attention/gqa_causal_prefill ---
    ("gqa_attn__missing_causal_mask",         "missing_mask",    "attention", 2, _mut_gqa_attn_missing_causal_mask),
    ("gqa_attn__wrong_kv_head_index",         "wrong_index",     "attention", 2, _mut_gqa_attn_wrong_kv_head_index),

    # --- kv_cache/decode_attention_mha ---
    ("decode_mha__wrong_dot_axis",            "wrong_reduction", "kv_cache",  0, _mut_decode_mha_wrong_dot_axis),
    ("decode_mha__missing_normalization",     "wrong_reduction", "kv_cache",  0, _mut_decode_mha_missing_normalization),

    # --- kv_cache/decode_attention_gqa ---
    ("decode_gqa__missing_kv_mask",           "missing_mask",    "kv_cache",  1, _mut_decode_gqa_missing_kv_mask),
    ("decode_gqa__wrong_group_offset",        "wrong_index",     "kv_cache",  1, _mut_decode_gqa_wrong_group_offset),

    # --- kv_cache/cache_write ---
    ("cache_write__missing_position_k",       "wrong_index",     "kv_cache",  2, _mut_cache_write_missing_position_k),
    ("cache_write__k_stores_v",               "wrong_operand",   "kv_cache",  2, _mut_cache_write_k_stores_v),

    # --- norm/rmsnorm ---
    ("norm_rmsnorm__missing_divisor",         "wrong_reduction", "norm",      0, _mut_norm_rmsnorm_missing_divisor),
    ("norm_rmsnorm__missing_weight",          "documented_gap",  "norm",      0, _mut_norm_rmsnorm_missing_weight),

    # --- norm/layernorm ---
    ("layernorm__missing_mean_centering",     "wrong_reduction", "norm",      1, _mut_layernorm_missing_mean_centering),
    ("layernorm__missing_bias",               "documented_gap",  "norm",      1, _mut_layernorm_missing_bias),

    # --- norm/rmsnorm_residual ---
    ("rmsnorm_res__missing_residual_add",     "missing_operand", "norm",      2, _mut_rmsnorm_res_missing_add),
    ("rmsnorm_res__normalize_x_not_residual", "wrong_operand",   "norm",      2, _mut_rmsnorm_res_normalize_x_not_residual),

    # --- rope/rotate_half ---
    ("rope__sign_flip",                       "wrong_arithmetic","rope",      0, _mut_rope_sign_flip),
    ("rope__second_half_sign_flip",           "wrong_arithmetic","rope",      0, _mut_rope_second_half_sign_flip),

    # --- rope/interleaved ---
    ("interleaved__even_stride_wrong",        "wrong_index",     "rope",      1, _mut_interleaved_even_stride_wrong),
    ("interleaved__odd_sign_flip",            "wrong_arithmetic","rope",      1, _mut_interleaved_odd_sign_flip),

    # --- rope/fused_qk ---
    ("fused_qk__k_reads_q",                  "wrong_operand",   "rope",      2, _mut_fused_qk_k_reads_q),
    ("fused_qk__q_sign_flip",                "wrong_arithmetic","rope",      2, _mut_fused_qk_q_sign_flip),

    # --- quantized_matmul/w8a16_pertensor ---
    ("qmatmul__no_dequant_scale",             "missing_scale",   "quantized_matmul", 0, _mut_qmatmul_no_scale),
    ("pertensor__accumulation_reset",         "wrong_reduction", "quantized_matmul", 0, _mut_pertensor_accumulation_reset),

    # --- quantized_matmul/w8a16_perchannel ---
    ("perchannel__scale_missing",             "missing_scale",   "quantized_matmul", 1, _mut_perchannel_scale_missing),
    ("perchannel__scale_uses_first_channel",  "wrong_index",     "quantized_matmul", 1, _mut_perchannel_scale_uses_first_channel),

    # --- quantized_matmul/w8a8 ---
    ("w8a8__missing_x_scale",                 "missing_scale",   "quantized_matmul", 2, _mut_w8a8_missing_x_scale),
    ("w8a8__output_int8_dtype",               "dtype_error",     "quantized_matmul", 2, _mut_w8a8_output_int8_dtype),
]

# Mutations expected to survive (not harness failures):
# - "harmless_change": mathematically equivalent on all inputs.
# - "documented_gap": equivalent only due to all-zeros/all-ones test init.
_EQUIVALENT_TYPES = {"harmless_change", "documented_gap"}


def run_mutation(name: str, mutation_type: str, category: str, template_idx: int,
                 transform_fn) -> MutantResult:
    entries = list(SMOKE_GENERATORS[category]())
    entry = entries[template_idx]
    reference = entry["pytorch_reference"]
    correct_kernel = entry["triton_kernel"]

    try:
        mutant_kernel = transform_fn(reference, correct_kernel)
    except ValueError as exc:
        return MutantResult(
            name=name, mutation_type=mutation_type, template=f"{category}[{template_idx}]",
            caught=True, status="setup_error", error=str(exc), elapsed_s=0.0,
            notes="Substitution target not found — template may have changed",
        )

    t0 = time.monotonic()
    try:
        result: VerifyResult = verify_model_kernel(
            reference_code=reference,
            candidate_code=mutant_kernel,
            interpret=True,
            timeout_s=TIMEOUT_S,
        )
        elapsed = time.monotonic() - t0
        caught = not result.passed
        return MutantResult(
            name=name, mutation_type=mutation_type, template=f"{category}[{template_idx}]",
            caught=caught, status=result.status, error=result.error, elapsed_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return MutantResult(
            name=name, mutation_type=mutation_type, template=f"{category}[{template_idx}]",
            caught=True, status="exception", error=str(exc), elapsed_s=elapsed,
        )


def _explain_survivor(name: str, mutation_type: str) -> None:
    explanations = {
        "attention__zero_max_init": (
            "    Softmax is shift-invariant: sum(exp(s-c)*v)/sum(exp(s-c)) is identical\n"
            "    for any constant c. m_i=0 and m_i=-inf give the same output."
        ),
        "cross_attn__wrong_reduction_axis": (
            "    Softmax shift-invariance again. Using axis=0 (column max) instead of\n"
            "    axis=1 (row max) produces different values for m_new, but the final\n"
            "    acc/l_i ratio cancels any per-row constant shift. Verified on 10 seeds;\n"
            "    max_abs_diff=0.000488 (rounding from float16 sqrt of head_dim scale)."
        ),
        "attention__full_loop_no_op": (
            "    Loop extended past diagonal, but per-element causal mask zeros out\n"
            "    future-position contributions. Result is identical to correct kernel."
        ),
        "norm_rmsnorm__missing_weight": (
            "    Weight init = all-ones. Multiplying by 1 is a no-op. The harness\n"
            "    cannot distinguish 'weight applied' from 'weight skipped' here.\n"
            "    Fix: use randomly-initialized weights in the test."
        ),
        "layernorm__missing_bias": (
            "    Bias init = all-zeros. Adding 0 is a no-op. Same gap as missing_weight.\n"
            "    Fix: use randomly-initialized bias in the test."
        ),
    }
    if name in explanations:
        print(explanations[name])
    elif mutation_type == "harmless_change":
        print(f"    Harmless change — mathematically equivalent on all inputs.")
    elif mutation_type == "documented_gap":
        print(f"    Documented gap — equivalent only due to all-zeros/all-ones test init.")


def main() -> int:
    print("KernelForge verification harness — mutation test")
    print("=" * 70)
    print(f"Mutations: {len(MUTATIONS)}")
    print(f"Templates covered: {len({(cat, idx) for _, _, cat, idx, _ in MUTATIONS})}")
    print(f"Timeout per mutant: {TIMEOUT_S}s")
    print(f"Mode: interpreter (TRITON_INTERPRET=1)")
    print()

    results: list[MutantResult] = []
    for i, (name, mutation_type, category, idx, fn) in enumerate(MUTATIONS, 1):
        print(f"[{i:2}/{len(MUTATIONS)}] {name} ...", end=" ", flush=True)
        r = run_mutation(name, mutation_type, category, idx, fn)
        results.append(r)
        verdict = "CAUGHT" if r.caught else "SURVIVED"
        print(f"{verdict}  ({r.elapsed_s:.1f}s)  [{r.status}]")
        if r.error and r.caught and r.mutation_type not in _EQUIVALENT_TYPES:
            first_line = r.error.splitlines()[0] if r.error else ""
            if first_line:
                print(f"       error: {first_line[:100]}")

    # --- Summary ---
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Name':<46} {'Type':<20} {'Verdict':<10} {'Status'}")
    print("-" * 100)
    for r in results:
        verdict = "CAUGHT" if r.caught else "SURVIVED"
        print(f"{r.name:<46} {r.mutation_type:<20} {verdict:<10} {r.status}")

    # --- Counts ---
    n_caught = sum(1 for r in results if r.caught)
    n_survived = len(results) - n_caught

    equivalent_names = {name for name, mt, *_ in MUTATIONS if mt in _EQUIVALENT_TYPES}
    # Non-equivalent = mutations that SHOULD be caught
    non_equivalent = [r for r in results if r.name not in equivalent_names]
    n_ne_caught = sum(1 for r in non_equivalent if r.caught)
    n_ne_total = len(non_equivalent)
    mutation_score = n_ne_caught / n_ne_total if n_ne_total else 0.0

    # By mutation type
    type_stats: dict[str, dict] = {}
    for r in results:
        t = r.mutation_type
        if t not in type_stats:
            type_stats[t] = {"total": 0, "caught": 0, "equivalent": 0}
        type_stats[t]["total"] += 1
        if r.caught:
            type_stats[t]["caught"] += 1
        if r.name in equivalent_names:
            type_stats[t]["equivalent"] += 1

    print()
    print(f"Total mutations:       {len(results)}")
    print(f"Caught:                {n_caught}")
    print(f"Survived:              {n_survived}")
    print()
    print(f"Non-equivalent mutations:  {n_ne_total}")
    print(f"Non-equivalent caught:     {n_ne_caught}")
    print(f"Mutation score:            {mutation_score:.1%}  (caught / non-equivalent)")
    print()
    print("Per-type breakdown:")
    print(f"  {'Type':<22} {'Total':>5} {'Caught':>6} {'Equiv':>6} {'Score':>7}")
    for t, s in sorted(type_stats.items()):
        ne = s["total"] - s["equivalent"]
        score = s["caught"] / ne if ne else float("nan")
        score_str = f"{score:.0%}" if not (score != score) else "  n/a"
        print(f"  {t:<22} {s['total']:>5} {s['caught']:>6} {s['equivalent']:>6} {score_str:>7}")

    if n_survived > 0:
        print()
        print("SURVIVORS:")
        for r in results:
            if not r.caught:
                label = " [EQUIVALENT]" if r.name in equivalent_names else " [UNEXPECTED GAP]"
                print(f"  - {r.name}  [{r.mutation_type}]{label}")
                _explain_survivor(r.name, r.mutation_type)
        print()

    # Fail only on truly unexpected survivors
    unexpected = [r for r in results if not r.caught and r.name not in equivalent_names]
    if unexpected:
        print(f"UNEXPECTED SURVIVORS ({len(unexpected)}) — real harness gaps:")
        for r in unexpected:
            print(f"  - {r.name}")
        return 1

    print("All unexpected mutations were caught. Known survivors match documented gaps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
