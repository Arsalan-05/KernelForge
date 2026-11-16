#!/usr/bin/env python3
"""
Mutation test of the KernelForge verification harness.

Creates deliberately broken Triton kernel variants, runs each through the
interpreter harness (verify_model_kernel with interpret=True), and reports
how many are caught (incorrect result detected) vs survived (not caught).
Survivors indicate harness coverage gaps; each one is investigated below.

Usage (from repo root, inside the kernelforge-verify Docker image):

    docker run --rm -v "$PWD":/work kernelforge-verify python scripts/mutation_test.py

Takes ~3-5 minutes (smoke shapes, serial, interpreter mode).

What counts as "caught":
  - result.passed is False (harness correctly rejects the mutant)
  - status is "error", "timeout", or "crash" (mutant fails to run)

What counts as "survived":
  - result.passed is True (harness incorrectly accepts the mutant)
"""

from __future__ import annotations

import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Allow running from repo root or from within Docker (/work is the mount point)
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data.templates import SMOKE_GENERATORS
from verification.verify_kernel import VerifyResult, verify_model_kernel

# Interpreter mode is slow; give each mutant enough time.
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
# Mutation definitions
# ---------------------------------------------------------------------------
# Each mutation is a (name, mutation_type, template_key, transform_fn) tuple.
# transform_fn(reference, kernel) -> mutant_kernel applies the mutation to the
# kernel source (string) and returns the mutant.  The reference is unchanged.
# ---------------------------------------------------------------------------

def _apply_substitution(kernel: str, old: str, new: str, label: str) -> str:
    """Replace old with new exactly once; raise if not found."""
    count = kernel.count(old)
    if count == 0:
        raise ValueError(f"[{label}] substitution target not found:\n  {old!r}")
    if count > 1:
        raise ValueError(f"[{label}] substitution target found {count} times (expected 1):\n  {old!r}")
    return kernel.replace(old, new, 1)


# 1. attention/causal_prefill — missing causal mask
#    The correct kernel masks future keys:
#      s = tl.where((offs_m[:, None] >= cols[None, :]) & n_mask[None, :], s, float("-inf"))
#    Mutant: drops the causal part, so every key position is visible to every query.
def _mut_attention_missing_causal_mask(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "s = tl.where((offs_m[:, None] >= cols[None, :]) & n_mask[None, :], s, float(\"-inf\"))",
        "s = tl.where(n_mask[None, :], s, float(\"-inf\"))  # BUG: causal mask removed",
        "attention_missing_causal_mask",
    )


# 2. attention/causal_prefill — running-max initialised to 0 instead of -inf
#    Does NOT change the output: softmax is shift-invariant, so any initial
#    m_i value is absorbed by the l_i denominator.  This is a "harmless_change"
#    mutation — the kernel is semantically correct regardless of m_i init.
#    Included to verify the harness does NOT raise false positives on correct
#    (but suboptimal for numerical stability) code.
def _mut_attention_zero_max_init(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "m_i = tl.full((BLOCK_M,), float(\"-inf\"), dtype=tl.float32)",
        "m_i = tl.zeros((BLOCK_M,), dtype=tl.float32)  # NOTE: different init; output unchanged by shift-invariance",
        "attention_zero_max_init",
    )


# 3. attention/causal_prefill — extend loop past diagonal (no-op mutant)
#    The correct kernel stops at `hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)`.
#    Mutant: loops over all KV positions, but the per-element causal mask still
#    fires, so the results are identical.  Expected to SURVIVE (not caught) —
#    this tests that the harness doesn't falsely flag a correct-but-slower kernel.
def _mut_attention_full_loop(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)",
        "hi = seq_len  # NOTE: loop extended past diagonal; causal mask still applied",
        "attention_full_loop",
    )


# 4. norm/rmsnorm — missing divisor in RMS computation
#    Correct: mean_sq = tl.sum(x * x, axis=0) / n_cols
#    Mutant:  mean_sq = tl.sum(x * x, axis=0)
#    Effect: inv_rms is sqrt(n_cols) times smaller than it should be, so each
#    activation gets multiplied by sqrt(n_cols) — clearly wrong for n_cols > 1.
def _mut_norm_rmsnorm_missing_divisor(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "mean_sq = tl.sum(x * x, axis=0) / n_cols",
        "mean_sq = tl.sum(x * x, axis=0)  # BUG: should divide by n_cols",
        "norm_rmsnorm_missing_divisor",
    )


# 5. norm/rmsnorm — missing weight multiplication (all-ones gap mutant)
#    Correct: y = x * inv_rms * weight
#    Mutant:  y = x * inv_rms
#    Both Model and ModelNew initialise weight to ones, so for this entry the
#    mutant output equals the correct output.  Expected to SURVIVE — this is a
#    genuine harness coverage gap: testing with default (all-ones) weight
#    cannot detect a missing weight application.
def _mut_norm_rmsnorm_missing_weight(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "y = x * inv_rms * weight",
        "y = x * inv_rms  # BUG: weight skipped (survives due to all-ones init)",
        "norm_rmsnorm_missing_weight",
    )


# 6. rope/rotate_half — sign flip in first output half
#    Correct: tl.store(..., (x1 * cos - x2 * sin).to(tl.float16), mask=mask)
#    Mutant:  tl.store(..., (x1 * cos + x2 * sin).to(tl.float16), mask=mask)
#    Effect: the x1-half of the rotation gains the wrong sign, so the rotation
#    angle is reflected instead of applied — always incorrect for non-zero sin.
def _mut_rope_sign_flip(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "tl.store(out_ptr + x_offs, (x1 * cos - x2 * sin).to(tl.float16), mask=mask)",
        "tl.store(out_ptr + x_offs, (x1 * cos + x2 * sin).to(tl.float16), mask=mask)  # BUG: sign flipped",
        "rope_sign_flip",
    )


# 7. quantized_matmul/w8a16_pertensor — dequantisation scale skipped
#    Correct: w = w.to(tl.float16) * scale
#    Mutant:  w = w.to(tl.float16)
#    Effect: weights stay as int8-valued fp16; output is off by the scale
#    factor (typically ~0.001–0.1 for typical weight ranges).
def _mut_qmatmul_no_scale(ref: str, kernel: str) -> str:
    return _apply_substitution(
        kernel,
        "w = w.to(tl.float16) * scale",
        "w = w.to(tl.float16)  # BUG: scale not applied",
        "qmatmul_no_scale",
    )


MUTATIONS = [
    # (name, mutation_type, smoke_template_key, transform_fn)
    # smoke_template_key: index into the list returned by SMOKE_GENERATORS[category]
    ("attention__missing_causal_mask",  "missing_mask",    "attention",        0, _mut_attention_missing_causal_mask),
    ("attention__zero_max_init",        "harmless_change", "attention",        0, _mut_attention_zero_max_init),
    ("attention__full_loop_no_op",      "harmless_change", "attention",        0, _mut_attention_full_loop),
    ("norm_rmsnorm__missing_divisor",   "wrong_reduction", "norm",             0, _mut_norm_rmsnorm_missing_divisor),
    ("norm_rmsnorm__missing_weight",    "missing_weight",  "norm",             0, _mut_norm_rmsnorm_missing_weight),
    ("rope__sign_flip",                 "wrong_arithmetic","rope",             0, _mut_rope_sign_flip),
    ("qmatmul__no_dequant_scale",       "missing_scale",   "quantized_matmul", 0, _mut_qmatmul_no_scale),
]


def run_mutation(name: str, mutation_type: str, category: str, template_idx: int,
                 transform_fn) -> MutantResult:
    # Pick the smoke entry
    entries = list(SMOKE_GENERATORS[category]())
    entry = entries[template_idx]
    reference = entry["pytorch_reference"]
    correct_kernel = entry["triton_kernel"]

    # Apply mutation
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
            caught=caught, status=result.status,
            error=result.error,
            elapsed_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return MutantResult(
            name=name, mutation_type=mutation_type, template=f"{category}[{template_idx}]",
            caught=True, status="exception", error=str(exc), elapsed_s=elapsed,
        )


def main() -> int:
    print("KernelForge verification harness — mutation test")
    print("=" * 60)
    print(f"Mutations: {len(MUTATIONS)}")
    print(f"Timeout per mutant: {TIMEOUT_S}s")
    print(f"Mode: interpreter (TRITON_INTERPRET=1)")
    print()

    results: list[MutantResult] = []
    for i, (name, mutation_type, category, idx, fn) in enumerate(MUTATIONS, 1):
        print(f"[{i}/{len(MUTATIONS)}] {name} ...", end=" ", flush=True)
        r = run_mutation(name, mutation_type, category, idx, fn)
        results.append(r)
        verdict = "CAUGHT" if r.caught else "SURVIVED"
        print(f"{verdict}  ({r.elapsed_s:.1f}s)  [{r.status}]")
        if r.error and r.caught:
            # Print first line of error for context
            first_line = r.error.splitlines()[0] if r.error else ""
            if first_line:
                print(f"       error: {first_line[:100]}")

    # Summary table
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Name':<42} {'Type':<20} {'Verdict':<10} {'Status'}")
    print("-" * 90)
    for r in results:
        verdict = "CAUGHT" if r.caught else "SURVIVED"
        print(f"{r.name:<42} {r.mutation_type:<20} {verdict:<10} {r.status}")

    n_caught = sum(1 for r in results if r.caught)
    n_survived = len(results) - n_caught
    n_expected_survived = sum(
        1 for _, mt, *_ in MUTATIONS
        if mt == "harmless_change" or mt == "missing_weight"
    )

    print()
    print(f"Caught:   {n_caught}/{len(results)}")
    print(f"Survived: {n_survived}/{len(results)}")
    print()

    if n_survived > 0:
        print("SURVIVORS (harness coverage gaps):")
        for r in results:
            if not r.caught:
                print(f"  - {r.name}  [{r.mutation_type}]")
                _explain_survivor(r.name)
        print()

    # Distinguish expected survivors from unexpected ones.
    # Expected to survive: harmless_change and missing_weight types are documented gaps.
    expected_survivor_types = {"harmless_change", "missing_weight"}
    unexpected_survived = [
        r for r in results
        if not r.caught and r.mutation_type not in expected_survivor_types
    ]

    if unexpected_survived:
        print(f"UNEXPECTED SURVIVORS ({len(unexpected_survived)}) — these are real gaps:")
        for r in unexpected_survived:
            print(f"  - {r.name}")
        print()
        # Return non-zero only for genuinely unexpected survivors
        return 1

    print("All unexpected mutations were caught. Known survivors match documented gaps.")
    return 0


def _explain_survivor(name: str) -> None:
    explanations = {
        "attention__zero_max_init": (
            "    Why it survives: online softmax is shift-invariant.  The final output\n"
            "    is sum(exp(s_j - c) * v_j) / sum(exp(s_j - c)) for any constant c,\n"
            "    so choosing c=0 (m_init=0) vs c=-inf gives identical results.  The\n"
            "    mutant is semantically correct (but loses numerical stability for very\n"
            "    negative scores on hardware).  The harness correctly accepts it."
        ),
        "attention__full_loop_no_op": (
            "    Why it survives: the loop extension processes extra KV tiles that the\n"
            "    per-element causal mask (offs_m >= cols) zeros out anyway.  The result\n"
            "    is identical to the correct kernel — the mutant is correct-but-slower,\n"
            "    not wrong.  The harness correctly accepts it."
        ),
        "norm_rmsnorm__missing_weight": (
            "    Why it survives: both Model and ModelNew initialise their weight\n"
            "    parameter to all-ones.  Multiplying by ones is a no-op, so the mutant\n"
            "    output equals the correct output.  Gap: test inputs cannot distinguish\n"
            "    'weight applied' from 'weight skipped' when weight == 1 everywhere.\n"
            "    Fix: test with randomly-initialised weights (requires seeding both\n"
            "    Model and ModelNew to the same state before running)."
        ),
    }
    if name in explanations:
        print(explanations[name])


if __name__ == "__main__":
    sys.exit(main())
