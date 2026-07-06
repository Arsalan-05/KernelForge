# Dataset schema — Op → Kernel pairs (inference-serving scope)

## Scope (narrowed from generic PyTorch ops)

Phase 1 targets **LLM-inference-serving kernel patterns** only, across five
categories (`norm_rope` was split into `norm` and `rope` after the initial
proposal, since normalization and rotary positional encoding are distinct
op families that warrant separate template libraries):

| Category | What it covers | Why it matters for positioning |
|---|---|---|
| `quantized_matmul` | INT8 weight/activation-quantized GEMM, dequant-fused GEMM | The core cost lever in serving — lower precision = more tokens/sec/GPU |
| `attention` | Causal/cross/grouped-query full-sequence attention (prefill) | The op that dominates prefill latency |
| `kv_cache` | Decode-step attention against a growing K/V cache, cache writes | The op that dominates decode latency/throughput — most direct link to Cohere/NVIDIA serving-efficiency framing |
| `norm` | RMSNorm, LayerNorm, fused residual-add + RMSNorm | Small but *frequent* ops; fusing them removes launch/memory overhead that adds up at serving scale |
| `rope` | Rotary positional embedding (rotate-half, interleaved-pair, fused Q+K) | Same frequency argument as `norm` — applied to every Q/K vector, every layer, every forward pass |

This replaces the earlier generic-op plan (elementwise/reduction/matmul
grab-bag) — those are still useful for Phase 0 harness sanity checks
(`examples/kernels/`) but are no longer the dataset's focus.

## Source-of-truth code convention

Both `pytorch_reference` and `triton_kernel` are stored using
**KernelBench's own convention** (`Model` / `ModelNew` classes +
module-level `get_inputs()` / `get_init_inputs()` functions), confirmed
from [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
(MIT-licensed):

```python
class Model(nn.Module):          # reference.py
    def __init__(self, *init_args): ...
    def forward(self, *runtime_args) -> Tensor | tuple[Tensor, ...]: ...

class ModelNew(nn.Module):       # triton_kernel candidate, same __init__/forward signature
    ...

def get_init_inputs(): ...       # -> list of args passed to __init__
def get_inputs(): ...            # -> list of tensors passed to forward
```

Reasons to match this convention instead of inventing our own:
- It's the convention KernelBench itself, its forks (KernelBench-v2,
  TritonForge), and most published LLM-kernel-generation work already use
  — synthetic generation prompts and any future comparison against those
  benchmarks stay apples-to-apples.
- `get_init_inputs()` cleanly handles ops that need constructor-time state
  (quantization scales, RMSNorm gain, a persistent K/V cache buffer) that a
  bare `solution(x, y)` function can't express without extra plumbing.
- KernelBench's own correctness/timing conventions are worth reusing
  directly: **tolerance is precision-dependent** — `float32: atol=rtol=1e-4`,
  `float16`/`bfloat16: atol=rtol=1e-2` — and **speedup = ref_runtime /
  candidate_runtime**, exactly matching what `verify_kernel.py` already
  computes.

**Adapter built:** `verification/verify_kernel.py` now exposes
`verify_model_kernel(reference_code, candidate_code, ...)` alongside the
original function-contract `verify_kernel()`. It loads `Model`/`ModelNew`,
seeds the RNG identically immediately before *each* construction (not just
before `get_inputs()`) — required for any op with random internal state,
like the quantized-matmul templates' weights — then times and compares
`Model.forward` against `ModelNew.forward` exactly as the function contract
does. Covered by `tests/test_verify_kernel.py` (correct/incorrect/missing-
class/seeded-random-init/tuple-output cases, all on CPU stand-ins).

## Field reference

| Field | Type | Notes |
|---|---|---|
| `id` | string | stable slug, e.g. `kv_cache__decode_attention_mha__b4_s1024` |
| `category` | enum | one of `quantized_matmul`, `attention`, `kv_cache`, `norm`, `rope` |
| `op_name` | string | short human name |
| `description` | string | plain-English op description; becomes the instruction-tuning "Instruction" field |
| `dtype` | enum | `float16` \| `bfloat16` \| `float32` — inference-serving examples default to `float16` |
| `pytorch_reference` | string | full source of `Model` + `get_inputs()` + `get_init_inputs()` |
| `triton_kernel` | string | full source of `@triton.jit` kernel(s) + `ModelNew` |
| `optimization_explanation` | string | **new field** — plain-English explanation of the fusion/tiling/quantization technique applied and why it helps at serving scale. This is what's shown alongside the code in the Phase 6 tool, and what the fine-tune should learn to produce, not just the code. |
| `tolerance` | `{atol, rtol}` | precision-dependent, per KernelBench's convention above |
| `test_shapes` | list | optional extra `get_init_inputs()`/`get_inputs()` overrides for benchmarking at prefill-like vs decode-like sizes |
| `provenance.source` | enum | `hand_written` \| `kernelbench_adapted` \| `synthetic_llm` |
| `provenance.kernelbench_ref` | string \| null | e.g. `level1/19_ReLU.py`, if directly adapted from KernelBench |
| `provenance.license` | string \| null | `MIT` when `kernelbench_ref` is set |
| `provenance.notes` | string | anything else worth recording about where this came from |
| `verification.verified` | bool | `false` until run through `verify_kernel.py` on a real GPU |
| `verification.correct` | bool \| null | populated post-verification |
| `verification.speedup` | float \| null | populated post-verification |
| `verification.device` | string \| null | e.g. `"NVIDIA T4 (Kaggle)"` |
| `verification.verified_at` | string \| null | ISO date |

## Instruction-tuning projection

At training time, each verified entry projects to the instruction-tuning
triple from the original plan, with the explanation appended to the
output rather than dropped:

```
Instruction: {description}
Input:       {pytorch_reference}
Output:      {triton_kernel}
             # Optimization: {optimization_explanation}
```

## Status

Schema approved; bulk generation built and run. `data/build_dataset.py`
combines a template library (`data/templates/`, 3 parameterized templates
per category x 20 shape variants = 60 entries/category, 300 total) with
`verify_model_kernel()` to produce `data/verified/dataset.jsonl` (passing
entries only) and `data/raw/rejected.jsonl` (everything else, with the
failure attached).

**All 300/300 are currently rejected** — every one reports
`status: "error"`, `"device 'cuda' requested but no CUDA device is
available"`. This machine has no CUDA GPU and Triton has no macOS/Windows
or CPU backend, so `verify_model_kernel(..., device="cuda")` cannot get past
its device check locally; this is an environment limit, not a correctness
signal on the 300 generated kernels. The harness's own mechanics (subprocess
sandboxing, per-construction seeding, tuple-output comparison, timeout/
crash handling) are validated separately and locally via
`tests/test_verify_kernel.py`'s CPU stand-ins. To get a real pass/fail
verdict, run the identical command on Kaggle/Colab:

```bash
pip install -r requirements.txt   # picks up triton on Linux automatically
python data/build_dataset.py --device cuda
```

The 3 examples in `data/examples/` (from the original schema proposal)
remain as documentation/reference and are separate from the 300 bulk-
generated entries above.
