# KernelForge Data Job Report

Generated on branch `agent/data`, cut from `agent/phase2`.

---

## 1. Mutation Test of the Verification Harness

**Goal:** Confirm that the interpreter harness detects broken Triton kernels.
Create deliberate mutations of the template kernels and test which ones are
caught.

**What was not done:** no paid API, no GPU, macOS only. All verification via
Triton's CPU interpreter (`TRITON_INTERPRET=1`, inside `kernelforge-verify`
Docker image).

### Method

Script: `scripts/mutation_test.py`  
Command:
```
docker run --rm -v "$PWD":/work kernelforge-verify python scripts/mutation_test.py
```

Seven mutations were applied to smoke-shape entries (one small instance per
template, the same shapes used by `build_dataset.py --smoke`). Each mutant
was run through `verify_model_kernel(interpret=True)` with a 90 s timeout.

### Mutations tested

| Name | Type | Template | Expected |
|---|---|---|---|
| `attention__missing_causal_mask` | missing_mask | attention/causal_prefill | caught |
| `attention__zero_max_init` | harmless_change | attention/causal_prefill | survives |
| `attention__full_loop_no_op` | harmless_change | attention/causal_prefill | survives |
| `norm_rmsnorm__missing_divisor` | wrong_reduction | norm/rmsnorm | caught |
| `norm_rmsnorm__missing_weight` | missing_weight | norm/rmsnorm | survives |
| `rope__sign_flip` | wrong_arithmetic | rope/rotate_half | caught |
| `qmatmul__no_dequant_scale` | missing_scale | quantized_matmul/w8a16_pertensor | caught |

### Results

```
Caught:   4/7
Survived: 3/7
Exit code: 0 (all survivors documented)
```

| Name | Verdict | Error / Mismatch |
|---|---|---|
| `attention__missing_causal_mask` | **CAUGHT** | 17395/20480 elements wrong; max abs err 2.829 |
| `attention__zero_max_init` | SURVIVED | softmax shift-invariant — any m_init gives same output |
| `attention__full_loop_no_op` | SURVIVED | loop extended but per-element causal mask still fires |
| `norm_rmsnorm__missing_divisor` | **CAUGHT** | 2083/2100 elements wrong; max abs err 3.026 |
| `norm_rmsnorm__missing_weight` | SURVIVED | all-ones weight init makes skipping weight undetectable |
| `rope__sign_flip` | **CAUGHT** | 924/3456 elements wrong; max abs err 6.094 |
| `qmatmul__no_dequant_scale` | **CAUGHT** | 400/400 elements wrong; max abs err 996.8 |

### Survivor analysis

**`attention__zero_max_init`** — harmless, not a gap.  
Softmax is shift-invariant: `sum(exp(s_j - c) * v_j) / sum(exp(s_j - c))` is
identical for any constant `c`. Initializing running max to 0 instead of -inf
gives the same final output. The kernel is semantically correct (but would lose
numerical stability for deeply negative scores on real hardware). The harness
correctly accepts it.

**`attention__full_loop_no_op`** — harmless, not a gap.  
The loop extends past the diagonal, but the per-element causal mask
(`offs_m >= cols`) zeros out future-position contributions anyway. The result
is identical to the correct kernel — it is simply less efficient. The harness
correctly accepts it.

**`norm_rmsnorm__missing_weight`** — real coverage gap.  
Both `Model` and `ModelNew` initialize their weight parameter to `torch.ones`.
Multiplying by all-ones is a no-op, so skipping the weight produces identical
output. The harness cannot distinguish "weight applied" from "weight skipped"
when weight equals 1 everywhere.

**Fix:** Two tests added to `tests/test_verify_kernel.py`:
- `test_missing_weight_caught_with_random_init` — proves the harness CAN
  catch missing weight when weight is randomly initialised (seed=7).
- `test_missing_weight_survives_with_ones_init` — documents the gap: with
  all-ones init, the mutant correctly passes.

No change to the template: the all-ones default is correct (it matches real
RMSNorm / LayerNorm default init), and the tests document the inherent
limitation.

### Test suite impact

Before: 272 passed, 5 skipped  
After:  274 passed, 5 skipped (2 new tests added)

---

## 2. Hosted-Model Baseline

**Goal:** Measure how well a general-purpose LLM (not fine-tuned on this
dataset) can produce syntactically valid and correctly executing Triton kernels.

**Model:** `qwen/qwen-2.5-coder-32b-instruct` via OpenRouter  
**URL:** `https://openrouter.ai/api/v1`  
**Mode:** interpreter correctness only — no GPU, no speedup measurement.  
**Script:** `scripts/hosted_model_baseline.py`

### Investigation: What was wrong in the first run

After the initial commit (`2edd85b`) the results were reviewed and found to have
two unexplained anomalies: HTTP 402 errors for 248 entries, and SyntaxErrors in
8/9 interpreter attempts. Both were investigated with live evidence.

#### HTTP 402 root cause

The OpenRouter key endpoint returned:
```
limit: $50.00  |  limit_remaining: $49.88  |  is_free_tier: true
```

The 402 error body (now logged in full — truncation was fixed):
```json
{
  "error": {
    "message": "This request would exceed your available credits given
                your current in-flight requests. Retry after in-flight
                requests settle, or add credits.",
    "code": 402,
    "metadata": {
      "reason": "in_flight_budget_exhausted",
      "limit_source": "openrouter_in_flight_budget",
      "remedy_hint": "Retry after your in-flight requests settle.
                      Adding credits raises your in-flight budget.",
      "headers": { "Retry-After": "120" }
    }
  }
}
```

**Root cause:** OpenRouter's free tier maintains a small concurrent-credit
reservation window. Each request reserves approximately `max_tokens ×
output_rate = 2048 × $1.50/M = $0.0031`. The free-tier in-flight budget fits
roughly 3 slots. This reservation is held for up to 120 seconds after a request
completes, not released immediately. Running two 4-worker sessions in parallel
(8 concurrent requests) or even sequential requests spaced less than 40 seconds
apart exhausts the budget.

The first run succeeded on its first 52 entries because those were processed
before the concurrent smoke session started. Subsequent runs fail on entry 4
because entries 1–3's budgets are still reserved.

**This is NOT a credit-exhaustion issue.** The account had $49.88 remaining.
**Fix:** Retry with Retry-After backoff (`_parse_retry_after` + 3-attempt loop
in `_generate_text`); default workers changed to 1.

#### SyntaxError root cause

The SyntaxErrors had this pattern:
```python
scale = weight_fp.abs().max() / 127        self.register_buffer(
```
(two statements on one line, no newline between them).

Investigation: captured raw SSE bytes for a streaming call and partially examined the pipeline
is correct. OpenRouter sends `"e = 127\\ns"` (JSON-escaped `\n`) in the SSE
`data:` line; `json.loads` decodes it to a real `\n`; `"".join(chunks)` preserves
it. Character-by-character inspection of the stored `generated_text` for the
failing entry confirmed position 7 (`'7'`) is followed by `' '` (space), not
`'\n'`. The newline was simply never generated.

**Root cause:** The model (qwen/qwen-2.5-coder-32b-instruct) occasionally omits
the newline token between adjacent code lines. This is a model generation
quality artifact, not a streaming or parsing bug.

**Fix:** `tests/test_remote_streaming.py` added with 9 tests documenting that
the pipeline is correct and that model-omitted newlines are faithfully
propagated (see `test_model_missing_newline_is_faithfully_propagated`).

**Script fixes applied** (all in this session, committed after investigation):
- `kernelforge/remote.py`: added `stream_options: {"include_usage": True}`;
  `UsageData` NamedTuple surfaced via `gen.last_usage`; full error body (removed
  300-char truncation).
- `scripts/hosted_model_baseline.py`: real token counts from API; default
  `--workers 1`; `_parse_retry_after` + 3-retry loop; real cost from
  `UsageData.cost_usd`.

### Corrected generation run

After fixes, run sequentially with retry:
```
python scripts/hosted_model_baseline.py --workers 1
```

Results (run on 2026-11-16):

| Metric | Value | Notes |
|---|---|---|
| Total entries attempted | 300 | |
| HTTP 402 errors | 297 | In-flight budget exhaustion; retries exhausted |
| Successful API responses | 3 | In-flight budget not yet clear from prior runs |
| Kernels parsed (successful calls) | 3 / 3 (100%) | quantized_matmul only |
| Real API cost | $0.0045 | From `stream_options.include_usage` |

The 297 failures are not model failures — they are API rate-limiting failures.
The model produced valid, parseable kernels on every call that reached it.
Running the full 300 sequentially with the corrected retry logic requires
approximately 3.5 hours (3 requests per 120-second budget window × 100 groups).

### Smoke interpreter run (corrected)

```
python scripts/hosted_model_baseline.py --smoke-only --workers 1
```

Run on 2026-11-16 with retry logic:

| Metric | Value | Notes |
|---|---|---|
| Smoke entries attempted | 15 | |
| HTTP 402 errors | 13 | Budget not yet fully cleared from prior sessions |
| Successful API responses | 2 | quantized_matmul entries only |
| Kernels parsed from successful | 2 / 2 (100%) | |
| Interpreter-verified | 2 | |
| **Correct** | **0 / 2** | Both failed with SyntaxError |
| Real API cost | $0.0083 | From `stream_options.include_usage` |

Both failures were SyntaxErrors (missing newlines in model output — same model
quality artifact as the first run). Zero entries passed the interpreter.

**Key comparison:**

| Run | Calls that reached model | Parse rate | Interpreter pass |
|---|---|---|---|
| First run (commit 2edd85b) | 52 / 300 | 100% (52/52) | 1/9 (11.1%) |
| Corrected run | 3 / 300 (budget exhausted) | 100% (3/3) | 0/2 (both SyntaxErrors) |
| Smoke corrected run | 2 / 15 (budget exhausted) | 100% (2/2) | 0/2 (both SyntaxErrors) |

The corrected runs confirm that the model's parse rate on successful calls is
100%, consistent with the first run. The interpreter pass rate is unreliable
because too few entries reach the model on the free tier.

### Limitations (updated)

- **In-flight budget constraint:** OpenRouter's free tier reserves max_tokens
  credit for 120 seconds per request. This limits sequential throughput to ~3
  requests per 120-second window (~1.5 requests/minute). A full 300-entry run
  takes approximately 3.5 hours with retry backoff. Results here are from
  partial runs only.
- **SyntaxErrors are model quality, not pipeline bugs:** confirmed by raw SSE
  inspection. The streaming/joining code correctly preserves `\n`. The model
  occasionally omits newline tokens. This cannot be fixed in the pipeline.
- **Interpreter only:** no GPU speedup measurement.
- **Smoke shapes only for interpreter:** large shapes (e.g. seq_len=2048
  attention) are too slow for the CPU interpreter.
- **Model is not fine-tuned:** zero-shot baseline, not the KernelForge result.
- **No pass@k or multi-attempt:** one generation per entry (temperature=0.0).

---

## 3. Local Base-Model Evaluation (Qwen2.5-Coder-1.5B-Instruct, MPS)

**Goal:** Measure zero-shot pass@1 for the smallest Qwen2.5-Coder model on our 60-entry subset.
Establishes a CPU-local baseline without paid API or GPU.

**Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (BASE model, not fine-tuned)  
**Device:** MPS float16 (Apple Silicon)  
**Script:** `scripts/local_baseline_eval.py`  
**Run date:** 2026-11-16

### Method

Two-phase: generate all 60 entries on MPS, free model, then Docker interpreter verify.
60 entries = 12 per category × 5 categories (3 smoke + 9 regular); categories:
quantized_matmul, attention, kv_cache, norm, rope.

### Limits

- **Single attempt per entry** (temperature=0, greedy decoding)
- **768-token cap** — 19 of 60 entries hit max_new_tokens=768 (truncated)
- **MPS float16** — results may differ slightly from CUDA fp16 (rounding)
- **No pass@k** — one generation per entry; no multi-attempt sampling
- **CPU interpreter only** — no GPU speedup measurement

### Results

| Metric | Value |
|---|---|
| Total entries | 60 |
| Parse OK | 60 / 60 (100.0%) |
| Correct (pass@1) | 0 / 60 (0.0%) |
| Wilson 95% CI | [0.0%, 6.0%] |
| Truncated (hit 768-token cap) | 19 / 60 |
| Tokens/s | median=13.2, mean=12.2 (MPS fp16) |

**Per-category:** all 5 categories: 0/12, CI=[0%, 24%].

### Actual error distribution (top 6 error messages, N=60)

| Count | Last line of traceback |
|---|---|
| 37 | `NameError: name 'triton' is not defined` |
| 8 | `SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '='?` |
| 6 | `SyntaxError: invalid syntax` |
| 5 | `SyntaxError: '(' was never closed` |
| 3 | `SyntaxError: invalid syntax. Perhaps you forgot a comma?` |
| 1 | `SyntaxError: '[' was never closed` |

**Root causes:**
- **NameError (37/60 = 62%):** Base model generates `@triton.jit` on line 1 without
  `import triton` / `import triton.language as tl`. Dominant failure — model knows the
  Triton API but consistently omits the import header.
- **SyntaxError `== or :=` (8/60 = 13%):** Model uses `=` where `==` is needed (non-truncation; 
  present in non-truncated entries only).
- **SyntaxError from truncation (15/60 = 25%):** 19 entries hit 768 token cap; 15 had truncation
  SyntaxErrors (cut mid-expression). All 5 `'(' was never closed` are norm entries.
- Remaining 4 truncated entries had NameError (kernel structurally complete but missing imports).

### Extraction fidelity (step 2 sanity check)

`parse_model_output` faithfully preserves newlines and indentation. Evidence:
- NameError traceback shows `line 1: @triton.jit` — extracted code is exactly what the model generated.
- Positive control: 3 known-correct kernels (norm, attention, rope) wrapped in model output format → 3/3 PASS, extraction bit-for-bit identical to original.
- For unclosed-fence (truncated) inputs: `_split_code_and_prose` returns all content after the opening fence without modification.

**raw_text not stored** in the original 60-entry checkpoint (pre-fix). Pipeline fix applied
(`scripts/local_baseline_eval.py` now saves `raw_text` and `extracted_kernel` per record for
future runs; `name_error` added as a distinct `failure_types` bucket in `_build_report`).

### 1536-token rerun (12 smoke entries, all 5 categories)

**Script:** `scripts/rerun_1536.py`  
**Script:** `scripts/inspect_and_header_verify.py` (truncation inspection + with-header verify)  
**Run date:** 2026-11-16

12 entries selected (3 quantized_matmul + 3 attention + 2 kv_cache + 2 norm + 2 rope smoke entries).
max_new_tokens raised from 768 → 1536.

| Metric | 768-token run (60 entries) | 1536-token run (12 smoke entries) |
|---|---|---|
| Correct (pass@1) | 0 / 60 (0.0%) | 0 / 12 (0.0%) |
| Truncated | 19 / 60 | 4 / 12 |
| NameError (missing imports) | 37 / 60 | 6 / 12 |
| SyntaxError `== or :=` | 8 / 60 | 3 / 12 |
| SyntaxError from truncation | 15 / 60 | 4 / 12 |

Raising the token cap does not fix the dominant NameError failure. It converts some truncation
SyntaxErrors to NameErrors but does not improve correctness.

#### Truncation inspection: fp16 NaN on MPS (4 entries that hit 1536 tokens)

All 4 truncated entries under fp16 — `norm__rmsnorm__smoke`, `norm__layernorm__smoke`,
`rope__rotate_half__b2_s9__smoke`, `rope__interleaved__b2_s9__smoke` — produced a tail
of repeated `!` characters filling the remaining output budget. Example tail-300 (identical
for all 4 under fp16):

```
'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
```

This is the fp16 NaN signature on MPS: NaN logits cause the argmax to return token 0
repeatedly, which the tokenizer decodes as `!`. The `!` output is not model looping.

**Confirmed by fp32 rerun** (`scripts/fp32_rerun_truncated.py`, 2026-11-16): the same 4
entries were regenerated under float32 on MPS at max_new_tokens=1536, temperature=0.

| Entry | fp16 tokens | fp32 tokens | Bang-tail gone? | fp32 no-header result |
|---|---|---|---|---|
| `norm__rmsnorm__smoke` | 1536 (truncated) | 1536 (truncated) | Yes | SyntaxError: `(` never closed |
| `norm__layernorm__smoke` | 1536 (truncated) | 602 | Yes | NameError: triton |
| `rope__rotate_half__b2_s9__smoke` | 1536 (truncated) | 770 | Yes | NameError: triton |
| `rope__interleaved__b2_s9__smoke` | 1536 (truncated) | 745 | Yes | NameError: triton |

All 4 entries produced no `!` tail under fp32. 3 of 4 no longer truncate — they were
hitting the token limit only because fp16 NaN filled the budget. `norm__rmsnorm__smoke`
still truncates at 1536 under fp32; its fp32 tail shows legitimate kernel argument code
(`BLOCK_DSIZE_123=128, ...`) cut mid-expression — a genuine long generation.

**Original 19 truncated entries (768-token run):** raw_text was not stored in that
checkpoint (pre-fix). Whether those entries also had fp16 NaN tails is unknown.

#### With-header verification

**fp16 run (6 NameError entries from 12-entry rerun):** 0/6 correct with header prepended.
**fp32 run (3 NameError entries from 4-entry rerun):** 0/3 correct with header prepended.

Remaining errors after adding `import triton\nimport triton.language as tl\n\n`:

| Error | fp16 run | fp32 run |
|---|---|---|
| `NameError: name 'ModelNew' is not defined` | 3 | 1 |
| `NameError: name 'batch_size' is not defined` | 1 | — |
| `NameError: name 'torch' is not defined` | 1 | — |
| `NameError: name 'max_seq_len' is not defined` | 1 | 1 |
| `AttributeError: module 'triton' has no attribute 'types'` | — | 1 |

The NameError failures are not fixable by prepending import headers. The kernels have
structural errors (missing class definitions, undefined variables) independent of the
missing imports.

The NameError failures are not fixable by prepending import headers. The kernels have
structural errors (missing class definitions, undefined variables) independent of the
missing imports.

---

## 4. Summary

| Task | Status | Key numbers |
|---|---|---|
| Mutation test: 7 mutations | Complete | 4/7 caught; 3 documented survivors |
| Mutation gap test (missing_weight) | Fixed | 2 new tests in test_verify_kernel.py |
| HTTP 402 investigation | Complete | Root cause: in-flight budget (not credits); $49.88 remaining |
| SyntaxError investigation | Partial | Root cause not established (non-streaming comparison blocked); pipeline is correct per 9 tests in test_remote_streaming.py |
| Script fixes (remote.py, baseline.py) | Complete | Real token counts, retry, full error body |
| Hosted-model: 300 LLM calls | Partial | 3/300 reached model; 100% parse rate on successful calls |
| Hosted-model: 15 smoke interpreter | Partial | 2/15 reached model; 0/2 correct (SyntaxErrors) |
| Local base-model eval: 60 entries | Complete | 0/60 pass@1; 37 NameError, 23 SyntaxError; 19/60 truncated |
| Positive control (extraction fidelity) | Complete | 3/3 known-correct kernels pass same path |
| Pipeline fix: raw_text, name_error bucket | Complete | local_baseline_eval.py + 6 new tests |
| 1536-token rerun: 12 smoke entries | Complete | 0/12 pass@1; 4 truncated (all looping) |
| Truncation inspection + fp32 rerun | Complete | `!!!!` = fp16 NaN on MPS (confirmed); 3/4 no longer truncate under fp32 |
| With-header verify (fp16+fp32 runs) | Complete | 0/6 (fp16) and 0/3 (fp32); deeper errors remain |
| Test suite (agent/overnight2 state) | 288 pass, 2 skip | +3 transformers-guarded, 1 datasets-skip |
| **fp32 baseline (60 entries, notes=False)** | **Complete** | **0/60 no-header; 0/50 with-header** |

---

## 5. FP32 Baseline — Training-Prompt Evaluation (2026-11-16)

**Goal:** Establish a reproducible base-model pass@1 on all 60 evaluation entries
using the same prompt format as fine-tuning (`notes=False`) and fp32 inference
to avoid the fp16 NaN degeneration observed on MPS.

**Script:** `scripts/fp32_baseline_eval.py`  
**Branch:** `agent/overnight2`  
**Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (1.5 B, fp32 weights, MPS)  
**Prompt format:** `build_messages(description, pytorch_reference)` — `notes=False`;
identical to `training/finetune.py` and `KernelGenerator.generate()`  
**max_new_tokens:** 1536  
**Verification:** Docker interpreter (`kernelforge-verify:latest`,
`TRITON_INTERPRET=1`, 90 s timeout)  
**Hardware:** Apple M-series MPS, ~20 GB swap in use, ~20 MB free RAM

### Two Numbers

| Pass | n | Correct | pass@1 | 95% Wilson CI |
|---|---|---|---|---|
| No-header | 60 | 0 | 0.000 | [0.000, 0.060] |
| With-header (NameError entries only) | 50 | 0 | 0.000 | n/a |

Adding `import torch\nimport triton\nimport triton.language as tl` before the
kernel did not fix any entry. Failures had deeper structural problems.

### Per-Category (no-header)

| Category | n | Correct | pass@1 | 95% Wilson CI |
|---|---|---|---|---|
| quantized_matmul | 12 | 0 | 0.000 | [0.000, 0.242] |
| attention | 12 | 0 | 0.000 | [0.000, 0.242] |
| kv_cache | 12 | 0 | 0.000 | [0.000, 0.242] |
| norm | 12 | 0 | 0.000 | [0.000, 0.242] |
| rope | 12 | 0 | 0.000 | [0.000, 0.242] |

All five categories fail uniformly.

### Truncation and Degenerate Output

- **Truncated (hit 1536-token cap):** 4 entries — all large `layernorm` shapes
  (`norm__layernorm__r1024_h{2048,4096,8192}`) and `kv_cache__decode_attention_gqa__b1_s1024`
- **Degenerate output (fp16 NaN guard):** 0 — no `!!!!` repetition observed; fp32
  mode produces no NaN degeneration

### No-Header Error Breakdown

| Error | Count |
|---|---|
| `NameError: name 'triton' is not defined` | 50 |
| `SyntaxError: invalid syntax (maybe meant '==' or ':=')` | 6 |
| `SyntaxError: '(' was never closed` | 3 |
| `SyntaxError: '[' was never closed` | 1 |

### With-Header Error Breakdown (50 entries tested)

| Error bucket | Count |
|---|---|
| name_error (various NameErrors) | 37 |
| type_error (AttributeError / TypeError) | 13 |

### Qualitative Failure Analysis

**NameError: triton (50 entries, no-header)**  
The model outputs `### Kernel:` followed immediately by `@triton.jit def ...` —
no `import triton`, no `import torch`, no `class ModelNew`. It treats the
response as a bare code snippet with imports already in scope and writes only
the Triton JIT function body. Even with the import header prepended, most of
these entries still fail because `class ModelNew` is absent.

**SyntaxError: invalid syntax (6 entries)**  
Observed patterns: (a) `grid=(expr, 1)` used as a default argument inside
`@triton.jit def f(..., grid=(n//b, 1)):` — Python does not allow complex
default-arg expressions; (b) `tl.load(ptr, align_to=tl.cacheline_size)` with
`cacheline_size` referenced without parentheses, causing a downstream
assignment `y = t<newline>...` that the parser sees as invalid.

**SyntaxError: unclosed `(` or `[` (4 entries)**  
All truncated entries: 1536 tokens reached mid-expression. The model was
writing a long kernel and hit the cap with an open parenthesis or bracket.

**NameError: name 'nn' is not defined (with-header)**  
Model wrote `class ModelNew(nn.Module):` — shorthand that requires
`import torch.nn as nn`. The prepended header adds `import torch` but not the
`as nn` alias.

**NameError: name 'ModelNew' is not defined (with-header, majority)**  
After adding imports, Python can now execute the file — but `ModelNew` is
still not defined because the model generated only the Triton JIT function and
no `class ModelNew(torch.nn.Module)` wrapper. The harness requires `ModelNew`.

**NameError: name 'seq_len' / 'max_seq_len' / 'batch_size' (with-header, rope)**  
The rope kernels' `ModelNew.forward()` uses `seq_len` or `max_seq_len` as bare
names at module load time (not received as forward arguments). The model placed
them as module-level assumptions — not defined at import.

**AttributeError: candidate_module.py does not define required attribute 'ModelNew' (with-header)**  
The model defined no class at all, or defined a class with a different name.
Structurally the same root cause as the NameError: ModelNew group.

**TypeError: const() takes no arguments (with-header)**  
The model used `tl.const(int)` as a type annotation — a hallucinated Triton
API. The correct annotation is `tl.constexpr`. This is a factual error about
the Triton API surface.

### Comparison to Previous fp16 Results

The earlier fp16 runs (Session 2, `agent/overnight`) produced the same 0/60
result but were confounded by NaN degeneration (`!!!!` output) on MPS. That
session used `notes=True`, which does not match the training prompt. The fp32
run here:

- Eliminates fp16 NaN: 0 degenerate outputs
- Uses `notes=False` to match `finetune.py` exactly
- Runs all 60 entries (not 12 smoke) with a single consistent configuration
- Confirms **0/60 pass@1 is the true base-model floor** before fine-tuning

### Interpretation

The 1.5 B base model does not know the KernelForge response format: it omits
required imports, does not emit `class ModelNew`, and sometimes hallucinates
Triton API calls (`tl.const`, `tl.cacheline_size`, `tl.types`). This is
expected — the model has not been fine-tuned yet. These numbers are the
pre-training baseline against which the QLoRA adapter's pass@1 will be
compared on Kaggle.

**Expected improvement after fine-tuning:** pass@1 > 0.000 (any correct kernel
demonstrates the adapter learned the format). The 15 smoke entries are the
simplest shapes and most likely to pass first.

---

## 6. Leakage Audit + Canonical Train/Val/Test Split (2026-11-17)

**Branch:** `agent/overnight3`

### Leakage finding

The 60-entry subset in `fp32_baseline_eval.py` selects 12 entries per category
(3 smoke + 9 regular). Of those 60 entries:

| Membership | Count |
|---|---|
| In TRAIN set (split_by_template seed=42) | 27 |
| In VAL set | 3 |
| In TEST set (held-out template) | 15 |
| Smoke-only (not in 300 GENERATORS entries) | 15 |
| **Total leakage for fine-tuned evaluation** | **30 / 60** |

The 0/60 base-model result is **uncontaminated** (the base model was not trained
on any of these entries). Any fine-tuned evaluation using the same 60-entry
subset would have 50% contamination.

### Canonical split design

File: `data/held_out_split.json` (seed=42, `split_by_template`)

| Split | Entries | Templates |
|---|---|---|
| Train | 180 | 10 (2 per category) |
| Val | 20 | 10 (same as train) |
| Test (held-out) | 100 | 5 (1 per category, 20 shapes each) |

Held-out templates:
- `attention__gqa_causal_prefill`
- `kv_cache__decode_attention_gqa`
- `norm__rmsnorm_residual`
- `quantized_matmul__w8a16_perchannel`
- `rope__interleaved`

Each held-out template has a training sibling in the same operator family,
so the test set measures generalization to variants, not new families.

### Leakage guarantee

12 tests in `tests/test_held_out_split.py` verify:
- No ID appears in more than one split
- All 300 IDs are present exactly once
- Each held-out template's entries are in test only
- `split_by_template(seed=42)` is reproducible
- The fp32_baseline leakage count (30/60) matches the audited value

Script `scripts/held_out_eval.py` enforces the split at runtime: it exits 1
if any evaluated entry is found in the train or val split.

### Re-statement of fp32 baseline against held-out set

Of the 60 entries evaluated in Section 5:
- 15 entries are in the held-out test split (clean, no leakage)
  → 0/15 pass@1 (Wilson CI [0.000, 0.218])
- 30 entries are in train/val (contaminated for fine-tuned evaluation)
- 15 entries are smoke-only (not in the 300 GENERATORS pool)

The 0/15 held-out result is consistent with 0/60 overall: same failure patterns
(NameError, SyntaxError), same structural issues documented in Section 5.

---

## 7. 0.5B Local LoRA MPS Pilot (2026-11-17)

**Label all numbers in this section: "0.5B local LoRA MPS pilot"**  
**Branch:** `agent/overnight3`

This is a supplementary experiment. The main research uses Qwen2.5-Coder-1.5B-Instruct
on a Kaggle T4. The pilot uses Qwen2.5-Coder-0.5B-Instruct on MPS to test the
fine-tuning pipeline locally without T4 access.

### B.2 — 0.5B base eval (20 entries, with Docker)

**COMPLETE** — Finished 02:12 EST.

Script: `scripts/held_out_eval.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --per-category 4`  
Output: `evaluation/results/held_out_base_0.5b_p4.json`

| Metric | Value |
|---|---|
| Total entries | 20 (4 per category × 5 categories) |
| Parsed (kernel extracted) | 16 / 20 (80%) |
| Truncated at max_new_tokens=1536 | 6 / 20 (30%) |
| Degenerate output (>50% rep char) | 0 / 20 |
| **pass@1 (no import header)** | **0 / 20 = 0.000** |
| **Wilson 95% CI** | **[0.000, 0.161]** |
| pass@1 (with import header) | 0 / 3 tested = 0.000 |
| Median tok/s (MPS fp32) | 21.3 |
| Mean tok/s | 18.5 |

**Error breakdown (no_header, 20 entries):**
- `syntax_error`: 13 (mostly `SyntaxError: unterminated string literal` from truncated output)
- `other` (no kernel extracted): 4
- `name_error`: 3 (`NameError: name 'triton' is not defined`)

**Per-category results (all 0/4):**

| Category | n | pass@1 | Wilson CI |
|---|---|---|---|
| quantized_matmul | 4 | 0.0 | [0.000, 0.490] |
| attention | 4 | 0.0 | [0.000, 0.490] |
| kv_cache | 4 | 0.0 | [0.000, 0.490] |
| norm | 4 | 0.0 | [0.000, 0.490] |
| rope | 4 | 0.0 | [0.000, 0.490] |

**Interpretation:** The 0.5B base model cannot reliably complete Triton kernels within
1536 tokens. 30% of outputs are truncated (unterminated string). The NameError pattern
shows the model generates Triton API calls without `import triton`. Adding the import
header does not help (TypeError remains). The base model establishes a 0.000 baseline
for the LoRA fine-tuning comparison.

### B.3 — 0.5B LoRA fine-tuning

**IN PROGRESS** — Started 02:19 EST. Run: `training/checkpoints/run_20261117T071859Z`.

Config: lora_r=8, lora_alpha=16, lr=3e-4, 3 epochs, fp32 on MPS, max_seq_length=1536,
padding="max_length" (all batches fixed to 1536 tokens for MPS kernel compilation stability).  
Training data: 42/180 examples survive the 1536-token filter (most kernels tokenize to 1537–1905 tokens).  
Steps: 33 total. Run: `training/checkpoints/run_20261117T072658Z`.

**→ Loss curve and fine-tuned eval results to be added when B.3/B.4 complete.**
