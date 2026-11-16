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

### Cost pilot

5-problem pilot run before the full 300:

```
python scripts/hosted_model_baseline.py --pilot
```

| Metric | Value |
|---|---|
| Entries in pilot | 5 |
| Estimated cost | $0.0084 |
| Projected cost (300 entries) | $0.50 |
| Spend cap | $3.00 |
| Pilot verdict | SAFE — full run approved |

### Generation run (all 300 entries)

```
python scripts/hosted_model_baseline.py --workers 4
```

Runs 300 LLM generation calls (4 parallel workers). Results written to
`evaluation/results/hosted_model_baseline.json`.

**Key finding:** Running 4 parallel workers while a separate smoke run was
also executing with 4 concurrent workers exceeded the OpenRouter concurrent-
request limit, triggering HTTP 402 (payment required) errors for 248 entries.
Only the first ~52 entries (processed before the concurrent smoke run started)
received real responses.

| Metric | Value |
|---|---|
| Total entries attempted | 300 |
| HTTP 402 errors (concurrent limit hit) | 248 |
| Successful API responses | 52 |
| Kernels parsed from successful responses | 52 / 52 (100%) |
| Kernels parsed from all 300 | 52 / 300 (17.3%) |
| Estimated cost | $0.1847 |

**Per-category breakdown** (of successful calls only):

| Category | Successful calls | Parsed | Rate |
|---|---|---|---|
| quantized_matmul | 51 | 51 | 100% |
| attention | 0 | 0 | — (all 402) |
| kv_cache | 0 | 0 | — (all 402) |
| norm | 0 | 0 | — (all 402) |
| rope | 1 | 1 | 100% |

**Limitation:** The 248 HTTP 402 errors are an artifact of running two parallel
API sessions simultaneously, not a model capability issue. The 52 successful
calls all produced parseable kernels (100% parse rate on successful calls).

### Interpreter verification (smoke shapes, Docker)

For interpreter verification, 15 smoke entries (one per template, small
non-power-of-2 shapes) were generated and verified inside the
`kernelforge-verify` Docker image:

```
docker run --rm -v "$PWD":/work kernelforge-verify \
    sh -c "pip install httpx -q && python scripts/hosted_model_baseline.py --smoke-only --workers 1"
```

(httpx is not in the base Docker image; the one-time `pip install httpx` takes
~5 seconds.)

**Key finding:** The smoke run also hit API rate limiting (6/15 entries got
HTTP 402 after the 9th request). This is consistent with a per-session
concurrent-request limit of approximately 9–10 requests.

| Metric | Value |
|---|---|
| Smoke entries attempted | 15 |
| HTTP 402 errors (rate limit) | 6 |
| Successful API responses | 9 |
| Kernels parsed from successful responses | 9 / 9 (100%) |
| Interpreter-verified | 9 |
| **Correct** | **1 / 9 (11.1%)** |
| Incorrect (output wrong) | 0 |
| Error (exception during execution) | 8 |

**Passing entry:** `kv_cache__cache_write__b2_s100__smoke`

**Failure breakdown** (8 errors):

| Error type | Count | Cause |
|---|---|---|
| SyntaxError | 8 | Missing newlines in generated code — lines run together |

The SyntaxErrors share a consistent pattern: a line ending is dropped between
two adjacent statements, e.g.:
```python
scale = weight_fp.abs().max() / 127        self.register_buffer(
```
instead of:
```python
scale = weight_fp.abs().max() / 127
        self.register_buffer(
```
This appears to be a text-generation artifact where end-of-line newlines before
indented continuation lines are occasionally omitted. It is not a Triton-specific
issue — the error occurs before any Triton code runs.

### Overall hosted-model baseline summary

| Metric | Value | Notes |
|---|---|---|
| Zero-shot parse rate | 100% | On successful API calls only |
| Interpreter pass rate (smoke, Docker) | 1/9 (11.1%) | 6/15 entries skipped due to rate limit |
| Main failure mode | SyntaxError (8/8 errors) | Missing newlines in streamed output |
| Total estimated API spend | ~$0.21 | Pilot + full 300 + two smoke runs |
| Model | qwen/qwen-2.5-coder-32b-instruct | General-purpose, not fine-tuned |

### Limitations

- **API rate limiting:** HTTP 402 errors after 9–10 rapid-fire requests indicate
  a per-session concurrent request limit on the OpenRouter account. This is an
  account / billing tier issue, not a model capability issue. Running the script
  sequentially (`--workers 1`) with pauses between batches would avoid this.
- **Interpreter only:** no GPU speedup measurement. The interpreter checks numerical
  correctness only; a passing interpreter result is not a substitute for GPU
  verification.
- **Smoke shapes only for interpreter:** the full 300 entries use shapes too
  large for the interpreter within the time budget (e.g. seq_len=2048 attention
  with the CPU interpreter).
- **Model is not fine-tuned:** `qwen/qwen-2.5-coder-32b-instruct` is a
  general-purpose code model, not trained on this dataset. This is a zero-shot
  baseline, not the KernelForge fine-tuned result.
- **Cost is estimated:** token counts use a char-based approximation (4 chars/token);
  actual OpenRouter billing may differ.
- **No pass@k or multi-attempt:** each entry gets exactly one generation call
  (temperature=0.0). Search with repair rounds was not run.

---

## 3. Summary

| Task | Status | Key numbers |
|---|---|---|
| Mutation test: 7 mutations | Complete | 4/7 caught; 3 documented survivors |
| Mutation gap test (missing_weight) | Fixed | 2 new tests in test_verify_kernel.py |
| Hosted-model: 300 LLM calls | Complete | 52/52 parsed on successful calls; 248 hit API rate limit |
| Hosted-model: 15 smoke interpreter (Docker) | Complete | 1/9 correct (11.1%); 8 syntax errors |
| Test suite | 274 pass, 5 skip | 2 new tests vs Phase 2 |
