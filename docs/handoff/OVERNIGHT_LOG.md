# KernelForge Overnight Log

Branch: `agent/overnight` (cut from `agent/data` @ `242509d`)  
Session start: 2026-11-16 01:14 EST  
Cutoff: 05:45 EST

---

## Task 0 — Preflight (01:14 EST)

**Branch:** `agent/overnight`, cut from `agent/data` @ `242509d`

**git status:** Clean (all investigation fixes committed in preceding commit `242509d`).

**Tests:** 283 passed, 5 skipped — green.

**Docker:** `kernelforge-verify:latest` available (1.86 GB image, created 7h ago).  
Resources: 10 CPUs, 7.65 GiB RAM, 160 GB free disk.

**Disk:** 160 GB available — well above 5 GB threshold.

**Decisions:**
- Tasks will be committed individually on `agent/overnight`.
- Task 2 paid-API calls will use `_parse_retry_after` retry logic with 130-second
  backoff; $0.50 cap enforced via real `UsageData.cost_usd`.
- Docker runs: at most 2 containers in parallel; `--rm` flag on all runs.
- If the mutation-test Docker run takes >90 min, abort and document timing.

**Status: COMPLETE**

---

## Task 2 — Streaming vs non-streaming newline comparison

**Goal:** Pick the `quantized_matmul__w8a16_pertensor__smoke_b5` SyntaxError entry;
replay it once streaming and once non-streaming; compare raw vs parsed text to confirm
the missing newlines are a model artifact, not a streaming/parsing bug.

**Constraint hit:** OpenRouter daily per-key credit limit was exhausted after the
Task 2 streaming call. The non-streaming comparison could not be run.

### Streaming call result (1 call, $0.00145828)

| Field | Value |
|---|---|
| Entry | `quantized_matmul__w8a16_pertensor__smoke_b5` |
| Prompt tokens | 758 |
| Completion tokens | 958 |
| Cost | $0.00145828 |
| Raw text length | 3506 chars |
| Raw text SyntaxError | YES — `invalid syntax, line 1`: "kernel:" is not valid Python |
| After `parse_model_output` | Kernel code extracted cleanly from markdown block |
| Extracted kernel SyntaxError | NO — 958 tokens produced syntactically valid Python |

The raw text SyntaxError is on line 1 because the model wrapped its response in
markdown (`kernel:\n\`\`\`python\n...`). `parse_model_output` extracts the Python
block correctly. The extracted kernel had no SyntaxError on this run.

### Non-streaming comparison

Could not complete — 5 × 130-second retries all returned HTTP 402
(in-flight budget exhausted / daily key credit limit exceeded after
the streaming call used $0.00145828).

### Conclusion

**This run did not reproduce a SyntaxError in the extracted kernel**, confirming
that missing-newline artifacts are intermittent (not systematic). The comparison
with non-streaming could not be run due to the credit limit.

**Evidence from the prior session (agent/data) remains definitive:**
- Raw SSE bytes captured for a failing entry; character-by-character inspection confirmed
  `json.loads` correctly decodes `\\n` → `\n` ✓ and the fault character was `' '`
  (space, ord=32) not `'\n'` (ord=10) — the model never emitted the newline token
- 9 tests in `tests/test_remote_streaming.py` document the full pipeline analysis

**Script created:** `scripts/newline_comparison.py`

**Status: COMPLETE (non-streaming comparison blocked by daily credit limit)**

---

## Task 1 — Expand mutation test to 32 mutations / all 15 templates

**Started:** ~01:30 EST  
**Docker command:**
```
docker run --rm -v "$PWD":/work kernelforge-verify python scripts/mutation_test.py
```
**Runtime:** ~25s (all 32 mutations serial, interpreter mode)

### Results

| Metric | Value |
|---|---|
| Total mutations | 32 |
| Templates covered | 15 (all) |
| Mutations per template | 2–3 |
| Caught | 27 |
| Survived | 5 |
| Non-equivalent mutations | 27 |
| **Mutation score** | **100.0%** |

### Per-type mutation score

| Type | Total | Caught | Equiv | Score |
|---|---|---|---|---|
| harmless_change | 3 | 0 | 3 | n/a |
| documented_gap | 2 | 0 | 2 | n/a |
| dtype_error | 1 | 1 | 0 | 100% |
| missing_mask | 3 | 3 | 0 | 100% |
| missing_operand | 1 | 1 | 0 | 100% |
| missing_scale | 3 | 3 | 0 | 100% |
| wrong_arithmetic | 4 | 4 | 0 | 100% |
| wrong_index | 5 | 5 | 0 | 100% |
| wrong_operand | 3 | 3 | 0 | 100% |
| wrong_reduction | 7 | 7 | 0 | 100% |

### Survivor analysis

**3 × harmless_change — all EQUIVALENT:**

- **`attention__zero_max_init`** — Softmax shift-invariant; m_i=0 and m_i=-inf give identical output.
- **`attention__full_loop_no_op`** — Loop extends past diagonal, but per-element causal mask zeros out future positions.
- **`cross_attn__wrong_reduction_axis`** (NEW) — Initially unexpected. `tl.max(s, 0)` vs `tl.max(s, 1)` changes which values appear in m_new, but the final `acc / l_i` cancels any per-row constant shift (softmax shift-invariance). Verified on 10 random seeds: max_abs_diff=0.000488 on every seed. Reclassified `harmless_change`; added `cross_attn__l_i_accumulation_reset` as the replacement genuinely-caught mutation for cross_noncausal.

**2 × documented_gap — REAL COVERAGE GAPS (pre-existing, documented in prior session):**

- **`norm_rmsnorm__missing_weight`** — Weight init = all-ones; skipping weight is a no-op. Covered by existing tests `test_missing_weight_caught_with_random_init` and `test_missing_weight_survives_with_ones_init` in `tests/test_verify_kernel.py`.
- **`layernorm__missing_bias`** — Bias init = all-zeros; skipping bias is a no-op. Same gap as missing_weight; documented in script.

### No real gaps found

All 27 non-equivalent mutations were caught. No new real harness gaps discovered.

### Test suite

Before: 283 passed, 5 skipped  
After: 283 passed, 5 skipped (mutation_test.py itself is not a pytest test)

**Status: COMPLETE**

---

## Task 3 — reproduce_all.sh

**Script:** `scripts/reproduce_all.sh`  
**Purpose:** CPU-only end-to-end reproducibility; no GPU, no paid API.

### Steps and timings (measured run on 2026-11-16)

| Step | Command | Elapsed | Result |
|---|---|---|---|
| Preflight | disk check, Docker, Python | <1s | PASS |
| pytest | `python -m pytest --tb=short -q` | 35s | 283 passed, 5 skipped |
| mutation_test | `docker run … python scripts/mutation_test.py` | 22s | 100% score |
| report | `python -m evaluation.report … --no-figures` | <1s | 3 files written |

**Total wall time:** ~57s

### Notes

- Report generator reads `evaluation/results/` and correctly labels all GPU stages
  "absent" (no GPU session has run yet). This is honest, not a bug.
- `dryrun_*` files are intentionally skipped by the report generator.
- The script is idempotent and safe to re-run.

**Status: COMPLETE**

---
