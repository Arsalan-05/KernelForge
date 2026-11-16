# Corrections to the repo docs (apply after the branches are merged)

Each entry gives the file, the line (as of `agent/d`, commit `4b63e8a`), the exact current text and the
exact replacement. Session A is editing `MASTER_PLAN.md` and `DOCUMENTATION.md` in parallel, so
**line numbers will drift. Match on the quoted text, not the line number.** If a quoted text no longer
exists after the merge, check whether Session A already fixed it.

The evidence for each correction is in [`SOURCES.md`](SOURCES.md) (citations) or the code file named.

---

## 1. TritonBench speedup claim (source contradicts it)

TritonBench reports the *best* speedup among correct operators: 1.56× on TritonBench-G and 1.91× on
TritonBench-T. Its best execution accuracy is 23.91% (G) and 53.01% (T). It makes no "rarely exceeds"
claim.

**`MASTER_PLAN.md`, line 48**

Current:
```
- **TritonBench** (Li et al., Feb 2025): Triton-specific benchmark; frontier models peak below 24% correctness on real-world kernels, and speedup rarely exceeds 1.5×.
```
Replacement:
```
- **TritonBench** (Li et al., ACL Findings 2025): Triton-specific benchmark with 184 real-world GitHub operators (TritonBench-G) and a PyTorch-aligned set (TritonBench-T). Best execution accuracy 23.91% on G and 53.01% on T; best speedup among correct operators 1.56× (G) and 1.91× (T).
```

## 2. Kevin generates CUDA, not Triton

Kevin (arXiv:2507.11948) is "the first model trained with multi-turn RL for CUDA kernel generation";
base model QwQ-32B.

**`MASTER_PLAN.md`, line 50**

Current:
```
- **Kevin-32B** (Baronio et al., 2025): multi-turn RL for kernel self-refinement. Our repair loop uses the same multi-turn feedback idea **at inference time, without training**.
```
Replacement:
```
- **Kevin-32B** (Baronio et al., 2025): QwQ-32B trained with multi-turn RL to write and refine **CUDA** (not Triton) kernels. Our repair loop uses the same multi-turn feedback idea **at inference time, without training**.
```

**`MASTER_PLAN.md`, line 37** (also covers item 4)

Current:
```
1. **Data-efficient specialization instead of broad, RL-heavy coverage.** Kevin-32B, AutoTriton and TritonRL rely on large-scale RL and heavy compute to chase generic operator coverage. Few ask how good a narrowly specialized model can get with a small, curated supervised dataset, zero RL, and a free-tier compute budget.
```
Replacement:
```
1. **Data-efficient specialization instead of broad, RL-heavy coverage.** Kevin-32B (CUDA), AutoTriton and TritonRL (Triton; supervised fine-tuning followed by RL) rely on RL and substantial compute to chase generic operator coverage. Few ask how good a narrowly specialized model can get with a small, curated supervised dataset, zero RL, and a free-tier compute budget.
```

**`README.md`, line 34**

Current:
```
behind Kevin-32B, run at inference time instead of trained with RL.
```
Replacement:
```
behind Kevin-32B (a CUDA model), run at inference time instead of trained with RL.
```

**`docs/project_writeup.md`, lines 64–65**

Current:
```
exact mismatch description or traceback from its best failed attempt. Kevin-32B learns
this multi-turn refinement with RL; here it runs at inference time on the same budget.
```
Replacement:
```
exact mismatch description or traceback from its best failed attempt. Kevin-32B learns
this multi-turn refinement with RL (for CUDA kernels); here it runs at inference time, without training.
```

`DOCUMENTATION.md` line 247 ("the inference-time version of what Kevin-32B trains with RL") and
`MASTER_PLAN.md` line 344 are accurate as written. No change needed.

## 3. KernelLLM is SFT-only

The KernelLLM model card describes supervised instruction fine-tuning only (about 25k PyTorch–Triton
pairs plus synthetic samples, 10 epochs, 16 GPUs, about 12 h). No RL.

**`MASTER_PLAN.md`, line 49**

Current:
```
- **KernelLLM** (Fisches et al., 2025): 8B Llama-3.1 fine-tuned for PyTorch→Triton. The closest prior work to the original framing.
```
Replacement:
```
- **KernelLLM** (Fisches et al., 2025): 8B Llama-3.1 Instruct, supervised fine-tuning only (no RL) on ~25k PyTorch→Triton pairs plus synthetic samples. The closest prior work: same training paradigm as ours, with ~100× more data and 16 GPUs.
```

**`docs/project_writeup.md`, line 41.** Already fixed on `agent/d` (commit `4b63e8a`). It previously read
`| **KernelLLM** (Fisches et al., 2025) | 8B Llama fine-tuned for PyTorch→Triton | No explanation field; RL-scale adjacent |`.

## 4. AutoTriton and TritonRL do SFT, then RL

Both papers train an 8B model with supervised fine-tuning first, then RL (AutoTriton: GRPO with
rule-based + execution rewards; TritonRL: verifiable rewards with Hierarchical Reward Decomposition).

**`MASTER_PLAN.md`, line 51**

Current:
```
- **AutoTriton / TritonRL** (Li et al., Woo et al., 2025): RL-trained Triton models.
```
Replacement:
```
- **AutoTriton / TritonRL** (Li et al., Woo et al., 2025): 8B Triton models trained with supervised fine-tuning followed by RL with execution-based rewards.
```

**`docs/project_writeup.md`, line 43.** Already fixed on `agent/d` (commit `4b63e8a`). It previously read
`| **AutoTriton / TritonRL** (2025) | RL for Triton | Heavy training budget |`.

## 5. "Flash-decoding style" overstates the decode templates

Flash-Decoding splits the KV sequence across programs and reduces the splits with a log-sum-exp step.
Our decode kernels use one program per (batch, head) or (batch, KV head) with a tiled online-softmax loop,
and do **not** split the KV sequence. `data/templates/kv_cache.py` line 7 already says
"without the split-KV second pass".

**`MASTER_PLAN.md`, line 150**

Current:
```
- Decode attention is now a tiled loop with online softmax (flash-decoding style). GQA packs a whole query group per KV head into one `tl.dot`.
```
Replacement:
```
- Decode attention is now a tiled loop with online softmax over the KV cache (one program per batch × head; no Flash-Decoding split-KV pass). GQA packs a whole query group per KV head into one `tl.dot`.
```

`DOCUMENTATION.md` line 622 ("Decode attention is now a tiled loop with online softmax") is accurate.
No change needed.

## 6. The 261/300 Docker pre-flight has no artifact in the repo

No report backs these numbers. `data/raw/` contains only `interpret_report.json` (the 15/15 smoke
check), and nothing records the full-grid run, the 39 OOM kills, or the 9–12 GB peak estimate.

**`MASTER_PLAN.md`, lines 393–396**

Current:
```
Every artifact (dataset, reports, logs, adapter) ends up in `/kaggle/working/kernelforge_outputs.zip`. Local pre-flight (Oct 1, 2026, Docker interpreter, all 300 grid shapes):
- **261/300 pass and 0 are incorrect.**
- The other 39 were killed for running out of memory (8 parallel jobs in an 8 GB Docker VM). They are the largest shapes: attention/RoPE at batch 8–16 × sequence 2048, and norms at 16k–65k rows.
- On a 16 GB T4 the largest references peak at roughly 9–12 GB, so a few may still run out of memory. The dataset report now lists `oom` separately from incorrect kernels (per category and per template), so a memory limit isn't mistaken for a broken template.
```
Replacement (option A, recommended: keep it but mark it unverified):
```
Every artifact (dataset, reports, logs, adapter) ends up in `/kaggle/working/kernelforge_outputs.zip`. Local pre-flight (Oct 1, 2026, Docker interpreter, all 300 grid shapes). **Unverified: no report from this run is in the repo, so treat these as notes, not results.**
- Observed: no shape failed as incorrect; the largest shapes (attention/RoPE at batch 8–16 × sequence 2048, norms at 16k–65k rows) were killed for running out of memory with 8 parallel jobs in an 8 GB Docker VM.
- The largest references may come close to the T4's 16 GB, so a few may still run out of memory on the GPU. The dataset report lists `oom` separately from incorrect kernels (per category and per template), so a memory limit isn't mistaken for a broken template.
- To make this a result, re-run `python data/build_dataset.py --interpret --jobs 2` in Docker and commit the report.
```
Replacement (option B: remove it):
```
Every artifact (dataset, reports, logs, adapter) ends up in `/kaggle/working/kernelforge_outputs.zip`. The dataset report lists `oom` separately from incorrect kernels (per category and per template), so a memory limit on the 16 GB T4 isn't mistaken for a broken template.
```

Notes on option A's command (checked in `data/build_dataset.py`):
- Without `--smoke`, the interpreter path runs the full 300-entry grid.
- It writes to the same `data/raw/interpret_report.json` as the smoke run, overwriting the committed
  15/15 smoke report. Copy that file aside first, or commit the full-grid report under a new name.
- `--jobs 2` is a guess at a setting that fits in 8 GB, not a tested value.

## 7. "None of these output a natural-language rationale" (too strong)

I checked abstracts and model cards only. TritonRL trains the model to "think" before coding, and
Kevin starts from a reasoning model, so their outputs include reasoning text. That's not the same as
an evaluated explanation field, but the absolute claim isn't supported.

**`MASTER_PLAN.md`, line 55**

Current:
```
None of these output a natural-language rationale. RL approaches need large domain-specific datasets and compute, so nobody reports what is achievable without that budget.
```
Replacement:
```
None of the works we reviewed treats a natural-language explanation of the optimization as an evaluated output (some produce reasoning traces, which are not scored as explanations). RL approaches need large domain-specific datasets and compute, and we found no report of what is achievable without that budget.
```

---

## Already correct (checked, no change)

- `README.md` line 18 and `MASTER_PLAN.md` line 221: "positioned against KernelLLM / Kevin-32B / AutoTriton, not as a claim of beating them".
- `README.md` line 167 and `docs/project_writeup.md` line 139: coverage limitation vs KernelBench/TritonBench.
- `docs/project_writeup.md` lines 40–43: prior-work rows, fixed on `agent/d` in `4b63e8a`.
