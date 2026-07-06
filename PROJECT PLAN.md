# Project Plan: KernelForge
### A data-efficient, explainable LLM specialist for LLM-serving kernel optimization

**Status:** Phase 2 complete (harness now supports both a bare-function contract and KernelBench's Model/ModelNew contract). Phase 1 in progress: schema finalized at 5 categories, 300 candidate entries generated via a template library, run through the harness — 0/300 verified so far, blocked on this machine lacking a CUDA GPU (not a correctness finding). See below.

---

## Positioning (read this before anything else)

This project sits in an active, fairly crowded research area — LLM-driven GPU kernel generation. Prior work already exists and is more resourced than a solo student project can compete with (see **Prior Work** below). This plan is deliberately **not** trying to beat that work on broad coverage or raw speedup numbers. Instead it targets two specific, still-open gaps:

1. **Data-efficient specialization over broad-but-shallow RL-heavy coverage.** Most existing systems (Kevin-32B, AutoTriton, TritonRL) rely on large-scale RL training and heavy compute to chase generic operator coverage. Almost nobody asks: *how good can a narrowly-specialized model get with a small, carefully curated supervised dataset and zero RL, on a real solo/free-tier compute budget?*
2. **Explainability.** Every system surveyed outputs a kernel plus a benchmark number. None of them generate a human-readable explanation of *why* the optimization works (which tiling/fusion/memory-access decision was made and why). This is a genuinely unclaimed, cheap-to-add angle.

**Scope narrowing:** instead of generic PyTorch ops (vector add, softmax, arbitrary matmul — the KernelBench/TritonBench territory), this project focuses specifically on **LLM-inference-serving kernel patterns** — the category of ops that actually matter for serving efficiency (relevant to Cohere's real business, and NVIDIA/Cerebras' compute-efficiency focus). Fewer categories, deeper coverage, each annotated with an explanation.

**Be honest about this in any write-up or interview.** State plainly: *"a small, data-efficient model specialized for LLM-serving kernel patterns, with explanation generation, built without RL or large-scale compute — explicitly positioned against and citing prior work, not claiming to surpass it on raw benchmarks."* That framing reads as informed and credible. Claiming novelty without acknowledging KernelLLM/Kevin-32B/AutoTriton would not.

---

## Prior Work (cite this explicitly in Phase 7 write-up)

- **KernelBench** (Stanford Scaling Intelligence Lab, Feb 2025) — the standard benchmark for evaluating LLM-generated GPU kernels; 250 PyTorch ML workloads.
- **TritonBench** (Li et al., Feb 2025) — hardware-aware benchmark specifically for Triton generation; shows even frontier models (GPT-o1, DeepSeek-R1) peak below 24% correctness on real-world complexity kernels, with speedup rarely exceeding 1.5x.
- **KernelLLM** (Fisches et al., 2025) — an 8B Llama-3.1-based model fine-tuned specifically for PyTorch→Triton translation. The closest direct prior work to this project's original framing.
- **Kevin-32B** (Baronio et al., 2025) — 32B model fine-tuned via multi-turn reinforcement learning for kernel self-refinement.
- **AutoTriton / TritonRL** (Li et al., Woo et al., 2025) — RL-trained models specifically for Triton programming.
- **AI CUDA Engineer** (Sakana AI, Lange et al., 2025) — agentic, training-free PyTorch-to-CUDA translation with iterative optimization.
- **Liger Kernel** — production hand-optimized Triton kernels for LLM training (not LLM-generated, but relevant prior art for the kernel patterns this project targets).

**Key fact from the literature that motivates the "explainability" gap:** none of the above output a natural-language rationale alongside the generated kernel — they output code and a benchmark number only.

**Key fact that motivates the "data-efficient/narrow" gap:** RL-based approaches require large-scale domain-specific datasets and substantial computational resources, which limits their practical applicability — nobody is asking what's achievable without that budget.

---

## Phase 0 — Foundations (Week 1–2) — ✅ Complete
Learned Triton basics, studied KernelBench structure, set up dev environment (Kaggle GPU notebook, dependencies).

---

## Phase 1 — Scope Lock & Dataset Strategy (Week 2–4) — 🔄 In progress (rescoped)

### 1.1 Task definition — locked
**Op → Kernel + Explanation**: input a PyTorch reference implementation of an LLM-serving-relevant op, output (a) a Triton kernel implementing it, and (b) a short natural-language explanation of the optimization technique applied.

### 1.2 Narrowed category scope — locked at 5
`norm` and `rope` ended up as two separate categories (not one combined
"norm_fusion") once template-writing started, since RMSNorm/LayerNorm and
RoPE are different enough op families to warrant separate template
libraries:
- **`quantized_matmul`** — W8A16 (per-tensor, per-channel) and W8A8 quantized linear layers
- **`attention`** — causal self-attention, non-causal cross-attention, grouped-query attention (all prefill/full-sequence)
- **`kv_cache`** — single-query decode-step attention (MHA and GQA), KV-cache write
- **`norm`** — RMSNorm, LayerNorm, fused residual-add + RMSNorm
- **`rope`** — rotate-half RoPE, interleaved-pair RoPE, fused Q+K RoPE

60 verified-candidate examples per category (the low end of the 60–150
target range, chosen for initial tractability — trivial to widen later,
see Phase 1.4).

### 1.3 Finalized schema
The actual implemented schema (`data/SCHEMA.md`) ended up richer than the
original draft below — field names changed (`reference_code`/`candidate_code`
→ `pytorch_reference`/`triton_kernel` to match KernelBench's `Model`/
`ModelNew` convention; `explanation` → `optimization_explanation`;
`verified`/`speedup` → a nested `verification` object), and it gained
`dtype`, `tolerance`, `test_shapes`, and `provenance` fields. Original draft
for reference:
```
{
  "id": "unique_id",
  "category": "quantized_matmul | attention | kv_cache | norm | rope",
  "instruction": "Write an optimized Triton kernel for: [op description]",
  "reference_code": "[PyTorch reference implementation]",
  "candidate_code": "[Triton kernel implementation]",
  "explanation": "[1-3 sentences: what optimization was applied and why — e.g., 'Fused the RMSNorm and activation into a single kernel to avoid an extra memory round-trip, since this op is memory-bandwidth bound.']",
  "input_spec": [{"shape": [...], "dtype": "...", "value_range": [...]}],
  "verified": true/false,
  "speedup": null  // filled in after running through verification harness
}
```
Schema proposed with 3 example entries (`data/examples/`), reviewed and
approved before bulk generation — done.

### 1.4 Dataset sourcing — bulk generation done, GPU verification pending
Rather than hand-writing 300 distinct kernels, built a template library
(`data/templates/`: 3 hand-written templates per category × 20 shape
variants) plus `data/build_dataset.py`, which generates every entry and
runs it through `verify_model_kernel()` — **every example must pass the
Phase 2 verification harness before being included in the final dataset**,
exactly as planned; unverified examples go to `data/raw/rejected.jsonl`,
not `dataset.jsonl`.

Ran on this machine: **300 generated, 0 verified, 300 rejected — all with
the identical error `"device 'cuda' requested but no CUDA device is
available"`**. This is an environment limit (Triton requires Linux +
NVIDIA; this machine is macOS with no GPU), not a finding about the 300
kernels' correctness — the uniform single-error-message pattern across all
300 is itself the evidence. Real pass/fail numbers require running
`python data/build_dataset.py --device cuda` on Kaggle/Colab — not done
yet.

**Deliverable for Phase 1:** `data/verified/dataset.jsonl` with 300+
verified (op, kernel, explanation) triples across the 5 narrowed
categories, each with a passing verification result. **Currently blocked**
on running the existing, complete pipeline on an actual GPU.

---

## Phase 2 — Verification Harness (Week 3–5) — ✅ Complete

`verification/verify_kernel.py` + `sandbox_runner.py` built and tested,
now supporting **two contracts**:
- **Function contract** (`verify_kernel`) — bare `solution(*tensors)`, used by the Phase 0 example kernels.
- **Model contract** (`verify_model_kernel`) — KernelBench's `Model`/`ModelNew` + `get_inputs`/`get_init_inputs`, which is what the Phase 1 dataset actually stores. Seeds the RNG immediately before *each* of `Model`/`ModelNew` construction (not just before `get_inputs()`), needed for ops with random internal state like quantized weights.

11 unit tests passing (6 function-contract: correctness, incorrect-candidate
detection, error handling, missing entry point, timeout, seed determinism;
5 model-contract: correctness, incorrect-candidate detection, missing
`ModelNew`, per-construction seeding, tuple-output comparison) — all on
CPU stand-ins, since Triton itself can't run on this machine.

Manually verified end-to-end with a real hand-written vector-add Triton
kernel: `correct=True`, `speedup=0.48x` (expected — vector add is
memory-bound and doesn't benefit from Triton over PyTorch's native op; real
gains show up on fused/compound ops, which is exactly what the narrowed
Phase 1 scope now targets).

The harness is task-agnostic (works on any reference/candidate code pair)
and is now the engine behind Phase 1's `build_dataset.py` as well as the
Phase 0 example kernels.

---

## Phase 3 — Model Selection & Baseline (Week 4–5)

1. **Candidate models** (all runnable on free-tier T4 with 4-bit quant + LoRA/Unsloth):
   - Qwen2.5-Coder (1.5B or 7B) — recommended default
   - DeepSeek-Coder (1.3B or 6.7B)
   - StarCoder2 (3B)
2. Run baseline eval: feed the narrowed category inputs to the un-fine-tuned base model, run through the verification harness, record % correct, average speedup, **and** manually assess whether the base model produces any explanation at all (it likely won't without prompting for it — note this as part of the "before" state).

**Deliverable for Phase 3:** Baseline report broken down by the 5 categories (not a single aggregate number) — this is important since some categories (e.g., quantized matmul) will likely be much harder than others (e.g., norm fusion), and category-level breakdown is what makes the eval in Phase 5 meaningful.

---

## Phase 4 — Fine-Tuning (Week 5–7)

1. **Method:** LoRA/QLoRA via Unsloth.
2. Train on the schema from 1.3 — the model needs to learn to output kernel code **and** the explanation field, so the training format should present both as expected output (e.g., structured as "kernel:\n...\n\nexplanation:\n...").
3. Split 85/10/5 train/val/test. Conservative hyperparameters: LoRA rank 16–32, lr ~2e-4, 3–5 epochs.
4. Expect 2–4 iteration cycles — watch for the explanation field specifically; it's easy for the model to produce generic/templated explanations rather than genuinely op-specific ones. Manually spot-check explanation quality every iteration.

**Deliverable for Phase 4:** Saved LoRA adapter/checkpoint + training logs.

---

## Phase 5 — Evaluation (Week 7–8)

1. Run held-out test set through the fine-tuned model → verification harness.
2. Compare against Phase 3 baseline, **broken down by category**:

   | Category | Base: % correct | Base: avg speedup | Fine-tuned: % correct | Fine-tuned: avg speedup |
   |---|---|---|---|---|
   | Quantized matmul | | | | |
   | Attention variants | | | | |
   | KV-cache ops | | | | |
   | Norm fusion | | | | |
   | RoPE | | | | |

3. **Evaluate the explanation field separately** — this doesn't have a clean automatic metric like correctness/speedup. Do a manual rubric-based review of 20–30 sampled explanations: does it correctly identify the actual optimization technique used? Is it specific to the op, or generic boilerplate? Score this qualitatively and report it honestly (e.g., "18/25 explanations correctly identified the specific optimization technique; 7 were generic or slightly inaccurate").
4. Be honest about failure modes and category-level weaknesses — this is expected given the harder categories (quantized matmul, attention) and is fine to report as-is.

**Deliverable for Phase 5:** Evaluation report with the category-level comparison table + explanation-quality rubric review + 3–5 concrete example outputs (good and bad).

---

## Phase 6 — Wrap It in a Tool (Week 8–10)

1. **Interface:** paste a PyTorch snippet (or pick from category-specific templates: quantized linear, attention block, KV-cache update, norm+activation, RoPE) → get back the Triton kernel, correctness check, benchmark comparison, **and the generated explanation displayed alongside the code**. The explanation is a differentiating feature of this tool versus prior work — make sure the UI treats it as a first-class output, not an afterthought.
2. **Stack:** FastAPI/Flask backend serving the fine-tuned model; Gradio frontend (fastest to build for "paste code → see result + explanation").

**Deliverable for Phase 6:** Deployed tool with public link + demo GIF showing the explanation feature specifically.

---

## Phase 7 — Write-Up & Positioning (Week 10–12)

1. **README structure:** problem → prior work (cited explicitly, see list above) → this project's specific, narrow contribution (data-efficient specialization on LLM-serving kernels + explainability) → dataset methodology → category-level results table → explanation-quality review → limitations → future work.
2. **State the honest framing directly in the README's first paragraph:** this is a data-efficient, narrowly-scoped exploration positioned against and citing existing broader systems, not a claim of beating them on raw benchmarks.
3. **Limitations to state explicitly:**
   - Small model / small dataset scale vs. 8B–32B RL-trained prior work
   - Narrow category coverage (5 LLM-serving-relevant categories) vs. broad generic op coverage in KernelBench/TritonBench
   - No RL — purely supervised fine-tuning
   - Explanation quality assessed via manual rubric, not an automated metric (there isn't a great one yet — worth naming as a real open problem)

**Deliverable for Phase 7:** Polished GitHub repo + README + optional blog post.

---

## Phase 8 — Efficiency & Scaling (Stretch goal, post-v1)

Once Phases 1–7 are done and you have real numbers, optional directions:

**A. Making the model itself cheaper to serve:**
- Post-training quantization (AWQ/GPTQ/bitsandbytes 8-bit or 4-bit) — benchmark accuracy before/after quantizing on your eval set.
- Merge LoRA adapters into base weights to remove adapter-lookup overhead.
- Distill the fine-tuned model into a smaller one trained on its own outputs.
- Serve via vLLM or TensorRT-LLM instead of plain `generate()` for faster inference (TensorRT-LLM is especially fitting given the NVIDIA-relevant framing).

**B. Making the generated kernels better/faster (the actual product quality — more valuable long-term):**
- Teach the model to emit `@triton.autotune`-decorated kernels that try multiple configs rather than fixed block sizes.
- Add more fused multi-op examples specifically, since single ops (like vector add) often don't beat PyTorch eager — the real wins are in fusion, which the narrowed category scope already leans toward.
- Rejection-sampling / best-of-N filtering using the verification harness as a data-quality filter: generate multiple candidates, keep only those that pass correctness **and** exceed a speedup threshold, fine-tune further on those (a lightweight alternative to full RL).

---

## Suggested Repo Structure

```
kernelforge/
├── README.md
├── data/
│   ├── raw/
│   ├── verified/              # dataset.jsonl lives here
│   └── build_dataset.py
├── verification/
│   └── verify_kernel.py       # done
├── tests/
│   └── test_verify_kernel.py  # done
├── training/
│   ├── finetune.py
│   ├── configs/
│   └── checkpoints/           # gitignored
├── evaluation/
│   ├── baseline_eval.py
│   ├── finetuned_eval.py
│   └── results/                # category-level tables + explanation rubric
├── app/
│   ├── backend/
│   └── frontend/
└── docs/
    ├── project_writeup.md
    └── demo.gif
```

---

## Timeline Summary

| Weeks | Phase | Status |
|---|---|---|
| 1–2 | Foundations | ✅ Done |
| 2–5 | Dataset (narrowed scope + explanation field) + verification harness | 🔄 Harness done; dataset schema/templates/generation pipeline done, 300 candidates generated, GPU verification pending (blocked on local hardware) |
| 4–5 | Model selection + category-level baseline | Not started |
| 5–7 | Fine-tuning (iterate 2–4x) | Not started |
| 7–8 | Category-level evaluation + explanation review | Not started |
| 8–10 | Tool (with explanation as first-class feature) | Not started |
| 10–12 | Write-up citing prior work honestly | Not started |
| Post-v1 | Efficiency & scaling stretch goals | Optional |

---

## Immediate Next Step

The dataset generation pipeline is complete and produces 300 candidate
entries across the 5 categories, but **verification is blocked on this
machine's lack of a CUDA GPU** — every entry currently reports
`status: "error"` ("no CUDA device available"), which is an environment
limit, not a finding about the kernels.

**Run this on Kaggle or Colab (Linux + NVIDIA GPU) to get real numbers:**
```bash
pip install -r requirements.txt
python data/build_dataset.py --device cuda
```
This populates `data/verified/dataset.jsonl` with whatever fraction of the
300 generated candidates actually compiles, runs, and matches the
reference within tolerance, and prints a per-category pass/fail/speedup
breakdown. Once that's in hand:
- For any template with a low pass rate, fix the template (not each
  individual failing shape variant) and regenerate.
- Decide whether 300 (60/category) is enough or whether to widen shape
  grids / add templates for under-represented categories before moving to
  Phase 3 (model selection + baseline).
