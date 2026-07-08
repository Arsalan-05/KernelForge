# KernelForge — Project Write-Up (Phase 7)

## Honest positioning (read this first)

KernelForge is a **data-efficient, narrowly-scoped exploration** of LLM-driven
Triton kernel generation for **LLM inference serving** — not a claim of beating
large RL-trained systems on broad KernelBench coverage. It targets two gaps the
literature leaves open:

1. **Data-efficient specialization** — how far can supervised fine-tuning on a
   small, curated dataset go without RL or 8B–32B-scale compute?
2. **Explainability** — every prior system outputs code + a benchmark number;
   none routinely produce a human-readable rationale for *why* the optimization
   works.

## Problem

LLM serving throughput is dominated by a small set of recurring GPU ops:
quantized matmul, attention (prefill), KV-cache decode, normalization, and RoPE.
Hand-optimized Triton kernels (e.g. Liger Kernel) show large wins, but writing
them is expert-heavy. LLMs can draft kernels, but correctness and speed must be
verified automatically — and practitioners still need to understand *what*
changed.

## Prior work (cite explicitly)

| Work | Contribution | Gap vs KernelForge |
|---|---|---|
| **KernelBench** (Stanford SIL, 2025) | 250 PyTorch ML workloads, standard eval harness | Broad generic ops; no explanations |
| **TritonBench** (Li et al., 2025) | Hardware-aware Triton benchmark | Frontier models <24% correct on hard kernels |
| **KernelLLM** (Fisches et al., 2025) | 8B Llama fine-tuned for PyTorch→Triton | No explanation field; RL-scale adjacent |
| **Kevin-32B** (Baronio et al., 2025) | 32B + multi-turn RL self-refinement | Large compute; no explanations |
| **AutoTriton / TritonRL** (2025) | RL for Triton | Heavy training budget |
| **AI CUDA Engineer** (Sakana, 2025) | Agentic iterative CUDA translation | Training-free but no explanation output |
| **Liger Kernel** | Hand-optimized Triton for LLM training | Not LLM-generated; relevant kernel patterns |

## This project's contribution

1. **Narrow scope:** 5 LLM-serving categories (`quantized_matmul`, `attention`,
   `kv_cache`, `norm`, `rope`) with 60 verified examples each (300 total target).
2. **Structured output:** kernel + `optimization_explanation` field, trained and
   surfaced in the demo UI as a first-class artifact.
3. **Reproducible harness:** subprocess-isolated `verify_model_kernel()` matching
   KernelBench's `Model`/`ModelNew` contract — same pass/fail definition at
   dataset-build time and eval time.
4. **Solo-compute budget:** QLoRA via Unsloth on a free-tier T4; no RL.

## Dataset methodology

- Template library: 3 hand-written templates per category × 20 shape variants.
- `data/build_dataset.py` generates entries and runs each through the harness;
  only passing entries land in `data/verified/dataset.jsonl`.
- Instruction-tuning projection: `description` + `pytorch_reference` →
  `triton_kernel` + `optimization_explanation`.

## Evaluation design

| Phase | Script | What it measures |
|---|---|---|
| 3 Baseline | `evaluation/baseline_eval.py` | Un-fine-tuned model: % correct, avg speedup, explanation presence |
| 4 Fine-tune | `training/finetune.py` | LoRA adapter on 85/10/5 split |
| 5 Eval | `evaluation/finetuned_eval.py` | Category-level before/after table |
| 5 Rubric | `evaluation/explanation_review.py` | Manual 0/1/2 scoring on 25 samples |

**Category-level results table** (fill after GPU runs):

| Category | Base: % correct | Base: avg speedup | FT: % correct | FT: avg speedup |
|---|---|---|---|---|
| quantized_matmul | | | | |
| attention | | | | |
| kv_cache | | | | |
| norm | | | | |
| rope | | | | |

**Explanation quality** (manual rubric, report honestly):
- 2 = identifies the specific optimization technique
- 1 = partially correct / generic
- 0 = wrong or missing

## Limitations (state explicitly)

- Small model (1.5B–7B) and ~300-example dataset vs. 8B–32B RL systems.
- Narrow category coverage vs. KernelBench/TritonBench breadth.
- No RL — purely supervised fine-tuning.
- Explanation quality has no automated metric; rubric review is manual.
- Real verification requires Linux + NVIDIA (Triton has no macOS build).

## Future work

- `@triton.autotune` in training targets for config search.
- Best-of-N rejection sampling via the harness (lightweight RL alternative).
- Post-training quant + adapter merge for cheaper serving of the codegen model.
- Widen shape grids / add templates for under-performing categories.

## Reproduction checklist

```bash
# 1. Verify dataset (Kaggle/Colab)
python data/build_dataset.py --device cuda

# 2. Baseline eval
python evaluation/baseline_eval.py --device cuda

# 3. Fine-tune
python training/finetune.py --config training/configs/finetune_default.json

# 4. Fine-tuned eval
python evaluation/finetuned_eval.py --adapter training/checkpoints/run_*/adapter --device cuda

# 5. Explanation rubric
python evaluation/explanation_review.py --report evaluation/results/finetuned_*.json --output evaluation/results/rubric_review.json

# 6. Demo tool
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
python app/frontend/gradio_app.py
```
