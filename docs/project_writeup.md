# KernelForge — Project Write-Up (Phase 7)

## Try it

Live demo: runs on demand from a Kaggle/Colab GPU session (`python app/launch.py --share` prints a
temporary `*.trycloudflare.com` link); a recorded walkthrough goes in `docs/assets/demo.gif`.

> **The demo currently runs the base model (Qwen2.5-Coder, not fine-tuned).** It shows the
> full pipeline — prompt → generated kernel → verification harness → correctness/speedup
> result — but generation quality reflects the base model, not a trained one. No
> fine-tuned adapter or evaluation numbers exist yet.

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

1. **Data-efficient specialization on LLM-serving ops:** 5 categories
   (`quantized_matmul`, `attention`, `kv_cache`, `norm`, `rope`) with 60 verified
   examples each (300 total target), rather than generic KernelBench breadth.
2. **Explainability:** kernel + `optimization_explanation` as a first-class output —
   the fine-tuning target, and already surfaced side by side in the demo UI.
3. **One verification harness, everywhere:** the subprocess-isolated
   `verify_model_kernel()` (KernelBench's `Model`/`ModelNew` contract) gates the
   dataset, the eval numbers, *and* what the live demo shows. Nothing in the demo is
   presented as verified unless the harness actually ran.

On top of that harness sits a **training-free, verifier-guided kernel search**
(`kernelforge/search.py`): best-of-N sampling (one batched generation, up to 4
candidates) filtered by the harness, then up to 2 repair rounds that give the model the
exact mismatch description or traceback from its best failed attempt. Kevin-32B learns
this multi-turn refinement with RL; here it runs at inference time on the same budget.

All of this is built on a solo-compute budget: QLoRA via Unsloth on a free-tier T4,
no RL.

## System architecture: one verification harness, everywhere

```
Web studio → FastAPI POST /generate/stream → kernel search (N samples → harness → select or repair)
          → base LLM (LoRA adapter later) + verification harness (GPU, or Triton interpreter) → result
```

The serving layer is a first-class component, not a wrapper bolted on at the end:

- **`app/web/`** — pick one of the 15 templates (or paste a PyTorch op), choose the
  harness (GPU or interpreter), the number of samples, and repair rounds. Every
  candidate streams into a timeline with its own verdict. The selected kernel is shown
  with its optimization explanation and a correctness + speedup card; each pipeline
  stage (GPU queue, generate, parse, verify, select) is shown as it happens. A
  persistent badge states which model variant is serving (base / fine-tuned / mock).
- **`app/backend/main.py`** — FastAPI service that runs the search, streams per-candidate
  events, caches harness results across requests, reports server stats, and serves the UI.
- **`kernelforge/search.py`** — the search engine, independent of FastAPI so evaluation
  can reuse it for pass@1 vs best-of-N comparisons.
- **`verification/`** — the same subprocess-isolated harness used everywhere.

The point: the harness that filters the dataset at build time, and that will grade the
baseline and fine-tuned models, is the same one grading what a visitor sees live in the
demo. Swapping the base model for the fine-tuned adapter is a config change
(`KERNELFORGE_ADAPTER_PATH`), not a new pipeline — so demo results and eval results are
directly comparable by construction.

## Dataset methodology

- Template library: 3 hand-written templates per category × 20 shape variants.
- `data/build_dataset.py` generates entries and runs each through the harness;
  only passing entries land in `data/verified/dataset.jsonl`.
- Instruction-tuning projection: `description` + `pytorch_reference` →
  `triton_kernel` + `optimization_explanation`.
- Local pre-check: every template runs through Triton's CPU interpreter in Docker on a
  small shape before any GPU time is spent. This caught real bugs, including attention
  references that raised on mixed-dtype matmuls and a W8A8 reference using an int32
  matmul CUDA doesn't support. After the fixes (and a move to FlashAttention-style tiled
  kernels), all 15 templates pass. The interpreter checks correctness only; speedups
  come from the GPU run.

## Evaluation design

| Phase | Script | What it measures |
|---|---|---|
| 3 Baseline | `evaluation/baseline_eval.py` | Un-fine-tuned model: % correct, avg speedup, explanation presence |
| 4 Fine-tune | `training/finetune.py` | QLoRA adapter; test set = one held-out template per category |
| 5 Eval | `evaluation/finetuned_eval.py` | Category-level before/after table |
| 5 Rubric | `evaluation/explanation_review.py` | Manual 0/1/2 scoring on 25 samples |
| 3 + 5 Search | `kernelforge/search.py` | pass@1 vs best-of-4 vs best-of-4 + 1 repair, same harness |

**Category-level results table** (fill after GPU runs):

| Category | Base: % correct | Base: avg speedup | FT: % correct | FT: avg speedup | FT best-of-4 + repair: % correct |
|---|---|---|---|---|---|
| quantized_matmul | | | | | |
| attention | | | | | |
| kv_cache | | | | | |
| norm | | | | | |
| rope | | | | | |

**Explanation quality** (manual rubric, report honestly):
- 2 = identifies the specific optimization technique
- 1 = partially correct / generic
- 0 = wrong or missing

## Limitations (state explicitly)

- Small model (1.5B–7B) and ~300-example dataset vs. 8B–32B RL systems.
- Narrow category coverage vs. KernelBench/TritonBench breadth.
- No RL — purely supervised fine-tuning; the search improves results at inference
  time only, at the cost of extra generation and harness runs per request.
- Explanation quality has no automated metric; rubric review is manual.
- Real verification requires Linux + NVIDIA (Triton has no macOS build). The CPU
  interpreter checks correctness only and is never reported as GPU-verified.

## Roadmap

Full plan: [`MASTER_PLAN.md`](../MASTER_PLAN.md).

1. **Demo + kernel search — built.** Remaining: the first live Kaggle GPU session and the GIF.
2. **Complete the dataset** — templates fixed and interpreter-checked; GPU-verify all 300.
3. **Baseline eval** — category-level numbers for the un-fine-tuned model, with and without search.
4. **Fine-tune the model** — QLoRA adapter, then swap it into the demo.
5. **Fine-tuned eval + explanation rubric** — fill in the results tables above.

## Future work

- `@triton.autotune` in training targets for config search.
- Rejection-sampling fine-tuning: run the search over the training prompts and fine-tune
  on kernels that pass and beat a speedup threshold (lightweight RL alternative).
- Post-training quant + adapter merge for cheaper serving of the codegen model.
- Widen shape grids / add templates for under-performing categories.

## Reproduction checklist

```bash
# 0. Demo (base model + live verification; Kaggle/Colab)
python app/launch.py --share

# 1. Verify dataset (local interpreter pre-check in Docker, then GPU on Kaggle/Colab)
docker run --rm -v "$PWD":/work kernelforge-verify python data/build_dataset.py --interpret --smoke --jobs 8
python data/build_dataset.py --device cuda

# 2. Baseline eval
python evaluation/baseline_eval.py --device cuda

# 3. Fine-tune
python training/finetune.py --config training/configs/finetune_default.json

# 4. Fine-tuned eval
python evaluation/finetuned_eval.py --adapter training/checkpoints/run_*/adapter --device cuda

# 5. Explanation rubric
python evaluation/explanation_review.py --report evaluation/results/finetuned_*.json --output evaluation/results/rubric_review.json

# 6. Demo with the fine-tuned adapter
python app/launch.py --share --adapter training/checkpoints/run_*/adapter
```
