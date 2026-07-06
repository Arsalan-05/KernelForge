# KernelForge

Fine-tuned LLM that generates and optimizes Triton GPU kernels from PyTorch ops — with automated correctness verification and speedup benchmarking against eager execution.

See [DOCUMENTATION.md](DOCUMENTATION.md) for the full project plan, architecture, and current status.

## Status

**Phase 2 (verification harness) — done, locally testable.**
**Phase 1 (dataset) — in progress:** schema approved (5 inference-serving
categories), 300 candidate op→kernel pairs generated via a template
library, and run through the harness — **0/300 verified so far, because
this machine has no CUDA GPU** (Triton requires Linux + NVIDIA; see below),
not because the kernels are known to be wrong. Real numbers require running
the same command on Kaggle/Colab. Everything else (fine-tuning, evaluation,
app) is not yet started.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Runs anywhere (CPU, no Triton needed) — validates the harness itself
pytest tests/ -v

# Runs only on Linux + NVIDIA GPU (Kaggle/Colab) — verifies a real Triton kernel
python verification/verify_kernel.py \
    examples/kernels/softmax/reference.py \
    examples/kernels/softmax/candidate_triton.py \
    --shape 4096 4096 --device cuda

# Generate + verify the full dataset (300 entries, 5 categories) — Kaggle/Colab only
python data/build_dataset.py --device cuda
```

## Repo layout

```
verification/   correctness + benchmark harness (verify_kernel.py, sandbox_runner.py)
examples/kernels/  hand-written Triton kernels used to prove out the harness (Phase 0)
tests/          CPU-only tests for harness mechanics (no GPU/Triton required)
data/templates/ template library generating dataset entries (5 categories x 3 templates x 20 shapes)
data/build_dataset.py  generates + verifies the full dataset -> data/verified/dataset.jsonl
data/SCHEMA.md  dataset entry schema + status
training/       LoRA/QLoRA fine-tuning scripts (Phase 4, not yet populated)
evaluation/     baseline vs. fine-tuned comparison (Phase 3/5, not yet populated)
app/            demo tool: backend + frontend (Phase 6, not yet populated)
```
