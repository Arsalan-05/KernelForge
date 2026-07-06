# KernelForge

Fine-tuned LLM that generates and optimizes Triton GPU kernels from PyTorch ops — with automated correctness verification and speedup benchmarking against eager execution.

See [DOCUMENTATION.md](DOCUMENTATION.md) for the full project plan, architecture, and current status.

## Status

**Phase 2 (verification harness) — done, locally testable.** Everything
else (dataset, fine-tuning, evaluation, app) is not yet started.

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
```

## Repo layout

```
verification/   correctness + benchmark harness (verify_kernel.py, sandbox_runner.py)
examples/kernels/  hand-written Triton kernels used to prove out the harness (Phase 0)
tests/          CPU-only tests for harness mechanics (no GPU/Triton required)
data/           dataset sourcing + verified training pairs (Phase 1, not yet populated)
training/       LoRA/QLoRA fine-tuning scripts (Phase 4, not yet populated)
evaluation/     baseline vs. fine-tuned comparison (Phase 3/5, not yet populated)
app/            demo tool: backend + frontend (Phase 6, not yet populated)
```
