# KernelForge

A data-efficient, explainable LLM specialist for **LLM-serving kernel optimization** —
not a claim of beating large RL-trained systems on broad KernelBench coverage.
Given a PyTorch reference op, KernelForge generates an optimized Triton kernel
**and** a human-readable explanation of the optimization technique, with
automated correctness verification and speedup benchmarking.

See [DOCUMENTATION.md](DOCUMENTATION.md) for architecture and harness details.
See [docs/project_writeup.md](docs/project_writeup.md) for positioning, prior work, and evaluation design.

## Status

| Phase | State |
|---|---|
| 0 — Foundations | Done (example kernels) |
| 1 — Dataset | Pipeline done; GPU verification pending (run on Kaggle/Colab) |
| 2 — Verification harness | Done |
| 3 — Baseline eval | Done (`evaluation/baseline_eval.py`) |
| 4 — Fine-tuning | Done (`training/finetune.py`) |
| 5 — Evaluation + rubric | Done (`evaluation/finetuned_eval.py`, `explanation_review.py`) |
| 6 — Demo tool | Done (FastAPI + Gradio) |
| 7 — Write-up | Done (`docs/project_writeup.md`) |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Harness tests (CPU, no Triton)
pytest tests/ -v

# Generate + verify dataset (Linux + NVIDIA GPU)
python data/build_dataset.py --device cuda

# Baseline eval (un-fine-tuned model)
python evaluation/baseline_eval.py --device cuda

# Fine-tune (QLoRA via Unsloth on T4)
python training/finetune.py --config training/configs/finetune_default.json

# Fine-tuned eval + category comparison
python evaluation/finetuned_eval.py --adapter training/checkpoints/run_*/adapter --device cuda

# Demo tool
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
python app/frontend/gradio_app.py
```

Dry-run the eval pipeline locally (no GPU/model):

```bash
python evaluation/baseline_eval.py --dry-run --device cpu
```

## Repo layout

```
kernelforge/       shared prompts, parsing, dataset, model loading
verification/      correctness + benchmark harness
data/              schema, templates, build_dataset.py
training/          LoRA fine-tuning + configs
evaluation/        baseline/finetuned eval + explanation rubric
app/               FastAPI backend + Gradio frontend
docs/              project write-up (Phase 7)
examples/kernels/  Phase 0 hand-written kernels
tests/             harness + utility tests
```

## Positioning

This project explicitly cites KernelBench, TritonBench, KernelLLM, Kevin-32B,
AutoTriton, and related work. Its narrow contribution is **supervised specialization
on 5 LLM-serving kernel categories with explanation generation**, built on a
solo/free-tier compute budget without RL.
