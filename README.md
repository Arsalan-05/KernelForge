# KernelForge

**Training a small model (Qwen2.5-Coder-1.5B) to write Triton GPU kernels, with an automated checker that tests every kernel against PyTorch.**

**Demo studio:** [kernelforge-production.up.railway.app](https://kernelforge-production.up.railway.app/)

Generates a kernel with a general-purpose hosted model (qwen/qwen3-coder via an API), then verifies it with KernelForge's harness. The fine-tuned adapter is not served yet.

**Full documentation:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## What it does

- Given a PyTorch reference op, the model writes a Triton kernel plus a short natural-language explanation.
- A subprocess-isolated verification harness checks every kernel against PyTorch, following KernelBench's `Model`/`ModelNew` contract.
- Covers five LLM-serving operator categories (`quantized_matmul`, `attention`, `kv_cache`, `norm`, `rope`) built from 15 op templates.
- Fine-tunes Qwen2.5-Coder-1.5B-Instruct with supervised QLoRA on one Kaggle T4 GPU (no RL).

## Results

| Model | Quantization | Eval set | pass@1 | Status |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct (base) | 4-bit NF4 | 88 held-out (T4) | **0 / 88** | verified |
| Qwen2.5-Coder-1.5B-Instruct + QLoRA | 4-bit NF4 | 88 held-out (T4) | pending | re-run in progress |

Dataset, held-out templates and the baseline error breakdown: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#evaluation-details).

---

## Status

Harness and baseline verified on a Kaggle T4. Fine-tuning rerun in progress.

## Stack

Python, PyTorch, Triton, QLoRA, FastAPI, Docker, Kaggle T4

---

## Quick start (no GPU)

```bash
git clone https://github.com/Arsalan-05/KernelForge.git && cd KernelForge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/reproduce_all.sh   # harness + mutation test + report; expected: "All steps PASSED"
```

More commands (pre-flight, split-leakage check, mock-model server, container): [`docs/COMMANDS.md`](docs/COMMANDS.md).
GPU session and demo on Kaggle/Colab: [`docs/KAGGLE_RUN.md`](docs/KAGGLE_RUN.md).

## Tests

```bash
pytest tests/ -v   # some tests skip without Docker or triton
```

## Limitations

- Fine-tuned results are pending; only the base-model baseline is verified on a T4.
- Small model and narrow scope: about 300 training examples from 15 op templates, fp16 forward passes only.
- The CPU interpreter checks correctness only; it has no timing.
- All shape variants within a template share the same kernel logic, so there are 15 distinct kernels.
- Subprocess isolation + rlimits is suitable for research; it is not a hardened sandbox.
- Triton requires Linux + NVIDIA; macOS uses the CPU interpreter mode in Docker.
