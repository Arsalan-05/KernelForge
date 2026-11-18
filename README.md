# KernelForge

Training a small model (Qwen2.5-Coder-1.5B) to write Triton GPU kernels, with an automated checker
that tests every kernel against PyTorch. Given a PyTorch reference op, the model writes a Triton kernel
**plus** a short natural-language explanation. Correctness is checked by a subprocess-isolated
verification harness following KernelBench's `Model`/`ModelNew` contract.

Demo studio (runs the base model, fine-tuned adapter in progress): https://kernelforge-production.up.railway.app/

Scope: five LLM-serving operator categories
(`quantized_matmul`, `attention`, `kv_cache`, `norm`, `rope`) built from 15 op templates, supervised
QLoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct with a target of 60 verified examples per category
(about 300), one Kaggle T4 GPU, no RL.

## What is done vs not done

| Component | Status |
|---|---|
| Verification harness (`verification/`) | Done; Triton CPU-interpreter check works; validated on a Kaggle T4 GPU |
| Dataset (15 op templates, 5 categories) | Verified on a Kaggle T4 GPU: 274 of 300 reference kernels pass |
| Base-model eval | Run on a Kaggle T4 (4-bit, zero-shot) on the 88 held-out tasks: 0 passed |
| Canonical held-out split (`data/held_out_split.json`) | Done; held-out templates per category; no-leakage tests pass |
| Smaller-model pilot (MPS, local) | LoRA pilot run locally; results not reported |
| QLoRA fine-tuning (`training/finetune.py`) | First T4 run crashed (`warmup_ratio` removed in transformers v5); fix applied, rerun in progress |
| Mutation testing | Done; surviving mutations are mathematically equivalent (softmax shift-invariance) |
| Demo UI (`app/`) | Built; runs on the base model; Kaggle/Colab GPU needed for GPU mode |
| Container hosting (Railway) | Dockerfile + `railway.json` done; interpreter-only (no GPU on Railway) |
| GPU session runner (`scripts/gpu_session.py`) | Done; ran on a Kaggle T4 (dataset + baseline completed, finetune rerun in progress) |
| Test suite | Passes on CPU (macOS and Linux); some tests require Docker or a GPU |

**Fine-tuned results are pending.** The base-model baseline (0/88 on the held-out set) is verified on a T4.

## Quick start (fresh clone, no GPU)

```bash
git clone https://github.com/Arsalan-05/KernelForge.git
cd KernelForge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the CPU test suite
pytest tests/ -v
# Expected: tests pass (some skip without Docker or triton)

# CPU reproducibility check (harness + mutation test + report)
bash scripts/reproduce_all.sh
# Expected: "All steps PASSED"

# Pre-flight check (packages, configs, dataset status)
python scripts/preflight_check.py --no-gpu

# Split-leakage check
python scripts/check_split_leakage.py
# Expected: "PASSED — no leakage detected."
```

## GPU session (Kaggle T4)

Import [`notebooks/kaggle_gpu_session.ipynb`](notebooks/kaggle_gpu_session.ipynb) into Kaggle
(Accelerator: GPU T4, Internet: on). Or run directly:

```bash
# In a Kaggle notebook cell:
!git clone --depth 1 -b main https://github.com/Arsalan-05/KernelForge.git
%cd KernelForge
!pip install -q -r requirements-demo.txt
!python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"

# GPU-verify dataset + baseline eval (no fine-tuning):
!python scripts/gpu_session.py --budget-hours 11 --out /kaggle/working/kernelforge_outputs.zip

# Full run (+ QLoRA fine-tune + fine-tuned eval):
!python scripts/gpu_session.py --finetune --budget-hours 11 --out /kaggle/working/kernelforge_outputs.zip
```

All results are written to `kernelforge_outputs.zip`. See
[docs/handoff/KAGGLE_RUNBOOK.md](docs/handoff/KAGGLE_RUNBOOK.md) for the full runbook,
expected output files, resume instructions, and a post-session checklist.

## Results

| Model | Quantization | Eval set | pass@1 | Status |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct (base) | 4-bit NF4 | 88 held-out (T4) | **0 / 88** | verified |
| Qwen2.5-Coder-1.5B-Instruct + QLoRA | 4-bit NF4 | 88 held-out (T4) | pending | re-run in progress |

- **Hardware**: 2× Tesla T4 (one used), Kaggle notebook
- **Dataset**: 274 / 300 candidates verified on T4; split 167 train / 19 val / 88 held-out test
- **Held-out templates** (1 per family): `attention__gqa_causal_prefill`,
  `kv_cache__decode_attention_gqa`, `norm__layernorm`, `quantized_matmul__w8a16_perchannel`,
  `rope__fused_qk`
- **Baseline error breakdown** (88 entries, all generated-code errors, 0 harness errors):
  `tl.Tensor` hallucination (40), undefined helper (16), JIT-outside-kernel (10),
  syntax truncation (5), other (17)

Reference-kernel speedups are measurements of hand-written templates, not model output.

## Run the demo (Kaggle/Colab)

Triton requires Linux + NVIDIA GPU. In a Kaggle/Colab GPU notebook cell:

```bash
!git clone https://github.com/Arsalan-05/KernelForge.git
%cd KernelForge
!pip install -q -r requirements-demo.txt
!python app/launch.py --share
```

This prints a temporary public URL (`Public: https://….trycloudflare.com`). Open it, pick an
op, and hit **Generate**. The kernel streams token by token, then goes through parse and harness
verification. Set **Samples** (1–4) and **Repair** (off/1/2) to run a kernel search.

**Note:** the demo currently runs the base model (not fine-tuned). The fine-tuned adapter can be
loaded with `--adapter training/checkpoints/<run>/adapter` once training is done.

For local development without a GPU:

```bash
KERNELFORGE_MOCK_MODEL=1 uvicorn app.backend.main:app --port 8000
# open http://localhost:8000
```

## Run in a container (Railway or any container host)

The root `Dockerfile` and `railway.json` define a CPU-only container. Railway builds
the image, checks `/health`, and binds `$PORT`. **No GPU on Railway — kernels are checked for
correctness only via the Triton interpreter; there is no timing.**

Required service variables:

| Variable | Purpose |
|---|---|
| `KERNELFORGE_LLM_MODEL` | Model ID for an OpenAI-compatible API (e.g. `gpt-4o-mini`). Unset = mock mode. |
| `KERNELFORGE_LLM_BASE_URL` | API base URL (default `https://api.openai.com/v1`) |
| `KERNELFORGE_LLM_API_KEY` | API key (never passed to the kernel sandbox) |
| `KERNELFORGE_RATE_LIMIT` | Runs per visitor IP per minute (default 6) |

```bash
docker build -t kernelforge-web . && docker run --rm -p 8000:8000 kernelforge-web
```

## Repo layout

```
app/               FastAPI backend + streaming API + web studio UI + launcher
verification/      correctness + benchmark harness (subprocess sandbox, GPU or Triton interpreter)
kernelforge/       shared prompts, parsing, dataset, model loading, kernel search
docker/            CPU image with Triton's interpreter for local correctness checks
scripts/           gpu_session.py (one-command GPU run), fp32_baseline_eval.py, kaggle_eval.py, held_out_eval.py
notebooks/         Kaggle notebook that runs the GPU session
data/              schema, templates (15 total), build_dataset.py, held_out_split.json (canonical split)
training/          QLoRA fine-tuning + configs
evaluation/        baseline/fine-tuned eval + explanation rubric + report generator
docs/              paper skeleton, project writeup, handoff docs, results (populated post-GPU)
tests/             harness, library, search, API, split integrity (most run on CPU; some need Docker/GPU)
```

## Limitations

- **No GPU results yet.** The fine-tuning and GPU evaluation pipeline is written but has not been
  run on a GPU.
- **Small model and narrow scope.** Qwen2.5-Coder-1.5B, a target of about 300 training examples from
  15 op templates, five operator categories, fp16 forward passes only. Not comparable to larger
  RL-trained systems on broad benchmarks.
- **Interpreter ≠ GPU.** The CPU interpreter checks correctness only; it has no timing and uses
  much more memory than GPU mode. Shapes that run out of memory in the interpreter may fit on a T4;
  this is untested.
- **Template diversity.** Within a template, all shape variants share the same kernel logic and
  explanation. The effective number of distinct kernels is 15.
- **Sandbox.** Subprocess isolation + rlimits is suitable for research; it is not a hardened
  sandbox for untrusted code on a public server.
- **Triton requires Linux + NVIDIA.** macOS development uses the CPU interpreter mode in Docker.
