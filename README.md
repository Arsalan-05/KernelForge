# KernelForge

KernelForge is a **data-efficient Triton kernel generator** for LLM-serving operators. Given a
PyTorch reference op, it generates an optimized Triton kernel **plus** a short natural-language
explanation of the optimization. Correctness is checked by a subprocess-isolated verification
harness following KernelBench's `Model`/`ModelNew` contract.

Scope: five operator families relevant to LLM inference serving
(`quantized_matmul`, `attention`, `kv_cache`, `norm`, `rope`), supervised QLoRA fine-tuning of
Qwen2.5-Coder-1.5B-Instruct on a few hundred harness-verified examples, one Kaggle T4 GPU,
no RL.

## What is done vs not done

| Component | Status |
|---|---|
| Verification harness (`verification/`) | Done; CPU interpreter mode (285/300 pass, 15 OOM in 8 GB VM); **GPU mode validated on T4** (274/300 pass, 26 rejected) |
| Dataset (15 templates × 20 shapes = 300 candidates) | CPU: 285/300 pass; **GPU (T4): 274/300 verified** (12 incorrect, 9 timeout, 3 timeout, 2 OOM) |
| Base-model eval (fp32, local, 60 entries) | Done: **0/60 pass@1** (95% CI [0.0%, 6.0%]); every failure is a missing import or syntax error |
| Canonical held-out split (`data/held_out_split.json`) | Done: 180 train / 20 val / 100 test (5 held-out templates, 1 per category); 12 no-leakage tests pass |
| 0.5B base eval (MPS pilot, 20 entries) | Done: **0/20 pass@1** (95% CI [0.0%, 16.1%]); 80% parse rate, 30% truncated; 0.5B local LoRA MPS pilot |
| 0.5B LoRA fine-tuning (MPS pilot) | Done: 10/33 steps (~0.95 epochs); loss 1.3384→0.7778; adapter at `training/checkpoints/run_20261117T072658Z/adapter` |
| 0.5B fine-tuned eval (MPS pilot, 20 entries) | Done: **0/20 pass@1** (95% CI [0.0%, 16.1%]); **100% parse rate** (was 80% base); 2 reach Triton interpreter |
| T4 baseline eval (4-bit, 88 held-out entries) | Done: **0/88 pass@1**; dominant error: hallucinated `tl.Tensor` (40/88); 0 harness errors |
| QLoRA fine-tuning (`training/finetune.py`) | **Crashed on T4**: `warmup_ratio` removed in transformers v5 (Kaggle's image); fix: pin `transformers<5` |
| Mutation testing | 29/29 non-equivalent mutations caught; 3 surviving mutations are mathematically equivalent (softmax shift-invariance) |
| Demo UI (`app/`) | Built; runs on base model; Kaggle/Colab GPU needed |
| Deployment (Railway) | Dockerfile + `railway.json` done; interpreter-only (no GPU on Railway) |
| GPU session runner (`scripts/gpu_session.py`) | Done; **ran on T4** (dataset + baseline ok, finetune crashed) |
| Test suite | 332 tests collected; passes on CPU (macOS and Linux); some tests require Docker or a GPU |

**GPU fine-tuning crashed** on the first Kaggle T4 run (`warmup_ratio` removed in transformers v5).
The dataset (274/300) and baseline (0/88 pass@1) are validated on GPU. Fix the install pin and re-run.

## Quick start (fresh clone, no GPU)

```bash
git clone https://github.com/Arsalan-05/KernelForge.git
cd KernelForge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the CPU test suite
pytest tests/ -v
# Expected: ~332 passed, 1 skipped  (some tests skip without Docker or triton)

# CPU reproducibility check (harness + mutation test + report)
bash scripts/reproduce_all.sh
# Expected: "All steps PASSED" in ~60 s

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

# GPU-verify dataset + baseline eval (no fine-tuning, ~6-7 h on T4):
!python scripts/gpu_session.py --budget-hours 11 --out /kaggle/working/kernelforge_outputs.zip

# Full run (+ QLoRA fine-tune + fine-tuned eval, ~8-9 h):
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

<details>
<summary>Earlier local baselines (fp32 MPS, different entry subsets)</summary>

**1.5B base, fp32 on MPS, 60 entries** (`max_new_tokens=1536`, temperature 0):
0/60 pass@1 (95% CI [0.000, 0.060]). Dominant failure: 50/60 omit Triton imports.
The 60-entry subset has 30 entries overlapping train/val (leakage for fine-tuned eval);
the 15 clean held-out entries give 0/15 pass@1 (CI [0.000, 0.218]).

**0.5B base, fp32 on MPS, 20 entries**: 0/20 pass@1 (CI [0.000, 0.161]), 80% parse rate.
**0.5B fine-tuned (10/33 steps, MPS pilot)**: 0/20 pass@1, but parse rate rose to 100%
and 2 entries reached the Triton interpreter runtime.
</details>

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

## Deploy (Railway or any container host)

The root `Dockerfile` and `railway.json` define a CPU-only always-on deployment. Railway builds
the image, checks `/health`, and binds `$PORT`. **No GPU on Railway — kernels are checked for
correctness only via the Triton interpreter; there are no speedup measurements.**

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
tests/             332 tests: harness, library, search, API, split integrity (most run on CPU; some need Docker/GPU)
```

## Limitations

- **No GPU results yet.** The fine-tuning and GPU evaluation pipeline is complete but has not been
  run. All result numbers above come from the local CPU/MPS baseline.
- **Small model and narrow scope.** 1.5B parameters, ~300 training examples from 15 templates,
  five operator families, fp16 forward passes only. Not comparable to 8B–32B RL-trained systems
  on broad benchmarks.
- **Interpreter ≠ GPU.** The CPU interpreter checks correctness only; it has no timing and uses
  10–15× more memory than GPU mode. The 15 OOM shapes in the interpreter may fit on a T4; this
  is untested.
- **Template diversity.** Within a template, all 20 shape variants share the same kernel logic and
  explanation. The effective number of distinct kernels is 15, not 300.
- **Sandbox.** Subprocess isolation + rlimits is suitable for research; it is not a hardened
  sandbox for untrusted code on a production server.
- **Triton requires Linux + NVIDIA.** macOS development uses the CPU interpreter mode in Docker.
