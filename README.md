# KernelForge

<!-- Replace with a real capture from the first Kaggle/Colab run (base model, badge visible):
     ![KernelForge demo](docs/assets/demo.gif) -->

**Try it:** the demo runs on a Kaggle/Colab GPU session — see [Run the demo](#run-the-demo). A recorded
walkthrough (`docs/assets/demo.gif`) will be added from the first live GPU run.

> **Demo status: running the base model** (Qwen2.5-Coder-1.5B-Instruct, not fine-tuned). The fine-tuned
> LoRA adapter drops into the same pipeline once training is done. The UI shows which model is running
> in a banner at the top.

KernelForge is a **data-efficient, explainable LLM specialist for LLM-serving GPU kernels**. Given a
PyTorch reference op, it generates an optimized **Triton** kernel **plus** a short natural-language
explanation of *why* the optimization helps. Correctness and speedup are checked by a
**subprocess-isolated verification harness** that matches KernelBench's `Model`/`ModelNew` contract.
It's deliberately narrow (five serving-relevant categories, supervised QLoRA on a small curated
dataset, no RL) and is positioned against KernelLLM / Kevin-32B / AutoTriton, not as a claim of
beating them on broad benchmarks.

**Three contributions:**
1. **Data-efficient specialization** on LLM-*serving* ops (`quantized_matmul`, `attention`, `kv_cache`,
   `norm`, `rope`), not generic KernelBench breadth.
2. **Explainability**: kernel + optimization rationale as a first-class output.
3. **One verification harness, everywhere**: the same `verify_model_kernel` call gates the training
   dataset, the eval numbers, *and* what the live demo shows. Nothing in the demo is shown as verified
   unless the harness actually ran; when it didn't, the UI says why.

On top of the harness sits a **training-free kernel search** (`kernelforge/search.py`). It samples up
to 4 candidates in one batched generation and verifies each one, sharing a single harness run between
identical kernels. The fastest correct kernel wins. If none passes, the exact harness feedback (for
example "6599/20480 elements outside tolerance; max abs error 0.00195 at index (1, 1, 1, 33)", or the
traceback) goes back to the model for up to 2 repair rounds. This is the multi-turn refinement idea
behind Kevin-32B, run at inference time instead of trained with RL.

## Current status

The **demo studio and kernel search are built and run on the base model**; launch them on a
Kaggle/Colab GPU for a shareable link. All 15 dataset templates were fixed and **pass Triton's CPU
interpreter** (correctness only, run locally in Docker). GPU verification of the 300 candidates is the
next step. Fine-tuning and evaluation code is written but **has not been run yet**, so there are no
result numbers to report.

| Piece | State |
|---|---|
| Verification harness | Done, GPU-validated; GPU mode (correctness + speedup) and interpreter mode (CPU, correctness only) |
| Demo studio + kernel search (best-of-N, repair, result cache, `/stats`) | Built; 75 tests (74 on macOS, plus 1 interpreter test that runs in Docker); first live GPU session + GIF pending |
| Public deployment (Railway: Dockerfile + `railway.json`, hosted-model option, code policy, rate limits) | Built; interpreter-only (Railway has no GPU) |
| Dataset (15 templates × 20 shapes, 5 categories) | Templates fixed; 15/15 pass the interpreter; GPU verification pending |
| Fine-tuning (QLoRA, `training/finetune.py`) | Code written, not yet run |
| Evaluation (`evaluation/`) | Code written, waiting on dataset + adapter |

See [DOCUMENTATION.md](DOCUMENTATION.md) for how it works, [docs/project_writeup.md](docs/project_writeup.md)
for positioning and prior work, and [MASTER_PLAN.md](MASTER_PLAN.md) for the roadmap, study track and next steps.

## Run the demo

Triton needs Linux + an NVIDIA GPU, so the demo runs on Kaggle (GPU T4, **Internet on**) or Colab (T4 runtime).
In a notebook cell:

```bash
!git clone https://github.com/Arsalan-05/KernelForge.git
%cd KernelForge
!pip install -q -r requirements-demo.txt
!python app/launch.py --share
```

The first run downloads the model (about 3 GB) and loads it before the server reports ready. It then
prints `Public: https://….trycloudflare.com`, a temporary Cloudflare quick-tunnel link that works until
the cell stops (no account needed; `cloudflared` is downloaded automatically). Open it, pick an op from
the library, and hit **Generate** (or ⌘/Ctrl+Enter): the kernel streams in token by token, then goes
through parse and harness verification, with a live pipeline view and a correctness/speedup card.
Set **Samples** (1–4) and **Repair** (off/1/2) to run a kernel search instead: every candidate appears
in a timeline with its verdict, and you can click any one to inspect its kernel, raw output, and harness
result. The **Harness** switch picks the GPU (correctness + speedup) or Triton's interpreter (CPU,
correctness only, on a small shape of the same op).

Useful flags:

- `--adapter training/checkpoints/<run>/adapter` runs a fine-tuned adapter instead; the badge switches to "fine-tuned"
- `--max-new-tokens 1024` gives faster, shorter generations (default 2048)
- `--mock` runs with no model or GPU; it echoes the hand-written template kernels, for UI testing only (red banner)

Without `launch.py` (e.g. for development), the backend serves the UI itself at `/` and the API docs at `/docs`:

```bash
KERNELFORGE_MOCK_MODEL=1 uvicorn app.backend.main:app --port 8000   # open http://localhost:8000
```

## Deploy (Railway, or any container host)

The root `Dockerfile` and `railway.json` define an always-on web deployment. Railway builds the
Dockerfile, checks `/health`, and binds `$PORT`. That image has CPU torch and Triton's interpreter,
so kernels are **checked for correctness only; there are no GPU speedups on Railway**. Pushing to
`main` redeploys.

Service variables (Railway → service → Variables):

| Variable | Purpose |
|---|---|
| `KERNELFORGE_LLM_MODEL` | Model id for an OpenAI-compatible API (e.g. `gpt-4o-mini`, or a Qwen coder model on Groq/Together/OpenRouter). Unset means mock mode. |
| `KERNELFORGE_LLM_BASE_URL` | API base URL, default `https://api.openai.com/v1` |
| `KERNELFORGE_LLM_API_KEY` | API key (never passed to the kernel sandbox) |
| `KERNELFORGE_RATE_LIMIT` | Runs per visitor IP per minute (default 6) |

The image sets `KERNELFORGE_PUBLIC=1`, which turns on:
- a static code policy on references and generated kernels (torch/triton/math imports only; no file, process or dunder access)
- per-IP rate limiting (HTTP 429)
- a bounded queue (HTTP 503)

A hosted model is labelled as such in the UI (a blue badge); it isn't KernelForge's fine-tuned model.

```bash
docker build -t kernelforge-web . && docker run --rm -p 8000:8000 kernelforge-web   # same image locally
```

## Other commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/ -v                                      # harness, search engine + app tests (CPU, no Triton)
python data/build_dataset.py --device cuda            # generate + verify dataset (GPU)

# Correctness-check all 15 templates with Triton's CPU interpreter (no GPU; needs Docker)
docker build -f docker/verify.Dockerfile -t kernelforge-verify .
docker run --rm -v "$PWD":/work kernelforge-verify python data/build_dataset.py --interpret --smoke --jobs 8
python evaluation/baseline_eval.py --device cuda      # baseline eval (base model)
python training/finetune.py --config training/configs/finetune_default.json
python evaluation/finetuned_eval.py --adapter training/checkpoints/run_*/adapter --device cuda
python evaluation/baseline_eval.py --dry-run --device cpu   # eval pipeline dry-run, no GPU/model
```

## Repo layout

```
app/               FastAPI backend (streaming API, template catalog), web studio UI (app/web), launcher
verification/      correctness + benchmark harness (subprocess sandbox, GPU or Triton interpreter)
kernelforge/       shared prompts, parsing, dataset, model loading, kernel search engine
docker/            CPU image with Triton's interpreter for local correctness checks
data/              schema, templates, build_dataset.py
training/          QLoRA fine-tuning + configs
evaluation/        baseline/fine-tuned eval + explanation rubric
docs/              project write-up
examples/kernels/  hand-written Triton kernels (vector add, ReLU, fused softmax)
tests/             harness, library, and API tests
```

## Limitations

- The demo currently runs the **base model**; the fine-tuned adapter is in progress on the same pipeline.
- Small model (1.5B) and a ~300-example target vs 8B–32B RL-trained systems.
- Narrow category coverage vs KernelBench/TritonBench.
- No RL; explanation quality is scored with a manual rubric.
- Triton requires Linux + NVIDIA; local macOS development covers scripting, the harness, and
  interpreter correctness checks in Docker. An interpreter pass says nothing about speed and isn't
  reported as GPU-verified.
- The demo's sandbox is subprocess isolation + timeout inside a temporary Kaggle/Colab session. That
  protects the server process from hangs and crashes; it is **not** a hardened sandbox for untrusted
  code on an always-on public server.
