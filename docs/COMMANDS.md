# KernelForge commands

Moved from the top-level README. GPU and Kaggle commands are in [KAGGLE_RUN.md](KAGGLE_RUN.md).

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

## Local development without a GPU

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
