# Correctness-only Triton verification on machines without an NVIDIA GPU
# (e.g. Apple Silicon via Docker Desktop). Kernels run through Triton's CPU
# interpreter; no timings are produced. GPU speedups still need Kaggle/Colab.
#
#   docker build -f docker/verify.Dockerfile -t kernelforge-verify .
#   docker run --rm -v "$PWD":/work kernelforge-verify \
#       python data/build_dataset.py --interpret --smoke
FROM python:3.11-slim

RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir triton==3.5.0 "numpy<2.3" pytest pyyaml
# numpy 2.3+ refuses int() on 1-element arrays, which Triton 3.5's interpreter
# does for every runtime loop bound.

ENV TRITON_INTERPRET=1 \
    PYTHONUNBUFFERED=1
WORKDIR /work
