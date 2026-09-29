# Public web deployment (Railway or any container host without a GPU).
# Serves the API + UI; kernels are checked with Triton's CPU interpreter
# (correctness only, no timings). Generation uses a hosted OpenAI-compatible
# model when KERNELFORGE_LLM_MODEL is set, otherwise mock mode.
#
#   docker build -t kernelforge-web .
#   docker run --rm -p 8000:8000 kernelforge-web
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    KERNELFORGE_PUBLIC=1 \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY requirements-server.txt .
RUN pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-server.txt

COPY . .

RUN useradd --create-home --uid 10001 kernelforge \
 && chown -R kernelforge /app
USER kernelforge

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 75"]
