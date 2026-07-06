# Example kernels (Phase 0 deliverable)

Hand-written Triton kernels, one per op, each paired with a plain PyTorch
`reference.py` and a `notes.md` explaining *why* (or why not) the kernel is
faster. These exist to prove out the harness end-to-end and to seed the
first few dataset examples in Phase 1.

| Kernel | Pattern | Speedup source |
|---|---|---|
| [vector_add](vector_add/) | elementwise | none (bandwidth-bound baseline) |
| [relu](relu/) | elementwise | none (bandwidth-bound baseline) |
| [softmax](softmax/) | reduction + elementwise, fused | avoids ~5 eager kernel launches / memory round-trips |

## Requirements

Triton only ships wheels for Linux + NVIDIA GPUs — these will **not** run
on macOS or Windows, with or without a GPU. Run them on Kaggle or Colab:

```bash
pip install -r requirements.txt   # picks up triton on Linux automatically
python verification/verify_kernel.py \
    examples/kernels/softmax/reference.py \
    examples/kernels/softmax/candidate_triton.py \
    --shape 4096 4096 --device cuda
```

Locally (macOS/Windows), use the harness's CPU test suite
(`tests/test_verify_kernel.py`) to validate the harness mechanics against
plain-Python stand-ins instead — see [DOCUMENTATION.md](../../DOCUMENTATION.md).
