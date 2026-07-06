"""
Runs inside its own subprocess, spawned by verify_kernel.py.

Takes one argument: the path to a JSON payload file describing the reference
implementation, the candidate (generated) implementation, and how to build
inputs for both. Always emits exactly one JSON object on stdout, even on
failure — the parent process treats any other outcome (non-JSON stdout,
non-zero exit without JSON, timeout) as a crash.

Isolating this in its own process means a kernel that segfaults, deadlocks,
or wedges the GPU only takes down this subprocess, not the harness or any
long-lived worker that's driving dataset generation.
"""

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

try:
    import resource

    def _limit_memory(max_bytes: int) -> None:
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

except ImportError:  # resource is POSIX-only; skip the limit on other platforms
    def _limit_memory(max_bytes: int) -> None:
        pass


def _emit(result: dict) -> None:
    print(json.dumps(result))
    sys.stdout.flush()


def _load_entry_point(source_path: Path, module_name: str, entry_point: str):
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, entry_point):
        raise AttributeError(
            f"{source_path.name} does not define a callable named '{entry_point}'"
        )
    return getattr(module, entry_point)


def _build_inputs(input_spec: list, device: str, seed: int):
    import torch

    torch.manual_seed(seed)
    tensors = []
    for spec in input_spec:
        dtype = getattr(torch, spec["dtype"])
        low, high = spec.get("value_range", (-1.0, 1.0))
        t = torch.empty(spec["shape"], dtype=dtype, device=device)
        t.uniform_(low, high)
        tensors.append(t)
    return tensors


def _as_tuple(value):
    return value if isinstance(value, (tuple, list)) else (value,)


def _check_correct(candidate_out, reference_out, atol: float, rtol: float) -> bool:
    import torch

    cand = _as_tuple(candidate_out)
    ref = _as_tuple(reference_out)
    if len(cand) != len(ref):
        return False
    for c, r in zip(cand, ref):
        if c.shape != r.shape:
            return False
        if not torch.allclose(c.float(), r.float(), atol=atol, rtol=rtol, equal_nan=False):
            return False
    return True


def _time_fn(fn, inputs, device: str, warmup: int, iters: int):
    import torch

    for _ in range(warmup):
        fn(*inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*inputs)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / 1000.0)  # ms -> s
        else:
            t0 = time.perf_counter()
            out = fn(*inputs)
            samples.append(time.perf_counter() - t0)
    samples.sort()
    mid = len(samples) // 2
    median = samples[mid] if len(samples) % 2 else (samples[mid - 1] + samples[mid]) / 2
    return out, median


def run(payload: dict) -> dict:
    import torch

    device = payload["device"]
    if device == "cuda" and not torch.cuda.is_available():
        return {"status": "error", "error": "device 'cuda' requested but no CUDA device is available"}

    entry_point = payload["entry_point"]
    atol = payload["atol"]
    rtol = payload["rtol"]
    warmup = payload["warmup"]
    iters = payload["iters"]
    seed = payload["seed"]

    reference_fn = _load_entry_point(Path(payload["reference_path"]), "reference_module", entry_point)
    candidate_fn = _load_entry_point(Path(payload["candidate_path"]), "candidate_module", entry_point)

    ref_inputs = _build_inputs(payload["input_spec"], device, seed)
    cand_inputs = _build_inputs(payload["input_spec"], device, seed)  # identical seed -> identical values

    ref_out, ref_time = _time_fn(reference_fn, ref_inputs, device, warmup, iters)
    cand_out, cand_time = _time_fn(candidate_fn, cand_inputs, device, warmup, iters)

    correct = _check_correct(cand_out, ref_out, atol, rtol)
    speedup = (ref_time / cand_time) if cand_time > 0 else float("inf")

    return {
        "status": "ok",
        "correct": correct,
        "reference_time_s": ref_time,
        "candidate_time_s": cand_time,
        "speedup": speedup,
    }


def main() -> None:
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text())

    mem_limit = payload.get("memory_limit_bytes")
    if mem_limit:
        try:
            _limit_memory(mem_limit)
        except Exception:
            pass  # best-effort; not all platforms/kernels allow this

    try:
        result = run(payload)
    except Exception:
        result = {"status": "error", "error": traceback.format_exc()}

    _emit(result)


if __name__ == "__main__":
    main()
