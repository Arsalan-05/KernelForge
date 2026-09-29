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
from typing import Optional

try:
    import resource

    def _limit_memory(max_bytes: int) -> None:
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

    def _limit_process(cpu_s: Optional[int], file_bytes: Optional[int]) -> None:
        for limit, value in (
            (resource.RLIMIT_CORE, 0),
            (resource.RLIMIT_CPU, cpu_s),
            (resource.RLIMIT_FSIZE, file_bytes),
        ):
            if value is None:
                continue
            try:
                _, hard = resource.getrlimit(limit)
                if hard != resource.RLIM_INFINITY:
                    value = min(value, hard)
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass

except ImportError:  # resource is POSIX-only; skip the limit on other platforms
    def _limit_memory(max_bytes: int) -> None:
        pass

    def _limit_process(cpu_s: Optional[int], file_bytes: Optional[int]) -> None:
        pass


def _emit(result: dict) -> None:
    print(json.dumps(result))
    sys.stdout.flush()


def _load_module(source_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_entry_point(source_path: Path, module_name: str, entry_point: str):
    module = _load_module(source_path, module_name)
    if not hasattr(module, entry_point):
        raise AttributeError(
            f"{source_path.name} does not define a callable named '{entry_point}'"
        )
    return getattr(module, entry_point)


def _require_attr(module, name: str, filename: str):
    if not hasattr(module, name):
        raise AttributeError(f"{filename} does not define required attribute '{name}'")
    return getattr(module, name)


def _to_device(tensors, device: str):
    return [t.to(device) if hasattr(t, "to") else t for t in tensors]


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


def _compare(candidate_out, reference_out, atol: float, rtol: float):
    """Return (correct, mismatch description or None)."""
    import torch

    cand = _as_tuple(candidate_out)
    ref = _as_tuple(reference_out)
    if len(cand) != len(ref):
        return False, f"candidate returned {len(cand)} output(s), reference returned {len(ref)}"
    for i, (c, r) in enumerate(zip(cand, ref)):
        label = f"output {i}" if len(ref) > 1 else "output"
        if not hasattr(c, "shape"):
            return False, f"{label} is {type(c).__name__}, expected a tensor"
        if c.shape != r.shape:
            return False, f"{label} shape {tuple(c.shape)} != reference shape {tuple(r.shape)}"
        cf, rf = c.detach().float().cpu(), r.detach().float().cpu()
        if not torch.allclose(cf, rf, atol=atol, rtol=rtol, equal_nan=False):
            if torch.isnan(cf).any() and not torch.isnan(rf).any():
                return False, f"{label} contains NaN ({int(torch.isnan(cf).sum())} elements)"
            diff = (cf - rf).abs().nan_to_num(nan=float("inf"))
            bad = diff > atol + rtol * rf.abs()
            flat = int(diff.argmax())
            idx = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat), diff.shape)) if diff.ndim else ()
            return False, (
                f"{label} mismatch: {int(bad.sum())}/{bad.numel()} elements outside atol={atol}, rtol={rtol}; "
                f"max abs error {float(diff.max()):.4g} at index {idx} "
                f"(candidate {float(cf.flatten()[flat]):.4g}, reference {float(rf.flatten()[flat]):.4g})"
            )
    return True, None


def _check_correct(candidate_out, reference_out, atol: float, rtol: float) -> bool:
    return _compare(candidate_out, reference_out, atol, rtol)[0]


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


def run_function_contract(payload: dict, device: str) -> dict:
    """Reference/candidate are each a bare `solution(*tensors) -> tensor` function."""
    entry_point = payload["entry_point"]
    warmup = payload["warmup"]
    iters = payload["iters"]
    seed = payload["seed"]

    reference_fn = _load_entry_point(Path(payload["reference_path"]), "reference_module", entry_point)
    candidate_fn = _load_entry_point(Path(payload["candidate_path"]), "candidate_module", entry_point)

    ref_inputs = _build_inputs(payload["input_spec"], device, seed)
    cand_inputs = _build_inputs(payload["input_spec"], device, seed)  # identical seed -> identical values

    ref_out, ref_time = _time_fn(reference_fn, ref_inputs, device, warmup, iters)
    cand_out, cand_time = _time_fn(candidate_fn, cand_inputs, device, warmup, iters)
    return _result(cand_out, ref_out, ref_time, cand_time, payload)


def _result(cand_out, ref_out, ref_time, cand_time, payload: dict) -> dict:
    correct, mismatch = _compare(cand_out, ref_out, payload["atol"], payload["rtol"])
    if payload.get("interpret"):
        return {"status": "ok", "correct": correct, "error": mismatch, "interpreted": True}
    speedup = (ref_time / cand_time) if cand_time > 0 else float("inf")
    return {
        "status": "ok",
        "correct": correct,
        "reference_time_s": ref_time,
        "candidate_time_s": cand_time,
        "speedup": speedup,
        "error": mismatch,
    }


def run_model_contract(payload: dict, device: str) -> dict:
    """
    Reference is a KernelBench-style module: `Model(nn.Module)` +
    `get_inputs()` + `get_init_inputs()`. Candidate is a `ModelNew(nn.Module)`
    with the same __init__/forward signature as `Model`.

    `Model` and `ModelNew` are constructed independently, so any op with
    random internal state (e.g. quantization weights) needs the RNG seeded
    identically immediately before *each* construction -- not just before
    get_inputs() -- or the two will silently compare against different
    random weights instead of the same op.
    """
    import torch

    warmup = payload["warmup"]
    iters = payload["iters"]
    seed = payload["seed"]

    reference_module = _load_module(Path(payload["reference_path"]), "reference_module")
    candidate_module = _load_module(Path(payload["candidate_path"]), "candidate_module")

    ref_filename = Path(payload["reference_path"]).name
    cand_filename = Path(payload["candidate_path"]).name
    Model = _require_attr(reference_module, "Model", ref_filename)
    get_inputs = _require_attr(reference_module, "get_inputs", ref_filename)
    get_init_inputs = _require_attr(reference_module, "get_init_inputs", ref_filename)
    ModelNew = _require_attr(candidate_module, "ModelNew", cand_filename)

    torch.manual_seed(seed)
    init_inputs = get_init_inputs()

    torch.manual_seed(seed)
    ref_model = Model(*init_inputs).to(device).eval()

    torch.manual_seed(seed)
    cand_model = ModelNew(*init_inputs).to(device).eval()

    torch.manual_seed(seed)
    base_inputs = _to_device(get_inputs(), device)
    ref_inputs = [t.clone() if hasattr(t, "clone") else t for t in base_inputs]
    cand_inputs = [t.clone() if hasattr(t, "clone") else t for t in base_inputs]

    if payload.get("interpret"):
        warmup, iters = 0, 1

    with torch.no_grad():
        ref_out, ref_time = _time_fn(ref_model.forward, ref_inputs, device, warmup, iters)
        cand_out, cand_time = _time_fn(cand_model.forward, cand_inputs, device, warmup, iters)
    return _result(cand_out, ref_out, ref_time, cand_time, payload)


def run(payload: dict) -> dict:
    import torch

    device = payload["device"]
    if device == "cuda" and not torch.cuda.is_available():
        return {"status": "error", "error": "device 'cuda' requested but no CUDA device is available"}

    contract = payload.get("contract", "function")
    if contract == "model":
        return run_model_contract(payload, device)
    return run_function_contract(payload, device)


def main() -> None:
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text())

    mem_limit = payload.get("memory_limit_bytes")
    if mem_limit:
        try:
            _limit_memory(mem_limit)
        except Exception:
            pass  # best-effort; not all platforms/kernels allow this
    _limit_process(payload.get("cpu_limit_s"), payload.get("file_size_limit_bytes"))

    try:
        result = run(payload)
    except Exception:
        result = {"status": "error", "error": traceback.format_exc()}

    _emit(result)


if __name__ == "__main__":
    main()
