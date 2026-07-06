"""
Verification harness: given a reference PyTorch implementation and a
candidate implementation (a hand-written or LLM-generated Triton kernel,
wrapped in a plain Python function), determines whether the candidate is
correct and how much faster it is than the reference.

This is the most-reused piece of infrastructure in the project — it's used
to (1) filter LLM-generated training examples down to ones that actually
work, and (2) evaluate the fine-tuned model against the base model, using
the exact same pass/fail and speedup definitions in both places.

Two contracts are supported:

- **function** (`verify_kernel`): reference and candidate each define a
  callable with the same name (default: "solution") taking one or more
  tensors and returning a tensor or tuple of tensors.
- **model** (`verify_model_kernel`): reference defines a KernelBench-style
  `Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`; candidate
  defines a `ModelNew(nn.Module)` with the same __init__/forward signature.
  This is the contract the dataset schema (see data/SCHEMA.md) stores
  examples in.

Execution happens in a subprocess (see sandbox_runner.py) so a candidate
that hangs, segfaults, or wedges the GPU can't take down the caller.
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_SANDBOX_RUNNER = Path(__file__).parent / "sandbox_runner.py"


@dataclass
class TensorSpec:
    shape: list
    dtype: str = "float32"
    value_range: tuple = (-1.0, 1.0)

    def to_dict(self) -> dict:
        return {"shape": self.shape, "dtype": self.dtype, "value_range": list(self.value_range)}


@dataclass
class VerifyResult:
    status: str  # "ok" | "error" | "timeout" | "crash"
    correct: Optional[bool] = None
    speedup: Optional[float] = None
    reference_time_s: Optional[float] = None
    candidate_time_s: Optional[float] = None
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == "ok" and bool(self.correct)


def _run_sandbox(payload: dict, timeout_s: float) -> VerifyResult:
    with tempfile.TemporaryDirectory(prefix="kernelforge_verify_") as tmpdir:
        payload_path = Path(tmpdir) / "payload.json"
        payload_path.write_text(json.dumps(payload))

        try:
            proc = subprocess.run(
                [sys.executable, str(_SANDBOX_RUNNER), str(payload_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return VerifyResult(status="timeout", error=f"candidate exceeded {timeout_s}s timeout")

        stdout = proc.stdout.strip()
        if not stdout:
            return VerifyResult(
                status="crash",
                error=f"no output from sandbox (exit code {proc.returncode}); stderr:\n{proc.stderr}",
            )

        try:
            # sandbox_runner emits exactly one JSON object as its last stdout line
            result = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError:
            return VerifyResult(
                status="crash",
                error=f"unparseable sandbox output (exit code {proc.returncode}):\n{stdout}\n{proc.stderr}",
            )

        return VerifyResult(**result)


def verify_kernel(
    reference_code: str,
    candidate_code: str,
    input_spec: list[TensorSpec],
    entry_point: str = "solution",
    device: str = "cuda",
    atol: float = 1e-2,
    rtol: float = 1e-2,
    warmup: int = 10,
    iters: int = 50,
    seed: int = 0,
    timeout_s: float = 30.0,
    memory_limit_bytes: Optional[int] = None,
) -> VerifyResult:
    """Run `candidate_code` against `reference_code` and report correctness + speedup.

    `reference_code` / `candidate_code` are the full source text of Python
    modules, each defining a top-level function named `entry_point`.
    """
    with tempfile.TemporaryDirectory(prefix="kernelforge_verify_src_") as tmpdir:
        tmp = Path(tmpdir)
        reference_path = tmp / "reference_module.py"
        candidate_path = tmp / "candidate_module.py"
        reference_path.write_text(reference_code)
        candidate_path.write_text(candidate_code)

        payload = {
            "contract": "function",
            "reference_path": str(reference_path),
            "candidate_path": str(candidate_path),
            "entry_point": entry_point,
            "device": device,
            "input_spec": [s.to_dict() if isinstance(s, TensorSpec) else s for s in input_spec],
            "atol": atol,
            "rtol": rtol,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
            "memory_limit_bytes": memory_limit_bytes,
        }
        return _run_sandbox(payload, timeout_s)


def verify_model_kernel(
    reference_code: str,
    candidate_code: str,
    device: str = "cuda",
    atol: float = 1e-2,
    rtol: float = 1e-2,
    warmup: int = 10,
    iters: int = 50,
    seed: int = 0,
    timeout_s: float = 30.0,
    memory_limit_bytes: Optional[int] = None,
) -> VerifyResult:
    """Run a KernelBench-style candidate against its reference and report correctness + speedup.

    `reference_code` is the full source of a module defining `Model`,
    `get_inputs()`, and `get_init_inputs()`. `candidate_code` is the full
    source of a module defining `ModelNew` with the same __init__/forward
    signature as `Model`. This is the contract dataset entries are stored
    in (see data/SCHEMA.md) — `Model` supplies both the reference forward
    pass and the shared get_inputs/get_init_inputs used to build `ModelNew`.
    """
    with tempfile.TemporaryDirectory(prefix="kernelforge_verify_src_") as tmpdir:
        tmp = Path(tmpdir)
        reference_path = tmp / "reference_module.py"
        candidate_path = tmp / "candidate_module.py"
        reference_path.write_text(reference_code)
        candidate_path.write_text(candidate_code)

        payload = {
            "contract": "model",
            "reference_path": str(reference_path),
            "candidate_path": str(candidate_path),
            "device": device,
            "atol": atol,
            "rtol": rtol,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
            "memory_limit_bytes": memory_limit_bytes,
        }
        return _run_sandbox(payload, timeout_s)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify a candidate kernel against a reference implementation.")
    parser.add_argument("reference", type=Path, help="Path to reference module (defines `solution`)")
    parser.add_argument("candidate", type=Path, help="Path to candidate module (defines `solution`)")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096], help="Shape of each input tensor")
    parser.add_argument("--num-inputs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--entry-point", default="solution")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    input_spec = [TensorSpec(shape=args.shape) for _ in range(args.num_inputs)]

    result = verify_kernel(
        reference_code=args.reference.read_text(),
        candidate_code=args.candidate.read_text(),
        input_spec=input_spec,
        entry_point=args.entry_point,
        device=args.device,
        atol=args.atol,
        rtol=args.rtol,
        warmup=args.warmup,
        iters=args.iters,
        timeout_s=args.timeout,
    )

    print(json.dumps(result.__dict__, indent=2))
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    _cli()
