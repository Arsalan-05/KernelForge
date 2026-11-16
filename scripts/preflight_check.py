"""Pre-flight check for a KernelForge Kaggle/Colab GPU session.

Runs all checks that do not require a GPU and optionally GPU-dependent ones when
a GPU is available. Exits 0 if all required checks pass, 1 if any fail.

Usage:
    python scripts/preflight_check.py           # all checks
    python scripts/preflight_check.py --no-gpu  # skip GPU-dependent checks

What it checks:
  1. Python version (>= 3.9)
  2. Required Python packages (torch, triton, unsloth if present)
  3. GPU availability and memory (>= MIN_GPU_GB GB recommended)
  4. Dataset status: verified dataset exists and is non-empty; or unverified candidates exist
  5. Config files present
  6. Test suite passes (optional, slow)
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_GPU_GB = 14.0  # Recommend at least 14 GB for norm shapes at r65536

REQUIRED_PACKAGES = ["torch", "datasets", "peft"]
OPTIONAL_PACKAGES = {"triton": "needed for --device cuda", "unsloth": "faster training (optional)"}
REQUIRED_CONFIGS = [
    "training/configs/finetune_default.json",
    "training/configs/qwen2.5-coder-1.5b.json",
]


def check(description: str, ok: bool, detail: str = "", warning: bool = False) -> bool:
    """Print a check result. Returns True if the check passed."""
    icon = "✓" if ok else ("⚠" if warning else "✗")
    status = "ok" if ok else ("warn" if warning else "FAIL")
    line = f"  [{status:4s}] {description}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU-dependent checks")
    parser.add_argument("--run-tests", action="store_true", help="Run the test suite (slow)")
    args = parser.parse_args()

    print(f"KernelForge pre-flight check (ROOT: {ROOT})\n")
    failures = 0

    # 1. Python version
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= 9
    if not check(f"Python {major}.{minor}", ok, f"need >= 3.9"):
        failures += 1

    # 2. Required packages
    print("\nPackages:")
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            check(pkg, True, f"version {version}")
        except ImportError:
            check(pkg, False, "not installed")
            failures += 1

    for pkg, note in OPTIONAL_PACKAGES.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            check(f"{pkg} (optional)", True, f"version {version}")
        except ImportError:
            check(f"{pkg} (optional)", True, f"not installed — {note}", warning=True)

    # 3. GPU checks
    if not args.no_gpu:
        print("\nGPU:")
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
            if not check("CUDA available", cuda_ok):
                failures += 1
            elif cuda_ok:
                name = torch.cuda.get_device_name(0)
                total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
                mem_ok = total_gb >= MIN_GPU_GB
                check(f"GPU: {name}", True, f"{total_gb:.1f} GB")
                if not check(f"GPU memory >= {MIN_GPU_GB:.0f} GB", mem_ok,
                             f"got {total_gb:.1f} GB — 15 OOM shapes may still OOM on T4"):
                    # This is a warning, not a failure (T4 is 16 GB which is >= 14)
                    check(f"GPU memory >= {MIN_GPU_GB:.0f} GB", mem_ok, warning=True)
                # Free memory
                free_gb = (torch.cuda.get_device_properties(0).total_memory
                           - torch.cuda.memory_reserved(0)) / 2**30
                check("GPU free memory", free_gb > 1.0, f"{free_gb:.1f} GB free")
        except ImportError:
            check("CUDA (torch not installed)", False)
            failures += 1

    # 4. Dataset status
    print("\nDataset:")
    verified = ROOT / "data" / "verified" / "dataset.jsonl"
    templates_dir = ROOT / "data" / "templates"
    if verified.exists() and verified.stat().st_size > 0:
        try:
            count = sum(1 for _ in verified.open())
            check("GPU-verified dataset", True, f"{count} entries in {verified}")
        except Exception as e:
            check("GPU-verified dataset", False, f"error reading: {e}")
            failures += 1
    else:
        check("GPU-verified dataset", False,
              "data/verified/dataset.jsonl missing or empty — run build_dataset.py --device cuda",
              warning=True)
        # Fallback: unverified candidates
        try:
            sys.path.insert(0, str(ROOT))
            from kernelforge.dataset import load_template_candidates
            candidates = load_template_candidates()
            check("Unverified template candidates", bool(candidates),
                  f"{len(candidates)} entries (dry-run only, not for real eval)")
        except Exception as e:
            check("Unverified template candidates", False, str(e))
            failures += 1

    # 5. Config files
    print("\nConfig files:")
    for rel in REQUIRED_CONFIGS:
        path = ROOT / rel
        if check(rel, path.exists()):
            try:
                json.loads(path.read_text())
            except json.JSONDecodeError as e:
                check(f"{rel} valid JSON", False, str(e))
                failures += 1

    # 6. OOM shape reminder
    print("\nKnown OOM shapes (from interpret_report_2026-11-15.json):")
    report_path = ROOT / "data" / "raw" / "interpret_report_2026-11-15.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            oom_count = sum(
                1 for shapes in report.get("templates", {}).values()
                if isinstance(shapes, list)
                for s in shapes
                if s.get("status") == "crash"
            )
            msg = (f"{oom_count} shapes OOM'd in 8 GB VM — may still OOM on T4 (16 GB). "
                   "They are kept in the dataset; `oom` is counted separately from incorrect.")
            check("OOM shape reminder", True, msg, warning=True)
        except Exception:
            pass

    # 7. Optional: test suite
    if args.run_tests:
        print("\nTest suite:")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        passed = result.returncode == 0
        if not check("pytest tests/", passed, result.stdout.strip().splitlines()[-1] if result.stdout else ""):
            failures += 1
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    # Summary
    print(f"\n{'=' * 50}")
    if failures == 0:
        print("All required checks passed.")
    else:
        print(f"{failures} required check(s) failed. Fix them before running the GPU session.")
        sys.exit(1)


if __name__ == "__main__":
    main()
