"""One GPU session (Kaggle/Colab T4) that produces every artifact the project is waiting on.

Stages, in order; each is logged and a failure doesn't stop the later ones
that don't depend on it:

    env       GPU / torch / Triton versions
    harness   tests/test_verify_kernel.py on the real GPU
    dataset   data/build_dataset.py --device cuda  -> data/verified/dataset.jsonl
    baseline  evaluation/baseline_eval.py on held-out templates (base model)
    finetune  training/finetune.py (QLoRA)                    [opt-in: --finetune]
    eval      evaluation/finetuned_eval.py on the same split  [runs after finetune]

Everything ends up in one folder (and a .zip next to it) to download:

    python scripts/gpu_session.py                      # dataset + baseline (~1.5 h on a T4)
    python scripts/gpu_session.py --finetune           # + QLoRA + fine-tuned eval (~+1 h)
    python scripts/gpu_session.py --stages dataset     # just one stage
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_STAGES = ["env", "harness", "dataset", "baseline", "finetune", "eval"]


def _default_out() -> Path:
    kaggle = Path("/kaggle/working")
    return (kaggle if kaggle.is_dir() else ROOT) / "kernelforge_outputs"


def _run(name: str, cmd: list[str], log_dir: Path, timeout_s: float) -> dict:
    """Run a stage, streaming output to the console and a log file."""
    log_path = log_dir / f"{name}.log"
    print(f"\n{'=' * 70}\n[{name}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait(timeout=max(1.0, timeout_s - (time.time() - start)))
        except subprocess.TimeoutExpired:
            proc.kill()
            log.write(f"\n[stage killed after {timeout_s}s]\n")
    status = "ok" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
    elapsed = round(time.time() - start, 1)
    print(f"[{name}] {status} in {elapsed}s", flush=True)
    return {"status": status, "seconds": elapsed, "log": log_path.name}


def _env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def _latest(pattern: str) -> Path | None:
    found = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def _collect(out: Path) -> list[str]:
    copied = []
    for rel in ["data/verified/dataset.jsonl", "data/raw/generation_report.json", "data/raw/rejected.jsonl"]:
        src = ROOT / rel
        if src.exists():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    for report in (ROOT / "evaluation" / "results").glob("*.json"):
        dst = out / "evaluation" / "results" / report.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report, dst)
        copied.append(str(dst.relative_to(out)))
    run = _latest("training/checkpoints/run_*")
    if run is not None:
        for name in ("adapter", "split_info.json", "held_out_test.jsonl"):
            src = run / name
            dst = out / "training" / run.name / name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            else:
                continue
            copied.append(str(dst.relative_to(out)))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=None)
    parser.add_argument("--finetune", action="store_true", help="also run QLoRA fine-tuning and its evaluation")
    parser.add_argument("--baseline-limit", type=int, default=25,
                        help="held-out examples for each eval (round-robin across categories)")
    parser.add_argument("--max-new-tokens", type=int, default=1536, help="generation cap for evals")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stages = args.stages or [s for s in ALL_STAGES if args.finetune or s not in ("finetune", "eval")]
    out = args.out or _default_out()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    session = {"started_at": datetime.now(timezone.utc).isoformat(), "stages": {}}
    session["env"] = _env_info()
    print(json.dumps(session["env"], indent=2))
    if not session["env"].get("cuda_available") and any(s != "env" for s in stages):
        print("\nNo CUDA GPU visible. On Kaggle: Settings -> Accelerator -> GPU T4, then restart.", file=sys.stderr)
        sys.exit(2)

    model_config = ROOT / "training" / "configs" / "qwen2.5-coder-1.5b.json"
    eval_config = out / "eval_model_config.json"
    cfg = json.loads(model_config.read_text())
    cfg["max_new_tokens"] = args.max_new_tokens
    eval_config.write_text(json.dumps(cfg, indent=2))

    plan = {
        "harness": ([py, "-m", "pytest", "-q", "tests/test_verify_kernel.py"], 900),
        "dataset": ([py, "data/build_dataset.py", "--device", "cuda"], 3 * 3600),
        "baseline": ([py, "evaluation/baseline_eval.py", "--device", "cuda", "--config", str(eval_config),
                      "--limit", str(args.baseline_limit)], 3 * 3600),
        "finetune": ([py, "training/finetune.py", "--config", "training/configs/finetune_default.json"], 4 * 3600),
    }
    for stage in stages:
        if stage == "env":
            continue
        if stage in ("baseline", "finetune") and not (ROOT / "data/verified/dataset.jsonl").exists():
            session["stages"][stage] = {"status": "skipped (no verified dataset)"}
            continue
        if stage == "eval":
            run = _latest("training/checkpoints/run_*")
            if run is None or not (run / "adapter").exists():
                session["stages"][stage] = {"status": "skipped (no adapter)"}
                continue
            cmd = [py, "evaluation/finetuned_eval.py", "--device", "cuda", "--config", str(eval_config),
                   "--adapter", str(run / "adapter"), "--checkpoint-run", str(run),
                   "--limit", str(args.baseline_limit)]
            session["stages"][stage] = _run(stage, cmd, logs, 3 * 3600)
            continue
        cmd, timeout_s = plan[stage]
        session["stages"][stage] = _run(stage, cmd, logs, timeout_s)

    report = ROOT / "data/raw/generation_report.json"
    if report.exists():
        session["dataset_report"] = json.loads(report.read_text())
    session["finished_at"] = datetime.now(timezone.utc).isoformat()
    session["artifacts"] = _collect(out)
    (out / "session.json").write_text(json.dumps(session, indent=2))

    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"\n{'=' * 70}\nSession summary")
    for stage, result in session["stages"].items():
        print(f"  {stage:<10} {result['status']}" + (f"  ({result['seconds']}s)" if "seconds" in result else ""))
    print(f"\nArtifacts: {out}\nDownload:  {archive}")


if __name__ == "__main__":
    main()
