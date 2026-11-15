"""One GPU session (Kaggle/Colab T4) that produces every artifact the project is waiting on.

Stages, in order:

    env       GPU / torch / Triton versions (fails if no CUDA GPU is visible)
    harness   tests/test_verify_kernel.py on the real GPU
    dataset   data/build_dataset.py --device cuda  -> data/verified/dataset.jsonl
    baseline  evaluation/baseline_eval.py on held-out templates (base model)
    finetune  training/finetune.py (QLoRA)                    [opt-in: --finetune]
    eval      evaluation/finetuned_eval.py on the same split  [runs after finetune]

Built so one "Run All" always leaves something usable to download:
  * A stage that raises, exits non-zero or times out is marked failed; the
    stages after it still run unless they need its output (no GPU -> GPU stages
    skipped; no verified dataset -> baseline/finetune skipped; no adapter -> eval
    skipped).
  * After every stage (and every few minutes during a long one) the artifacts
    collected so far, stage_status.json and session.json are zipped to
    <out>.zip, i.e. /kaggle/working/kernelforge_outputs.zip on Kaggle. The zip
    is replaced atomically, so a session killed mid-zip keeps the previous one.
  * --budget-hours: a stage that can't fit in the remaining time is skipped
    cleanly, and each stage's timeout is capped at what's left, so the final
    zip is always written before the platform kills the session.
  * --resume / --resume-from: the dataset stage continues from the progress log
    of an earlier run instead of starting over.

    python scripts/gpu_session.py                      # dataset + baseline
    python scripts/gpu_session.py --finetune           # + QLoRA + fine-tuned eval
    python scripts/gpu_session.py --stages dataset     # just one stage
    python scripts/gpu_session.py --budget-hours 11    # stop cleanly inside a 12 h limit
    python scripts/gpu_session.py --resume-from /kaggle/input/prev/kernelforge_outputs.zip
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
ALL_STAGES = ["env", "harness", "dataset", "baseline", "finetune", "eval"]
DATASET = ROOT / "data/verified/dataset.jsonl"
CUDA_PROGRESS = "data/raw/progress_cuda.jsonl"  # written by build_dataset.py --device cuda

# Seconds kept back from the budget for the final collect + zip.
FINAL_RESERVE_S = 300
# How often a long stage refreshes the zip while it is still running.
SNAPSHOT_EVERY_S = 600


class StageFailed(Exception):
    """A stage ran but did not succeed (non-zero exit, timeout, missing GPU, ...)."""


@dataclass
class Stage:
    name: str
    run: Callable[[float], dict]  # gets its timeout in seconds; returns extra info; raises on failure
    max_seconds: float            # hard cap for this stage
    min_seconds: float = 0.0      # skip unless at least this much budget is left (rough guess, not measured)
    skip_reason: Callable[[], Optional[str]] = lambda: None  # checked right before running


# --------------------------------------------------------------------------- #
# Stage runner (no GPU or subprocess knowledge, so it's unit-testable on CPU)
# --------------------------------------------------------------------------- #

def run_stages(
    stages: list[Stage],
    after_stage: Callable[[list[dict]], None],
    budget_s: Optional[float] = None,
    reserve_s: float = FINAL_RESERVE_S,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict]:
    """Run stages in order. One failing never stops the rest or the artifact saving.

    Returns one status record per stage:
        {"stage", "status": "ok"|"failed"|"skipped", "seconds", "error"}
    and calls after_stage(records_so_far) after each one (that's where the zip is
    written). A Ctrl-C / notebook interrupt marks the current stage failed, saves,
    and skips the rest.
    """
    start = clock()
    records: list[dict] = []
    interrupted = False

    for stage in stages:
        record = {"stage": stage.name, "status": "skipped", "seconds": 0.0, "error": None}
        reason = "session interrupted" if interrupted else stage.skip_reason()
        timeout = stage.max_seconds
        if reason is None and budget_s is not None:
            remaining = budget_s - (clock() - start) - reserve_s
            if remaining < max(stage.min_seconds, 1.0):
                reason = f"time budget: {remaining:.0f}s left, stage needs at least {stage.min_seconds:.0f}s"
            timeout = min(timeout, remaining)

        if reason is not None:
            record["error"] = reason
            print(f"\n[{stage.name}] skipped: {reason}", flush=True)
        else:
            t0 = clock()
            try:
                record.update(stage.run(timeout) or {})
                record["status"] = "ok"
            except KeyboardInterrupt:
                record.update(status="failed", error="interrupted")
                interrupted = True
            except Exception as exc:  # noqa: BLE001 - any stage error must not stop the session
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                if not isinstance(exc, StageFailed):
                    traceback.print_exc()
            record["seconds"] = round(clock() - t0, 1)
            print(f"[{stage.name}] {record['status']} in {record['seconds']}s"
                  + (f": {record['error']}" if record["error"] else ""), flush=True)

        records.append(record)
        try:
            after_stage(records)
        except Exception:  # noqa: BLE001 - a failed save must not kill the stages after it
            traceback.print_exc()
    return records


# --------------------------------------------------------------------------- #
# Subprocess stages
# --------------------------------------------------------------------------- #

def run_command(
    name: str,
    cmd: list[str],
    log_dir: Path,
    timeout_s: float,
    on_tick: Optional[Callable[[], None]] = None,
    tick_every_s: float = SNAPSHOT_EVERY_S,
    cwd: Path = ROOT,
) -> dict:
    """Run one stage command, streaming output to the console and <log_dir>/<name>.log.

    A watchdog timer kills the process at timeout_s (reading stdout blocks until
    the process exits, so a timeout on wait() alone would never fire). on_tick is
    called at most every tick_every_s while output is flowing, which is how a long
    stage refreshes the zip. Raises StageFailed on a non-zero exit or a timeout.
    """
    log_path = log_dir / f"{name}.log"
    print(f"\n{'=' * 70}\n[{name}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    timed_out = threading.Event()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)

        def kill():
            timed_out.set()
            proc.kill()

        watchdog = threading.Timer(max(1.0, timeout_s), kill)
        watchdog.start()
        last_tick = time.monotonic()
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                if on_tick is not None and time.monotonic() - last_tick >= tick_every_s:
                    log.flush()
                    on_tick()
                    last_tick = time.monotonic()
            proc.wait()
        except BaseException:  # Ctrl-C: don't leave the child running on the GPU
            proc.kill()
            proc.wait()
            raise
        finally:
            watchdog.cancel()
        if timed_out.is_set():
            log.write(f"\n[stage killed after {timeout_s:.0f}s]\n")
    if timed_out.is_set():
        raise StageFailed(f"timed out after {timeout_s:.0f}s (log: {log_path.name})")
    if proc.returncode != 0:
        raise StageFailed(f"exit code {proc.returncode} (log: {log_path.name})")
    return {"log": log_path.name}


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
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


def _latest(pattern: str) -> Optional[Path]:
    found = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def _has_lines(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #

def collect(out: Path, root: Path = ROOT) -> list[str]:
    """Copy whatever artifacts exist right now into `out` (missing ones are skipped)."""
    copied = []
    for rel in ["data/verified/dataset.jsonl", "data/raw/generation_report.json", "data/raw/rejected.jsonl",
                CUDA_PROGRESS]:
        src = root / rel
        if src.exists():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    for report in (root / "evaluation" / "results").glob("*.json"):
        dst = out / "evaluation" / "results" / report.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report, dst)
        copied.append(str(dst.relative_to(out)))
    runs = sorted((root / "training" / "checkpoints").glob("run_*"), key=lambda p: p.stat().st_mtime)
    if runs:
        run = runs[-1]
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


def zip_outputs(out: Path) -> Path:
    """Zip `out` to <out>.zip via a temp file + rename, so a crash mid-zip keeps the old zip."""
    final = out.with_name(out.name + ".zip")
    tmp = out.with_name(out.name + ".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out))
    os.replace(tmp, final)
    return final


def snapshot(out: Path, session: dict, records: list[dict], root: Path = ROOT) -> Path:
    """Write stage_status.json + session.json, collect artifacts, and refresh the zip."""
    session["stages"] = {r["stage"]: r for r in records}  # dict form, read by the notebook
    report = root / "data/raw/generation_report.json"
    if report.exists():
        try:
            session["dataset_report"] = json.loads(report.read_text())
        except json.JSONDecodeError:
            pass
    session["artifacts"] = collect(out, root)
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    (out / "stage_status.json").write_text(json.dumps(records, indent=2))
    (out / "session.json").write_text(json.dumps(session, indent=2))
    return zip_outputs(out)


def restore_progress(source: Path, root: Path = ROOT) -> bool:
    """Copy a previous session's dataset progress log (from its outputs folder or
    zip) into place so build_dataset --resume continues it. True if found."""
    dst = root / CUDA_PROGRESS
    dst.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        src = source / CUDA_PROGRESS
        if src.exists():
            shutil.copy2(src, dst)
            return True
        return False
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            if CUDA_PROGRESS in zf.namelist():
                dst.write_bytes(zf.read(CUDA_PROGRESS))
                return True
    return False


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

def build_stages(args, names: list[str], logs: Path, eval_config: Path, session: dict,
                 tick: Callable[[], None]) -> list[Stage]:
    py = sys.executable

    def no_gpu() -> Optional[str]:
        return None if session["env"].get("cuda_available") else "no CUDA GPU visible"

    def needs_dataset() -> Optional[str]:
        return no_gpu() or (None if _has_lines(DATASET) else "no verified dataset")

    def needs_adapter() -> Optional[str]:
        run = _latest("training/checkpoints/run_*")
        return no_gpu() or (None if run is not None and (run / "adapter").exists() else "no adapter")

    def command(name, cmd_fn):
        return lambda timeout: run_command(name, cmd_fn(timeout), logs, timeout, on_tick=tick)

    def env_stage(_timeout):
        if not session["env"].get("cuda_available"):
            raise StageFailed("no CUDA GPU visible. On Kaggle: Settings -> Accelerator -> GPU T4, then restart")
        return {"gpu": session["env"].get("gpu")}

    def dataset_cmd(timeout):
        # Ask build_dataset to stop *starting* verifications a few minutes before
        # the watchdog would kill it, so it writes a partial dataset + report.
        margin_s = 60 + 180  # one verification's timeout + time to write the report
        cmd = [py, "data/build_dataset.py", "--device", "cuda",
               "--max-minutes", f"{max(1.0, (timeout - margin_s) / 60):.1f}"]
        return cmd + (["--resume"] if args.resume else [])

    def finetuned_eval_cmd(_timeout):
        run = _latest("training/checkpoints/run_*")
        return [py, "evaluation/finetuned_eval.py", "--device", "cuda", "--config", str(eval_config),
                "--adapter", str(run / "adapter"), "--checkpoint-run", str(run),
                "--limit", str(args.baseline_limit)]

    # min_seconds are rough floors so a stage isn't started with no chance to finish;
    # they are guesses, not measured on a T4.
    all_stages = {
        "env": Stage("env", env_stage, max_seconds=60),
        "harness": Stage("harness", command("harness", lambda t: [py, "-m", "pytest", "-q",
                                                                   "tests/test_verify_kernel.py"]),
                         max_seconds=900, min_seconds=120, skip_reason=no_gpu),
        "dataset": Stage("dataset", command("dataset", dataset_cmd),
                         max_seconds=4 * 3600, min_seconds=900, skip_reason=no_gpu),
        "baseline": Stage("baseline", command("baseline", lambda t: [
                              py, "evaluation/baseline_eval.py", "--device", "cuda", "--config", str(eval_config),
                              "--limit", str(args.baseline_limit)]),
                          max_seconds=3 * 3600, min_seconds=900, skip_reason=needs_dataset),
        "finetune": Stage("finetune", command("finetune", lambda t: [
                              py, "training/finetune.py", "--config", "training/configs/finetune_default.json"]),
                          max_seconds=4 * 3600, min_seconds=1800, skip_reason=needs_dataset),
        "eval": Stage("eval", command("eval", finetuned_eval_cmd),
                      max_seconds=3 * 3600, min_seconds=900, skip_reason=needs_adapter),
    }
    return [all_stages[n] for n in names]


def _default_out() -> Path:
    kaggle = Path("/kaggle/working")
    return (kaggle if kaggle.is_dir() else ROOT) / "kernelforge_outputs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=None)
    parser.add_argument("--finetune", action="store_true", help="also run QLoRA fine-tuning and its evaluation")
    parser.add_argument("--baseline-limit", type=int, default=25,
                        help="held-out examples for each eval (round-robin across categories)")
    parser.add_argument("--max-new-tokens", type=int, default=1536, help="generation cap for evals")
    parser.add_argument("--budget-hours", type=float, default=None,
                        help="wall-clock budget for the whole session; late stages are skipped, not killed")
    parser.add_argument("--resume", action="store_true",
                        help="dataset stage continues from data/raw/progress_cuda.jsonl")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="previous kernelforge_outputs folder or .zip to resume the dataset stage from")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    names = args.stages or [s for s in ALL_STAGES if args.finetune or s not in ("finetune", "eval")]
    out = args.out or _default_out()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    session = {"started_at": datetime.now(timezone.utc).isoformat(), "budget_hours": args.budget_hours,
               "env": env_info()}
    print(json.dumps(session["env"], indent=2))

    if args.resume_from is not None:
        found = restore_progress(args.resume_from)
        print(f"Resume from {args.resume_from}: {'progress log restored' if found else 'no progress log found'}")
        args.resume = args.resume or found

    model_config = ROOT / "training" / "configs" / "qwen2.5-coder-1.5b.json"
    eval_config = out / "eval_model_config.json"
    cfg = json.loads(model_config.read_text())
    cfg["max_new_tokens"] = args.max_new_tokens
    eval_config.write_text(json.dumps(cfg, indent=2))

    records: list[dict] = []

    def save(current: list[dict]) -> None:
        records[:] = current
        snapshot(out, session, records)

    def tick() -> None:  # mid-stage refresh: the records so far are complete, the running stage isn't listed yet
        try:
            snapshot(out, session, records)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    stages = build_stages(args, names, logs, eval_config, session, tick)
    budget_s = args.budget_hours * 3600 if args.budget_hours else None
    records = run_stages(stages, after_stage=save, budget_s=budget_s)

    session["finished_at"] = datetime.now(timezone.utc).isoformat()
    archive = snapshot(out, session, records)
    print(f"\n{'=' * 70}\nSession summary")
    for r in records:
        print(f"  {r['stage']:<10} {r['status']:<8} {r['seconds']:>8}s  {r['error'] or ''}")
    print(f"\nArtifacts: {out}\nDownload:  {archive}")
    return 1 if any(r["status"] == "failed" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
