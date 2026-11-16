"""Run the KernelForge ablation matrix dry-run or for real on a GPU.

Usage:
    python scripts/run_ablations.py --dry-run      # CPU smoke run with mock model
    python scripts/run_ablations.py --conditions base_zero_shot base_few_shot_other
    python scripts/run_ablations.py                 # requires GPU + verified dataset

The matrix is defined in evaluation/ablation_matrix.json. Each condition maps to
one eval script invocation. This script:
  1. Reads the matrix.
  2. For fine-tuned conditions, writes a temporary config with the LoRA overrides.
  3. Launches the eval script as a subprocess with the right flags.
  4. Collects exit codes and report paths.
  5. Writes ablations_summary.json to evaluation/results/ with per-condition status.

All numbers in ablations_summary.json are TODO until real GPU runs; the summary
records the command, exit code and report path so the caller knows whether each
condition ran and where its output is.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = ROOT / "evaluation" / "ablation_matrix.json"
RESULTS_DIR = ROOT / "evaluation" / "results"


def load_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


def _adapter_for(condition: dict, results_dir: Path) -> Path | None:
    """Return the adapter path for a fine-tuned condition, or None for base."""
    if condition["model"] != "finetuned":
        return None
    rank = condition.get("lora_r", 16)
    # Adapter is expected at training/checkpoints/run_<id>/adapter
    # For dry-run / mock, we let the eval script handle the missing adapter.
    adapter = ROOT / "training" / "checkpoints" / f"run_{condition['id']}" / "adapter"
    return adapter


def _build_command(
    condition: dict,
    matrix: dict,
    adapter_path: Path | None,
    dry_run: bool,
    extra: list[str],
) -> list[str]:
    """Build the subprocess command for one condition."""
    script = ROOT / condition["script"]
    shared = matrix.get("shared_flags", {})
    cmd = [
        sys.executable, str(script),
        "--max-new-tokens", str(shared.get("max_new_tokens", 2048)),
        "--split", str(shared.get("split", "template")),
        "--seed", str(shared.get("seed", 42)),
    ]
    if dry_run:
        cmd += ["--dry-run", "--device", "cpu"]
    else:
        cmd += ["--device", shared.get("device", "cuda")]

    for flag in condition.get("extra_flags", []):
        cmd.append(flag)

    if adapter_path is not None:
        cmd += ["--adapter", str(adapter_path)]

    cmd += extra
    return cmd


def _write_ft_config(condition: dict, base_config_path: Path, tmp_dir: Path) -> Path:
    """Write a fine-tune config with LoRA overrides for this condition."""
    base_cfg = json.loads(base_config_path.read_text())
    cfg = {**base_cfg, **condition.get("finetune_overrides", {})}
    path = tmp_dir / f"config_{condition['id']}.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def run_conditions(
    conditions: list[dict],
    matrix: dict,
    dry_run: bool,
    extra_flags: list[str],
) -> list[dict]:
    """Run each condition and return a list of result dicts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_config = ROOT / matrix["base_config"]
    results = []
    with tempfile.TemporaryDirectory(prefix="kernelforge_ablation_") as tmp:
        tmp = Path(tmp)
        for cond in conditions:
            print(f"\n{'='*60}")
            print(f"Condition: {cond['id']}  —  {cond['description']}")
            adapter = _adapter_for(cond, RESULTS_DIR)
            cmd = _build_command(cond, matrix, adapter, dry_run, extra_flags)
            if cond.get("finetune_overrides"):
                cfg_path = _write_ft_config(cond, base_config, tmp)
                # The fine-tune config path is not passed to eval scripts (they
                # read from the adapter's directory); it's written here for
                # reference only and recorded in the summary.
            else:
                cfg_path = base_config
            print("Command:", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=False)
            status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
            # Find the report file written by this run (newest .json in results/)
            reports = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
            latest = str(reports[-1]) if reports else None
            result = {
                "id": cond["id"],
                "description": cond["description"],
                "model": cond["model"],
                "few_shot_source": cond.get("few_shot_source"),
                "lora_r": cond.get("lora_r"),
                "status": status,
                "command": cmd,
                "report": latest,
                "config": str(cfg_path),
                "note": "TODO: values are populated from the report after a real GPU run",
            }
            results.append(result)
            print(f"  → {status}; report: {latest}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(MATRIX_PATH),
                        help="Path to ablation_matrix.json (default: evaluation/ablation_matrix.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU smoke run with --dry-run flag; no GPU or real model needed")
    parser.add_argument("--conditions", nargs="*",
                        help="Run only these condition IDs (default: all)")
    parser.add_argument("--extra-flags", nargs="*", default=[],
                        help="Extra flags passed to every eval script invocation")
    args = parser.parse_args()

    matrix = json.loads(Path(args.matrix).read_text())
    all_conditions = matrix["conditions"]
    if args.conditions:
        selected = [c for c in all_conditions if c["id"] in args.conditions]
        missing = set(args.conditions) - {c["id"] for c in selected}
        if missing:
            print(f"WARNING: unknown condition ids: {missing}", file=sys.stderr)
    else:
        selected = all_conditions

    if not selected:
        print("No conditions to run.", file=sys.stderr)
        sys.exit(1)

    results = run_conditions(selected, matrix, args.dry_run, args.extra_flags or [])

    summary_path = RESULTS_DIR / "ablations_summary.json"
    summary_path.write_text(json.dumps({
        "matrix": str(Path(args.matrix).resolve()),
        "dry_run": args.dry_run,
        "conditions": results,
    }, indent=2))
    print(f"\nSummary written to {summary_path}")

    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        print(f"\nFailed conditions: {[r['id'] for r in failed]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
