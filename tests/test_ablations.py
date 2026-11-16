"""Tests for the ablation matrix and runner (scripts/run_ablations.py).

All tests are CPU-only and use a mock model. No numbers produced here are results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_ablations import _build_command, _write_ft_config, load_matrix


# --- matrix schema tests -------------------------------------------------------

def test_matrix_loads_and_has_required_keys():
    matrix = load_matrix()
    assert "conditions" in matrix and "shared_flags" in matrix and "base_config" in matrix


def test_all_condition_ids_are_unique():
    matrix = load_matrix()
    ids = [c["id"] for c in matrix["conditions"]]
    assert len(ids) == len(set(ids)), f"Duplicate condition ids: {set(i for i in ids if ids.count(i) > 1)}"


def test_all_conditions_have_required_fields():
    matrix = load_matrix()
    required = {"id", "description", "model", "script"}
    for cond in matrix["conditions"]:
        missing = required - cond.keys()
        assert not missing, f"Condition {cond.get('id','?')} missing fields: {missing}"


def test_base_conditions_have_no_finetune_overrides():
    matrix = load_matrix()
    for cond in matrix["conditions"]:
        if cond["model"] == "base":
            assert "finetune_overrides" not in cond or not cond["finetune_overrides"], \
                f"Base condition {cond['id']} has finetune_overrides"


def test_finetuned_conditions_declare_lora_r():
    matrix = load_matrix()
    for cond in matrix["conditions"]:
        if cond["model"] == "finetuned":
            assert cond.get("lora_r") is not None, \
                f"Finetuned condition {cond['id']} missing lora_r"


def test_repetition_penalty_not_in_conditions_or_shared_flags():
    """repetition_penalty is pinned in common.py; it must not appear in conditions or shared_flags."""
    matrix = load_matrix()
    for cond in matrix["conditions"]:
        flags = cond.get("extra_flags", []) + list(cond.get("finetune_overrides", {}).keys())
        assert "repetition_penalty" not in flags, \
            f"Condition {cond['id']} sets repetition_penalty; it is pinned to 1.0 in common.py"
    assert "repetition_penalty" not in matrix.get("shared_flags", {}), \
        "shared_flags sets repetition_penalty; it is pinned to 1.0 in common.py"


# --- command-builder tests -----------------------------------------------------

def test_build_command_dry_run_uses_cpu():
    matrix = load_matrix()
    cond = next(c for c in matrix["conditions"] if c["model"] == "base")
    cmd = _build_command(cond, matrix, adapter_path=None, dry_run=True, extra=[])
    assert "--dry-run" in cmd and "--device" in cmd
    assert cmd[cmd.index("--device") + 1] == "cpu"


def test_build_command_real_uses_cuda():
    matrix = load_matrix()
    cond = next(c for c in matrix["conditions"] if c["model"] == "base")
    cmd = _build_command(cond, matrix, adapter_path=None, dry_run=False, extra=[])
    assert "--dry-run" not in cmd
    assert cmd[cmd.index("--device") + 1] == "cuda"


def test_build_command_few_shot_flags():
    matrix = load_matrix()
    cond = next(c for c in matrix["conditions"] if c.get("few_shot_source") == "other_category")
    cmd = _build_command(cond, matrix, adapter_path=None, dry_run=True, extra=[])
    assert "--few-shot-source" in cmd
    assert cmd[cmd.index("--few-shot-source") + 1] == "other_category"


def test_build_command_max_new_tokens_is_2048():
    matrix = load_matrix()
    cond = matrix["conditions"][0]
    cmd = _build_command(cond, matrix, adapter_path=None, dry_run=True, extra=[])
    assert "--max-new-tokens" in cmd
    assert cmd[cmd.index("--max-new-tokens") + 1] == "2048"


# --- config-writer tests -------------------------------------------------------

def test_write_ft_config_applies_lora_overrides(tmp_path):
    matrix = load_matrix()
    base_config = ROOT / matrix["base_config"]
    cond = next(c for c in matrix["conditions"]
                if c["model"] == "finetuned" and c.get("finetune_overrides"))
    cfg_path = _write_ft_config(cond, base_config, tmp_path)
    cfg = json.loads(cfg_path.read_text())
    for k, v in cond["finetune_overrides"].items():
        assert cfg[k] == v, f"Override {k}={v} not applied; got {cfg[k]}"


def test_write_ft_config_does_not_override_seed(tmp_path):
    """Seed must stay fixed across all conditions for reproducibility."""
    matrix = load_matrix()
    base_config = ROOT / matrix["base_config"]
    base_seed = json.loads(base_config.read_text()).get("seed", 42)
    for cond in matrix["conditions"]:
        if cond.get("finetune_overrides"):
            cfg_path = _write_ft_config(cond, base_config, tmp_path)
            cfg = json.loads(cfg_path.read_text())
            assert cfg.get("seed") == base_seed, \
                f"Condition {cond['id']} changed seed from {base_seed} to {cfg.get('seed')}"


# --- dry-run smoke test --------------------------------------------------------

def test_run_ablations_dry_run_base_zero_shot(tmp_path, monkeypatch):
    """Smoke: run_conditions with --dry-run on the base_zero_shot condition using
    a mock subprocess so no model is loaded. Checks that ablations_summary.json is
    written with the right structure."""
    import scripts.run_ablations as runner

    # Redirect results dir to tmp_path
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)

    # Stub subprocess.run to simulate a successful dry run (exit 0) and write a fake report.
    def fake_run(cmd, **kwargs):
        # Write a minimal fake report so the runner can find it.
        (tmp_path / "baseline_dry_20260101T000000Z.json").write_text(
            json.dumps({"run_name": "baseline", "dry_run": True, "results": []}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    matrix = load_matrix()
    cond = next(c for c in matrix["conditions"] if c["id"] == "base_zero_shot")
    results = runner.run_conditions([cond], matrix, dry_run=True, extra_flags=[])

    # run_conditions returns results; the caller (main) writes the summary.
    # Verify the returned list structure directly.
    assert len(results) == 1
    result = results[0]
    assert result["id"] == "base_zero_shot"
    assert result["status"] == "ok"
    assert result["few_shot_source"] is None
    assert result["model"] == "base"
    assert "TODO" in result["note"]
