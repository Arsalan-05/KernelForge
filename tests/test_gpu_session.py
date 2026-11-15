"""CPU-only tests for scripts/gpu_session.py: the stage runner, the time budget,
the subprocess watchdog, and artifact saving (status JSON + atomic zip).

No GPU, model or real pipeline stage runs here. Stages are small Python
callables or `python -c` one-liners; every artifact is a synthetic file in a
tmp dir.
"""

import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "gpu_session.py"
_spec = importlib.util.spec_from_file_location("gpu_session", _PATH)
gs = importlib.util.module_from_spec(_spec)
sys.modules["gpu_session"] = gs  # @dataclass looks its module up in sys.modules
_spec.loader.exec_module(gs)


def ok(_timeout):
    return {"detail": "fine"}


def boom(_timeout):
    raise RuntimeError("synthetic failure")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- run_stages

def test_failed_stage_does_not_stop_later_stages_or_saving():
    saves = []
    records = gs.run_stages(
        [gs.Stage("a", ok, 10), gs.Stage("b", boom, 10), gs.Stage("c", ok, 10)],
        after_stage=lambda recs: saves.append([r["status"] for r in recs]),
    )
    assert [r["status"] for r in records] == ["ok", "failed", "ok"]
    assert records[1]["error"] == "RuntimeError: synthetic failure"
    assert records[0]["detail"] == "fine"
    assert saves == [["ok"], ["ok", "failed"], ["ok", "failed", "ok"]]  # saved after every stage


def test_status_records_have_the_documented_fields():
    (record,) = gs.run_stages([gs.Stage("a", ok, 10)], after_stage=lambda r: None)
    assert {"stage", "status", "seconds", "error"} <= set(record)


def test_skip_reason_skips_without_running():
    ran = []
    records = gs.run_stages(
        [gs.Stage("a", lambda t: ran.append(1), 10, skip_reason=lambda: "no adapter")],
        after_stage=lambda r: None,
    )
    assert ran == [] and records[0]["status"] == "skipped" and records[0]["error"] == "no adapter"


def test_a_crashing_save_does_not_stop_the_session():
    def bad_save(_records):
        raise OSError("disk full (synthetic)")

    records = gs.run_stages([gs.Stage("a", ok, 10), gs.Stage("b", ok, 10)], after_stage=bad_save)
    assert [r["status"] for r in records] == ["ok", "ok"]


def test_interrupt_marks_stage_failed_saves_and_skips_the_rest():
    def interrupted(_timeout):
        raise KeyboardInterrupt

    saves = []
    records = gs.run_stages(
        [gs.Stage("a", interrupted, 10), gs.Stage("b", ok, 10)],
        after_stage=lambda recs: saves.append(len(recs)),
    )
    assert records[0]["status"] == "failed" and records[0]["error"] == "interrupted"
    assert records[1]["status"] == "skipped"
    assert saves == [1, 2]


# ---------------------------------------------------------------- budget

def test_budget_skips_late_stages_cleanly():
    clock = FakeClock()

    def takes(seconds):
        def run(_timeout):
            clock.t += seconds
            return {}
        return run

    records = gs.run_stages(
        [gs.Stage("a", takes(500), 1000, min_seconds=100),
         gs.Stage("b", takes(1), 1000, min_seconds=600)],
        after_stage=lambda r: None, budget_s=1000, reserve_s=100, clock=clock,
    )
    # After "a", 1000 - 500 - 100 = 400 s remain, less than b's 600 s floor.
    assert records[0]["status"] == "ok"
    assert records[1]["status"] == "skipped" and "time budget" in records[1]["error"]


def test_budget_caps_each_stage_timeout():
    clock = FakeClock()
    seen = []
    gs.run_stages([gs.Stage("a", lambda t: seen.append(t), max_seconds=10_000)],
                  after_stage=lambda r: None, budget_s=3600, reserve_s=300, clock=clock)
    assert seen == [3300]


def test_no_budget_uses_stage_cap():
    seen = []
    gs.run_stages([gs.Stage("a", lambda t: seen.append(t), max_seconds=42)], after_stage=lambda r: None)
    assert seen == [42]


# ---------------------------------------------------------------- run_command

def test_run_command_success_writes_log(tmp_path):
    result = gs.run_command("hello", [sys.executable, "-c", "print('hi')"], tmp_path, timeout_s=30, cwd=tmp_path)
    assert result == {"log": "hello.log"}
    assert "hi" in (tmp_path / "hello.log").read_text()


def test_run_command_nonzero_exit_raises(tmp_path):
    with pytest.raises(gs.StageFailed, match="exit code 3"):
        gs.run_command("bad", [sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path, timeout_s=30, cwd=tmp_path)


def test_run_command_timeout_kills_a_running_stage(tmp_path):
    # The child keeps stdout open while sleeping; before the watchdog fix this
    # blocked until the child finished on its own.
    start = time.monotonic()
    with pytest.raises(gs.StageFailed, match="timed out"):
        gs.run_command("slow", [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(30)"],
                       tmp_path, timeout_s=1, cwd=tmp_path)
    assert time.monotonic() - start < 15
    assert "stage killed" in (tmp_path / "slow.log").read_text()


def test_run_command_ticks_while_output_flows(tmp_path):
    ticks = []
    script = "import time\nfor i in range(4):\n    print(i, flush=True)\n    time.sleep(0.3)"
    gs.run_command("tick", [sys.executable, "-c", script], tmp_path, timeout_s=30,
                   on_tick=lambda: ticks.append(1), tick_every_s=0.25, cwd=tmp_path)
    assert len(ticks) >= 2


# ---------------------------------------------------------------- artifacts

def _fake_root(tmp_path):
    root = tmp_path / "repo"
    (root / "data/verified").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    (root / "data/verified/dataset.jsonl").write_text('{"id": "synthetic__tpl__0"}\n')
    (root / gs.CUDA_PROGRESS).write_text('{"id": "synthetic__tpl__0", "retried": false, "result": {}}\n')
    return root


def test_snapshot_writes_status_json_and_zip(tmp_path):
    root = _fake_root(tmp_path)
    out = tmp_path / "kernelforge_outputs"
    out.mkdir()
    records = [{"stage": "dataset", "status": "failed", "seconds": 1.0, "error": "exit code 1"}]

    archive = gs.snapshot(out, {"env": {}}, records, root=root)

    assert archive == tmp_path / "kernelforge_outputs.zip"
    assert json.loads((out / "stage_status.json").read_text()) == records
    names = set(zipfile.ZipFile(archive).namelist())
    assert {"stage_status.json", "session.json", "data/verified/dataset.jsonl", gs.CUDA_PROGRESS} <= names
    assert not (tmp_path / "kernelforge_outputs.zip.tmp").exists()


def test_snapshot_with_nothing_produced_still_zips(tmp_path):
    out = tmp_path / "kernelforge_outputs"
    out.mkdir()
    archive = gs.snapshot(out, {"env": {}}, [], root=tmp_path / "empty_repo")
    assert "stage_status.json" in zipfile.ZipFile(archive).namelist()


def test_restore_progress_from_zip_and_folder(tmp_path):
    root = _fake_root(tmp_path)
    out = tmp_path / "prev_outputs"
    out.mkdir()
    gs.snapshot(out, {"env": {}}, [], root=root)
    expected = (root / gs.CUDA_PROGRESS).read_text()

    for source in (tmp_path / "prev_outputs.zip", out):
        target = tmp_path / f"new_repo_{source.suffix or 'dir'}"
        assert gs.restore_progress(source, root=target)
        assert (target / gs.CUDA_PROGRESS).read_text() == expected


def test_restore_progress_reports_missing_log(tmp_path):
    empty = tmp_path / "empty_outputs"
    empty.mkdir()
    assert gs.restore_progress(empty, root=tmp_path / "repo") is False


# ---------------------------------------------------------------- zip contents

def test_collect_includes_raw_results_reviews_and_run_files_but_not_checkpoints(tmp_path):
    root = _fake_root(tmp_path)
    (root / "data/raw/generation_report.json").write_text("{}")
    (root / "evaluation/results").mkdir(parents=True)
    (root / "evaluation/results/baseline_20261115T000000Z.json").write_text("{}")
    (root / "evaluation/reviews").mkdir()
    (root / "evaluation/reviews/explanation_review_scored.json").write_text("{}")
    old = root / "training/checkpoints/run_old"
    new = root / "training/checkpoints/run_new"
    for run in (old, new):
        (run / "adapter").mkdir(parents=True)
        (run / "adapter/adapter_model.safetensors").write_text("synthetic weights")
        (run / "split_info.json").write_text("{}")
        (run / "checkpoint-30").mkdir()
        (run / "checkpoint-30/optimizer.pt").write_text("synthetic optimizer state")
        (run / "checkpoint-30/trainer_state.json").write_text("{}")
    import os
    os.utime(old, (1, 1))  # make run_new the latest run

    out = tmp_path / "out"
    copied = set(gs.collect(out, root=root))

    assert {"data/raw/generation_report.json", gs.CUDA_PROGRESS,
            "evaluation/results/baseline_20261115T000000Z.json",
            "evaluation/reviews/explanation_review_scored.json",
            "training/run_old/split_info.json", "training/run_new/split_info.json",
            "training/run_new/checkpoint-30/trainer_state.json",
            "training/run_new/adapter/adapter_model.safetensors"} <= copied
    assert not any(c.endswith("optimizer.pt") for c in copied)       # no checkpoint weights
    assert "training/run_old/adapter/adapter_model.safetensors" not in copied  # adapter: latest run only


def test_session_json_records_the_zip_file_list(tmp_path):
    root = _fake_root(tmp_path)
    out = tmp_path / "kernelforge_outputs"
    out.mkdir()
    archive = gs.snapshot(out, {"env": {}}, [], root=root)
    session = json.loads((out / "session.json").read_text())
    assert session["zip_files"] == sorted(zipfile.ZipFile(archive).namelist())


def test_large_zip_prints_a_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gs, "ZIP_WARN_BYTES", 0)
    out = tmp_path / "kernelforge_outputs"
    out.mkdir()
    (out / "x.txt").write_text("synthetic")
    gs.zip_outputs(out)
    assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------- eval size

def test_eval_summary_flags_full_and_partial_held_out_sets(tmp_path):
    report = tmp_path / "baseline_x.json"
    report.write_text(json.dumps({"results": [{}] * 100}))
    full = gs.eval_summary(report, held_out_total=100, eval_limit=None)
    assert full["prompts_evaluated"] == 100 and full["full_held_out_set"] is True

    report.write_text(json.dumps({"results": [{}] * 25}))
    capped = gs.eval_summary(report, held_out_total=100, eval_limit=25)
    assert capped["prompts_evaluated"] == 25 and capped["full_held_out_set"] is False

    missing = gs.eval_summary(None, held_out_total=100, eval_limit=None)
    assert missing["prompts_evaluated"] is None and missing["full_held_out_set"] is False


def test_new_report_ignores_reports_from_before_the_stage(tmp_path):
    old = tmp_path / "baseline_old.json"
    old.write_text("{}")
    before = {old}
    assert gs._new_report("baseline", before, results_dir=tmp_path) is None
    new = tmp_path / "baseline_new.json"
    new.write_text("{}")
    (tmp_path / "finetuned_new.json").write_text("{}")
    assert gs._new_report("baseline", before, results_dir=tmp_path) == new


def _capture_baseline_cmd(monkeypatch, tmp_path, argv):
    """Run the baseline stage with run_command stubbed out; return the command it would run."""
    import argparse
    seen = {}
    monkeypatch.setattr(gs, "run_command", lambda name, cmd, *a, **k: seen.setdefault("cmd", cmd) and {})
    monkeypatch.setattr(gs, "held_out_count", lambda: 100)
    monkeypatch.setattr(gs, "_new_report", lambda prefix, before: None)
    args = argparse.Namespace(eval_limit=None, resume=False)
    for k, v in argv.items():
        setattr(args, k, v)
    session = {"env": {"cuda_available": True}}
    (stage,) = gs.build_stages(args, ["baseline"], tmp_path, tmp_path / "cfg.json", session, tick=lambda: None)
    record = stage.run(60)
    return seen["cmd"], record


def test_default_eval_uses_all_held_out_prompts(monkeypatch, tmp_path):
    cmd, record = _capture_baseline_cmd(monkeypatch, tmp_path, {})
    assert "--limit" not in cmd
    assert record["eval_limit"] is None and record["held_out_total"] == 100


def test_eval_limit_is_passed_through(monkeypatch, tmp_path):
    cmd, record = _capture_baseline_cmd(monkeypatch, tmp_path, {"eval_limit": 25})
    assert cmd[-2:] == ["--limit", "25"] and record["eval_limit"] == 25
