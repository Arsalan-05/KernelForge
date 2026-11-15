"""Tests for evaluation/report (python -m evaluation.report).

ALL DATA IN THIS FILE IS SYNTHETIC. The fixtures below are made-up numbers that
exist only to exercise parsing, missing-stage handling, intervals and table
rendering. They are not results and must never be presented as results.

Where possible the fixtures are written by the project's *real* writers
(evaluation.common.save_eval_report, data.build_dataset._write_dataset), fed
synthetic VerifyResults, so a change to an artifact format breaks these tests
instead of silently breaking the report.
"""

from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import data.build_dataset as build_dataset
import evaluation.common as common
from evaluation.report.__main__ import build_report, main
from evaluation.report.artifacts import Artifacts, CATEGORIES, CategoryResult, load_artifacts, parse_eval_report
from evaluation.report.figures import make_figures
from evaluation.report.latex import FRAGMENT_NAMES, combined, fragments
from evaluation.report.render import (
    NOT_RUN,
    limitations,
    phase5_table,
    prop_cell,
    readme_snippet,
    stage_status,
)
from evaluation.report.stats import Proportion, geomean, mean, median, wilson_interval
from kernelforge.dataset import CATEGORIES as DATASET_CATEGORIES
from verification.verify_kernel import VerifyResult

SYNTHETIC_KERNEL_OUTPUT = (
    "kernel:\n```python\nimport triton\nclass ModelNew:\n    pass\n```\n\n"
    "explanation:\nSYNTHETIC explanation for tests."
)


# --------------------------------------------------------------------------- synthetic fixture builders


def _synthetic_entry_results(category_outcomes: dict[str, list[tuple]], *, interpreted=False):
    """[(status, correct, speedup), ...] per category -> EntryEvalResults (SYNTHETIC)."""
    results = []
    for category, outcomes in category_outcomes.items():
        for i, (status, correct, speedup) in enumerate(outcomes):
            v = VerifyResult(status=status, correct=correct, speedup=speedup, interpreted=interpreted)
            results.append(common.evaluate_generated_kernel(
                {"id": f"{category}__synthetic_tpl__s{i}", "category": category, "pytorch_reference": ""},
                SYNTHETIC_KERNEL_OUTPUT, skip_verification=True))
            results[-1].verification = v
    return results


def _write_synthetic_eval(out_dir: Path, monkeypatch, run_name: str, outcomes, *, adapter=None,
                          interpreted=False, extra=None) -> Path:
    monkeypatch.setattr(common, "RESULTS_DIR", out_dir / "evaluation" / "results")
    results = _synthetic_entry_results(outcomes, interpreted=interpreted)
    path = common.save_eval_report(
        run_name=run_name, model_name="SYNTHETIC-model", adapter_path=adapter, device="cuda",
        results=results, metrics=common.aggregate_metrics(results),
        extra={"dry_run": False, "split": "template", "held_out_templates": ["attention__synthetic_tpl"],
               **(extra or {})},
    )
    # save_eval_report names files by second; make names unique for back-to-back calls.
    unique = path.with_name(f"{run_name}_{len(list(path.parent.glob('*.json')))}.json")
    path.rename(unique)
    return unique


def _write_synthetic_dataset_report(out_dir: Path, monkeypatch) -> None:
    raw = out_dir / "data" / "raw"
    monkeypatch.setattr(build_dataset, "VERIFIED_PATH", out_dir / "data" / "verified" / "dataset.jsonl")
    monkeypatch.setattr(build_dataset, "REJECTED_PATH", raw / "rejected.jsonl")
    monkeypatch.setattr(build_dataset, "REPORT_PATH", raw / "generation_report.json")
    rows = [
        ("norm__synthetic_rms", VerifyResult("ok", True, 2.0)),
        ("norm__synthetic_rms", VerifyResult("ok", True, 4.0)),
        ("norm__synthetic_rms", VerifyResult("ok", False)),
        ("rope__synthetic_half", VerifyResult("crash", None, error="CUDA out of memory")),
        ("rope__synthetic_half", VerifyResult("timeout", None)),
        ("rope__synthetic_half", VerifyResult("error", None, error="SYNTHETIC failure")),
    ]
    results = [({"id": f"{tpl}__s{i}", "category": tpl.split("__")[0]}, r) for i, (tpl, r) in enumerate(rows)]
    build_dataset._write_dataset(results, SimpleNamespace(device="cuda", smoke=False))


BASE_OUTCOMES = {  # SYNTHETIC: 2/4 correct in norm, 1/2 in rope, none elsewhere
    "norm": [("ok", True, 1.5), ("ok", True, 2.5), ("ok", False, None), ("error", None, None)],
    "rope": [("ok", True, float("inf")), ("timeout", None, None)],
}


# --------------------------------------------------------------------------- statistics


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert wilson_interval(0, 10) == (pytest.approx(0.0), pytest.approx(0.2775, abs=1e-4))
    assert wilson_interval(10, 10) == (pytest.approx(0.7225, abs=1e-4), pytest.approx(1.0))
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_proportion_small_flag_and_means():
    assert Proportion(1, 19).small and not Proportion(1, 20).small
    assert mean([1.0, 3.0, float("inf"), None]) == 2.0
    assert median([3.0, 1.0, 2.0, 4.0]) == 2.5 and mean([]) is None


def test_categories_match_dataset_module():
    assert CATEGORIES == tuple(DATASET_CATEGORIES)


# --------------------------------------------------------------------------- parsing


def test_parses_report_written_by_save_eval_report(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path, monkeypatch, "baseline", BASE_OUTCOMES)
    art = load_artifacts(tmp_path)
    run = art.run("base")
    assert run is not None and run.model_name == "SYNTHETIC-model" and run.split == "template"
    assert (run.categories["norm"].n, run.categories["norm"].correct) == (4, 2)
    assert run.categories["norm"].speedups == [1.5, 2.5]
    assert run.categories["rope"].speedups == [] and run.categories["rope"].infinite_speedups == 1
    overall = run.overall()
    assert (overall.n, overall.correct) == (6, 3)


def test_dry_run_and_parse_only_reports_are_excluded(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path, monkeypatch, "baseline", BASE_OUTCOMES, extra={"dry_run": True})
    _write_synthetic_eval(tmp_path, monkeypatch, "finetuned", BASE_OUTCOMES, adapter="a",
                          extra={"skip_verification": True})
    art = load_artifacts(tmp_path)
    assert art.runs == {}
    reasons = " ".join(why for _, why in art.excluded)
    assert "dry run" in reasons and "parse-only" in reasons


def test_search_mode_from_key_or_run_name():
    report = {"run_name": "finetuned", "adapter_path": "x", "results": [
        {"id": "a", "category": "norm", "verification": {"status": "ok", "correct": True, "speedup": 1.0}}]}
    run, _ = parse_eval_report({**report, "search": {"num_candidates": 4, "repair_rounds": 1}}, "s.json")
    assert run.key == ("ft", 4, 1, "unspecified") and run.mode_label == "best-of-4 + repair"
    run, _ = parse_eval_report({**report, "run_name": "baseline_best4", "adapter_path": None}, "s.json")
    assert run.key == ("base", 4, 0, "unspecified")
    run, _ = parse_eval_report(report, "s.json")
    assert run.key == ("ft", 1, 0, "unspecified") and run.mode_label == "pass@1"


def test_summary_only_report_falls_back_to_per_category():
    report = {"run_name": "baseline", "per_category": {
        "norm": {"total": 5, "correct": 2, "passed": 2, "avg_speedup": 1.25}}}
    run, _ = parse_eval_report(report, "s.json")
    assert run.from_summary and run.categories["norm"].avg_speedup == 1.25
    assert run.categories["norm"].speedup_n == 2


def test_newest_report_wins_and_older_is_listed(tmp_path, monkeypatch):
    old = _write_synthetic_eval(tmp_path, monkeypatch, "baseline", {"norm": [("ok", False, None)]})
    data = json.loads(old.read_text())
    data["generated_at"] = "2000-01-01T00:00:00+00:00"
    old.write_text(json.dumps(data))
    _write_synthetic_eval(tmp_path, monkeypatch, "baseline", BASE_OUTCOMES)
    art = load_artifacts(tmp_path)
    assert art.run("base").categories["norm"].n == 4
    assert art.superseded == [str(old.relative_to(tmp_path))]


def test_parses_generation_report_written_by_build_dataset(tmp_path, monkeypatch):
    _write_synthetic_dataset_report(tmp_path, monkeypatch)
    ds = load_artifacts(tmp_path).dataset
    norm, rope = ds.per_category["norm"], ds.per_category["rope"]
    assert (norm.passed, norm.incorrect, norm.total) == (2, 1, 3)
    assert (rope.oom, rope.timeout, rope.other) == (1, 1, 1)
    assert ds.per_template["norm__synthetic_rms"].passed == 2


def test_explanation_review_counts_only_scored_items(tmp_path):
    review = {"source_report": "evaluation/results/finetuned_x.json", "rubric": {}, "items": [
        {"rubric_score": 2}, {"rubric_score": 1}, {"rubric_score": None}]}
    (tmp_path / "explanation_review_ft.json").write_text(json.dumps(review))
    r = load_artifacts(tmp_path).reviews[0]
    assert r.model == "ft" and r.scores == [2, 1] and r.n_items == 3 and r.mean_score == 1.5


def test_bad_json_is_reported_not_fatal(tmp_path):
    (tmp_path / "session.json").write_text("{not json")
    art = load_artifacts(tmp_path)
    assert art.problems and "session.json" in art.problems[0]


# --------------------------------------------------------------------------- missing stages & rendering


def test_empty_outputs_say_not_run_and_never_crash(tmp_path):
    art = load_artifacts(tmp_path)
    table = phase5_table(art)
    assert all(cell == NOT_RUN for row in table[1:] for cell in row[1:])
    assert all(status.startswith("absent") for _, status in stage_status(art))
    assert "No evaluation results yet" in readme_snippet(art)
    assert any("No model evaluation results" in line for line in limitations(art))


def test_session_status_explains_missing_stage(tmp_path):
    session = {"stages": {"baseline": {"status": "skipped (no verified dataset)"}}}
    (tmp_path / "session.json").write_text(json.dumps(session))
    status = dict(stage_status(load_artifacts(tmp_path)))
    assert status["Base, pass@1"] == "absent (session: skipped (no verified dataset))"


def test_prop_cell_formats_and_flags_small_n():
    assert prop_cell(Proportion(5, 10)) == "50% (5/10) [24–76]†"
    assert prop_cell(Proportion(10, 20)) == "50% (10/20) [30–70]"
    assert prop_cell(Proportion(0, 0)) == "no samples"
    assert prop_cell(Proportion(5, 10), latex=True) == "50\\% (5/10) [24--76]$^\\dagger$"


def test_phase5_table_cells(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path, monkeypatch, "baseline", BASE_OUTCOMES)
    table = phase5_table(load_artifacts(tmp_path))
    assert table[0][1:5] == ["Base % correct", "Base mean speedup", "Base geomean speedup", "Base fast_1"]
    rows = {r[0]: r for r in table}
    assert rows["Norm"][1] == "50% (2/4) [15–85]†"
    assert rows["Norm"][2] == "2.00× (n=2)†"  # mean of the SYNTHETIC 1.5x and 2.5x
    assert rows["Norm"][3] == "1.94× (n=2)†"  # sqrt(1.5 * 2.5)
    assert rows["Norm"][4] == "50% (2/4) [15–85]†"  # both correct kernels beat 1x
    assert rows["RoPE"][2] == "– (none timed)"  # the only correct rope kernel had speedup=inf
    assert rows["RoPE"][4] == "0% (0/2) [0–66]†"  # inf is not counted as fast
    assert rows["Attention"][1] == "no samples"
    assert all(rows["Norm"][i] == NOT_RUN for i in range(5, 10))  # no FT runs
    assert rows["All"][1] == "50% (3/6) [19–81]†"


def test_interpreter_runs_report_no_speedup(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path, monkeypatch, "baseline", BASE_OUTCOMES, interpreted=True)
    art = load_artifacts(tmp_path)
    assert phase5_table(art)[4][2] == "n/a (interpreter)"  # Norm row, Base speedup
    assert any("Interpreter-only" in line for line in limitations(art))


def test_latex_fragments_are_escaped_and_wrapped(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path, monkeypatch, "baseline", BASE_OUTCOMES)
    tex = combined(fragments(load_artifacts(tmp_path)))
    assert "Base \\% correct" in tex and "2.00$\\times$" in tex and "Base fast\\_1" in tex
    assert "\\resizebox{\\columnwidth}{!}{%" in tex and "\\begin{tabular}{l" + "r" * 9 + "}" in tex
    assert "booktabs" in tex and "graphicx" in tex  # packages are stated
    body = [line for line in tex.splitlines() if not line.startswith("%")]  # TeX comments may hold a bare %
    assert all("%" not in line.replace("\\%", "").rstrip("%") for line in body)


def test_limitations_flag_small_n_and_split(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path, monkeypatch, "baseline", BASE_OUTCOMES)
    text = " ".join(limitations(load_artifacts(tmp_path)))
    assert "Small samples" in text and "attention__synthetic_tpl" in text


# --------------------------------------------------------------------------- end to end


def test_build_report_from_zip_writes_all_files(tmp_path, monkeypatch):
    outputs = tmp_path / "kernelforge_outputs"
    _write_synthetic_eval(outputs, monkeypatch, "baseline", BASE_OUTCOMES)
    _write_synthetic_eval(outputs, monkeypatch, "finetuned", BASE_OUTCOMES, adapter="adapter",
                          extra={"search": {"num_candidates": 4, "repair_rounds": 1}})
    _write_synthetic_dataset_report(outputs, monkeypatch)
    archive = tmp_path / "kernelforge_outputs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in outputs.rglob("*"):
            zf.write(path, path.relative_to(outputs).as_posix())

    out = tmp_path / "report"
    written = build_report(archive, out, figures=False)
    names = {p.name for p in written}
    assert {"results.md", "readme_results_snippet.md", "tables.tex", "summary.json", "table_categories.tex",
            "table_search.tex", "table_dataset.tex"} <= names
    assert not {"table_training.tex", "table_rubric.tex"} & names  # no training run, no review
    md = (out / "results.md").read_text()
    assert "| Fine-tuned, best-of-4 + repair | present" in md
    assert "| Fine-tuned, pass@1 | absent" in md
    summary = json.loads((out / "summary.json").read_text())
    assert summary["runs"]["base/unspecified/pass@1"]["all"]["correct"] == 3
    assert math.isclose(summary["runs"]["base/unspecified/pass@1"]["per_category"]["norm"]["avg_speedup"], 2.0)


def test_figures_are_drawn_when_matplotlib_present(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    outputs = tmp_path / "outputs"
    _write_synthetic_eval(outputs, monkeypatch, "baseline", BASE_OUTCOMES)
    out = tmp_path / "report"
    build_report(outputs, out)
    assert (out / "figures" / "correctness_by_category.png").stat().st_size > 0
    assert (out / "figures" / "speedup_distribution.png").stat().st_size > 0
    assert not (out / "figures" / "search_gain.png").exists()  # no search run
    assert "search_gain: not drawn (no search runs" in (out / "results.md").read_text()


def test_cli_rejects_missing_source(tmp_path, capsys):
    assert main([str(tmp_path / "nope.zip"), "--out", str(tmp_path / "o")]) == 2
    assert "does not exist" in capsys.readouterr().err


# --------------------------------------------------------------------------- speedup metrics (SYNTHETIC numbers)


def test_geomean_and_fast_1_on_synthetic_speedups():
    # SYNTHETIC: 4 attempts, 3 correct; one timed 2x, one 0.5x, one at 0 s (inf).
    c = CategoryResult(n=4, correct=3, speedups=[2.0, 0.5], infinite_speedups=1)
    assert c.avg_speedup == 1.25  # arithmetic mean
    assert c.geomean_speedup == pytest.approx(1.0)  # 2x and 0.5x cancel
    assert (c.fast_1.successes, c.fast_1.n) == (1, 4)  # only the 2x kernel; n = ALL attempts
    assert geomean([4.0, 1.0, float("inf"), -1.0]) == pytest.approx(2.0)


def test_summary_only_report_has_no_geomean_or_fast_1():
    report = {"run_name": "baseline", "per_category": {
        "norm": {"total": 5, "correct": 2, "passed": 2, "avg_speedup": 1.25}}}
    run, _ = parse_eval_report(report, "SYNTHETIC.json")
    assert run.categories["norm"].geomean_speedup is None and run.categories["norm"].fast_1 is None
    art = Artifacts(source="SYNTHETIC", runs={run.key: run})
    norm = {r[0]: r for r in phase5_table(art)}["Norm"]
    assert norm[2] == "1.25× (n=2)†" and norm[3] == norm[4] == "n/a (summary-only report)"


def test_summary_json_and_limitations_carry_new_metrics(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    _write_synthetic_eval(outputs, monkeypatch, "baseline", BASE_OUTCOMES)
    build_report(outputs, tmp_path / "report", figures=False)
    norm = json.loads((tmp_path / "report" / "summary.json").read_text())["runs"]["base/unspecified/pass@1"]["per_category"]
    assert norm["norm"]["geomean_speedup"] == pytest.approx(math.sqrt(1.5 * 2.5))
    assert (norm["norm"]["fast_1"]["k"], norm["norm"]["fast_1"]["n"]) == (2, 4)
    assert norm["rope"]["infinite_speedups_excluded"] == 1
    text = " ".join(limitations(load_artifacts(outputs)))
    assert "fast_1" in text and "geometric mean" in text and "speedup = inf" in text


# --------------------------------------------------------------------------- LaTeX fragment skipping


def _tex_files(out: Path) -> set[str]:
    return {p.name for p in out.glob("*.tex")}


def test_no_data_writes_no_latex_at_all(tmp_path):
    (tmp_path / "src").mkdir()
    build_report(tmp_path / "src", tmp_path / "out", figures=False)
    assert _tex_files(tmp_path / "out") == set()  # not even tables.tex: the paper shows its TODO boxes


def test_only_fragments_with_data_are_written(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path / "src", monkeypatch, "baseline", BASE_OUTCOMES)  # pass@1 only
    build_report(tmp_path / "src", tmp_path / "out", figures=False)
    # pass@1 alone is not a search result, and there is no dataset/training/review data.
    assert _tex_files(tmp_path / "out") == {"table_categories.tex", "tables.tex"}


def test_interpreter_dataset_check_is_not_a_dataset_table(tmp_path):
    report = {"templates": {"norm/rmsnorm": [{"id": "x", "passed": True, "status": "ok"}]}, "smoke": True}
    (tmp_path / "interpret_report.json").write_text(json.dumps(report))  # SYNTHETIC
    assert "table_dataset" not in fragments(load_artifacts(tmp_path))


def test_stale_fragment_is_removed_when_its_data_is_gone(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path / "src", monkeypatch, "baseline", BASE_OUTCOMES)
    out = tmp_path / "out"
    out.mkdir()
    (out / "table_search.tex").write_text("% left over from an older SYNTHETIC run")
    build_report(tmp_path / "src", out, figures=False)
    assert not (out / "table_search.tex").exists()


def test_rubric_fragment_needs_scored_items(tmp_path):
    review = {"source_report": "baseline_x.json", "rubric": {}, "items": [{"rubric_score": None}]}
    (tmp_path / "review.json").write_text(json.dumps(review))  # SYNTHETIC, unscored
    assert "table_rubric" not in fragments(load_artifacts(tmp_path))
    review["items"] = [{"rubric_score": 2}, {"rubric_score": 0}]
    (tmp_path / "review.json").write_text(json.dumps(review))
    tex = fragments(load_artifacts(tmp_path))["table_rubric"]
    assert "Base & 2/2 & 1.00 & 1 / 0 / 1" in tex


def _write_synthetic_training_run(root: Path, *, with_state: bool) -> None:
    run = root / "training" / "run_20990101T000000Z"
    run.mkdir(parents=True)
    (run / "split_info.json").write_text(json.dumps({  # SYNTHETIC split
        "train": 9, "val": 1, "test": 5, "split": "template", "held_out_templates": ["norm__synthetic_tpl"],
        "config": {"model_name": "SYNTHETIC-model", "lora_r": 16, "lora_alpha": 32, "learning_rate": 2e-4,
                   "num_epochs": 3, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8},
        "created_at": "2099-01-01T00:00:00+00:00"}))
    if with_state:
        for step, loss in ((10, 9.9), (20, 0.5)):  # SYNTHETIC losses; the newest checkpoint must win
            (run / f"checkpoint-{step}").mkdir()
            (run / f"checkpoint-{step}" / "trainer_state.json").write_text(json.dumps({
                "epoch": step / 10, "global_step": step,
                "log_history": [{"loss": loss + 1, "step": step}, {"eval_loss": loss + 0.1, "step": step},
                                {"train_loss": loss, "train_runtime": 3600.0 * step / 10, "step": step}]}))
        (run / "training_metrics.json").write_text(json.dumps({"peak_memory_gb": 7.5}))


def test_training_fragment_from_trainer_state(tmp_path):
    _write_synthetic_training_run(tmp_path, with_state=True)
    art = load_artifacts(tmp_path)
    assert art.training.trainer_state_source.endswith("checkpoint-20/trainer_state.json")
    tex = fragments(art)["table_training"]
    for expected in ("Final train loss & 0.5000", "Final validation loss & 0.6000", "Training wall-clock & 2.00 h",
                     "Peak GPU memory & 7.5 GB", "Epochs completed & 2", "Train / val / test entries & 9 / 1 / 5",
                     "norm\\_\\_synthetic\\_tpl"):
        assert expected in tex, expected


def test_training_fragment_without_trainer_state_has_no_loss_rows(tmp_path):
    _write_synthetic_training_run(tmp_path, with_state=False)
    tex = fragments(load_artifacts(tmp_path))["table_training"]
    assert "Epochs (configured) & 3" in tex
    assert "loss" not in tex.lower() and "wall-clock" not in tex and "memory" not in tex


def test_no_training_artifacts_no_training_fragment(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path, monkeypatch, "finetuned", BASE_OUTCOMES, adapter="a")
    art = load_artifacts(tmp_path)
    assert "table_training" not in fragments(art)
    assert any("Training loss is not in the artifacts" in line for line in limitations(art))


# --------------------------------------------------------------------------- matplotlib missing


def test_report_works_without_matplotlib(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # makes `import matplotlib` raise ImportError
    outputs = tmp_path / "outputs"
    _write_synthetic_eval(outputs, monkeypatch, "baseline", BASE_OUTCOMES)
    made, skipped = make_figures(load_artifacts(outputs), tmp_path / "unused")
    assert made == [] and len(skipped) == 3
    assert all("evaluation/report/requirements.txt" in why for _, why in skipped)
    build_report(outputs, tmp_path / "report")
    md = (tmp_path / "report" / "results.md").read_text()
    assert "correctness_by_category: not drawn (matplotlib is not installed" in md
    assert not (tmp_path / "report" / "figures").exists()
    assert (tmp_path / "report" / "table_categories.tex").exists()  # tables still written


# --------------------------------------------------------------------------- session outputs & new eval keys
# Every fixture below is SYNTHETIC: made-up reports in the format the other sessions
# are moving to, written directly (all keys optional) to exercise the report.

from evaluation.report.render import (  # noqa: E402
    coverage_table,
    coverage_warnings,
    search_table,
    session_stage_table,
    summary_json,
    token_limit_table,
    training_table,
)


def _row_dict(i: int, category: str, correct: bool, *, speedup=1.5, hit=None, prefix="s") -> dict:
    v = {"status": "ok", "correct": correct, "speedup": speedup if correct else None, "interpreted": False}
    row = {"id": f"{category}__synthetic_tpl__{prefix}{i}", "category": category, "verification": v,
           "outcome": "correct" if correct else "incorrect", "generated_text": "SYNTHETIC",
           "generated_explanation": "SYNTHETIC"}
    if hit is not None:
        row["hit_token_limit"] = hit
    return row


def _write_report(root: Path, name: str, rows: list[dict], **top) -> Path:
    path = root / "evaluation" / "results" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"run_name": name.split("_2")[0], "model_name": "SYNTHETIC-model",
              "generated_at": top.pop("generated_at", "2099-01-01T00:00:00"), "results": rows, **top}
    path.write_text(json.dumps(report))
    return path


def test_stage_status_table_from_stage_status_json(tmp_path):
    (tmp_path / "stage_status.json").write_text(json.dumps([  # SYNTHETIC session
        {"stage": "env", "status": "ok", "seconds": 1.2, "error": None},
        {"stage": "dataset", "status": "failed", "seconds": 3600.0, "error": "StageFailed: killed after 3600s"},
        {"stage": "finetune", "status": "skipped", "seconds": 0.0, "error": "time budget: 10s left"},
        {"stage": "baseline", "status": "ok", "seconds": 900.0, "error": None,
         "prompts_evaluated": 25, "held_out_total": 100}]))
    art = load_artifacts(tmp_path)
    rows = {r[0]: r for r in session_stage_table(art)}
    assert rows["dataset"][1:] == ["FAILED", "3600", "StageFailed: killed after 3600s"]
    assert rows["finetune"][1] == "SKIPPED" and "time budget" in rows["finetune"][3]
    assert rows["baseline"][3] == "prompts scored 25 of 100"
    out = tmp_path / "out"
    build_report(tmp_path, out, figures=False)
    md = (out / "results.md").read_text()
    assert md.index("## Session stages") < md.index("## Category results")


def test_stage_status_falls_back_to_session_json(tmp_path):
    (tmp_path / "session.json").write_text(json.dumps({"stages": {"dataset": {"status": "ok", "seconds": 5}}}))
    assert session_stage_table(load_artifacts(tmp_path))[1][:3] == ["dataset", "ok", "5"]


def test_coverage_warns_loudly_when_not_full_set(tmp_path):
    rows = [_row_dict(i, "norm", i % 2 == 0) for i in range(5)]  # SYNTHETIC: 5 of 100 held-out
    _write_report(tmp_path, "baseline_20990101T000000Z", rows, prompt_mode="zero_shot")
    (tmp_path / "stage_status.json").write_text(json.dumps([
        {"stage": "baseline", "status": "ok", "seconds": 1, "error": None,
         "eval_report": "baseline_20990101T000000Z.json", "prompts_evaluated": 5, "held_out_total": 100,
         "full_held_out_set": False}]))
    art = load_artifacts(tmp_path)
    assert coverage_table(art)[1][1:4] == ["5", "100", "NO"]
    assert any("NOT the full held-out set" in w for w in coverage_warnings(art))
    out = tmp_path / "out"
    build_report(tmp_path, out, figures=False)
    md = (out / "results.md").read_text()
    assert md.index("WARNING: these results are incomplete") < md.index("## Session stages")
    assert "Warning: incomplete results" in (out / "readme_results_snippet.md").read_text()


def test_partial_report_is_flagged(tmp_path):
    _write_report(tmp_path, "finetuned_x", [_row_dict(0, "rope", True)], adapter_path="a", partial=True)
    art = load_artifacts(tmp_path)
    assert art.run("ft").partial
    assert any("marked partial" in w for w in coverage_warnings(art))


def test_incremental_log_without_final_report_is_a_partial_run(tmp_path):
    log = tmp_path / "evaluation" / "results" / "finetuned_20990101T000000Z.jsonl"
    log.parent.mkdir(parents=True)
    lines = [{"run_name": "finetuned", "adapter_path": "a", "prompt_mode": "zero_shot"},  # metadata line
             _row_dict(0, "norm", True), _row_dict(1, "norm", False)]
    log.write_text("\n".join(json.dumps(x) for x in lines) + "\n{truncated")  # killed mid-write
    art = load_artifacts(tmp_path)
    run = art.run("ft", 1, 0, "zero_shot")
    assert run.incremental and run.partial and (run.overall().n, run.overall().correct) == (2, 1)
    assert any("killed before writing its report" in w for w in coverage_warnings(art))
    assert any("unreadable line" in p for p in art.problems)


def test_incremental_log_is_ignored_when_final_report_exists(tmp_path):
    _write_report(tmp_path, "baseline_20990101T000000Z", [_row_dict(i, "norm", True) for i in range(3)])
    log = tmp_path / "evaluation" / "results" / "baseline_20990101T000000Z.jsonl"
    log.write_text(json.dumps(_row_dict(0, "norm", False)) + "\n")
    art = load_artifacts(tmp_path)
    assert art.run("base").overall().n == 3 and not art.run("base").incremental
    assert str(log.relative_to(tmp_path)) in art.superseded


def test_dryrun_files_are_skipped(tmp_path):
    _write_report(tmp_path, "dryrun_baseline_x", [_row_dict(0, "norm", True)])
    art = load_artifacts(tmp_path)
    assert art.runs == {} and "dryrun" in art.excluded[0][1]


def test_zero_shot_and_few_shot_are_never_pooled(tmp_path):
    _write_report(tmp_path, "baseline_zero", [_row_dict(i, "norm", i < 1) for i in range(4)],
                  prompt_mode="zero_shot")  # SYNTHETIC 1/4
    _write_report(tmp_path, "baseline_few", [_row_dict(i, "norm", i < 3) for i in range(4)],
                  prompt_mode="few_shot")  # SYNTHETIC 3/4
    art = load_artifacts(tmp_path)
    assert art.run("base") is None  # ambiguous: never silently pick one
    table = phase5_table(art)
    header, norm = table[0], {r[0]: r for r in table}["Norm"]
    zero, few = header.index("Base zero-shot % correct"), header.index("Base few-shot % correct")
    assert norm[zero].startswith("25% (1/4)") and norm[few].startswith("75% (3/4)")
    assert not any("(4/8)" in cell for row in table for cell in row)  # no pooled 4/8 anywhere
    assert any("never averaged together" in line for line in limitations(art))


def test_token_limit_truncations_and_share_of_failures(tmp_path):
    rows = [_row_dict(0, "norm", True, hit=True), _row_dict(1, "norm", False, hit=True),
            _row_dict(2, "norm", False, hit=False), _row_dict(3, "norm", False, hit=False)]  # SYNTHETIC
    _write_report(tmp_path, "baseline_x", rows, max_new_tokens=2048)
    art = load_artifacts(tmp_path)
    row = token_limit_table(art)[1]
    assert row[1:5] == ["2048", "4", "2", "3"]
    assert row[5].startswith("1: 33% (1/3)")  # 1 of 3 failures was cut off
    assert any("2 of 4 outputs were cut off at max_new_tokens=2048" in line for line in limitations(art))


def test_token_limit_unknown_is_not_zero(tmp_path):
    _write_report(tmp_path, "baseline_x", [_row_dict(0, "norm", False)])  # no hit_token_limit anywhere
    assert token_limit_table(load_artifacts(tmp_path)) == []
    _write_report(tmp_path, "baseline_x", [_row_dict(0, "norm", False)], hit_token_limit=1)  # count only
    assert token_limit_table(load_artifacts(tmp_path))[1][5] == "not recorded per output"


def test_search_compared_with_pass1_on_the_same_prompts(tmp_path):
    # SYNTHETIC: pass@1 on 10 prompts (first 4 all wrong, last 6 all right = 60% overall);
    # search on the first 4 only. The fair pass@1 is 0/4, not 6/10.
    pass1 = [_row_dict(i, "norm", i >= 4) for i in range(10)]
    search = [_row_dict(i, "norm", i < 3) for i in range(4)]
    _write_report(tmp_path, "baseline_p1", pass1, prompt_mode="zero_shot")
    _write_report(tmp_path, "baseline_s", search, prompt_mode="zero_shot",
                  search={"num_candidates": 4, "repair_rounds": 1, "search_limit": 4, "prompts_scored": 4})
    art = load_artifacts(tmp_path)
    rows = search_table(art)
    all_col = rows[0].index("All")
    assert rows[1][1] == "same 4 (search_limit 4)" and rows[1][2] == "pass@1, same prompts"
    assert rows[1][all_col].startswith("0% (0/4)")
    assert rows[2][2] == "best-of-4 + repair" and rows[2][all_col].startswith("75% (3/4)")
    assert not any("(6/10)" in cell for row in rows for cell in row)
    text = " ".join(limitations(art))
    assert "never with pass@1 on all prompts" in text
    summary = json.loads(json.dumps(summary_json(art)))
    comparison = summary["search_comparisons"][0]
    assert comparison["pass1_same_prompts"]["n"] == 4 and comparison["search_runs"]["best-of-4 + repair"]["k"] == 3


def test_search_prompts_missing_from_pass1_are_reported(tmp_path):
    _write_report(tmp_path, "baseline_p1", [_row_dict(i, "norm", True) for i in range(2)])  # SYNTHETIC
    _write_report(tmp_path, "baseline_s", [_row_dict(i, "norm", True) for i in range(3)],
                  search={"num_candidates": 4, "repair_rounds": 0, "search_limit": 3})
    art = load_artifacts(tmp_path)
    rows = search_table(art)
    assert rows[1][2] == "pass@1, same prompts (2 of 3 scored)"
    # The like-for-like row: the search run restricted to the 2 prompts pass@1 also scored.
    assert rows[3][1] == "shared 2" and rows[3][2] == "best-of-4, prompts pass@1 also scored"
    assert rows[3][rows[0].index("All")].startswith("100% (2/2)")
    assert any("pass@1 scored only 2 of the 3 search prompts" in w for w in coverage_warnings(art))


def test_training_metrics_memory_and_wall_clock(tmp_path):
    run = tmp_path / "training" / "checkpoints" / "run_20990101T000000Z"
    run.mkdir(parents=True)
    (run / "training_metrics.json").write_text(json.dumps(  # SYNTHETIC
        {"wall_clock_s": 5400, "peak_memory_bytes": 9.5e9, "steps": 66}))
    (run / "train_log.json").write_text(json.dumps([{"loss": 1.2, "step": 10}, {"train_loss": 0.8, "step": 66}]))
    art = load_artifacts(tmp_path)
    rows = dict((r[0], r[1]) for r in training_table(art)[1:])
    assert rows["Training wall-clock"] == "1.50 h" and rows["Peak GPU memory"] == "9.5 GB"
    assert rows["Optimizer steps"] == "66" and rows["Final train loss"] == "0.8000"
    assert "Peak GPU memory & 9.5 GB" in fragments(art)["table_training"]


def test_partial_dataset_build_and_orphan_progress_log(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "generation_report.json").write_text(json.dumps({  # SYNTHETIC
        "device": "cuda", "partial": True, "run_stats": {"resumed": 3, "not_run": ["a", "b"]},
        "per_category": {"norm": {"verified": 1, "rejection_reasons": {}}}, "per_template": {}}))
    (raw / "progress_interpret.jsonl").write_text(json.dumps({"id": "x", "result": {"status": "ok", "correct": True}}))
    warnings = coverage_warnings(load_artifacts(tmp_path))
    assert "GPU dataset build is PARTIAL: 2 candidates were not verified." in warnings
    assert any("Dataset build (interpret) did not finish" in w and "1 verifications (1 passed)" in w
               for w in warnings)


def test_wrong_types_never_crash_or_invent_numbers(tmp_path):
    _write_report(tmp_path, "baseline_x", [{"id": "a", "category": "norm", "verification": "garbage",
                                            "hit_token_limit": "yes"}, "not a row"],
                  outcomes="nope", search="nope", max_new_tokens="lots", partial=None, prompt_mode=None)
    out = tmp_path / "out"
    build_report(tmp_path, out, figures=False)
    run = load_artifacts(tmp_path).run("base")
    assert (run.overall().n, run.overall().correct) == (1, 0) and run.max_new_tokens is None
    assert run.token_limit() is None and run.key == ("base", 1, 0, "unspecified")
