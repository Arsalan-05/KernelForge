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
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import data.build_dataset as build_dataset
import evaluation.common as common
from evaluation.report.__main__ import build_report, main
from evaluation.report.artifacts import CATEGORIES, load_artifacts, parse_eval_report
from evaluation.report.render import (
    NOT_RUN,
    latex_fragments,
    limitations,
    phase5_table,
    prop_cell,
    readme_snippet,
    stage_status,
)
from evaluation.report.stats import Proportion, mean, median, wilson_interval
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
    assert run.key == ("ft", 4, 1) and run.mode_label == "best-of-4 + repair"
    run, _ = parse_eval_report({**report, "run_name": "baseline_best4", "adapter_path": None}, "s.json")
    assert run.key == ("base", 4, 0)
    run, _ = parse_eval_report(report, "s.json")
    assert run.key == ("ft", 1, 0) and run.mode_label == "pass@1"


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
    assert status["Base model, pass@1"] == "absent (session: skipped (no verified dataset))"


def test_prop_cell_formats_and_flags_small_n():
    assert prop_cell(Proportion(5, 10)) == "50% (5/10) [24–76]†"
    assert prop_cell(Proportion(10, 20)) == "50% (10/20) [30–70]"
    assert prop_cell(Proportion(0, 0)) == "no samples"
    assert prop_cell(Proportion(5, 10), latex=True) == "50\\% (5/10) [24--76]$^\\dagger$"


def test_phase5_table_cells(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path, monkeypatch, "baseline", BASE_OUTCOMES)
    rows = {r[0]: r for r in phase5_table(load_artifacts(tmp_path))}
    assert rows["Norm"][1] == "50% (2/4) [15–85]†"
    assert rows["Norm"][2] == "2.00× (n=2)†"
    assert rows["RoPE"][2] == "– (0 correct)"  # the only correct rope kernel had speedup=inf
    assert rows["Attention"][1] == "no samples"
    assert rows["Norm"][3] == NOT_RUN and rows["Norm"][5] == NOT_RUN  # no FT runs
    assert rows["All"][1] == "50% (3/6) [19–81]†"


def test_interpreter_runs_report_no_speedup(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path, monkeypatch, "baseline", BASE_OUTCOMES, interpreted=True)
    art = load_artifacts(tmp_path)
    assert phase5_table(art)[4][2] == "n/a (interpreter)"  # Norm row, Base speedup
    assert any("Interpreter-only" in line for line in limitations(art))


def test_latex_fragments_are_escaped(tmp_path, monkeypatch):
    _write_synthetic_eval(tmp_path, monkeypatch, "baseline", BASE_OUTCOMES)
    tex = latex_fragments(load_artifacts(tmp_path))
    assert "Base \\% correct" in tex and "2.00$\\times$" in tex and "\\begin{tabular}{lrrrrr}" in tex
    body = tex.split("\n", 1)[1]  # first line is a TeX comment, which may contain a bare %
    assert "\\%" in body and " % " not in body


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
    assert {"results.md", "readme_results_snippet.md", "tables.tex", "summary.json"} <= names
    md = (out / "results.md").read_text()
    assert "Fine-tuned model, best-of-4 + repair | present" in md
    assert "Fine-tuned model, pass@1 | absent" in md
    summary = json.loads((out / "summary.json").read_text())
    assert summary["runs"]["base/pass@1"]["all"]["correct"] == 3
    assert math.isclose(summary["runs"]["base/pass@1"]["per_category"]["norm"]["avg_speedup"], 2.0)


def test_figures_are_drawn_when_matplotlib_present(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    outputs = tmp_path / "outputs"
    _write_synthetic_eval(outputs, monkeypatch, "baseline", BASE_OUTCOMES)
    out = tmp_path / "report"
    build_report(outputs, out)
    assert (out / "figures" / "correctness_by_category.png").stat().st_size > 0
    assert (out / "figures" / "speedup_distribution.png").stat().st_size > 0
    assert not (out / "figures" / "search_gain.png").exists()  # only one search mode ran
    assert "search_gain: not drawn (fewer than two search modes" in (out / "results.md").read_text()


def test_cli_rejects_missing_source(tmp_path, capsys):
    assert main([str(tmp_path / "nope.zip"), "--out", str(tmp_path / "o")]) == 2
    assert "does not exist" in capsys.readouterr().err
