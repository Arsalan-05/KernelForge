"""Eval/training run mechanics: incremental results, resume, time limits, search,
few-shot, greedy decoding, and the blind explanation review.

CPU only. The "model" is a fake generator and the dataset is the unverified
template candidates written to a temp file and marked verified for the test;
no number produced here is a result.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluation.common as common
from kernelforge.dataset import load_template_candidates, make_split
from kernelforge.prompts import format_response

ROOT = Path(__file__).parent.parent
MODEL_CONFIG = ROOT / "training/configs/qwen2.5-coder-1.5b.json"


class FakeGenerator:
    """Stands in for KernelGenerator. pass@1 returns the reference answer (so it
    parses); search samples are seed-dependent junk (so they don't), which makes
    repair rounds run and lets tests check that sampling is seeded."""

    backend = "fake"

    def __init__(self, entries, crash_after=None):
        self.by_reference = {e["pytorch_reference"]: e for e in entries}
        self.calls = []
        self.crash_after = crash_after
        self.last_hit_token_limit = False

    def generate(self, description, pytorch_reference, temperature=None, example=None):
        if self.crash_after is not None and len(self.calls) >= self.crash_after:
            raise RuntimeError("simulated kill")
        self.calls.append({"temperature": temperature, "example": example["id"] if example else None,
                           "id": self.by_reference[pytorch_reference]["id"]})
        return format_response(self.by_reference[pytorch_reference])

    def stream_chat(self, messages, num_sequences=1, temperature=None):
        import torch

        for i in range(num_sequences):
            yield i, f"sample {torch.randint(0, 10**9, (1,)).item()} at T={temperature}"

    def count_tokens(self, text):
        return len(text.split())


@pytest.fixture(scope="module")
def candidates():
    return load_template_candidates()


@pytest.fixture
def env(tmp_path, monkeypatch, candidates):
    """Synthetic 'verified' dataset in tmp, results dir in tmp, fake generator."""
    dataset = tmp_path / "dataset.jsonl"
    with open(dataset, "w") as f:
        for e in candidates:
            f.write(json.dumps({**e, "verification": {"verified": True}}) + "\n")
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path / "results")
    made = []

    def factory(config, crash_after=None):
        gen = FakeGenerator(candidates, crash_after)
        made.append(gen)
        return gen

    import evaluation.baseline_eval as baseline_eval

    monkeypatch.setattr(baseline_eval, "KernelGenerator", factory)
    base_args = ["--dataset", str(dataset), "--config", str(MODEL_CONFIG), "--device", "cpu", "--skip-verification"]
    return SimpleNamespace(tmp=tmp_path, base_args=base_args, made=made, factory=factory, monkeypatch=monkeypatch,
                           module=baseline_eval)


def _run(env, *extra):
    path = env.module.main(env.base_args + list(extra))
    return json.loads(Path(path).read_text())


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# --- pass@1 --------------------------------------------------------------------

def test_pass_at_1_is_greedy_on_all_held_out_prompts(env):
    report = _run(env)
    gen = env.made[-1]
    assert len(gen.calls) == 100 and {c["temperature"] for c in gen.calls} == {0.0}
    assert report["prompts_scored"] == report["prompts_planned"] == 100 and not report["partial"]
    assert report["decoding"] == "greedy" and report["max_new_tokens"] == 2048
    assert report["prompt_mode"] == "zero_shot" and {c["example"] for c in gen.calls} == {None}
    assert "search" not in report


def test_killed_run_resumes_without_redoing_prompts(env):
    env.monkeypatch.setattr(env.module, "KernelGenerator", lambda cfg: env.factory(cfg, crash_after=7))
    with pytest.raises(RuntimeError, match="simulated kill"):
        _run(env)
    log = env.tmp / "results" / "baseline.jsonl"
    assert sum(r["type"] == "row" for r in _rows(log)) == 7  # finished prompts survived the kill

    env.monkeypatch.setattr(env.module, "KernelGenerator", env.factory)
    report = _run(env, "--resume")
    assert len(env.made[-1].calls) == 93
    assert not report["partial"] and len({r["id"] for r in report["results"]}) == 100


def test_resume_refuses_different_settings(env):
    _run(env, "--limit", "3")
    with pytest.raises(SystemExit, match="settings differ"):
        _run(env, "--limit", "3", "--resume", "--max-new-tokens", "1024")


def test_rerun_without_resume_keeps_the_old_log(env):
    _run(env, "--limit", "2")
    _run(env, "--limit", "2")
    logs = sorted(p.name for p in (env.tmp / "results").glob("baseline*.jsonl"))
    assert len(logs) == 2 and "baseline.jsonl" in logs


def test_max_minutes_writes_a_partial_report(env):
    class ExpiresAfter:  # deadline that passes before the 5th prompt
        def __init__(self, max_minutes):
            self.checks = 0

        def expired(self):
            self.checks += 1
            return self.checks >= 5

    env.monkeypatch.setattr(common, "Deadline", ExpiresAfter)
    report = _run(env, "--max-minutes", "1")
    assert report["partial"] and report["prompts_scored"] == 4 and report["prompts_planned"] == 100


def test_corrupt_last_line_is_ignored_on_resume(env):
    _run(env, "--limit", "3")
    log = env.tmp / "results" / "baseline.jsonl"
    log.write_text(log.read_text() + '{"type": "row", "pha')  # killed mid-write
    report = _run(env, "--limit", "4", "--resume")
    assert report["prompts_scored"] == 4 and len(env.made[-1].calls) == 1


# --- search ----------------------------------------------------------------------

def test_search_runs_on_a_stratified_subset_and_records_its_settings(env):
    report = _run(env, "--num-candidates", "3", "--repair-rounds", "1")
    search = report["search"]
    assert {k: search[k] for k in ("num_candidates", "repair_rounds", "search_limit", "prompts_scored")} == \
        {"num_candidates": 3, "repair_rounds": 1, "search_limit": 25, "prompts_scored": 25}
    assert report["prompts_scored"] == 100  # pass@1 still covers every held-out prompt
    per_category = {}
    for row in search["results"]:
        per_category[row["category"]] = per_category.get(row["category"], 0) + 1
    assert set(per_category.values()) == {5} and len(per_category) == 5
    # Junk samples are unparseable, so the repair round ran: 3 samples + 1 repair.
    assert {row["num_candidates"] for row in search["results"]} == {4}
    assert search["round0_temperature"] == 0.7 and search["seed"] == 1234


def test_search_sampling_is_seeded(env):
    def texts(*extra):
        report = _run(env, "--limit", "5", "--search-limit", "5", "--num-candidates", "2", *extra)
        return [row["generated_text"] for row in report["search"]["results"]]

    assert texts("--seed", "7") == texts("--seed", "7")
    assert texts("--seed", "7") != texts("--seed", "8")


# --- few-shot ----------------------------------------------------------------------

def test_few_shot_examples_come_from_the_train_split_only(env, candidates):
    report = _run(env, "--few-shot", "--num-candidates", "2", "--search-limit", "5")
    train, val, test = make_split(candidates, "template")
    train_ids, test_ids = {e["id"] for e in train}, {e["id"] for e in test}
    category = {e["id"]: e["category"] for e in candidates}
    rows = report["results"] + report["search"]["results"]
    assert report["prompt_mode"] == "few_shot" and report["run_name"] == "baseline_few_shot"
    for row in rows:
        assert row["example_id"] in train_ids and row["example_id"] not in test_ids
        assert category[row["example_id"]] == row["category"]


def test_finetuned_comparison_ignores_few_shot_baselines(tmp_path):
    from evaluation.finetuned_eval import load_baseline_report

    rows = [{"id": "a"}]
    (tmp_path / "baseline_few_shot_20260102T000000Z.json").write_text(
        json.dumps({"prompt_mode": "few_shot", "results": rows, "per_category": {"tag": "few"}}))
    (tmp_path / "baseline_20260101T000000Z.json").write_text(
        json.dumps({"prompt_mode": "zero_shot", "results": rows, "per_category": {"tag": "zero"}}))
    assert load_baseline_report(tmp_path, ["a"])[0] == {"tag": "zero"}


# --- decoding ----------------------------------------------------------------------

def test_temperature_zero_means_greedy_decoding():
    from kernelforge.model import GenerationConfig, KernelGenerator

    gen = KernelGenerator(GenerationConfig(max_new_tokens=2048))
    gen._tokenizer = SimpleNamespace(eos_token_id=0)
    greedy = gen._generate_kwargs(0.0)
    assert greedy["do_sample"] is False and "temperature" not in greedy and greedy["max_new_tokens"] == 2048
    sampled = gen._generate_kwargs(0.7)
    assert sampled["do_sample"] is True and sampled["temperature"] == 0.7


# --- training ----------------------------------------------------------------------

def test_time_limit_callback_saves_adapter_and_stops(tmp_path):
    from transformers import TrainerControl, TrainerState

    from training.finetune import make_time_limit_callback

    saved = []
    model = SimpleNamespace(save_pretrained=saved.append)
    state = TrainerState(global_step=3, max_steps=10)

    control = make_time_limit_callback(None, tmp_path).on_step_end(None, state, TrainerControl(), model=model)
    assert not control.should_training_stop and not saved

    callback = make_time_limit_callback(0.0, tmp_path / "adapter")  # deadline long past
    control = callback.on_step_end(None, state, TrainerControl(), model=model)
    assert control.should_training_stop and saved == [str(tmp_path / "adapter")]
    assert callback.stopped_at_step == 3


def test_finetune_dry_run_time_limit_writes_logs_into_run_folder(tmp_path, monkeypatch):
    """End to end on a tiny random model; needs the Qwen tokenizer in the local HF cache."""
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not in local HF cache (no network in tests)")
    import sys

    from training import finetune

    run = tmp_path / "dryrun_test"
    monkeypatch.setattr(sys, "argv", ["finetune.py", "--dry-run", "--max-minutes", "0", "--output-dir", str(run)])
    finetune.main()
    info = json.loads((run / "split_info.json").read_text())
    assert info["partial"] and info["stopped_at_step"] == 1
    assert (run / "adapter" / "adapter_config.json").exists()
    assert json.loads((run / "train_log.json").read_text())  # loss history in the run folder itself
    assert (run / "trainer_state.json").exists()


# --- blind explanation review --------------------------------------------------------

def _synthetic_report(path, run_name, prompt_mode, entries, text):
    rows = [{"id": e["id"], "category": e["category"], "generated_explanation": f"{text} {e['id']}"}
            for e in entries]
    path.write_text(json.dumps({"run_name": run_name, "prompt_mode": prompt_mode, "model_name": "m",
                                "dry_run": True, "results": rows}))
    return path


def test_blind_review_hides_models_and_summarize_joins_the_key(tmp_path, candidates):
    from evaluation import explanation_review as er

    test = make_split(candidates, "template")[2]
    a = _synthetic_report(tmp_path / "baseline_x.json", "baseline", "zero_shot", test, "BASE")
    b = _synthetic_report(tmp_path / "finetuned_y.json", "finetuned", "zero_shot", test, "FT")
    review_path = tmp_path / "review.json"
    er.main(["--report", str(a), str(b), "--output", str(review_path), "--blind", "--sample", "6"])

    review = json.loads(review_path.read_text())
    key = json.loads(er.key_path_for(review_path).read_text())
    assert review["blind"] and len(review["items"]) == 12
    assert all("source" not in item and "run_name" not in json.dumps(item) for item in review["items"])
    assert set(key) == {item["sample_id"] for item in review["items"]}
    assert sorted(s["run_name"] for s in key.values()) == ["baseline"] * 6 + ["finetuned"] * 6

    # Score: 2 for every fine-tuned sample, 0 for every baseline one (the scorer can't see
    # which is which; the test peeks at the synthetic text instead).
    answers = []
    for item in review["items"]:
        answers += ["2" if item["generated_explanation"].startswith("FT") else "0", ""]
    er.score_review(review_path, ask=lambda prompt: answers.pop(0))

    with pytest.raises(SystemExit):  # a blind review can't be summarized without its key
        er.main(["--summarize", str(review_path), "--key", str(tmp_path / "missing.json")])
    er.main(["--summarize", str(review_path)])
    summary = json.loads(review_path.read_text())["summary"]
    assert summary["scored"] == 12 and summary["unscored"] == 0
    assert summary["by_model"]["finetuned (zero_shot)"]["mean_score"] == 2
    assert summary["by_model"]["baseline (zero_shot)"]["mean_score"] == 0


def test_scoring_can_stop_and_resume(tmp_path, candidates):
    from evaluation import explanation_review as er

    test = make_split(candidates, "template")[2]
    a = _synthetic_report(tmp_path / "baseline_x.json", "baseline", "zero_shot", test, "BASE")
    review_path = tmp_path / "review.json"
    er.main(["--report", str(a), "--output", str(review_path), "--sample", "3"])
    answers = ["1", "note", "x", "q"]  # score one, an invalid key, then quit
    er.score_review(review_path, ask=lambda prompt: answers.pop(0))
    scores = [i["rubric_score"] for i in json.loads(review_path.read_text())["items"]]
    assert scores == [1, None, None]
    answers = ["2", "", "0", ""]
    er.score_review(review_path, ask=lambda prompt: answers.pop(0))
    assert [i["rubric_score"] for i in json.loads(review_path.read_text())["items"]] == [1, 2, 0]
