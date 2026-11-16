"""Pre-GPU checks for the training + evaluation pipeline.

Everything here runs on CPU without Triton. Entries come from the
*unverified* template candidates (data/templates), which have the same ids,
templates and text as the GPU-verified dataset will; nothing here is a result.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from kernelforge.dataset import load_eval_entries, load_template_candidates, make_split, template_of
from kernelforge.model import render_prompt
from kernelforge.parsing import parse_model_output
from kernelforge.prompts import build_messages, format_response
from training.finetune import IGNORE_INDEX, build_sft_dataset, build_sft_features

ROOT = Path(__file__).parent.parent
FINETUNE_CONFIG = json.loads((ROOT / "training/configs/finetune_default.json").read_text())


@pytest.fixture(scope="module")
def candidates():
    return load_template_candidates()


# --- split -----------------------------------------------------------------

def test_template_split_counts_match_docs(candidates):
    """MASTER_PLAN: 300 entries -> 180 train / 20 val / 100 test, one held-out template per category."""
    train, val, test = make_split(candidates, "template")
    assert (len(train), len(val), len(test)) == (180, 20, 100)
    held_out = Counter(template_of(e) for e in test)
    assert len(held_out) == 5 and set(held_out.values()) == {20}
    assert Counter(e["category"] for e in test) == {c: 20 for c in Counter(e["category"] for e in candidates)}


def test_template_split_has_no_leakage(candidates):
    train, val, test = make_split(candidates, "template")
    test_templates = {template_of(e) for e in test}
    assert not test_templates & {template_of(e) for e in train + val}
    assert not {e["id"] for e in test} & {e["id"] for e in train + val}
    # Same kernel source must never appear on both sides (shape constants aside, a
    # template's kernel is fixed, so this is what a shape-level split would leak).
    assert not {e["triton_kernel"] for e in test} & {e["triton_kernel"] for e in train + val}


def test_finetune_and_eval_scripts_select_the_same_test_set(candidates):
    """finetune.py splits with its config; the eval scripts call make_split with defaults."""
    ratios = {"val_ratio": FINETUNE_CONFIG.get("val_ratio", 0.10)}
    from_finetune = make_split(candidates, FINETUNE_CONFIG.get("split", "template"),
                               seed=FINETUNE_CONFIG.get("seed", 42), **ratios)[2]
    from_eval = make_split(candidates, "template")[2]
    assert [e["id"] for e in from_finetune] == [e["id"] for e in from_eval]


def test_real_runs_never_fall_back_to_schema_examples(tmp_path):
    empty = tmp_path / "dataset.jsonl"
    empty.write_text("")
    assert load_eval_entries(empty, dry_run=False) == ([], "none")
    entries, source = load_eval_entries(empty, dry_run=True)
    assert len(entries) == 300 and "UNVERIFIED" in source


# --- prompt / target round trip ----------------------------------------------

def test_training_target_round_trips_through_the_parser(candidates):
    """What the model is trained to emit must parse back to exactly the kernel and explanation."""
    for entry in candidates:
        parsed = parse_model_output(format_response(entry))
        assert parsed.triton_kernel == entry["triton_kernel"].strip(), entry["id"]
        assert parsed.optimization_explanation == entry["optimization_explanation"].strip(), entry["id"]


def test_finetune_config_matches_documented_hyperparameters():
    c = FINETUNE_CONFIG
    assert (c["lora_r"], c["lora_alpha"], c["learning_rate"], c["num_epochs"]) == (16, 32, 2e-4, 3)
    assert (c["per_device_train_batch_size"], c["gradient_accumulation_steps"]) == (1, 8)


# --- loss masking ------------------------------------------------------------

class FakeChatTokenizer:
    """Character-level stand-in for an HF tokenizer with a ChatML chat template.

    `add_special_tokens=True` prepends a BOS id, so tests can catch a double BOS.
    """

    chat_template = "chatml"
    BOS = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(ch) for ch in text]
        return {"input_ids": ([self.BOS] if add_special_tokens else []) + ids}


def _decode(ids):
    return "".join(chr(i) for i in ids)


def test_loss_mask_covers_only_the_assistant_answer(candidates):
    tok = FakeChatTokenizer()
    entry = candidates[0]
    feat = build_sft_features(entry, tok, max_seq_length=100_000)
    labels, ids = feat["labels"], feat["input_ids"]
    n_prompt = labels.index(next(t for t in labels if t != IGNORE_INDEX))

    # The masked part is exactly the inference prompt, with no extra BOS.
    prompt = render_prompt(tok, build_messages(entry["description"], entry["pytorch_reference"]))
    assert _decode(ids[:n_prompt]) == prompt
    assert all(t == IGNORE_INDEX for t in labels[:n_prompt])
    # Everything after it is supervised: the two-section answer plus the end-of-turn marker.
    assert labels[n_prompt:] == ids[n_prompt:]
    assert _decode(labels[n_prompt:]) == format_response(entry) + "<|im_end|>\n"
    assert len(ids) == len(labels) == len(feat["attention_mask"])


def test_overlong_examples_are_dropped_not_truncated(candidates):
    tok = FakeChatTokenizer()
    lengths = [len(build_sft_features(e, tok, 100_000)["input_ids"]) for e in candidates[:4]]
    cap = sorted(lengths)[1]  # two of the four fit
    feats, dropped = build_sft_dataset(candidates[:4], tok, cap)
    assert len(feats) + len(dropped) == 4 and len(dropped) == sum(n > cap for n in lengths)
    assert all(len(f["input_ids"]) <= cap for f in feats)


def test_tokenizer_without_chat_template_is_rejected(candidates):
    class NoTemplate(FakeChatTokenizer):
        chat_template = None

    with pytest.raises(ValueError, match="chat template"):
        build_sft_features(candidates[0], NoTemplate(), 100_000)


def _cached_qwen_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(FINETUNE_CONFIG["model_name"], local_files_only=True)
    except Exception:
        return None


def test_loss_mask_with_real_qwen_tokenizer(candidates):
    """Same checks with the real chat template, if the tokenizer is in the local HF cache."""
    tok = _cached_qwen_tokenizer()
    if tok is None:
        pytest.skip("Qwen tokenizer not in local HF cache (no network in tests)")
    end_of_turn = tok.eos_token  # <|im_end|> for Qwen2.5-Instruct
    longest = 0
    for entry in candidates[::10]:
        feat = build_sft_features(entry, tok, FINETUNE_CONFIG["max_seq_length"])
        assert feat is not None, f"{entry['id']} exceeds max_seq_length"
        answer = [t for t in feat["labels"] if t != IGNORE_INDEX]
        assert tok.decode(answer).startswith(format_response(entry) + end_of_turn)
        longest = max(longest, len(feat["input_ids"]))
    assert longest < FINETUNE_CONFIG["max_seq_length"]


# --- evaluation plumbing -----------------------------------------------------

def test_eval_outcomes_separate_parse_failures_from_unverified(candidates):
    from evaluation.common import evaluate_generated_kernel

    entry = candidates[0]
    ok = evaluate_generated_kernel(entry, format_response(entry), skip_verification=True)
    assert ok.outcome == "not_verified" and ok.has_explanation
    junk = evaluate_generated_kernel(entry, "Sorry, I can't help with that.", skip_verification=True)
    assert junk.outcome == "no_kernel"
    assert not junk.has_explanation  # raw text without a kernel is not an explanation


def test_report_keeps_full_explanation(candidates):
    from evaluation.common import evaluate_generated_kernel, result_row
    from evaluation.explanation_review import _generated_explanation

    entry = max(candidates, key=lambda e: len(e["triton_kernel"]))
    row = result_row(evaluate_generated_kernel(entry, format_response(entry), skip_verification=True))
    row = json.loads(json.dumps(row))  # as stored in the JSONL / report
    assert row["generated_explanation"] == entry["optimization_explanation"].strip()
    assert entry["optimization_explanation"].strip() not in row["generated_text_preview"]
    assert _generated_explanation(row) == entry["optimization_explanation"].strip()


def test_baseline_comparison_skips_dry_runs_and_other_test_sets(tmp_path):
    from evaluation.finetuned_eval import load_baseline_report

    def write(name, ids, dry):
        rows = [{"id": i} for i in ids]
        (tmp_path / name).write_text(json.dumps({"dry_run": dry, "results": rows, "per_category": {"tag": name}}))

    write("baseline_20260101T000000Z.json", ["a", "b"], False)
    write("baseline_20260102T000000Z.json", ["a", "c"], False)  # newer, different entries
    write("baseline_20260103T000000Z.json", ["a", "b"], True)   # newest, dry run
    per_category, name = load_baseline_report(tmp_path, ["b", "a"])
    assert name == "baseline_20260101T000000Z.json" and per_category == {"tag": name}
    assert load_baseline_report(tmp_path, ["z"]) == (None, None)


# --- peak_memory_gb ----------------------------------------------------------

def test_peak_memory_gb_is_none_without_gpu(tmp_path, monkeypatch, candidates):
    """When CUDA is not available, peak_memory_gb must be None in split_info.json.

    Stubs out load_eval_entries and train() so the full model is never loaded;
    only the split_info update logic in main() is exercised.
    """
    import sys
    import training.finetune as finetune

    # Stub train() to return what it would on CPU (no GPU memory available).
    fake_stats = {
        "partial": False, "stopped_at_step": None, "steps_planned": 10,
        "backend": "transformers", "dropped_too_long": [], "supervised_token_fraction": 0.5,
        "longest_example_tokens": 100, "peak_memory_gb": None,
    }
    monkeypatch.setattr(finetune, "train", lambda *a, **kw: fake_stats)

    # Stub load_eval_entries so no GPU dataset file is needed.
    monkeypatch.setattr(finetune, "load_eval_entries",
                        lambda *a, **kw: (candidates, "unverified template candidates (test stub)"))

    # Stub build_tiny_random_model so transformers is not needed; return a dummy path string.
    monkeypatch.setattr(finetune, "build_tiny_random_model",
                        lambda *a, **kw: run / "dummy_model")

    run = tmp_path / "test_peak_mem"
    monkeypatch.setattr(sys, "argv", ["finetune.py", "--dry-run", "--output-dir", str(run)])
    finetune.main()
    info = json.loads((run / "split_info.json").read_text())
    assert "peak_memory_gb" in info, "split_info.json must contain peak_memory_gb key"
    assert info["peak_memory_gb"] is None, "peak_memory_gb must be None when no GPU is present"
