"""CPU-only tests for scripts/check_token_lengths.py. A fake tokenizer
(one token per character, plus a fixed header) stands in for Qwen's, so no
download is needed; the numbers here are synthetic, not real token counts."""

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_token_lengths.py"
_spec = importlib.util.spec_from_file_location("check_token_lengths", _PATH)
ctl = importlib.util.module_from_spec(_spec)
sys.modules["check_token_lengths"] = ctl
_spec.loader.exec_module(ctl)


class CharTokenizer:
    """Chat template = role headers + content; one token per character."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text) if tokenize else text


def test_target_tokens_are_the_assistant_turn_only():
    entry = {"id": "synthetic__tpl__0", "category": "synthetic", "description": "d",
             "pytorch_reference": "ref", "triton_kernel": "KERNEL", "optimization_explanation": "WHY"}
    target, total = ctl.target_token_count(entry, CharTokenizer())
    response = "kernel:\nKERNEL\n\nexplanation:\nWHY"
    assert target == len(response) + len("<end>")  # content + end-of-turn, header excluded
    assert total > target


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert ctl.percentile(values, 95) == 95
    assert ctl.percentile(values, 100) == 100
    assert ctl.percentile([7], 95) == 7


def test_summarize_counts_over_cap_and_pct():
    summary = ctl.summarize({"a": [100, 200], "b": [900, 1200]}, cap=1000)
    assert summary["per_category"]["b"] == {"n": 2, "max": 1200, "p95": 1200, "mean": 1050.0, "over_cap": 1}
    assert summary["overall"]["max"] == 1200 and summary["overall"]["over_cap"] == 1
    assert summary["overall"]["max_pct_of_cap"] == 120.0
