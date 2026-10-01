"""Tests for prompt formatting, output parsing, and dataset utilities."""

import json
from pathlib import Path

import pytest

from kernelforge.dataset import load_example_entries, split_dataset
from kernelforge.parsing import parse_model_output
from kernelforge.prompts import format_instruction, format_training_example

EXAMPLES_DIR = Path(__file__).parent.parent / "data" / "examples"


def test_load_example_entries():
    entries = load_example_entries()
    assert len(entries) >= 3
    assert all("pytorch_reference" in e for e in entries)


def test_split_dataset_stratified():
    entries = load_example_entries() * 20  # repeat for enough samples
    train, val, test = split_dataset(entries, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05)
    assert len(train) + len(val) + len(test) == len(entries)
    assert len(test) >= 1


def test_format_instruction_contains_reference():
    entry = load_example_entries()[0]
    text = format_instruction(entry["description"], entry["pytorch_reference"])
    assert entry["description"] in text
    assert "kernel:" in text.lower() or "explanation:" in text.lower()


def test_format_training_example_roundtrip_fields():
    entry = load_example_entries()[0]
    example = format_training_example(entry)
    assert example["id"] == entry["id"]
    assert entry["triton_kernel"].strip() in example["response"]
    assert entry["optimization_explanation"].strip() in example["response"]


def test_parse_structured_output():
    text = "kernel:\nimport triton\n\nclass ModelNew: pass\n\nexplanation:\nFused ops to reduce memory traffic."
    parsed = parse_model_output(text)
    assert parsed.has_kernel
    assert parsed.has_explanation
    assert "ModelNew" in parsed.triton_kernel
    assert "memory traffic" in parsed.optimization_explanation


@pytest.mark.parametrize("kernel_header, explanation_header", [
    ("### kernel:", "### explanation:"),
    ("**Kernel:**", "**Explanation:**"),
])
def test_parse_markdown_decorated_headers(kernel_header, explanation_header):
    text = f"{kernel_header}\n```python\nimport triton\nclass ModelNew: pass\n```\n\n{explanation_header}\nTiles the K loop."
    parsed = parse_model_output(text)
    assert parsed.triton_kernel == "import triton\nclass ModelNew: pass"
    assert parsed.optimization_explanation == "Tiles the K loop."


def test_parse_fenced_code_block():
    text = "```python\nimport triton\nclass ModelNew: pass\n```\n\nFused the norm and residual."
    parsed = parse_model_output(text)
    assert parsed.has_kernel
    assert "Fused" in parsed.optimization_explanation


def test_parse_empty_output():
    parsed = parse_model_output("")
    assert not parsed.has_kernel
    assert not parsed.has_explanation


def _grid_entries():
    from data.templates import GENERATORS

    return [e for make in GENERATORS.values() for e in make()]


def test_template_split_holds_out_whole_ops():
    from kernelforge.dataset import split_by_template, template_of

    entries = _grid_entries()
    train, val, test = split_by_template(entries, seed=42)
    assert len(train) + len(val) + len(test) == len(entries)
    seen = {template_of(e) for e in train + val}
    held_out = {template_of(e) for e in test}
    assert not seen & held_out
    assert len(held_out) == 5 and {e["category"] for e in test} == {e["category"] for e in entries}
    assert split_by_template(entries, seed=42)[2] == test  # deterministic


def test_interleave_by_category_samples_every_category_first():
    from kernelforge.dataset import interleave_by_category

    entries = _grid_entries()
    first = interleave_by_category(entries)[:5]
    assert len({e["category"] for e in first}) == 5


def test_dataset_report_separates_oom_from_broken_kernels(tmp_path, monkeypatch, capsys):
    from argparse import Namespace

    from data import build_dataset
    from verification.verify_kernel import VerifyResult

    for name in ("VERIFIED_PATH", "REJECTED_PATH", "REPORT_PATH"):
        monkeypatch.setattr(build_dataset, name, tmp_path / f"{name}.json")
    entries = [e for e in _grid_entries() if e["category"] == "norm"][:3]
    results = [
        (entries[0], VerifyResult(status="ok", correct=True, speedup=2.0)),
        (entries[1], VerifyResult(status="error", error="torch.OutOfMemoryError: CUDA out of memory")),
        (entries[2], VerifyResult(status="ok", correct=False, error="max abs error 3.1")),
    ]
    build_dataset._write_dataset(results, Namespace(device="cuda", smoke=False))
    report = json.loads((tmp_path / "REPORT_PATH.json").read_text())
    assert report["per_category"]["norm"]["rejection_reasons"] == {"oom": 1, "incorrect": 1}
    template = report["per_template"]["norm__rmsnorm"]
    assert template["verified"] == 1 and template["median_speedup"] == 2.0
    assert "(oom)" in capsys.readouterr().out
