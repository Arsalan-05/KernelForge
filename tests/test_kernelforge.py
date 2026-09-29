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
