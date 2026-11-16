"""Parsing edge cases seen from general-purpose and truncated model outputs.

All model outputs below are hand-written synthetic strings, not real generations.
"""

from kernelforge.parsing import parse_model_output

KERNEL = "import torch\nimport triton\n\nclass ModelNew(torch.nn.Module):\n    pass"


def test_prose_before_fence_in_kernel_section():
    text = f"kernel:\nHere is the optimized kernel:\n```python\n{KERNEL}\n```\n\nexplanation:\nFuses the ops."
    parsed = parse_model_output(text)
    assert parsed.triton_kernel == KERNEL
    assert parsed.optimization_explanation == "Fuses the ops."


def test_inline_explanation_label_after_fence():
    text = f"kernel:\n```python\n{KERNEL}\n```\n\n**Explanation:** Tiles K so the tile stays in SRAM."
    parsed = parse_model_output(text)
    assert parsed.triton_kernel == KERNEL
    assert parsed.optimization_explanation == "Tiles K so the tile stays in SRAM."


def test_missing_explanation_is_none_not_kernel_text():
    parsed = parse_model_output(f"kernel:\n```python\n{KERNEL}\n```")
    assert parsed.triton_kernel == KERNEL
    assert parsed.optimization_explanation is None
    assert not parsed.has_explanation


def test_unclosed_fence_from_truncated_generation():
    text = f"kernel:\n```python\n{KERNEL}\n    def forward(self, x):"
    parsed = parse_model_output(text)
    assert parsed.triton_kernel.startswith("import torch")
    assert "```" not in parsed.triton_kernel


def test_fallback_prefers_block_with_modelnew_over_echoed_reference():
    reference = "import torch\n\nclass Model(torch.nn.Module):\n    pass"
    text = (f"The reference is:\n```python\n{reference}\n```\nMy kernel:\n```python\n{KERNEL}\n```\n"
            "Explanation: one pass over memory.")
    parsed = parse_model_output(text)
    assert parsed.triton_kernel == KERNEL
    assert parsed.optimization_explanation == "one pass over memory."


def test_comment_mentioning_kernel_is_not_a_section_header():
    code = f"{KERNEL}\n# kernel: launch one program per row"
    parsed = parse_model_output(f"kernel:\n{code}\n\nexplanation:\nRow-parallel.")
    assert parsed.triton_kernel == code
    assert parsed.optimization_explanation == "Row-parallel."


def test_no_kernel_at_all_is_reported_as_missing():
    parsed = parse_model_output("I cannot write this kernel.")
    assert not parsed.has_kernel
