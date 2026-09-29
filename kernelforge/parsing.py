"""Parse model-generated text into kernel code and explanation."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedOutput:
    triton_kernel: str | None
    optimization_explanation: str | None
    raw_text: str

    @property
    def has_kernel(self) -> bool:
        return bool(self.triton_kernel and self.triton_kernel.strip())

    @property
    def has_explanation(self) -> bool:
        return bool(self.optimization_explanation and self.optimization_explanation.strip())


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def parse_model_output(text: str) -> ParsedOutput:
    """Extract kernel and explanation from model output.

    Accepts the structured format (kernel:/explanation: sections) or falls back
    to heuristics (fenced code block + trailing prose).
    """
    raw = text.strip()
    if not raw:
        return ParsedOutput(None, None, raw)

    # Section headers may carry markdown decoration from general-purpose
    # models: "### kernel:", "**Explanation:**".
    header = r"(?:^|\n)[#>*_ \t]*{name}[*_ \t]*:[*_ \t]*\n"
    kernel_match = re.search(
        header.format(name="kernel") + r"(.*?)(?=" + header.format(name="explanation") + r"|\Z)",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    explanation_match = re.search(
        header.format(name="explanation") + r"(.*)\Z",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if kernel_match:
        kernel = _strip_code_fences(kernel_match.group(1).strip())
        explanation = explanation_match.group(1).strip() if explanation_match else None
        return ParsedOutput(kernel or None, explanation or None, raw)

    # Fallback: first fenced Python block is the kernel; remainder is explanation.
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        kernel = fence_match.group(1).strip()
        remainder = raw[fence_match.end() :].strip()
        explanation = remainder or None
        return ParsedOutput(kernel, explanation, raw)

    # Last resort: if it looks like Python with triton imports, treat whole thing as kernel.
    if "triton" in raw.lower() and "modelnew" in raw.lower():
        return ParsedOutput(raw, None, raw)

    return ParsedOutput(None, raw if raw else None, raw)
