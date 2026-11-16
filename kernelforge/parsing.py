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


# A complete fenced block: ```python\n ... \n```
_FENCED_BLOCK = re.compile(r"```[\w+-]*[ \t]*\n(.*?)\s*```", flags=re.DOTALL)
# An opening fence line on its own, used when the closing fence never arrived.
_OPENING_FENCE = re.compile(r"```[\w+-]*[ \t]*\n")
# An inline label such as "Explanation:" or "**Explanation:**" at the start of prose.
_EXPLANATION_LABEL = re.compile(r"^[#>*_ \t]*explanation[*_ \t]*:[*_ \t]*", flags=re.IGNORECASE)


def _pick_fenced_block(text: str) -> re.Match | None:
    """The fenced block that defines ModelNew, else the first one.

    General-purpose models sometimes echo the reference in its own block
    before the kernel; that block has `class Model` but no `ModelNew`.
    """
    blocks = list(_FENCED_BLOCK.finditer(text))
    for block in blocks:
        if "ModelNew" in block.group(1):
            return block
    return blocks[0] if blocks else None


def _split_code_and_prose(text: str) -> tuple[str, str]:
    """Return (code, prose after the code) for a kernel section.

    Handles: a bare module, a fenced module, prose before the fence, prose
    after the fence, and an unclosed fence (generation cut off at
    max_new_tokens), in which case everything after the fence line is code.
    """
    text = text.strip()
    block = _pick_fenced_block(text)
    if block:
        return block.group(1).strip(), text[block.end():].strip()
    opening = _OPENING_FENCE.search(text)
    if opening:
        return text[opening.end():].strip(), ""
    # No opening fence; drop a stray closing fence at the end, if any.
    return re.sub(r"\n?[ \t]*```[ \t]*\Z", "", text).strip(), ""


def _strip_code_fences(text: str) -> str:
    return _split_code_and_prose(text)[0]


def _clean_explanation(text: str) -> str | None:
    text = _EXPLANATION_LABEL.sub("", text.strip(), count=1).strip()
    return text or None


def is_degenerate_output(text: str, tail_chars: int = 300, threshold: float = 0.5) -> bool:
    """True when >threshold of the last tail_chars characters are one repeated character.

    Catches fp16/bf16 NaN degeneration on MPS and similar hardware where NaN
    logits cause argmax to return token 0 repeatedly (observed as '!!!...!').
    threshold=0.5 means more than half the tail is a single character.
    """
    if not text:
        return False
    tail = text[-tail_chars:]
    if len(tail) < 2:
        return False
    most_common = max(tail.count(c) for c in set(tail))
    return most_common / len(tail) > threshold


def parse_model_output(text: str) -> ParsedOutput:
    """Extract kernel and explanation from model output.

    Accepts the structured format (kernel:/explanation: sections) or falls back
    to heuristics (fenced code block + trailing prose).
    """
    raw = text.strip()
    if not raw:
        return ParsedOutput(None, None, raw)

    # Section headers may carry markdown decoration from general-purpose
    # models: "### kernel:", "**Explanation:**". A header must sit on its own
    # line, so a "# kernel: ..." comment inside the code is not a header.
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
        kernel, after_code = _split_code_and_prose(kernel_match.group(1))
        if explanation_match:
            explanation = explanation_match.group(1).strip() or None
        else:
            # No "explanation:" header line, but prose after the closing fence
            # (e.g. "**Explanation:** The kernel fuses ...") is the explanation.
            explanation = _clean_explanation(after_code)
        return ParsedOutput(kernel or None, explanation, raw)

    # Fallback: the fenced Python block (preferring one with ModelNew) is the
    # kernel; prose after it is the explanation.
    block = _pick_fenced_block(raw)
    if block:
        kernel = block.group(1).strip()
        return ParsedOutput(kernel or None, _clean_explanation(raw[block.end():]), raw)

    # Last resort: if it looks like Python with triton imports, treat whole thing as kernel.
    if "triton" in raw.lower() and "modelnew" in raw.lower():
        return ParsedOutput(_strip_code_fences(raw), None, raw)

    return ParsedOutput(None, raw if raw else None, raw)
