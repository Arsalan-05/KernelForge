"""Prompt templates for training and inference."""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are a GPU kernel engineer specializing in LLM inference serving. "
    "Given a PyTorch reference implementation, write an optimized Triton kernel "
    "using KernelBench's ModelNew convention, and explain the optimization technique."
)

KERNEL_CONTRACT = (
    "Requirements for the kernel source:\n"
    "- One self-contained module: import only torch, triton, triton.language (as tl) and math. "
    "Don't import the reference or any `kernelbench` package; they aren't available.\n"
    "- Define `class ModelNew(torch.nn.Module)` with the same __init__ and forward signatures as "
    "`Model`, returning outputs with the same shapes and dtypes.\n"
    "- Allocate outputs on the inputs' device (e.g. torch.empty_like(x)); never hardcode 'cuda'.\n"
    "- Don't redefine `get_inputs`, `get_init_inputs` or the shape constants; the harness supplies them."
)

OUTPUT_FORMAT_INSTRUCTION = (
    f"{KERNEL_CONTRACT}\n\n"
    "Respond with exactly two sections:\n"
    "kernel:\n"
    "<full Python source with @triton.jit kernel(s) and ModelNew>\n\n"
    "explanation:\n"
    "<1-3 sentences describing the optimization technique and why it helps at serving scale>"
)


def format_instruction(description: str, pytorch_reference: str) -> str:
    """Build the user-facing instruction for inference."""
    return (
        f"{description}\n\n"
        f"PyTorch reference:\n```python\n{pytorch_reference.strip()}\n```\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTION}"
    )


def build_messages(description: str, pytorch_reference: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_instruction(description, pytorch_reference)},
    ]


_TEMP_PATH = re.compile(r"/[^\s\"']*kernelforge_verify_src_[^/\s]+/(candidate|reference)_module\.py")
_MAX_FEEDBACK_LINES = 25


def _trim_error(error: str) -> str:
    error = _TEMP_PATH.sub(lambda m: f"{m.group(1)}.py", error.strip())
    lines = error.splitlines()
    if len(lines) > _MAX_FEEDBACK_LINES:
        lines = ["..."] + lines[-_MAX_FEEDBACK_LINES:]
    return "\n".join(lines)[-3000:]


def format_repair_feedback(parse_status: str, verification: dict | None) -> str:
    """Turn a failed attempt into the next user turn of a repair conversation."""
    if parse_status != "ok":
        problem = (
            "Your previous response could not be used: it did not contain a `kernel:` section with "
            "a complete Python module defining a `ModelNew` class."
        )
    elif verification and verification.get("status") == "ok" and verification.get("correct") is False:
        problem = (
            "Your previous kernel ran but produced wrong results when checked against the PyTorch "
            f"reference on identical inputs:\n{_trim_error(verification.get('error') or 'outputs did not match')}"
        )
    else:
        status = (verification or {}).get("status", "error")
        problem = (
            f"Your previous kernel failed in the verification harness (status: {status}):\n"
            f"{_trim_error((verification or {}).get('error') or 'no error message')}"
        )
    return (
        f"{problem}\n\n"
        "Fix it. Keep `ModelNew` with the same __init__ and forward signatures as `Model`, return "
        "outputs with the same shapes, and keep the kernel in Triton.\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTION}"
    )


def build_repair_messages(description: str, pytorch_reference: str, previous_output: str,
                          parse_status: str, verification: dict | None) -> list[dict]:
    """Only the latest attempt is kept, so context stays bounded across repair rounds."""
    return build_messages(description, pytorch_reference) + [
        {"role": "assistant", "content": previous_output},
        {"role": "user", "content": format_repair_feedback(parse_status, verification)},
    ]


def format_training_example(entry: dict) -> dict:
    """Project a schema entry into an instruction-tuning triple."""
    instruction = format_instruction(entry["description"], entry["pytorch_reference"])
    response = (
        f"kernel:\n{entry['triton_kernel'].strip()}\n\n"
        f"explanation:\n{entry['optimization_explanation'].strip()}"
    )
    return {
        "id": entry["id"],
        "category": entry["category"],
        "instruction": instruction,
        "response": response,
        "pytorch_reference": entry["pytorch_reference"],
        "triton_kernel": entry["triton_kernel"],
        "optimization_explanation": entry["optimization_explanation"],
        "tolerance": entry.get("tolerance", {"atol": 1e-2, "rtol": 1e-2}),
    }
