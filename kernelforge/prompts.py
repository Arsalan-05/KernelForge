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

TRITON_NOTES = (
    "Triton 3.x pitfalls to avoid:\n"
    "- Write softmax by hand: m = tl.max(x, axis=1); p = tl.exp(x - m[:, None]); p / tl.sum(p, axis=1)[:, None]. "
    "There is no axis argument on tl.softmax.\n"
    "- There is no tl.isnan/tl.isinf; use x != x. Math lives in tl (tl.exp, tl.log, tl.sqrt, tl.rsqrt, tl.where).\n"
    "- tl.arange(0, BLOCK) needs a constexpr power-of-two BLOCK; mask the tail and pass other= to masked tl.load.\n"
    "- tl.dot needs 2-D blocks with every dimension >= 16 and matching input dtypes; accumulate in tl.float32.\n"
    "- For causal masks use a large finite negative (e.g. -1e9) or guard fully-masked rows, otherwise "
    "exp(-inf - -inf) produces NaN.\n"
    "- Cast explicitly with x.to(tl.float32) and store in the output's dtype."
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


def format_response(entry: dict) -> str:
    return (
        f"kernel:\n{entry['triton_kernel'].strip()}\n\n"
        f"explanation:\n{entry['optimization_explanation'].strip()}"
    )


def build_messages(description: str, pytorch_reference: str, example: dict | None = None,
                   notes: bool = False) -> list[dict]:
    """`example` (a dataset entry for a *different* op) is shown as a solved prior turn;
    `notes` appends TRITON_NOTES to the system prompt. Both are for general-purpose models."""
    system = f"{SYSTEM_PROMPT}\n\n{TRITON_NOTES}" if notes else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]
    if example is not None:
        messages += [
            {"role": "user", "content": format_instruction(example["description"], example["pytorch_reference"])},
            {"role": "assistant", "content": format_response(example)},
        ]
    messages.append({"role": "user", "content": format_instruction(description, pytorch_reference)})
    return messages


_TEMP_PATH = re.compile(r"/[^\s\"']*kernelforge_verify_src_[^/\s]+/(candidate|reference)_module\.py")
_MAX_FEEDBACK_LINES = 25


def _trim_error(error: str) -> str:
    error = _TEMP_PATH.sub(lambda m: f"{m.group(1)}.py", error.strip())
    lines = error.splitlines()
    if len(lines) > _MAX_FEEDBACK_LINES:
        lines = ["..."] + lines[-_MAX_FEEDBACK_LINES:]
    return "\n".join(lines)[-3000:]


_REPAIR_HINTS = [
    (re.compile(r"module 'triton\.language' has no attribute '(\w+)'"),
     lambda m: f"`tl.{m.group(1)}` does not exist in Triton 3.x. Build it from tl.exp/tl.max/tl.sum/tl.where "
               "or plain comparisons instead."),
    (re.compile(r"(\w+)\(\) got an unexpected keyword argument '(\w+)'"),
     lambda m: f"`{m.group(1)}` takes no `{m.group(2)}=` argument in Triton 3.x; compute it explicitly "
               "(e.g. softmax via tl.max, tl.exp and tl.sum along the axis)."),
    (re.compile(r"contains NaN|contains inf", re.I),
     lambda m: "NaN/inf usually means a fully-masked row (exp(-inf - -inf)) or an unmasked out-of-bounds load. "
               "Use a large finite negative for masking, pass other= to masked loads, and keep the running "
               "max/sum in float32."),
    (re.compile(r"arange's (range|arguments) must be a power of 2|arange.*power of 2", re.I),
     lambda m: "tl.arange bounds must be constexpr powers of two; round BLOCK up with "
               "triton.next_power_of_2 and mask the tail."),
    (re.compile(r"(shape mismatch|incompatible dimensions|must be >= 16)", re.I),
     lambda m: "Check block shapes: tl.dot needs 2-D operands with every dimension >= 16 and matching inner "
               "dimensions; transpose with tl.trans where needed."),
    (re.compile(r"SyntaxError"),
     lambda m: "Return the full module again as valid Python; check indentation and line breaks around the "
               "reported line."),
    (re.compile(r"code policy"),
     lambda m: "Import only torch, triton, triton.language and math, and don't touch files, processes, "
               "getattr/eval or dunder attributes."),
]


def repair_hints(error: str) -> list[str]:
    """Targeted fixes for failure patterns general-purpose models hit in Triton code."""
    hints = []
    for pattern, render in _REPAIR_HINTS:
        m = pattern.search(error or "")
        if m:
            hints.append(render(m))
    return hints


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
    hints = repair_hints((verification or {}).get("error") or "")
    if hints:
        problem += "\n\nLikely fix:\n" + "\n".join(f"- {h}" for h in hints)
    return (
        f"{problem}\n\n"
        "Fix it. Keep `ModelNew` with the same __init__ and forward signatures as `Model`, return "
        "outputs with the same shapes, and keep the kernel in Triton.\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTION}"
    )


def build_repair_messages(description: str, pytorch_reference: str, previous_output: str,
                          parse_status: str, verification: dict | None,
                          example: dict | None = None, notes: bool = False) -> list[dict]:
    """Only the latest attempt is kept, so context stays bounded across repair rounds."""
    return build_messages(description, pytorch_reference, example, notes) + [
        {"role": "assistant", "content": previous_output},
        {"role": "user", "content": format_repair_feedback(parse_status, verification)},
    ]


def format_training_example(entry: dict) -> dict:
    """Project a schema entry into an instruction-tuning triple."""
    instruction = format_instruction(entry["description"], entry["pytorch_reference"])
    response = format_response(entry)
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
