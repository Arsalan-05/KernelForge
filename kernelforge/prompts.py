"""Prompt templates for training and inference."""

SYSTEM_PROMPT = (
    "You are a GPU kernel engineer specializing in LLM inference serving. "
    "Given a PyTorch reference implementation, write an optimized Triton kernel "
    "using KernelBench's ModelNew convention, and explain the optimization technique."
)

OUTPUT_FORMAT_INSTRUCTION = (
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
