"""Shared helper for assembling schema-conformant dataset entries (see data/SCHEMA.md)."""


def make_entry(
    id: str,
    category: str,
    op_name: str,
    description: str,
    dtype: str,
    pytorch_reference: str,
    triton_kernel: str,
    optimization_explanation: str,
    tolerance: dict,
    provenance_notes: str,
    test_shapes: list = None,
    kernelbench_ref: str = None,
) -> dict:
    return {
        "id": id,
        "category": category,
        "op_name": op_name,
        "description": description,
        "dtype": dtype,
        "pytorch_reference": pytorch_reference,
        "triton_kernel": triton_kernel,
        "optimization_explanation": optimization_explanation,
        "tolerance": tolerance,
        "test_shapes": test_shapes or [],
        "provenance": {
            "source": "hand_written" if kernelbench_ref is None else "kernelbench_adapted",
            "kernelbench_ref": kernelbench_ref,
            "license": "MIT" if kernelbench_ref else None,
            "notes": provenance_notes,
        },
        "verification": {
            "verified": False,
            "correct": None,
            "speedup": None,
            "device": None,
            "verified_at": None,
        },
    }


FP16_TOLERANCE = {"atol": 1e-2, "rtol": 1e-2}
