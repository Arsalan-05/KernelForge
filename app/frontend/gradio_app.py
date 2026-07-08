"""Gradio frontend for KernelForge (Phase 6)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import httpx

API_URL = os.environ.get("KERNELFORGE_API_URL", "http://localhost:8000")


def _fetch_templates(category: str | None = None) -> list[dict]:
    params = {"category": category} if category else {}
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{API_URL}/templates", params=params)
        resp.raise_for_status()
        return resp.json()


def _load_template(template_id: str) -> tuple[str, str, str]:
    if not template_id:
        return "", "", ""
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{API_URL}/templates/{template_id}")
        resp.raise_for_status()
        data = resp.json()
        return data["description"], data["pytorch_reference"], data.get("reference_explanation", "")


def generate_kernel(description: str, pytorch_reference: str, verify: bool, device: str):
    if not description.strip() or not pytorch_reference.strip():
        return "Provide both a description and PyTorch reference.", "", "", ""

    payload = {
        "description": description,
        "pytorch_reference": pytorch_reference,
        "verify": verify,
        "device": device,
    }
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(f"{API_URL}/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

    kernel = data.get("triton_kernel") or "(no kernel parsed)"
    explanation = data.get("optimization_explanation") or "(no explanation parsed)"
    verification = data.get("verification")

    verify_text = "Verification skipped."
    if verification:
        if verification.get("passed"):
            verify_text = (
                f"PASSED — correct={verification['correct']}, "
                f"speedup={verification.get('speedup', 'n/a')}x"
            )
        else:
            verify_text = (
                f"FAILED — status={verification.get('status')}, "
                f"correct={verification.get('correct')}, error={verification.get('error', '')}"
            )

    return kernel, explanation, verify_text, data.get("raw_output", "")


def build_ui() -> gr.Blocks:
    categories = ["(all)", "quantized_matmul", "attention", "kv_cache", "norm", "rope"]

    with gr.Blocks(title="KernelForge", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# KernelForge\n"
            "Paste a PyTorch op (KernelBench `Model` convention) and get an optimized "
            "**Triton kernel** plus a human-readable **optimization explanation**."
        )

        with gr.Row():
            category_filter = gr.Dropdown(categories, value="(all)", label="Category filter")
            template_dropdown = gr.Dropdown([], label="Load template", interactive=True)

        description = gr.Textbox(label="Op description", lines=2)
        pytorch_reference = gr.Code(label="PyTorch reference", language="python", lines=18)

        with gr.Row():
            verify_checkbox = gr.Checkbox(value=True, label="Verify kernel (requires CUDA)")
            device = gr.Radio(["cuda", "cpu"], value="cuda", label="Device")

        generate_btn = gr.Button("Generate kernel + explanation", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Generated Triton kernel")
                kernel_output = gr.Code(label="Triton kernel", language="python", lines=20)
            with gr.Column():
                gr.Markdown("### Optimization explanation")
                explanation_output = gr.Textbox(label="Why this is faster", lines=10)

        verification_output = gr.Textbox(label="Verification result", lines=2)
        raw_output = gr.Textbox(label="Raw model output", lines=6, visible=False)

        def refresh_templates(cat):
            c = None if cat == "(all)" else cat
            try:
                templates = _fetch_templates(c)
                choices = [(f"{t['op_name']} ({t['id']})", t["id"]) for t in templates]
                return gr.Dropdown(choices=choices, value=None)
            except Exception as exc:
                return gr.Dropdown(choices=[(f"API error: {exc}", "")], value=None)

        category_filter.change(refresh_templates, inputs=[category_filter], outputs=[template_dropdown])
        demo.load(refresh_templates, inputs=[category_filter], outputs=[template_dropdown])

        template_dropdown.change(
            lambda tid: _load_template(tid),
            inputs=[template_dropdown],
            outputs=[description, pytorch_reference, explanation_output],
        )

        generate_btn.click(
            generate_kernel,
            inputs=[description, pytorch_reference, verify_checkbox, device],
            outputs=[kernel_output, explanation_output, verification_output, raw_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", "7860")))
