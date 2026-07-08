"""FastAPI backend for the KernelForge demo tool (Phase 6)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kernelforge.dataset import CATEGORIES, load_example_entries, load_training_data
from kernelforge.model import GenerationConfig, KernelGenerator, load_generation_config
from kernelforge.parsing import parse_model_output
from verification.verify_kernel import verify_model_kernel

app = FastAPI(
    title="KernelForge",
    description="Generate optimized Triton kernels with explanations from PyTorch ops.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_generator: KernelGenerator | None = None
_templates: list[dict] = []


class GenerateRequest(BaseModel):
    description: str = Field(..., description="Plain-English op description")
    pytorch_reference: str = Field(..., description="PyTorch reference source (Model + get_inputs/get_init_inputs)")
    verify: bool = Field(True, description="Run verification harness on generated kernel")
    device: str = Field("cuda", description="cuda or cpu")
    tolerance_atol: float = 0.01
    tolerance_rtol: float = 0.01


class GenerateResponse(BaseModel):
    triton_kernel: Optional[str]
    optimization_explanation: Optional[str]
    raw_output: str
    verification: Optional[dict] = None


class TemplateInfo(BaseModel):
    id: str
    category: str
    op_name: str
    description: str


def _get_generator() -> KernelGenerator:
    global _generator
    if _generator is None:
        config_path = os.environ.get(
            "KERNELFORGE_MODEL_CONFIG",
            str(Path(__file__).parent.parent.parent / "training" / "configs" / "qwen2.5-coder-1.5b.json"),
        )
        gen_config = load_generation_config(config_path)
        adapter = os.environ.get("KERNELFORGE_ADAPTER_PATH")
        if adapter:
            gen_config.adapter_path = adapter
        _generator = KernelGenerator(gen_config)
    return _generator


def _load_templates() -> list[dict]:
    global _templates
    if not _templates:
        _templates = load_training_data(fallback_to_examples=True)
    return _templates


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/categories")
def list_categories():
    return {"categories": list(CATEGORIES)}


@app.get("/templates", response_model=list[TemplateInfo])
def list_templates(category: Optional[str] = None):
    templates = _load_templates()
    if category:
        templates = [t for t in templates if t["category"] == category]
    return [
        TemplateInfo(
            id=t["id"],
            category=t["category"],
            op_name=t["op_name"],
            description=t["description"],
        )
        for t in templates
    ]


@app.get("/templates/{template_id}")
def get_template(template_id: str):
    for t in _load_templates():
        if t["id"] == template_id:
            return {
                "id": t["id"],
                "category": t["category"],
                "op_name": t["op_name"],
                "description": t["description"],
                "pytorch_reference": t["pytorch_reference"],
                "reference_explanation": t.get("optimization_explanation"),
            }
    raise HTTPException(status_code=404, detail="Template not found")


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    generator = _get_generator()
    try:
        raw = generator.generate(req.description, req.pytorch_reference)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    parsed = parse_model_output(raw)
    verification_dict = None

    if req.verify and parsed.has_kernel:
        result = verify_model_kernel(
            req.pytorch_reference,
            parsed.triton_kernel,
            device=req.device,
            atol=req.tolerance_atol,
            rtol=req.tolerance_rtol,
        )
        verification_dict = {
            "status": result.status,
            "correct": result.correct,
            "speedup": result.speedup,
            "reference_time_s": result.reference_time_s,
            "candidate_time_s": result.candidate_time_s,
            "passed": result.passed,
            "error": result.error,
        }

    return GenerateResponse(
        triton_kernel=parsed.triton_kernel,
        optimization_explanation=parsed.optimization_explanation,
        raw_output=raw,
        verification=verification_dict,
    )


class VerifyRequest(BaseModel):
    pytorch_reference: str
    triton_kernel: str
    device: str = "cuda"
    atol: float = 0.01
    rtol: float = 0.01


@app.post("/verify")
def verify_only(req: VerifyRequest):
    result = verify_model_kernel(
        req.pytorch_reference,
        req.triton_kernel,
        device=req.device,
        atol=req.atol,
        rtol=req.rtol,
    )
    return {
        "status": result.status,
        "correct": result.correct,
        "speedup": result.speedup,
        "reference_time_s": result.reference_time_s,
        "candidate_time_s": result.candidate_time_s,
        "passed": result.passed,
        "error": result.error,
    }
