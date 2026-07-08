"""Model loading and text generation for baseline and fine-tuned eval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.prompts import SYSTEM_PROMPT, format_instruction


@dataclass
class GenerationConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    adapter_path: str | None = None
    max_new_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.95
    load_in_4bit: bool = True
    max_seq_length: int = 8192


class KernelGenerator:
    """Wraps Unsloth (preferred on Linux+GPU) or transformers+PEFT fallback."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._backend = None

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            self._load_unsloth()
            self._backend = "unsloth"
        except ImportError:
            self._load_transformers()
            self._backend = "transformers"

    def _load_unsloth(self) -> None:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=None,
        )

        if self.config.adapter_path:
            model = FastLanguageModel.for_inference(model)
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.config.adapter_path)

        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = None
        if self.config.load_in_4bit:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                quant_config = None

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.config.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.config.adapter_path)

        model.eval()
        self._model = model
        self._tokenizer = tokenizer

    def generate(self, description: str, pytorch_reference: str) -> str:
        self.load()
        user_content = format_instruction(description, pytorch_reference)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_content}\n\nAssistant:"

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @property
    def backend(self) -> str | None:
        return self._backend


def load_generation_config(path: Path | str) -> GenerationConfig:
    """Load a GenerationConfig from a JSON or YAML file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return GenerationConfig(**data)
