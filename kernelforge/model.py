"""Model loading and text generation for baseline and fine-tuned eval."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from kernelforge.prompts import SYSTEM_PROMPT, build_messages


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

    def _encode_prompt(self, description: str, pytorch_reference: str) -> dict:
        return self._encode_messages(build_messages(description, pytorch_reference))

    def _encode_messages(self, messages: list[dict]) -> dict:
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            turns = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages[1:])
            prompt = f"{SYSTEM_PROMPT}\n\n{turns}\n\nAssistant:"

        inputs = self._tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self._model.device) for k, v in inputs.items()}

    def _generate_kwargs(self, temperature: float | None = None) -> dict:
        temperature = self.config.temperature if temperature is None else temperature
        return {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": temperature,
            "top_p": self.config.top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

    def generate(self, description: str, pytorch_reference: str) -> str:
        self.load()
        import torch

        inputs = self._encode_prompt(description, pytorch_reference)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **self._generate_kwargs())

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(self, description: str, pytorch_reference: str) -> Iterator[str]:
        """Yield decoded text chunks as they're generated.

        Closing the iterator early stops generation on the GPU before returning,
        so a caller holding a device lock can release it safely afterwards.
        """
        self.load()
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        stop = threading.Event()
        errors: list[BaseException] = []

        class _StopOnEvent(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return stop.is_set()

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs = self._encode_prompt(description, pytorch_reference)

        def _run() -> None:
            try:
                with torch.no_grad():
                    self._model.generate(
                        **inputs,
                        **self._generate_kwargs(),
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([_StopOnEvent()]),
                    )
            except BaseException as exc:  # surfaced to the consumer below
                errors.append(exc)
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            stop.set()
            thread.join()
        if errors:
            raise errors[0]

    def stream_chat(
        self, messages: list[dict], num_sequences: int = 1, temperature: float | None = None
    ) -> Iterator[tuple[int, str]]:
        """Sample `num_sequences` completions in one batched generate(), yielding
        (sequence_index, text_chunk) as tokens arrive.

        Same cancellation contract as stream(): closing the iterator stops
        generation before it returns.
        """
        self.load()
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        stop = threading.Event()
        errors: list[BaseException] = []

        class _StopOnEvent(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return stop.is_set()

        streamer = BatchTextStreamer(self._tokenizer, num_sequences, self._tokenizer.eos_token_id)
        inputs = self._encode_messages(messages)
        kwargs = self._generate_kwargs(temperature)
        if num_sequences > 1:
            kwargs.update(num_return_sequences=num_sequences, do_sample=True)

        def _run() -> None:
            try:
                with torch.no_grad():
                    self._model.generate(
                        **inputs, **kwargs, streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([_StopOnEvent()]),
                    )
            except BaseException as exc:  # surfaced to the consumer below
                errors.append(exc)
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            for item in streamer:
                yield item
        finally:
            stop.set()
            thread.join()
        if errors:
            raise errors[0]

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    @property
    def backend(self) -> str | None:
        return self._backend


class BatchTextStreamer:
    """transformers streamer for batched sampling (TextIteratorStreamer only allows batch 1).

    generate() calls put() once with the prompt ids, then once per step with a
    (batch,) tensor of new token ids. Each row is decoded incrementally; text is
    held back while it ends in an incomplete UTF-8 sequence, and a row's cache is
    reset at each newline so decoding stays linear in output length.
    """

    def __init__(self, tokenizer, num_sequences: int, eos_token_id: int | None):
        import queue

        self._tokenizer = tokenizer
        self._eos = eos_token_id
        self._queue: "queue.Queue" = queue.Queue()
        self._caches: list[list[int]] = [[] for _ in range(num_sequences)]
        self._emitted = [0] * num_sequences
        self._finished = [False] * num_sequences
        self._prompt_seen = False
        self._sentinel = object()

    def put(self, value) -> None:
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        tokens = value.reshape(-1).tolist()
        for i, token in enumerate(tokens[: len(self._caches)]):
            if self._finished[i]:
                continue
            if token == self._eos:
                self._finished[i] = True
                self._flush(i)
                continue
            self._caches[i].append(token)
            self._emit(i)

    def _decode(self, i: int) -> str:
        return self._tokenizer.decode(self._caches[i], skip_special_tokens=True)

    def _emit(self, i: int) -> None:
        text = self._decode(i)
        if text.endswith("\ufffd"):
            return
        chunk = text[self._emitted[i]:]
        if text.endswith("\n"):
            self._caches[i], self._emitted[i] = [], 0
        else:
            self._emitted[i] = len(text)
        if chunk:
            self._queue.put((i, chunk))

    def _flush(self, i: int) -> None:
        chunk = self._decode(i)[self._emitted[i]:]
        self._caches[i], self._emitted[i] = [], 0
        if chunk:
            self._queue.put((i, chunk))

    def end(self) -> None:
        for i in range(len(self._caches)):
            if not self._finished[i]:
                self._flush(i)
        self._queue.put(self._sentinel)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                return
            yield item


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
