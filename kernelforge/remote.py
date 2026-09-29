"""Generation through an OpenAI-compatible chat-completions API.

Lets the demo run on hosts without a GPU (e.g. Railway): the search engine,
harness and UI are unchanged, only the text comes from a hosted model. This is
a general-purpose model, not KernelForge's fine-tuned adapter, and the UI says so.

Works with any server that speaks POST {base_url}/chat/completions with
`stream: true` (OpenAI, Groq, Together, Fireworks, OpenRouter, vLLM, Ollama).
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Iterator, Optional

import httpx

_DONE = object()


class RemoteGenerationError(RuntimeError):
    pass


class RemoteChatGenerator:
    """Same interface as KernelGenerator.stream_chat / count_tokens.

    N candidates are N parallel streaming requests; chunks are yielded as
    (sequence_index, text) in arrival order, so all candidates fill in at once.
    """

    backend = "remote"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.95,
        timeout_s: float = 120.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_s = timeout_s
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0))

    def load(self) -> None:
        pass

    def count_tokens(self, text: str) -> int:
        # No tokenizer for an arbitrary hosted model; ~4 chars/token is the usual estimate.
        return max(1, round(len(text) / 4)) if text else 0

    def _stream_one(self, index: int, messages: list[dict], temperature: float,
                    out: queue.Queue, stop: threading.Event) -> None:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "top_p": self.top_p,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "text/event-stream"}
        try:
            with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                     json=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    detail = resp.read().decode(errors="replace")[:300]
                    raise RemoteGenerationError(f"HTTP {resp.status_code} from the model API: {detail}")
                for line in resp.iter_lines():
                    if stop.is_set():
                        return
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise RemoteGenerationError(f"model API error: {chunk['error']}")
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            out.put((index, text))
        except Exception as exc:  # surfaced to the consumer, not swallowed
            out.put((index, exc))
        finally:
            out.put((index, _DONE))

    def stream_chat(
        self, messages: list[dict], num_sequences: int = 1, temperature: Optional[float] = None
    ) -> Iterator[tuple[int, str]]:
        temp = self.temperature if temperature is None else temperature
        out: queue.Queue = queue.Queue()
        stop = threading.Event()
        threads = [
            threading.Thread(target=self._stream_one, args=(i, messages, temp, out, stop), daemon=True)
            for i in range(num_sequences)
        ]
        for t in threads:
            t.start()

        errors: dict[int, Exception] = {}
        remaining = num_sequences
        try:
            while remaining:
                index, item = out.get()
                if item is _DONE:
                    remaining -= 1
                elif isinstance(item, Exception):
                    errors[index] = item
                else:
                    yield index, item
        finally:
            stop.set()

        if len(errors) == num_sequences:
            first = errors[min(errors)]
            raise first if isinstance(first, RemoteGenerationError) else RemoteGenerationError(
                f"{type(first).__name__}: {first}")
        for index, exc in sorted(errors.items()):
            yield index, f"\n[generation failed for this candidate: {exc}]"
