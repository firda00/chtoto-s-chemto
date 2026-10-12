"""Фейковый OpenAI-совместимый сервер (in-process, через httpx-транспорт) для тестов и демо.

Отдаёт SSE-стрим `/v1/chat/completions` с заданными TTFT и скоростью генерации, а также
`/v1/models`, `/api/tags`, `/api/show`, `/api/pull` (нативный API Ollama).

Время может идти по «виртуальным часам» `FakeClock`: генератор стрима двигает часы вперёд перед
выдачей каждого чанка, поэтому TTFT/tps измеряются детерминированно и мгновенно.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx

from local_llm.client import MockLLM
from local_llm.presets import QUANTS


class FakeClock:
    """Виртуальные часы, совместимые по вызову с `time.perf_counter`."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass(frozen=True)
class MockProfile:
    ttft_s: float = 0.2
    tps: float = 50.0
    max_tokens: int = 64


def simulated_profile(model: str, base: MockProfile) -> MockProfile:
    """Скорость «как у реального GPU»: декодинг упирается в память, tps ∝ 1/бит-на-вес.

    База `base` соответствует q4_K_M; модель с `q8_0` в имени получит ~0.57× tps.
    Prefill (TTFT) упирается в вычисления сильнее, чем в память, поэтому масштабируется мягче (√).
    """
    ref_bits = QUANTS["q4_K_M"].bits
    for name, spec in QUANTS.items():
        if name.lower() in model.lower():
            factor = ref_bits / spec.bits
            return MockProfile(
                ttft_s=round(base.ttft_s / factor**0.5, 6),
                tps=round(base.tps * factor, 6),
                max_tokens=base.max_tokens,
            )
    return base


def _sse(obj: dict | str) -> bytes:
    payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return f"data: {payload}\n\n".encode()


def _chunk(model: str, idx: int, delta: dict, finish: str | None = None) -> dict:
    return {
        "id": f"chatcmpl-mock-{idx}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def stream_events(
    model: str,
    tokens: list[str],
    profile: MockProfile,
    *,
    clock: FakeClock | None = None,
    prompt_tokens: int = 0,
    include_usage: bool = True,
) -> Iterator[bytes]:
    """SSE-события: роль → токены (с задержками) → finish → usage → [DONE]."""

    def wait(dt: float) -> None:
        if clock is not None:
            clock.advance(dt)
        elif dt > 0:
            time.sleep(dt)

    yield _sse(_chunk(model, 0, {"role": "assistant", "content": ""}))
    for i, tok in enumerate(tokens):
        wait(profile.ttft_s if i == 0 else 1.0 / profile.tps)
        yield _sse(_chunk(model, i + 1, {"content": tok}))
    yield _sse(_chunk(model, len(tokens) + 1, {}, finish="stop"))
    if include_usage:
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(tokens),
            "total_tokens": prompt_tokens + len(tokens),
        }
        yield _sse({"id": "chatcmpl-mock-usage", "choices": [], "model": model, "usage": usage})
    yield _sse("[DONE]")


def make_handler(
    *,
    clock: FakeClock | None = None,
    profile: MockProfile | None = None,
    profile_for: Callable[[str], MockProfile] | None = None,
    models: list[str] | None = None,
    fail_models: set[str] | None = None,
    vary_length: bool = False,
) -> Callable[[httpx.Request], httpx.Response]:
    """Обработчик запросов для `httpx.MockTransport` или `respx` (`side_effect=`)."""
    base = profile or MockProfile()
    pick = profile_for or (lambda _model: base)
    known = models or ["qwen2.5:7b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"]
    failing = fail_models or set()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content or b"{}") if request.method == "POST" else {}

        if path.endswith("/chat/completions"):
            model = body.get("model", "mock")
            if model in failing:
                return httpx.Response(500, json={"error": {"message": f"model {model} crashed"}})
            prof = pick(model)
            prompt = " ".join(str(m.get("content", "")) for m in body.get("messages", []))
            limit = min(int(body.get("max_tokens") or prof.max_tokens), prof.max_tokens)
            if vary_length:
                # длина ответа детерминированно зависит от промпта: 40–100% от лимита
                h = int(hashlib.sha256(prompt.encode()).hexdigest()[:4], 16) % 100
                limit = max(1, int(limit * (0.4 + 0.6 * h / 100)))
            words = MockLLM(model=model).answer_text(prompt, n_words=max(limit - 1, 1)).split(" ")
            tokens = [w if i == 0 else " " + w for i, w in enumerate(words)][:limit]
            if not body.get("stream"):
                text = "".join(tokens)
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-mock",
                        "object": "chat.completion",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt.split()),
                            "completion_tokens": len(tokens),
                            "total_tokens": len(prompt.split()) + len(tokens),
                        },
                    },
                )
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=stream_events(
                    model,
                    tokens,
                    prof,
                    clock=clock,
                    prompt_tokens=len(prompt.split()),
                    include_usage=include_usage,
                ),
            )
        if path.endswith("/models"):
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": m, "object": "model"} for m in known]},
            )
        if path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": m, "model": m, "size": 4_700_000_000} for m in known]},
            )
        if path == "/api/show":
            name = body.get("model") or body.get("name")
            if name not in known:
                return httpx.Response(404, json={"error": f"model '{name}' not found"})
            return httpx.Response(
                200,
                json={
                    "modelfile": f"FROM {name}\n",
                    "parameters": "num_ctx 4096",
                    "details": {"quantization_level": "Q4_K_M", "parameter_size": "7.6B"},
                },
            )
        if path == "/api/pull":
            return httpx.Response(200, json={"status": "success"})
        if path == "/health":
            return httpx.Response(200)
        return httpx.Response(404, json={"error": f"no route {path}"})

    return handler


def make_mock_transport(**kwargs) -> httpx.MockTransport:
    return httpx.MockTransport(make_handler(**kwargs))
