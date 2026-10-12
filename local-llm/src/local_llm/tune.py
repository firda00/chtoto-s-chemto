"""Подбор пресета + квантования + длины контекста под бюджет VRAM и цель (latency/quality).

Эвристика:
- кандидаты = все пресеты × все квантования бэкенда; для каждой пары берётся максимальный
  контекст из сетки CTX_STEPS, который помещается в `vram × (1 − headroom)` и не превышает
  потолок цели (latency: 8k, balanced: 16k, quality: 32k);
- качество = quality(модели) × quality(квантования);
- скорость декодинга ≈ 1 / размер весов (генерация упирается в пропускную способность памяти);
- score = w_q × (качество/макс)² + w_s × √(скорость/макс), веса зависят от цели.
  Квадрат усиливает разрыв в качестве между классами моделей (3B vs 8B), корень отражает
  убывающую пользу скорости сверх «комфортной для чтения».
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from local_llm.presets import (
    Backend,
    MemoryEstimate,
    ModelPreset,
    get_quant,
    load_presets,
)

Target = Literal["latency", "balanced", "quality"]
TARGETS: tuple[Target, ...] = ("latency", "balanced", "quality")

# (вес качества, вес скорости)
TARGET_WEIGHTS: dict[Target, tuple[float, float]] = {
    "latency": (0.3, 0.7),
    "balanced": (0.6, 0.4),
    "quality": (0.85, 0.15),
}
TARGET_CTX_CAP: dict[Target, int] = {"latency": 8192, "balanced": 16384, "quality": 32768}
CTX_STEPS: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768, 65536, 131072)

DEFAULT_HEADROOM = 0.10  # 10% VRAM оставляем под ОС/драйвер/фрагментацию


class Candidate(BaseModel):
    preset_id: str
    display_name: str
    backend: Backend
    quant: str
    ctx: int
    memory: MemoryEstimate
    quality: float
    speed: float
    score: float = 0.0
    ollama_tag: str | None = None
    vllm_repo: str | None = None


class Recommendation(BaseModel):
    vram_gb: float
    usable_gb: float
    target: Target
    backend: Backend
    min_ctx: int
    best: Candidate | None
    alternatives: list[Candidate]

    @property
    def ok(self) -> bool:
        return self.best is not None


def _max_fitting_ctx(
    preset: ModelPreset,
    quant: str,
    usable_gb: float,
    min_ctx: int,
    ctx_cap: int,
    kv_bits: float,
) -> tuple[int, MemoryEstimate] | None:
    limit = min(preset.max_ctx, ctx_cap)
    steps = [c for c in CTX_STEPS if min_ctx <= c <= limit]
    if not steps and min_ctx <= preset.max_ctx:
        steps = [min_ctx]
    for ctx in sorted(steps, reverse=True):
        est = preset.estimate(quant, ctx, kv_bits=kv_bits)
        if est.total_gb <= usable_gb:
            return ctx, est
    return None


def candidates(
    vram_gb: float,
    *,
    target: Target = "balanced",
    backend: Backend = "ollama",
    min_ctx: int = 4096,
    max_ctx: int | None = None,
    headroom: float = DEFAULT_HEADROOM,
    kv_bits: float = 16,
    presets: dict[str, ModelPreset] | None = None,
) -> list[Candidate]:
    """Все конфигурации, помещающиеся в бюджет, отсортированные по score (лучшие первыми)."""
    if vram_gb <= 0:
        raise ValueError("vram_gb должен быть > 0")
    if target not in TARGET_WEIGHTS:
        raise ValueError(f"target должен быть одним из {TARGETS}")
    presets = presets if presets is not None else load_presets()
    usable = vram_gb * (1.0 - headroom)
    ctx_cap = TARGET_CTX_CAP[target] if max_ctx is None else max_ctx
    ctx_cap = max(ctx_cap, min_ctx)

    found: list[Candidate] = []
    for preset in presets.values():
        for quant in preset.quants(backend):
            fit = _max_fitting_ctx(preset, quant, usable, min_ctx, ctx_cap, kv_bits)
            if fit is None:
                continue
            ctx, est = fit
            found.append(
                Candidate(
                    preset_id=preset.id,
                    display_name=preset.display_name,
                    backend=backend,
                    quant=quant,
                    ctx=ctx,
                    memory=est,
                    quality=round(preset.quality * get_quant(quant).quality, 4),
                    speed=round(1.0 / est.weights_gb, 4),
                    ollama_tag=preset.ollama.tag(quant) if backend == "ollama" else None,
                    vllm_repo=preset.vllm.repo(quant) if backend == "vllm" else None,
                )
            )
    if not found:
        return []

    wq, ws = TARGET_WEIGHTS[target]
    q_max = max(c.quality for c in found)
    s_max = max(c.speed for c in found)
    for c in found:
        c.score = round(wq * (c.quality / q_max) ** 2 + ws * (c.speed / s_max) ** 0.5, 4)
    found.sort(key=lambda c: (c.score, c.quality, c.ctx), reverse=True)
    return found


def recommend(
    vram_gb: float,
    *,
    target: Target = "balanced",
    backend: Backend = "ollama",
    min_ctx: int = 4096,
    max_ctx: int | None = None,
    headroom: float = DEFAULT_HEADROOM,
    kv_bits: float = 16,
    top: int = 5,
    presets: dict[str, ModelPreset] | None = None,
) -> Recommendation:
    found = candidates(
        vram_gb,
        target=target,
        backend=backend,
        min_ctx=min_ctx,
        max_ctx=max_ctx,
        headroom=headroom,
        kv_bits=kv_bits,
        presets=presets,
    )
    return Recommendation(
        vram_gb=vram_gb,
        usable_gb=round(vram_gb * (1.0 - headroom), 3),
        target=target,
        backend=backend,
        min_ctx=min_ctx,
        best=found[0] if found else None,
        alternatives=found[1 : max(1, top)],
    )
