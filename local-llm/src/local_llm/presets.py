"""Каталог пресетов моделей, справочник квантований и оценка потребления памяти.

Формула оценки (GiB):

    weights  = params × bits_per_weight / 8 × overhead
    kv_cache = 2 (K и V) × n_layers × n_kv_heads × head_dim × ctx_len × batch × kv_bits / 8
    total    = weights + kv_cache + runtime   (runtime — CUDA-контекст, буферы, ~0.5 GiB)

`overhead` (по умолчанию 1.10) покрывает активации, embedding/output-слои в более
высокой точности и фрагментацию аллокатора.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Backend = Literal["ollama", "vllm"]
BACKENDS: tuple[Backend, ...] = ("ollama", "vllm")
VRAM_TIERS: tuple[int, ...] = (8, 12, 16, 24)
GIB = 1024**3

DEFAULT_OVERHEAD = 1.10
DEFAULT_RUNTIME_GB = 0.5


class QuantSpec(BaseModel):
    """Формат квантования: эффективные биты на вес и относительное качество."""

    name: str
    backends: tuple[Backend, ...]
    bits: float = Field(gt=0, description="Эффективные биты на вес (с учётом scale/zero-point)")
    quality: float = Field(gt=0, le=1, description="Сохранение качества относительно fp16")
    description: str


QUANTS: dict[str, QuantSpec] = {
    q.name: q
    for q in [
        QuantSpec(
            name="q4_K_M",
            backends=("ollama",),
            bits=4.85,
            quality=0.95,
            description="GGUF k-quant 4 бит (смешанный). Дефолт Ollama, баланс размер/качество",
        ),
        QuantSpec(
            name="q5_K_M",
            backends=("ollama",),
            bits=5.69,
            quality=0.975,
            description="GGUF k-quant 5 бит. Заметно ближе к fp16, +17% памяти к q4_K_M",
        ),
        QuantSpec(
            name="q8_0",
            backends=("ollama",),
            bits=8.5,
            quality=0.995,
            description="GGUF 8 бит. Практически без потерь, в ~1.9 раза меньше fp16",
        ),
        QuantSpec(
            name="fp16",
            backends=("ollama", "vllm"),
            bits=16.0,
            quality=1.0,
            description="Без квантования (fp16/bf16). Эталон качества, максимум памяти",
        ),
        QuantSpec(
            name="awq",
            backends=("vllm",),
            bits=4.25,
            quality=0.955,
            description="AWQ int4 (group 128). Быстрые Marlin-ядра в vLLM, качество ~q4_K_M",
        ),
        QuantSpec(
            name="gptq",
            backends=("vllm",),
            bits=4.25,
            quality=0.945,
            description="GPTQ int4 (group 128). Широкий выбор готовых чекпоинтов",
        ),
        QuantSpec(
            name="fp8",
            backends=("vllm",),
            bits=8.0,
            quality=0.99,
            description="FP8 W8A8 (Ada/Hopper: RTX 40xx, L4, H100). Почти без потерь, x2 к fp16",
        ),
    ]
}


def get_quant(name: str) -> QuantSpec:
    try:
        return QUANTS[name]
    except KeyError:
        raise KeyError(f"Неизвестное квантование {name!r}. Доступны: {', '.join(QUANTS)}") from None


def quants_for(backend: Backend) -> list[QuantSpec]:
    return [q for q in QUANTS.values() if backend in q.backends]


# ---------------------------------------------------------------------------
# Оценка памяти
# ---------------------------------------------------------------------------


class MemoryEstimate(BaseModel):
    weights_gb: float
    kv_cache_gb: float
    runtime_gb: float
    total_gb: float

    def fits(self, vram_gb: float, headroom: float = 0.0) -> bool:
        """Помещается ли в `vram_gb`, оставив долю `headroom` свободной."""
        return self.total_gb <= vram_gb * (1.0 - headroom)


def weights_gb(params_b: float, quant_bits: float, overhead: float = DEFAULT_OVERHEAD) -> float:
    if params_b <= 0 or quant_bits <= 0:
        raise ValueError("params_b и quant_bits должны быть > 0")
    return params_b * 1e9 * quant_bits / 8 / GIB * overhead


def kv_cache_gb(
    ctx_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int = 128,
    kv_bits: float = 16,
    batch: int = 1,
) -> float:
    """Размер KV-кэша: 2 тензора (K, V) на слой × kv-головы × head_dim × токены."""
    if ctx_len < 0 or batch < 1:
        raise ValueError("ctx_len >= 0, batch >= 1")
    return 2 * n_layers * n_kv_heads * head_dim * ctx_len * batch * kv_bits / 8 / GIB


def _resolve_bits(quant_bits: float | str) -> float:
    if isinstance(quant_bits, str):
        return get_quant(quant_bits).bits
    return float(quant_bits)


def memory_breakdown(
    params_b: float,
    quant_bits: float | str,
    ctx_len: int,
    *,
    n_layers: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    kv_bits: float = 16,
    batch: int = 1,
    overhead: float = DEFAULT_OVERHEAD,
    runtime_gb: float = DEFAULT_RUNTIME_GB,
) -> MemoryEstimate:
    w = weights_gb(params_b, _resolve_bits(quant_bits), overhead)
    kv = kv_cache_gb(ctx_len, n_layers, n_kv_heads, head_dim, kv_bits, batch)
    return MemoryEstimate(
        weights_gb=round(w, 3),
        kv_cache_gb=round(kv, 3),
        runtime_gb=runtime_gb,
        total_gb=round(w + kv + runtime_gb, 3),
    )


def estimate_memory_gb(
    params_b: float,
    quant_bits: float | str,
    ctx_len: int,
    *,
    n_layers: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    kv_bits: float = 16,
    batch: int = 1,
    overhead: float = DEFAULT_OVERHEAD,
    runtime_gb: float = DEFAULT_RUNTIME_GB,
) -> float:
    """Оценка VRAM (GiB): params × bits/8 × overhead + KV-кэш + runtime.

    `quant_bits` — число бит на вес (например 4.85) или имя квантования (`"q4_K_M"`).
    Архитектурные параметры по умолчанию соответствуют Llama-3.1-8B (32 слоя, 8 KV-голов, GQA).
    """
    return memory_breakdown(
        params_b,
        quant_bits,
        ctx_len,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        kv_bits=kv_bits,
        batch=batch,
        overhead=overhead,
        runtime_gb=runtime_gb,
    ).total_gb


# ---------------------------------------------------------------------------
# Пресеты моделей
# ---------------------------------------------------------------------------


class ArchSpec(BaseModel):
    n_layers: int = Field(gt=0)
    n_kv_heads: int = Field(gt=0)
    head_dim: int = Field(default=128, gt=0)


class OllamaSpec(BaseModel):
    tag_prefix: str = Field(description="Например 'qwen2.5:7b-instruct'; тег = prefix-quant")
    quants: list[str] = Field(default_factory=lambda: ["q4_K_M", "q5_K_M", "q8_0", "fp16"])

    def tag(self, quant: str) -> str:
        return f"{self.tag_prefix}-{quant}"


class VLLMSpec(BaseModel):
    hf_repo: str = Field(description="Базовый чекпоинт (bf16)")
    quant_repos: dict[str, str] = Field(
        default_factory=dict,
        description="Готовые квантованные чекпоинты; fp8 без репо = динамическое квантование",
    )

    def repo(self, quant: str) -> str:
        if quant in ("fp16", "fp8") and quant not in self.quant_repos:
            return self.hf_repo
        try:
            return self.quant_repos[quant]
        except KeyError:
            raise KeyError(f"Для {self.hf_repo} нет чекпоинта {quant!r}") from None

    @property
    def quants(self) -> list[str]:
        order = list(QUANTS)
        return sorted({"fp16", "fp8", *self.quant_repos}, key=order.index)


class GenerationDefaults(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    repeat_penalty: float = Field(default=1.05, ge=0.5, le=2)


class TierRecommendation(BaseModel):
    quant: str
    ctx: int = Field(gt=0)


class ModelPreset(BaseModel):
    id: str
    display_name: str
    family: str
    params_b: float = Field(gt=0, description="Число параметров, млрд")
    max_ctx: int = Field(gt=0)
    quality: float = Field(
        gt=0, le=1, description="Относительный индекс качества модели (эвристика)"
    )
    arch: ArchSpec
    ollama: OllamaSpec
    vllm: VLLMSpec
    generation: GenerationDefaults = Field(default_factory=GenerationDefaults)
    recommended: dict[Backend, dict[int, TierRecommendation | None]] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def _validate(self) -> ModelPreset:
        for q in self.ollama.quants:
            if "ollama" not in get_quant(q).backends:
                raise ValueError(f"{self.id}: {q} не поддерживается Ollama")
        for q in self.vllm.quant_repos:
            if "vllm" not in get_quant(q).backends:
                raise ValueError(f"{self.id}: {q} не поддерживается vLLM")
        for backend, tiers in self.recommended.items():
            for tier, rec in tiers.items():
                if tier not in VRAM_TIERS:
                    raise ValueError(
                        f"{self.id}: неизвестный VRAM-тир {tier} (ожидаются {VRAM_TIERS})"
                    )
                if rec is None:
                    continue
                if rec.quant not in self.quants(backend):
                    raise ValueError(f"{self.id}: {rec.quant} недоступно для {backend}")
                if rec.ctx > self.max_ctx:
                    raise ValueError(f"{self.id}: ctx {rec.ctx} > max_ctx {self.max_ctx}")
        return self

    def quants(self, backend: Backend) -> list[str]:
        return list(self.ollama.quants) if backend == "ollama" else self.vllm.quants

    def estimate(
        self,
        quant: str,
        ctx_len: int,
        *,
        kv_bits: float = 16,
        batch: int = 1,
        overhead: float = DEFAULT_OVERHEAD,
        runtime_gb: float = DEFAULT_RUNTIME_GB,
    ) -> MemoryEstimate:
        return memory_breakdown(
            self.params_b,
            get_quant(quant).bits,
            ctx_len,
            n_layers=self.arch.n_layers,
            n_kv_heads=self.arch.n_kv_heads,
            head_dim=self.arch.head_dim,
            kv_bits=kv_bits,
            batch=batch,
            overhead=overhead,
            runtime_gb=runtime_gb,
        )


# ---------------------------------------------------------------------------
# Загрузка каталога
# ---------------------------------------------------------------------------


def default_presets_dir() -> Path:
    env = os.environ.get("LOCAL_LLM_PRESETS_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for candidate in (here.parents[1] / "presets", here / "_presets"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Каталог пресетов не найден; задайте LOCAL_LLM_PRESETS_DIR")


def load_preset_file(path: str | Path) -> ModelPreset:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался YAML-объект")
    return ModelPreset.model_validate(data)


@lru_cache(maxsize=8)
def _load_cached(directory: str) -> dict[str, ModelPreset]:
    result: dict[str, ModelPreset] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        preset = load_preset_file(path)
        if preset.id in result:
            raise ValueError(f"Дублирующийся id пресета {preset.id!r} в {path}")
        result[preset.id] = preset
    if not result:
        raise FileNotFoundError(f"В {directory} нет *.yaml пресетов")
    return result


def load_presets(directory: str | Path | None = None) -> dict[str, ModelPreset]:
    d = Path(directory) if directory else default_presets_dir()
    return dict(_load_cached(str(d.resolve())))


def get_preset(preset_id: str, directory: str | Path | None = None) -> ModelPreset:
    presets = load_presets(directory)
    if preset_id in presets:
        return presets[preset_id]
    normalized = preset_id.lower().replace("_", "-").replace(" ", "")
    for p in presets.values():
        if normalized in (p.id.lower(), p.display_name.lower().replace(" ", "")):
            return p
    raise KeyError(f"Пресет {preset_id!r} не найден. Доступны: {', '.join(presets)}")
