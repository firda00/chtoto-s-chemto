"""Эмбеддеры.

* HashingEmbedder — детерминированный feature hashing на numpy (слова + символьные 3-граммы).
  Без сети и моделей: используется в тестах и как fallback «из коробки».
* FastEmbedEmbedder — ONNX-модель через `fastembed` (extra `local`), лениво загружается
  при первом вызове. По умолчанию BAAI/bge-small-en-v1.5 (384 измерения).
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rag_system.text import tokenize, words

DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
_KNOWN_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "intfloat/multilingual-e5-small": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}


@runtime_checkable
class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    """Детерминированный «мешок признаков» с хешированием в фиксированное пространство.

    Признаки: стемы слов (вес 1.0) и символьные 3-граммы слов (вес 0.3) — последние
    дают устойчивость к словоформам и опечаткам. Знак признака тоже берётся из хеша,
    чтобы коллизии в среднем гасили друг друга. Вес — сублинейный TF (1 + log tf).
    """

    name = "hashing"

    def __init__(self, dim: int = 512, ngram_weight: float = 0.3) -> None:
        self._dim = dim
        self.ngram_weight = ngram_weight

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _hash(feature: str) -> int:
        return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest())

    def _features(self, text: str) -> Counter[str]:
        feats: Counter[str] = Counter()
        for tok in tokenize(text):
            feats["w:" + tok] += 1
        for w in words(text):
            padded = f"<{w}>"
            for i in range(len(padded) - 2):
                feats["g:" + padded[i : i + 3]] += 1
        return feats

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for feat, tf in self._features(text).items():
            h = self._hash(feat)
            idx = h % self._dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            weight = 1.0 if feat.startswith("w:") else self.ngram_weight
            vec[idx] += sign * weight * (1.0 + math.log(tf))
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class FastEmbedEmbedder:
    """Обёртка над fastembed.TextEmbedding с ленивой загрузкой модели."""

    name = "fastembed"

    def __init__(self, model_name: str = DEFAULT_FASTEMBED_MODEL, **kwargs: Any) -> None:
        self.model_name = model_name
        self._kwargs = kwargs
        self._model: Any = None
        self._dim: int | None = _KNOWN_DIMS.get(model_name)

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("fastembed") is not None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - зависит от окружения
                raise RuntimeError(
                    "fastembed не установлен. Установите extra: `uv sync --extra local` "
                    "или используйте EMBEDDER=hashing."
                ) from exc
            self._model = TextEmbedding(model_name=self.model_name, **self._kwargs)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["probe"]).shape[1])
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = list(self._load().embed(list(texts)))
        return l2_normalize(np.stack(vecs))

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        # у bge/e5 есть отдельный префикс для запросов — fastembed добавляет его в query_embed
        vecs = list(model.query_embed([text])) if hasattr(model, "query_embed") else None
        if not vecs:
            vecs = list(model.embed([text]))
        return l2_normalize(np.asarray(vecs[0]))


def get_embedder(name: str = "auto", model: str = DEFAULT_FASTEMBED_MODEL) -> Embedder:
    """auto → fastembed, если пакет установлен, иначе hashing."""
    name = (name or "auto").lower()
    if name == "auto":
        name = "fastembed" if FastEmbedEmbedder.available() else "hashing"
    if name == "hashing":
        return HashingEmbedder()
    if name == "fastembed":
        if not FastEmbedEmbedder.available():
            raise RuntimeError(
                "EMBEDDER=fastembed, но пакет fastembed не установлен: `uv sync --extra local`."
            )
        return FastEmbedEmbedder(model_name=model)
    raise ValueError(f"Неизвестный эмбеддер: {name!r} (auto | fastembed | hashing)")
