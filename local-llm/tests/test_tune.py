import pytest

from local_llm.presets import get_quant, load_presets
from local_llm.tune import TARGETS, candidates, recommend


@pytest.mark.parametrize("vram", [8, 12, 16, 24])
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("backend", ["ollama", "vllm"])
def test_recommendation_always_fits_budget(vram, target, backend):
    rec = recommend(vram, target=target, backend=backend)
    assert rec.best is not None
    assert rec.best.memory.total_gb <= vram * 0.9
    assert rec.best.ctx >= 4096
    assert rec.best.quant in load_presets()[rec.best.preset_id].quants(backend)
    for alt in rec.alternatives:
        assert alt.memory.total_gb <= vram * 0.9
        assert alt.score <= rec.best.score


@pytest.mark.parametrize(
    "vram, target, expected",
    [
        (8, "latency", ("llama3.2-3b", "q4_K_M")),
        (8, "balanced", ("qwen3-8b", "q4_K_M")),
        (12, "quality", ("qwen3-8b", "q8_0")),
        (16, "quality", ("qwen2.5-14b", "q5_K_M")),
        (24, "quality", ("qwen2.5-14b", "q8_0")),
    ],
)
def test_known_recommendations(vram, target, expected):
    best = recommend(vram, target=target).best
    assert (best.preset_id, best.quant) == expected


def test_latency_target_picks_lighter_weights_than_quality():
    for vram in (12, 16, 24):
        fast = recommend(vram, target="latency").best
        good = recommend(vram, target="quality").best
        assert fast.memory.weights_gb < good.memory.weights_gb
        assert fast.quality <= good.quality


def test_quality_does_not_decrease_with_more_vram():
    qualities = [recommend(v, target="quality").best.quality for v in (8, 12, 16, 24)]
    assert qualities == sorted(qualities)


def test_14b_never_recommended_at_8gb():
    for target in TARGETS:
        ids = {c.preset_id for c in candidates(8, target=target)}
        assert "qwen2.5-14b" not in ids


def test_ctx_respects_target_cap_and_min_ctx():
    assert all(c.ctx <= 8192 for c in candidates(24, target="latency"))
    assert all(c.ctx >= 16384 for c in candidates(24, target="quality", min_ctx=16384))
    big = recommend(24, target="quality", max_ctx=131072, min_ctx=65536).best
    assert big.ctx >= 65536 and big.preset_id.startswith("llama")


def test_vllm_candidates_use_vllm_quants_and_repos():
    rec = recommend(12, backend="vllm", target="quality")
    for c in [rec.best, *rec.alternatives]:
        assert "vllm" in get_quant(c.quant).backends
        assert c.vllm_repo and c.ollama_tag is None


def test_too_small_budget_returns_no_recommendation():
    rec = recommend(2, target="quality")
    assert rec.best is None and not rec.ok and rec.alternatives == []


def test_invalid_args():
    with pytest.raises(ValueError):
        recommend(0)
    with pytest.raises(ValueError):
        recommend(8, target="cheap")  # type: ignore[arg-type]
