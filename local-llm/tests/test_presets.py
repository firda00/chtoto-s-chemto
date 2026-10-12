import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from local_llm.presets import (
    VRAM_TIERS,
    ModelPreset,
    get_preset,
    load_preset_file,
    load_presets,
)

PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"
EXPECTED = {"qwen2.5-7b", "qwen2.5-14b", "qwen3-8b", "llama3.1-8b", "llama3.2-3b"}


def test_catalog_contains_required_models():
    presets = load_presets()
    assert set(presets) >= EXPECTED


def test_each_yaml_is_valid_and_id_matches_filename():
    for path in PRESETS_DIR.glob("*.yaml"):
        preset = load_preset_file(path)
        assert preset.id == path.stem


@pytest.mark.parametrize("preset_id", sorted(EXPECTED))
def test_recommendations_cover_all_tiers_and_fit_budget(preset_id):
    """Курируемые рекомендации в YAML должны помещаться в 90% VRAM по формуле оценки."""
    p = get_preset(preset_id)
    for backend in ("ollama", "vllm"):
        tiers = p.recommended[backend]
        assert set(tiers) == set(VRAM_TIERS)
        for tier, rec in tiers.items():
            if rec is None:
                # «не помещается» должно быть правдой даже для самого лёгкого кванта с 4k
                lightest = min(p.quants(backend), key=lambda q: p.estimate(q, 4096).total_gb)
                assert not p.estimate(lightest, 4096).fits(tier, headroom=0.1)
                continue
            est = p.estimate(rec.quant, rec.ctx)
            assert est.fits(tier, headroom=0.1), (preset_id, backend, tier, est)


def test_recommendations_monotonic_in_vram():
    for p in load_presets().values():
        for backend in ("ollama", "vllm"):
            sizes = [
                p.estimate(r.quant, r.ctx).total_gb if r else 0.0
                for _, r in sorted(p.recommended[backend].items())
            ]
            assert sizes == sorted(sizes), (p.id, backend)


def test_ollama_tags_and_vllm_repos():
    p = get_preset("qwen2.5-7b")
    assert p.ollama.tag("q4_K_M") == "qwen2.5:7b-instruct-q4_K_M"
    assert p.vllm.repo("awq") == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert p.vllm.repo("fp8") == "Qwen/Qwen2.5-7B-Instruct"  # динамическое FP8
    assert p.vllm.quants == ["fp16", "awq", "gptq", "fp8"]
    with pytest.raises(KeyError):
        get_preset("llama3.2-3b").vllm.repo("gptq")


def test_get_preset_is_forgiving_and_errors_list_options():
    assert get_preset("Qwen2.5-7B").id == "qwen2.5-7b"
    with pytest.raises(KeyError, match=re.escape("qwen2.5-7b")):
        get_preset("mistral-7b")


def _base_dict() -> dict:
    return yaml.safe_load((PRESETS_DIR / "qwen2.5-7b.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["ollama"].update(quants=["awq"]), "Ollama"),
        (lambda d: d["vllm"]["quant_repos"].update(q4_K_M="x/y"), "vLLM"),
        (lambda d: d["recommended"]["ollama"].update({10: {"quant": "q4_K_M", "ctx": 1}}), "тир"),
        (lambda d: d["recommended"]["vllm"].update({8: {"quant": "q8_0", "ctx": 4096}}), "vllm"),
        (
            lambda d: d["recommended"]["ollama"].update({8: {"quant": "q4_K_M", "ctx": 10**6}}),
            "ctx",
        ),
        (lambda d: d.update(params_b=-1), "params_b"),
    ],
)
def test_validation_rejects_bad_presets(mutate, message):
    data = _base_dict()
    mutate(data)
    with pytest.raises(ValidationError, match=message):
        ModelPreset.model_validate(data)


def test_custom_presets_dir(tmp_path, monkeypatch):
    data = _base_dict()
    data["id"] = "custom-7b"
    (tmp_path / "custom.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("LOCAL_LLM_PRESETS_DIR", str(tmp_path))
    assert list(load_presets()) == ["custom-7b"]


def test_duplicate_ids_rejected(tmp_path):
    data = yaml.safe_dump(_base_dict())
    (tmp_path / "a.yaml").write_text(data, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(data, encoding="utf-8")
    with pytest.raises(ValueError, match="Дублирующийся"):
        load_presets(tmp_path)
