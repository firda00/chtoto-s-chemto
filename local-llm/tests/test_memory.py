import math

import pytest

from local_llm.presets import (
    GIB,
    QUANTS,
    estimate_memory_gb,
    kv_cache_gb,
    memory_breakdown,
    weights_gb,
)


def test_weights_formula_matches_hand_calculation():
    # 7 млрд × 4 бит / 8 = 3.5e9 байт; × overhead 1.0
    assert weights_gb(7, 4, overhead=1.0) == pytest.approx(3.5e9 / GIB)
    assert weights_gb(7, 4, overhead=1.2) == pytest.approx(3.5e9 / GIB * 1.2)


def test_kv_cache_llama31_8b_at_8k_is_exactly_one_gib():
    # 2 × 32 слоя × 8 KV-голов × 128 × 8192 токенов × 2 байта = 1 GiB
    assert kv_cache_gb(8192, n_layers=32, n_kv_heads=8, head_dim=128) == pytest.approx(1.0)


def test_kv_cache_scales_linearly_with_ctx_batch_and_bits():
    base = kv_cache_gb(4096, 28, 4)
    assert kv_cache_gb(8192, 28, 4) == pytest.approx(2 * base)
    assert kv_cache_gb(4096, 28, 4, batch=4) == pytest.approx(4 * base)
    assert kv_cache_gb(4096, 28, 4, kv_bits=8) == pytest.approx(base / 2)


def test_estimate_total_is_sum_of_components():
    est = memory_breakdown(8.03, 4.85, 8192, n_layers=32, n_kv_heads=8, runtime_gb=0.5)
    assert est.kv_cache_gb == pytest.approx(1.0, abs=1e-3)
    assert est.total_gb == pytest.approx(est.weights_gb + est.kv_cache_gb + 0.5, abs=2e-3)
    expected = 8.03e9 * 4.85 / 8 / GIB * 1.10 + 1.0 + 0.5
    assert estimate_memory_gb(8.03, 4.85, 8192) == pytest.approx(expected, abs=2e-3)


def test_estimate_accepts_quant_name():
    by_name = estimate_memory_gb(7.61, "q4_K_M", 4096, n_layers=28, n_kv_heads=4)
    by_bits = estimate_memory_gb(7.61, QUANTS["q4_K_M"].bits, 4096, n_layers=28, n_kv_heads=4)
    assert by_name == by_bits


def test_lower_bits_need_less_memory():
    order = ["fp16", "q8_0", "q5_K_M", "q4_K_M"]
    totals = [estimate_memory_gb(8, q, 4096) for q in order]
    assert totals == sorted(totals, reverse=True)
    # fp16 → q4_K_M экономит больше 60% памяти на весах
    assert weights_gb(8, 4.85) / weights_gb(8, 16) < 0.4


def test_fits_with_headroom():
    est = memory_breakdown(8, 4.85, 4096)
    assert est.fits(est.total_gb)
    assert not est.fits(est.total_gb, headroom=0.1)
    assert est.fits(math.ceil(est.total_gb / 0.9) + 0.01, headroom=0.1)


@pytest.mark.parametrize(
    "args",
    [(0, 4, 1024), (7, 0, 1024), (-1, 4, 1024)],
)
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        estimate_memory_gb(*args)


def test_unknown_quant_name_raises():
    with pytest.raises(KeyError, match="q3_bogus"):
        estimate_memory_gb(7, "q3_bogus", 1024)
