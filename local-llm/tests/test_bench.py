import csv
import json

import httpx
import pytest
import respx

from local_llm.bench import (
    BenchConfig,
    compare_rows,
    load_prompts,
    load_run,
    measure_request,
    percentile,
    run_benchmark,
    save_csv,
    save_json,
    summary_table,
)
from local_llm.mock_server import (
    FakeClock,
    MockProfile,
    make_handler,
    make_mock_transport,
    simulated_profile,
    stream_events,
)

BASE = "http://bench.test/v1"


def _cfg(**kw) -> BenchConfig:
    data = {"base_url": BASE, "model": "qwen2.5:7b-instruct-q4_K_M", "warmup": 0, "max_tokens": 10}
    data.update(kw)
    return BenchConfig(**data)


@respx.mock
def test_streaming_ttft_and_tps_are_deterministic_with_respx():
    """TTFT 200 мс, затем 9 токенов по 50 мс → decode = 9 / 0.45 = 20 tok/s."""
    clock = FakeClock()
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=make_handler(
            clock=clock, profile=MockProfile(ttft_s=0.2, tps=20, max_tokens=10)
        )
    )
    with httpx.Client() as client:
        r = measure_request(client, _cfg(), "Привет", clock=clock)

    assert r.ok and r.error is None
    assert r.ttft_s == pytest.approx(0.2)
    assert r.completion_tokens == 10
    assert r.tokens_from_usage
    assert r.decode_tps == pytest.approx(20.0)
    assert r.total_s == pytest.approx(0.65)
    assert r.e2e_tps == pytest.approx(10 / 0.65, rel=1e-3)

    sent = json.loads(route.calls.last.request.content)
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["max_tokens"] == 10
    assert route.calls.last.request.headers["authorization"] == "Bearer local"


@respx.mock
def test_without_usage_tokens_are_counted_by_chunks():
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        tokens = ["a", "b", "c", "d", "e"]
        body = stream_events(
            "m", tokens, MockProfile(ttft_s=0.1, tps=10), clock=clock, include_usage=False
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    respx.post(f"{BASE}/chat/completions").mock(side_effect=handler)
    with httpx.Client() as client:
        r = measure_request(client, _cfg(), "x", clock=clock)
    assert not r.tokens_from_usage
    assert r.completion_tokens == 5
    assert r.ttft_s == pytest.approx(0.1)
    assert r.decode_tps == pytest.approx(10.0)


@respx.mock
def test_http_error_is_recorded_not_raised():
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    run = run_benchmark(_cfg(), ["a", "b"])
    assert run.summary.n_errors == 2
    assert all("500" in r.error for r in run.results)
    assert run.summary.ttft_ms.p50 is None


def test_connection_error_is_recorded():
    def boom(request):
        raise httpx.ConnectError("connection refused")

    run = run_benchmark(_cfg(), ["a"], transport=httpx.MockTransport(boom))
    assert run.results[0].ok is False
    assert "ConnectError" in run.results[0].error


def test_run_benchmark_prompts_x_repeats_with_warmup():
    clock = FakeClock()
    calls = []
    handler = make_handler(clock=clock, profile=MockProfile(ttft_s=0.1, tps=50, max_tokens=8))

    def counting(request):
        calls.append(request)
        return handler(request)

    run = run_benchmark(
        _cfg(repeats=3, warmup=2, max_tokens=8),
        ["p1", "p2", "p3", "p4"],
        transport=httpx.MockTransport(counting),
        clock=clock,
    )
    assert len(calls) == 2 + 4 * 3  # прогрев не попадает в результаты
    assert len(run.results) == 12
    assert {(r.prompt_idx, r.repeat) for r in run.results} == {
        (i, j) for i in range(4) for j in range(3)
    }
    s = run.summary
    assert s.n_errors == 0
    assert s.ttft_ms.p50 == pytest.approx(100.0)
    assert s.decode_tps.mean == pytest.approx(50.0)
    assert s.total_completion_tokens == 12 * 8
    assert s.latency_s.p95 == pytest.approx(0.1 + 7 / 50, abs=1e-4)


def test_compare_two_configs_quant_simulation():
    """q8_0 в симуляции медленнее q4_K_M пропорционально битам на вес."""
    base = MockProfile(ttft_s=0.2, tps=40, max_tokens=16)
    runs = []
    for model in ("qwen2.5:7b-instruct-q4_K_M", "qwen2.5:7b-instruct-q8_0"):
        clock = FakeClock()
        transport = make_mock_transport(
            clock=clock, profile_for=lambda m: simulated_profile(m, base)
        )
        runs.append(
            run_benchmark(
                _cfg(model=model, max_tokens=16), ["a", "b"], transport=transport, clock=clock
            )
        )
    a, b = runs[0].summary, runs[1].summary
    assert a.decode_tps.mean == pytest.approx(40.0)
    assert b.decode_tps.mean == pytest.approx(40.0 * 4.85 / 8.5, rel=1e-3)
    rows = {r["metric"]: r for r in compare_rows(a, b)}
    assert rows["decode tok/s (mean)"]["winner"] == a.label
    assert rows["decode tok/s (mean)"]["delta_pct"] == pytest.approx(-42.9, abs=0.15)
    assert rows["TTFT p50, мс"]["winner"] == a.label
    assert rows["Ошибок"]["winner"] is None
    table = summary_table(runs)
    assert table.row_count == 2


def test_json_and_csv_roundtrip(tmp_path):
    clock = FakeClock()
    run = run_benchmark(
        _cfg(label="A", api_key="secret"),
        ["x", "y"],
        transport=make_mock_transport(clock=clock),
        clock=clock,
    )
    jp = save_json(run, tmp_path / "out" / "run.json")
    assert "secret" not in jp.read_text(encoding="utf-8")  # ключ не сохраняется
    loaded = load_run(jp)
    assert loaded.summary == run.summary
    cp = save_csv(run, tmp_path / "run.csv")
    rows = list(csv.DictReader(cp.open(encoding="utf-8")))
    assert len(rows) == 2 and rows[0]["label"] == "A"
    assert float(rows[0]["ttft_ms"]) == pytest.approx(200.0)


def test_percentile_and_prompts(tmp_path):
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([0, 10], 95) == pytest.approx(9.5)
    f = tmp_path / "p.txt"
    f.write_text("# comment\n\nfirst\n  second  \n", encoding="utf-8")
    assert load_prompts(f) == ["first", "second"]
    f.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts(f)


def test_project_prompts_file():
    from pathlib import Path

    prompts = load_prompts(Path(__file__).resolve().parents[1] / "data" / "prompts.txt")
    assert 8 <= len(prompts) <= 15
    assert any("а" in p for p in prompts) and any(p.isascii() for p in prompts)
