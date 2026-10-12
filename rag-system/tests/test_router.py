from rag_system.llm import MockLLM
from rag_system.router import KeywordRouter, LLMRouter

COLLECTIONS = ["company", "faq", "policies", "pricing", "schedule", "services"]


def test_llm_router_uses_scripted_decision() -> None:
    llm = MockLLM(['{"collection": "pricing", "tools": [], "reason": "price question"}'])
    route = LLMRouter(llm).route("Сколько стоит Flex?", COLLECTIONS)
    assert route.collection == "pricing"
    assert route.tools == []
    assert route.router == "llm"
    call = llm.calls[0]
    assert call["json_schema"] is not None
    assert "pricing" in call["prompt"]


def test_llm_router_all_and_tool_filtering() -> None:
    llm = MockLLM([{"collection": "all", "tools": ["calculator", "rm_rf"], "reason": ""}])
    route = LLMRouter(llm).route("120000 * 3", COLLECTIONS)
    assert route.collection is None
    assert route.tools == ["calculator"]  # неизвестный инструмент отброшен


def test_llm_router_falls_back_on_invalid_output() -> None:
    # Пустой JSON (поведение MockLLM без сценария) не проходит строгую схему → keyword fallback
    route = LLMRouter(MockLLM()).route("Можно ли прийти с собакой? Какие правила?", COLLECTIONS)
    assert route.router == "keyword-fallback"
    assert route.collection == "policies"


def test_llm_router_falls_back_on_unknown_collection() -> None:
    llm = MockLLM(['{"collection": "weather", "tools": [], "reason": "?"}'])
    route = LLMRouter(llm).route("Какая скорость Wi-Fi?", COLLECTIONS)
    assert route.router == "keyword-fallback"
    assert "unknown collection" in route.reason
    assert route.collection == "faq"


def test_keyword_router_decisions() -> None:
    r = KeywordRouter()
    assert r.route("Сколько стоит переговорная?", COLLECTIONS).collection == "pricing"
    assert r.route("What is your refund policy?", COLLECTIONS).collection == "policies"
    assert r.route("Когда проходит нетворкинг?", COLLECTIONS).collection == "schedule"
    assert r.route("Hello there", COLLECTIONS).collection is None  # ничего не совпало → все
    assert r.route("Сколько будет 2 + 2?", COLLECTIONS).tools == ["calculator"]
    assert r.route("Какая сегодня дата?", COLLECTIONS).tools == ["get_date"]
