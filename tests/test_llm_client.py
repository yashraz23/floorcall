"""The LLM client's money rules (OpenRouter), against a mock transport. No network, no key, no spend."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from floorcall.config import LLMSettings, Settings
from floorcall.llm.client import BudgetExceededError, Ledger, LLMClient, LLMError

MODEL = "openai/gpt-oss-20b"  # ceiling $0.075 in / $0.30 out per million
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}
MSGS = [{"role": "user", "content": "is this an escalation?"}]


def reply(
    content: str = '{"label": "no"}',
    prompt: int = 1000,
    completion: int = 100,
    cost: float | None = 0.00002,
    provider: str | None = "DeepInfra",
) -> dict[str, Any]:
    usage: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion}
    if cost is not None:
        usage["cost"] = cost
    body: dict[str, Any] = {"choices": [{"message": {"content": content}}], "usage": usage}
    if provider is not None:
        body["provider"] = provider
    return body


class Server:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(request.headers)
        return self.responses.pop(0)


def client(
    tmp_path: Path, server: Server, **overrides: Any
) -> tuple[LLMClient, Ledger, list[float]]:
    settings = LLMSettings(**overrides)
    ledger = Ledger(tmp_path / "ledger.sqlite", settings)
    sleeps: list[float] = []
    c = LLMClient(
        settings, "test-key", ledger, transport=httpx.MockTransport(server), sleep=sleeps.append
    )
    return c, ledger, sleeps


def test_the_charge_is_openrouters_reported_cost(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=reply(cost=0.0000123)))
    c, ledger, _ = client(tmp_path, server)
    out = c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert out.cost_usd == pytest.approx(0.0000123)
    assert out.provider == "DeepInfra"
    assert ledger.spent() == pytest.approx(0.0000123)
    s = ledger.summary()
    assert s["calls_by_provider"] == {"DeepInfra": 1}
    assert s["by_purpose"][0]["calls_costed_by_upper_bound"] == 0


def test_a_missing_cost_is_charged_at_its_upper_bound(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=reply(cost=None)))
    c, ledger, _ = client(tmp_path, server)
    out = c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert out.cost_usd == pytest.approx((1000 * 0.075 + 100 * 0.30) / 1e6)
    assert ledger.summary()["by_purpose"][0]["calls_costed_by_upper_bound"] == 1


def test_every_request_caps_price_and_requires_compliant_providers(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=reply()))
    c, _, _ = client(tmp_path, server, provider_pins={})  # the unpinned request shape
    c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    body = server.requests[0]
    assert body["model"] == "openai/gpt-oss-20b"  # never a ":batch" variant
    assert body["provider"] == {
        "require_parameters": True,
        "max_price": {"prompt": 0.075, "completion": 0.30},
    }
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["reasoning"] == {"effort": "low", "exclude": True}
    assert body["temperature"] == 0 and body["seed"] == LLMSettings().seed
    assert server.headers[0]["authorization"] == "Bearer test-key"


def test_a_pinned_model_goes_to_one_endpoint_with_no_fallback(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=reply(provider="Crusoe")))
    c, _, _ = client(tmp_path, server)
    out = c.chat_json(model="openai/gpt-oss-120b", messages=MSGS, schema=SCHEMA, purpose="t")
    assert out.provider == "Crusoe"
    assert server.requests[0]["provider"] == {
        "require_parameters": True,
        "max_price": {"prompt": 0.15, "completion": 0.60},
        "only": ["crusoe/bf16"],
        "order": ["crusoe/bf16"],
        "allow_fallbacks": False,
        "quantizations": ["bf16"],
    }


@pytest.mark.parametrize("served_by", ["DekaLLM", None])
def test_a_pinned_model_served_elsewhere_is_rejected_but_charged(
    tmp_path: Path, served_by: str | None
) -> None:
    server = Server(
        httpx.Response(200, json=reply(provider=served_by, cost=0.00004)),
        httpx.Response(200, json=reply(provider="Crusoe")),
    )
    c, ledger, _ = client(tmp_path, server)
    kw: dict[str, Any] = {"model": "openai/gpt-oss-120b", "messages": MSGS, "schema": SCHEMA}
    with pytest.raises(LLMError, match="not the pinned crusoe/bf16"):
        c.chat_json(**kw, purpose="t")
    assert ledger.spent() == pytest.approx(0.00004)
    again = c.chat_json(**kw, purpose="t")  # the rejected answer was not cached
    assert not again.cached and len(server.requests) == 2


def test_responses_are_cached(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=reply()))
    c, ledger, _ = client(tmp_path, server)
    first = c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    again = c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert again.cached and again.cost_usd == 0.0 and again.data == first.data
    assert len(server.requests) == 1
    assert ledger.spent() == pytest.approx(first.cost_usd)


def test_bypassing_the_cache_calls_again(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=reply()), httpx.Response(200, json=reply()))
    c, _, _ = client(tmp_path, server)
    for _ in range(2):
        c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t", use_cache=False)
    assert len(server.requests) == 2


def test_the_stop_is_475_of_5_and_refuses_before_any_request(tmp_path: Path) -> None:
    server = Server()
    _, ledger, _ = client(tmp_path, server)
    assert ledger.stop_usd == pytest.approx(4.75)
    c2, ledger2, _ = client(tmp_path / "small", server, budget_usd=0.0001)
    with pytest.raises(BudgetExceededError, match="stop"):
        c2.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert server.requests == []
    assert ledger2.spent() == 0.0


def test_spend_carried_over_from_earlier_runs_counts(tmp_path: Path) -> None:
    settings = LLMSettings(budget_usd=1.0, budget_margin=1.0)
    path = tmp_path / "l.sqlite"
    first = Ledger(path, settings)
    first.settle(
        0.0,
        model=MODEL,
        provider="x",
        purpose="earlier",
        prompt_tokens=0,
        completion_tokens=0,
        reported_cost=0.99,
        latency_ms=1.0,
    )
    later = Ledger(path, settings)  # a new run, the same ledger file
    with pytest.raises(BudgetExceededError):
        later.reserve(MODEL, 100_000, 100_000)  # worst case $0.0375 on top of $0.99


def test_a_model_without_a_price_ceiling_is_refused(tmp_path: Path) -> None:
    c, _, _ = client(tmp_path, Server())
    with pytest.raises(BudgetExceededError, match="no price ceiling"):
        c.chat_json(model="openai/gpt-oss-20b:batch", messages=MSGS, schema=SCHEMA, purpose="t")


def test_reservations_hold_the_stop_under_concurrency(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite", LLMSettings(budget_usd=1.0, budget_margin=1.0))
    ledger.reserve(MODEL, 1_000_000, 1_000_000)  # worst case $0.375
    ledger.reserve(MODEL, 1_000_000, 1_000_000)  # $0.75 in flight
    with pytest.raises(BudgetExceededError):
        ledger.reserve(MODEL, 1_000_000, 1_000_000)


def test_rate_limits_are_retried_honouring_retry_after(tmp_path: Path) -> None:
    server = Server(
        httpx.Response(429, headers={"retry-after": "7"}),
        httpx.Response(503),
        httpx.Response(200, json=reply()),
    )
    c, ledger, sleeps = client(tmp_path, server)
    c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert sleeps[0] == 7.0 and len(server.requests) == 3
    assert ledger.summary()["by_purpose"][0]["calls"] == 1


def test_a_client_error_is_not_retried_and_costs_nothing(tmp_path: Path) -> None:
    server = Server(httpx.Response(400, json={"error": {"message": "bad request"}}))
    c, ledger, _ = client(tmp_path, server)
    with pytest.raises(LLMError, match="400"):
        c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert ledger.spent() == 0.0
    ledger.reserve(MODEL, 100, 100)  # the failed call's reservation was released


def test_unparseable_content_is_still_billed(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=reply(content="not json", cost=0.00005)))
    c, ledger, _ = client(tmp_path, server)
    with pytest.raises(LLMError, match="JSON"):
        c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert ledger.spent() == pytest.approx(0.00005)


def test_the_key_never_reaches_a_settings_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    # run.json stores settings.model_dump(): a plain-string key would be written into every run
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-this-must-not-leak")
    dumped = json.dumps(Settings().model_dump(mode="json"))
    assert "sk-or-this-must-not-leak" not in dumped
    assert Settings().openrouter_api_key is not None
