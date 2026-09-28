"""The Groq client's money rules, against a mock transport. No network, no key, no spend."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from floorcall.config import LLMSettings, Settings
from floorcall.llm.groq import BudgetExceededError, GroqClient, Ledger, LLMError

MODEL = "openai/gpt-oss-20b"  # $0.075 in / $0.30 out per million
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}
MSGS = [{"role": "user", "content": "is this an escalation?"}]


def reply(
    content: str = '{"label": "no"}', prompt: int = 1000, completion: int = 100
) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


class Server:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return self.responses.pop(0)


def client(
    tmp_path: Path, server: Server, **overrides: Any
) -> tuple[GroqClient, Ledger, list[float]]:
    settings = LLMSettings(**overrides)
    ledger = Ledger(tmp_path / "ledger.sqlite", settings)
    sleeps: list[float] = []
    c = GroqClient(
        settings, "test-key", ledger, transport=httpx.MockTransport(server), sleep=sleeps.append
    )
    return c, ledger, sleeps


def test_a_call_is_priced_from_its_usage_and_cached(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=reply()))
    c, ledger, _ = client(tmp_path, server)
    first = c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert first.data == {"label": "no"} and not first.cached
    assert first.cost_usd == pytest.approx((1000 * 0.075 + 100 * 0.30) / 1e6)
    again = c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert again.cached and again.cost_usd == 0.0
    assert len(server.requests) == 1  # the cache answered the second call
    assert ledger.spent() == pytest.approx(first.cost_usd)


def test_the_request_asks_for_strict_json_and_is_deterministic(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=reply()))
    c, _, _ = client(tmp_path, server)
    c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    body = server.requests[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["temperature"] == 0 and body["seed"] == LLMSettings().seed


def test_bypassing_the_cache_calls_again(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=reply()), httpx.Response(200, json=reply()))
    c, ledger, _ = client(tmp_path, server)
    c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t", use_cache=False)
    c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t", use_cache=False)
    assert len(server.requests) == 2
    assert ledger.summary()["by_purpose"][0]["calls"] == 2


def test_the_cap_refuses_before_any_request_is_sent(tmp_path: Path) -> None:
    server = Server()
    c, ledger, _ = client(tmp_path, server, budget_usd=0.0001)
    with pytest.raises(BudgetExceededError, match="cap"):
        c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert server.requests == []
    assert ledger.spent() == 0.0


def test_a_model_without_a_price_is_refused(tmp_path: Path) -> None:
    c, _, _ = client(tmp_path, Server())
    with pytest.raises(BudgetExceededError, match="no price"):
        c.chat_json(model="some/unpriced-model", messages=MSGS, schema=SCHEMA, purpose="t")


def test_reservations_hold_the_cap_under_concurrency(tmp_path: Path) -> None:
    settings = LLMSettings(budget_usd=1.0, budget_margin=1.0)
    ledger = Ledger(tmp_path / "l.sqlite", settings)
    ledger.reserve(MODEL, 1_000_000, 1_000_000)  # worst case $0.375
    ledger.reserve(MODEL, 1_000_000, 1_000_000)  # $0.75 reserved in flight
    with pytest.raises(BudgetExceededError):
        ledger.reserve(MODEL, 1_000_000, 1_000_000)  # would reach $1.125


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
    server = Server(httpx.Response(200, json=reply(content="not json")))
    c, ledger, _ = client(tmp_path, server)
    with pytest.raises(LLMError, match="JSON"):
        c.chat_json(model=MODEL, messages=MSGS, schema=SCHEMA, purpose="t")
    assert ledger.spent() > 0


def test_the_key_never_reaches_a_settings_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    # run.json stores settings.model_dump(): a plain-string key would be written into every run
    monkeypatch.setenv("GROQ_API_KEY", "gsk_this_must_not_leak")
    dumped = json.dumps(Settings().model_dump(mode="json"))
    assert "gsk_this_must_not_leak" not in dumped
    assert Settings().groq_api_key is not None
