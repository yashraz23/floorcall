"""An OpenRouter client (OpenAI-compatible chat API) with a hard spending stop and a response cache.

Only two things in floorcall call an LLM: the D4 training labeller and the prompted-LLM baseline
(DECISIONS.md D-030; provider D-031). Both go through here, so both obey one budget:

- **Cost is OpenRouter's.** Every response carries `usage.cost`, which OpenRouter reports in US
  dollars. The ledger records that figure for each billed call, with the provider that served it.
  A response that lacked one would be recorded at its upper bound (tokens x the max price) and
  marked as estimated, so spend is never under-counted.
- **The stop is $4.75** (`budget_usd * budget_margin`), covering every run, past and future. Every
  request carries `provider.max_price`, so no provider above the configured per-token ceiling is
  used, which makes a call's worst case a bound rather than a guess: a conservative prompt-token
  estimate plus the full output allowance, at the ceiling. That worst case is *reserved* before the
  call, and the call is refused if spent + reserved + worst case would pass the stop.
  Reservations make the check hold under concurrency.
- **Only compliant providers.** `provider.require_parameters` routes only to providers that support
  every parameter sent: strict JSON-schema output, the seed, reasoning effort.
- **Cache.** Responses are cached by the exact request, so a rerun costs nothing and returns the same
  answer. The latency benchmark bypasses it, since a cached answer has no network in it.

The API key comes from Settings (OPENROUTER_API_KEY, a SecretStr). It goes into one request header,
and nowhere else: not into the ledger, the cache, results, or any message or exception.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from floorcall.config import LLMSettings


class BudgetExceededError(RuntimeError):
    """The call could push total LLM spend past the stop."""


class LLMError(RuntimeError):
    """The API failed, or answered with something that is not the requested JSON."""


@dataclass(frozen=True)
class ChatResult:
    data: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float  # request built, sent, answered and parsed; network included
    cached: bool
    provider: str | None


def _estimate_prompt_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    # Deliberately high: about 3 characters a token for English, plus per-message overhead.
    return sum(len(m["content"]) // 3 + 8 for m in messages) + 16


class Ledger:
    def __init__(self, path: Path, settings: LLMSettings) -> None:
        self.settings = settings
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._reserved = 0.0
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS calls (at TEXT, model TEXT, provider TEXT, purpose TEXT, "
                "prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL, "
                "cost_source TEXT, latency_ms REAL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)"
            )

    @property
    def stop_usd(self) -> float:
        return self.settings.budget_usd * self.settings.budget_margin

    def max_price(self, model: str) -> tuple[float, float]:
        if model not in self.settings.max_price_per_million:
            raise BudgetExceededError(
                f"no price ceiling for {model!r}: its spend cannot be bounded"
            )
        return self.settings.max_price_per_million[model]

    def upper_bound(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        p_in, p_out = self.max_price(model)
        return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6

    def _spent(self) -> float:
        return float(self._db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()[0])

    def spent(self) -> float:
        with self._lock:
            return self._spent()

    def reserve(self, model: str, prompt_tokens: int, max_output: int) -> float:
        worst = self.upper_bound(model, prompt_tokens, max_output)
        with self._lock:
            spent = self._spent()
            if spent + self._reserved + worst > self.stop_usd:
                raise BudgetExceededError(
                    f"spent ${spent:.4f} + reserved ${self._reserved:.4f} + this call's worst case "
                    f"${worst:.4f} would pass the ${self.stop_usd:.2f} stop "
                    f"({self.settings.budget_margin:.0%} of the ${self.settings.budget_usd:.2f} cap)"
                )
            self._reserved += worst
        return worst

    def settle(
        self,
        reserved: float,
        *,
        model: str,
        provider: str | None,
        purpose: str,
        prompt_tokens: int,
        completion_tokens: int,
        reported_cost: float | None,
        latency_ms: float,
    ) -> float:
        if reported_cost is not None:
            cost, source = reported_cost, "openrouter"
        else:
            cost, source = self.upper_bound(model, prompt_tokens, completion_tokens), "upper_bound"
        with self._lock:
            self._reserved -= reserved
            self._db.execute(
                "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    model,
                    provider,
                    purpose,
                    prompt_tokens,
                    completion_tokens,
                    cost,
                    source,
                    latency_ms,
                ),
            )
        return cost

    def release(self, reserved: float) -> None:
        with self._lock:
            self._reserved -= reserved

    def cache_get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, key: str, response: Mapping[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(response))
            )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            rows = self._db.execute(
                "SELECT purpose, model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), "
                "SUM(cost_usd), SUM(cost_source = 'upper_bound') FROM calls "
                "GROUP BY purpose, model ORDER BY purpose"
            ).fetchall()
            providers = self._db.execute(
                "SELECT COALESCE(provider, '?'), COUNT(*) FROM calls GROUP BY provider ORDER BY 2 DESC"
            ).fetchall()
        return {
            "budget_usd": self.settings.budget_usd,
            "stop_usd": self.stop_usd,
            "spent_usd": sum(r[5] for r in rows),
            "by_purpose": [
                {
                    "purpose": r[0],
                    "model": r[1],
                    "calls": r[2],
                    "prompt_tokens": r[3],
                    "completion_tokens": r[4],
                    "cost_usd": r[5],
                    "calls_costed_by_upper_bound": r[6],
                }
                for r in rows
            ],
            "calls_by_provider": dict(providers),
        }


class LLMClient:
    def __init__(
        self,
        settings: LLMSettings,
        api_key: str,
        ledger: Ledger,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "floorcall"},
            timeout=settings.timeout_s,
            transport=transport,
        )

    def _payload(
        self, model: str, messages: Sequence[Mapping[str, str]], schema: Mapping[str, Any]
    ) -> dict[str, Any]:
        p_in, p_out = self.ledger.max_price(model)
        return {
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": 0,
            "seed": self.settings.seed,
            "max_tokens": self.settings.max_output_tokens,
            "reasoning": {"effort": self.settings.reasoning_effort, "exclude": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": dict(schema)},
            },
            "provider": {
                "require_parameters": True,
                "max_price": {"prompt": p_in, "completion": p_out},
            },
        }

    def chat_json(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        purpose: str,
        use_cache: bool = True,
    ) -> ChatResult:
        payload = self._payload(model, messages, schema)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if use_cache and (hit := self.ledger.cache_get(key)) is not None:
            return ChatResult(
                hit["data"],
                hit["prompt_tokens"],
                hit["completion_tokens"],
                0.0,
                0.0,
                True,
                hit.get("provider"),
            )

        reserved = self.ledger.reserve(
            model, _estimate_prompt_tokens(messages), self.settings.max_output_tokens
        )
        try:
            t0 = time.perf_counter()
            body = self._post(payload)
            usage = body.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            reported = usage.get("cost")
            reported_cost = float(reported) if isinstance(reported, int | float) else None
            provider = body.get("provider") if isinstance(body.get("provider"), str) else None
            data: Any = None
            parse_error: Exception | None = None
            try:
                data = json.loads(body["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                parse_error = exc
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except BaseException:
            self.ledger.release(reserved)
            raise
        # A billed call is recorded even when its content is unusable: the money is spent.
        cost = self.ledger.settle(
            reserved,
            model=model,
            provider=provider,
            purpose=purpose,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reported_cost=reported_cost,
            latency_ms=latency_ms,
        )
        if parse_error is not None or not isinstance(data, dict):
            raise LLMError(f"{model} ({provider}) did not return the requested JSON: {parse_error}")
        self.ledger.cache_put(
            key,
            {
                "data": data,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "provider": provider,
            },
        )
        return ChatResult(data, prompt_tokens, completion_tokens, cost, latency_ms, False, provider)

    def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                r = self._http.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last = exc
            else:
                if r.status_code == 200:
                    body: dict[str, Any] = r.json()
                    if "error" in body and "choices" not in body:
                        raise LLMError(f"error in a 200 response: {str(body['error'])[:300]}")
                    return body
                if r.status_code not in (408, 409, 429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                last = LLMError(f"HTTP {r.status_code}")
                retry_after = r.headers.get("retry-after")
                if retry_after is not None and attempt < self.settings.max_retries:
                    try:
                        self._sleep(min(60.0, float(retry_after)))
                        continue
                    except ValueError:
                        pass
            if attempt < self.settings.max_retries:
                self._sleep(min(30.0, 2.0**attempt))
        raise LLMError(f"gave up after {self.settings.max_retries + 1} attempts: {last}")

    def close(self) -> None:
        self._http.close()
