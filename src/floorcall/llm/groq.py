"""A small Groq client (OpenAI-compatible chat API) with a hard spending cap and a response cache.

Only two things in floorcall call an LLM: the D4 training labeller and the prompted-LLM baseline
(DECISIONS.md D-030). Both go through here, so both obey one budget:

- **Ledger.** Every billed call is recorded in SQLite with its token usage and cost, priced from
  `LLMSettings.prices_per_million`. The cap (`budget_usd`, Yash's $5) covers every run, past and
  future. Before a call, its worst case (a conservative prompt-token estimate plus the full output
  allowance) is *reserved*; the call is refused if spent + reserved + worst case would pass
  `budget_usd * budget_margin`. Reservations make the check hold under concurrency. A model with
  no known price is refused, because its spend could not be capped.
- **Cache.** Responses are cached by the exact request. A rerun costs nothing and returns the
  same answer. The latency benchmark bypasses the cache, since a cached answer has no network in it.
- **Output.** Strict JSON-schema structured outputs (supported on the gpt-oss models), parsed and
  returned as a dict. Groq returns no log-probabilities (its API: "not yet supported by any of our
  models"), so any probability an LLM gives here is one it *states*.

The API key comes from Settings (GROQ_API_KEY, SecretStr) and is never logged.
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
    """The call could push total Groq spend past the cap."""


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
                "CREATE TABLE IF NOT EXISTS calls (at TEXT, model TEXT, purpose TEXT, "
                "prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL, latency_ms REAL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)"
            )

    def price(self, model: str) -> tuple[float, float]:
        if model not in self.settings.prices_per_million:
            raise BudgetExceededError(f"no price for {model!r}: its spend cannot be capped")
        return self.settings.prices_per_million[model]

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        p_in, p_out = self.price(model)
        return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6

    def spent(self) -> float:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()
        return float(row[0])

    def reserve(self, model: str, prompt_tokens: int, max_output: int) -> float:
        worst = self.cost(model, prompt_tokens, max_output)
        limit = self.settings.budget_usd * self.settings.budget_margin
        with self._lock:
            spent = float(
                self._db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()[0]
            )
            if spent + self._reserved + worst > limit:
                raise BudgetExceededError(
                    f"spent ${spent:.4f} + reserved ${self._reserved:.4f} + this call's worst case "
                    f"${worst:.4f} would pass ${limit:.2f} ({self.settings.budget_margin:.0%} of the "
                    f"${self.settings.budget_usd:.2f} cap)"
                )
            self._reserved += worst
        return worst

    def settle(
        self,
        reserved: float,
        model: str,
        purpose: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> float:
        cost = self.cost(model, prompt_tokens, completion_tokens)
        with self._lock:
            self._reserved -= reserved
            self._db.execute(
                "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    model,
                    purpose,
                    prompt_tokens,
                    completion_tokens,
                    cost,
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
                "SUM(cost_usd) FROM calls GROUP BY purpose, model ORDER BY purpose"
            ).fetchall()
        return {
            "budget_usd": self.settings.budget_usd,
            "spent_usd": sum(r[5] for r in rows),
            "by_purpose": [
                {
                    "purpose": r[0],
                    "model": r[1],
                    "calls": r[2],
                    "prompt_tokens": r[3],
                    "completion_tokens": r[4],
                    "cost_usd": r[5],
                }
                for r in rows
            ],
        }


class GroqClient:
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
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=settings.timeout_s,
            transport=transport,
        )

    def _payload(
        self, model: str, messages: Sequence[Mapping[str, str]], schema: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": 0,
            "seed": self.settings.seed,
            "max_completion_tokens": self.settings.max_output_tokens,
            "reasoning_effort": self.settings.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": dict(schema)},
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
                hit["data"], hit["prompt_tokens"], hit["completion_tokens"], 0.0, 0.0, True
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
            try:
                data = json.loads(body["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                data = None
                parse_error: Exception | None = exc
            else:
                parse_error = None
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except BaseException:
            self.ledger.release(reserved)
            raise
        # A billed call is recorded even when its content is unusable: the money is spent.
        cost = self.ledger.settle(
            reserved, model, purpose, prompt_tokens, completion_tokens, latency_ms
        )
        if parse_error is not None or not isinstance(data, dict):
            raise LLMError(f"{model} did not return the requested JSON: {parse_error}")
        self.ledger.cache_put(
            key,
            {"data": data, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        )
        return ChatResult(data, prompt_tokens, completion_tokens, cost, latency_ms, False)

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
