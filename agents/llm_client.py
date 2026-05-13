"""OpenRouter client for BRAG (Block 5+).

Thin wrapper over OpenRouter's OpenAI-compatible chat completions endpoint.
Stdlib-only HTTP (urllib) so we don't pull in another dependency. Handles:

- API key from env (OPENROUTER_API_KEY) or .env
- Retry with exponential backoff on 429/5xx, honoring Retry-After
- JSON-mode requests via ``response_format={"type": "json_object"}``
- App attribution via HTTP-Referer / X-Title headers (optional but
  recommended by OpenRouter for free-tier visibility)

Default model is ``deepseek/deepseek-chat`` — chosen 2026-05-05 for prose
fact extraction (Block 5). Other agents (Planner, Verifier, Refutation,
Generator) will pass their own model slug.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-chat"
DEFAULT_REFERER = "https://github.com/danielwipert/cas.brag"
DEFAULT_TITLE = "cas.brag"

_RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader — only sets keys that are not already in os.environ.
    Idempotent and silent if the file is missing."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


class LLMError(RuntimeError):
    """Raised when the LLM call fails after all retries."""


@dataclass
class LLMResponse:
    content: str
    model: str
    finish_reason: str
    usage: dict[str, int]
    raw: dict[str, Any]


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        referer: str = DEFAULT_REFERER,
        title: str = DEFAULT_TITLE,
        max_retries: int = 5,
        backoff_base: float = 1.5,
        request_timeout: float = 120.0,
    ) -> None:
        if api_key is None:
            _load_dotenv()
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError(
                "OPENROUTER_API_KEY not set. Put it in .env or export it."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.referer = referer
        self.title = title
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.request_timeout = request_timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        response_format: dict[str, str] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra:
            body.update(extra)

        data = self._post_with_retry("/chat/completions", body)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish = choice.get("finish_reason", "")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected OpenRouter response shape: {data!r}") from e
        return LLMResponse(
            content=content,
            model=data.get("model", body["model"]),
            finish_reason=finish,
            usage=dict(data.get("usage") or {}),
            raw=data,
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[Any, LLMResponse]:
        """Convenience: requests JSON mode and parses the response content.

        Returns ``(parsed, response)``. Raises ``LLMError`` if the model
        replies with non-JSON content despite the json_object request."""
        resp = self.chat(
            messages,
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            parsed = json.loads(resp.content)
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Model {resp.model} returned non-JSON in JSON mode: "
                f"{resp.content[:500]!r}"
            ) from e
        return parsed, resp

    def _post_with_retry(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url, data=payload, headers=headers, method="POST"
                )
                with urllib.request.urlopen(
                    req, timeout=self.request_timeout
                ) as resp:
                    raw = resp.read().decode("utf-8")
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as e:
                    # 200 OK but body is not JSON — typically a transient
                    # upstream proxy hiccup (gateway HTML page, truncated
                    # response). Treat like a 5xx and retry with backoff.
                    last_err = LLMError(
                        f"OpenRouter non-JSON 200 body on {path} "
                        f"(decode error: {e}): {raw[:300]!r}"
                    )
                    if attempt == self.max_retries:
                        raise last_err
                    sleep = self._compute_sleep(attempt, retry_after=None)
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:1000]
                except Exception:
                    pass
                last_err = LLMError(
                    f"OpenRouter HTTP {e.code} on {path}: {err_body}"
                )
                if e.code not in _RETRY_STATUSES or attempt == self.max_retries:
                    raise last_err
                sleep = self._compute_sleep(attempt, retry_after=e.headers.get("Retry-After"))
            except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
                # OSError covers ConnectionResetError, BrokenPipeError,
                # ConnectionAbortedError, socket.timeout, etc. — anything
                # the socket layer raises when the upstream connection
                # drops mid-stream. http.client.HTTPException covers
                # IncompleteRead / BadStatusLine on truncated responses.
                last_err = LLMError(f"Network error on {path}: {type(e).__name__}: {e}")
                if attempt == self.max_retries:
                    raise last_err
                sleep = self._compute_sleep(attempt, retry_after=None)
            time.sleep(sleep)
        # unreachable, but mypy-friendly
        raise last_err or LLMError("retry loop exited without resolution")

    def _compute_sleep(self, attempt: int, *, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        # exponential backoff with jitter: base ** attempt + [0, 0.5)
        return (self.backoff_base ** attempt) + random.random() * 0.5
