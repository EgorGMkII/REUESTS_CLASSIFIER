from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any

LLMCallable = Callable[[str], Any]

_LLM_CALL_LOCK = threading.Lock()
_LAST_LLM_CALL_AT = 0.0


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return max(0, int(value))
    except ValueError:
        return default


def _is_rate_limit_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    text = f"{type(error).__name__}: {error}".lower()
    return "429" in text and ("rate" in text or "too many requests" in text)


def _wait_for_llm_slot() -> None:
    global _LAST_LLM_CALL_AT

    interval = _env_float("HYDRA_LLM_MIN_INTERVAL_SECONDS", 0.0)
    if interval <= 0:
        return

    with _LLM_CALL_LOCK:
        now = time.monotonic()
        wait_seconds = _LAST_LLM_CALL_AT + interval - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _LAST_LLM_CALL_AT = time.monotonic()


def _invoke_with_controls(invoker: Callable[[str], Any], prompt: str) -> Any:
    max_retries = _env_int("HYDRA_LLM_MAX_RETRIES", 0)
    retry_seconds = _env_float("HYDRA_LLM_RETRY_SECONDS", 65.0)

    for attempt in range(max_retries + 1):
        _wait_for_llm_slot()
        try:
            return invoker(prompt)
        except Exception as error:
            if not _is_rate_limit_error(error) or attempt >= max_retries:
                raise
            time.sleep(retry_seconds)
    raise RuntimeError("unreachable LLM retry state")


def invoke_default_llm(prompt: str) -> Any:
    # Import lazily so importing and testing classification never creates a client.
    from llm_module import get_langchain_openai_chat_model

    return _invoke_with_controls(get_langchain_openai_chat_model().invoke, prompt)


def invoke_preprocessor_llm(prompt: str) -> Any:
    # Import lazily so importing and testing classification never creates a client.
    from llm_module import get_preprocessor_chat_model

    return _invoke_with_controls(get_preprocessor_chat_model().invoke, prompt)


def response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks)
    if isinstance(response, dict):
        return json.dumps(response, ensure_ascii=False)
    return ""


def parse_json_response(response: Any) -> dict[str, Any]:
    text = response_text(response).strip()
    if not text:
        raise ValueError("LLM returned an empty response")
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        text = fenced.group(1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def valid_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Confidence must be numeric")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1")
    return confidence
