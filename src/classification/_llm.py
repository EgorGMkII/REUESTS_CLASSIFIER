from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

LLMCallable = Callable[[str], Any]


def invoke_default_llm(prompt: str) -> Any:
    # Import lazily so importing and testing classification never creates a client.
    from llm_module import get_langchain_openai_chat_model

    return get_langchain_openai_chat_model().invoke(prompt)


def invoke_preprocessor_llm(prompt: str) -> Any:
    # Import lazily so importing and testing classification never creates a client.
    from llm_module import get_preprocessor_chat_model

    return get_preprocessor_chat_model().invoke(prompt)


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
