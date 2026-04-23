from __future__ import annotations

import json
from typing import TypeVar

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


def _coerce_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text_parts.append(str(part.get("text", "")))
                continue
            text_parts.append(str(part))
        return "".join(text_parts)
    return str(content)


def _extract_json_payload(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    json.loads(stripped)
    return stripped


def invoke_structured_model(
    *,
    model_name: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    response_model: type[ResponseModelT],
) -> tuple[ResponseModelT, str, str]:
    client = ChatOpenAI(model=model_name, temperature=temperature)
    response = client.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    raw_content = _coerce_message_content(response.content).strip()
    json_payload = _extract_json_payload(raw_content)
    structured_output = response_model.model_validate_json(json_payload)
    return structured_output, user_prompt, raw_content
