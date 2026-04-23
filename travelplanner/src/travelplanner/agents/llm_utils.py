from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    api_key_env: str | None
    default_base_url: str | None = None
    base_url_env: str | None = None
    organization_env: str | None = None


OPENAI_COMPATIBLE_PROVIDERS: dict[str, OpenAICompatibleProvider] = {
    "openai": OpenAICompatibleProvider(
        api_key_env="OPENAI_API_KEY",
        organization_env="OPENAI_ORG_ID",
    ),
    "openrouter": OpenAICompatibleProvider(
        api_key_env="OPENROUTER_API_KEY",
        default_base_url="https://openrouter.ai/api/v1",
        base_url_env="OPENROUTER_BASE_URL",
    ),
    "groq": OpenAICompatibleProvider(
        api_key_env="GROQ_API_KEY",
        default_base_url="https://api.groq.com/openai/v1",
        base_url_env="GROQ_BASE_URL",
    ),
    "ollama": OpenAICompatibleProvider(
        api_key_env="OLLAMA_API_KEY",
        default_base_url="http://localhost:11434/v1",
        base_url_env="OLLAMA_BASE_URL",
    ),
}


def _get_env_value(name: str | None) -> str | None:
    if name is None:
        return None

    value = os.getenv(name)
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _parse_model_name(model_name: str) -> tuple[str, str]:
    provider_name, separator, provider_model = model_name.partition(":")
    if separator == "":
        return "openai", model_name.strip()

    provider_name = provider_name.strip().lower()
    provider_model = provider_model.strip()
    if not provider_name or not provider_model:
        raise ValueError(
            "Model names must use either '<model>' or '<provider>:<model>' syntax."
        )

    return provider_name, provider_model


def make_chat_model(*, model_name: str, temperature: float) -> ChatOpenAI:
    provider_name, provider_model = _parse_model_name(model_name)
    provider = OPENAI_COMPATIBLE_PROVIDERS.get(provider_name)
    if provider is None:
        supported_providers = ", ".join(sorted(OPENAI_COMPATIBLE_PROVIDERS))
        raise ValueError(
            f"Unsupported model provider '{provider_name}'. "
            f"Supported providers: {supported_providers}."
        )

    api_key = _get_env_value(provider.api_key_env)
    if (
        provider.api_key_env is not None
        and api_key is None
        and provider_name != "ollama"
    ):
        raise ValueError(
            f"Missing API key for provider '{provider_name}'. "
            f"Set the {provider.api_key_env} environment variable."
        )
    if provider_name == "ollama" and api_key is None:
        api_key = "ollama"

    base_url = _get_env_value(provider.base_url_env) or provider.default_base_url
    organization = _get_env_value(provider.organization_env)
    return ChatOpenAI(
        model=provider_model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=base_url,
        openai_organization=organization,
    )


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
    client = make_chat_model(model_name=model_name, temperature=temperature)
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
