from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from ollama import Client as OllamaClient
from pydantic import BaseModel


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


# ─── Provider configuration ──────────────────────────────────────────────────

@dataclass(frozen=True)
class OpenAICompatibleProvider:
    """Configuration for an OpenAI-compatible provider.

    Attributes:
        api_key_env: Name of the environment variable containing the provider API
            key. May be `None` if the provider does not require one.
        default_base_url: Default OpenAI-compatible base URL for the provider.
        base_url_env: Name of the environment variable that overrides the base
            URL.
        organization_env: Name of the environment variable containing the
            provider organization ID, if supported.
    """

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
        default_base_url="https://api.ollama.ai/v1",  # Ollama Cloud API
        base_url_env="OLLAMA_BASE_URL",
    ),
}


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _get_env_value(name: str | None) -> str | None:
    """Returns a normalized environment variable value.

    Args:
        name: Environment variable name to read. If `None`, no lookup is
            performed.

    Returns:
        The stripped environment variable value, or `None` if the variable name
        is `None`, the variable is unset, or the value is empty after stripping.
    """

    if name is None:
        return None

    value = os.getenv(name)
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _parse_model_name(model_name: str) -> tuple[str, str]:
    """Parses a model string into provider and provider model name.

    Args:
        model_name: Model identifier in either `<model>` or
            `<provider>:<model>` format.

    Returns:
        A `(provider_name, provider_model)` tuple. Bare model names default to
        the `openai` provider.

    Raises:
        ValueError: If a provider-prefixed model name is malformed, such as a
            missing provider or missing model segment.
    """

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


# ─── OpenAI-compatible client builder ──────────────────────────────────────

def make_chat_model(*, model_name: str, temperature: float) -> ChatOpenAI:
    """Builds a chat model client from a provider-aware model name.

    Args:
        model_name: Model identifier in either `<model>` or
            `<provider>:<model>` format.
        temperature: Sampling temperature passed to the chat model client.

    Returns:
        A configured `ChatOpenAI` client pointed at the resolved provider,
        model, API key, base URL, and organization settings.

    Raises:
        ValueError: If the provider is not supported.
        ValueError: If the provider requires an API key and the expected
            environment variable is missing.
    """

    provider_name, provider_model = _parse_model_name(model_name)
    provider = OPENAI_COMPATIBLE_PROVIDERS.get(provider_name)
    if provider is None:
        supported_providers = ", ".join(sorted(OPENAI_COMPATIBLE_PROVIDERS))
        raise ValueError(
            f"Unsupported model provider '{provider_name}'. "
            f"Supported providers: {supported_providers}."
        )

    api_key = _get_env_value(provider.api_key_env)
    if provider.api_key_env is not None and api_key is None:
        raise ValueError(
            f"Missing API key for provider '{provider_name}'. "
            f"Set the {provider.api_key_env} environment variable."
        )

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
    """Converts LangChain message content into plain text.

    Args:
        content: Raw message content returned by the chat model. This may be a
            string, a list of structured parts, or another serializable object.

    Returns:
        A single string representation of the content. List payloads are
        flattened by concatenating their text parts.
    """

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
    """Extracts and validates a JSON payload from model output text.

    Args:
        text: Raw model output that should contain JSON, optionally wrapped in a
            fenced code block.

    Returns:
        The JSON payload as a string, with outer code fences removed if present.

    Raises:
        json.JSONDecodeError: If the extracted text is not valid JSON.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    json.loads(stripped)
    return stripped


# ─── Ollama Cloud helpers (identical to hotel_search_agent) ──────────────────

def _call_llm(system_prompt: str, user_prompt: str, model_name: str, temperature: float = 0.0) -> str:
    """Call LLM with explicit provider branching.

    Provider dispatch (matches ``_parse_model_name`` / ``make_chat_model``):
    -  ``openrouter:*``   → LangChain ChatOpenAI (OpenRouter API)
    -  ``ollama:*``       → native ``ollama.Client`` (cloud or local)
    -  bare name / ``openai:*`` → LangChain ChatOpenAI (OpenAI API)

    Bare model names (no prefix) default to **OpenAI**, as documented in AGENTS.md.
    Use the ``ollama:`` prefix when you intend to call an Ollama model.

    Args:
        system_prompt: System message.
        user_prompt: User message.
        model_name: Model identifier. Examples:
            - ``openrouter:anthropic/claude-3.5-sonnet``
            - ``ollama:nemotron-3-super``
            - ``ollama:nemotron-3-super``
            - ``gpt-5-mini`` (bare → OpenAI)
        temperature: Sampling temperature.

    Returns:
        Raw response text from the LLM.

    Raises:
        ValueError: If required environment variables are missing.
    """
    if model_name.startswith("openrouter:"):
        # ── OpenRouter path (LangChain) ──────────────────────────────────
        llm = make_chat_model(model_name=model_name, temperature=temperature)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        return response.content

    if model_name.startswith("ollama:"):
        # ── Ollama path (native client) ──────────────────────────────────
        provider_model = model_name[len("ollama:"):].strip()
        api_key = _get_env_value("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError(
                "OLLAMA_API_KEY environment variable is required for Ollama models. "
                "Get an API key at https://ollama.com/api_keys"
            )

        host = _get_env_value("OLLAMA_BASE_URL") or "https://ollama.com"
        client = OllamaClient(
            host=host,
            headers={"Authorization": f"Bearer {api_key}"},
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = client.chat(model=provider_model, messages=messages, stream=False)
        return response["message"]["content"]

    # ── Default: OpenAI (or any other OpenAI-compatible provider) ──────
    llm = make_chat_model(model_name=model_name, temperature=temperature)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content


# ─── Main structured-model entrypoint ─────────────────────────────────────────

def invoke_structured_model(
    *,
    model_name: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    response_model: type[ResponseModelT],
) -> tuple[ResponseModelT, str, str]:
    """Invokes a chat model and parses its JSON response into a Pydantic model.

    Provider dispatch:
    * ``openrouter:*``          → LangChain ChatOpenAI (team credits)
    * ``openai:*``, ``groq:*``  → LangChain ChatOpenAI
    * everything else           → native ``ollama.Client`` (cloud / local)

    Ollama requires ``OLLAMA_API_KEY`` set in the environment.  If unset, a
    ``ValueError`` is raised.

    Args:
        model_name: Model identifier.  Examples::
            - ``openrouter:anthropic/claude-3.5-sonnet``
            - ``openai:gpt-5-mini``
            - ``nemotron-3-super``
            - ``mistral``
        temperature: Sampling temperature for the model call.
        system_prompt: System message sent to the model.
        user_prompt: User message sent to the model.
        response_model: Pydantic model class used to validate the JSON payload
            returned by the model.

    Returns:
        A tuple of (validated model instance, user_prompt sent, raw text from LLM).

    Raises:
        ValueError: If required credentials are missing.
        json.JSONDecodeError: If the model response is not valid JSON after
            optional code-fence removal.
        pydantic.ValidationError: If the JSON payload does not match
            ``response_model``.
    """

    raw_content = _call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        temperature=temperature,
    )

    json_payload = _extract_json_payload(raw_content)
    structured_output = response_model.model_validate_json(json_payload)
    return structured_output, user_prompt, raw_content
