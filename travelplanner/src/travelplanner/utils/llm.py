from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from travelplanner.utils.runtime_monitor import record_llm_call


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


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
        default_base_url="http://localhost:11434/v1",
        base_url_env="OLLAMA_BASE_URL",
    ),
}


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


def extract_token_usage(response: object) -> tuple[int, int]:
    """Extract `(input_tokens, output_tokens)` from a LangChain response."""

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        tokens_in = int(
            usage_metadata.get("input_tokens")
            or usage_metadata.get("prompt_tokens")
            or 0
        )
        tokens_out = int(
            usage_metadata.get("output_tokens")
            or usage_metadata.get("completion_tokens")
            or 0
        )
        return tokens_in, tokens_out

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage", {})
        if isinstance(token_usage, dict):
            tokens_in = int(token_usage.get("prompt_tokens") or 0)
            tokens_out = int(token_usage.get("completion_tokens") or 0)
            return tokens_in, tokens_out

    return 0, 0


def invoke_structured_model(
    *,
    model_name: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    response_model: type[ResponseModelT],
) -> tuple[ResponseModelT, str, str]:
    """Invokes a chat model and parses its JSON response into a Pydantic model.

    Args:
        model_name: Model identifier in either `<model>` or
            `<provider>:<model>` format.
        temperature: Sampling temperature for the model call.
        system_prompt: System message sent to the model.
        user_prompt: User message sent to the model.
        response_model: Pydantic model class used to validate the JSON payload
            returned by the model.

    Returns:
        A tuple containing:

        - The validated `response_model` instance.
        - The exact `user_prompt` string that was sent.
        - The raw text content returned by the model before JSON parsing.

    Raises:
        ValueError: If the provider is unsupported or a required API key is
            missing.
        json.JSONDecodeError: If the model response is not valid JSON after
            optional code-fence removal.
        pydantic.ValidationError: If the JSON payload does not match
            `response_model`.
    """

    client = make_chat_model(model_name=model_name, temperature=temperature)
    response = client.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    tokens_in, tokens_out = extract_token_usage(response)
    record_llm_call(
        model_name=model_name,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
    raw_content = _coerce_message_content(response.content).strip()
    json_payload = _extract_json_payload(raw_content)
    structured_output = response_model.model_validate_json(json_payload)
    return structured_output, user_prompt, raw_content
