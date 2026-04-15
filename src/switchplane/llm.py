"""LLM integration for Switchplane applications.

Provides a default build_llm that uses native LangChain adapters.
Apps can override this with their own implementation (e.g. for
API gateways).

Requires one or more optional LangChain adapter packages:
  - langchain-anthropic (Claude models)
  - langchain-google-genai (Gemini models)
  - langchain-openai (OpenAI models)
"""

from typing import NamedTuple

from langchain_core.language_models.chat_models import BaseChatModel

from switchplane.config import DEFAULT_MODEL


class ModelInfo(NamedTuple):
    name: str
    context_window: int


# Well-known public models — apps can use these immediately
MODELS: dict[str, ModelInfo] = {
    # Anthropic — 200k context
    "claude-sonnet-4-20250514": ModelInfo("claude-sonnet-4-20250514", 200_000),
    "claude-sonnet-4-5-20250929": ModelInfo("claude-sonnet-4-5-20250929", 200_000),
    "claude-sonnet-4-6": ModelInfo("claude-sonnet-4-6", 200_000),
    "claude-opus-4-20250514": ModelInfo("claude-opus-4-20250514", 200_000),
    "claude-opus-4-6-v1": ModelInfo("claude-opus-4-6-v1", 200_000),
    "claude-haiku-4-5-20251001": ModelInfo("claude-haiku-4-5-20251001", 200_000),
    # Google — 1M context
    "gemini-2.0-flash": ModelInfo("gemini-2.0-flash", 1_000_000),
    "gemini-2.5-pro": ModelInfo("gemini-2.5-pro", 1_000_000),
    "gemini-2.5-flash": ModelInfo("gemini-2.5-flash", 1_000_000),
    # OpenAI — 128k context
    "gpt-4o": ModelInfo("gpt-4o", 128_000),
    "gpt-4o-mini": ModelInfo("gpt-4o-mini", 128_000),
}


def context_window(model: str) -> int:
    """Return the context window size for a model."""
    info = MODELS.get(model)
    return info.context_window if info else 200_000


def build_llm(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Instantiate a LangChain chat model.

    Routes to the appropriate LangChain adapter based on model name prefix:
      - claude-* -> ChatAnthropic (requires langchain-anthropic)
      - gemini-* -> ChatGoogleGenerativeAI (requires langchain-google-genai)
      - gpt-*    -> ChatOpenAI (requires langchain-openai)
    """
    if model.startswith("claude"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Claude models require langchain-anthropic: pip install langchain-anthropic") from None
        kwargs = {"model": model}
        if api_key:
            kwargs["anthropic_api_key"] = api_key
        if base_url:
            kwargs["anthropic_api_url"] = base_url
        return ChatAnthropic(**kwargs)
    elif model.startswith("gemini"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "Gemini models require langchain-google-genai: pip install langchain-google-genai"
            ) from None
        kwargs = {"model": model}
        if api_key:
            kwargs["google_api_key"] = api_key
        return ChatGoogleGenerativeAI(**kwargs)
    elif model.startswith("gpt"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("OpenAI models require langchain-openai: pip install langchain-openai") from None
        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    else:
        raise ValueError(f"Unknown model prefix: {model}. Expected claude-*, gemini-*, or gpt-*")
