"""Tests for switchplane.llm — model registry and build_llm factory."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from switchplane.llm import (
    DEFAULT_MODEL,
    MODELS,
    ModelInfo,
    build_llm,
    context_window,
)


# ---------------------------------------------------------------------------
# ModelInfo / MODELS registry
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_is_named_tuple(self):
        info = ModelInfo(name="my-model", context_window=100_000)
        assert info.name == "my-model"
        assert info.context_window == 100_000

    def test_tuple_unpacking(self):
        name, window = ModelInfo("x", 42)
        assert name == "x"
        assert window == 42


class TestModelsRegistry:
    def test_default_model_present(self):
        assert DEFAULT_MODEL in MODELS

    def test_all_anthropic_models_have_200k_context(self):
        anthropic = [k for k in MODELS if k.startswith("claude")]
        assert anthropic, "No Anthropic models in registry"
        for key in anthropic:
            assert MODELS[key].context_window == 200_000, key

    def test_all_gemini_models_have_1m_context(self):
        gemini = [k for k in MODELS if k.startswith("gemini")]
        assert gemini, "No Gemini models in registry"
        for key in gemini:
            assert MODELS[key].context_window == 1_000_000, key

    def test_all_openai_models_have_128k_context(self):
        openai = [k for k in MODELS if k.startswith("gpt")]
        assert openai, "No OpenAI models in registry"
        for key in openai:
            assert MODELS[key].context_window == 128_000, key

    def test_model_name_matches_key(self):
        for key, info in MODELS.items():
            assert info.name == key


# ---------------------------------------------------------------------------
# context_window
# ---------------------------------------------------------------------------


class TestContextWindow:
    def test_known_anthropic_model(self):
        assert context_window("claude-sonnet-4-20250514") == 200_000

    def test_known_gemini_model(self):
        assert context_window("gemini-2.0-flash") == 1_000_000

    def test_known_openai_model(self):
        assert context_window("gpt-4o") == 128_000

    def test_unknown_model_falls_back_to_200k(self):
        assert context_window("unknown-future-model") == 200_000

    def test_empty_string_falls_back(self):
        assert context_window("") == 200_000

    def test_returns_int(self):
        assert isinstance(context_window(DEFAULT_MODEL), int)


# ---------------------------------------------------------------------------
# build_llm — helpers
# ---------------------------------------------------------------------------


def _mock_module(cls_name: str) -> tuple[MagicMock, MagicMock]:
    """Return (module_mock, class_mock) with class_mock set as cls_name attr."""
    cls_mock = MagicMock()
    mod_mock = MagicMock()
    setattr(mod_mock, cls_name, cls_mock)
    return mod_mock, cls_mock


# ---------------------------------------------------------------------------
# build_llm — Anthropic (claude-*)
# ---------------------------------------------------------------------------


class TestBuildLlmAnthropic:
    def test_returns_chat_anthropic_instance(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            result = build_llm("claude-sonnet-4-20250514")
        assert result is cls.return_value

    def test_passes_model_name(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            build_llm("claude-opus-4-20250514")
        cls.assert_called_once_with(model="claude-opus-4-20250514")

    def test_passes_api_key_when_provided(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            build_llm("claude-sonnet-4-20250514", api_key="sk-ant-test")
        _, kwargs = cls.call_args
        assert kwargs.get("anthropic_api_key") == "sk-ant-test"

    def test_passes_base_url_when_provided(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            build_llm("claude-sonnet-4-20250514", base_url="https://my-proxy/")
        _, kwargs = cls.call_args
        assert kwargs.get("anthropic_api_url") == "https://my-proxy/"

    def test_omits_api_key_when_none(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            build_llm("claude-sonnet-4-20250514")
        _, kwargs = cls.call_args
        assert "anthropic_api_key" not in kwargs

    def test_omits_base_url_when_none(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            build_llm("claude-sonnet-4-20250514")
        _, kwargs = cls.call_args
        assert "anthropic_api_url" not in kwargs

    def test_import_error_raises_with_hint(self):
        with patch.dict(sys.modules, {"langchain_anthropic": None}):
            with pytest.raises(ImportError, match="langchain-anthropic"):
                build_llm("claude-sonnet-4-20250514")


# ---------------------------------------------------------------------------
# build_llm — Google (gemini-*)
# ---------------------------------------------------------------------------


class TestBuildLlmGemini:
    def test_returns_chat_google_instance(self):
        mod, cls = _mock_module("ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": mod}):
            result = build_llm("gemini-2.0-flash")
        assert result is cls.return_value

    def test_passes_model_name(self):
        mod, cls = _mock_module("ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": mod}):
            build_llm("gemini-2.5-pro")
        cls.assert_called_once_with(model="gemini-2.5-pro")

    def test_passes_api_key_when_provided(self):
        mod, cls = _mock_module("ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": mod}):
            build_llm("gemini-2.0-flash", api_key="gai-key")
        _, kwargs = cls.call_args
        assert kwargs.get("google_api_key") == "gai-key"

    def test_omits_api_key_when_none(self):
        mod, cls = _mock_module("ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": mod}):
            build_llm("gemini-2.0-flash")
        _, kwargs = cls.call_args
        assert "google_api_key" not in kwargs

    def test_base_url_not_forwarded(self):
        """Gemini adapter has no base_url kwarg; build_llm must not pass it."""
        mod, cls = _mock_module("ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": mod}):
            build_llm("gemini-2.0-flash", base_url="https://proxy/")
        _, kwargs = cls.call_args
        assert "base_url" not in kwargs

    def test_import_error_raises_with_hint(self):
        with patch.dict(sys.modules, {"langchain_google_genai": None}):
            with pytest.raises(ImportError, match="langchain-google-genai"):
                build_llm("gemini-2.0-flash")

    def test_import_error_when_not_installed(self):
        # langchain_google_genai is not installed in this environment —
        # removing any cached mock ensures the real absence is exercised.
        saved = sys.modules.pop("langchain_google_genai", None)
        try:
            with pytest.raises(ImportError, match="langchain-google-genai"):
                build_llm("gemini-2.0-flash")
        finally:
            if saved is not None:
                sys.modules["langchain_google_genai"] = saved


# ---------------------------------------------------------------------------
# build_llm — OpenAI (gpt-*)
# ---------------------------------------------------------------------------


class TestBuildLlmOpenAI:
    def test_returns_chat_openai_instance(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            result = build_llm("gpt-4o")
        assert result is cls.return_value

    def test_passes_model_name(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            build_llm("gpt-4o-mini")
        cls.assert_called_once_with(model="gpt-4o-mini")

    def test_passes_api_key_when_provided(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            build_llm("gpt-4o", api_key="sk-openai")
        _, kwargs = cls.call_args
        assert kwargs.get("api_key") == "sk-openai"

    def test_passes_base_url_when_provided(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            build_llm("gpt-4o", base_url="https://oai-proxy/v1")
        _, kwargs = cls.call_args
        assert kwargs.get("base_url") == "https://oai-proxy/v1"

    def test_omits_api_key_when_none(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            build_llm("gpt-4o")
        _, kwargs = cls.call_args
        assert "api_key" not in kwargs

    def test_omits_base_url_when_none(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            build_llm("gpt-4o")
        _, kwargs = cls.call_args
        assert "base_url" not in kwargs

    def test_import_error_raises_with_hint(self):
        with patch.dict(sys.modules, {"langchain_openai": None}):
            with pytest.raises(ImportError, match="langchain-openai"):
                build_llm("gpt-4o")

    def test_import_error_when_not_installed(self):
        saved = sys.modules.pop("langchain_openai", None)
        try:
            with patch.dict(sys.modules, {"langchain_openai": None}):
                with pytest.raises(ImportError, match="langchain-openai"):
                    build_llm("gpt-4o")
        finally:
            if saved is not None:
                sys.modules["langchain_openai"] = saved


# ---------------------------------------------------------------------------
# build_llm — unknown prefix
# ---------------------------------------------------------------------------


class TestBuildLlmUnknownPrefix:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model prefix"):
            build_llm("llama-3-70b")

    def test_error_message_includes_model(self):
        with pytest.raises(ValueError, match="llama-3-70b"):
            build_llm("llama-3-70b")

    def test_error_message_lists_known_prefixes(self):
        with pytest.raises(ValueError, match=r"claude-\*"):
            build_llm("mistral-large")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Unknown model prefix"):
            build_llm("")
