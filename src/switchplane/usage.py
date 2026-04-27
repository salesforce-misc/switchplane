"""LLM usage accounting helpers."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field


class ModelPricing(NamedTuple):
    """USD prices per one million tokens."""

    input_per_million: float
    output_per_million: float


# Public list-price approximations. Keep this deliberately small; unknown
# models still produce token records, just without an estimated dollar cost.
MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4-20250514": ModelPricing(3.0, 15.0),
    "claude-sonnet-4-5-20250929": ModelPricing(3.0, 15.0),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0),
    "claude-opus-4-20250514": ModelPricing(15.0, 75.0),
    "claude-opus-4-6-v1": ModelPricing(15.0, 75.0),
    "claude-haiku-4-5-20251001": ModelPricing(1.0, 5.0),
    "gpt-4o": ModelPricing(2.5, 10.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gemini-2.0-flash": ModelPricing(0.10, 0.40),
    "gemini-2.5-flash": ModelPricing(0.30, 2.50),
    "gemini-2.5-pro": ModelPricing(1.25, 10.0),
}


class LLMUsageRecord(BaseModel):
    """Structured accounting record for a single LLM call."""

    model_config = ConfigDict(str_strip_whitespace=True)

    task_id: str
    model: str
    node_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    estimated_raw_prompt_tokens: int | None = None
    estimated_tokens_saved: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def estimate_text_tokens(text: str) -> int:
    """Cheap text token estimate for before/after comparisons.

    Providers expose exact usage after a call, but savings estimates often need
    a pre-call approximation. Four characters per token is a conservative,
    model-agnostic rule of thumb for English/code-like text.
    """

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate USD cost for a model if pricing is known."""

    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    cost = (prompt_tokens / 1_000_000 * pricing.input_per_million) + (
        completion_tokens / 1_000_000 * pricing.output_per_million
    )
    return round(cost, 6)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_token_counts(response: Any) -> tuple[int | None, int | None, int | None]:
    """Extract provider-reported token counts from common LangChain responses."""

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt = _coerce_int(usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens"))
        completion = _coerce_int(
            usage_metadata.get("output_tokens")
            or usage_metadata.get("completion_tokens")
            or usage_metadata.get("generated_tokens")
        )
        total = _coerce_int(usage_metadata.get("total_tokens"))
        if prompt is not None or completion is not None or total is not None:
            return prompt, completion, total

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if isinstance(token_usage, dict):
            prompt = _coerce_int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens"))
            completion = _coerce_int(token_usage.get("completion_tokens") or token_usage.get("output_tokens"))
            total = _coerce_int(token_usage.get("total_tokens"))
            if prompt is not None or completion is not None or total is not None:
                return prompt, completion, total

    return None, None, None


def llm_usage_from_response(
    response: Any,
    *,
    task_id: str,
    model: str,
    node_name: str,
    fallback_prompt_text: str = "",
    fallback_completion_text: str = "",
    estimated_raw_prompt_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> LLMUsageRecord:
    """Build an ``LLMUsageRecord`` from a LangChain response."""

    prompt_tokens, completion_tokens, total_tokens = extract_token_counts(response)
    token_source = "provider"

    if prompt_tokens is None:
        prompt_tokens = estimate_text_tokens(fallback_prompt_text)
        token_source = "estimated"
    if completion_tokens is None:
        completion_tokens = estimate_text_tokens(fallback_completion_text)
        token_source = "estimated"
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    estimated_tokens_saved = None
    if estimated_raw_prompt_tokens is not None:
        estimated_tokens_saved = max(0, estimated_raw_prompt_tokens - prompt_tokens)

    meta = dict(metadata or {})
    meta.setdefault("token_source", token_source)

    return LLMUsageRecord(
        task_id=task_id,
        model=model,
        node_name=node_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(model, prompt_tokens, completion_tokens),
        estimated_raw_prompt_tokens=estimated_raw_prompt_tokens,
        estimated_tokens_saved=estimated_tokens_saved,
        metadata=meta,
    )
