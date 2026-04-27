from types import SimpleNamespace

from switchplane.usage import estimate_cost_usd, estimate_text_tokens, extract_token_counts, llm_usage_from_response


def test_estimate_text_tokens_uses_four_chars_per_token():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2


def test_extract_token_counts_from_usage_metadata():
    response = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    assert extract_token_counts(response) == (10, 5, 15)


def test_extract_token_counts_from_response_metadata():
    response = SimpleNamespace(response_metadata={"token_usage": {"prompt_tokens": 20, "completion_tokens": 7}})

    assert extract_token_counts(response) == (20, 7, None)


def test_estimate_cost_for_known_model():
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75


def test_usage_from_response_uses_provider_counts():
    response = SimpleNamespace(
        content="ok",
        usage_metadata={"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
    )

    usage = llm_usage_from_response(
        response,
        task_id="task1",
        model="claude-sonnet-4-20250514",
        node_name="summarize",
        estimated_raw_prompt_tokens=1_000,
    )

    assert usage.task_id == "task1"
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 125
    assert usage.estimated_tokens_saved == 900
    assert usage.metadata["token_source"] == "provider"


def test_usage_from_response_falls_back_to_estimates():
    response = SimpleNamespace(content="hello world")

    usage = llm_usage_from_response(
        response,
        task_id="task1",
        model="unknown-model",
        node_name="summarize",
        fallback_prompt_text="abcd",
        fallback_completion_text=response.content,
    )

    assert usage.prompt_tokens == 1
    assert usage.completion_tokens == 3
    assert usage.total_tokens == 4
    assert usage.estimated_cost_usd is None
    assert usage.metadata["token_source"] == "estimated"
