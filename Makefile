.PHONY: test test-cov install-test install-e2e e2e lint static format formatcheck clean

install-test:
	uv pip install -e ".[test]"

install-e2e:
	uv pip install -e ".[test]" -e examples/hello

install-examples:
	uv pip install -e examples/hello -e examples/weather -e examples/devops -e examples/chatbot

test:
	uv run pytest tests/ -v -n auto

itest:
	ITEST=1 uv run pytest -n auto tests/ -v

e2e:
	E2E=1 uv run pytest tests/e2e/ -v --timeout=30

test-cov:
	uv run pytest tests/ -v -n auto --cov=switchplane --cov-report=term-missing

lint:
	uv run python -m py_compile src/switchplane/*.py

static:
	uv run ruff check src/ tests/ examples/

format:
	uv run ruff format src/ tests/ examples/

formatcheck:
	uv run ruff format --check src/ tests/ examples/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov
