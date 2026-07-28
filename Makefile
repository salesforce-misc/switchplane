.PHONY: test test-unit test-examples test-cov install-test install-examples install-e2e e2e lint static format formatcheck clean

# Example test suites live outside `testpaths`, so they are named explicitly here.
# Any examples/<name>/tests/ directory is picked up automatically.
EXAMPLE_TESTS := $(wildcard examples/*/tests)

install-test:
	uv pip install -e ".[test]"

install-e2e:
	uv pip install -e ".[test]" -e examples/hello

install-examples:
	uv pip install -e examples/hello -e examples/weather -e examples/devops -e examples/chatbot -e examples/quality

test: test-unit test-examples

test-unit:
	uv run pytest tests/ -v -n auto

# Requires the example packages on the path: make install-examples
test-examples:
ifeq ($(EXAMPLE_TESTS),)
	@echo "No example test suites found; skipping."
else
	uv run pytest $(EXAMPLE_TESTS) -v -n auto
endif

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
