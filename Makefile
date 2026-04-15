.PHONY: test test-cov install-test lint static format formatcheck clean

install-test:
	uv pip install -e ".[test]"

install-examples:
	uv pip install -e examples/hello -e examples/weather -e examples/devops -e examples/chatbot

test:
	pytest tests/ -v -n auto

itest:
	ITEST=1 pytest -n auto tests/ -v 

test-cov:
	pytest tests/ -v -n auto --cov=switchplane --cov-report=term-missing

lint:
	python -m py_compile src/switchplane/*.py

static:
	ruff check src/ tests/ examples/

format:
	ruff format src/ tests/ examples/

formatcheck:
	ruff format --check src/ tests/ examples/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov
