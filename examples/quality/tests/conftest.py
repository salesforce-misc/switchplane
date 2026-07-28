"""Test configuration and shared fixtures for examples/quality.

The import-path seam: no example package is installed in .venv, so we inject
the example root onto sys.path at session scope. This works only because the
example has no compiled dependencies (numpy, pandas, etc.).
"""

from pathlib import Path

import pytest

# Resolve the quality root (one level up from tests/)
QUALITY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _inject_quality_on_path():
    """Inject the quality example root onto sys.path so imports resolve.

    Guard: skip if langchain_core is absent (indicates switchplane[llm] not installed).
    """
    pytest.importorskip("langchain_core")
    import sys

    sys.path.insert(0, str(QUALITY_ROOT))
    yield
    sys.path.remove(str(QUALITY_ROOT))
