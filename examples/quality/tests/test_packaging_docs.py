"""Contracts for the Quality example's installation metadata and documentation links."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

QUALITY_ROOT = Path(__file__).parents[1]
REPO_ROOT = QUALITY_ROOT.parents[1]


def test_quality_requires_compatible_switchplane():
    config = tomllib.loads((QUALITY_ROOT / "pyproject.toml").read_text())

    assert "switchplane[llm]>=0.11.0" in config["project"]["dependencies"]


def test_quality_declares_local_editable_switchplane_source():
    config = tomllib.loads((QUALITY_ROOT / "pyproject.toml").read_text())

    assert config["tool"]["uv"]["sources"]["switchplane"] == {
        "path": "../..",
        "editable": True,
    }


def test_provider_pool_doc_link_targets_existing_example_section():
    readme = (QUALITY_ROOT / "README.md").read_text()
    match = re.search(r"\[Provider pool docs\]\(([^)]+)\)", readme)
    assert match, "Quality README must link to provider-pool documentation"

    relative_path, separator, anchor = match.group(1).partition("#")
    target = (QUALITY_ROOT / relative_path).resolve()
    assert target.exists()
    assert target == (REPO_ROOT / "examples" / "quality" / "README.md").resolve()
    assert separator and anchor

    headings = re.findall(r"^#{1,6}\s+(.+)$", target.read_text(), flags=re.MULTILINE)
    slugs = {re.sub(r"[^a-z0-9 -]", "", heading.lower()).strip().replace(" ", "-") for heading in headings}
    assert anchor in slugs
