"""Drift guards for the fresh-checkout quickstart contract."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from kestrel_sovereign.multi_agent.config import (
    DEFAULT_AGENT_START_PORT,
    DEFAULT_HOST_PORT,
)
from kestrel_sovereign.setup.steps.agent import DEFAULT_QUICKSTART_AGENT_NAME
from scripts import check_docs_links


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = PROJECT_ROOT / "QUICKSTART.md"


def _quickstart_text() -> str:
    return QUICKSTART.read_text(encoding="utf-8")


def _normalized_quickstart_text() -> str:
    return " ".join(_quickstart_text().split())


def test_supported_python_display_tracks_package_metadata():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        requires_python = tomllib.load(file)["project"]["requires-python"]

    specifier = SpecifierSet(requires_python)
    supported = [
        minor
        for minor in range(8, 21)
        if Version(f"3.{minor}") in specifier
    ]
    assert supported
    assert supported == list(range(supported[0], supported[-1] + 1))

    displayed_range = f"Python 3.{supported[0]}–3.{supported[-1]}"
    assert displayed_range in _quickstart_text()


def test_happy_path_uses_host_start_and_documents_named_start_alternative():
    text = _quickstart_text()
    normalized = _normalized_quickstart_text()
    name = DEFAULT_QUICKSTART_AGENT_NAME

    assert f"agent named `{name}`" in normalized
    assert f"port `{DEFAULT_HOST_PORT}`" in normalized
    assert f"port `{DEFAULT_AGENT_START_PORT}`" in normalized
    assert (
        f"http://localhost:{DEFAULT_HOST_PORT}/api/agents/{name}"
        in text
    )

    setup_position = text.index("\nuv run kestrel setup --quickstart\n")
    host_start_position = text.index("\nuv run kestrel start\n")
    named_start_position = text.index(f"\nuv run kestrel start {name}\n")
    assert setup_position < host_start_position < named_start_position

    extra_create_position = text.index("uv run kestrel create MyAgent")
    extra_start_position = text.index("uv run kestrel start MyAgent")
    assert extra_create_position < extra_start_position


def test_health_and_api_examples_match_public_and_authenticated_contracts():
    text = _quickstart_text()

    assert '{"status":"ok","agent_initialized":true}' in text
    assert '# {"status":"ok"}' not in text
    assert '"X-API-Key: $KESTREL_API_KEY"' in text
    assert "$KESTREL_URL/health/detailed" in text
    assert "$KESTREL_AGENT_API/v1/models" in text
    assert "$KESTREL_AGENT_API/v1/chat/completions" in text
    assert "The minimal readiness probe is intentionally public" in text


def test_doctor_is_not_described_as_a_live_provider_probe():
    text = _normalized_quickstart_text()

    assert "Doctor is read-only" in text
    assert "it deliberately does not contact LLM providers" in text
    assert "doctor` will remain the readiness gate until Ollama is running" not in text


def test_privacy_and_encryption_boundaries_are_not_overstated():
    text = _normalized_quickstart_text()

    assert "does not promise that every interaction is remembered forever" in text
    assert "Saved-item bodies (`saved_items.content`)" in text
    assert "RAG document-chunk bodies" in text
    assert "remain plaintext columns today" in text
    assert "`KESTREL_DB_KEY` does not encrypt the database" in text


def test_all_quickstart_relative_links_resolve():
    assert check_docs_links.check_file(QUICKSTART) == []
