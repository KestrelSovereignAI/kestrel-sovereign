"""Contracts for bootstrap/example model selection defaults."""

from pathlib import Path

import pytest
import tomllib

from kestrel_sovereign.features.council.feature import CouncilFeature
from scripts.setup_demo_agent import build_demo_kestrel_toml


def test_demo_kestrel_toml_uses_auto_selection_with_hints():
    """The demo agent's kestrel.toml [llm] block must use the vendor/route
    schema with model='auto' on every route — no hardcoded model IDs."""
    config = tomllib.loads(build_demo_kestrel_toml())
    llm = config["llm"]

    assert llm["route_priority"] == ["anthropic:api", "ollama:local"]

    anthropic_api = llm["vendors"]["anthropic"]["routes"]["api"]
    assert anthropic_api["model"] == "auto"
    assert anthropic_api["selection_hints"] == ["opus"]
    assert anthropic_api["api_key_env"] == "ANTHROPIC_API_KEY"

    ollama_local = llm["vendors"]["ollama"]["routes"]["local"]
    assert ollama_local["model"] == "auto"
    assert ollama_local["host"] == "http://localhost:11434"

    # No hardcoded concrete IDs anywhere in the rendered config.
    rendered = build_demo_kestrel_toml()
    assert "claude-opus-4-6" not in rendered
    assert "llama3.2:latest" not in rendered


@pytest.mark.asyncio
async def test_council_members_help_uses_auto_model_example():
    feature = CouncilFeature(agent=None)
    feature.config = None

    result = await feature.list_members()

    assert 'model = "auto"' in result
    assert "claude-opus-4-5-20251101" not in result


def test_run_docker_remote_does_not_inject_hidden_default_model():
    """Epic #1050 tier 3 ported ``scripts/run_docker_remote.sh`` to
    :mod:`kestrel_sovereign.cli_docker_remote`. The property the legacy
    test guarded — ``run_docker_remote`` does not inject a hidden
    default model — must still hold for the Python entry point.
    """
    from kestrel_sovereign import cli_docker_remote

    src = Path(cli_docker_remote.__file__).read_text(encoding="utf-8")

    assert "DEFAULT_LLM_MODEL" not in src
