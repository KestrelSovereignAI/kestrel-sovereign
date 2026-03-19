"""Contracts for bootstrap/example model selection defaults."""

from pathlib import Path

import pytest

from kestrel_sovereign.features.council.feature import CouncilFeature
from scripts.setup_demo_agent import build_demo_llm_config


def test_demo_llm_config_uses_auto_selection_with_hints():
    config = build_demo_llm_config()

    assert 'provider_priority = ["anthropic", "ollama"]' in config
    assert '[anthropic]\nmodel = "auto"\nselection_hints = ["opus"]' in config
    assert '[ollama]\nhost = "http://localhost:11434"\nmodel = "auto"' in config
    assert "claude-opus-4-6" not in config
    assert 'model = "llama3.2:latest"' not in config


@pytest.mark.asyncio
async def test_council_members_help_uses_auto_model_example():
    feature = CouncilFeature(agent=None)
    feature.config = None

    result = await feature.list_members()

    assert 'model = "auto"' in result
    assert "claude-opus-4-5-20251101" not in result


def test_run_docker_remote_does_not_inject_hidden_default_model():
    script = Path("scripts/run_docker_remote.sh").read_text(encoding="utf-8")

    assert "DEFAULT_LLM_MODEL" not in script
