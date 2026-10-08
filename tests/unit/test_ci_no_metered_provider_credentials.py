"""Routine CI cannot enable metered tests through repo or inherited secrets."""

from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("job", ["integration-tests", "llm-tests"])
def test_provider_credentials_are_explicitly_blank_in_routine_test_steps(job):
    path = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
    workflow = yaml.safe_load(path.read_text())
    steps = [
        step
        for step in workflow["jobs"][job]["steps"]
        if "pytest " in step.get("run", "")
    ]
    assert steps, f"No pytest step found for {job}; credential guard needs review"
    for step in steps:
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "TAVILY_API_KEY",
            "REPLICATE_API_TOKEN",
            "RUNPOD_API_KEY",
            "XAI_API_KEY",
        ):
            assert step["env"].get(key) == "", f"{job} must explicitly blank {key}"
