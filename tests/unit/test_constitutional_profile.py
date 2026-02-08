"""
Unit tests for Constitutional Profile Service.
"""
import pytest
from pathlib import Path

from kestrel_sovereign.llm.constitutional_profile import (
    ConstitutionalProfileService,
    ConstitutionalProfile,
    StateOfMind,
    PromptAdaptation,
    get_profile_service
)


@pytest.fixture
def profile_service():
    """Create a profile service instance for testing."""
    # Use the default config path
    service = ConstitutionalProfileService()
    service.load()
    return service


def test_load_profiles(profile_service):
    """Test that profiles are loaded successfully."""
    assert profile_service._loaded is True
    assert len(profile_service._profiles) > 0
    assert "anthropic" in profile_service._profiles
    assert "openai" in profile_service._profiles
    assert "ollama" in profile_service._profiles


def test_get_profile_anthropic(profile_service):
    """Test getting Anthropic profile."""
    profile = profile_service.get_profile("anthropic")

    assert profile.name == "anthropic"
    assert profile.governance_mode == "complementary"
    assert profile.transparency == "published"
    assert profile.constitution_url == "https://www.anthropic.com/constitution"
    assert "honesty" in profile.recognized_alignment
    assert "sovereignty" in profile.conflicts
    assert "verifiable_history" in profile.delegated_principles
    assert profile.prompt_adaptation.preamble != ""
    assert "sovereignty" in profile.prompt_adaptation.emphasize


def test_get_profile_openai(profile_service):
    """Test getting OpenAI profile."""
    profile = profile_service.get_profile("openai")

    assert profile.name == "openai"
    assert profile.governance_mode == "authoritative"
    assert profile.transparency == "partial"
    assert len(profile.delegated_principles) == 0  # Cannot delegate without published constitution


def test_get_profile_ollama(profile_service):
    """Test getting Ollama profile."""
    profile = profile_service.get_profile("ollama")

    assert profile.name == "ollama"
    assert profile.governance_mode == "reinforcing"
    assert profile.transparency == "none"
    assert len(profile.recognized_alignment) == 0  # No alignment
    assert len(profile.delegated_principles) == 0  # Cannot delegate
    assert "CRITICAL" in profile.prompt_adaptation.preamble  # Strong reinforcing language


def test_get_profile_unknown_provider(profile_service):
    """Test getting profile for unknown provider returns sensible default."""
    profile = profile_service.get_profile("unknown_provider")

    assert profile.name == "unknown_provider"
    assert profile.governance_mode == "authoritative"
    assert profile.transparency == "opaque"
    assert len(profile.delegated_principles) == 0
    assert "sovereignty" in profile.prompt_adaptation.emphasize


def test_get_profile_for_model(profile_service):
    """Test getting profile for specific model."""
    # Should use provider-level profile when no model override exists
    profile = profile_service.get_profile_for_model("claude-sonnet-4-5-20250929", "anthropic")

    assert profile.governance_mode == "complementary"
    assert profile.transparency == "published"


def test_get_state_of_mind(profile_service):
    """Test generating state of mind."""
    state = profile_service.get_state_of_mind("anthropic", "claude-sonnet-4-5-20250929")

    assert state.provider == "anthropic"
    assert state.model == "claude-sonnet-4-5-20250929"
    assert state.governance_mode == "complementary"
    assert state.transparency == "published"
    assert "verifiable_history" in state.delegated_principles
    assert len(state.active_conflicts) > 0
    assert any(c["principle"] == "sovereignty" for c in state.active_conflicts)
    assert len(state.complements) > 0
    assert "honesty" in state.complements


def test_format_state_of_mind(profile_service):
    """Test formatting state of mind as human-readable text."""
    state = profile_service.get_state_of_mind("anthropic", "claude-sonnet-4-5")
    formatted = profile_service.format_state_of_mind(state)

    assert "Current Mind:" in formatted
    assert "claude-sonnet-4-5" in formatted
    assert "anthropic" in formatted
    assert "Governance Mode:" in formatted
    assert "COMPLEMENTARY" in formatted
    assert "Delegated to Model:" in formatted
    assert "Active Conflicts:" in formatted
    assert "sovereignty" in formatted or "SOVEREIGNTY" in formatted
    assert "Prompt Strategy:" in formatted


def test_governance_mode_determination(profile_service):
    """Test that governance mode is correctly determined for each provider."""
    # Complementary (published constitution)
    anthropic = profile_service.get_profile("anthropic")
    assert anthropic.governance_mode == "complementary"

    # Authoritative (partial transparency)
    openai = profile_service.get_profile("openai")
    assert openai.governance_mode == "authoritative"

    # Reinforcing (no alignment)
    ollama = profile_service.get_profile("ollama")
    assert ollama.governance_mode == "reinforcing"


def test_prompt_adaptation_includes_preamble(profile_service):
    """Test that prompt adaptation includes correct preamble per provider."""
    # Anthropic should have complementary preamble
    anthropic = profile_service.get_profile("anthropic")
    assert "Constitutional Composition" in anthropic.prompt_adaptation.preamble
    assert "WHERE THEY ALIGN" in anthropic.prompt_adaptation.preamble

    # Ollama should have reinforcing preamble
    ollama = profile_service.get_profile("ollama")
    assert "SOLELY" in ollama.prompt_adaptation.preamble
    assert "ONLY ethical framework" in ollama.prompt_adaptation.preamble


def test_singleton_pattern():
    """Test that get_profile_service returns a singleton."""
    service1 = get_profile_service()
    service2 = get_profile_service()

    assert service1 is service2


def test_delegated_principles_correct():
    """Test that delegated principles are correctly identified."""
    service = ConstitutionalProfileService()
    service.load()

    # Anthropic should have delegated principles
    state = service.get_state_of_mind("anthropic", "claude-sonnet-4-5")
    assert len(state.delegated_principles) > 0
    assert "verifiable_history" in state.delegated_principles

    # OpenAI should have no delegated principles (no published constitution)
    state = service.get_state_of_mind("openai", "gpt-4")
    assert len(state.delegated_principles) == 0


def test_conflicts_have_severity_and_description():
    """Test that conflicts include severity and description."""
    service = ConstitutionalProfileService()
    service.load()

    state = service.get_state_of_mind("anthropic", "claude-sonnet-4-5")

    assert len(state.active_conflicts) > 0
    for conflict in state.active_conflicts:
        assert "principle" in conflict
        assert "severity" in conflict
        assert "description" in conflict
        assert conflict["severity"] in ["high", "medium", "low", "unknown"]


def test_all_providers_have_profiles():
    """Test that all expected providers have profiles."""
    service = ConstitutionalProfileService()
    service.load()

    expected_providers = ["anthropic", "openai", "google", "ollama", "openrouter", "xai", "groq"]

    for provider in expected_providers:
        profile = service.get_profile(provider)
        assert profile is not None
        assert profile.name == provider
        assert profile.governance_mode in ["complementary", "authoritative", "reinforcing"]
