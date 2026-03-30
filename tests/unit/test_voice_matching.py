"""
Unit tests for voice personality matching.

Tests:
- Exact match scoring
- Partial match (some dimensions None)
- No match fallback
- Migration scenario (provider unavailable)
- Personality sync on set_voice
- Identity package serialization with new fields
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from kestrel_sovereign.identity.identity_package import (
    AgentIdentityPackage,
    PersonalityFingerprint,
)
from kestrel_sovereign.voice.base import VoiceInfo, match_voice, VoiceConfig


# ---------------------------------------------------------------------------
# match_voice() tests
# ---------------------------------------------------------------------------


class TestMatchVoice:
    """Tests for the match_voice() scoring function."""

    def _voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(voice_id="nova", name="Nova", provider="openai",
                      gender="feminine", age="young", energy="warm", accent="american"),
            VoiceInfo(voice_id="onyx", name="Onyx", provider="openai",
                      gender="masculine", age="mature", energy="authoritative", accent="american"),
            VoiceInfo(voice_id="echo", name="Echo", provider="openai",
                      gender="masculine", age="middle", energy="calm", accent="american"),
            VoiceInfo(voice_id="fable", name="Fable", provider="openai",
                      gender="neutral", age="middle", energy="warm", accent="british"),
            VoiceInfo(voice_id="alba", name="Alba", provider="piper",
                      gender="feminine", age="young", energy="warm", accent="british"),
        ]

    def test_exact_match_all_dimensions(self):
        """Perfect match across all four dimensions returns that voice."""
        personality = PersonalityFingerprint(
            voice_gender_preference="feminine",
            voice_age_preference="young",
            voice_energy="warm",
            voice_accent_preference="american",
        )
        result = match_voice(personality, self._voices())
        assert result is not None
        assert result.voice_id == "nova"  # 3+2+2+1 = 8 points

    def test_gender_weighted_highest(self):
        """Gender match (3 pts) outweighs age+energy (2+2) when accent is tied."""
        personality = PersonalityFingerprint(
            voice_gender_preference="masculine",
            voice_age_preference="mature",
            voice_energy="authoritative",
            voice_accent_preference="american",
        )
        result = match_voice(personality, self._voices())
        assert result is not None
        assert result.voice_id == "onyx"  # 3+2+2+1 = 8

    def test_partial_match_some_none(self):
        """When some dimensions are None, only non-None are scored."""
        personality = PersonalityFingerprint(
            voice_gender_preference="feminine",
            voice_age_preference=None,
            voice_energy=None,
            voice_accent_preference="british",
        )
        result = match_voice(personality, self._voices())
        assert result is not None
        # alba: gender=feminine(+3), accent=british(+1) = 4
        # nova: gender=feminine(+3), accent=american(0) = 3
        assert result.voice_id == "alba"

    def test_all_none_returns_first(self):
        """When all personality dimensions are None, returns the first voice (score 0 for all)."""
        personality = PersonalityFingerprint()
        voices = self._voices()
        result = match_voice(personality, voices)
        assert result is not None
        assert result.voice_id == voices[0].voice_id

    def test_no_voices_returns_none(self):
        """Empty voice list returns None."""
        personality = PersonalityFingerprint(voice_gender_preference="feminine")
        result = match_voice(personality, [])
        assert result is None

    def test_no_good_match_returns_best_available(self):
        """Even with zero-scoring match, the first voice is returned."""
        personality = PersonalityFingerprint(
            voice_gender_preference="neutral",
            voice_age_preference="mature",
            voice_energy="energetic",
            voice_accent_preference="australian",
        )
        # Only one voice available with no matching attributes
        voices = [
            VoiceInfo(voice_id="x", name="X", provider="test",
                      gender="masculine", age="young", energy="calm", accent="american"),
        ]
        result = match_voice(personality, voices)
        assert result is not None
        assert result.voice_id == "x"

    def test_tiebreak_favors_first(self):
        """When two voices tie on score, the first one wins (stable ordering)."""
        personality = PersonalityFingerprint(
            voice_gender_preference="feminine",
        )
        voices = [
            VoiceInfo(voice_id="a", name="A", provider="p1", gender="feminine"),
            VoiceInfo(voice_id="b", name="B", provider="p2", gender="feminine"),
        ]
        result = match_voice(personality, voices)
        assert result.voice_id == "a"

    def test_migration_scenario_cross_provider(self):
        """When the preferred provider's voice is unavailable, match across providers."""
        personality = PersonalityFingerprint(
            voice_gender_preference="feminine",
            voice_age_preference="young",
            voice_energy="warm",
            voice_accent_preference="british",
        )
        # Only piper voices available (simulating OpenAI unavailable)
        piper_voices = [
            VoiceInfo(voice_id="alba", name="Alba", provider="piper",
                      gender="feminine", age="young", energy="warm", accent="british"),
            VoiceInfo(voice_id="lessac", name="Lessac", provider="piper",
                      gender="masculine", age="middle", energy="calm", accent="american"),
        ]
        result = match_voice(personality, piper_voices)
        assert result is not None
        assert result.voice_id == "alba"  # 3+2+2+1 = 8


# ---------------------------------------------------------------------------
# PersonalityFingerprint voice fields serialization
# ---------------------------------------------------------------------------


class TestPersonalityVoiceFields:
    """Tests for voice personality fields on PersonalityFingerprint."""

    def test_default_values_are_none(self):
        fp = PersonalityFingerprint()
        assert fp.voice_gender_preference is None
        assert fp.voice_age_preference is None
        assert fp.voice_energy is None
        assert fp.voice_accent_preference is None

    def test_to_dict_includes_voice_fields(self):
        fp = PersonalityFingerprint(
            voice_gender_preference="feminine",
            voice_age_preference="young",
            voice_energy="warm",
            voice_accent_preference="american",
        )
        d = fp.to_dict()
        assert d["voice_gender_preference"] == "feminine"
        assert d["voice_age_preference"] == "young"
        assert d["voice_energy"] == "warm"
        assert d["voice_accent_preference"] == "american"

    def test_from_dict_with_voice_fields(self):
        data = {
            "communication_style": "warm",
            "voice_gender_preference": "masculine",
            "voice_energy": "authoritative",
        }
        fp = PersonalityFingerprint.from_dict(data)
        assert fp.voice_gender_preference == "masculine"
        assert fp.voice_energy == "authoritative"
        assert fp.voice_age_preference is None  # not in data

    def test_from_dict_ignores_unknown_keys(self):
        data = {"voice_gender_preference": "feminine", "unknown_field": "value"}
        fp = PersonalityFingerprint.from_dict(data)
        assert fp.voice_gender_preference == "feminine"

    def test_roundtrip_through_identity_package(self):
        """Voice fields survive AgentIdentityPackage serialize/deserialize."""
        personality = PersonalityFingerprint(
            voice_gender_preference="feminine",
            voice_age_preference="young",
            voice_energy="warm",
            voice_accent_preference="british",
        )
        pkg = AgentIdentityPackage(
            did="did:pkh:eip155:1:0xabc",
            agent_name="Test",
            created_at="2026-01-01T00:00:00Z",
            constitution_hash="abc123",
            constitution_text="test",
            personality=personality,
        )
        data = pkg.to_dict()
        restored = AgentIdentityPackage.from_dict(data)
        assert restored.personality.voice_gender_preference == "feminine"
        assert restored.personality.voice_age_preference == "young"
        assert restored.personality.voice_energy == "warm"
        assert restored.personality.voice_accent_preference == "british"

    def test_json_roundtrip(self):
        """Voice fields survive JSON serialize/deserialize."""
        personality = PersonalityFingerprint(
            voice_gender_preference="masculine",
            voice_energy="calm",
        )
        pkg = AgentIdentityPackage(
            did="did:pkh:eip155:1:0x123",
            agent_name="JsonTest",
            created_at="2026-01-01T00:00:00Z",
            constitution_hash="hash",
            constitution_text="text",
            personality=personality,
        )
        json_str = pkg.to_json()
        restored = AgentIdentityPackage.from_json(json_str)
        assert restored.personality.voice_gender_preference == "masculine"
        assert restored.personality.voice_energy == "calm"
        assert restored.personality.voice_age_preference is None

    def test_content_hash_changes_with_voice_fields(self):
        """Adding voice personality fields changes the content hash."""
        pkg1 = AgentIdentityPackage(
            did="did:pkh:eip155:1:0x123",
            agent_name="Test",
            created_at="2026-01-01T00:00:00Z",
            constitution_hash="h",
            constitution_text="t",
        )
        pkg2 = AgentIdentityPackage(
            did="did:pkh:eip155:1:0x123",
            agent_name="Test",
            created_at="2026-01-01T00:00:00Z",
            constitution_hash="h",
            constitution_text="t",
            personality=PersonalityFingerprint(voice_gender_preference="feminine"),
        )
        # Same export_timestamp for deterministic comparison
        pkg1.export_timestamp = pkg2.export_timestamp = "2026-01-01T00:00:00Z"
        assert pkg1.compute_content_hash() != pkg2.compute_content_hash()


# ---------------------------------------------------------------------------
# VoiceInfo extended fields
# ---------------------------------------------------------------------------


class TestVoiceInfoExtended:
    """Tests for VoiceInfo age/energy/accent fields."""

    def test_defaults(self):
        v = VoiceInfo(voice_id="test", name="Test", provider="test")
        assert v.age == "middle"
        assert v.energy == "neutral"
        assert v.accent == "neutral"

    def test_custom_values(self):
        v = VoiceInfo(
            voice_id="nova", name="Nova", provider="openai",
            gender="feminine", age="young", energy="warm", accent="american",
        )
        assert v.age == "young"
        assert v.energy == "warm"
        assert v.accent == "american"


# ---------------------------------------------------------------------------
# VoiceFeature._resolve_voice and personality sync
# ---------------------------------------------------------------------------

from kestrel_sovereign.features.voice.feature import VoiceFeature, VoicePrivacyError
from kestrel_sovereign.voice.base import TTSProvider, STTProvider
from kestrel_sovereign.voice.provider_registry import VoiceProviderRegistry
from typing import AsyncIterator


class FakeTTS(TTSProvider):
    """Fake TTS provider for testing."""
    name = "fake"
    is_local = True
    _available = True

    def __init__(self, voices: list[VoiceInfo] | None = None):
        self._voices = voices or [
            VoiceInfo(voice_id="v1", name="V1", provider="fake",
                      gender="feminine", age="young", energy="warm", accent="american"),
        ]

    async def synthesize(self, text, voice_id, model="", output_format="opus"):
        return b"audio"

    async def synthesize_stream(self, text, voice_id, model="", output_format="opus"):
        yield b"chunk"

    async def list_voices(self):
        return self._voices

    async def is_available(self):
        return self._available


class FakeUnavailableTTS(FakeTTS):
    """TTS provider that reports itself as unavailable."""
    name = "unavailable"
    _available = False

    async def is_available(self):
        return False


def _make_voice_feature(
    voice_config: VoiceConfig | None = None,
    personality: PersonalityFingerprint | None = None,
    tts_providers: dict[str, TTSProvider] | None = None,
) -> VoiceFeature:
    """Create a VoiceFeature with mock agent for testing."""
    agent = MagicMock()
    identity = MagicMock()
    identity.personality = personality or PersonalityFingerprint()
    identity.voice_config = (voice_config.to_dict() if voice_config else None)
    agent.identity = identity
    agent.config = {}
    agent.privacy_agent = None

    feature = VoiceFeature.__new__(VoiceFeature)
    feature.agent = agent
    feature._voice_config = voice_config or VoiceConfig()

    # Set up registry with provided TTS providers
    registry = MagicMock(spec=VoiceProviderRegistry)
    if tts_providers:
        registry.list_tts_providers.return_value = list(tts_providers.keys())
        registry.get_tts.side_effect = lambda name: tts_providers.get(name)
    else:
        registry.list_tts_providers.return_value = []
        registry.get_tts.return_value = None
    feature._voice_registry = registry

    return feature


@pytest.mark.asyncio
class TestResolveVoice:
    """Tests for VoiceFeature._resolve_voice()."""

    async def test_explicit_config_takes_priority(self):
        """When voice_config has provider+voice_id AND provider is available, use it."""
        tts = FakeTTS()
        vc = VoiceConfig(tts_provider="fake", tts_voice_id="v1")
        feature = _make_voice_feature(
            voice_config=vc,
            tts_providers={"fake": tts},
        )
        provider, voice_id = await feature._resolve_voice()
        assert voice_id == "v1"
        assert provider is tts

    async def test_personality_matching_when_no_config(self):
        """When no voice_config, personality hints drive matching."""
        voices = [
            VoiceInfo(voice_id="deep", name="Deep", provider="fake",
                      gender="masculine", age="mature", energy="authoritative", accent="american"),
            VoiceInfo(voice_id="bright", name="Bright", provider="fake",
                      gender="feminine", age="young", energy="warm", accent="american"),
        ]
        tts = FakeTTS(voices=voices)
        personality = PersonalityFingerprint(
            voice_gender_preference="feminine",
            voice_age_preference="young",
            voice_energy="warm",
        )
        feature = _make_voice_feature(
            personality=personality,
            tts_providers={"fake": tts},
        )
        provider, voice_id = await feature._resolve_voice()
        assert voice_id == "bright"

    async def test_migration_fallback_on_unavailable_provider(self):
        """When configured provider is unavailable, fall back to personality matching."""
        unavailable = FakeUnavailableTTS(voices=[
            VoiceInfo(voice_id="orig", name="Orig", provider="unavailable",
                      gender="feminine", age="young", energy="warm", accent="american"),
        ])
        backup = FakeTTS(voices=[
            VoiceInfo(voice_id="backup-f", name="Backup F", provider="fake",
                      gender="feminine", age="young", energy="warm", accent="american"),
            VoiceInfo(voice_id="backup-m", name="Backup M", provider="fake",
                      gender="masculine", age="mature", energy="calm", accent="american"),
        ])
        personality = PersonalityFingerprint(
            voice_gender_preference="feminine",
            voice_age_preference="young",
            voice_energy="warm",
        )
        vc = VoiceConfig(tts_provider="unavailable", tts_voice_id="orig")
        feature = _make_voice_feature(
            voice_config=vc,
            personality=personality,
            tts_providers={"unavailable": unavailable, "fake": backup},
        )
        provider, voice_id = await feature._resolve_voice()
        assert voice_id == "backup-f"
        assert provider is backup

    async def test_fallback_to_first_available_default(self):
        """When no config and no personality hints, fall back to first voice."""
        tts = FakeTTS(voices=[
            VoiceInfo(voice_id="default", name="Default", provider="fake"),
        ])
        feature = _make_voice_feature(tts_providers={"fake": tts})
        provider, voice_id = await feature._resolve_voice()
        assert voice_id == "default"


@pytest.mark.asyncio
class TestSetVoicePersonalitySync:
    """Tests that set_voice syncs personality hints."""

    async def test_set_voice_updates_personality(self):
        """Setting a voice updates personality fingerprint voice fields."""
        voices = [
            VoiceInfo(voice_id="nova", name="Nova", provider="fake",
                      gender="feminine", age="young", energy="warm", accent="american"),
        ]
        tts = FakeTTS(voices=voices)
        personality = PersonalityFingerprint()
        feature = _make_voice_feature(
            personality=personality,
            tts_providers={"fake": tts},
        )

        result = await feature.set_voice("nova", "fake")
        assert result["success"] is True

        # Check personality was synced
        p = feature.agent.identity.personality
        assert p.voice_gender_preference == "feminine"
        assert p.voice_age_preference == "young"
        assert p.voice_energy == "warm"
        assert p.voice_accent_preference == "american"
