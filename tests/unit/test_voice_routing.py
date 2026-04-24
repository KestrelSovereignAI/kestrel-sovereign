"""
Unit tests for the voice path resolver.

The resolver is a pure function — no network, no registry probing — so tests
just build a :class:`VoiceRoutingContext` and assert the returned route. The
matrix covers every (privacy_preset × llm_vendor × installed-provider set)
combination that produces a distinct decision.

Naming convention: each test maps to a rule in ``routing.resolve``'s docstring.
"""

import pytest

from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.voice.routing import (
    InstalledProviders,
    UserVoicePreferences,
    VoiceRoute,
    VoiceRoutingContext,
    PROVIDER_DEEPGRAM,
    PROVIDER_ELEVENLABS,
    PROVIDER_FASTER_WHISPER,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_REALTIME,
    PROVIDER_PIPER,
    resolve,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _all_installed() -> InstalledProviders:
    """Every provider installed — lets tests focus on privacy + preferences."""
    return InstalledProviders(
        tts={PROVIDER_PIPER, PROVIDER_ELEVENLABS, PROVIDER_OPENAI},
        stt={PROVIDER_FASTER_WHISPER, PROVIDER_OPENAI, PROVIDER_DEEPGRAM},
        conversation={PROVIDER_OPENAI_REALTIME},
        tts_local={PROVIDER_PIPER},
        stt_local={PROVIDER_FASTER_WHISPER},
    )


def _ctx(
    *,
    privacy: str = "normal",
    llm_vendor: str = "anthropic",
    installed: InstalledProviders | None = None,
    prefs: UserVoicePreferences | None = None,
) -> VoiceRoutingContext:
    return VoiceRoutingContext(
        llm_vendor=llm_vendor,
        privacy_config=get_privacy_preset(privacy),
        installed=installed if installed is not None else _all_installed(),
        preferences=prefs if prefs is not None else UserVoicePreferences(),
    )


# ---------------------------------------------------------------------------
# Rule 1: Local-only (ephemeral / isolated)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("privacy", ["ephemeral", "isolated"])
def test_local_only_selects_piper_and_faster_whisper(privacy: str) -> None:
    route = resolve(_ctx(privacy=privacy, llm_vendor="anthropic"))
    assert route.path == "local"
    assert route.tts_provider == PROVIDER_PIPER
    assert route.stt_provider == PROVIDER_FASTER_WHISPER
    assert route.conversation_provider is None
    assert "Local-only" in route.reason


@pytest.mark.parametrize("privacy", ["ephemeral", "isolated"])
def test_local_only_ignores_openai_llm_vendor(privacy: str) -> None:
    """OpenAI chat LLM must not trigger Realtime when privacy forbids cloud."""
    route = resolve(_ctx(privacy=privacy, llm_vendor="openai"))
    assert route.path == "local"
    assert route.conversation_provider is None


@pytest.mark.parametrize("privacy", ["ephemeral", "isolated"])
def test_local_only_ignores_cloud_preference(privacy: str) -> None:
    prefs = UserVoicePreferences(preferred_tts=PROVIDER_ELEVENLABS)
    route = resolve(_ctx(privacy=privacy, prefs=prefs))
    assert route.path == "local"
    assert route.tts_provider == PROVIDER_PIPER  # override ignored
    assert "ignored preferred_tts='elevenlabs'" in route.reason


@pytest.mark.parametrize("privacy", ["ephemeral", "isolated"])
def test_local_only_returns_none_when_no_local_tts(privacy: str) -> None:
    installed = InstalledProviders(
        tts={PROVIDER_OPENAI},  # cloud only, not installable under local-only
        stt={PROVIDER_FASTER_WHISPER},
        stt_local={PROVIDER_FASTER_WHISPER},
    )
    route = resolve(_ctx(privacy=privacy, installed=installed))
    assert route.path is None
    assert "install piper-tts" in route.reason


@pytest.mark.parametrize("privacy", ["ephemeral", "isolated"])
def test_local_only_returns_none_when_no_local_stt(privacy: str) -> None:
    installed = InstalledProviders(
        tts={PROVIDER_PIPER},
        stt={PROVIDER_OPENAI},
        tts_local={PROVIDER_PIPER},
    )
    route = resolve(_ctx(privacy=privacy, installed=installed))
    assert route.path is None
    assert "install faster-whisper" in route.reason


@pytest.mark.parametrize("privacy", ["ephemeral", "isolated"])
def test_local_only_accepts_third_party_local_provider(privacy: str) -> None:
    """Third-party local providers registered via entry_points should count.

    The resolver must not hardcode which provider names are "local" — the
    registry's is_local flag is the source of truth, relayed through
    ``tts_local``/``stt_local``.
    """
    installed = InstalledProviders(
        tts={"custom_edge_tts"},
        stt={"custom_edge_stt"},
        tts_local={"custom_edge_tts"},
        stt_local={"custom_edge_stt"},
    )
    route = resolve(_ctx(privacy=privacy, installed=installed))
    assert route.path == "local"
    assert route.tts_provider == "custom_edge_tts"
    assert route.stt_provider == "custom_edge_stt"


# ---------------------------------------------------------------------------
# Rule 2: Realtime (OpenAI LLM + cloud-allowed + prefer_realtime + provider installed)
# ---------------------------------------------------------------------------


def test_realtime_path_selected_for_openai_llm_and_normal_privacy() -> None:
    route = resolve(_ctx(privacy="normal", llm_vendor="openai"))
    assert route.path == "realtime"
    assert route.conversation_provider == PROVIDER_OPENAI_REALTIME
    assert route.tts_provider is None  # Realtime owns the whole turn
    assert route.stt_provider is None
    assert "Realtime path" in route.reason


def test_realtime_path_selected_for_openai_llm_and_public_privacy() -> None:
    route = resolve(_ctx(privacy="public", llm_vendor="openai"))
    assert route.path == "realtime"


def test_realtime_declined_when_user_disables_prefer_realtime() -> None:
    prefs = UserVoicePreferences(prefer_realtime=False)
    route = resolve(_ctx(privacy="normal", llm_vendor="openai", prefs=prefs))
    assert route.path == "pipeline"
    assert "user declined Realtime" in route.reason


def test_realtime_skipped_when_provider_not_installed() -> None:
    installed = InstalledProviders(
        tts=_all_installed().tts,
        stt=_all_installed().stt,
        conversation=set(),  # no realtime provider
    )
    route = resolve(_ctx(privacy="normal", llm_vendor="openai", installed=installed))
    assert route.path == "pipeline"
    assert "openai_realtime provider not installed" in route.reason


def test_realtime_not_selected_for_non_openai_llm() -> None:
    for vendor in ("anthropic", "google", "ollama", "", None):
        route = resolve(_ctx(privacy="normal", llm_vendor=vendor))
        assert route.path == "pipeline", f"vendor={vendor}"
        # Reason explains that Realtime required OpenAI.
        assert "Realtime requires OpenAI" in route.reason


def test_realtime_blocked_by_ephemeral_privacy_even_on_openai() -> None:
    route = resolve(_ctx(privacy="ephemeral", llm_vendor="openai"))
    assert route.path == "local"


# ---------------------------------------------------------------------------
# Rule 3: Anonymous pipeline — privacy-safe defaults
# ---------------------------------------------------------------------------


def test_anonymous_defaults_to_privacy_safe_providers() -> None:
    route = resolve(_ctx(privacy="anonymous", llm_vendor="anthropic"))
    assert route.path == "pipeline"
    assert route.tts_provider == PROVIDER_PIPER
    assert route.stt_provider == PROVIDER_FASTER_WHISPER
    assert "ANONYMOUS privacy prefers privacy-safe defaults" in route.reason


def test_anonymous_honors_explicit_cloud_preference() -> None:
    prefs = UserVoicePreferences(
        preferred_tts=PROVIDER_ELEVENLABS,
        preferred_stt=PROVIDER_OPENAI,
    )
    route = resolve(_ctx(privacy="anonymous", prefs=prefs))
    assert route.path == "pipeline"
    assert route.tts_provider == PROVIDER_ELEVENLABS
    assert route.stt_provider == PROVIDER_OPENAI


def test_anonymous_does_not_trigger_realtime_even_on_openai_llm() -> None:
    """ANONYMOUS means the user opted for scrubbed storage; keep biometrics local by default."""
    route = resolve(_ctx(privacy="anonymous", llm_vendor="openai"))
    assert route.path == "pipeline"
    assert route.tts_provider == PROVIDER_PIPER


# ---------------------------------------------------------------------------
# Rule 4: Cloud pipeline (NORMAL/PUBLIC, non-OpenAI LLM or Realtime unavailable)
# ---------------------------------------------------------------------------


def test_cloud_pipeline_prefers_elevenlabs_tts_and_openai_stt() -> None:
    route = resolve(_ctx(privacy="normal", llm_vendor="anthropic"))
    assert route.path == "pipeline"
    assert route.tts_provider == PROVIDER_ELEVENLABS  # v3 tag support preferred
    assert route.stt_provider == PROVIDER_OPENAI


def test_cloud_pipeline_falls_back_to_openai_tts_without_elevenlabs() -> None:
    installed = _all_installed()
    installed.tts.remove(PROVIDER_ELEVENLABS)
    route = resolve(_ctx(privacy="normal", llm_vendor="anthropic", installed=installed))
    assert route.tts_provider == PROVIDER_OPENAI


def test_cloud_pipeline_falls_back_to_deepgram_without_openai_stt() -> None:
    installed = _all_installed()
    installed.stt.remove(PROVIDER_OPENAI)
    route = resolve(_ctx(privacy="normal", llm_vendor="anthropic", installed=installed))
    assert route.stt_provider == PROVIDER_DEEPGRAM


def test_cloud_pipeline_falls_back_to_local_stt_last() -> None:
    installed = _all_installed()
    installed.stt.discard(PROVIDER_OPENAI)
    installed.stt.discard(PROVIDER_DEEPGRAM)
    route = resolve(_ctx(privacy="normal", llm_vendor="anthropic", installed=installed))
    assert route.stt_provider == PROVIDER_FASTER_WHISPER


def test_cloud_pipeline_honors_explicit_tts_preference() -> None:
    prefs = UserVoicePreferences(preferred_tts=PROVIDER_OPENAI)
    route = resolve(_ctx(privacy="normal", llm_vendor="anthropic", prefs=prefs))
    assert route.tts_provider == PROVIDER_OPENAI


def test_cloud_pipeline_ignores_non_installed_preference() -> None:
    installed = _all_installed()
    installed.tts.discard(PROVIDER_ELEVENLABS)
    prefs = UserVoicePreferences(preferred_tts=PROVIDER_ELEVENLABS)
    route = resolve(
        _ctx(privacy="normal", llm_vendor="anthropic", installed=installed, prefs=prefs)
    )
    # Falls back to OpenAI (next in fallback chain), since ElevenLabs isn't installed.
    assert route.tts_provider == PROVIDER_OPENAI


# ---------------------------------------------------------------------------
# Rule 5: No path
# ---------------------------------------------------------------------------


def test_no_path_when_no_tts_installed_in_cloud_mode() -> None:
    installed = InstalledProviders(
        tts=set(),
        stt={PROVIDER_OPENAI},
    )
    route = resolve(_ctx(privacy="normal", installed=installed))
    assert route.path is None
    assert "no TTS" in route.reason


def test_no_path_when_no_stt_installed_in_cloud_mode() -> None:
    installed = InstalledProviders(
        tts={PROVIDER_OPENAI},
        stt=set(),
    )
    route = resolve(_ctx(privacy="normal", installed=installed))
    assert route.path is None
    assert "no STT" in route.reason


def test_no_path_when_nothing_installed() -> None:
    route = resolve(_ctx(privacy="normal", installed=InstalledProviders()))
    assert route.path is None


# ---------------------------------------------------------------------------
# VoiceRoute invariants
# ---------------------------------------------------------------------------


def test_voiceroute_is_available_true_when_path_set() -> None:
    route = VoiceRoute(path="pipeline", tts_provider="piper", stt_provider="faster_whisper")
    assert route.is_available


def test_voiceroute_is_available_false_when_path_none() -> None:
    route = VoiceRoute(path=None, reason="nothing installed")
    assert not route.is_available


# ---------------------------------------------------------------------------
# Realtime preferred_conversation override
# ---------------------------------------------------------------------------


def test_realtime_honors_explicit_preferred_conversation() -> None:
    """If a hypothetical second realtime provider is installed, the user's choice wins."""
    installed = _all_installed()
    installed.conversation.add("custom_realtime")
    prefs = UserVoicePreferences(preferred_conversation="custom_realtime")
    route = resolve(_ctx(privacy="normal", llm_vendor="openai", installed=installed, prefs=prefs))
    assert route.path == "realtime"
    assert route.conversation_provider == "custom_realtime"


def test_realtime_falls_back_to_pipeline_when_preferred_conversation_missing() -> None:
    prefs = UserVoicePreferences(preferred_conversation="nonexistent")
    route = resolve(_ctx(privacy="normal", llm_vendor="openai", prefs=prefs))
    assert route.path == "pipeline"
