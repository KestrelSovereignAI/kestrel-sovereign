"""
Voice path resolver — single source of truth for picking the voice path.

Given ``(llm_vendor, privacy_config, installed_providers, user_preferences)``
returns a :class:`VoiceRoute` describing which providers and transport to use.

Every caller (``endpoints/voice_realtime.py``, ``endpoints/voice.py``,
``features/voice/feature.py``, the frontend UI badge) must consult
:func:`resolve` and honor the returned route. Callers must not reimplement
privacy gating or provider selection — the rules live here and only here.

Design notes:

* :func:`resolve` is pure: takes a context, returns a route. No I/O, no network.
* The route carries provider *names* (e.g. ``"openai_realtime"``) — never model
  IDs. Model selection happens inside each provider via runtime discovery,
  per the repo-wide no-hardcoded-models rule.
* Rule ordering is documented in :func:`resolve`'s docstring and inline; change
  it deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from kestrel_sovereign.privacy import PrivacyConfig


VoicePath = Literal["realtime", "pipeline", "local"]


# Provider name constants. These are identifiers that VoiceProviderRegistry
# (and its conversation-provider sibling, ticket #725) recognize. They are
# NOT model IDs. Adding a new provider: register it in the appropriate
# registry; the resolver only cares about presence via InstalledProviders.
PROVIDER_PIPER = "piper"
PROVIDER_FASTER_WHISPER = "faster_whisper"
PROVIDER_OPENAI = "openai"
PROVIDER_ELEVENLABS = "elevenlabs"
PROVIDER_DEEPGRAM = "deepgram"
PROVIDER_OPENAI_REALTIME = "openai_realtime"


@dataclass
class UserVoicePreferences:
    """Per-agent voice preferences, typically loaded from ``AgentIdentityPackage.voice_config``.

    Empty fields mean "no preference — let the resolver pick." An explicit value
    overrides the resolver's default when compatible with privacy + availability;
    if the preference is unreachable (cloud provider in local-only mode, or a
    provider not installed) the resolver ignores it and records the reason.
    """

    preferred_tts: Optional[str] = None
    preferred_stt: Optional[str] = None
    preferred_conversation: Optional[str] = None
    # When True (default) and an OpenAI chat model is active, the resolver
    # selects the Realtime path in NORMAL/PUBLIC privacy. Set False to stay on
    # the cascaded Pipeline even on OpenAI (e.g. to keep STT→our-LLM→TTS
    # symmetry, or to use a non-OpenAI TTS voice).
    prefer_realtime: bool = True


@dataclass
class InstalledProviders:
    """Snapshot of which voice providers are registered and currently available.

    Callers pass what the registry reports today; the resolver does not itself
    probe availability. Keeping this as a dataclass (not the registry handle)
    keeps :func:`resolve` pure and trivially unit-testable.

    ``tts_local``/``stt_local`` hold the subset whose provider instance reports
    ``is_local=True``. The resolver uses these to enforce the local-only rule
    without hardcoding which provider names are "local" — third-party local
    providers registered via entry_points are first-class.
    """

    tts: set[str] = field(default_factory=set)
    stt: set[str] = field(default_factory=set)
    conversation: set[str] = field(default_factory=set)
    tts_local: set[str] = field(default_factory=set)
    stt_local: set[str] = field(default_factory=set)


@dataclass
class VoiceRoutingContext:
    """Everything :func:`resolve` needs to decide."""

    llm_vendor: Optional[str]  # "openai", "anthropic", "google", "ollama", ...
    privacy_config: PrivacyConfig
    installed: InstalledProviders
    preferences: UserVoicePreferences = field(default_factory=UserVoicePreferences)


@dataclass
class VoiceRoute:
    """Resolver output — describes the selected voice transport and providers.

    ``path is None`` means no usable route exists under the current context
    (e.g. EPHEMERAL privacy with no local TTS installed). The UI must disable
    the voice button in this case and surface ``reason``.

    ``blocked_tts`` and ``blocked_stt`` name the user-preferred provider the
    resolver *rejected* because the current privacy mode forbade it (typically
    a cloud provider under local-only mode). These let callers surface the old
    "Cannot use X in Y privacy mode" error shape without re-deriving the
    policy. ``None`` means no preference was blocked.
    """

    path: Optional[VoicePath]
    tts_provider: Optional[str] = None
    stt_provider: Optional[str] = None
    conversation_provider: Optional[str] = None
    reason: str = ""
    blocked_tts: Optional[str] = None
    blocked_stt: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self.path is not None


# ---------------------------------------------------------------------------
# Helpers — internal, do not export
# ---------------------------------------------------------------------------


def _pick(preferred: Optional[str], fallbacks: list[str], installed: set[str]) -> Optional[str]:
    """Return the first available choice, honoring an explicit preference.

    Resolution order:

    1. ``preferred`` if installed.
    2. Any name from ``fallbacks`` (priority order).
    3. Any installed provider (sorted alphabetically for determinism) — this
       catches third-party providers registered via entry_points whose names
       we don't hardcode here.

    Returns ``None`` only when ``installed`` is empty.
    """
    if preferred and preferred in installed:
        return preferred
    for candidate in fallbacks:
        if candidate in installed:
            return candidate
    if installed:
        return sorted(installed)[0]
    return None


def _pick_local(preferred: Optional[str], fallbacks: list[str], local_installed: set[str]) -> Optional[str]:
    """Like :func:`_pick` but restrict to the ``local_installed`` set.

    A non-local preference is ignored in local-only mode; the caller records
    the override in ``reason`` and surfaces ``blocked_tts``/``blocked_stt``.
    Falls back to any local provider when preferred + explicit fallbacks all
    miss — third-party local providers registered via entry_points are
    first-class.
    """
    if preferred and preferred in local_installed:
        return preferred
    for candidate in fallbacks:
        if candidate in local_installed:
            return candidate
    if local_installed:
        return sorted(local_installed)[0]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve(ctx: VoiceRoutingContext) -> VoiceRoute:
    """Pick the best legal voice path for ``ctx``.

    Rule order (first match wins):

    1. **Local-only.** Privacy forbids cloud (``not allows_cloud_llm()``).
       Select Piper TTS + Faster-Whisper STT. Ignore cloud preferences; record
       the override. If neither local provider is installed, return
       ``path=None`` with an actionable reason.

    2. **Anonymous pipeline.** Privacy is ANONYMOUS (cloud allowed but
       ``storage="scrubbed"``). Default to privacy-safer Piper TTS and
       Faster-Whisper STT unless the user explicitly opted into a cloud
       provider. ANONYMOUS intentionally never triggers Realtime — even when
       text storage is scrubbed, streaming raw biometric voice to a cloud
       provider defeats the mode's purpose. Runs before the Realtime rule
       for exactly this reason.

    3. **Realtime.** Privacy is NORMAL/PUBLIC *and* the user's chat LLM vendor
       is ``openai`` *and* ``preferences.prefer_realtime`` *and* a realtime
       conversation provider is installed. Select the ``openai_realtime``
       conversation provider; ``tts_provider``/``stt_provider`` stay ``None``
       because Realtime owns the whole turn.

    4. **Cloud pipeline.** Privacy is NORMAL/PUBLIC and either the LLM is not
       OpenAI, the user declined Realtime, or no realtime provider is
       installed. Prefer ElevenLabs v3 for TTS (inline tag support); fall back
       to OpenAI TTS. STT prefers OpenAI transcribe, falls back to Deepgram,
       then Faster-Whisper.

    5. **No path.** None of the above produced a viable combination. Return
       ``path=None`` with an actionable reason so the UI can explain.
    """
    privacy = ctx.privacy_config
    prefs = ctx.preferences
    installed = ctx.installed

    # -- Rule 1: Local-only ------------------------------------------------
    if not privacy.allows_cloud_llm():
        # Flag rejected cloud preferences independently of whether a local
        # fallback exists — callers use blocked_tts/blocked_stt to surface
        # the precise "Cannot use X in Y privacy mode" error to users even
        # when the opposite channel is missing entirely. "Not local" means
        # "installed but is_local=False" — also a cloud-only preference is
        # blocked when nothing by that name exists locally.
        blocked_tts = (
            prefs.preferred_tts
            if prefs.preferred_tts and prefs.preferred_tts not in installed.tts_local
            else None
        )
        blocked_stt = (
            prefs.preferred_stt
            if prefs.preferred_stt and prefs.preferred_stt not in installed.stt_local
            else None
        )

        tts = _pick_local(
            prefs.preferred_tts,
            [PROVIDER_PIPER],
            installed.tts_local,
        )
        stt = _pick_local(
            prefs.preferred_stt,
            [PROVIDER_FASTER_WHISPER],
            installed.stt_local,
        )

        if tts is None or stt is None:
            missing = []
            if tts is None:
                missing.append("local TTS (install piper-tts)")
            if stt is None:
                missing.append("local STT (install faster-whisper)")
            return VoiceRoute(
                path=None,
                tts_provider=tts,
                stt_provider=stt,
                reason=(
                    f"Voice unavailable in local-only privacy mode: missing "
                    f"{' and '.join(missing)}."
                ),
                blocked_tts=blocked_tts,
                blocked_stt=blocked_stt,
            )

        override_note = ""
        if blocked_tts:
            override_note += (
                f" (ignored preferred_tts='{blocked_tts}': cloud provider"
                f" blocked in local-only mode.)"
            )
        if blocked_stt:
            override_note += (
                f" (ignored preferred_stt='{blocked_stt}': cloud provider"
                f" blocked in local-only mode.)"
            )

        return VoiceRoute(
            path="local",
            tts_provider=tts,
            stt_provider=stt,
            reason="Local-only pipeline: privacy mode blocks cloud providers." + override_note,
            blocked_tts=blocked_tts,
            blocked_stt=blocked_stt,
        )

    # -- Rule 2: Anonymous pipeline ---------------------------------------
    # Cloud is allowed but storage is scrubbed. Default to privacy-safer
    # local TTS/STT unless the user explicitly preferred a cloud provider.
    # This deliberately runs before the Realtime rule — ANONYMOUS should not
    # stream raw biometric voice to the cloud just because the chat LLM is
    # OpenAI. If the user wants Realtime, they should switch to NORMAL.
    vendor = (ctx.llm_vendor or "").lower()
    if privacy.requires_anonymization():
        tts = _pick(
            prefs.preferred_tts,
            [PROVIDER_PIPER, PROVIDER_ELEVENLABS, PROVIDER_OPENAI],
            installed.tts,
        )
        stt = _pick(
            prefs.preferred_stt,
            [PROVIDER_FASTER_WHISPER, PROVIDER_OPENAI, PROVIDER_DEEPGRAM],
            installed.stt,
        )
        if tts is None or stt is None:
            return _no_path_for_missing_pipeline(installed)
        return VoiceRoute(
            path="pipeline",
            tts_provider=tts,
            stt_provider=stt,
            reason=(
                f"Pipeline path: ANONYMOUS privacy prefers privacy-safe defaults "
                f"(TTS={tts}, STT={stt})."
            ),
        )

    # -- Rule 3: Realtime -------------------------------------------------
    # Privacy is NORMAL/PUBLIC (cloud OK + not scrubbed), the chat LLM vendor
    # is OpenAI, the user wants Realtime, and a realtime conversation provider
    # is installed. Never silently switches the user's LLM — the vendor check
    # requires they already chose OpenAI.
    if (
        vendor == "openai"
        and prefs.prefer_realtime
        and (
            prefs.preferred_conversation in installed.conversation
            if prefs.preferred_conversation
            else PROVIDER_OPENAI_REALTIME in installed.conversation
        )
    ):
        conv = prefs.preferred_conversation or PROVIDER_OPENAI_REALTIME
        return VoiceRoute(
            path="realtime",
            conversation_provider=conv,
            reason=(
                f"Realtime path: OpenAI chat LLM with {conv} provider and "
                f"cloud-allowing privacy mode."
            ),
        )

    # -- Rule 4: Cloud pipeline (NORMAL / PUBLIC, non-OpenAI or no realtime) -
    tts = _pick(
        prefs.preferred_tts,
        [PROVIDER_ELEVENLABS, PROVIDER_OPENAI, PROVIDER_PIPER],
        installed.tts,
    )
    stt = _pick(
        prefs.preferred_stt,
        [PROVIDER_OPENAI, PROVIDER_DEEPGRAM, PROVIDER_FASTER_WHISPER],
        installed.stt,
    )
    if tts is None and stt is None:
        return _no_path_for_missing_pipeline(installed)
    if tts is None or stt is None:
        # Partial route: one channel resolved, the other didn't. Return
        # path=None so full voice sessions know to disable, but populate the
        # resolved channel so single-direction callers (TTS-only `speak`,
        # STT-only `transcribe`) can still proceed.
        missing = "TTS" if tts is None else "STT"
        return VoiceRoute(
            path=None,
            tts_provider=tts,
            stt_provider=stt,
            reason=f"Partial voice availability: no {missing} provider installed.",
        )

    # Explain why Realtime wasn't picked when the user is on OpenAI. This lets
    # the UI tooltip display an accurate reason.
    if vendor == "openai" and prefs.prefer_realtime:
        realtime_missing = PROVIDER_OPENAI_REALTIME not in installed.conversation
        if realtime_missing:
            reason_suffix = " (Realtime unavailable: openai_realtime provider not installed.)"
        else:
            reason_suffix = ""
    elif vendor == "openai" and not prefs.prefer_realtime:
        reason_suffix = " (user declined Realtime: prefer_realtime=False)"
    else:
        reason_suffix = f" (LLM vendor '{vendor or 'unknown'}' — Realtime requires OpenAI)"

    return VoiceRoute(
        path="pipeline",
        tts_provider=tts,
        stt_provider=stt,
        reason=f"Pipeline path: cloud-allowing privacy (TTS={tts}, STT={stt})." + reason_suffix,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _no_path_for_missing_pipeline(installed: InstalledProviders) -> VoiceRoute:
    missing: list[str] = []
    if not installed.tts:
        missing.append("TTS")
    if not installed.stt:
        missing.append("STT")
    if not missing:
        # Both sets non-empty but _pick still returned None — means the
        # preferred provider wasn't installed and no fallback covered it.
        # Surface the fact rather than lying about the cause.
        reason = "Voice unavailable: could not select a provider from installed options."
    else:
        reason = f"Voice unavailable: no {' and '.join(missing)} provider installed."
    return VoiceRoute(path=None, reason=reason)
