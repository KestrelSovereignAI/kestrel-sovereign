"""
Voice Provider Registry.

Manages TTS, STT, and (new in #725) ConversationProvider registration,
discovery, and routing. Mirrors the pattern in
``kestrel_sovereign/llm/provider_registry.py``.

External packages register providers via entry_points. TTS/STT share one
group for backward compat; conversation (speech-to-speech) providers live
in a separate group because they're factories for live sessions, not
bytes-to-text transforms::

    [project.entry-points."kestrel_sovereign.voice_providers"]
    ElevenLabsTTS = "kestrel_voice_elevenlabs:ElevenLabsTTSProvider"
    ElevenLabsSTT = "kestrel_voice_elevenlabs:ElevenLabsSTTProvider"

    [project.entry-points."kestrel_sovereign.conversation_providers"]
    OpenAIRealtime = "kestrel_voice_openai:OpenAIRealtimeConversationProvider"
"""
import logging
from dataclasses import dataclass
from typing import Optional

from kestrel_sdk.voice import ConversationProvider
from kestrel_sovereign.entrypoints import discover_entry_point_classes
from .base import TTSProvider, STTProvider

logger = logging.getLogger(__name__)

VOICE_PROVIDER_ENTRY_POINT_GROUP = "kestrel_sovereign.voice_providers"
CONVERSATION_PROVIDER_ENTRY_POINT_GROUP = "kestrel_sovereign.conversation_providers"


@dataclass
class ProviderDiagnostic:
    """One row in the provider-status surface.

    Captures every attempted provider — registered, unavailable, or
    import-broken — so the UI can show a real reason ("API key lacks
    voices_read") instead of "no voices reported." Filled by the registry at
    boot. /voice/providers/status enriches each TTS row with a live
    ``list_voices`` probe.
    """

    name: str                          # entry-point name, e.g. "ElevenLabsTTSProvider"
    provider_name: Optional[str]       # provider.name; None if instantiation failed
    kind: str                          # "tts" | "stt" | "conversation"
    registered: bool
    is_local: bool = False
    init_error: Optional[str] = None
    available_error: Optional[str] = None
    voice_count: Optional[int] = None
    voice_list_error: Optional[str] = None
    install_hint: Optional[str] = None


def _install_hint_for(ep_name: str, init_error: Optional[str], avail_error: Optional[str]) -> Optional[str]:
    """Map common failure shapes to a one-liner the user can act on.

    Returns ``None`` when no specific hint applies — the UI falls back to
    showing the raw error from ``init_error`` / ``available_error``.
    """
    name = (ep_name or "").lower()
    blob = " ".join(filter(None, [init_error or "", avail_error or ""])).lower()
    if "elevenlabs" in name:
        if "voices_read" in blob or "missing_permission" in blob:
            return ("Your ElevenLabs API key is missing the `voices_read` scope. "
                    "Edit at elevenlabs.io/app/settings/api-keys and grant "
                    "voices_read + text_to_speech.")
        if "is_available() returned false" in blob:
            return "Set ELEVENLABS_API_KEY in your environment, then restart the host."
    if "deepgram" in name:
        if "import failed" in blob or "cannot import name" in blob or "no module named 'deepgram." in blob:
            return ("kestrel-voice-deepgram needs an update for the installed "
                    "deepgram-sdk version. Upgrade the package or pin the SDK.")
        if "is_available() returned false" in blob:
            return "Set DEEPGRAM_API_KEY in your environment, then restart the host."
    if "openai" in name and "is_available() returned false" in blob:
        return "Set OPENAI_API_KEY in your environment, then restart the host."
    return None


class VoiceProviderRegistry:
    """Registry for TTS, STT, and ConversationProvider instances."""

    def __init__(self, config: dict):
        """Initialize the voice provider registry.

        Args:
            config: Voice configuration dictionary (the [voice] section from kestrel.toml).
        """
        self._tts_providers: dict[str, TTSProvider] = {}
        self._stt_providers: dict[str, STTProvider] = {}
        self._conversation_providers: dict[str, ConversationProvider] = {}
        self._config = config
        self._initialized = False
        # Every entry-point we attempted, registered or not. Keyed implicitly
        # by entry-point name so a single package contributing both TTS+STT
        # appears twice (one diagnostic per role).
        self._diagnostics: list[ProviderDiagnostic] = []

    def diagnostics(self) -> list[ProviderDiagnostic]:
        """Snapshot of every provider attempted at boot, registered or not."""
        return list(self._diagnostics)

    async def initialize(self) -> None:
        """Discover and initialize available providers based on config and installed packages.

        Iterates through the priority lists in config and attempts to register
        each provider. Providers that aren't available (missing packages, no API keys)
        are skipped with a warning.

        After built-in providers, scans entry_points for external voice providers.
        Built-in providers win on name collisions.
        """
        if self._initialized:
            return

        tts_priority = self._config.get("tts_provider_priority", [])
        stt_priority = self._config.get("stt_provider_priority", [])

        for provider_name in tts_priority:
            try:
                provider = self._create_tts_provider(provider_name)
                if provider and await provider.is_available():
                    self.register_tts(provider)
                    logger.info(f"Initialized TTS provider: {provider_name}")
                else:
                    logger.warning(f"TTS provider '{provider_name}' not available. Skipping.")
            except Exception as e:
                logger.error(f"Failed to initialize TTS provider '{provider_name}': {e}")

        for provider_name in stt_priority:
            try:
                provider = self._create_stt_provider(provider_name)
                if provider and await provider.is_available():
                    self.register_stt(provider)
                    logger.info(f"Initialized STT provider: {provider_name}")
                else:
                    logger.warning(f"STT provider '{provider_name}' not available. Skipping.")
            except Exception as e:
                logger.error(f"Failed to initialize STT provider '{provider_name}': {e}")

        # Phase 2: Discover external providers via entry_points
        await self._discover_entrypoint_providers()

        # Phase 3: Discover conversation (speech-to-speech) providers.
        await self._discover_conversation_providers()

        self._initialized = True

    async def _discover_entrypoint_providers(self) -> None:
        """Scan entry_points for external TTS/STT providers.

        Discovered providers that are subclasses of TTSProvider or STTProvider
        are instantiated with config and registered. Built-in providers
        (already registered) win on name collisions.
        """
        # Single scan — entry points can provide either TTS or STT (or both)
        # We check against both base classes
        tts_classes = discover_entry_point_classes(VOICE_PROVIDER_ENTRY_POINT_GROUP, TTSProvider)
        stt_classes = discover_entry_point_classes(VOICE_PROVIDER_ENTRY_POINT_GROUP, STTProvider)

        for ep_name, cls in tts_classes.items():
            await self._try_register("tts", ep_name, cls)
        for ep_name, cls in stt_classes.items():
            await self._try_register("stt", ep_name, cls)

        # Capture entry points that failed to even import so the user can
        # see a real reason in /providers/status (e.g. deepgram-sdk SDK
        # version mismatch). discover_entry_point_classes drops these
        # silently — we re-scan with raw importlib to record the failure.
        await self._record_import_failures()

    async def _try_register(self, kind: str, ep_name: str, cls) -> None:
        """Instantiate, register if available, ALWAYS record a diagnostic.

        Single seam where availability decisions are recorded — no silent
        ``except: pass``. Every outcome (registered, init-failed, unavailable,
        name-collision) becomes a ProviderDiagnostic that
        /voice/providers/status can show.
        """
        provider = None
        provider_name = None
        init_error: Optional[str] = None
        available_error: Optional[str] = None
        registered = False
        is_local = False
        try:
            provider_config = self._config.get(ep_name, {})
            provider = cls(config=provider_config)
            provider_name = provider.name
            is_local = bool(getattr(provider, "is_local", False))
            existing = self._tts_providers if kind == "tts" else self._stt_providers
            if provider_name in existing:
                available_error = f"shadowed by already-registered '{provider_name}'"
            else:
                try:
                    available = await provider.is_available()
                except Exception as e:
                    available_error = f"is_available() raised: {e}"
                else:
                    if available:
                        if kind == "tts":
                            self.register_tts(provider)
                        else:
                            self.register_stt(provider)
                        registered = True
                        logger.info(f"Registered entry_point {kind.upper()} provider: {ep_name}")
                    else:
                        available_error = "is_available() returned False"
        except Exception as e:
            init_error = f"{type(e).__name__}: {e}"
            logger.warning(f"Failed to load entry_point {kind.upper()} provider '{ep_name}': {e}")

        self._diagnostics.append(
            ProviderDiagnostic(
                name=ep_name,
                provider_name=provider_name,
                kind=kind,
                registered=registered,
                is_local=is_local,
                init_error=init_error,
                available_error=available_error,
                install_hint=_install_hint_for(ep_name, init_error, available_error),
            )
        )

    async def _record_import_failures(self) -> None:
        """Find entry points whose underlying module failed to import."""
        try:
            from importlib.metadata import entry_points
        except Exception:
            return
        already = {d.name for d in self._diagnostics}
        for group in (VOICE_PROVIDER_ENTRY_POINT_GROUP, CONVERSATION_PROVIDER_ENTRY_POINT_GROUP):
            try:
                eps = entry_points(group=group)
            except Exception:
                continue
            for ep in eps:
                if ep.name in already:
                    continue
                try:
                    ep.load()
                except Exception as e:
                    kind = "conversation" if group == CONVERSATION_PROVIDER_ENTRY_POINT_GROUP else "tts"
                    msg = f"import failed: {type(e).__name__}: {e}"
                    self._diagnostics.append(
                        ProviderDiagnostic(
                            name=ep.name,
                            provider_name=None,
                            kind=kind,
                            registered=False,
                            init_error=msg,
                            install_hint=_install_hint_for(ep.name, msg, None),
                        )
                    )
                    logger.warning(f"Entry point '{ep.name}' from '{group}' failed to import: {e}")

    async def _discover_conversation_providers(self) -> None:
        """Scan entry_points for ConversationProvider implementations.

        Conversation providers (speech-to-speech) live in a separate group
        from TTS/STT because they own the full turn and have a different
        contract. Discovered providers that are subclasses of
        ``ConversationProvider`` are instantiated with their config section
        and registered if available.
        """
        classes = discover_entry_point_classes(
            CONVERSATION_PROVIDER_ENTRY_POINT_GROUP, ConversationProvider
        )
        for ep_name, cls in classes.items():
            provider = None
            provider_name = None
            init_error: Optional[str] = None
            available_error: Optional[str] = None
            registered = False
            try:
                provider_config = self._config.get(ep_name, {})
                provider = cls(config=provider_config)
                provider_name = provider.name
                if provider_name in self._conversation_providers:
                    available_error = f"shadowed by already-registered '{provider_name}'"
                else:
                    try:
                        available = await provider.is_available()
                    except Exception as e:
                        available_error = f"is_available() raised: {e}"
                    else:
                        if available:
                            self.register_conversation(provider)
                            registered = True
                            logger.info(f"Registered entry_point conversation provider: {ep_name}")
                        else:
                            available_error = "is_available() returned False"
            except Exception as e:
                init_error = f"{type(e).__name__}: {e}"
                logger.warning(f"Failed to load entry_point conversation provider '{ep_name}': {e}")

            self._diagnostics.append(
                ProviderDiagnostic(
                    name=ep_name,
                    provider_name=provider_name,
                    kind="conversation",
                    registered=registered,
                    init_error=init_error,
                    available_error=available_error,
                    install_hint=_install_hint_for(ep_name, init_error, available_error),
                )
            )

    # Cloud voice providers (openai, elevenlabs, deepgram) are discovered
    # via entry_points from their respective packages:
    #   kestrel-voice-openai, kestrel-voice-elevenlabs, kestrel-voice-deepgram

    def _create_tts_provider(self, name: str) -> Optional[TTSProvider]:
        """Create a TTS provider by name.

        Local providers (piper) are imported from core. Cloud providers
        (openai, elevenlabs) are discovered via entry_points.
        """
        if name == "piper":
            try:
                from .piper_tts import PiperTTSProvider
                piper_config = self._config.get("piper", {})
                return PiperTTSProvider(piper_config)
            except ImportError:
                logger.warning("piper-tts package not installed. Skipping piper TTS.")
                return None

        # Cloud providers are discovered via entry_points in _discover_entrypoint_providers()
        logger.debug(f"TTS provider '{name}' not a built-in; will check entry_points.")
        return None

    def _create_stt_provider(self, name: str) -> Optional[STTProvider]:
        """Create an STT provider by name.

        Local providers (faster_whisper) are imported from core. Cloud
        providers (openai, deepgram) are discovered via entry_points.
        """
        if name == "faster_whisper":
            try:
                from .faster_whisper_stt import FasterWhisperSTTProvider
                provider_config = self._config.get("faster_whisper", {})
                return FasterWhisperSTTProvider(config=provider_config)
            except ImportError:
                logger.warning("faster-whisper package not installed. Skipping faster_whisper STT.")
                return None

        # Cloud providers are discovered via entry_points in _discover_entrypoint_providers()
        logger.debug(f"STT provider '{name}' not a built-in; will check entry_points.")
        return None

    def register_tts(self, provider: TTSProvider) -> None:
        """Register a TTS provider.

        Args:
            provider: TTSProvider instance to register.
        """
        self._tts_providers[provider.name] = provider

    def register_stt(self, provider: STTProvider) -> None:
        """Register an STT provider.

        Args:
            provider: STTProvider instance to register.
        """
        self._stt_providers[provider.name] = provider

    def get_tts(self, name: str) -> Optional[TTSProvider]:
        """Get a TTS provider by name.

        Args:
            name: Provider name.

        Returns:
            TTSProvider if found, None otherwise.
        """
        return self._tts_providers.get(name)

    def get_stt(self, name: str) -> Optional[STTProvider]:
        """Get an STT provider by name.

        Args:
            name: Provider name.

        Returns:
            STTProvider if found, None otherwise.
        """
        return self._stt_providers.get(name)

    def get_local_tts(self) -> list[TTSProvider]:
        """Return only local (privacy-safe) TTS providers."""
        return [p for p in self._tts_providers.values() if p.is_local]

    def get_local_stt(self) -> list[STTProvider]:
        """Return only local (privacy-safe) STT providers."""
        return [p for p in self._stt_providers.values() if p.is_local]

    def list_tts_providers(self) -> list[str]:
        """List registered TTS provider names."""
        return list(self._tts_providers.keys())

    def list_stt_providers(self) -> list[str]:
        """List registered STT provider names."""
        return list(self._stt_providers.keys())

    # ------------------------------------------------------------------
    # Conversation (speech-to-speech) providers — see #725
    # ------------------------------------------------------------------

    def register_conversation(self, provider: ConversationProvider) -> None:
        """Register a ConversationProvider."""
        self._conversation_providers[provider.name] = provider

    def get_conversation(self, name: str) -> Optional[ConversationProvider]:
        """Get a ConversationProvider by name."""
        return self._conversation_providers.get(name)

    def list_conversation_providers(self) -> list[str]:
        """List registered ConversationProvider names."""
        return list(self._conversation_providers.keys())

    def get_local_conversation(self) -> list[ConversationProvider]:
        """Return only local (privacy-safe) conversation providers.

        Currently always empty in practice — realtime speech-to-speech models
        are cloud-only. Kept symmetric with ``get_local_tts``/``get_local_stt``
        for future local realtime support.
        """
        return [p for p in self._conversation_providers.values() if p.is_local]
