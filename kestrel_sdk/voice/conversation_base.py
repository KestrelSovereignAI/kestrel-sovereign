"""
ConversationProvider — speech-to-speech provider contract.

Sibling to :class:`TTSProvider` and :class:`STTProvider`, but with a
fundamentally different shape. Speech-to-speech models (OpenAI Realtime and
the Kestrel-agnostic realtime APIs that follow) own the **full conversational
turn**: audio in → LLM reasoning → audio out, all within one session. They
can't be modeled as ``bytes ↔ text`` transforms because:

* The LLM lives inside the session, not on the caller's side.
* Tool calls happen mid-utterance and need a live round-trip.
* Interruption (barge-in) is a first-class control, not an emergent property
  of chunk scheduling.
* Turn detection (voice activity, semantic VAD) is configured on the session
  and emits events rather than being polled.

This module defines:

* :class:`ConversationProvider` — factory for sessions.
* :class:`ConversationSession` — the live session handle.
* The :class:`ConversationEvent` union — everything a session can emit.
* :class:`AudioFormat`, :class:`TurnDetectionConfig`, :class:`ToolDef` —
  supporting types with no Kestrel-specific coupling.

Providers implementing this ABC register via the entry-points group
``kestrel_sovereign.conversation_providers``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional, Union


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioFormat:
    """Wire format for audio frames on a Realtime session.

    OpenAI Realtime accepts PCM16 at 24 kHz mono little-endian; other future
    providers may accept Opus or higher sample rates. Providers should accept
    the formats listed in :meth:`ConversationProvider.supported_audio_formats`
    (added when a second vendor joins); callers pick one per session.
    """

    sample_rate: int = 24000
    encoding: Literal["pcm16", "opus"] = "pcm16"
    channels: int = 1


@dataclass(frozen=True)
class TurnDetectionConfig:
    """How the session decides when the user is done speaking.

    ``mode`` values:

    * ``"server_vad"`` — provider-side voice activity detection; auto-commits
      the user's audio buffer on speech-stop.
    * ``"semantic_vad"`` — content-aware: waits for natural end-of-utterance
      cues in the partial transcript. Higher latency, fewer interruptions.
    * ``"none"`` — caller drives turn boundaries explicitly (rare; used when
      the frontend already has its own VAD).
    """

    mode: Literal["server_vad", "semantic_vad", "none"] = "server_vad"
    silence_ms: int = 500
    threshold: float = 0.5
    # When True, the session automatically starts producing a response when
    # the user's turn ends. Set False to let the agent explicitly trigger
    # responses (e.g. "the agent was interrupted; wait for the user to
    # prompt again").
    create_response: bool = True


@dataclass(frozen=True)
class ToolDef:
    """Tool exposed to the realtime session.

    Mirrors the shape of OpenAI's tool schema so providers can forward it
    verbatim. ``parameters_schema`` is a JSON Schema dict. We intentionally
    do NOT import the sovereign-side Tool/ToolCategory types here — the SDK
    stays dependency-free so third-party packages can register conversation
    providers without pulling in the whole agent runtime.
    """

    name: str
    description: str
    parameters_schema: dict


# ---------------------------------------------------------------------------
# Event union
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionCreatedEvent:
    session_id: str
    kind: Literal["session.created"] = "session.created"


@dataclass(frozen=True)
class SessionUpdatedEvent:
    session_id: str
    kind: Literal["session.updated"] = "session.updated"


@dataclass(frozen=True)
class SpeechStartedEvent:
    """User started speaking. Callers should flush playback for barge-in."""

    kind: Literal["input_audio.speech_started"] = "input_audio.speech_started"


@dataclass(frozen=True)
class SpeechStoppedEvent:
    """User stopped speaking. The session will now process their utterance."""

    kind: Literal["input_audio.speech_stopped"] = "input_audio.speech_stopped"


@dataclass(frozen=True)
class TranscriptDeltaEvent:
    """Partial transcript of the user's in-progress utterance."""

    text: str
    is_final: bool = False
    kind: Literal["input_audio.transcript_delta"] = "input_audio.transcript_delta"


@dataclass(frozen=True)
class TranscriptFinalEvent:
    """Final (committed) transcript of the user's utterance."""

    text: str
    kind: Literal["input_audio.transcript_final"] = "input_audio.transcript_final"


@dataclass(frozen=True)
class ResponseAudioDeltaEvent:
    """One chunk of agent-spoken audio (in the session's configured format)."""

    pcm_chunk: bytes
    kind: Literal["response.audio_delta"] = "response.audio_delta"


@dataclass(frozen=True)
class ResponseTextDeltaEvent:
    """One chunk of the agent's spoken text (running transcript for UI)."""

    text: str
    kind: Literal["response.text_delta"] = "response.text_delta"


@dataclass(frozen=True)
class ResponseDoneEvent:
    """The agent finished this response turn."""

    kind: Literal["response.done"] = "response.done"


@dataclass(frozen=True)
class ToolCallRequestedEvent:
    """The agent wants to invoke a registered tool.

    The caller executes the tool and calls
    :meth:`ConversationSession.commit_tool_result` with the matching
    ``call_id``.
    """

    call_id: str
    name: str
    arguments: dict
    kind: Literal["response.tool_call_requested"] = "response.tool_call_requested"


@dataclass(frozen=True)
class ErrorEvent:
    """Session-level error. May or may not be fatal; check ``code``."""

    message: str
    code: Optional[str] = None
    kind: Literal["error"] = "error"


ConversationEvent = Union[
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptDeltaEvent,
    TranscriptFinalEvent,
    ResponseAudioDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseDoneEvent,
    ToolCallRequestedEvent,
    ErrorEvent,
]


# ---------------------------------------------------------------------------
# ABCs
# ---------------------------------------------------------------------------


class ConversationSession(ABC):
    """Live speech-to-speech session. One per voice conversation turn/call.

    Lifecycle:

    1. Caller obtains via :meth:`ConversationProvider.create_session`.
    2. Call :meth:`send_audio` with user PCM chunks.
    3. Consume :meth:`receive` (async iterator) for events.
    4. On :class:`ToolCallRequestedEvent`: execute the tool, then
       :meth:`commit_tool_result`.
    5. On barge-in from the UI: :meth:`cancel_response`.
    6. End with :meth:`close`.

    Implementations must be safe to call :meth:`close` multiple times.
    """

    session_id: str

    @abstractmethod
    async def send_audio(self, pcm_chunk: bytes) -> None:
        """Append one PCM chunk to the input buffer.

        The chunk must match the session's configured :class:`AudioFormat`.
        """

    @abstractmethod
    def receive(self) -> AsyncIterator[ConversationEvent]:
        """Stream events from the session. Exits when the session closes."""

    @abstractmethod
    async def commit_tool_result(self, call_id: str, result: Any) -> None:
        """Return the result of a tool the session requested.

        ``call_id`` matches :class:`ToolCallRequestedEvent.call_id`. ``result``
        will be serialized via ``json.dumps`` by the provider.
        """

    @abstractmethod
    async def update_instructions(self, instructions: str) -> None:
        """Replace the session's ``instructions`` field mid-session.

        Used by the Realtime tag adapter (#724) to surface per-turn tag
        directives composed from ``[excited]``/``[whispering]``/etc.
        """

    @abstractmethod
    async def cancel_response(self) -> None:
        """Interrupt the in-flight agent response (barge-in).

        Implementations should stop emitting :class:`ResponseAudioDeltaEvent`
        as soon as possible; callers typically also flush their playback
        buffer.
        """

    @abstractmethod
    async def close(self) -> None:
        """Terminate the session. Idempotent."""


class ConversationProvider(ABC):
    """Factory for :class:`ConversationSession` instances.

    Registered via the ``kestrel_sovereign.conversation_providers``
    entry-points group. ``name`` matches the string the routing resolver
    (#723) stores in ``VoiceRoute.conversation_provider``.
    """

    name: str
    is_local: bool  # False for any cloud-backed provider; used by privacy gating.

    @abstractmethod
    async def create_session(
        self,
        *,
        voice: str,
        instructions: str,
        tools: list[ToolDef],
        turn_detection: TurnDetectionConfig,
        audio_format: AudioFormat,
    ) -> ConversationSession:
        """Open a new live session configured for this turn."""

    @abstractmethod
    async def discover_models(self) -> list[str]:
        """Runtime-discovered realtime-capable model IDs.

        Never hardcoded — providers call out to their vendor's ``/models``
        endpoint (filtered to realtime-capable) and cache. Empty list is
        allowed (provider discovered nothing; caller surfaces a useful
        error).
        """

    @abstractmethod
    async def list_voices(self) -> list[str]:
        """Runtime-discovered voice IDs this provider exposes.

        Same rule as :meth:`discover_models`: no hardcoded names. Returns
        bare voice IDs rather than :class:`VoiceInfo` to keep the SDK free
        of the metadata taxonomy — providers surface richer metadata via
        their TTS provider instance if they have one.
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """True if the provider can currently open sessions.

        Typically checks for API key presence and a cheap reachability probe.
        """
