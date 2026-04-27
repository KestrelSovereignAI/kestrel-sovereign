"""
Unit tests for the ConversationProvider ABC.

The ABC describes the shape of speech-to-speech providers (OpenAI Realtime
and successors). These tests exercise the contract via a
:class:`FakeConversationProvider` that scripts a full session: audio in,
events out, a tool call round-trip, a mid-session instructions update, a
barge-in cancel, and a clean close. If the ABC is wrong, the fake can't
implement it cleanly — that's the signal.

No network. No real providers. Pure contract verification.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from kestrel_sdk.voice import (
    AudioFormat,
    ConversationEvent,
    ConversationProvider,
    ConversationSession,
    ErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseDoneEvent,
    ResponseTextDeltaEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    ToolCallRequestedEvent,
    ToolDef,
    TranscriptDeltaEvent,
    TranscriptFinalEvent,
    TurnDetectionConfig,
    VoiceInfo,
)
from kestrel_sovereign.voice.provider_registry import VoiceProviderRegistry


# ---------------------------------------------------------------------------
# Fake session + provider — scripts a canned conversation turn
# ---------------------------------------------------------------------------


class FakeConversationSession(ConversationSession):
    """A scripted session used only to prove the ABC contract.

    On creation, queues a canned event script to be consumed by ``receive()``.
    Callers can push audio and tool results; the fake records them for
    assertions and surfaces a ``response.done`` once the tool is committed.
    """

    def __init__(
        self,
        session_id: str,
        *,
        voice: str,
        instructions: str,
        tools: list[ToolDef],
        audio_format: AudioFormat,
    ) -> None:
        self.session_id = session_id
        self._queue: asyncio.Queue[ConversationEvent | None] = asyncio.Queue()
        self._closed = False
        self._audio_received: list[bytes] = []
        self._tool_results: dict[str, Any] = {}
        self._instructions = instructions
        self._instructions_updates: list[str] = []
        self._audio_format = audio_format
        self._voice = voice
        self._tools = tools
        self._canceled = False

    # -- public surface used by tests --
    @property
    def audio_received(self) -> list[bytes]:
        return self._audio_received

    @property
    def tool_results(self) -> dict[str, Any]:
        return self._tool_results

    @property
    def instructions_updates(self) -> list[str]:
        return self._instructions_updates

    @property
    def canceled(self) -> bool:
        return self._canceled

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def current_instructions(self) -> str:
        return self._instructions

    # -- enqueue helpers for the test script --
    async def push_event(self, event: ConversationEvent) -> None:
        await self._queue.put(event)

    async def end_stream(self) -> None:
        await self._queue.put(None)  # sentinel

    # -- ConversationSession implementation --
    async def send_audio(self, pcm_chunk: bytes) -> None:
        if self._closed:
            raise RuntimeError("send_audio after close")
        self._audio_received.append(pcm_chunk)

    async def receive(self) -> AsyncIterator[ConversationEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def commit_tool_result(self, call_id: str, result: Any) -> None:
        self._tool_results[call_id] = result

    async def update_instructions(self, instructions: str) -> None:
        self._instructions_updates.append(instructions)
        self._instructions = instructions

    async def cancel_response(self) -> None:
        self._canceled = True

    async def close(self) -> None:
        # Idempotent per the ABC docstring.
        self._closed = True


class FakeConversationProvider(ConversationProvider):
    name = "fake_realtime"
    is_local = False

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self.sessions_created: list[FakeConversationSession] = []

    async def create_session(
        self,
        *,
        voice: str,
        instructions: str,
        tools: list[ToolDef],
        turn_detection: TurnDetectionConfig,
        audio_format: AudioFormat,
    ) -> ConversationSession:
        session = FakeConversationSession(
            session_id=f"sess_{len(self.sessions_created)}",
            voice=voice,
            instructions=instructions,
            tools=tools,
            audio_format=audio_format,
        )
        self.sessions_created.append(session)
        # Emit session.created so callers can observe the handshake via the
        # normal event stream (matches OpenAI Realtime semantics).
        await session.push_event(SessionCreatedEvent(session_id=session.session_id))
        return session

    async def discover_models(self) -> list[str]:
        # A real provider would query /v1/models; the fake returns a fixed
        # set to prove the contract shape. The list contents are opaque to
        # the registry — no hardcoded-model policing applies to a fake.
        return ["fake-realtime-flagship", "fake-realtime-mini"]

    async def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(voice_id="fake-voice-a", name="Fake A", provider="fake_realtime"),
            VoiceInfo(voice_id="fake-voice-b", name="Fake B", provider="fake_realtime"),
        ]

    async def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> FakeConversationProvider:
    return FakeConversationProvider()


@pytest.fixture
def tool_echo() -> ToolDef:
    return ToolDef(
        name="echo",
        description="Return the input verbatim.",
        parameters_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


# ---------------------------------------------------------------------------
# Provider-level contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_exposes_name_and_is_local(provider: FakeConversationProvider) -> None:
    assert provider.name == "fake_realtime"
    assert provider.is_local is False


@pytest.mark.asyncio
async def test_provider_discovers_models_at_runtime(provider: FakeConversationProvider) -> None:
    models = await provider.discover_models()
    assert models
    assert all(isinstance(m, str) for m in models)


@pytest.mark.asyncio
async def test_provider_lists_voices_at_runtime(provider: FakeConversationProvider) -> None:
    voices = await provider.list_voices()
    assert voices
    assert all(isinstance(v, VoiceInfo) for v in voices)
    assert all(v.provider == provider.name for v in voices)


@pytest.mark.asyncio
async def test_provider_is_available(provider: FakeConversationProvider) -> None:
    assert await provider.is_available() is True


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_returns_session_with_id(
    provider: FakeConversationProvider, tool_echo: ToolDef
) -> None:
    session = await provider.create_session(
        voice="fake-voice-a",
        instructions="Speak warmly.",
        tools=[tool_echo],
        turn_detection=TurnDetectionConfig(),
        audio_format=AudioFormat(),
    )
    assert session.session_id == "sess_0"


@pytest.mark.asyncio
async def test_session_emits_session_created_first(
    provider: FakeConversationProvider, tool_echo: ToolDef
) -> None:
    session = await provider.create_session(
        voice="fake-voice-a",
        instructions="",
        tools=[],
        turn_detection=TurnDetectionConfig(),
        audio_format=AudioFormat(),
    )
    await session.end_stream()
    events = [e async for e in session.receive()]
    assert len(events) >= 1
    assert isinstance(events[0], SessionCreatedEvent)
    assert events[0].session_id == session.session_id


@pytest.mark.asyncio
async def test_send_audio_records_chunks(
    provider: FakeConversationProvider, tool_echo: ToolDef
) -> None:
    session = await provider.create_session(
        voice="fake-voice-a",
        instructions="",
        tools=[],
        turn_detection=TurnDetectionConfig(mode="server_vad"),
        audio_format=AudioFormat(sample_rate=24000, encoding="pcm16"),
    )
    await session.send_audio(b"\x00\x01\x02")
    await session.send_audio(b"\x03\x04\x05")
    assert session.audio_received == [b"\x00\x01\x02", b"\x03\x04\x05"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_close_is_idempotent(provider: FakeConversationProvider) -> None:
    session = await provider.create_session(
        voice="v",
        instructions="",
        tools=[],
        turn_detection=TurnDetectionConfig(),
        audio_format=AudioFormat(),
    )
    await session.close()
    await session.close()  # must not raise
    assert session.closed  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_send_audio_after_close_raises(
    provider: FakeConversationProvider,
) -> None:
    session = await provider.create_session(
        voice="v",
        instructions="",
        tools=[],
        turn_detection=TurnDetectionConfig(),
        audio_format=AudioFormat(),
    )
    await session.close()
    with pytest.raises(RuntimeError):
        await session.send_audio(b"\x00")


# ---------------------------------------------------------------------------
# Event stream — speech + response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_turn_event_sequence(
    provider: FakeConversationProvider, tool_echo: ToolDef
) -> None:
    session = await provider.create_session(
        voice="fake-voice-a",
        instructions="",
        tools=[tool_echo],
        turn_detection=TurnDetectionConfig(),
        audio_format=AudioFormat(),
    )
    # Script a full turn: user speaks, partial transcript, final transcript,
    # response audio + text, response done.
    await session.push_event(SpeechStartedEvent())
    await session.push_event(TranscriptDeltaEvent(text="Hello", is_final=False))
    await session.push_event(SpeechStoppedEvent())
    await session.push_event(TranscriptFinalEvent(text="Hello there"))
    await session.push_event(ResponseTextDeltaEvent(text="Hi!"))
    await session.push_event(ResponseAudioDeltaEvent(pcm_chunk=b"\xaa\xbb"))
    await session.push_event(ResponseDoneEvent())
    await session.end_stream()

    events = [e async for e in session.receive()]
    kinds = [e.kind for e in events]
    # SessionCreated is first (enqueued by create_session), then the scripted turn.
    assert kinds == [
        "session.created",
        "input_audio.speech_started",
        "input_audio.transcript_delta",
        "input_audio.speech_stopped",
        "input_audio.transcript_final",
        "response.text_delta",
        "response.audio_delta",
        "response.done",
    ]


# ---------------------------------------------------------------------------
# Tool call round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_requested_and_committed(
    provider: FakeConversationProvider, tool_echo: ToolDef
) -> None:
    session = await provider.create_session(
        voice="v",
        instructions="",
        tools=[tool_echo],
        turn_detection=TurnDetectionConfig(),
        audio_format=AudioFormat(),
    )
    await session.push_event(
        ToolCallRequestedEvent(
            call_id="call_abc",
            name="echo",
            arguments={"text": "hi"},
        )
    )
    await session.end_stream()

    events = [e async for e in session.receive()]
    tool_calls = [e for e in events if isinstance(e, ToolCallRequestedEvent)]
    assert len(tool_calls) == 1
    assert tool_calls[0].call_id == "call_abc"
    assert tool_calls[0].name == "echo"

    # Caller commits the result, session records it.
    await session.commit_tool_result("call_abc", {"text": "hi"})
    assert session.tool_results == {"call_abc": {"text": "hi"}}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Mid-session instruction updates (for the Realtime tag adapter, #724)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_instructions_replaces_session_field(
    provider: FakeConversationProvider,
) -> None:
    session = await provider.create_session(
        voice="v",
        instructions="Speak calmly.",
        tools=[],
        turn_detection=TurnDetectionConfig(),
        audio_format=AudioFormat(),
    )
    await session.update_instructions("Speak with excited energy.")
    assert session.current_instructions == "Speak with excited energy."  # type: ignore[attr-defined]
    assert session.instructions_updates == ["Speak with excited energy."]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Barge-in
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_response_marks_session_canceled(
    provider: FakeConversationProvider,
) -> None:
    session = await provider.create_session(
        voice="v",
        instructions="",
        tools=[],
        turn_detection=TurnDetectionConfig(),
        audio_format=AudioFormat(),
    )
    assert not session.canceled  # type: ignore[attr-defined]
    await session.cancel_response()
    assert session.canceled  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Error events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_event_is_surfaced_through_receive(
    provider: FakeConversationProvider,
) -> None:
    session = await provider.create_session(
        voice="v",
        instructions="",
        tools=[],
        turn_detection=TurnDetectionConfig(),
        audio_format=AudioFormat(),
    )
    await session.push_event(ErrorEvent(message="rate limit", code="429"))
    await session.end_stream()

    events = [e async for e in session.receive()]
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].message == "rate limit"
    assert errors[0].code == "429"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_registry_accepts_conversation_provider() -> None:
    registry = VoiceProviderRegistry({})
    provider = FakeConversationProvider()
    registry.register_conversation(provider)
    assert registry.get_conversation("fake_realtime") is provider
    assert "fake_realtime" in registry.list_conversation_providers()


def test_registry_list_is_empty_by_default() -> None:
    registry = VoiceProviderRegistry({})
    assert registry.list_conversation_providers() == []


def test_registry_get_local_conversation_filters_by_is_local() -> None:
    registry = VoiceProviderRegistry({})
    registry.register_conversation(FakeConversationProvider())  # is_local=False
    # A cloud provider should not appear in the local list.
    assert registry.get_local_conversation() == []


# ---------------------------------------------------------------------------
# Audio format defaults match OpenAI Realtime spec
# ---------------------------------------------------------------------------


def test_audio_format_defaults_pcm16_24khz_mono() -> None:
    fmt = AudioFormat()
    assert fmt.sample_rate == 24000
    assert fmt.encoding == "pcm16"
    assert fmt.channels == 1


def test_turn_detection_defaults_server_vad() -> None:
    td = TurnDetectionConfig()
    assert td.mode == "server_vad"
    assert td.create_response is True


def test_tool_def_fields() -> None:
    t = ToolDef(name="t", description="d", parameters_schema={"type": "object"})
    assert t.name == "t"
    assert t.description == "d"
    assert t.parameters_schema == {"type": "object"}
