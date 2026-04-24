"""Kestrel SDK — Voice provider interfaces."""

from .base import TTSProvider, STTProvider, VoiceInfo, VoiceConfig, match_voice, split_sentences
from .conversation_base import (
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
)

__all__ = [
    "TTSProvider",
    "STTProvider",
    "VoiceInfo",
    "VoiceConfig",
    "match_voice",
    "split_sentences",
    # Conversation (speech-to-speech) provider contract.
    "ConversationProvider",
    "ConversationSession",
    "ConversationEvent",
    "AudioFormat",
    "TurnDetectionConfig",
    "ToolDef",
    # Event types.
    "SessionCreatedEvent",
    "SessionUpdatedEvent",
    "SpeechStartedEvent",
    "SpeechStoppedEvent",
    "TranscriptDeltaEvent",
    "TranscriptFinalEvent",
    "ResponseAudioDeltaEvent",
    "ResponseTextDeltaEvent",
    "ResponseDoneEvent",
    "ToolCallRequestedEvent",
    "ErrorEvent",
]
