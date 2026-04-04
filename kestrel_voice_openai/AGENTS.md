# kestrel-voice-openai — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel_voice_openai/
├── pyproject.toml
├── README.md
└── kestrel_voice_openai/
    ├── __init__.py
    ├── openai_tts.py    # OpenAITTSProvider — text-to-speech
    └── openai_stt.py    # OpenAISTTProvider — speech-to-text
```

## Entry Points

- `kestrel_sovereign.voice_providers`: `OpenAITTS = "kestrel_voice_openai.openai_tts:OpenAITTSProvider"`
- `kestrel_sovereign.voice_providers`: `OpenAISTT = "kestrel_voice_openai.openai_stt:OpenAISTTProvider"`

## Key Files to Read First

1. `kestrel_voice_openai/openai_tts.py` — TTS provider implementation
2. `kestrel_voice_openai/openai_stt.py` — STT provider implementation

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Requires `OPENAI_API_KEY` for API access
- This package provides both TTS and STT — two separate providers
- Registered under `kestrel_sovereign.voice_providers`, not `kestrel_sovereign.features`
