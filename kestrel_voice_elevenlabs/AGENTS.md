# kestrel-voice-elevenlabs — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel_voice_elevenlabs/
├── pyproject.toml
├── README.md
└── kestrel_voice_elevenlabs/
    ├── __init__.py
    └── elevenlabs_tts.py    # ElevenLabsTTSProvider implementation
```

## Entry Points

- `kestrel_sovereign.voice_providers`: `ElevenLabsTTS = "kestrel_voice_elevenlabs.elevenlabs_tts:ElevenLabsTTSProvider"`

## Key Files to Read First

1. `kestrel_voice_elevenlabs/elevenlabs_tts.py` — Complete TTS provider implementation

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Requires `ELEVENLABS_API_KEY` for API access
- This is a voice provider, not a feature — registered under `kestrel_sovereign.voice_providers`
