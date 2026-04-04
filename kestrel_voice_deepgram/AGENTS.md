# kestrel-voice-deepgram — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel_voice_deepgram/
├── pyproject.toml
├── README.md
└── kestrel_voice_deepgram/
    ├── __init__.py
    └── deepgram_stt.py    # DeepgramSTTProvider implementation
```

## Entry Points

- `kestrel_sovereign.voice_providers`: `DeepgramSTT = "kestrel_voice_deepgram.deepgram_stt:DeepgramSTTProvider"`

## Key Files to Read First

1. `kestrel_voice_deepgram/deepgram_stt.py` — Complete STT provider implementation

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Requires `DEEPGRAM_API_KEY` for API access
- This is a voice provider, not a feature — registered under `kestrel_sovereign.voice_providers`
