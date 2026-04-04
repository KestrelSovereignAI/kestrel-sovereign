# kestrel-voice-openai

OpenAI text-to-speech and speech-to-text voice providers for Kestrel Sovereign. Provides both TTS and STT capabilities via the OpenAI API, enabling full voice interaction with agents.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-voice-openai.git
```

## Dependencies

- `kestrel-sovereign-sdk`
- `openai>=1.93.2`

## Usage

Once installed, both `OpenAITTS` and `OpenAISTT` providers are automatically discovered by kestrel-sovereign via the `kestrel_sovereign.voice_providers` entry point.

## Configuration

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
