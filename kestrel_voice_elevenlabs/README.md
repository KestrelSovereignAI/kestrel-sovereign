# kestrel-voice-elevenlabs

ElevenLabs text-to-speech voice provider for Kestrel Sovereign. Provides high-quality, natural-sounding voice output for agent interactions using the ElevenLabs API.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-voice-elevenlabs.git
```

## Dependencies

- `kestrel-sovereign-sdk`
- `elevenlabs>=1.0.0`

## Usage

Once installed, the `ElevenLabsTTS` provider is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.voice_providers` entry point.

## Configuration

| Variable | Description |
|----------|-------------|
| `ELEVENLABS_API_KEY` | ElevenLabs API key |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
