# kestrel-voice-deepgram

Deepgram speech-to-text voice provider for Kestrel Sovereign. Provides real-time speech recognition for agent voice input using the Deepgram API.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-voice-deepgram.git
```

## Dependencies

- `kestrel-sovereign-sdk`
- `deepgram-sdk>=4.0.0`

## Usage

Once installed, the `DeepgramSTT` provider is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.voice_providers` entry point.

## Configuration

| Variable | Description |
|----------|-------------|
| `DEEPGRAM_API_KEY` | Deepgram API key |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
