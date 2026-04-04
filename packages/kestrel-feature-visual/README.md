# kestrel-feature-visual

Visual identity for Kestrel Sovereign agents — avatar generation, selfies, and LoRA training for character consistency. Uses Replicate for image generation with support for training custom models to maintain a consistent visual identity across generated images.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-feature-visual.git
```

## Dependencies

- `kestrel-sovereign-sdk`
- `kestrel-sovereign`
- `replicate>=1.0.4`
- `httpx>=0.27.0`

## Usage

Once installed, the `VisualIdentityFeature` is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.features` entry point.

## Configuration

| Variable | Description |
|----------|-------------|
| `REPLICATE_API_TOKEN` | Replicate API token for image generation |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e ".[test]"
uv run pytest
```
