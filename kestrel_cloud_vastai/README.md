# kestrel-cloud-vastai

Vast.ai GPU marketplace cloud provider for Kestrel Sovereign. Provides access to the GPU marketplace for cost-effective compute provisioning, with instance search, creation, SSH training workflows, and lifecycle management.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-cloud-vastai.git
```

## Dependencies

- `kestrel-sovereign-sdk`
- `kestrel-sovereign`
- `vastai-sdk>=0.1.0`

## Usage

Once installed, the `VastAIFeature` is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.features` entry point.

## Configuration

| Variable | Description |
|----------|-------------|
| `VASTAI_API_KEY` | Vast.ai API key |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
