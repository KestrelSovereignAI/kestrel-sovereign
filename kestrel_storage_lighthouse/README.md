# kestrel-storage-lighthouse

Lighthouse IPFS pinning storage provider for Kestrel Sovereign. Provides persistent, decentralized storage via the Lighthouse pinning service, ensuring sovereign data remains available on IPFS with guaranteed persistence.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-storage-lighthouse.git
```

## Dependencies

- `kestrel-sovereign-sdk[crypto]`
- `httpx>=0.27.0`

## Usage

Once installed, the `LighthouseProvider` is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.storage_providers` entry point.

## Configuration

| Variable | Description |
|----------|-------------|
| `LIGHTHOUSE_API_KEY` | Lighthouse API key for IPFS pinning |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
