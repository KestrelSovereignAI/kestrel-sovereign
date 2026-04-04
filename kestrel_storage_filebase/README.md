# kestrel-storage-filebase

Filebase S3-compatible IPFS storage provider for Kestrel Sovereign. Provides IPFS storage through a familiar S3-compatible API, making it easy to store and retrieve sovereign data with automatic IPFS pinning.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-storage-filebase.git
```

## Dependencies

- `kestrel-sovereign-sdk[crypto]`
- `boto3>=1.35.0`

## Usage

Once installed, the `FilebaseProvider` is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.storage_providers` entry point.

## Configuration

| Variable | Description |
|----------|-------------|
| `FILEBASE_ACCESS_KEY` | Filebase S3 access key |
| `FILEBASE_SECRET_KEY` | Filebase S3 secret key |
| `FILEBASE_BUCKET` | Filebase bucket name |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
