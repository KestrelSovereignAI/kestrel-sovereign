# kestrel-storage-storacha

Storacha (web3.storage) storage provider for Kestrel Sovereign. Provides decentralized storage with UCAN-based authentication and content addressing, ensuring sovereign data ownership through cryptographic delegation proofs.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-storage-storacha.git
```

## Dependencies

- `kestrel-sovereign-sdk[crypto]`
- `httpx>=0.27.0`
- `cbor2>=5.6.0`

## Usage

Once installed, the `StorachaProvider` is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.storage_providers` entry point.

## Configuration

| Variable | Description |
|----------|-------------|
| `STORACHA_PRINCIPAL_KEY` | Storacha principal key for authentication |
| `STORACHA_DELEGATION_PROOF` | UCAN delegation proof |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
