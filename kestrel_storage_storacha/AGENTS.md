# kestrel-storage-storacha — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel_storage_storacha/
├── pyproject.toml
├── README.md
└── kestrel_storage_storacha/
    ├── __init__.py
    ├── storacha_provider.py    # StorachaProvider entry point
    ├── storacha_rest.py        # Storacha REST API client
    └── storacha_ucan.py        # UCAN authentication handling
```

## Entry Points

- `kestrel_sovereign.storage_providers`: `StorachaProvider = "kestrel_storage_storacha.storacha_provider:StorachaProvider"`

## Key Files to Read First

1. `kestrel_storage_storacha/storacha_provider.py` — Main provider implementation
2. `kestrel_storage_storacha/storacha_ucan.py` — UCAN-based authentication
3. `kestrel_storage_storacha/storacha_rest.py` — REST API client

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Requires UCAN delegation proofs for authentication
- This is a storage provider, not a feature — registered under `kestrel_sovereign.storage_providers`
- Uses SDK crypto extras for encryption at rest
