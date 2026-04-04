# kestrel-storage-lighthouse — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel_storage_lighthouse/
├── pyproject.toml
├── README.md
└── kestrel_storage_lighthouse/
    ├── __init__.py
    ├── lighthouse_provider.py    # LighthouseProvider entry point
    └── lighthouse_rest.py        # Lighthouse REST API client
```

## Entry Points

- `kestrel_sovereign.storage_providers`: `LighthouseProvider = "kestrel_storage_lighthouse.lighthouse_provider:LighthouseProvider"`

## Key Files to Read First

1. `kestrel_storage_lighthouse/lighthouse_provider.py` — Main provider implementation
2. `kestrel_storage_lighthouse/lighthouse_rest.py` — REST API client

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Requires `LIGHTHOUSE_API_KEY` for IPFS pinning access
- This is a storage provider, not a feature — registered under `kestrel_sovereign.storage_providers`
- Uses SDK crypto extras for encryption at rest
