# kestrel-storage-filebase — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel_storage_filebase/
├── pyproject.toml
├── README.md
└── kestrel_storage_filebase/
    ├── __init__.py
    └── filebase_provider.py    # FilebaseProvider implementation
```

## Entry Points

- `kestrel_sovereign.storage_providers`: `FilebaseProvider = "kestrel_storage_filebase.filebase_provider:FilebaseProvider"`

## Key Files to Read First

1. `kestrel_storage_filebase/filebase_provider.py` — Complete provider implementation

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Requires Filebase S3 credentials for storage access
- This is a storage provider, not a feature — registered under `kestrel_sovereign.storage_providers`
- Uses S3-compatible API via boto3 — familiar to AWS developers
