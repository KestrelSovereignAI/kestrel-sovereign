# kestrel-sovereign-sdk — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
sdk/
├── pyproject.toml
├── README.md
└── kestrel_sdk/
    ├── features/          # Base Feature class, Tool, Hook interfaces
    ├── protocols/         # Protocol definitions for providers
    ├── utils/             # Shared utilities
    └── security/          # Encryption helpers (optional [crypto] extra)
```

## Entry Points

None — this is a library dependency, not a feature plugin.

## Key Files to Read First

1. `kestrel_sdk/features/base.py` — Feature, Tool, and Hook base classes
2. `kestrel_sdk/__init__.py` — Public API surface

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- This SDK must remain lightweight — avoid adding heavy dependencies
- All feature packages depend on this; breaking changes affect the entire ecosystem
- When adding new base classes, ensure backward compatibility
