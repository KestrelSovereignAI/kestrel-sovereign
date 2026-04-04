# kestrel-feature-spawn — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel-feature-spawn/
├── pyproject.toml
├── README.md
└── src/
    └── kestrel_feature_spawn/
        ├── __init__.py
        ├── spawn/
        │   ├── feature.py              # SpawnFeature entry point
        │   ├── lifecycle.py            # Child agent lifecycle management
        │   ├── mandate.py              # Task mandate/delegation
        │   ├── scoped_constitution.py  # Constitutional scope for children
        │   ├── delegated_wallet.py     # Wallet delegation to children
        │   └── endpoints.py            # Spawn API endpoints
        └── bridge/
            ├── feature.py              # BridgeFeature entry point
            ├── protocol.py             # Bridge communication protocol
            └── router.py               # Message routing
```

## Entry Points

- `kestrel_sovereign.features`: `SpawnFeature = "kestrel_feature_spawn.spawn.feature:SpawnFeature"`
- `kestrel_sovereign.features`: `BridgeFeature = "kestrel_feature_spawn.bridge.feature:BridgeFeature"`

## Key Files to Read First

1. `src/kestrel_feature_spawn/spawn/feature.py` — Spawn feature and tools
2. `src/kestrel_feature_spawn/spawn/lifecycle.py` — Child agent lifecycle
3. `src/kestrel_feature_spawn/bridge/feature.py` — Bridge feature and tools

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Uses src layout — imports are from `kestrel_feature_spawn`, not `src.kestrel_feature_spawn`
- Child agents inherit a scoped constitution from their parent
- Wallet delegation requires the `[wallet]` extra
