# kestrel-feature-visual — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
packages/kestrel-feature-visual/
├── pyproject.toml
├── README.md
└── src/
    └── kestrel_feature_visual/
        ├── __init__.py
        └── feature.py    # VisualIdentityFeature — avatar/selfie/LoRA tools
```

## Entry Points

- `kestrel_sovereign.features`: `VisualIdentityFeature = "kestrel_feature_visual.feature:VisualIdentityFeature"`

## Key Files to Read First

1. `src/kestrel_feature_visual/feature.py` — Complete feature implementation

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Uses src layout — imports are from `kestrel_feature_visual`, not `src.kestrel_feature_visual`
- Requires `REPLICATE_API_TOKEN` for image generation
- LoRA training creates custom models for visual consistency
