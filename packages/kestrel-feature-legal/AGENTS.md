# kestrel-feature-legal — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
packages/kestrel-feature-legal/
├── pyproject.toml
├── README.md
└── src/
    └── kestrel_feature_legal/
        ├── __init__.py
        ├── incorporate_tool.py       # Incorporation tool for agents
        ├── models.py                 # Legal entity data models
        ├── operating_agreement.py    # Operating Agreement generator
        └── wyoming_dao.py            # Articles of Organization generator
```

## Entry Points

None currently registered — this package is used as a library by other features.

## Key Files to Read First

1. `src/kestrel_feature_legal/wyoming_dao.py` — Articles of Organization generation
2. `src/kestrel_feature_legal/operating_agreement.py` — Operating Agreement generation
3. `src/kestrel_feature_legal/incorporate_tool.py` — Agent incorporation tool

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Uses src layout — imports are from `kestrel_feature_legal`, not `src.kestrel_feature_legal`
- Generated documents map Kestrel constitutional governance to Wyoming legal structures
- Legal document generation is deterministic given the same inputs
