# kestrel-cloud-vastai — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel_cloud_vastai/
├── pyproject.toml
├── README.md
└── kestrel_cloud_vastai/
    ├── __init__.py
    ├── feature.py          # VastAIFeature entry point
    ├── core.py             # Core Vast.ai operations
    ├── http_api.py         # HTTP API client
    ├── manager.py          # Instance lifecycle management
    ├── models.py           # Data models and GPU profiles
    ├── ssh_training.py     # SSH-based training workflows
    └── workflows.py        # Compute workflow orchestration
```

## Entry Points

- `kestrel_sovereign.features`: `VastAIFeature = "kestrel_cloud_vastai.feature:VastAIFeature"`

## Key Files to Read First

1. `kestrel_cloud_vastai/feature.py` — Main feature class and tools
2. `kestrel_cloud_vastai/manager.py` — Instance lifecycle management
3. `kestrel_cloud_vastai/models.py` — GPU profiles and instance models

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Requires `VASTAI_API_KEY` for marketplace access
- GPU instances incur costs — always clean up after use
- Instance search finds best price/performance GPU offers
