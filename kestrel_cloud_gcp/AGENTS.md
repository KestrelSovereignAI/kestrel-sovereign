# kestrel-cloud-gcp — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel_cloud_gcp/
├── pyproject.toml
├── README.md
└── kestrel_cloud_gcp/
    ├── __init__.py
    ├── cloudrun.py             # CloudRunProvider entry point
    └── compute/
        ├── __init__.py
        ├── feature.py          # GCPComputeFeature entry point
        ├── core.py             # Core compute operations
        ├── manager.py          # Instance lifecycle management
        ├── models.py           # GPU profiles and data models
        ├── ssh_training.py     # SSH-based training workflows
        └── workflows.py        # Compute workflow orchestration
```

## Entry Points

- `kestrel_sovereign.features`: `GCPComputeFeature = "kestrel_cloud_gcp.compute.feature:GCPComputeFeature"`
- `kestrel_sovereign.cloud_providers`: `CloudRunProvider = "kestrel_cloud_gcp.cloudrun:CloudRunProvider"`

## Key Files to Read First

1. `kestrel_cloud_gcp/compute/feature.py` — GCP Compute feature and tools
2. `kestrel_cloud_gcp/cloudrun.py` — Cloud Run provider
3. `kestrel_cloud_gcp/compute/manager.py` — Instance lifecycle

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Requires valid GCP credentials and project configuration
- GPU instances incur costs — always clean up after use
- Cloud Run scales to zero by default for cost efficiency
