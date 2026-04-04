# kestrel-cloud-azure — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel_cloud_azure/
├── pyproject.toml
├── README.md
└── kestrel_cloud_azure/
    ├── __init__.py
    └── azure_container.py    # AzureContainerProvider implementation
```

## Entry Points

- `kestrel_sovereign.cloud_providers`: `AzureContainerProvider = "kestrel_cloud_azure.azure_container:AzureContainerProvider"`

## Key Files to Read First

1. `kestrel_cloud_azure/azure_container.py` — Complete provider implementation

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Requires Azure credentials configured via Azure Identity
- Container Apps are serverless — they scale to zero when idle
- This is a cloud provider, not a feature — registered under `kestrel_sovereign.cloud_providers`
