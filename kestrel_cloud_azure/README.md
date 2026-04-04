# kestrel-cloud-azure

Azure Container Apps cloud provider for Kestrel Sovereign. Enables deployment to Azure Container Apps for serverless container hosting with automatic scaling.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-cloud-azure.git
```

## Dependencies

- `kestrel-sovereign-sdk`
- `azure-mgmt-appcontainers>=3.0.0`
- `azure-identity>=1.15.0`

## Usage

Once installed, the `AzureContainerProvider` is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.cloud_providers` entry point.

## Configuration

| Variable | Description |
|----------|-------------|
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `AZURE_RESOURCE_GROUP` | Azure resource group name |

Authentication uses Azure Identity defaults (managed identity, CLI, environment variables).

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
