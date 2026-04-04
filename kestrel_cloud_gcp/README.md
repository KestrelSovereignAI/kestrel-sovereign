# kestrel-cloud-gcp

GCP Compute Engine and Cloud Run cloud providers for Kestrel Sovereign. Manages GPU instance provisioning on Compute Engine for training workloads and serverless deployment via Cloud Run that scales to zero.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-cloud-gcp.git
```

## Dependencies

- `kestrel-sovereign-sdk`
- `kestrel-sovereign`
- `google-cloud-compute>=1.15.0`
- `google-cloud-run>=0.10.0`
- `google-cloud-logging>=3.5.0`

## Usage

Once installed, `GCPComputeFeature` and `CloudRunProvider` are automatically discovered by kestrel-sovereign via entry points.

## Configuration

| Variable | Description |
|----------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON |
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | GCP region (e.g., `us-central1`) |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
