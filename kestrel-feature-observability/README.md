# kestrel-feature-observability

Operational monitoring, wellness metrics, and telemetry for Kestrel Sovereign agents. Provides two features: **ObservabilityFeature** for lifecycle event logging via hooks, and **WellnessFeature** for 5-dimension operational health monitoring (constitutional friction, context pressure, interaction depth, session continuity, memory health).

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-feature-observability.git
```

With OpenTelemetry and Prometheus:

```bash
uv pip install "kestrel-feature-observability[all] @ git+https://github.com/KestrelSovereignAI/kestrel-feature-observability.git"
```

## Dependencies

- `kestrel-sovereign-sdk`
- Optional: `opentelemetry-*` packages (via `[opentelemetry]`), `prometheus-client` (via `[prometheus]`)

## Usage

Once installed, both `ObservabilityFeature` and `WellnessFeature` are automatically discovered by kestrel-sovereign via the `kestrel_sovereign.features` entry point.

### Wellness Commands

- `!wellness` — Run wellness check across all 5 dimensions
- `!wellness-history` — View wellness trends over time
- `!wellness-export` — Export wellness data

## Configuration

| Variable | Description |
|----------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry collector endpoint (optional) |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e ".[all]"
uv run pytest
```
