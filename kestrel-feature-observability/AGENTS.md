# kestrel-feature-observability — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel-feature-observability/
├── pyproject.toml
├── README.md
├── kestrel_feature_observability/
│   ├── __init__.py
│   ├── metrics.py               # Prometheus metrics definitions
│   ├── telemetry.py             # OpenTelemetry tracing setup
│   ├── endpoints/
│   │   ├── metrics.py           # /metrics HTTP endpoint
│   │   └── observability.py     # Observability API endpoints
│   ├── observability/
│   │   ├── feature.py           # ObservabilityFeature entry point
│   │   └── hook.py              # Lifecycle event hooks
│   └── wellness/
│       ├── feature.py           # WellnessFeature entry point
│       └── metrics.py           # 5-dimension wellness metrics
└── tests/
    ├── test_metrics.py
    └── test_telemetry.py
```

## Entry Points

- `kestrel_sovereign.features`: `ObservabilityFeature = "kestrel_feature_observability.observability.feature:ObservabilityFeature"`
- `kestrel_sovereign.features`: `WellnessFeature = "kestrel_feature_observability.wellness.feature:WellnessFeature"`

## Key Files to Read First

1. `kestrel_feature_observability/observability/feature.py` — Observability feature and hooks
2. `kestrel_feature_observability/wellness/feature.py` — Wellness feature and tools
3. `kestrel_feature_observability/wellness/metrics.py` — 5-dimension health metrics

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Wellness metrics are telemetry-only — they are NOT injected into agent reasoning
- ObservabilityFeature uses the hook system for event logging
- Prometheus and OpenTelemetry integrations are optional extras
