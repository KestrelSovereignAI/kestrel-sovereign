"""Agent reads stay in core; fleet ingestion/query belongs to the host feature.

The core observability router used to register an agent-scoped
``GET /api/observability/events`` that shadowed the fleet host feature's
tenant-aware query route (#2317). Core now cedes ``/events`` entirely to the
fleet host feature and keeps only the per-agent ``/summary`` and
``/metrics/{name}`` views.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.endpoints.observability import router


def test_core_does_not_register_events_route():
    app = FastAPI()
    app.include_router(router)
    app.state.agent = type("Agent", (), {"observability_store": None})()

    with TestClient(app) as client:
        # The fleet host feature owns /events now; core must not shadow it.
        assert client.get("/api/observability/events").status_code == 404
        assert client.post("/api/observability/events").status_code == 404


def test_core_keeps_summary_and_metrics_but_not_events_or_host_namespace():
    paths = {route.path for route in router.routes}

    assert "/api/observability/events" not in paths
    assert "/api/observability/summary" in paths
    assert "/api/observability/metrics/{metric_name}" in paths
    assert not any(path.startswith("/api/host/observability") for path in paths)
