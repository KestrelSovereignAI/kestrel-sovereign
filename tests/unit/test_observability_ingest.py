"""Tests for POST /api/observability/events ingest endpoint (#2062)."""
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.endpoints.observability import router as observability_router
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


@pytest_asyncio.fixture
async def store(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "obs-ingest.db"))
    await backend.connect()
    s = ObservabilityStore(backend)
    await s.initialize()
    return s


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(observability_router)
    app.state.agent = type("A", (), {"observability_store": store})()
    return TestClient(app)


def test_post_then_get_round_trip(client):
    body = {
        "agent_name": "talon:kestrel-sovereign#2051",
        "session_id": "issue-2051-soft-delete",
        "event_type": "tool_call",
        "tool_name": "Bash",
        "metadata": {"hook_event_type": "PreToolUse"},
    }
    resp = client.post("/api/observability/events", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["count"] == 1
    assert data["event_ids"] and not data["errors"]

    got = client.get(
        "/api/observability/events",
        params={"session_id": "issue-2051-soft-delete"},
    )
    assert got.status_code == 200
    events = got.json()["events"]
    assert len(events) == 1
    assert events[0]["event_id"] == data["event_ids"][0]
    assert events[0]["agent_name"] == "talon:kestrel-sovereign#2051"
    assert events[0]["tool_name"] == "Bash"
    assert events[0]["event_type"] == "tool_call"


def test_post_batch(client):
    body = {
        "events": [
            {
                "agent_name": "talon:x",
                "session_id": "sess-batch",
                "event_type": "tool_response",
                "tool_name": "Read",
                "duration_ms": 42,
                "success": True,
            },
            {
                "agent_name": "talon:x",
                "session_id": "sess-batch",
                "event_type": "error",
                "error_message": "boom",
                "metadata": {"error_type": "explosion"},
            },
            {
                "agent_name": "talon:x",
                "session_id": "sess-batch",
                "event_type": "metric",
                "metadata": {"metric_name": "files_touched", "metric_value": 3},
            },
        ]
    }
    resp = client.post("/api/observability/events", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 3

    events = client.get(
        "/api/observability/events", params={"session_id": "sess-batch"}
    ).json()["events"]
    types = {e["event_type"] for e in events}
    assert types == {"tool_response", "error", "metric"}

    tool_resp = next(e for e in events if e["event_type"] == "tool_response")
    assert tool_resp["duration_ms"] == 42
    assert tool_resp["success"] is True

    err = next(e for e in events if e["event_type"] == "error")
    assert err["error_message"] == "boom"
    assert err["metadata"]["error_type"] == "explosion"


def test_pushed_timestamp_is_preserved(client):
    # External producer emits telemetry after the fact: the stored row must
    # keep the pushed event time, not be re-stamped with "now".
    pushed = "2026-06-30T17:00:00+00:00"
    body = {
        "agent_name": "talon:x",
        "session_id": "sess-ts",
        "event_type": "tool_call",
        "tool_name": "Bash",
        "timestamp": "2026-06-30T17:00:00Z",
    }
    resp = client.post("/api/observability/events", json=body)
    assert resp.status_code == 200, resp.text

    events = client.get(
        "/api/observability/events", params={"session_id": "sess-ts"}
    ).json()["events"]
    assert len(events) == 1
    from datetime import datetime

    assert datetime.fromisoformat(events[0]["timestamp"]) == datetime.fromisoformat(pushed)


def test_pushed_timestamp_preserved_for_metric(client):
    pushed = "2026-06-30T17:00:00+00:00"
    body = {
        "agent_name": "talon:x",
        "session_id": "sess-ts-metric",
        "event_type": "metric",
        "metadata": {"metric_name": "files_touched", "metric_value": 3},
        "timestamp": "2026-06-30T17:00:00Z",
    }
    resp = client.post("/api/observability/events", json=body)
    assert resp.status_code == 200, resp.text

    events = client.get(
        "/api/observability/events", params={"session_id": "sess-ts-metric"}
    ).json()["events"]
    assert len(events) == 1
    from datetime import datetime

    assert datetime.fromisoformat(events[0]["timestamp"]) == datetime.fromisoformat(pushed)


def test_malformed_timestamp_returns_422(client):
    resp = client.post(
        "/api/observability/events",
        json={
            "agent_name": "a",
            "session_id": "s",
            "event_type": "tool_call",
            "timestamp": "not-a-timestamp",
        },
    )
    assert resp.status_code == 422


def test_unknown_event_type_returns_422(client):
    resp = client.post(
        "/api/observability/events",
        json={
            "agent_name": "a",
            "session_id": "s",
            "event_type": "not_a_real_type",
        },
    )
    assert resp.status_code == 422


def test_missing_required_fields_returns_422(client):
    # session_id is required.
    resp = client.post(
        "/api/observability/events",
        json={"agent_name": "a", "event_type": "tool_call"},
    )
    assert resp.status_code == 422


def test_inbound_metadata_is_redacted(client):
    body = {
        "agent_name": "talon:x",
        "session_id": "sess-redact",
        "event_type": "tool_call",
        "tool_name": "Bash",
        "metadata": {"api_key": "sk-secret-value", "command": "ls"},
    }
    resp = client.post("/api/observability/events", json=body)
    assert resp.status_code == 200, resp.text

    events = client.get(
        "/api/observability/events", params={"session_id": "sess-redact"}
    ).json()["events"]
    assert len(events) == 1
    meta = events[0]["metadata"]
    assert meta["api_key"] == "<redacted>"
    assert meta["command"] == "ls"


def test_returns_503_without_store():
    app = FastAPI()
    app.include_router(observability_router)
    app.state.agent = type("A", (), {"observability_store": None})()
    with TestClient(app) as c:
        resp = c.post(
            "/api/observability/events",
            json={"agent_name": "a", "session_id": "s", "event_type": "tool_call"},
        )
    assert resp.status_code == 503
