"""Private capability payloads never enter the canonical plaintext audit sinks."""

import json

import aiosqlite
import pytest

from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore, ToolDispatchEntry
from kestrel_sovereign.features.security.permissions import PermissionStore
from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.security.access_log import private_capability_tool_name
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


@pytest.mark.parametrize("name", [
    "artifact_publish_text", "artifact_list", "visual_computer_action", "visual_plan", "visual_computer_step",
    "first_owner__artifact_get", "feature_dispatch:visual_plan",
])
@pytest.mark.parametrize("mode", ["normal", "public", "anonymous", "ephemeral", "isolated"])
@pytest.mark.asyncio
async def test_private_dispatch_metrics_survive_without_payload_or_error_text(tmp_path, name, mode):
    backend = SQLiteBackend(str(tmp_path / "audit.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    store.set_privacy_config_provider(lambda: get_privacy_preset(mode))
    private = "PRIVATE-CONTENT-SENTINEL"
    try:
        await store.log_tool_dispatch(ToolDispatchEntry(
            agent_did="did:test:owner", session_id="session", turn_id="turn", tool_name=name,
            adapter="native", args_redacted={"text": private, "nested": {private: private}},
            result_status="error", error_class="ValueError", error_message=private,
            latency_ms=7, result_size_bytes=19,
        ), required=True)
        event = await store.log_tool_call("did:test:owner", name, metadata={"arguments": {"text": private}})
        await store.log_tool_response(event, False, 9, error_message=private)
        row = await backend.fetch_one("SELECT args_redacted,error_message,result_status,error_class,latency_ms,result_size_bytes FROM a2a_tool_dispatches")
        assert private not in str(tuple(row))
        assert row[1] is None and tuple(row[2:]) == ("error", "ValueError", 7, 19)
        event_row = await backend.fetch_one("SELECT metadata,error_message,event_type,duration_ms FROM a2a_observability WHERE id=?", (event,))
        assert private not in str(tuple(event_row))
        assert event_row[1] is None and tuple(event_row[2:]) == ("tool_response", 9)
        assert json.loads(row[0])  # Explicit elision marker, not missing audit.
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_security_decision_elides_private_arguments_at_its_actual_write(tmp_path):
    store = PermissionStore(str(tmp_path / "security.db"))
    await store.initialize()
    await store.log_decision("ArtifactsFeature", "artifact_publish_text", "tool_execution", "allowed", args_summary="PRIVATE-ARTIFACT-TITLE-AND-SOURCE")
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute("SELECT tool_name,decision,args_summary FROM security_audit_log")
        row = await cursor.fetchone()
    assert row[:2] == ("artifact_publish_text", "allowed")
    assert json.loads(row[2]) == {"_private_capability": True}


def test_other_tool_diagnostics_keep_existing_policy():
    assert not private_capability_tool_name("github_search")
    assert not private_capability_tool_name(None)
