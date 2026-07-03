"""Privacy-aware gating for the A2A observability sink (F076 / #2097).

Tool-call arguments and metadata are user content. Before this fix they
persisted to ``a2a_tool_dispatches`` / ``a2a_observability`` in EVERY privacy
mode — including EPHEMERAL ("nothing stored") and ANONYMOUS (no redaction).

These tests pin the layer-boundary behaviour:
- EPHEMERAL / ISOLATED: arg/metadata payloads are elided (content-free
  counts/latency/status still record).
- ANONYMOUS: persisted arg/metadata content is anonymized.
- NORMAL: payloads persist as before.
- ``purge_observability_since`` sweeps rows since the ephemeral watermark, and
  ``purge_ephemeral_session`` drives it through the storage wrapper.
"""

import json

import pytest

from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityStore,
    ToolDispatchEntry,
    _CONTENT_GATED_MARKER,
)
from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


def _pii_entry(**overrides) -> ToolDispatchEntry:
    base = dict(
        agent_did="did:test:emma",
        session_id="sess-1",
        turn_id="turn_1",
        tool_name="github",
        adapter="cli.github",
        args_redacted={
            "token": "secret-token",
            "note": "email me at alice@example.com",
        },
        result_status="success",
        error_class=None,
        error_message=None,
        latency_ms=42,
        result_size_bytes=17,
    )
    base.update(overrides)
    return ToolDispatchEntry(**base)


async def _store(tmp_path, name):
    backend = SQLiteBackend(str(tmp_path / name))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    return backend, store


async def _dispatch_row(backend):
    return await backend.fetch_one(
        "SELECT args_redacted, result_status, latency_ms FROM a2a_tool_dispatches "
        "WHERE agent_did=?",
        ("did:test:emma",),
    )


@pytest.mark.asyncio
async def test_ephemeral_elides_tool_dispatch_args_but_keeps_metrics(tmp_path):
    backend, store = await _store(tmp_path, "ephemeral.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("ephemeral"))
        await store.log_tool_dispatch(_pii_entry())

        row = await _dispatch_row(backend)
        assert row is not None, "row itself still persists (content-free metrics)"
        args = json.loads(row[0])
        assert _CONTENT_GATED_MARKER in args
        assert "alice@example.com" not in row[0]
        assert "secret-token" not in row[0]
        # counts / status / latency still metered
        assert row[1] == "success"
        assert row[2] == 42
        summary = await store.tool_failure_rate("did:test:emma")
        assert summary["total_calls"] == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_isolated_elides_tool_dispatch_args(tmp_path):
    backend, store = await _store(tmp_path, "isolated.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("isolated"))
        await store.log_tool_dispatch(_pii_entry())

        row = await _dispatch_row(backend)
        args = json.loads(row[0])
        assert _CONTENT_GATED_MARKER in args
        assert "alice@example.com" not in row[0]
        assert row[2] == 42  # latency still recorded
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_anonymous_redacts_tool_dispatch_args(tmp_path):
    backend, store = await _store(tmp_path, "anonymous.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("anonymous"))
        await store.log_tool_dispatch(_pii_entry())

        row = await _dispatch_row(backend)
        args = json.loads(row[0])
        # PII anonymized...
        assert "alice@example.com" not in row[0]
        # ...but the structure survives and secret-key redaction still applies
        assert args["token"] == "<redacted>"
        assert "note" in args
        assert row[1] == "success"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_normal_persists_tool_dispatch_args(tmp_path):
    backend, store = await _store(tmp_path, "normal.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("normal"))
        await store.log_tool_dispatch(_pii_entry())

        row = await _dispatch_row(backend)
        args = json.loads(row[0])
        # Normal mode keeps content (secret keys still redacted as before).
        assert args["note"] == "email me at alice@example.com"
        assert args["token"] == "<redacted>"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_no_provider_is_ungated(tmp_path):
    backend, store = await _store(tmp_path, "ungated.db")
    try:
        await store.log_tool_dispatch(_pii_entry())
        row = await _dispatch_row(backend)
        args = json.loads(row[0])
        assert args["note"] == "email me at alice@example.com"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_log_tool_call_metadata_elided_in_ephemeral(tmp_path):
    backend, store = await _store(tmp_path, "toolcall.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("ephemeral"))
        event_id = await store.log_tool_call(
            agent_name="did:test:emma",
            tool_name="github",
            session_id="sess-1",
            metadata={"pii": "alice@example.com"},
        )
        row = await backend.fetch_one(
            "SELECT metadata, tool_name FROM a2a_observability WHERE id=?",
            (event_id,),
        )
        assert "alice@example.com" not in row[0]
        assert _CONTENT_GATED_MARKER in json.loads(row[0])
        assert row[1] == "github"  # content-free metering intact
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_log_agent_response_metadata_elided_in_ephemeral(tmp_path):
    backend, store = await _store(tmp_path, "agentresp.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("ephemeral"))
        event_id = await store.log_agent_response(
            agent_name="did:test:emma",
            duration_ms=11,
            session_id="sess-1",
            metadata={"pii": "alice@example.com"},
        )
        row = await backend.fetch_one(
            "SELECT metadata, duration_ms FROM a2a_observability WHERE id=?",
            (event_id,),
        )
        assert "alice@example.com" not in row[0]
        assert _CONTENT_GATED_MARKER in json.loads(row[0])
        assert row[1] == 11  # latency metering intact
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_log_error_gates_metadata_and_message_but_keeps_error_type(tmp_path):
    backend, store = await _store(tmp_path, "logerror.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("ephemeral"))
        event_id = await store.log_error(
            agent_name="did:test:emma",
            error_type="ValueError",
            error_message="failed on alice@example.com",
            session_id="sess-1",
            metadata={"pii": "alice@example.com"},
        )
        row = await backend.fetch_one(
            "SELECT metadata, error_message FROM a2a_observability WHERE id=?",
            (event_id,),
        )
        meta = json.loads(row[0])
        assert "alice@example.com" not in row[0]
        assert _CONTENT_GATED_MARKER in meta
        # operational key survives the gate
        assert meta["error_type"] == "ValueError"
        # free-form message is content-bearing and elided
        assert row[1] == _CONTENT_GATED_MARKER
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_log_metric_gates_metadata_but_keeps_name_and_value(tmp_path):
    backend, store = await _store(tmp_path, "logmetric.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("ephemeral"))
        event_id = await store.log_metric(
            agent_name="did:test:emma",
            metric_name="turn_latency",
            metric_value=1.5,
            session_id="sess-1",
            metadata={"pii": "alice@example.com"},
        )
        row = await backend.fetch_one(
            "SELECT metadata FROM a2a_observability WHERE id=?",
            (event_id,),
        )
        meta = json.loads(row[0])
        assert "alice@example.com" not in row[0]
        assert _CONTENT_GATED_MARKER in meta
        # operational metering survives so get_metric_summary still works
        assert meta["metric_name"] == "turn_latency"
        assert meta["metric_value"] == 1.5
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_log_error_anonymizes_metadata_in_anonymous(tmp_path):
    backend, store = await _store(tmp_path, "logerror-anon.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("anonymous"))
        event_id = await store.log_error(
            agent_name="did:test:emma",
            error_type="ValueError",
            error_message="failed on alice@example.com",
            metadata={"note": "email alice@example.com"},
        )
        row = await backend.fetch_one(
            "SELECT metadata, error_message FROM a2a_observability WHERE id=?",
            (event_id,),
        )
        assert "alice@example.com" not in row[0]
        assert "alice@example.com" not in (row[1] or "")
        # structure + operational key survive anonymization
        meta = json.loads(row[0])
        assert "note" in meta
        assert meta["error_type"] == "ValueError"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_normal_persists_observability_metadata(tmp_path):
    backend, store = await _store(tmp_path, "normal-meta.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("normal"))
        event_id = await store.log_metric(
            agent_name="did:test:emma",
            metric_name="turn_latency",
            metric_value=1.5,
            metadata={"note": "email alice@example.com"},
        )
        row = await backend.fetch_one(
            "SELECT metadata FROM a2a_observability WHERE id=?",
            (event_id,),
        )
        meta = json.loads(row[0])
        assert meta["note"] == "email alice@example.com"
        assert meta["metric_name"] == "turn_latency"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_purge_observability_since_sweeps_rows(tmp_path):
    backend, store = await _store(tmp_path, "purge.db")
    try:
        # No gating here — we want real rows to exist, then sweep them.
        await store.log_tool_dispatch(_pii_entry())
        await store.log_tool_call(
            agent_name="did:test:emma", tool_name="github", session_id="sess-1"
        )

        before_disp = await _dispatch_row(backend)
        assert before_disp is not None

        # Watermark in the wrapper's space-separated shape; the sweep must
        # normalize it against the store's ISO timestamps and still match.
        counts = await store.purge_observability_since(
            "2000-01-01 00:00:00", agent_did="did:test:emma", agent_name="did:test:emma"
        )
        assert counts["a2a_tool_dispatches"] >= 1
        assert counts["a2a_observability"] >= 1

        assert await _dispatch_row(backend) is None
        remaining = await backend.fetch_one(
            "SELECT COUNT(*) FROM a2a_observability WHERE agent_name=?",
            ("did:test:emma",),
        )
        assert remaining[0] == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_purge_ephemeral_session_drives_observability_sweep(tmp_path):
    """The storage wrapper's ephemeral purge invokes the bound sweep (F076)."""
    backend, store = await _store(tmp_path, "wrapper-purge.db")
    try:
        class _FakeStorage:
            agent_id = "did:test:emma"

            async def purge_conversations_since(self, since, *, reason):
                return 0

            async def purge_agent_graph_nodes(self, *, since_iso):
                return 0

        # Enter EPHEMERAL first (records the watermark), THEN leak a dispatch
        # row that bypassed the gate — the sweep only scrubs rows since the
        # watermark, so ordering matters.
        wrapper = PrivacyEnforcingStorage(_FakeStorage(), "ephemeral")
        wrapper.set_observability_purge(
            lambda since: store.purge_observability_since(
                since, agent_did="did:test:emma", agent_name="did:test:emma"
            )
        )
        await store.log_tool_dispatch(_pii_entry())

        breakdown = await wrapper.purge_ephemeral_session()
        assert breakdown.get("a2a_tool_dispatches", 0) >= 1
        assert await _dispatch_row(backend) is None
    finally:
        await store.close()
