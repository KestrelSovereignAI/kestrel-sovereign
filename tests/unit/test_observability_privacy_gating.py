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
- The gates fail closed (#3920): a privacy provider or anonymizer that raises,
  or content nested past the anonymizer's depth cap, is elided, never stored.
"""

import json
import logging
import threading

import pytest

from kestrel_sovereign.a2a.stores.unified import observability_store as obs_module
from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityPurgeError,
    ObservabilityStore,
    ToolDispatchEntry,
    _CONTENT_GATED_MARKER,
    _anonymize_deep,
    redact_tool_args_json,
)
from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.storage.privacy_wrapper import (
    PrivacyEnforcingStorage,
    PurgeOutcome,
)


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
async def test_purge_observability_since_purges_the_whole_transition_second(tmp_path):
    """#3227: the wrapper's watermark is exact to the microsecond, but a row's
    ``timestamp`` can be an event time supplied after the fact, earlier than
    the write. A row stamped 300 ms before the watermark, in its second, is
    purged as it was while the watermark carried whole seconds; the last
    microsecond of the previous second survives."""
    from datetime import datetime, timedelta, timezone

    transition = datetime(2026, 9, 7, 12, 0, 5, 500000, tzinfo=timezone.utc)
    backend, store = await _store(tmp_path, "purge-second.db")
    try:
        for name, stamp in (
            ("previous-second", transition.replace(microsecond=0) - timedelta(microseconds=1)),
            ("before-watermark", transition - timedelta(milliseconds=300)),
            ("after-watermark", transition + timedelta(milliseconds=300)),
        ):
            await store.log_tool_call(
                agent_name="did:test:emma", tool_name=name, timestamp=stamp
            )

        counts = await store.purge_observability_since(
            "2026-09-07 12:00:05.500000",
            agent_did="did:test:emma",
            agent_name="did:test:emma",
        )

        assert counts["a2a_observability"] == 2
        rows = await backend.fetch_all(
            "SELECT tool_name FROM a2a_observability WHERE agent_name=?",
            ("did:test:emma",),
        )
        assert [row[0] for row in rows] == ["previous-second"]
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

            async def purge_channel_messages_since(self, since, *, reason):
                # The wrapper now sweeps channel_messages too (#2096/#2673); a
                # fake that omits it would raise AttributeError and the wrapper
                # would (correctly) record channel_messages as a required FAILED
                # store, masking this clean-path assertion.
                return 0

            async def purge_session_projection(self, *, reason):
                # ...and the #2959 projection, for the same reason. It is a
                # REQUIRED_TRACE_STORE — it holds no content, but its residue
                # names the agent — so omitting it here fails the mode exit
                # rather than the assertion this case is about.
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
        # This is a clean path: every required content sweep succeeded, so mode
        # exit is permitted (no FAILED required store).
        assert breakdown.required_sweep_failed is False
        assert breakdown.get("a2a_tool_dispatches", 0) >= 1
        assert await _dispatch_row(backend) is None
    finally:
        await store.close()


def _flaky_execute_factory(backend, fail_sql_fragment):
    """Return an ``execute`` shim that raises for one DELETE but delegates the
    rest, so a single table sweep fails while the others still run."""
    real_execute = backend.execute

    async def _flaky_execute(sql, params=()):
        if fail_sql_fragment in sql:
            raise RuntimeError("disk I/O error")
        return await real_execute(sql, params)

    return _flaky_execute


@pytest.mark.asyncio
async def test_purge_observability_since_raises_on_partial_failure(tmp_path):
    """#2673: a failed table sweep is surfaced as ``ObservabilityPurgeError``
    carrying per-table successes/failures — never collapsed into a clean 0. The
    OTHER independent sweep is still attempted."""
    backend, store = await _store(tmp_path, "obs-purge-fail.db")
    try:
        await store.log_tool_dispatch(_pii_entry())
        await store.log_tool_call(
            agent_name="did:test:emma", tool_name="github", session_id="sess-1"
        )

        backend.execute = _flaky_execute_factory(
            backend, "DELETE FROM a2a_tool_dispatches"
        )

        with pytest.raises(ObservabilityPurgeError) as excinfo:
            await store.purge_observability_since(
                "2000-01-01 00:00:00",
                agent_did="did:test:emma",
                agent_name="did:test:emma",
            )

        err = excinfo.value
        # The failed table is retained as a failure...
        assert "a2a_tool_dispatches" in err.failures
        # ...and the independent observability sweep still ran and succeeded.
        assert err.partial_counts.get("a2a_observability", 0) >= 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_purge_ephemeral_session_records_observability_failure(tmp_path):
    """#2673: an observability sweep failure is retained as FAILED (never a
    clean 0) but does NOT block mode exit — the sink is a non-required content
    store, so ``required_sweep_failed`` stays False while ``any_sweep_failed``
    trips."""
    backend, store = await _store(tmp_path, "obs-purge-wrapper-fail.db")
    try:
        class _FakeStorage:
            agent_id = "did:test:emma"

            async def purge_conversations_since(self, since, *, reason):
                return 0

            async def purge_agent_graph_nodes(self, *, since_iso):
                return 0

            async def purge_channel_messages_since(self, since, *, reason):
                return 0

            async def purge_session_projection(self, *, reason):
                # A REQUIRED_TRACE_STORE (#2959): omitting it would fail the
                # mode exit for a reason this case is not about.
                return 0

        backend.execute = _flaky_execute_factory(
            backend, "DELETE FROM a2a_tool_dispatches"
        )

        wrapper = PrivacyEnforcingStorage(_FakeStorage(), "ephemeral")
        wrapper.set_observability_purge(
            lambda since: store.purge_observability_since(
                since, agent_did="did:test:emma", agent_name="did:test:emma"
            )
        )
        await store.log_tool_dispatch(_pii_entry())

        report = await wrapper.purge_ephemeral_session()

        # The failed table reads as FAILED, not a clean 0.
        assert (
            report.store_results["a2a_tool_dispatches"].outcome is PurgeOutcome.FAILED
        )
        # Non-required content store: mode exit is still permitted...
        assert report.required_sweep_failed is False
        # ...but the purge is NOT clean.
        assert report.any_sweep_failed is True
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# log_llm_call gating (#2236 codex P1): wiring the store into LLMService made
# prompt/response previews persist for EVERY call, so the previews (plus
# error text, tool_calls, metadata) must honour privacy mode like every
# other content-bearing sink field. Counts/latency/model always persist.
# ---------------------------------------------------------------------------

async def _llm_row(backend):
    return await backend.fetch_one(
        "SELECT system_prompt_preview, user_prompt_preview, response_preview, "
        "error_message, tool_calls, metadata, input_tokens, duration_ms, model "
        "FROM a2a_llm_calls WHERE agent_did=?",
        ("did:test:emma",),
    )


async def _log_pii_llm_call(store):
    await store.log_llm_call(
        provider="openai",
        model="gpt-test",
        duration_ms=1200,
        success=True,
        agent_did="did:test:emma",
        system_prompt="You are Emma. The user is alice@example.com",
        user_prompt="my email is alice@example.com",
        response="Sure thing, alice@example.com!",
        error_message="upstream said: alice@example.com",
        tool_calls=[{"name": "mail", "args": {"to": "alice@example.com"}}],
        metadata={"note": "email me at alice@example.com"},
        input_tokens=10,
        output_tokens=5,
    )


@pytest.mark.asyncio
async def test_ephemeral_elides_llm_call_content_but_keeps_metrics(tmp_path):
    backend, store = await _store(tmp_path, "llm-ephemeral.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("ephemeral"))
        await _log_pii_llm_call(store)

        row = await _llm_row(backend)
        assert row is not None, "row itself still persists (content-free metrics)"
        for content_col in row[:6]:
            assert content_col is None or "alice@example.com" not in str(content_col)
        # counts / latency / model still metered
        assert row[6] == 10
        assert row[7] == 1200
        assert row[8] == "gpt-test"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_anonymous_redacts_llm_call_content(tmp_path):
    backend, store = await _store(tmp_path, "llm-anon.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("anonymous"))
        await _log_pii_llm_call(store)

        row = await _llm_row(backend)
        assert row is not None
        for content_col in row[:6]:
            assert content_col is None or "alice@example.com" not in str(content_col)
        # anonymized, not elided: prompt text survives minus the PII
        assert row[1] and "my email is" in row[1]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_normal_persists_llm_call_content(tmp_path):
    backend, store = await _store(tmp_path, "llm-normal.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("normal"))
        await _log_pii_llm_call(store)

        row = await _llm_row(backend)
        assert row is not None
        assert row[0] and "alice@example.com" in row[0]
        assert row[2] and "alice@example.com" in row[2]
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Fail-closed gates (#3920). The gates used to fail OPEN in two inner
# helpers: a privacy provider that raised read as "no policy configured", and
# ``_anonymize_deep`` returned the original string when the anonymizer raised
# and the untouched subtree below depth 8. Every content field must instead
# be elided while the row's metering still records.
#
# The scenario helpers below are shared with the PostgreSQL leg in
# tests/integration/test_observability_privacy_gating_backends.py.
# ---------------------------------------------------------------------------

# An email, so the real anonymizer removes it wherever it is applied: only a
# gate that skips anonymization, or stores the raw value, lets it persist.
LEAK = "leak-3920@example.com"
FAIL_CLOSED_AGENT = "did:test:fail-closed"


def nested(levels: int, leaf):
    """``leaf`` wrapped in ``levels`` dicts."""
    value = leaf
    for _ in range(levels):
        value = {"level": value}
    return value


# The leaf sits 10 levels below the metadata root: past the depth cap.
DEEP_METADATA = {"note": f"note {LEAK}", "deep": nested(9, LEAK)}
DISPATCH_ARGS = {"to": LEAK, "body": f"body {LEAK}"}
TOOL_CALLS = [{"name": "mail", "args": {"to": LEAK}}]


def _raising_provider():
    # The message carries content: the gate must log the exception type only.
    raise RuntimeError(f"privacy agent unavailable for {LEAK}")


def _raising_anonymizer(text):
    raise ValueError(f"cannot scrub {text}")


def provider_raises(store, monkeypatch):
    store.set_privacy_config_provider(_raising_provider)


def anonymizer_raises(store, monkeypatch):
    store.set_privacy_config_provider(lambda: get_privacy_preset("anonymous"))
    monkeypatch.setattr(obs_module, "_anonymize_text", _raising_anonymizer)


def nested_past_depth_cap(store, monkeypatch):
    # The real anonymizer: it works, but the deep leaf is never reached.
    store.set_privacy_config_provider(lambda: get_privacy_preset("anonymous"))


FAIL_CLOSED_SCENARIOS = {
    "provider-raises": provider_raises,
    "anonymizer-raises": anonymizer_raises,
    "nested-past-depth-cap": nested_past_depth_cap,
}


async def write_every_content_field(store, agent: str = FAIL_CLOSED_AGENT) -> None:
    """Put ``LEAK`` in every content field each store writer persists."""
    await store.log_llm_call(
        provider="openai",
        model="gpt-test",
        duration_ms=1200,
        success=False,
        agent_did=agent,
        system_prompt=f"system {LEAK}",
        user_prompt=f"user {LEAK}",
        response=f"response {LEAK}",
        error_message=f"error {LEAK}",
        tool_calls=TOOL_CALLS,
        metadata=DEEP_METADATA,
        input_tokens=10,
        output_tokens=5,
    )
    event_id = await store.log_tool_call(
        agent_name=agent, tool_name="mail", session_id="sess-1",
        metadata=DEEP_METADATA,
    )
    await store.log_tool_response(
        event_id, success=False, duration_ms=33, error_message=f"tool {LEAK}"
    )
    await store.log_tool_dispatch(
        ToolDispatchEntry(
            agent_did=agent,
            session_id="sess-1",
            turn_id="turn_1",
            tool_name="mail",
            adapter="cli.mail",
            args_redacted=DISPATCH_ARGS,
            result_status="error",
            error_class="ValueError",
            error_message=f"dispatch {LEAK}",
            latency_ms=42,
            result_size_bytes=17,
        ),
        required=True,
    )
    await store.log_agent_response(
        agent_name=agent, duration_ms=11, session_id="sess-1",
        metadata=DEEP_METADATA,
    )
    await store.log_error(
        agent_name=agent, error_type="ValueError", error_message=f"failed {LEAK}",
        metadata=DEEP_METADATA,
    )
    await store.log_metric(
        agent_name=agent, metric_name="turn_latency", metric_value=1.5,
        metadata=DEEP_METADATA,
    )


def as_json(value):
    """A JSON column as Python, whichever form the backend returns it in."""
    return value if isinstance(value, (dict, list)) else json.loads(value)


async def persisted_rows(backend, agent: str = FAIL_CLOSED_AGENT) -> list:
    """Every column of every row the writers above produced."""
    rows = []
    for table, column in (
        ("a2a_llm_calls", "agent_did"),
        ("a2a_observability", "agent_name"),
        ("a2a_tool_dispatches", "agent_did"),
    ):
        rows += await backend.fetch_all(
            f"SELECT * FROM {table} WHERE {column}=?", (agent,)
        )
    return rows


async def assert_nothing_leaked(backend, agent: str = FAIL_CLOSED_AGENT) -> None:
    rows = await persisted_rows(backend, agent)
    assert len(rows) == 6
    leaked = [str(value) for row in rows for value in row if LEAK in str(value)]
    assert leaked == []


async def assert_metering_recorded(store, backend, agent: str = FAIL_CLOSED_AGENT):
    """Counts, status and latency persist whatever happened to the content."""
    llm = await backend.fetch_all(
        "SELECT duration_ms, input_tokens, output_tokens, success, model "
        "FROM a2a_llm_calls WHERE agent_did=?",
        (agent,),
    )
    assert [tuple(row) for row in llm] == [
        (1200, 10, 5, store.to_bool_param(False), "gpt-test")
    ]
    events = await backend.fetch_all(
        "SELECT event_type, duration_ms, metadata FROM a2a_observability "
        "WHERE agent_name=?",
        (agent,),
    )
    assert sorted((row[0], row[1]) for row in events) == [
        ("agent_response", 11),
        ("error", None),
        ("metric", None),
        ("tool_response", 33),
    ]
    metas = {row[0]: as_json(row[2]) for row in events}
    assert metas["error"]["error_type"] == "ValueError"
    assert metas["metric"]["metric_name"] == "turn_latency"
    assert metas["metric"]["metric_value"] == 1.5
    dispatch = await backend.fetch_all(
        "SELECT result_status, error_class, latency_ms, result_size_bytes "
        "FROM a2a_tool_dispatches WHERE agent_did=?",
        (agent,),
    )
    assert [tuple(row) for row in dispatch] == [("error", "ValueError", 42, 17)]
    assert (await store.tool_failure_rate(agent))["total_calls"] == 1


async def assert_content_kept_as_today(backend, agent: str = FAIL_CLOSED_AGENT):
    """Unconfigured and NORMAL/PUBLIC: content persists exactly as before."""
    llm = await backend.fetch_one(
        "SELECT system_prompt_preview, user_prompt_preview, response_preview, "
        "error_message, tool_calls, metadata FROM a2a_llm_calls WHERE agent_did=?",
        (agent,),
    )
    assert tuple(llm[:4]) == (
        f"system {LEAK}", f"user {LEAK}", f"response {LEAK}", f"error {LEAK}"
    )
    assert as_json(llm[4]) == json.loads(redact_tool_args_json(TOOL_CALLS))
    assert as_json(llm[5]) == DEEP_METADATA
    events = await backend.fetch_all(
        "SELECT event_type, error_message, metadata FROM a2a_observability "
        "WHERE agent_name=?",
        (agent,),
    )
    by_type = {row[0]: (row[1], as_json(row[2])) for row in events}
    assert by_type["tool_response"] == (f"tool {LEAK}", DEEP_METADATA)
    assert by_type["agent_response"] == (None, DEEP_METADATA)
    assert by_type["error"] == (
        f"failed {LEAK}", {**DEEP_METADATA, "error_type": "ValueError"}
    )
    dispatch = await backend.fetch_one(
        "SELECT args_redacted, error_message FROM a2a_tool_dispatches "
        "WHERE agent_did=?",
        (agent,),
    )
    assert as_json(dispatch[0]) == DISPATCH_ARGS
    assert dispatch[1] == f"dispatch {LEAK}"


@pytest.fixture
def fresh_unreadable_policy_log(monkeypatch):
    """A process that has not yet logged an unreadable policy."""
    monkeypatch.setattr(obs_module, "_policy_unreadable_logged", threading.Event())


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", sorted(FAIL_CLOSED_SCENARIOS))
async def test_failed_gate_persists_no_content(
    tmp_path, monkeypatch, fresh_unreadable_policy_log, scenario
):
    backend, store = await _store(tmp_path, f"{scenario}.db")
    try:
        FAIL_CLOSED_SCENARIOS[scenario](store, monkeypatch)
        await write_every_content_field(store)

        await assert_nothing_leaked(backend)
        await assert_metering_recorded(store, backend)
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["provider-raises", "anonymizer-raises"])
async def test_failed_gate_elides_every_field_to_the_error_marker(
    tmp_path, monkeypatch, fresh_unreadable_policy_log, scenario
):
    backend, store = await _store(tmp_path, f"{scenario}-marker.db")
    try:
        FAIL_CLOSED_SCENARIOS[scenario](store, monkeypatch)
        await write_every_content_field(store)

        gated = {_CONTENT_GATED_MARKER: "error"}
        llm = await backend.fetch_one(
            "SELECT system_prompt_preview, user_prompt_preview, response_preview, "
            "error_message, tool_calls, metadata FROM a2a_llm_calls WHERE agent_did=?",
            (FAIL_CLOSED_AGENT,),
        )
        assert tuple(llm[:4]) == (_CONTENT_GATED_MARKER,) * 4
        assert as_json(llm[4]) == gated
        assert as_json(llm[5]) == gated
        events = await backend.fetch_all(
            "SELECT event_type, error_message, metadata FROM a2a_observability "
            "WHERE agent_name=?",
            (FAIL_CLOSED_AGENT,),
        )
        by_type = {row[0]: (row[1], as_json(row[2])) for row in events}
        assert by_type["tool_response"] == (_CONTENT_GATED_MARKER, gated)
        assert by_type["error"] == (
            _CONTENT_GATED_MARKER, {**gated, "error_type": "ValueError"}
        )
        dispatch = await backend.fetch_one(
            "SELECT args_redacted, error_message FROM a2a_tool_dispatches "
            "WHERE agent_did=?",
            (FAIL_CLOSED_AGENT,),
        )
        assert as_json(dispatch[0]) == gated
        assert dispatch[1] == _CONTENT_GATED_MARKER
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "normal", "public"])
async def test_unconfigured_and_full_storage_keep_content_as_before(tmp_path, mode):
    backend, store = await _store(tmp_path, f"kept-{mode}.db")
    try:
        if mode is not None:
            store.set_privacy_config_provider(lambda: get_privacy_preset(mode))
        await write_every_content_field(store)

        await assert_content_kept_as_today(backend)
        await assert_metering_recorded(store, backend)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_unreadable_policy_warns_once_naming_only_the_exception_type(
    tmp_path, caplog, fresh_unreadable_policy_log
):
    backend, store = await _store(tmp_path, "unreadable-log.db")
    try:
        store.set_privacy_config_provider(_raising_provider)
        with caplog.at_level(logging.DEBUG, logger=obs_module.__name__):
            await write_every_content_field(store)
            await write_every_content_field(store, agent="did:test:second")

        warnings = [
            record for record in caplog.records
            if record.name == obs_module.__name__ and record.levelno >= logging.WARNING
        ]
        assert len(warnings) == 1
        assert "RuntimeError" in warnings[0].getMessage()
        assert all(LEAK not in record.getMessage() for record in caplog.records)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_unreadable_policy_is_not_latched(tmp_path, fresh_unreadable_policy_log):
    """Once the provider reads again, its mode decides; nothing stays elided."""
    backend, store = await _store(tmp_path, "unreadable-recovers.db")
    try:
        store.set_privacy_config_provider(_raising_provider)
        await write_every_content_field(store, agent="did:test:while-unreadable")
        store.set_privacy_config_provider(lambda: get_privacy_preset("normal"))
        await write_every_content_field(store)

        await assert_content_kept_as_today(backend)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_provider_returning_no_config_elides(
    tmp_path, fresh_unreadable_policy_log
):
    """Only an unbound provider means "unconfigured"; a bound one that returns
    nothing is a policy that cannot be read."""
    backend, store = await _store(tmp_path, "provider-none.db")
    try:
        store.set_privacy_config_provider(lambda: None)
        await write_every_content_field(store)

        await assert_nothing_leaked(backend)
        await assert_metering_recorded(store, backend)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_deidentified_elides_content_like_the_storage_wrapper(tmp_path):
    """DEIDENTIFIED persists no raw content anywhere (the storage wrapper
    refuses every generic write); the observability sink used to keep it."""
    backend, store = await _store(tmp_path, "deidentified.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("deidentified"))
        await write_every_content_field(store)

        await assert_nothing_leaked(backend)
        await assert_metering_recorded(store, backend)
        dispatch = await backend.fetch_one(
            "SELECT args_redacted FROM a2a_tool_dispatches WHERE agent_did=?",
            (FAIL_CLOSED_AGENT,),
        )
        assert as_json(dispatch[0]) == {_CONTENT_GATED_MARKER: "deidentified"}
    finally:
        await store.close()


class _Opaque:
    """A non-JSON value: ``json_dumps(default=str)`` would store its ``str()``."""

    def __str__(self):
        return f"opaque {LEAK}"


@pytest.mark.asyncio
async def test_anonymous_anonymizes_non_json_leaves(tmp_path):
    backend, store = await _store(tmp_path, "anon-leaves.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset("anonymous"))
        event_id = await store.log_agent_response(
            agent_name=FAIL_CLOSED_AGENT,
            duration_ms=11,
            metadata={
                "tags": {LEAK},
                "pair": (LEAK, 1),
                "object": _Opaque(),
                "blob": LEAK.encode(),
                "count": 3,
                "ratio": 0.5,
                "flag": True,
                "missing": None,
            },
        )
        row = await backend.fetch_one(
            "SELECT metadata FROM a2a_observability WHERE id=?", (event_id,)
        )
        assert LEAK not in str(row[0])
        meta = as_json(row[0])
        assert len(meta["tags"]) == 1 and isinstance(meta["tags"], list)
        assert meta["pair"][1] == 1
        assert meta["object"].startswith("opaque ")
        assert meta["blob"] == f"<bytes:{len(LEAK.encode())}>"
        assert (meta["count"], meta["ratio"], meta["flag"], meta["missing"]) == (
            3, 0.5, True, None
        )
    finally:
        await store.close()


def test_anonymize_deep_lets_an_anonymizer_failure_propagate():
    with pytest.raises(ValueError):
        _anonymize_deep({"t": LEAK}, _raising_anonymizer)


def test_anonymize_deep_elides_past_the_depth_cap():
    def scrub(text):
        return text.replace(LEAK, "[EMAIL]")

    # The root is depth 0. A leaf at depth 8 is still anonymized, structure
    # intact...
    assert _anonymize_deep(nested(8, LEAK), scrub) == nested(8, "[EMAIL]")
    # ...anything at depth 9 is elided, whatever its type, subtrees included.
    assert _anonymize_deep(nested(9, LEAK), scrub) == nested(9, _CONTENT_GATED_MARKER)
    assert _anonymize_deep(nested(9, 7), scrub) == nested(9, _CONTENT_GATED_MARKER)
    assert _anonymize_deep(nested(10, LEAK), scrub) == nested(9, _CONTENT_GATED_MARKER)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "normal"])
async def test_operational_keys_do_not_mutate_the_callers_metadata(tmp_path, mode):
    backend, store = await _store(tmp_path, f"no-mutation-{mode}.db")
    try:
        if mode is not None:
            store.set_privacy_config_provider(lambda: get_privacy_preset(mode))
        metadata = {"note": "caller-owned"}
        await store.log_error(
            agent_name=FAIL_CLOSED_AGENT, error_type="ValueError",
            error_message="boom", metadata=metadata,
        )
        await store.log_metric(
            agent_name=FAIL_CLOSED_AGENT, metric_name="turn_latency",
            metric_value=1.5, metadata=metadata,
        )
        assert metadata == {"note": "caller-owned"}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_unrecognized_storage_tier_elides(tmp_path):
    """Content is kept only for a tier the gate knows allows it."""
    from kestrel_sovereign.privacy import PrivacyConfig

    backend, store = await _store(tmp_path, "unknown-tier.db")
    try:
        store.set_privacy_config_provider(lambda: PrivacyConfig(storage="unrecognized"))
        await write_every_content_field(store)

        await assert_nothing_leaked(backend)
        await assert_metering_recorded(store, backend)
        dispatch = await backend.fetch_one(
            "SELECT args_redacted FROM a2a_tool_dispatches WHERE agent_did=?",
            (FAIL_CLOSED_AGENT,),
        )
        assert as_json(dispatch[0]) == {_CONTENT_GATED_MARKER: "unrecognized"}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_anonymizer_failure_is_logged_without_its_message(
    tmp_path, monkeypatch, caplog
):
    """The anonymizer's exception text echoes the content it was scrubbing."""
    backend, store = await _store(tmp_path, "anon-failure-log.db")
    try:
        anonymizer_raises(store, monkeypatch)
        with caplog.at_level(logging.DEBUG, logger=obs_module.__name__):
            await write_every_content_field(store)

        gate_logs = [
            record.getMessage() for record in caplog.records
            if record.name == obs_module.__name__
            and "privacy gate failed" in record.getMessage()
        ]
        assert gate_logs and all("ValueError" in message for message in gate_logs)
        assert all(LEAK not in record.getMessage() for record in caplog.records)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# An elided row stays readable (#3920 review). The gate stores
# ``{_CONTENT_GATED_MARKER: <reason>}`` in ``a2a_llm_calls.tool_calls``, where
# ``LLMCallEvent`` holds a list; the reader used to hand that dict to the model
# and ``query_llm_calls`` raised ValidationError on every page that held the
# row, long after the privacy policy read again.
# ---------------------------------------------------------------------------

def _policy_recovers(store, monkeypatch):
    """The provider reads again and the real anonymizer is back."""
    store.set_privacy_config_provider(lambda: get_privacy_preset("normal"))
    monkeypatch.setattr(obs_module, "_anonymize_text", None)


async def read_every_row(store, agent: str = FAIL_CLOSED_AGENT) -> dict:
    """Every row the writers produced, through the store's own readers."""
    events = await store.query_events(agent_name=agent)
    return {
        "llm_calls": await store.query_llm_calls(agent_did=agent),
        "events": events,
        "event": await store.get_event(events[0].event_id),
        "failures": await store.recent_failures(agent),
    }


async def assert_elided_rows_read_back(store, monkeypatch, scenario) -> None:
    """Rows a failed gate elided read back through every reader, and keep
    reading once the policy recovers and new rows carry content again."""
    FAIL_CLOSED_SCENARIOS[scenario](store, monkeypatch)
    await write_every_content_field(store)
    _policy_recovers(store, monkeypatch)
    await write_every_content_field(store, agent="did:test:recovered")

    elided = await read_every_row(store)
    [call] = elided["llm_calls"]
    gated = {_CONTENT_GATED_MARKER: "error"}
    assert call.tool_calls == [gated]
    assert call.metadata == gated
    assert (
        call.system_prompt_preview, call.user_prompt_preview,
        call.response_preview, call.error_message,
    ) == (_CONTENT_GATED_MARKER,) * 4
    assert (call.duration_ms, call.input_tokens, call.output_tokens) == (1200, 10, 5)
    assert len(elided["events"]) == 4
    assert elided["failures"][0]["args_redacted"] == gated
    assert LEAK not in repr(elided)

    # The row written after recovery keeps its calls, exactly as today.
    [kept] = (await read_every_row(store, agent="did:test:recovered"))["llm_calls"]
    assert kept.tool_calls == json.loads(redact_tool_args_json(TOOL_CALLS))


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["provider-raises", "anonymizer-raises"])
async def test_elided_rows_read_back_after_the_policy_recovers(
    tmp_path, monkeypatch, fresh_unreadable_policy_log, scenario
):
    backend, store = await _store(tmp_path, f"{scenario}-read.db")
    try:
        await assert_elided_rows_read_back(store, monkeypatch, scenario)
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ephemeral", "isolated"])
async def test_mode_elided_tool_calls_read_back(tmp_path, mode):
    """The same marker has been stored in these modes since F076."""
    backend, store = await _store(tmp_path, f"{mode}-read.db")
    try:
        store.set_privacy_config_provider(lambda: get_privacy_preset(mode))
        await write_every_content_field(store)

        [call] = (await read_every_row(store))["llm_calls"]
        assert call.tool_calls == [{_CONTENT_GATED_MARKER: get_privacy_preset(mode).storage}]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_truncated_tool_calls_read_back(tmp_path):
    """Calls past the size cap are stored as one truncation marker."""
    backend, store = await _store(tmp_path, "truncated-read.db")
    try:
        # Strings are capped at 512 chars, so it takes several calls.
        calls = [{"name": "mail", "args": {"body": "x" * 500}}] * 10
        await store.log_llm_call(
            provider="openai", model="gpt-test", duration_ms=5,
            agent_did=FAIL_CLOSED_AGENT, tool_calls=calls,
        )

        [call] = await store.query_llm_calls(agent_did=FAIL_CLOSED_AGENT)
        [marker] = call.tool_calls
        assert marker["_truncated"] is True
        assert marker["preview"].startswith('[{"name": "mail"')
    finally:
        await store.close()
