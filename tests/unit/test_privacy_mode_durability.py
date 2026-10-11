"""The privacy mode is durable, and the mandate it displaced comes back (#3554).

The bug: ``!privacy deidentified`` persisted the forced local model as the
agent's model preference, but the mode itself lived only in the process. After
a restart the agent ran in NORMAL mode on the 1B local model; ``!privacy
normal`` then had nothing to restore.

The restart tests boot a real agent on real SQLite twice against the same
database. Only the routes are scripted: an Anthropic API route and a local
Ollama route, the pair in the issue's repro.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from kestrel_sovereign.agent.privacy_state import (
    FAIL_CLOSED_PRIVACY_MODE,
    PRIVACY_STATE_KEY,
    PRIVACY_STATE_UNRECORDED_MESSAGE,
    PrivacyState,
    PrivacyStateUnreadableError,
    SavedModelPreference,
    load_privacy_state,
    restore_privacy_state,
    write_privacy_state,
)
from kestrel_sovereign.privacy import (
    PRIVACY_MODES_MOST_RESTRICTIVE_FIRST,
    PrivacyMode,
    most_restrictive_privacy_mode,
    privacy_mode_to_config,
)

CLOUD = {"vendor": "anthropic", "route": "api", "model": "claude-haiku-4-5"}
LOCAL = {"vendor": "ollama", "route": "local", "model": "llama3.2:1b"}
UNPINNED = {"vendor": None, "route": None, "model": None}


def _routes() -> list[dict]:
    def route(spec: dict, *, is_local: bool) -> dict:
        return {
            "name": f"{spec['vendor']}:{spec['route']}",
            "vendor": spec["vendor"],
            "route": spec["route"],
            "model": spec["model"],
            "client": None,
            "adapter": None,
            "is_cloud": not is_local,
            "is_local": is_local,
            "base_url": None,
            "reasoning_effort": None,
            "selection_hints": [],
            "capabilities": {},
            "embedding_sibling": None,
        }

    return [route(CLOUD, is_local=False), route(LOCAL, is_local=True)]


@asynccontextmanager
async def _booted_agent(tmp_path, *, before_initialize=None):
    """Boot a real agent whose database lives in ``tmp_path``.

    Booting twice with the same ``tmp_path`` is a restart: same identity, same
    database, a fresh process-level agent and LLM service.
    """
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService

    identity_dir = tmp_path / "identity"
    did_file = tmp_path / "did.txt"
    if did_file.exists():
        did = did_file.read_text()
    else:
        identity_dir.mkdir()
        credentials = await create_kestrel_identity_async(
            output_dir=str(identity_dir), is_test_instance=True, agent_name="Privacy"
        )
        did = credentials.agent_did
        did_file.write_text(did)

    llm_service = LLMService()
    llm_service.providers = _routes()

    async def no_late_routes(*_args, **_kwargs):
        return None

    llm_service.finalize_providers = no_late_routes
    agent = KestrelAgent(
        did=did,
        storage_path=os.path.join(str(identity_dir), "kestrel_prime.db"),
        llm_service=llm_service,
    )
    try:
        if before_initialize is not None:
            await before_initialize(agent)
        await agent.initialize()
        yield agent
    finally:
        await agent.shutdown()
        await llm_service.close()


async def _seed_record(agent, value: str) -> None:
    """Write a raw privacy record into the agent's database before it boots."""
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    storage = AsyncStorage(agent.storage_path, agent_id=agent.did)
    await storage.initialize()
    try:
        await storage.db.execute(
            "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (agent.did, PRIVACY_STATE_KEY, value, datetime.now(timezone.utc).isoformat()),
        )
    finally:
        await storage.close()


async def _seed_model_preference(agent, preference: dict) -> None:
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    storage = AsyncStorage(agent.storage_path, agent_id=agent.did)
    await storage.initialize()
    try:
        await storage.db.execute(
            "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (
                agent.did,
                "model_preference",
                json.dumps(preference),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    finally:
        await storage.close()


async def _recorded(agent) -> PrivacyState | None:
    return await load_privacy_state(agent._raw_storage.db, agent.did)


async def _persisted_preference(agent) -> dict:
    rows = await agent._raw_storage.db.fetchall(
        "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
        (agent.did, "model_preference"),
    )
    return json.loads(rows[0][0])


def _mandate(agent) -> dict:
    return agent.llm_service.get_model_preference()


def _assert_consistent(agent, mode: PrivacyMode) -> None:
    """Every privacy state holder agrees on ``mode``."""
    assert agent.privacy_mode is mode
    assert agent.storage.privacy_mode is mode
    assert agent.privacy_agent.privacy_mode is mode


async def _pin_cloud_mandate(agent) -> None:
    agent.llm_service.set_model_preference(
        CLOUD["model"], CLOUD["vendor"], CLOUD["route"]
    )
    await agent.llm_service.drain_preference_persistence()


# --- The issue's repro and the scope ruling's tests -------------------------


@pytest.mark.asyncio
async def test_issue_repro_mode_and_mandate_agree_across_restarts(tmp_path):
    """#3554 repro: mandate anthropic, ``!privacy deidentified``, restart,
    ``!privacy normal``, restart.

    Before the fix the first restart came back in NORMAL on the local model
    and ``!privacy normal`` restored nothing.
    """
    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)
        reply = await agent.command_handler._cmd_privacy("!privacy deidentified")
        assert "to deidentified" in reply
        assert _mandate(agent) == LOCAL

    # Enter, restart: the mode and its local route both come back.
    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.DEIDENTIFIED)
        assert _mandate(agent) == LOCAL
        assert not agent.privacy_config.allows_cloud_llm()

        # Enter, restart, exit: the original mandate comes back.
        reply = await agent.command_handler._cmd_privacy("!privacy normal")
        assert "from deidentified to normal" in reply
        _assert_consistent(agent, PrivacyMode.NORMAL)
        assert _mandate(agent) == CLOUD
        assert await _persisted_preference(agent) == CLOUD
        assert await _recorded(agent) == PrivacyState(mode=PrivacyMode.NORMAL)

    # NORMAL comes back on the pre-mode mandate.
    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.NORMAL)
        assert _mandate(agent) == CLOUD


@pytest.mark.asyncio
async def test_enter_restart_restores_mode_route_and_saved_mandate(tmp_path):
    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)
        result = await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)
        assert result.applied is True
        assert result.model_switched == LOCAL
        assert await _recorded(agent) == PrivacyState(
            mode=PrivacyMode.DEIDENTIFIED,
            saved_preference=SavedModelPreference(**CLOUD),
        )
        assert await _persisted_preference(agent) == LOCAL

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.DEIDENTIFIED)
        assert _mandate(agent) == LOCAL
        assert agent._privacy_saved_preference == SavedModelPreference(**CLOUD)


@pytest.mark.asyncio
async def test_enter_then_exit_restores_original_mandate(tmp_path):
    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)
        await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)
        assert _mandate(agent) == LOCAL

        result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

        assert result.applied is True
        assert result.model_switched == CLOUD
        assert _mandate(agent) == CLOUD
        assert agent._privacy_saved_preference is None
        assert await _recorded(agent) == PrivacyState(mode=PrivacyMode.NORMAL)
        assert await _persisted_preference(agent) == CLOUD


@pytest.mark.asyncio
async def test_reentering_a_local_only_mode_keeps_the_saved_mandate(tmp_path):
    """A second local-only mode must not save the local model the first forced."""
    saved = SavedModelPreference(**CLOUD)
    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)
        await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)

        for mode in (PrivacyMode.DEIDENTIFIED, PrivacyMode.ISOLATED, PrivacyMode.ANONYMOUS):
            result = await agent.set_privacy_mode_with_effects(mode)
            assert result.applied is True
            assert _mandate(agent) == LOCAL
            assert agent._privacy_saved_preference == saved
            assert await _recorded(agent) == PrivacyState(mode=mode, saved_preference=saved)

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.ANONYMOUS)
        assert agent._privacy_saved_preference == saved
        await agent.set_privacy_mode_with_effects(PrivacyMode.PUBLIC)
        assert _mandate(agent) == CLOUD


@pytest.mark.asyncio
async def test_unpinned_agent_returns_to_default_routing_not_an_invented_pin(tmp_path):
    """No mandate before the mode means none after it: leaving the mode must
    not pin the first configured route, nor keep the forced local model."""
    async with _booted_agent(tmp_path) as agent:
        assert _mandate(agent) == UNPINNED
        await agent.set_privacy_mode_with_effects(PrivacyMode.ISOLATED)
        assert _mandate(agent) == LOCAL

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.ISOLATED)
        result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)
        assert result.model_switched is None
        assert _mandate(agent) == UNPINNED
        assert await _persisted_preference(agent) == UNPINNED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored",
    [
        "not json",
        json.dumps({"version": 1, "mode": "secret", "transition_from": None, "saved_preference": None}),
        json.dumps({"version": 2, "mode": "normal", "transition_from": None, "saved_preference": None}),
        json.dumps({"version": 1, "mode": "normal", "transition_from": "??", "saved_preference": None}),
        json.dumps({"mode": "deidentified"}),
    ],
    ids=["not-json", "unknown-mode", "future-version", "unknown-transition-from", "missing-keys"],
)
async def test_unreadable_recorded_mode_fails_closed(tmp_path, stored, caplog):
    """A record that cannot be read never falls back to NORMAL."""

    async def seed(agent):
        await _seed_model_preference(agent, CLOUD)
        await _seed_record(agent, stored)

    caplog.set_level(logging.ERROR, logger="kestrel_sovereign.agent.privacy_state")
    async with _booted_agent(tmp_path, before_initialize=seed) as agent:
        _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
        assert agent.privacy_mode is not PrivacyMode.NORMAL
        assert not agent.privacy_config.allows_cloud_llm()
        # The fail-closed mode is a local-only mode like any other: it comes
        # back on its local route, holding the mandate it displaced.
        assert _mandate(agent) == LOCAL
        assert agent._privacy_saved_preference == SavedModelPreference(**CLOUD)
        assert any("failing closed" in r.getMessage() for r in caplog.records)
        # The unreadable record is replaced by the mode boot failed closed to.
        assert await _recorded(agent) == PrivacyState(
            mode=FAIL_CLOSED_PRIVACY_MODE,
            saved_preference=SavedModelPreference(**CLOUD),
        )


@pytest.mark.asyncio
async def test_interrupted_exit_restores_the_more_restrictive_mode(tmp_path):
    """A process stopped mid-way through ``deidentified -> normal`` (the
    mandate already restored and persisted, the record not yet committed)
    comes back in DEIDENTIFIED on its local route, still holding the mandate."""
    pending = PrivacyState(
        mode=PrivacyMode.NORMAL,
        saved_preference=SavedModelPreference(**CLOUD),
        transition_from=PrivacyMode.DEIDENTIFIED,
    )

    async def seed(agent):
        await _seed_model_preference(agent, CLOUD)
        await _seed_record(agent, pending.encode())

    async with _booted_agent(tmp_path, before_initialize=seed) as agent:
        _assert_consistent(agent, PrivacyMode.DEIDENTIFIED)
        assert _mandate(agent) == LOCAL
        assert agent._privacy_saved_preference == SavedModelPreference(**CLOUD)

        await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)
        assert _mandate(agent) == CLOUD


@pytest.mark.asyncio
async def test_interrupted_enter_restores_the_local_mode_before_its_route_persisted(tmp_path):
    """Stopped after ``normal -> ephemeral`` was recorded but before the forced
    local route was persisted: EPHEMERAL comes back, on its local route."""
    pending = PrivacyState(
        mode=PrivacyMode.EPHEMERAL,
        saved_preference=SavedModelPreference(**CLOUD),
        transition_from=PrivacyMode.NORMAL,
    )

    async def seed(agent):
        await _seed_model_preference(agent, CLOUD)
        await _seed_record(agent, pending.encode())

    async with _booted_agent(tmp_path, before_initialize=seed) as agent:
        _assert_consistent(agent, PrivacyMode.EPHEMERAL)
        assert _mandate(agent) == LOCAL
        assert await _persisted_preference(agent) == LOCAL


@pytest.mark.asyncio
async def test_agent_without_a_record_boots_in_its_configured_mode(tmp_path):
    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.NORMAL)
        assert await _recorded(agent) is None
        assert _mandate(agent) == UNPINNED


# --- Refusal when the mode cannot be recorded -------------------------------


class _UnwritableDatabase:
    backend_type = "sqlite"

    def __init__(self):
        self.attempts = 0

    async def fetchall(self, *_args, **_kwargs):
        return []

    async def execute(self, *_args, **_kwargs):
        import sqlite3

        self.attempts += 1
        raise sqlite3.OperationalError("disk I/O error")


class _RecordingLLM:
    def __init__(self, preference: dict):
        self.providers = _routes()
        self._pref = dict(preference)
        self.calls: list[tuple] = []

    def get_model_preference(self):
        return dict(self._pref)

    def resolve_model_preference(self, model, vendor=None, route=None, *, validate=True):
        return {"vendor": vendor, "model": model, "route": route}

    def set_model_preference(self, model, vendor=None, route=None, *, validate=True):
        self.calls.append((model, vendor, route))
        self._pref = {"vendor": vendor, "model": model, "route": route}

    def clear_model_preference(self):
        self.calls.append(None)
        self._pref = dict(UNPINNED)


def _bare_agent(*, mode=PrivacyMode.NORMAL, preference=CLOUD, db=None):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from kestrel_sovereign.features.privacy import PrivacyAgent
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    agent = KestrelAgent.__new__(KestrelAgent)
    agent.did = "did:example:privacy"
    agent._privacy_mode = mode
    agent._raw_storage = SimpleNamespace(db=db)
    agent.storage = PrivacyEnforcingStorage(MagicMock(), mode)
    agent.privacy_agent = PrivacyAgent(MagicMock(), mode)
    agent.features = {}
    agent.llm_service = _RecordingLLM(preference)
    return agent


@pytest.mark.asyncio
async def test_transition_that_cannot_be_recorded_is_refused():
    """A local-only mode that would not survive a restart is not entered:
    nothing flips and the model preference is untouched."""
    db = _UnwritableDatabase()
    agent = _bare_agent(db=db)

    result = await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)

    assert db.attempts == 1
    assert result.applied is False
    assert result.persistence_failed is True
    assert result.message == PRIVACY_STATE_UNRECORDED_MESSAGE
    assert result.allows_cloud_llm is True
    _assert_consistent(agent, PrivacyMode.NORMAL)
    assert agent.llm_service.calls == []
    assert getattr(agent, "_privacy_saved_preference", None) is None


@pytest.mark.asyncio
async def test_endpoint_reports_an_unrecorded_transition_as_not_applied():
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from tests.unit.test_privacy_mode_model_restore_contracts import (
        _prepare_app,
        _restore_app,
    )

    agent = _bare_agent(db=_UnwritableDatabase())
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agent/privacy-mode",
                    headers={
                        "X-API-Key": "test-key",
                        "X-Kestrel-Allow-Destructive": "test-rail-bypass",
                    },
                    json={"mode": "deidentified"},
                )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert body["persistence_failed"] is True
        assert body["mode"] == "normal"
    finally:
        _restore_app(app, original)


@pytest.mark.asyncio
async def test_leaving_with_nothing_saved_keeps_the_current_preference_and_warns(caplog):
    """No saved mandate: the current preference stays, and the forced local
    model is not reported as a restored mandate."""
    agent = _bare_agent(mode=PrivacyMode.DEIDENTIFIED, preference=LOCAL)
    caplog.set_level(logging.WARNING)

    result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

    assert result.applied is True
    assert result.model_switched is None
    assert agent.llm_service.calls == []
    assert agent.llm_service.get_model_preference() == LOCAL
    assert any(
        "no saved model mandate" in record.getMessage() for record in caplog.records
    )


# --- The record and its resolution -----------------------------------------


def test_restrictiveness_order_covers_every_mode():
    assert set(PRIVACY_MODES_MOST_RESTRICTIVE_FIRST) == set(PrivacyMode)
    assert len(PRIVACY_MODES_MOST_RESTRICTIVE_FIRST) == len(PrivacyMode)
    assert FAIL_CLOSED_PRIVACY_MODE is PrivacyMode.EPHEMERAL
    # Every local-only mode is more restrictive than every cloud mode, so an
    # interrupted transition between the two always resolves local-only.
    for local in PrivacyMode:
        if privacy_mode_to_config(local).allows_cloud_llm():
            continue
        for cloud in PrivacyMode:
            if privacy_mode_to_config(cloud).allows_cloud_llm():
                assert most_restrictive_privacy_mode(local, cloud) is local
                assert most_restrictive_privacy_mode(cloud, local) is local


@pytest.mark.parametrize(
    "state",
    [
        PrivacyState(mode=PrivacyMode.NORMAL),
        PrivacyState(mode=PrivacyMode.EPHEMERAL, saved_preference=SavedModelPreference(**CLOUD)),
        PrivacyState(mode=PrivacyMode.ISOLATED, saved_preference=SavedModelPreference(None, None, None)),
        PrivacyState(
            mode=PrivacyMode.NORMAL,
            saved_preference=SavedModelPreference("anthropic", "claude-haiku-4-5", None),
            transition_from=PrivacyMode.DEIDENTIFIED,
        ),
    ],
)
def test_record_round_trips(state):
    assert PrivacyState.decode(state.encode()) == state


@pytest.mark.parametrize(
    "saved",
    [
        {"vendor": "anthropic", "route": "api"},
        {"vendor": "anthropic", "route": "api", "model": ""},
        {"vendor": "anthropic", "route": None, "model": None},
        {"vendor": 7, "route": "api", "model": "m"},
        ["anthropic", "api", "m"],
    ],
)
def test_malformed_saved_preference_makes_the_record_unreadable(saved):
    raw = json.dumps(
        {"version": 1, "mode": "isolated", "transition_from": None, "saved_preference": saved}
    )
    with pytest.raises(PrivacyStateUnreadableError):
        PrivacyState.decode(raw)


class _StoredRows:
    backend_type = "sqlite"

    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error
        self.writes: list[tuple] = []

    async def fetchall(self, _sql, _params):
        if self.error is not None:
            raise self.error
        return self.rows

    async def execute(self, _sql, params):
        self.writes.append(params)
        if params[1] == PRIVACY_STATE_KEY:
            self.rows = [(params[2],)]

    def records(self) -> list[PrivacyState]:
        return [
            PrivacyState.decode(params[2])
            for params in self.writes
            if params[1] == PRIVACY_STATE_KEY
        ]

    def mandates(self) -> list[dict]:
        return [
            json.loads(params[2])
            for params in self.writes
            if params[1] == "model_preference"
        ]


@pytest.mark.asyncio
async def test_restore_without_a_record_keeps_the_configured_mode():
    restored = await restore_privacy_state(
        _StoredRows(), "did:example:a", initial_mode=PrivacyMode.PUBLIC
    )
    assert restored.mode is PrivacyMode.PUBLIC
    assert restored.recorded is False
    assert restored.failed_closed is False


@pytest.mark.asyncio
async def test_restore_fails_closed_when_the_read_fails():
    import sqlite3

    restored = await restore_privacy_state(
        _StoredRows(error=sqlite3.OperationalError("database is locked")),
        "did:example:a",
        initial_mode=PrivacyMode.NORMAL,
    )
    assert restored.mode is FAIL_CLOSED_PRIVACY_MODE
    assert restored.failed_closed is True


@pytest.mark.asyncio
async def test_restore_resolves_an_interrupted_transition_to_the_more_restrictive_mode():
    for start, target in [
        (PrivacyMode.DEIDENTIFIED, PrivacyMode.NORMAL),
        (PrivacyMode.NORMAL, PrivacyMode.ISOLATED),
        (PrivacyMode.PUBLIC, PrivacyMode.NORMAL),
    ]:
        stored = PrivacyState(mode=target, transition_from=start).encode()
        restored = await restore_privacy_state(
            _StoredRows(rows=[(stored,)]), "did:example:a", initial_mode=PrivacyMode.NORMAL
        )
        assert restored.mode is most_restrictive_privacy_mode(start, target)
        assert restored.failed_closed is False


@pytest.mark.asyncio
async def test_write_binds_the_agent_and_key():
    db = _StoredRows()
    state = PrivacyState(mode=PrivacyMode.ANONYMOUS)
    await write_privacy_state(db, "did:example:a", state)
    [(agent_id, key, value, _updated_at)] = db.writes
    assert (agent_id, key) == ("did:example:a", PRIVACY_STATE_KEY)
    assert PrivacyState.decode(value) == state


@pytest.mark.asyncio
async def test_transition_refused_by_a_fact_lease_records_the_mode_still_in_force():
    """The transition is recorded before anything flips; when the storage
    wrapper then refuses it, the mandate and the record go back to the state
    still in force."""
    db = _StoredRows()
    agent = _bare_agent(db=db)
    agent.storage._acquire_explicit_fact_lease()
    try:
        result = await agent.set_privacy_mode_with_effects(PrivacyMode.ISOLATED)
    finally:
        agent.storage._release_explicit_fact_lease()

    assert result.retryable_conflict is True
    assert result.applied is False
    _assert_consistent(agent, PrivacyMode.NORMAL)
    assert agent.llm_service.calls == []
    saved = SavedModelPreference(**CLOUD)
    pending = PrivacyState(PrivacyMode.ISOLATED, saved, transition_from=PrivacyMode.NORMAL)
    assert db.records() == [
        pending,
        PrivacyState(PrivacyMode.ISOLATED, saved),
        # Put back: the pending record, the mandate in force, the origin.
        pending,
        PrivacyState(mode=PrivacyMode.NORMAL),
    ]
    assert db.mandates() == [LOCAL, CLOUD]


class _OriginUnwritableRows(_StoredRows):
    """Refuses settled records of ``refused``; everything else is written."""

    def __init__(self, refused):
        super().__init__()
        self.refused = refused

    async def execute(self, sql, params):
        import sqlite3

        if params[1] == PRIVACY_STATE_KEY:
            state = PrivacyState.decode(params[2])
            if state.transition_from is None and state.mode is self.refused:
                raise sqlite3.OperationalError("disk I/O error")
        await super().execute(sql, params)


@pytest.mark.asyncio
async def test_guard_refused_exit_never_leaves_normal_on_the_local_route():
    """The exit is recorded (settled NORMAL, mandate restored), then the
    storage wrapper refuses it, and the record it started from cannot be put
    back. The pending record is written again before the local route is, so
    a restart resumes DEIDENTIFIED with the saved mandate, never NORMAL on
    the local route with the saved mandate gone."""
    db = _OriginUnwritableRows(PrivacyMode.DEIDENTIFIED)
    agent = _bare_agent(mode=PrivacyMode.DEIDENTIFIED, preference=LOCAL, db=db)
    saved = SavedModelPreference(**CLOUD)
    agent._privacy_saved_preference = saved
    # DEIDENTIFIED storage refuses explicit facts; an in-flight session
    # projection read holds the same transition guard.
    agent.storage._active_session_projection_leases += 1
    try:
        result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)
    finally:
        agent.storage._active_session_projection_leases -= 1

    assert result.retryable_conflict is True
    _assert_consistent(agent, PrivacyMode.DEIDENTIFIED)
    pending = PrivacyState(
        PrivacyMode.NORMAL, saved, transition_from=PrivacyMode.DEIDENTIFIED
    )
    assert db.records() == [pending, PrivacyState(PrivacyMode.NORMAL), pending]
    assert db.mandates() == [CLOUD, LOCAL]
    restored = await restore_privacy_state(
        db, agent.did, initial_mode=PrivacyMode.NORMAL
    )
    assert (restored.mode, restored.saved_preference) == (PrivacyMode.DEIDENTIFIED, saved)


@pytest.mark.asyncio
async def test_endpoint_conflict_says_where_a_restart_resumes():
    """An entry the storage wrapper refused, whose record could not be put
    back: the 409 says a restart resumes in the mode the pending record names."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from tests.unit.test_privacy_mode_model_restore_contracts import (
        _prepare_app,
        _restore_app,
    )

    agent = _bare_agent(db=_OriginUnwritableRows(PrivacyMode.NORMAL))
    agent.storage._acquire_explicit_fact_lease()
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agent/privacy-mode",
                    headers={
                        "X-API-Key": "test-key",
                        "X-Kestrel-Allow-Destructive": "test-rail-bypass",
                    },
                    json={"mode": "isolated"},
                )
        assert response.status_code == 409
        assert response.json()["detail"].endswith(
            "Until then, a restart resumes in isolated mode."
        )
    finally:
        agent.storage._release_explicit_fact_lease()
        _restore_app(app, original)


@pytest.mark.asyncio
async def test_nothing_changes_until_the_transition_is_recorded():
    """Persist before mutating: at every write a transition makes, the live
    mode, every state holder, the saved mandate and the model preference are
    still those it started from."""
    seen: list[tuple] = []

    class _Watching(_StoredRows):
        async def execute(self, sql, params):
            seen.append(
                (
                    agent.privacy_mode,
                    agent.storage.privacy_mode,
                    agent.privacy_agent.privacy_mode,
                    getattr(agent, "_privacy_saved_preference", None),
                    agent.llm_service.get_model_preference(),
                )
            )
            await super().execute(sql, params)

    agent = _bare_agent(db=_Watching())
    saved = SavedModelPreference(**CLOUD)

    await agent.set_privacy_mode_with_effects(PrivacyMode.ISOLATED)
    assert seen == [(PrivacyMode.NORMAL,) * 3 + (None, CLOUD)] * 3
    _assert_consistent(agent, PrivacyMode.ISOLATED)
    assert agent.llm_service.get_model_preference() == LOCAL

    seen.clear()
    await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)
    assert seen == [(PrivacyMode.ISOLATED,) * 3 + (saved, LOCAL)] * 3
    _assert_consistent(agent, PrivacyMode.NORMAL)
    assert agent.llm_service.get_model_preference() == CLOUD


@pytest.mark.asyncio
async def test_applied_transition_is_recorded_before_and_after_it_flips():
    db = _StoredRows()
    agent = _bare_agent(db=db)

    await agent.set_privacy_mode_with_effects(PrivacyMode.ISOLATED)
    await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

    saved = SavedModelPreference(**CLOUD)
    assert db.records() == [
        PrivacyState(PrivacyMode.ISOLATED, saved, transition_from=PrivacyMode.NORMAL),
        PrivacyState(PrivacyMode.ISOLATED, saved),
        # Leaving keeps the saved mandate until the exit is recorded as applied.
        PrivacyState(PrivacyMode.NORMAL, saved, transition_from=PrivacyMode.ISOLATED),
        PrivacyState(PrivacyMode.NORMAL),
    ]
    assert agent.llm_service.calls == [
        (LOCAL["model"], LOCAL["vendor"], LOCAL["route"]),
        (CLOUD["model"], CLOUD["vendor"], CLOUD["route"]),
    ]
    assert db.mandates() == [LOCAL, CLOUD]


@pytest.mark.asyncio
async def test_mandate_is_persisted_before_the_transition_is_recorded_as_applied():
    """The applied record is the commit point: a restart that reads it must
    find the mandate it implies already persisted, never the one it replaced."""
    events: list[str] = []

    class _OrderedRows(_StoredRows):
        async def execute(self, sql, params):
            if params[1] == PRIVACY_STATE_KEY:
                state = PrivacyState.decode(params[2])
                events.append("pending" if state.transition_from else "applied")
            else:
                events.append("mandate written")
            await super().execute(sql, params)

    agent = _bare_agent(db=_OrderedRows())

    async def drain_preference_persistence():
        events.append("scheduled writes finished")

    agent.llm_service.drain_preference_persistence = drain_preference_persistence

    await agent.set_privacy_mode_with_effects(PrivacyMode.ISOLATED)
    await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

    assert events == [
        "pending",
        "scheduled writes finished",
        "mandate written",
        "applied",
    ] * 2


@pytest.mark.asyncio
async def test_failed_read_at_boot_never_leaves_a_later_boot_on_the_local_model(
    tmp_path, monkeypatch
):
    """A boot that cannot read the record fails closed and forces the local
    route, which is persisted as the model preference. It must record the mode
    it failed closed to, with the mandate it displaced: otherwise the next boot
    that reads the old NORMAL record comes back NORMAL on the local model,
    which is the bug this issue fixes."""
    import sqlite3

    from kestrel_sovereign.agent import privacy_state

    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)
        await agent.set_privacy_mode_with_effects(PrivacyMode.PUBLIC)

    async def unreadable(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patched:
        patched.setattr(privacy_state, "load_privacy_state", unreadable)
        async with _booted_agent(tmp_path) as agent:
            _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
            assert _mandate(agent) == LOCAL

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
        assert _mandate(agent) == LOCAL
        assert agent._privacy_saved_preference == SavedModelPreference(**CLOUD)
        await agent.set_privacy_mode_with_effects(PrivacyMode.PUBLIC)
        assert _mandate(agent) == CLOUD


# --- Boot records the mode it failed closed to (#3554 review P1) ------------


def _first_read_fails(monkeypatch):
    """Make boot's privacy-record read fail, as a locked database would. The
    read that recovers a saved mandate is a separate one, and succeeds."""
    import sqlite3

    from kestrel_sovereign.agent import privacy_state

    async def unreadable(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(privacy_state, "load_privacy_state", unreadable)


@pytest.mark.asyncio
async def test_fail_closed_boot_is_recorded_when_the_mandate_is_already_local(
    tmp_path, monkeypatch
):
    """An old PUBLIC record, a local pin, and a read that fails once.

    The local pin needs no routing change, so recording the fail-closed mode
    cannot hang off one. Before the fix nothing was written, and the next boot
    read the PUBLIC record and came back allowing cloud routes.

    The first boot reads the record normally. Booting in EPHEMERAL straight
    after inception would let its shutdown purge, whose watermark has
    one-second resolution, take inception's own rows with it."""

    async def seed(agent):
        await _seed_model_preference(agent, LOCAL)
        await _seed_record(agent, PrivacyState(mode=PrivacyMode.PUBLIC).encode())

    async with _booted_agent(tmp_path, before_initialize=seed) as agent:
        _assert_consistent(agent, PrivacyMode.PUBLIC)
        assert _mandate(agent) == LOCAL

    with monkeypatch.context() as patched:
        _first_read_fails(patched)
        async with _booted_agent(tmp_path) as agent:
            _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
            assert _mandate(agent) == LOCAL
            assert await _recorded(agent) == PrivacyState(mode=FAIL_CLOSED_PRIVACY_MODE)

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
        assert not agent.privacy_config.allows_cloud_llm()
        assert _mandate(agent) == LOCAL


@pytest.mark.asyncio
async def test_fail_closed_boot_keeps_the_saved_mandate_of_a_record_it_could_not_read(
    tmp_path, monkeypatch
):
    """The record boot could not read held the mandate DEIDENTIFIED displaced.
    Failing closed must not overwrite it with nothing: leaving the mode later
    still restores the original mandate."""
    saved = SavedModelPreference(**CLOUD)

    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)
        await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)
        assert _mandate(agent) == LOCAL
        assert await _recorded(agent) == PrivacyState(
            mode=PrivacyMode.DEIDENTIFIED, saved_preference=saved
        )

    with monkeypatch.context() as patched:
        _first_read_fails(patched)
        async with _booted_agent(tmp_path) as agent:
            _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
            assert agent._privacy_saved_preference == saved
            assert await _recorded(agent) == PrivacyState(
                mode=FAIL_CLOSED_PRIVACY_MODE, saved_preference=saved
            )

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
        assert agent._privacy_saved_preference == saved
        await agent.set_privacy_mode_with_effects(PrivacyMode.PUBLIC)
        assert _mandate(agent) == CLOUD


@pytest.mark.asyncio
async def test_fail_closed_boot_that_cannot_record_does_not_persist_the_forced_route(
    tmp_path, monkeypatch
):
    """Boot cannot read the record, fails closed, and cannot record that
    either. It still runs on the local route, but must not persist it as the
    mandate: the next boot reads the old PUBLIC record, and would otherwise
    come back PUBLIC on the local model with the cloud mandate gone."""
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    async def seed(agent):
        await _seed_model_preference(agent, CLOUD)
        await _seed_record(agent, PrivacyState(mode=PrivacyMode.PUBLIC).encode())

    async with _booted_agent(tmp_path, before_initialize=seed) as agent:
        _assert_consistent(agent, PrivacyMode.PUBLIC)

    import sqlite3

    execute = AsyncDatabase.execute

    async def record_write_fails(self, sql, params=()):
        if isinstance(params, (tuple, list)) and len(params) > 1 and params[1] == PRIVACY_STATE_KEY:
            raise sqlite3.OperationalError("disk I/O error")
        return await execute(self, sql, params)

    with monkeypatch.context() as patched:
        _first_read_fails(patched)
        patched.setattr(AsyncDatabase, "execute", record_write_fails)
        async with _booted_agent(tmp_path) as agent:
            _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
            assert _mandate(agent) == LOCAL
            await agent.llm_service.drain_preference_persistence()
            assert await _persisted_preference(agent) == CLOUD

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.PUBLIC)
        assert _mandate(agent) == CLOUD


@pytest.mark.asyncio
async def test_exit_whose_saved_mandate_no_longer_resolves_keeps_the_current_one():
    """A saved mandate the LLM service now refuses is neither installed nor
    persisted. As when nothing was saved, the current preference stays and
    the saved copy is cleared, but the user is told: holding it instead would
    let a later transition replay it over a mandate set since."""
    db = _StoredRows()
    agent = _bare_agent(mode=PrivacyMode.DEIDENTIFIED, preference=LOCAL, db=db)
    agent._privacy_saved_preference = SavedModelPreference(**CLOUD)

    def refuse(model, vendor=None, route=None, *, validate=True):
        raise ValueError(f"no configured route serves {model}")

    agent.llm_service.resolve_model_preference = refuse

    result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

    assert result.applied is True
    assert result.model_switched is None
    assert "anthropic:api/claude-haiku-4-5, could not be restored" in result.message
    assert agent.llm_service.calls == []
    assert agent.llm_service.get_model_preference() == LOCAL
    assert agent._privacy_saved_preference is None
    assert db.mandates() == [LOCAL]
    assert db.records()[-1] == PrivacyState(mode=PrivacyMode.NORMAL)


@pytest.mark.asyncio
async def test_entering_from_a_cloud_mode_saves_the_mandate_in_force_not_a_stale_one():
    """NORMAL still holds a saved mandate (boot restored it but could not
    record that), and the operator has since set another model. Entering a
    local-only mode saves the model in force, the operator's, not the stale
    saved copy that leaving would otherwise replay over it."""
    db = _StoredRows()
    chosen = {"vendor": "openai", "route": "api", "model": "gpt-5-mini"}
    agent = _bare_agent(mode=PrivacyMode.NORMAL, preference=chosen, db=db)
    agent._privacy_saved_preference = SavedModelPreference(**CLOUD)

    await agent.set_privacy_mode_with_effects(PrivacyMode.ISOLATED)
    assert agent._privacy_saved_preference == SavedModelPreference(**chosen)

    await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)
    assert agent.llm_service.get_model_preference() == chosen


@pytest.mark.asyncio
async def test_a_diverged_record_does_not_replace_the_operators_mandate():
    """A refused entry left the record DEIDENTIFIED, holding the mandate then
    in force, while the agent stayed NORMAL; the operator then set another
    model. Moving to PUBLIC records from DEIDENTIFIED (fail closed) but keeps
    the operator's model: in NORMAL nothing was displaced to restore."""
    db = _StoredRows()
    chosen = {"vendor": "openai", "route": "api", "model": "gpt-5-mini"}
    agent = _bare_agent(mode=PrivacyMode.NORMAL, preference=chosen, db=db)
    await write_privacy_state(
        db,
        agent.agent_id,
        PrivacyState(PrivacyMode.DEIDENTIFIED, SavedModelPreference(**CLOUD)),
    )

    result = await agent.set_privacy_mode_with_effects(PrivacyMode.PUBLIC)

    assert result.applied is True
    assert agent.llm_service.get_model_preference() == chosen
    assert db.mandates() == [chosen]
    assert db.records()[1] == PrivacyState(
        PrivacyMode.PUBLIC,
        SavedModelPreference(**CLOUD),
        transition_from=PrivacyMode.DEIDENTIFIED,
    )
    assert db.records()[-1] == PrivacyState(mode=PrivacyMode.PUBLIC)


@pytest.mark.asyncio
async def test_without_an_llm_service_the_saved_mandate_is_carried_as_it_is():
    """With no LLM service there is nothing to restore the saved mandate into,
    so no transition drops it: a later boot with one restores it."""
    db = _StoredRows()
    agent = _bare_agent(mode=PrivacyMode.DEIDENTIFIED, db=db)
    agent.llm_service = None
    saved = SavedModelPreference(**CLOUD)
    agent._privacy_saved_preference = saved

    for mode in (PrivacyMode.NORMAL, PrivacyMode.PUBLIC):
        result = await agent.set_privacy_mode_with_effects(mode)
        assert result.applied is True
        assert db.records()[-1] == PrivacyState(mode=mode, saved_preference=saved)
    assert db.mandates() == []


@pytest.mark.asyncio
async def test_boot_that_cannot_read_the_record_at_all_leaves_it_intact(
    tmp_path, monkeypatch
):
    """Both of boot's reads fail. It fails closed, but writes nothing over a
    record it never saw: that record holds the mandate DEIDENTIFIED displaced,
    and the next boot that can read it restores both."""
    import sqlite3

    from kestrel_sovereign.agent import privacy_state

    saved = SavedModelPreference(**CLOUD)
    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)
        await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)

    async def unreadable(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patched:
        patched.setattr(privacy_state, "_fetch_record_rows", unreadable)
        async with _booted_agent(tmp_path) as agent:
            _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
            assert _mandate(agent) == LOCAL

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.DEIDENTIFIED)
        assert agent._privacy_saved_preference == saved
        await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)
        assert _mandate(agent) == CLOUD


@pytest.mark.asyncio
async def test_unreadable_boot_neither_overwrites_the_record_nor_shadows_its_mandate(
    tmp_path, monkeypatch
):
    """The record holds the mandate ISOLATED displaced, but the persisted
    preference is not a local route (it was lost), so boot's reconcile would
    save that instead. With the record unreadable, reconcile writes nothing;
    and once the record can be read, leaving restores the record's mandate,
    not the one boot guessed."""
    import sqlite3

    from kestrel_sovereign.agent import privacy_state

    saved = SavedModelPreference(**CLOUD)

    async def seed(agent):
        await _seed_model_preference(agent, CLOUD)
        await _seed_record(
            agent, PrivacyState(mode=PrivacyMode.ISOLATED, saved_preference=saved).encode()
        )

    async with _booted_agent(tmp_path, before_initialize=seed) as agent:
        _assert_consistent(agent, PrivacyMode.ISOLATED)
        first_boot = agent
    await _seed_model_preference(first_boot, UNPINNED)

    async def unreadable(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patched:
        patched.setattr(privacy_state, "_fetch_record_rows", unreadable)
        async with _booted_agent(tmp_path) as agent:
            _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
            assert _mandate(agent) == LOCAL
            patched.undo()

            assert await _recorded(agent) == PrivacyState(
                mode=PrivacyMode.ISOLATED, saved_preference=saved
            )
            assert await _persisted_preference(agent) == UNPINNED

            result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

            assert result.applied is True
            assert _mandate(agent) == CLOUD
            assert await _recorded(agent) == PrivacyState(mode=PrivacyMode.NORMAL)


@pytest.mark.asyncio
async def test_malformed_record_is_repaired_keeping_a_saved_mandate_that_validates(
    tmp_path,
):
    """A record whose mode names nothing fails closed, but its saved mandate
    is still valid: the repaired record keeps it, even though the local pin
    needed no routing change."""
    stored = json.dumps(
        {
            "version": 1,
            "mode": "secret",
            "transition_from": None,
            "saved_preference": CLOUD,
        }
    )

    async def seed(agent):
        await _seed_model_preference(agent, LOCAL)
        await _seed_record(agent, stored)

    async with _booted_agent(tmp_path, before_initialize=seed) as agent:
        _assert_consistent(agent, FAIL_CLOSED_PRIVACY_MODE)
        assert _mandate(agent) == LOCAL
        assert await _recorded(agent) == PrivacyState(
            mode=FAIL_CLOSED_PRIVACY_MODE,
            saved_preference=SavedModelPreference(**CLOUD),
        )


@pytest.mark.asyncio
async def test_restore_recovers_nothing_from_a_record_that_cannot_be_reread():
    import sqlite3

    restored = await restore_privacy_state(
        _StoredRows(error=sqlite3.OperationalError("disk I/O error")),
        "did:example:a",
        initial_mode=PrivacyMode.NORMAL,
    )
    assert restored.mode is FAIL_CLOSED_PRIVACY_MODE
    assert restored.saved_preference is None
    # Nothing is known about a record that cannot be read: it is not replaced.
    assert restored.rewrite is False


@pytest.mark.asyncio
async def test_restore_marks_only_records_that_do_not_name_the_restored_state():
    settled = PrivacyState(mode=PrivacyMode.ISOLATED).encode()
    pending = PrivacyState(
        mode=PrivacyMode.NORMAL, transition_from=PrivacyMode.ISOLATED
    ).encode()

    for stored, rewrite in [(settled, False), (pending, True)]:
        restored = await restore_privacy_state(
            _StoredRows(rows=[(stored,)]),
            "did:example:a",
            initial_mode=PrivacyMode.NORMAL,
        )
        assert restored.rewrite is rewrite


# --- A transition that cannot be recorded is refused (#3554 review) ---------


def _is_mandate_write(params) -> bool:
    return (
        isinstance(params, (tuple, list))
        and len(params) > 1
        and params[1] == "model_preference"
    )


def _is_settled_record_write(params) -> bool:
    return (
        isinstance(params, (tuple, list))
        and len(params) > 2
        and params[1] == PRIVACY_STATE_KEY
        and PrivacyState.decode(params[2]).transition_from is None
    )


def _fail_writes(patched, agent, failing) -> None:
    """Fail every write the agent's real database is asked for that ``failing``
    picks, as a full disk would. Reads, and every other write, go through."""
    import sqlite3

    db = agent._raw_storage.db
    execute = db.execute

    async def execute_or_fail(sql, params=()):
        if failing(params):
            raise sqlite3.OperationalError("disk I/O error")
        return await execute(sql, params)

    patched.setattr(db, "execute", execute_or_fail)


@pytest.mark.asyncio
async def test_failed_exit_retried_and_failing_again_survives_a_restart(
    tmp_path, monkeypatch
):
    """Failed exit, repeated exit (also failing), restart.

    Real ``LLMService``; only the mandate write fails. Neither exit applies:
    the agent stays in DEIDENTIFIED on its local route, holding the saved
    mandate, and the pending record keeps both. Before the fix the first
    failed exit had already flipped to NORMAL and dropped the saved copy, so
    the second recorded ``normal -> normal`` with nothing saved, and the
    restart came back NORMAL on the local model with the original mandate
    gone."""
    saved = SavedModelPreference(**CLOUD)
    pending = PrivacyState(
        mode=PrivacyMode.NORMAL,
        saved_preference=saved,
        transition_from=PrivacyMode.DEIDENTIFIED,
    )
    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)
        await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)

        with monkeypatch.context() as patched:
            _fail_writes(patched, agent, _is_mandate_write)
            for _ in range(2):
                result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

                assert result.applied is False
                assert result.persistence_failed is True
                assert result.message == PRIVACY_STATE_UNRECORDED_MESSAGE
                _assert_consistent(agent, PrivacyMode.DEIDENTIFIED)
                assert _mandate(agent) == LOCAL
                assert agent._privacy_saved_preference == saved
                assert await _persisted_preference(agent) == LOCAL
                assert await _recorded(agent) == pending

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.DEIDENTIFIED)
        assert _mandate(agent) == LOCAL
        assert agent._privacy_saved_preference == saved

        result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

        assert result.applied is True
        assert _mandate(agent) == CLOUD
        assert await _persisted_preference(agent) == CLOUD
        assert await _recorded(agent) == PrivacyState(mode=PrivacyMode.NORMAL)


@pytest.mark.asyncio
async def test_failed_exit_then_a_successful_retry_settles_to_normal(
    tmp_path, monkeypatch
):
    """Failed exit, then a retry that succeeds: NORMAL, the original mandate
    restored and persisted, and no pending record or saved copy left."""
    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)
        await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)

        with monkeypatch.context() as patched:
            _fail_writes(patched, agent, _is_mandate_write)
            refused = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)
        assert refused.applied is False

        result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

        assert result.applied is True
        assert result.model_switched == CLOUD
        _assert_consistent(agent, PrivacyMode.NORMAL)
        assert _mandate(agent) == CLOUD
        assert agent._privacy_saved_preference is None
        assert await _persisted_preference(agent) == CLOUD
        assert await _recorded(agent) == PrivacyState(mode=PrivacyMode.NORMAL)

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.NORMAL)
        assert _mandate(agent) == CLOUD


@pytest.mark.asyncio
async def test_failed_settled_write_puts_back_the_state_the_exit_started_from(
    tmp_path, monkeypatch
):
    """The exit's mandate is written, then its settled record fails. The
    refused exit puts back the mandate still in force, then the record it
    started from, so storage again names the state the agent stays in."""
    saved = SavedModelPreference(**CLOUD)
    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)
        await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)

        def settled_normal(params) -> bool:
            return (
                _is_settled_record_write(params)
                and PrivacyState.decode(params[2]).mode is PrivacyMode.NORMAL
            )

        with monkeypatch.context() as patched:
            _fail_writes(patched, agent, settled_normal)
            result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

        assert result.applied is False
        assert result.persistence_failed is True
        _assert_consistent(agent, PrivacyMode.DEIDENTIFIED)
        assert _mandate(agent) == LOCAL
        assert await _persisted_preference(agent) == LOCAL
        assert await _recorded(agent) == PrivacyState(
            mode=PrivacyMode.DEIDENTIFIED, saved_preference=saved
        )


@pytest.mark.asyncio
async def test_a_later_failed_transition_keeps_what_a_pending_record_guarantees(
    tmp_path, monkeypatch
):
    """A refused entry whose record could not be put back leaves a pending
    record that resolves to DEIDENTIFIED, while the agent stays NORMAL. A
    later transition that fails as well starts from that record, not from the
    live state: its own pending record still resolves to DEIDENTIFIED and
    keeps the mandate DEIDENTIFIED displaced, so a restart comes back
    local-only, on its local route, and leaving restores the mandate."""
    saved = SavedModelPreference(**CLOUD)
    async with _booted_agent(tmp_path) as agent:
        await _pin_cloud_mandate(agent)

        with monkeypatch.context() as patched:
            _fail_writes(patched, agent, _is_settled_record_write)
            entry = await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)
            assert entry.applied is False
            assert entry.message == (
                PRIVACY_STATE_UNRECORDED_MESSAGE
                + " Until then, a restart resumes in deidentified mode."
            )
            _assert_consistent(agent, PrivacyMode.NORMAL)
            assert _mandate(agent) == CLOUD
            # The forced route was written, then the mandate in force put back.
            assert await _persisted_preference(agent) == CLOUD

            public = await agent.set_privacy_mode_with_effects(PrivacyMode.PUBLIC)
            assert public.applied is False
            _assert_consistent(agent, PrivacyMode.NORMAL)
            assert await _recorded(agent) == PrivacyState(
                mode=PrivacyMode.PUBLIC,
                saved_preference=saved,
                transition_from=PrivacyMode.DEIDENTIFIED,
            )

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.DEIDENTIFIED)
        assert _mandate(agent) == LOCAL
        assert agent._privacy_saved_preference == saved

        await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)
        assert _mandate(agent) == CLOUD


@pytest.mark.asyncio
async def test_boot_restore_whose_mandate_is_not_persisted_keeps_the_saved_copy(
    tmp_path, monkeypatch
):
    """A cloud mode that still holds a saved mandate restores it at boot. If
    the restored mandate cannot be persisted, the record keeps the saved copy
    so the next boot restores it again."""
    import sqlite3

    from kestrel_sovereign.storage.async_database import AsyncDatabase

    saved = SavedModelPreference(**CLOUD)

    async def seed(agent):
        await _seed_model_preference(agent, LOCAL)
        await _seed_record(
            agent, PrivacyState(mode=PrivacyMode.NORMAL, saved_preference=saved).encode()
        )

    execute = AsyncDatabase.execute

    async def mandate_write_fails(self, sql, params=()):
        if _is_mandate_write(params):
            raise sqlite3.OperationalError("disk I/O error")
        return await execute(self, sql, params)

    with monkeypatch.context() as patched:

        async def seed_then_fail_mandate_writes(agent):
            await seed(agent)
            patched.setattr(AsyncDatabase, "execute", mandate_write_fails)

        async with _booted_agent(
            tmp_path, before_initialize=seed_then_fail_mandate_writes
        ) as agent:
            _assert_consistent(agent, PrivacyMode.NORMAL)
            assert _mandate(agent) == CLOUD
            assert await _persisted_preference(agent) == LOCAL
            assert await _recorded(agent) == PrivacyState(
                mode=PrivacyMode.NORMAL, saved_preference=saved
            )

    async with _booted_agent(tmp_path) as agent:
        _assert_consistent(agent, PrivacyMode.NORMAL)
        assert _mandate(agent) == CLOUD
        assert await _persisted_preference(agent) == CLOUD
        assert await _recorded(agent) == PrivacyState(mode=PrivacyMode.NORMAL)


class _MandateUnwritableRows(_StoredRows):
    async def execute(self, sql, params):
        import sqlite3

        if _is_mandate_write(params):
            raise sqlite3.OperationalError("disk I/O error")
        await super().execute(sql, params)


def _deidentified_bare_agent_with_saved_mandate():
    agent = _bare_agent(
        mode=PrivacyMode.DEIDENTIFIED, preference=LOCAL, db=_MandateUnwritableRows()
    )
    agent._privacy_saved_preference = SavedModelPreference(**CLOUD)
    return agent


@pytest.mark.asyncio
async def test_transition_whose_mandate_cannot_be_written_is_refused():
    agent = _deidentified_bare_agent_with_saved_mandate()

    result = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)

    assert result.applied is False
    assert result.persistence_failed is True
    _assert_consistent(agent, PrivacyMode.DEIDENTIFIED)
    assert agent.llm_service.calls == []
    assert agent._privacy_saved_preference == SavedModelPreference(**CLOUD)
    # The pending record stays: putting back the record it started from
    # needs the mandate written first, and that write fails too.
    pending = PrivacyState(
        PrivacyMode.NORMAL,
        SavedModelPreference(**CLOUD),
        transition_from=PrivacyMode.DEIDENTIFIED,
    )
    assert agent._raw_storage.db.records() == [pending, pending]


@pytest.mark.asyncio
async def test_endpoint_reports_a_transition_whose_mandate_was_not_written_as_refused():
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from tests.unit.test_privacy_mode_model_restore_contracts import (
        _prepare_app,
        _restore_app,
    )

    agent = _deidentified_bare_agent_with_saved_mandate()
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agent/privacy-mode",
                    headers={
                        "X-API-Key": "test-key",
                        "X-Kestrel-Allow-Destructive": "test-rail-bypass",
                    },
                    json={"mode": "normal"},
                )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert body["persistence_failed"] is True
        assert body["mode"] == "deidentified"
        assert body["message"] == PRIVACY_STATE_UNRECORDED_MESSAGE
    finally:
        _restore_app(app, original)


# --- Where a transition starts from ------------------------------------------


@pytest.mark.parametrize(
    ("live_mode", "live_saved", "recorded_mode", "recorded_saved", "expected"),
    [
        # Nothing went wrong: the record names the live state.
        (
            PrivacyMode.DEIDENTIFIED,
            SavedModelPreference(**CLOUD),
            PrivacyMode.DEIDENTIFIED,
            SavedModelPreference(**CLOUD),
            PrivacyState(PrivacyMode.DEIDENTIFIED, SavedModelPreference(**CLOUD)),
        ),
        # A refused entry left a pending record that resolves local-only.
        (
            PrivacyMode.NORMAL,
            None,
            PrivacyMode.DEIDENTIFIED,
            SavedModelPreference(**CLOUD),
            PrivacyState(PrivacyMode.DEIDENTIFIED, SavedModelPreference(**CLOUD)),
        ),
        # A refused exit left the target recorded: the live mode is stricter.
        (
            PrivacyMode.ISOLATED,
            SavedModelPreference(**CLOUD),
            PrivacyMode.NORMAL,
            None,
            PrivacyState(PrivacyMode.ISOLATED, SavedModelPreference(**CLOUD)),
        ),
        # The stricter side has no saved copy; the other side's is kept.
        (
            PrivacyMode.EPHEMERAL,
            None,
            PrivacyMode.NORMAL,
            SavedModelPreference(**CLOUD),
            PrivacyState(PrivacyMode.EPHEMERAL, SavedModelPreference(**CLOUD)),
        ),
        # Both hold one: the record's, not the one boot guessed while the
        # record was unreadable.
        (
            PrivacyMode.EPHEMERAL,
            SavedModelPreference(None, None, None),
            PrivacyMode.ISOLATED,
            SavedModelPreference(**CLOUD),
            PrivacyState(PrivacyMode.EPHEMERAL, SavedModelPreference(**CLOUD)),
        ),
    ],
    ids=["agree", "pending-entry", "refused-exit", "saved-fallback", "record-wins"],
)
def test_transition_starts_from_the_more_restrictive_of_live_and_recorded(
    live_mode, live_saved, recorded_mode, recorded_saved, expected
):
    from kestrel_sovereign.agent.privacy_state import (
        RestoredPrivacyState,
        transition_origin,
    )

    recorded = RestoredPrivacyState(
        mode=recorded_mode, saved_preference=recorded_saved, recorded=True
    )
    assert transition_origin(live_mode, live_saved, recorded) == expected


@pytest.mark.asyncio
async def test_transition_is_refused_when_the_record_cannot_be_read():
    """The record is where a transition starts from; one it cannot read is
    refused before anything is written."""
    import sqlite3

    db = _StoredRows(error=sqlite3.OperationalError("database is locked"))
    agent = _bare_agent(db=db)

    result = await agent.set_privacy_mode_with_effects(PrivacyMode.DEIDENTIFIED)

    assert result.applied is False
    assert result.persistence_failed is True
    assert db.writes == []
    _assert_consistent(agent, PrivacyMode.NORMAL)
    assert agent.llm_service.calls == []
