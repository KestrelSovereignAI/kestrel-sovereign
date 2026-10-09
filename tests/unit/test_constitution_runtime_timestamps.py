"""Runtime state must preserve TIMESTAMPTZ intent through the actual adapter."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.constitution.runtime_state import (
    ConstitutionRuntimeState,
    ConstitutionRuntimeStateStore,
)
from kestrel_sovereign.storage.db.postgres import PostgresBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", [0, -5, 5.5])
async def test_state_and_event_timestamp_binds_preserve_utc_instants(offset):
    bound = []

    class Backend:
        backend_type = "postgres"

        @asynccontextmanager
        async def transaction(self):
            yield

        async def execute(self, query, params=()):
            bound.append((query, PostgresBackend._strip_tz(params)))
            return 1

        async def fetch_one(self, query, params=()):
            bound.append((query, PostgresBackend._strip_tz(params)))
            return (1,)

    utc = datetime(2026, 10, 9, 13, 57, tzinfo=timezone.utc)
    local = utc.astimezone(timezone(timedelta(hours=offset)))
    state = ConstitutionRuntimeState(
        agent_id="did:test:timestamp",
        safe_mode=False,
        safe_mode_reason=None,
        safe_mode_entered_at=local - timedelta(minutes=3),
        safe_mode_exited_at=local,
        safe_mode_exit_authorization="signed-owner",
        last_successful_audit_at=local,
        interaction_count=0,
        updated_at=local,
    )
    await ConstitutionRuntimeStateStore(Backend()).write(
        state, event_type="safe_mode_exited", event_authorization="signed-owner"
    )

    assert len(bound) == 2
    state_params, event_params = bound[0][1], bound[1][1]
    assert state_params[3] == utc - timedelta(minutes=3)
    for value in (state_params[3], state_params[4], state_params[6], state_params[10], event_params[4]):
        assert value.tzinfo == timezone.utc
    assert state_params[4] == state_params[6] == state_params[10] == event_params[4] == utc


@pytest.mark.parametrize("backend_type", ["sqlite", "postgres"])
@pytest.mark.parametrize("naive", [False, True])
def test_optional_and_legacy_naive_runtime_parameters_keep_utc_contract(backend_type, naive):
    store = ConstitutionRuntimeStateStore(SimpleNamespace(backend_type=backend_type))
    value = datetime(2026, 10, 9, 13, 57)
    if not naive:
        value = value.replace(tzinfo=timezone.utc)
    assert store._timestamp_param(None) is None
    parameter = store._timestamp_param(value)
    if backend_type == "postgres":
        actual, = PostgresBackend._strip_tz((parameter,))
        assert actual == value.replace(tzinfo=timezone.utc)
        assert actual.tzinfo == timezone.utc
    else:
        assert parameter == "2026-10-09T13:57:00+00:00"


def test_generic_legacy_timestamp_binding_is_not_changed():
    aware = datetime(2026, 10, 9, 8, 57, tzinfo=timezone(timedelta(hours=-5)))
    assert PostgresBackend._strip_tz((aware,)) == (aware.replace(tzinfo=None),)
