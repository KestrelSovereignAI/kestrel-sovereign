"""Fail-closed observability privacy gates on SQLite and PostgreSQL (#3920).

The unit tier proves the gates on SQLite. These cases run the same writers and
assertions against a real PostgreSQL backend too: a failed gate persists no
content, metering still records, and unconfigured or NORMAL storage keeps
content exactly as before.
"""

import pytest

from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_observability_privacy_gating import (
    FAIL_CLOSED_SCENARIOS,
    assert_content_kept_as_today,
    assert_elided_rows_read_back,
    assert_metering_recorded,
    assert_nothing_leaked,
    write_every_content_field,
)
from tests.utils.postgres_schema import postgres_test_url, with_search_path


@pytest.fixture(params=["sqlite", "postgres"])
async def observability_backend(request, tmp_path):
    """A connected backend; the PostgreSQL one in a schema of its own."""
    if request.param == "sqlite":
        backend = SQLiteBackend(str(tmp_path / "observability.db"))
        await backend.connect()
        try:
            yield backend
        finally:
            await backend.close()
        return

    import asyncpg

    from kestrel_sovereign.storage.db.postgres import PostgresBackend
    from tests.utils.postgres_schema import disposable_postgres_schema

    dsn = postgres_test_url()
    if not dsn:
        pytest.skip("TEST_POSTGRES_URL required for the PostgreSQL leg")
    control = await asyncpg.connect(dsn)
    try:
        async with disposable_postgres_schema(control, "obs_privacy") as schema:
            backend = PostgresBackend(with_search_path(dsn, schema))
            await backend.connect()
            try:
                yield backend
            finally:
                await backend.close()
    finally:
        await control.close()


async def _store(backend):
    store = ObservabilityStore(backend, owns_backend=False)
    await store.initialize()
    return store


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("scenario", sorted(FAIL_CLOSED_SCENARIOS))
async def test_failed_gate_persists_no_content_on_both_backends(
    observability_backend, monkeypatch, scenario
):
    store = await _store(observability_backend)
    FAIL_CLOSED_SCENARIOS[scenario](store, monkeypatch)
    await write_every_content_field(store)

    await assert_nothing_leaked(observability_backend)
    await assert_metering_recorded(store, observability_backend)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("mode", [None, "normal"])
async def test_unconfigured_and_normal_keep_content_on_both_backends(
    observability_backend, mode
):
    store = await _store(observability_backend)
    if mode is not None:
        store.set_privacy_config_provider(lambda: get_privacy_preset(mode))
    await write_every_content_field(store)

    await assert_content_kept_as_today(observability_backend)
    await assert_metering_recorded(store, observability_backend)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("scenario", ["provider-raises", "anonymizer-raises"])
async def test_elided_rows_read_back_on_both_backends(
    observability_backend, monkeypatch, scenario
):
    store = await _store(observability_backend)
    await assert_elided_rows_read_back(store, monkeypatch, scenario)
