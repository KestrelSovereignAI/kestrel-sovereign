"""Standalone restoration's accepted native owner is not a success on control.

This exercises actual SpawnFeature shutdown, actual PG work, original owned
Task cancellation and an actual retired advisory-lease denial. The accepted
restore handle/Agent view are declared component inputs, not Manager tree
recovery, full spawned Agent READY or retry authority.
"""

import asyncio
import secrets
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.features.spawn.feature import SpawnFeature
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["native-control", "native-cancel", "ordinary", "healthy"]
)
async def test_spawn_shutdown_retains_original_restore_control(native_pg, kind):
    backend, observer, schema = native_pg
    await backend.execute("CREATE TABLE restore_effects (id INTEGER PRIMARY KEY)")
    async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
        lease.require_live()
    feature = SpawnFeature(
        SimpleNamespace(did="did:test:restore", agent_id="did:test:restore")
    )
    await feature.initialize()
    entered, retired = asyncio.Event(), asyncio.Event()
    originals = []
    ordinary = OSError("ordinary original restoration failure")

    async def restore():
        await backend.execute("INSERT INTO restore_effects VALUES (1)")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            try:
                if kind.startswith("native"):
                    try:
                        lease.require_live()
                    except BaseException as original:
                        originals.append(original)
                        if kind == "native-cancel":
                            raise asyncio.CancelledError(
                                "original native restoration observation"
                            ) from original
                        raise
                if kind == "ordinary":
                    raise ordinary
            finally:
                retired.set()

    task = create_owned_task(
        restore(), name="accepted-standalone-restore", retain_originals=True
    )
    feature._standalone_restore_task = task
    try:
        await asyncio.wait_for(entered.wait(), 5)
        caught = None
        try:
            await feature.shutdown()
        except BaseException as error:
            caught = error
        assert task.done() and retired.is_set()
        assert feature._standalone_restore_task is None
        assert (
            await observer.fetchval(f'SELECT count(*) FROM "{schema}".restore_effects')
            == 1
        )
        if kind.startswith("native"):
            assert (
                originals
                and caught is not None
                and originals[0] in tuple(_explicit_control_chain(caught))
            ), "spawn shutdown normalized original native restore control"
            assert is_execution_control_error(caught)
        elif kind == "ordinary":
            assert caught is ordinary
        else:
            assert caught is None
    finally:
        if not task.done():
            task.cancel()
        await await_owned_task(task)
