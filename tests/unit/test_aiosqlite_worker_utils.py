"""Tests for aiosqlite worker-lifecycle synchronization helpers."""

import asyncio

import pytest

from tests.utils.aiosqlite_workers import wait_for_lifecycle_checkpoint


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_wins_without_waiting_for_owner() -> None:
    checkpoint = asyncio.Event()
    owner_release = asyncio.Event()

    async def owner() -> None:
        await owner_release.wait()

    owner_task = asyncio.create_task(owner())
    checkpoint.set()
    try:
        await wait_for_lifecycle_checkpoint(
            checkpoint.wait(),
            owner_task,
            description="checkpoint reached",
            harness_timeout=0.1,
        )
        assert not owner_task.done()
    finally:
        owner_release.set()
        await owner_task


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_rejects_premature_completion() -> None:
    async def owner() -> None:
        return None

    owner_task = asyncio.create_task(owner())
    await owner_task

    with pytest.raises(
        AssertionError,
        match="Lifecycle completed before checkpoint reached",
    ):
        await wait_for_lifecycle_checkpoint(
            asyncio.Event().wait(),
            owner_task,
            description="checkpoint reached",
            harness_timeout=0.1,
        )


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_fails_closed_when_both_are_ready() -> None:
    checkpoint = asyncio.Event()

    async def owner() -> None:
        return None

    owner_task = asyncio.create_task(owner())
    checkpoint.set()
    await owner_task

    with pytest.raises(
        AssertionError,
        match="Lifecycle completed before checkpoint reached",
    ):
        await wait_for_lifecycle_checkpoint(
            checkpoint.wait(),
            owner_task,
            description="checkpoint reached",
            harness_timeout=0.1,
        )


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_reports_owner_failure() -> None:
    failure = RuntimeError("owner failed")

    async def owner() -> None:
        raise failure

    owner_task = asyncio.create_task(owner())
    await asyncio.wait({owner_task})

    with pytest.raises(
        AssertionError,
        match="Lifecycle failed before checkpoint reached",
    ) as captured:
        await wait_for_lifecycle_checkpoint(
            asyncio.Event().wait(),
            owner_task,
            description="checkpoint reached",
            harness_timeout=0.1,
        )

    assert captured.value.__cause__ is failure


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_reports_owner_cancellation() -> None:
    owner_task = asyncio.create_task(asyncio.Event().wait())
    owner_task.cancel()
    await asyncio.wait({owner_task})

    with pytest.raises(
        AssertionError,
        match="Lifecycle cancelled before checkpoint reached",
    ):
        await wait_for_lifecycle_checkpoint(
            asyncio.Event().wait(),
            owner_task,
            description="checkpoint reached",
            harness_timeout=0.1,
        )


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_tolerates_contracted_owner_failure() -> None:
    """An owner whose failure IS the asserted behavior must not read as a flake.

    ``test_factory_dispose_error_preserves_cleanup_timeout_for_checked_out_worker``
    (in ``tests/unit/storage/test_db_backends.py``) patches the shutdown drain
    to 0.01s and asserts ``db.close()`` raises, so the owner terminating before
    the held worker publishes its checkpoint is a legitimate scheduling order —
    aiosqlite resolves the close future before it reaches the injected exit
    delay, making the race genuine rather than theoretical.  That is the only
    opted-out call site; the default fail-closed rule still applies everywhere
    else (see ``test_lifecycle_checkpoint_reports_owner_failure``).
    """
    failure = RuntimeError("worker did not terminate")
    checkpoint = asyncio.Event()

    async def owner() -> None:
        raise failure

    owner_task = asyncio.create_task(owner())
    await asyncio.wait({owner_task})
    assert owner_task.done()

    async def publish_checkpoint_after_owner_failed() -> None:
        await asyncio.sleep(0)
        checkpoint.set()

    publisher = asyncio.create_task(publish_checkpoint_after_owner_failed())
    try:
        await wait_for_lifecycle_checkpoint(
            checkpoint.wait(),
            owner_task,
            description="the delayed worker exit",
            harness_timeout=1.0,
            require_live_lifecycle=False,
        )
    finally:
        await publisher

    assert checkpoint.is_set()
    # The owner's failure stays intact for the caller to assert on itself.
    assert owner_task.exception() is failure


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_has_finite_deadlock_guard() -> None:
    owner_release = asyncio.Event()

    async def owner() -> None:
        await owner_release.wait()

    owner_task = asyncio.create_task(owner())
    try:
        with pytest.raises(
            AssertionError,
            match="made no progress within the 0.001s test-harness guard",
        ):
            await wait_for_lifecycle_checkpoint(
                asyncio.Event().wait(),
                owner_task,
                description="checkpoint reached",
                harness_timeout=0.001,
            )
    finally:
        owner_release.set()
        await owner_task
