"""Pure Task-owner policy checks; native ingress proof is an integration test."""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    gather_owned,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [asyncio.FIRST_EXCEPTION, "unsupported"])
async def test_invalid_race_policy_does_not_consume_operations(policy):
    consumed = []

    def operations():
        consumed.append(True)
        yield asyncio.sleep(0)

    with pytest.raises(ValueError, match="ALL_COMPLETED or FIRST_COMPLETED"):
        await gather_owned(operations(), return_when=policy)
    assert consumed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [asyncio.ALL_COMPLETED, asyncio.FIRST_COMPLETED])
async def test_empty_owned_gather_has_no_waiter(policy):
    assert await gather_owned(iter(()), return_when=policy) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [asyncio.ALL_COMPLETED, asyncio.FIRST_COMPLETED])
async def test_owned_gather_default_waits_but_race_retires_peer(policy):
    entered, release, winner = asyncio.Event(), asyncio.Event(), asyncio.Event()
    originals, cancellations = [], []

    async def first():
        await entered.wait()
        winner.set()
        return "winner"

    async def second():
        originals.append(asyncio.current_task())
        entered.set()
        try:
            await release.wait()
            return "second"
        except asyncio.CancelledError as error:
            cancellations.append(error)
            raise

    owner = create_owned_task(
        gather_owned((operation() for operation in (first, second)), return_when=policy)
    )
    try:
        await asyncio.wait_for(winner.wait(), 5)
        if policy == asyncio.ALL_COMPLETED:
            assert not owner.done() and not cancellations
            release.set()
        outcome = await asyncio.wait_for(await_owned_task(owner), 5)
        assert outcome.error is None
        assert originals[0].done()
        if policy == asyncio.FIRST_COMPLETED:
            assert cancellations and outcome.result == ["winner", None]
        else:
            assert not cancellations and outcome.result == ["winner", "second"]
    finally:
        release.set()
        await await_owned_task(owner)
