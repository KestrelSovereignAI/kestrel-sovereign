"""Durable, cross-agent ownership tests for externally provisioned routes."""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sovereign.features.channels.route_ownership import (
    ChannelRouteClaim,
    ChannelRouteOwnershipStore,
)
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


@pytest.mark.asyncio
async def test_route_claim_is_exclusive_reconcilable_and_does_not_leak_competing_agent(
    tmp_path,
) -> None:
    """Separate Core connections contend as independent host processes would."""
    db_path = str(tmp_path / "shared-core-routes.db")
    first_backend = SQLiteBackend(db_path)
    second_backend = SQLiteBackend(db_path)
    await first_backend.connect()
    await second_backend.connect()
    first = ChannelRouteOwnershipStore(first_backend)
    second = ChannelRouteOwnershipStore(second_backend)
    bot_identity = "telegram-bot:123456"
    first_agent = "did:test:telegram-route-first"
    second_agent = "did:test:telegram-route-second"

    try:
        first_claim, second_claim = await asyncio.gather(
            first.claim(
                channel_type="telegram",
                canonical_route_identity=bot_identity,
                agent_id=first_agent,
            ),
            second.claim(
                channel_type="telegram",
                canonical_route_identity=bot_identity,
                agent_id=second_agent,
            ),
        )
        assert sum(claim is not None for claim in (first_claim, second_claim)) == 1
        winner, winning_agent, winning_claim = (
            (first, first_agent, first_claim)
            if first_claim is not None
            else (second, second_agent, second_claim)
        )
        losing_store, losing_agent = (
            (second, second_agent) if first_claim is not None else (first, first_agent)
        )
        assert winning_claim is not None

        assert await winner.is_claimed_by(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=winning_agent,
        ) is True
        # A contending user can learn only that THEIR request failed.  The
        # primitive neither returns nor exposes the winning agent identity.
        assert await losing_store.is_claimed_by(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=losing_agent,
        ) is False
        assert not hasattr(losing_store, "get_owner")

        # Reassertion replaces the generation. A stale same-agent release and
        # a competing-agent release can never delete that replacement.
        replacement_claim = await winner.claim(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=winning_agent,
        )
        assert replacement_claim is not None
        assert replacement_claim != winning_claim
        assert await winner.release(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=winning_agent,
            claim=winning_claim,
        ) is False
        assert await losing_store.release(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=losing_agent,
            claim=ChannelRouteClaim(generation="competing-generation"),
        ) is False
        assert await winner.release(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=winning_agent,
            claim=replacement_claim,
        ) is True
        assert (
            await losing_store.claim(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=losing_agent,
            )
            is not None
        )
    finally:
        await second_backend.close()
        await first_backend.close()
