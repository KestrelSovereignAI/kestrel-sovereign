"""Durable, cross-agent ownership tests for externally provisioned routes."""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sovereign.features.channels.route_ownership import (
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
        assert [first_claim, second_claim].count(True) == 1
        winner, loser = (
            (first, first_agent) if first_claim else (second, second_agent)
        )
        losing_store, losing_agent = (
            (second, second_agent) if first_claim else (first, first_agent)
        )

        assert await winner.is_claimed_by(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=loser,
        ) is True
        # A contending user can learn only that THEIR request failed.  The
        # primitive neither returns nor exposes the winning agent identity.
        assert await losing_store.is_claimed_by(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=losing_agent,
        ) is False
        assert not hasattr(losing_store, "get_owner")

        # Same-agent reconciliation is idempotent, while another agent cannot
        # release a route it does not own.
        assert await winner.claim(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=loser,
        ) is True
        assert await losing_store.release(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=losing_agent,
        ) is False
        assert await winner.release(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=loser,
        ) is True
        assert await losing_store.claim(
            channel_type="telegram",
            canonical_route_identity=bot_identity,
            agent_id=losing_agent,
        ) is True
    finally:
        await second_backend.close()
        await first_backend.close()
