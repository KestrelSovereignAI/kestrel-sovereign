"""Reopen the original native-MCP agent in a separate owned Python process."""

import asyncio
import json
import os
import sys
from pathlib import Path

from fastapi.encoders import jsonable_encoder
from kestrel_sdk.payer_policy import PayerPolicy

from kestrel_sovereign.agent.custody import ReleaseOutcome
from kestrel_sovereign.kestrel_agent import (
    KestrelAgent,
    agent_shutdown_completion_owners,
)
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import memory_consolidator
from kestrel_sovereign.storage.db.sqlite import _aiosqlite_worker_is_alive
from kestrel_sovereign.testing.native_shutdown_budget import shutdown_fixture_agent


async def close_restart_agent(agent):
    """Join original completion owners and observe the actual SQLite workers."""
    backend = agent._raw_storage.db.backend
    connections = tuple(connection for connection in (
        backend._connection, *backend._retired_connection_closes,
    ) if connection is not None)
    await shutdown_fixture_agent(agent)
    await agent.wait_for_shutdown_completion(defer_cancellation=False)
    remaining_connections = (*connections, *backend._retired_connection_closes)
    if (agent._unreleased_feature_registry() or backend.is_connected
            or any(_aiosqlite_worker_is_alive(connection) for connection in remaining_connections)
            or any(not owner.done() for owner in agent_shutdown_completion_owners(agent))):
        raise RuntimeError("Restart shutdown did not retire its original cleanup and storage owners")
    return {"primary_sqlite_closed": True, "original_primary_workers": len(connections)}


async def main():
    request = json.load(sys.stdin)
    llm = LLMService()
    agent = KestrelAgent(did=request["did"], storage_path=request["storage_path"],
        llm_service=llm, allowed_features={"MemoryAgencyFeature"},
        payer_policy=PayerPolicy.model_validate(request["payer_policy"]))
    try:
        await agent.initialize()
        rows = await agent._raw_storage.conversation.get_messages_by_ids(request["receipts"])
        await agent.set_privacy_mode(PrivacyMode(request["mode"]))
        report = await agent.memory_system.consolidate()
        episodes = [episode.to_dict() for episode in await agent.memory_system.get_episodes()]
        repeat_report = await agent.memory_system.consolidate()
        repeat_episodes = [episode.to_dict() for episode in await agent.memory_system.get_episodes()]
        stored_episode_ids = await agent._raw_storage.db.fetchall(
            "SELECT id FROM memory_episodes WHERE agent_id = ?", (agent.did,))
        result = {"pid": os.getpid(), "did": agent.did,
            "core_source": str(Path(memory_consolidator.__file__).resolve()),
            "generation_disabled": agent.llm_service.disabled, "rows": rows, "report": report,
            "episodes": episodes, "repeat_report": repeat_report,
            "repeat_episodes": repeat_episodes, "stored_episode_ids": stored_episode_ids}
    finally:
        try:
            cleanup = await close_restart_agent(agent)
        finally:
            await llm.close()
    # Publish this storage witness only after original native cleanup is verified.
    result["cleanup"] = cleanup
    Path(request["receipt_path"]).write_text(json.dumps(jsonable_encoder(result)), encoding="utf-8")


async def main_with_retained_storage():
    """Exercise the actual child entry point with its native close unwired."""
    original_shutdown = KestrelAgent.shutdown
    original_stores = []

    async def shutdown_with_retained_storage(agent):
        storage = agent._raw_storage
        actual_close = storage.close
        original_stores.append(actual_close)

        async def retain_storage():
            return ReleaseOutcome.RETAINED

        storage.close = retain_storage
        try:
            return await original_shutdown(agent)
        finally:
            storage.close = actual_close

    KestrelAgent.shutdown = shutdown_with_retained_storage
    try:
        await main()
    finally:
        KestrelAgent.shutdown = original_shutdown
        # Keep the original failure, while physically releasing the exact
        # retained native worker so this test-owned child can terminate.
        for actual_close in original_stores:
            await actual_close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--retain-storage"]:
        asyncio.run(main_with_retained_storage())
    elif not sys.argv[1:]:
        asyncio.run(main())
    else:
        raise ValueError("Unknown native restart probe arguments")
