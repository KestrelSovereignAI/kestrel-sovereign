"""A real agent on real storage, booted for tests that script its model calls.

The agent loads only the features a test names, plus the mandatory set every
agent carries. Left unfiltered, ``initialize()`` also loads every optional
feature package installed in the environment, and their hooks make model calls
of their own through the scripted providers. A developer machine with
``kestrel-feature-reflection`` installed sees its STOP hook take a scripted
answer that CI, without the package, never hands out (#3578).
"""

from __future__ import annotations

from kestrel_sovereign.testing.native_shutdown_budget import shutdown_fixture_agent

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AbstractSet


@asynccontextmanager
async def booted_agent(
    tmp_path,
    *,
    name: str,
    provenance: str,
    features: AbstractSet[str],
    genesis: bool = True,
    released_schema: Path | None = None,
    is_test_instance: bool = True,
):
    """A booted agent past genesis and discovery, loading only ``features``
    (Feature class names) beyond the mandatory set.

    ``genesis=False`` leaves the genesis audit pending, so the agent's first
    turn runs it through its own auditor.

    ``released_schema`` names a snapshot from ``tests.shared.agent_store_schema``.
    The agent's database then starts as that release's fresh database, and the
    agent boots across whatever upgrade its stores need (#4002).

    ``is_test_instance=False`` creates a separate test-owned identity with
    interactive approvals enabled; the default identity is tagged headless.
    """
    from kestrel_sovereign.bootstrap import BootstrapState
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES
    from tests.shared.genesis_audit import complete_deterministic_genesis_audit

    db_path = Path(tmp_path) / "kestrel_prime.db"
    if released_schema is None:
        credentials = await create_kestrel_identity_async(
            output_dir=str(tmp_path), is_test_instance=is_test_instance, agent_name=name
        )
    else:
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from tests.shared.agent_store_schema import load

        Path(tmp_path).mkdir(parents=True, exist_ok=True)
        load(db_path, released_schema)
        database = await AsyncDatabase.sqlite(str(db_path))
        try:
            credentials = await create_kestrel_identity_async(
                output_dir=str(tmp_path),
                is_test_instance=is_test_instance,
                agent_name=name,
                database=database,
            )
        finally:
            await database.close()
    llm_service = LLMService()
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=str(db_path),
        llm_service=llm_service,
        allowed_features=set(features),
    )
    try:
        await agent.initialize()
        loaded = {type(feature).__name__ for feature in agent.features.values()}
        assert loaded == MANDATORY_FEATURES | features, (
            f"the agent loaded {sorted(loaded)}, not only the features asked for"
        )
        if genesis:
            await complete_deterministic_genesis_audit(agent, provenance=provenance)
        await agent.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
        yield agent
    finally:
        await shutdown_fixture_agent(agent)
        await llm_service.close()
