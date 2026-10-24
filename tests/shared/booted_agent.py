"""A real agent on real storage, booted for tests that script its model calls.

The agent loads only the features a test names, plus the mandatory set every
agent carries. Left unfiltered, ``initialize()`` also loads every optional
feature package installed in the environment, and their hooks make model calls
of their own through the scripted providers. A developer machine with
``kestrel-feature-reflection`` installed sees its STOP hook take a scripted
answer that CI, without the package, never hands out (#3578).
"""

from __future__ import annotations

from tests.shared.native_shutdown_budget import shutdown_fixture_agent

import os
from contextlib import asynccontextmanager
from typing import AbstractSet


@asynccontextmanager
async def booted_agent(
    tmp_path,
    *,
    name: str,
    provenance: str,
    features: AbstractSet[str],
    genesis: bool = True,
):
    """A booted agent past genesis and discovery, loading only ``features``
    (Feature class names) beyond the mandatory set.

    ``genesis=False`` leaves the genesis audit pending, so the agent's first
    turn runs it through its own auditor."""
    from kestrel_sovereign.bootstrap import BootstrapState
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES
    from tests.shared.genesis_audit import complete_deterministic_genesis_audit

    credentials = await create_kestrel_identity_async(
        output_dir=str(tmp_path), is_test_instance=True, agent_name=name
    )
    llm_service = LLMService()
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=os.path.join(str(tmp_path), "kestrel_prime.db"),
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
