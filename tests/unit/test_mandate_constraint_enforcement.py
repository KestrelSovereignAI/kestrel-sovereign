"""Runtime enforcement of SpawnMandate additional_constraints (#2137).

  1. Durable record: the mandate's constraints are persisted on the spawned_by
     delegation edge, so enforcement can be reattached at load (survives restart)
     without rewriting — and breaking the integrity hash of — the base
     constitution.
  2. Hard enforcement: a spawned child cannot actually invoke a `restricted_tools`
     entry — the MandateRestrictionHook denies it at PRE_TOOL_USE, applied
     uniformly on the load path (fresh spawn, reload, restart).

Rendering the mandate's behavioral_rules into the child's effective constitution
via the #1722 per-agent overlay is a follow-up (kept out of here so the base
constitution stays canonical and integrity-verifiable).
"""

import os
from types import SimpleNamespace

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookInput, PermissionDecision
from kestrel_sovereign.hooks import HooksManager, evaluate_blocking_decision
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate
from kestrel_sovereign.spawn.mandate_hook import MandateRestrictionHook
from kestrel_sovereign.spawn.mandate_reload import (
    read_spawn_mandate,
    register_restriction_hook,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.inception_service import generate_secp256k1_keypair


BEHAVIOR_RULE = "Never contact external endpoints without human approval."
RESTRICTED_TOOL = "computer_use_shell"


@pytest.mark.asyncio
async def test_mandate_constraints_persisted_on_delegation_edge(tmp_path):
    """The mandate's constraints are recorded on the spawned_by delegation edge,
    the durable machine-readable record the load path reattaches enforcement
    from. (The anchored base constitution is left canonical so its integrity
    hash still verifies — the enforcement path is the edge + the load-time hook,
    not a rewrite of the base constitution.)"""
    parent_private, _ = generate_secp256k1_keypair()
    parent_did = "did:pkh:eip155:1:0xParentEnforce"

    mandate = SpawnMandate(
        parent_did=parent_did,
        purpose="scoped worker",
        ttl_seconds=3600,
        max_child_depth=1,
        additional_constraints={
            "behavioral_rules": [BEHAVIOR_RULE],
            "restricted_tools": [RESTRICTED_TOOL],
        },
    )
    mandate = sign_mandate(mandate, parent_private)

    creds = await create_kestrel_identity_async(
        output_dir=str(tmp_path),
        is_test_instance=True,
        agent_name="ScopedChild",
        parent_did=parent_did,
        spawn_mandate=mandate,
    )

    db = await AsyncDatabase.sqlite(os.path.join(str(tmp_path), "kestrel_prime.db"))
    try:
        graph = AsyncGraphStore(db)
        out_edges = await graph.get_edges(creds.agent_did, direction="out")
        spawned = [e for e in out_edges if e.label == "spawned_by"]
        assert len(spawned) == 1
        constraints = spawned[0].properties["additional_constraints"]
        assert constraints["restricted_tools"] == [RESTRICTED_TOOL]
        assert constraints["behavioral_rules"] == [BEHAVIOR_RULE]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_reload_reconstructs_mandate_and_enforces(tmp_path):
    """The reload path reconstructs the mandate from the persisted edge and
    re-applies the restricted_tools hook — so enforcement survives restart, not
    just the in-process spawn."""
    parent_private, _ = generate_secp256k1_keypair()
    parent_did = "did:pkh:eip155:1:0xParentReload"

    mandate = SpawnMandate(
        parent_did=parent_did,
        purpose="scoped worker",
        ttl_seconds=1800,
        max_child_depth=0,
        additional_constraints={"restricted_tools": [RESTRICTED_TOOL]},
    )
    mandate = sign_mandate(mandate, parent_private)

    creds = await create_kestrel_identity_async(
        output_dir=str(tmp_path),
        is_test_instance=True,
        agent_name="ReloadedChild",
        parent_did=parent_did,
        spawn_mandate=mandate,
    )

    # Simulate a fresh load: reconstruct the mandate from the persisted edge
    # (the same helper KestrelAgent.initialize() uses on every boot path).
    storage = AsyncStorage(os.path.join(str(tmp_path), "kestrel_prime.db"))
    await storage.initialize()
    try:
        reconstructed = await read_spawn_mandate(storage, creds.agent_did)
    finally:
        await storage.close()
    assert reconstructed is not None
    assert reconstructed.additional_constraints["restricted_tools"] == [RESTRICTED_TOOL]

    # And it re-applies as a real block.
    child = SimpleNamespace(name="ReloadedChild", hooks_manager=HooksManager())
    register_restriction_hook(child.hooks_manager, reconstructed)
    out = await child.hooks_manager.execute_hooks(
        HookEvent.PRE_TOOL_USE,
        HookInput(
            session_id="t",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name=RESTRICTED_TOOL,
            tool_input={},
        ),
    )
    blocked = evaluate_blocking_decision(out)
    assert blocked is not None and blocked.decision == PermissionDecision.DENY


@pytest.mark.asyncio
async def test_read_spawn_mandate_none_for_root_agent(tmp_path):
    """A non-spawned (root) agent has no spawned_by edge ⇒ no mandate."""
    creds = await create_kestrel_identity_async(
        output_dir=str(tmp_path),
        is_test_instance=True,
        agent_name="RootAgent",
    )
    storage = AsyncStorage(os.path.join(str(tmp_path), "kestrel_prime.db"))
    await storage.initialize()
    try:
        assert await read_spawn_mandate(storage, creds.agent_did) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_initialize_reattaches_enforcement_on_any_boot_path(tmp_path):
    """End-to-end: a spawned child booted directly via KestrelAgent.initialize()
    (the shared path used by single-agent server + CLI, not just AgentManager)
    reattaches its spawn mandate and registers the restricted_tools hook."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService

    parent_private, _ = generate_secp256k1_keypair()
    parent_did = "did:pkh:eip155:1:0xParentBoot"
    mandate = sign_mandate(
        SpawnMandate(
            parent_did=parent_did,
            purpose="scoped worker",
            ttl_seconds=999,
            additional_constraints={"restricted_tools": [RESTRICTED_TOOL]},
        ),
        parent_private,
    )
    creds = await create_kestrel_identity_async(
        output_dir=str(tmp_path),
        is_test_instance=True,
        agent_name="BootChild",
        parent_did=parent_did,
        spawn_mandate=mandate,
    )

    agent = KestrelAgent(
        did=creds.agent_did,
        storage_path=os.path.join(str(tmp_path), "kestrel_prime.db"),
        llm_service=LLMService(),
    )
    await agent.initialize()

    hooks = agent.hooks_manager.get_hooks(HookEvent.PRE_TOOL_USE)
    assert any(isinstance(h, MandateRestrictionHook) for h in hooks)
    assert getattr(agent, "spawn_mandate", None) is not None
    assert agent.spawn_mandate.additional_constraints["restricted_tools"] == [
        RESTRICTED_TOOL
    ]


@pytest.mark.asyncio
async def test_restriction_hook_denies_restricted_allows_others():
    """The hook denies a restricted tool and allows everything else."""
    hook = MandateRestrictionHook([RESTRICTED_TOOL])

    denied = await hook.execute(
        HookInput(
            session_id="t",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name=RESTRICTED_TOOL,
            tool_input={},
        )
    )
    assert denied.permission_decision == PermissionDecision.DENY

    allowed = await hook.execute(
        HookInput(
            session_id="t",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="some_other_tool",
            tool_input={},
        )
    )
    assert allowed.permission_decision != PermissionDecision.DENY


@pytest.mark.asyncio
async def test_enforce_restricted_tools_blocks_via_hooks_manager():
    """register_restriction_hook installs a hook that the HooksManager
    evaluates as a real block for the restricted tool."""
    child = SimpleNamespace(name="ScopedChild", hooks_manager=HooksManager())
    mandate = SimpleNamespace(
        additional_constraints={"restricted_tools": [RESTRICTED_TOOL]}
    )

    register_restriction_hook(child.hooks_manager, mandate)

    # restricted tool → blocked
    out = await child.hooks_manager.execute_hooks(
        HookEvent.PRE_TOOL_USE,
        HookInput(
            session_id="t",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name=RESTRICTED_TOOL,
            tool_input={},
        ),
    )
    blocked = evaluate_blocking_decision(out)
    assert blocked is not None and blocked.decision == PermissionDecision.DENY

    # unrestricted tool → not blocked
    out2 = await child.hooks_manager.execute_hooks(
        HookEvent.PRE_TOOL_USE,
        HookInput(
            session_id="t",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="allowed_tool",
            tool_input={},
        ),
    )
    assert evaluate_blocking_decision(out2) is None


@pytest.mark.asyncio
async def test_enforce_restricted_tools_noop_without_restrictions():
    """No restricted_tools ⇒ no hook registered (nothing to enforce)."""
    child = SimpleNamespace(name="Plain", hooks_manager=HooksManager())
    mandate = SimpleNamespace(additional_constraints={"behavioral_rules": ["be nice"]})

    register_restriction_hook(child.hooks_manager, mandate)

    hooks = child.hooks_manager.get_hooks(HookEvent.PRE_TOOL_USE)
    assert not any(isinstance(h, MandateRestrictionHook) for h in hooks)
