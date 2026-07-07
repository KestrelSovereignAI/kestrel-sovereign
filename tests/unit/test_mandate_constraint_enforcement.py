"""Runtime enforcement of SpawnMandate additional_constraints (#2137).

Covers the two halves of enforcement:
  1. Durable: the mandate's constraints are woven into the child's *anchored*
     constitution (so they reach the system prompt and integrity anchoring) and
     recorded on the delegation edge.
  2. Hard: a spawned child cannot actually invoke a `restricted_tools` entry —
     the MandateRestrictionHook denies it at PRE_TOOL_USE.
"""

import os
from types import SimpleNamespace

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookInput, PermissionDecision
from kestrel_sovereign.hooks import HooksManager, evaluate_blocking_decision
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate
from kestrel_sovereign.spawn.mandate_hook import MandateRestrictionHook
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
from kestrel_sovereign.inception_service import generate_secp256k1_keypair


BEHAVIOR_RULE = "Never contact external endpoints without human approval."
RESTRICTED_TOOL = "computer_use_shell"


@pytest.mark.asyncio
async def test_mandate_constraints_woven_into_anchored_constitution(tmp_path):
    """The child's anchored constitution carries the mandate's behavioral rules
    and restricted-tools list, and the delegation edge records the constraints."""
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

        # (1) constraints persisted on the spawned_by edge
        spawned = [e for e in out_edges if e.label == "spawned_by"]
        assert len(spawned) == 1
        assert spawned[0].properties["additional_constraints"]["restricted_tools"] == [
            RESTRICTED_TOOL
        ]

        # (2) anchored constitution (governed_by target) contains the constraints
        governed = [e for e in out_edges if e.label == "governed_by"]
        assert len(governed) == 1
        files = AsyncFileStore(db)
        constitution = (await files.retrieve_file(governed[0].target_id)).decode("utf-8")
        assert "SPAWN MANDATE CONSTRAINTS" in constitution
        assert BEHAVIOR_RULE in constitution
        assert RESTRICTED_TOOL in constitution
    finally:
        await db.close()


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
    """AgentManager._enforce_restricted_tools registers a hook that the
    HooksManager evaluates as a real block for the restricted tool."""
    child = SimpleNamespace(name="ScopedChild", hooks_manager=HooksManager())
    mandate = SimpleNamespace(
        additional_constraints={"restricted_tools": [RESTRICTED_TOOL]}
    )

    AgentManager._enforce_restricted_tools(child, mandate)

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

    AgentManager._enforce_restricted_tools(child, mandate)

    hooks = child.hooks_manager.get_hooks(HookEvent.PRE_TOOL_USE)
    assert not any(isinstance(h, MandateRestrictionHook) for h in hooks)
