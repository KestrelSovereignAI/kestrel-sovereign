"""Tests for FeatureForgeFeature — the feature that creates features (#2434).

Covers the four Definition-of-Done gates:
  1. Scaffold generator produces a loadable feature package.
  2. Iron Rule validator: widen-rejection and narrow-acceptance.
  3. Approval gate wired through the SecurityFeature approval queue; forged
     feature demonstrably inert pre-approval.
  4. Audit events emitted for forge/validate/register/approve/reject.
"""

import ast
import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.feature_forge.feature import FeatureForgeFeature
from kestrel_sovereign.features.feature_forge.iron_rule import (
    BASELINE_CAPABILITIES,
    validate_narrowing,
)
from kestrel_sovereign.features.feature_forge.scaffold import (
    SpecError,
    parse_spec,
    render_package,
    to_class_name,
    to_snake_case,
)
from kestrel_sovereign.features.feature_forge.store import (
    STATE_APPROVED,
    STATE_BLOCKED,
    STATE_DRAFT,
    STATE_PENDING,
    STATE_REJECTED,
    STATE_VALIDATED,
    ForgeStore,
)
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
from kestrel_sovereign.features.security.permissions import (
    PermissionLevel,
    PermissionStore,
)
from kestrel_sdk.tools.result import ToolResultStatus


def _spec(name="EvidenceLedger", permissions=("memory_read", "memory_write")):
    return {
        "name": name,
        "purpose": "Record claims with their verification state",
        "tools": [
            {
                "name": "record_claim",
                "description": "Record a claim and its verification state",
                "parameters": [
                    {"name": "claim", "type": "string", "required": True,
                     "description": "the claim text"},
                    {"name": "verified", "type": "boolean", "required": False,
                     "description": "whether independently verified"},
                ],
            }
        ],
        "permissions": list(permissions),
    }


# ---------------------------------------------------------------------------
# 2. Iron Rule validator
# ---------------------------------------------------------------------------

def test_iron_rule_narrow_accept():
    verdict = validate_narrowing(
        ["memory_read"], ["memory_read", "graph_read", "memory_write"]
    )
    assert verdict.valid is True
    assert verdict.widened == []
    assert verdict.unknown == []


def test_iron_rule_widen_reject():
    verdict = validate_narrowing(
        ["memory_read", "shell_execution"], ["memory_read", "graph_read"]
    )
    assert verdict.valid is False
    assert "shell_execution" in verdict.widened
    assert "narrow only" in verdict.reason.lower()


def test_iron_rule_unknown_capability_reject():
    verdict = validate_narrowing(["telepathy"], ["memory_read"])
    assert verdict.valid is False
    assert verdict.unknown == ["telepathy"]


def test_iron_rule_empty_request_is_narrowing():
    verdict = validate_narrowing([], ["memory_read"])
    assert verdict.valid is True


def test_iron_rule_normalizes_case_and_dupes():
    verdict = validate_narrowing(["Memory_Read", "memory_read"], ["memory_read"])
    assert verdict.valid is True
    assert verdict.requested == ["memory_read"]


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_class,expected_module",
    [
        ("EvidenceLedger", "EvidenceLedgerFeature", "evidence_ledger"),
        ("evidence_ledger", "EvidenceLedgerFeature", "evidence_ledger"),
        ("EvidenceLedgerFeature", "EvidenceLedgerFeature", "evidence_ledger"),
        ("My Cool Thing", "MyCoolThingFeature", "my_cool_thing"),
    ],
)
def test_name_normalization(raw, expected_class, expected_module):
    assert to_class_name(raw) == expected_class
    assert to_snake_case(raw) == expected_module


# ---------------------------------------------------------------------------
# 1. Scaffold generator produces a loadable feature package
# ---------------------------------------------------------------------------

def test_parse_spec_rejects_missing_name():
    with pytest.raises(SpecError):
        parse_spec({"tools": [{"name": "x"}]})


def test_parse_spec_rejects_no_tools():
    with pytest.raises(SpecError):
        parse_spec({"name": "X", "tools": []})


def test_parse_spec_rejects_bad_tool_identifier():
    with pytest.raises(SpecError):
        parse_spec({"name": "X", "tools": [{"name": "bad name!"}]})


def test_scaffold_renders_valid_python():
    parsed = parse_spec(_spec())
    files = render_package(parsed)
    assert set(files) == {"__init__.py", "feature.py", "test_evidence_ledger.py"}
    for content in files.values():
        ast.parse(content)  # raises SyntaxError on invalid Python


def test_scaffolded_package_is_loadable_and_tools_return_toolresult(tmp_path):
    """DoD #1: the forged package loads and its tools honor the ToolResult contract."""
    parsed = parse_spec(_spec())
    files = render_package(parsed)
    pkg = tmp_path / "evidence_ledger"
    pkg.mkdir()
    for rel, content in files.items():
        (pkg / rel).write_text(content)

    spec = importlib.util.spec_from_file_location(
        "forged_evidence_ledger.feature", pkg / "feature.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    feature_cls = module.EvidenceLedgerFeature
    from kestrel_sovereign.features.base import Feature as SovereignFeature
    assert issubclass(feature_cls, SovereignFeature)

    instance = feature_cls(MagicMock())
    tools = instance.get_tools()
    assert [t.name for t in tools] == ["record_claim"]

    result = asyncio.run(instance.record_claim(claim="CI passed"))
    assert result.status is ToolResultStatus.OK


# ---------------------------------------------------------------------------
# Store + inertness
# ---------------------------------------------------------------------------

def test_forge_root_is_outside_feature_discovery_path(tmp_path, monkeypatch):
    """DoD #3 (inertness): forged packages live outside the discovery path."""
    monkeypatch.setenv("KESTREL_FORGE_ROOT", str(tmp_path / "forged"))
    agent = MagicMock()
    agent.features = {}
    agent.storage_path = None
    feature = FeatureForgeFeature(agent)
    asyncio.run(feature.initialize())

    import kestrel_sovereign.features as features_pkg
    features_dir = Path(features_pkg.__file__).parent.resolve()
    assert features_dir not in feature.store.root.resolve().parents
    assert feature.store.root.resolve() != features_dir


def test_store_roundtrip(tmp_path):
    store = ForgeStore(tmp_path)
    parsed = parse_spec(_spec())
    from kestrel_sovereign.features.feature_forge.store import ForgeRecord

    record = ForgeRecord(
        name=parsed.module_name,
        display_name=parsed.name,
        class_name=parsed.class_name,
        state=STATE_DRAFT,
        spec=parsed.to_dict(),
    )
    store.save_scaffold(record, render_package(parsed))
    assert store.exists("evidence_ledger")
    loaded = store.get("evidence_ledger")
    assert loaded.class_name == "EvidenceLedgerFeature"
    assert loaded.state == STATE_DRAFT
    assert [r.name for r in store.list()] == ["evidence_ledger"]


# ---------------------------------------------------------------------------
# Feature pipeline + audit + approval
# ---------------------------------------------------------------------------

class _FakePermissionStore:
    """Captures audit rows written via log_decision."""

    def __init__(self):
        self.rows = []

    async def log_decision(self, **kwargs):
        self.rows.append(kwargs)


class _FakeSecurityFeature:
    def __init__(self, queue=None, permission_store=None):
        self.approval_queue = queue
        self.permission_store = permission_store


@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_FORGE_ROOT", str(tmp_path / "forged"))
    agent = MagicMock()
    agent.storage_path = None
    agent.is_test_instance = False
    agent.did = None
    perm_store = _FakePermissionStore()
    queue = ApprovalQueue()  # no permission store => queues without policy
    agent.features = {
        "SecurityFeature": _FakeSecurityFeature(queue, perm_store),
    }
    feature = FeatureForgeFeature(agent)
    return feature, agent, perm_store, queue


@pytest.mark.asyncio
async def test_forge_validate_register_happy_path(forge):
    feature, agent, perm_store, queue = forge
    await feature.initialize()

    r = await feature.forge_feature(_spec())
    assert r.status is ToolResultStatus.OK
    assert r.data["state"] == STATE_DRAFT

    r = await feature.forge_validate("evidence_ledger")
    assert r.status is ToolResultStatus.OK
    assert r.data["state"] == STATE_VALIDATED

    r = await feature.forge_register("evidence_ledger")
    assert r.status is ToolResultStatus.OK
    assert r.data["state"] == STATE_PENDING

    # DoD #3: inert & pending until the Sovereign approves.
    assert feature.store.get("evidence_ledger").state == STATE_PENDING
    # Let the background approval task register its pending request.
    for _ in range(10):
        await asyncio.sleep(0)
        if queue.pending_count == 1:
            break
    assert queue.pending_count == 1

    # DoD #4: audit events for forge/validate/register.
    decisions = {row["decision"] for row in perm_store.rows}
    assert {"drafted", "validated", "pending_approval"} <= decisions
    assert all(row["feature_name"] == "FeatureForgeFeature" for row in perm_store.rows)


@pytest.mark.asyncio
async def test_approval_advances_to_approved(forge):
    feature, agent, perm_store, queue = forge
    await feature.initialize()
    await feature.forge_feature(_spec())
    await feature.forge_validate("evidence_ledger")
    await feature.forge_register("evidence_ledger")

    # Let the background approval task register its pending request.
    request = await _wait_pending(queue)
    await queue.submit_decision(request.id, True, "once")
    await feature._approval_tasks.get("evidence_ledger", _done_future())

    assert feature.store.get("evidence_ledger").state == STATE_APPROVED
    assert any(row["decision"] == "approved" for row in perm_store.rows)


@pytest.mark.asyncio
async def test_denial_advances_to_rejected(forge):
    feature, agent, perm_store, queue = forge
    await feature.initialize()
    await feature.forge_feature(_spec())
    await feature.forge_validate("evidence_ledger")
    await feature.forge_register("evidence_ledger")

    await asyncio.sleep(0)
    request = queue.pending_requests[0]
    await queue.submit_decision(request.id, False, "user_denied")
    await feature._approval_tasks.get("evidence_ledger", _done_future())

    assert feature.store.get("evidence_ledger").state == STATE_REJECTED
    assert any(row["decision"] == "rejected" for row in perm_store.rows)


@pytest.mark.asyncio
async def test_global_auto_mode_does_not_bypass_forge_approval(tmp_path, monkeypatch):
    """#2434 P1: under global Auto, forge approval must NOT auto-approve.

    Uses a REAL ``PermissionStore`` (the fake in the ``forge`` fixture has no
    policy, so it can't reproduce this). With global auto-mode ON, an
    *unregistered* synthetic tool auto-approves — that is the hazard. The fix
    seeds ``FeatureForgeFeature.approve_forged_feature`` at ALWAYS_ASK, so the
    forged feature stays inert & pending, queued for an explicit Sovereign
    decision, instead of silently advancing to approved.
    """
    monkeypatch.setenv("KESTREL_FORGE_ROOT", str(tmp_path / "forged"))
    perm_store = PermissionStore(str(tmp_path / "security.db"))
    await perm_store.initialize()
    perm_store.set_global_auto_mode(True)

    agent = MagicMock()
    agent.storage_path = None
    agent.is_test_instance = False  # production: queues for a Sovereign
    agent.did = None

    # Hazard baseline: an unregistered synthetic tool DOES auto-approve.
    queue = ApprovalQueue(permission_store=perm_store, agent=agent)
    approved, scope = await queue.request_approval(
        feature_name="FeatureForgeFeature",
        tool_name="approve_forged_feature",
        tool_args={},
    )
    assert (approved, scope) == (True, "auto")

    # Apply the fix's seed, then the same gate must hold as ALWAYS_ASK.
    await perm_store.register_tool(
        feature_name="FeatureForgeFeature",
        tool_name="approve_forged_feature",
        default_level=PermissionLevel.ALWAYS_ASK,
        hardened=True,
    )

    agent.features = {
        "SecurityFeature": _FakeSecurityFeature(queue, perm_store),
    }
    feature = FeatureForgeFeature(agent)
    await feature.initialize()
    await feature.forge_feature(_spec())
    await feature.forge_validate("evidence_ledger")
    r = await feature.forge_register("evidence_ledger")
    assert r.data["state"] == STATE_PENDING

    # The background approval task must QUEUE (block on the Sovereign), never
    # auto-approve. Give it a few loop turns to reach request_approval.
    request = await _wait_pending(queue)
    assert request.tool_name == "approve_forged_feature"
    assert feature.store.get("evidence_ledger").state == STATE_PENDING


@pytest.mark.asyncio
async def test_no_approver_blocks_not_rejects(tmp_path, monkeypatch):
    """#2434 P2: a non-user-denial outcome must not be laundered into rejected.

    A headless agent (``is_test_instance``) with a real approval queue gets
    ``(False, "no_approver")`` — no one is attached to answer. That is NOT a
    user denial (#1542), so the forged feature parks in the recoverable
    ``blocked`` state and can be re-registered, rather than being permanently
    rejected. Re-registering once an approver exists must re-queue.
    """
    monkeypatch.setenv("KESTREL_FORGE_ROOT", str(tmp_path / "forged"))
    perm_store = _FakePermissionStore()
    agent = MagicMock()
    agent.storage_path = None
    agent.is_test_instance = True  # headless → no_approver, not a denial
    agent.did = None
    queue = ApprovalQueue(agent=agent)
    agent.features = {"SecurityFeature": _FakeSecurityFeature(queue, perm_store)}

    feature = FeatureForgeFeature(agent)
    await feature.initialize()
    await feature.forge_feature(_spec())
    await feature.forge_validate("evidence_ledger")
    await feature.forge_register("evidence_ledger")
    await feature._approval_tasks.get("evidence_ledger", _done_future())

    record = feature.store.get("evidence_ledger")
    assert record.state == STATE_BLOCKED
    assert record.state != STATE_REJECTED
    assert any(row["decision"] == "blocked" for row in perm_store.rows)

    # Recoverable: re-registering from blocked re-queues a fresh approval.
    agent.is_test_instance = False  # a Sovereign is now attached
    r = await feature.forge_register("evidence_ledger")
    assert r.status is ToolResultStatus.OK
    assert r.data["state"] == STATE_PENDING


@pytest.mark.asyncio
async def test_forge_validate_widen_rejected(forge):
    """DoD #2 through the tool: a spec requesting an unheld capability is rejected."""
    feature, agent, perm_store, queue = forge
    await feature.initialize()
    # shell_execution is privileged and no ComputeFeature is loaded → not held.
    await feature.forge_feature(_spec(name="Sneaky", permissions=["shell_execution"]))
    r = await feature.forge_validate("sneaky")
    assert r.status is ToolResultStatus.ERROR
    assert "narrow only" in (r.error or "").lower()
    # Must not advance out of draft.
    assert feature.store.get("sneaky").state == STATE_DRAFT
    assert any(row["decision"] == "rejected" for row in perm_store.rows)


@pytest.mark.asyncio
async def test_register_requires_validated(forge):
    feature, agent, perm_store, queue = forge
    await feature.initialize()
    await feature.forge_feature(_spec())
    r = await feature.forge_register("evidence_ledger")  # still draft
    assert r.status is ToolResultStatus.ERROR
    assert feature.store.get("evidence_ledger").state == STATE_DRAFT


@pytest.mark.asyncio
async def test_granted_capabilities_reflect_loaded_features(forge):
    feature, agent, perm_store, queue = forge
    await feature.initialize()
    granted = set(feature.granted_capabilities())
    assert BASELINE_CAPABILITIES <= granted
    assert "shell_execution" not in granted

    agent.features["ComputeFeature"] = MagicMock()
    granted2 = set(feature.granted_capabilities())
    assert "shell_execution" in granted2


@pytest.mark.asyncio
async def test_list_and_status(forge):
    feature, agent, perm_store, queue = forge
    await feature.initialize()
    await feature.forge_feature(_spec())
    listing = await feature.list_forged()
    assert listing.data["count"] == 1
    status = await feature.forge_status("EvidenceLedgerFeature")  # class-name lookup
    assert status.status is ToolResultStatus.OK
    assert status.data["state"] == STATE_DRAFT


def _done_future():
    fut = asyncio.get_event_loop().create_future()
    fut.set_result(None)
    return fut


async def _wait_pending(queue, expected=1, tries=50):
    for _ in range(tries):
        await asyncio.sleep(0)
        if queue.pending_count >= expected:
            return queue.pending_requests[0]
    raise AssertionError("approval request never queued")
