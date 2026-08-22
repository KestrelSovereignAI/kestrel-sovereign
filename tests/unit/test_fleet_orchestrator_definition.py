"""The Fleet Orchestrator agent definition (#2321).

The Fleet Orchestrator dispatches coding work ONLY by starting
``fleet_coding_pipeline`` workflow runs; it never edits code, approves its own
consent gates. These tests assert the definition's scoped tool allowlist DENIES
write/edit tools, that its feature ceiling excludes code-editing features, and
that its constitution + mandate are shaped so the existing spawn-mandate
machinery (#2137 hard denial, #2226 feature ceiling, #2225 constitution
surfacing) enforces all three.
"""

import importlib
import json
from types import SimpleNamespace

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookInput, PermissionDecision
from kestrel_sovereign.hooks import HooksManager, evaluate_blocking_decision
from kestrel_sovereign.spawn.mandate_hook import MandateRestrictionHook
from kestrel_sovereign.spawn.mandate_reload import register_restriction_hook

from kestrel_sovereign.fleet import orchestrator as fo
from kestrel_sovereign.workflow_features import (
    FEATURE_ENTRY_POINT_GROUP,
    WORKFLOW_FEATURE_ENTRY_POINT_GROUP,
    WorkflowFeatureResolutionError,
)


WORKFLOW_PROVIDER_FEATURE = "ExampleCodingFeature"
WORKFLOW_PROVIDER_VALUE = "example_coding.feature:ExampleCodingFeature"
WORKFLOW_PROVIDER_DISTRIBUTION = "kestrel-feature-example-coding"


def _workflow_provider_entry_points():
    distribution = SimpleNamespace(name=WORKFLOW_PROVIDER_DISTRIBUTION)
    return [
        SimpleNamespace(
            group=WORKFLOW_FEATURE_ENTRY_POINT_GROUP,
            name=fo.FLEET_CODING_WORKFLOW_NAME,
            value=WORKFLOW_PROVIDER_VALUE,
            dist=distribution,
        ),
        SimpleNamespace(
            group=FEATURE_ENTRY_POINT_GROUP,
            name=WORKFLOW_PROVIDER_FEATURE,
            value=WORKFLOW_PROVIDER_VALUE,
            dist=distribution,
        ),
    ]


# ---------------------------------------------------------------------------
# Identity + attribution contract.
# ---------------------------------------------------------------------------


def test_friendly_name_and_slug():
    """The friendly name is the observability orchestrator stamp (#2302); the
    slug is the multi-agent routing key. Renaming either breaks attribution."""
    assert fo.FLEET_ORCHESTRATOR_NAME == "Fleet Orchestrator"
    assert fo.FLEET_ORCHESTRATOR_SLUG == "fleet-orchestrator"


# ---------------------------------------------------------------------------
# Feature ceiling — no code-editing / filesystem / shell features.
# ---------------------------------------------------------------------------


def test_feature_ceiling_has_the_dispatch_and_read_surface():
    for cls in (
        "WorkflowsFeature",
        "GitHubFeature",
        "SchedulerFeature",
        "ReflectionFeature",
    ):
        assert cls in fo.FEATURE_ALLOWLIST


def test_feature_ceiling_excludes_write_edit_features():
    """The structural 'no write/edit/file tools' guarantee: the features that
    provide fs_edit/fs_write/shell/write_script never load."""
    for cls in fo.FORBIDDEN_FEATURES:
        assert cls not in fo.FEATURE_ALLOWLIST
    assert "ComputeFeature" not in fo.FEATURE_ALLOWLIST
    assert "ComputerUseFeature" not in fo.FEATURE_ALLOWLIST


def test_mandatory_tool_floor_is_derived_from_real_feature_classes():
    from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURE_MODULES

    expected = set()
    for class_name, module_path in MANDATORY_FEATURE_MODULES.items():
        feature_class = getattr(importlib.import_module(module_path), class_name)
        registered = fo.registered_tool_names(feature_class)
        assert registered, f"{class_name} must expose its mandatory tool surface"
        expected.update(registered)

    assert fo.MANDATORY_TOOL_ALLOWLIST == frozenset(expected)
    assert expected <= fo.TOOL_ALLOWLIST


# ---------------------------------------------------------------------------
# The required test: the allowlist denies write/edit tools and direct writes.
# ---------------------------------------------------------------------------


def test_allowlist_denies_write_edit_and_mutation_tools():
    # Write / edit / file / shell tools are denied.
    for tool in ("fs_edit", "fs_write", "shell", "write_script", "execute_command", "apply_patch"):
        assert fo.is_tool_denied(tool), f"{tool} must be denied"
        assert not fo.is_tool_allowed(tool)

    # GitHub write tools are denied (real kestrel_feature_github @tool names).
    for tool in (
        "create_github_issue",
        "create_github_pull_request",
        "merge_github_pull_request",
        "add_github_issue_comment",
        "add_github_label",
        "remove_github_label",
        "close_github_issue",
        "reopen_github_issue",
    ):
        assert fo.is_tool_denied(tool)

    # Scheduler mutation tools are denied (P1 #2321).
    for tool in (
        "schedule_add",
        "schedule_add_deadline",
        "schedule_remove",
        "schedule_pause",
        "schedule_resume",
        "schedule_update",
        "schedule_record_outcome",
    ):
        assert fo.is_tool_denied(tool), f"{tool} must be denied"

    # Workflow-mutation tools are denied (only run/load/read are scoped in).
    for tool in ("workflow_cancel", "workflow_force_abort", "workflow_define"):
        assert fo.is_tool_denied(tool)

    # Mandatory Features load regardless of the feature ceiling, but these
    # concrete core tools would bypass Fleet's no-self-approval / workflow-only
    # dispatch boundary and therefore remain explicitly denied.
    assert fo.MANDATORY_MUTATION_TOOLS == frozenset(
        {
            "approve",
            "deny",
            "export_identity",
            "import_identity",
            "send_a2a_task",
            "set_permission",
        }
    )
    assert fo.MANDATORY_MUTATION_TOOLS <= fo.MANDATORY_TOOL_ALLOWLIST
    for tool in fo.MANDATORY_MUTATION_TOOLS:
        assert fo.is_tool_denied(tool)
        assert not fo.is_tool_allowed(tool)


def test_allowed_dispatch_and_read_tools_are_not_denied():
    # workflow_run is the ONLY dispatch surface — it must be allowed.
    assert fo.is_tool_allowed("workflow_run")
    assert not fo.is_tool_denied("workflow_run")
    # Read tools across the surfaces are allowed (real feature @tool names).
    for tool in (
        "ask_agent",
        "constitution",
        "lifecycle_status",
        "list_permissions",
        "pending_approvals",
        "send_a2a_message",
        "send_a2a_question",
        "verify_identity",
        "wait",
        "workflow_status",
        "list_github_issues",
        "get_github_repo_info",
    ):
        assert fo.is_tool_allowed(tool)
        assert not fo.is_tool_denied(tool)


def test_scheduler_read_status_tools_remain_available():
    """The Signals-view read/status scheduler tools are NOT denied — only the
    schedule-mutating tools are (P1 #2321)."""
    for tool in ("schedule_list", "schedule_history", "schedule_engagement"):
        assert not fo.is_tool_denied(tool)


def test_explicit_mandatory_mutations_are_the_only_allow_deny_overlap():
    # The full mandatory floor is derived into TOOL_ALLOWLIST, then this
    # explicit core mutation subset wins in is_tool_allowed and the runtime
    # hook. No other positive classification may overlap a denial.
    assert fo.TOOL_ALLOWLIST & fo.RESTRICTED_TOOLS == fo.MANDATORY_MUTATION_TOOLS


# ---------------------------------------------------------------------------
# Derived-not-hand-maintained deny-list audit (P2 #2321).
#
# For EVERY feature in the ceiling, enumerate the tool names it actually
# registers (the same @tool `_tool_schema["name"]` the runtime loads) and assert
# each is classified — allowed (read) OR denied (mutation) OR arg-restricted.
# A tool that is neither fails the audit, so a future feature version adding a
# tool can never silently widen the agent's reach: the test breaks until someone
# classifies it. In-tree features are always importable; external feature
# packages are skipped when not installed (covered wherever they ARE installed).
# ---------------------------------------------------------------------------


def _load_feature_cls(feature_name):
    module_path, cls_name = fo.FEATURE_TOOL_MODULES[feature_name]
    module = pytest.importorskip(
        module_path, reason=f"{feature_name} not installed in this environment"
    )
    return getattr(module, cls_name)


@pytest.mark.parametrize("feature_name", sorted(fo.FEATURE_ALLOWLIST))
def test_every_ceiling_feature_tool_is_classified(feature_name):
    """Fail-closed audit: no registered tool of a ceiling feature may be left
    unclassified. This is what makes the deny-list derived, not hand-maintained
    against intent — a new mutating tool in a feature upgrade trips this test."""
    cls = _load_feature_cls(feature_name)
    registered = fo.registered_tool_names(cls)
    assert registered, f"{feature_name} registered no @tool names — enumeration broke"
    unclassified = fo.unclassified_tool_names(cls)
    assert not unclassified, (
        f"{feature_name} exposes tool(s) that are neither allowed nor denied: "
        f"{sorted(unclassified)}. Classify each as a read (TOOL_ALLOWLIST) or a "
        f"mutation (RESTRICTED_TOOLS)."
    )


def test_memory_destructive_tools_are_denied():
    """Regression for the P2 audit gap: MemoryFeature bundles delete/purge tools
    the orchestrator must never invoke directly. They ARE in the deny-list."""
    for tool in (
        "delete_conversation",
        "delete_message_by_id",
        "delete_messages",
        "purge_conversation",
        "purge_message_by_id",
        "restore_conversation",
        "restore_message_by_id",
        "mark_superseded",
        "update_action_item",
        "confirm_person_match",
        "memory_consolidate",
    ):
        assert fo.is_tool_denied(tool), f"{tool} must be denied"
    # Reads stay available.
    for tool in ("search_memory", "recall_recent", "memory_status", "get_episodes"):
        assert fo.is_tool_allowed(tool)
        assert not fo.is_tool_denied(tool)


def test_reflection_mutation_tools_are_denied():
    """ReflectionFeature is in the ceiling for the `reflect` triage sweep + reads;
    its self-model/training mutation tools are hard-denied."""
    for tool in (
        "update_self_model",
        "propose_improvement",
        "create_improvement_ticket",
        "training_cycle",
    ):
        assert fo.is_tool_denied(tool), f"{tool} must be denied"
    for tool in ("reflect", "get_behavior_rules", "get_insights", "get_self_model"):
        assert fo.is_tool_allowed(tool)
        assert not fo.is_tool_denied(tool)


# ---------------------------------------------------------------------------
# Hard runtime enforcement via the EXISTING MandateRestrictionHook (#2137).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restriction_hook_hard_denies_dispatch_and_write_tools():
    hook = fo.build_restriction_hook()
    assert isinstance(hook, MandateRestrictionHook)
    assert HookEvent.PRE_TOOL_USE in hook.events

    for tool in (
        "fs_write",
        "shell",
        "create_github_pull_request",
        "workflow_cancel",
        "schedule_add",
        "schedule_add_deadline",
    ):
        out = await hook.execute(
            HookInput(
                session_id="t",
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name=tool,
                tool_input={},
            )
        )
        assert out.permission_decision == PermissionDecision.DENY, f"{tool} not denied"

    # Loading the dynamically resolved workflow-owner Feature must not expose
    # any of its direct tools unless core's generic Fleet policy classified the
    # tool in the positive ceiling.
    out = await hook.execute(
        HookInput(
            session_id="t",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="provider_direct_dispatch",
            tool_input={},
        )
    )
    assert out.permission_decision == PermissionDecision.DENY

    for tool, tool_input in (
        ("workflow_run", {"name": "fleet_coding_pipeline"}),
        ("list_github_issues", {}),
    ):
        out = await hook.execute(
            HookInput(
                session_id="t",
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name=tool,
                tool_input=tool_input,
            )
        )
        assert out.permission_decision != PermissionDecision.DENY, f"{tool} wrongly denied"


@pytest.mark.asyncio
async def test_build_and_reload_hooks_admit_mandatory_floor_only_within_ceiling():
    direct_hook = fo.build_restriction_hook()
    manager = HooksManager()
    register_restriction_hook(
        manager,
        SimpleNamespace(additional_constraints=fo.additional_constraints()),
    )

    permitted_floor = fo.MANDATORY_TOOL_ALLOWLIST - fo.MANDATORY_MUTATION_TOOLS
    for tool_name in sorted(permitted_floor):
        tool_input = {}
        direct = await direct_hook.execute(
            HookInput(
                session_id="t",
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name=tool_name,
                tool_input=tool_input,
            )
        )
        assert direct.permission_decision != PermissionDecision.DENY, tool_name

        reloaded = await manager.execute_hooks(
            HookEvent.PRE_TOOL_USE,
            HookInput(
                session_id="t",
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name=tool_name,
                tool_input=tool_input,
            ),
        )
        assert evaluate_blocking_decision(reloaded) is None, tool_name

    for tool_name in sorted(fo.MANDATORY_MUTATION_TOOLS):
        direct = await direct_hook.execute(
            HookInput(
                session_id="t",
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name=tool_name,
                tool_input={},
            )
        )
        assert direct.permission_decision == PermissionDecision.DENY, tool_name
        reloaded = await manager.execute_hooks(
            HookEvent.PRE_TOOL_USE,
            HookInput(
                session_id="t",
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name=tool_name,
                tool_input={},
            ),
        )
        assert evaluate_blocking_decision(reloaded) is not None, tool_name

    denied = await direct_hook.execute(
        HookInput(
            session_id="t",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="external_provider_direct_tool",
            tool_input={},
        )
    )
    assert denied.permission_decision == PermissionDecision.DENY

    denied = await manager.execute_hooks(
        HookEvent.PRE_TOOL_USE,
        HookInput(
            session_id="t",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="external_provider_direct_tool",
            tool_input={},
        ),
    )
    assert evaluate_blocking_decision(denied) is not None


@pytest.mark.asyncio
async def test_restriction_hook_scopes_workflow_run_to_fleet_coding_pipeline():
    """P1 (#2321): workflow_run is allowed ONLY for fleet_coding_pipeline. Any
    other workflow name (stalled_work_rescue, an arbitrary loaded definition, or
    a missing name) is hard-denied at the argument level — the tool-name
    allowlist alone can't express this."""
    hook = fo.build_restriction_hook()

    async def run(name):
        tool_input = {} if name is None else {"name": name}
        return await hook.execute(
            HookInput(
                session_id="t",
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name="workflow_run",
                tool_input=tool_input,
            )
        )

    # The single permitted workflow passes.
    ok = await run("fleet_coding_pipeline")
    assert ok.permission_decision != PermissionDecision.DENY

    # Every other name — including a builtin — is denied.
    for name in ("stalled_work_rescue", "some_other_workflow", "", None):
        out = await run(name)
        assert out.permission_decision == PermissionDecision.DENY, (
            f"workflow_run({name!r}) must be denied"
        )

    # The pure-Python mirror agrees with the hook.
    assert fo.is_tool_call_allowed("workflow_run", {"name": "fleet_coding_pipeline"})
    assert not fo.is_tool_call_allowed("workflow_run", {"name": "stalled_work_rescue"})
    assert not fo.is_tool_call_allowed("workflow_run", {})


@pytest.mark.asyncio
async def test_reload_path_registers_the_deny_hook_from_constraints():
    """The mandate the definition builds carries restricted_tools +
    restricted_tool_args, so the real reload seam (register_restriction_hook)
    installs a hook that blocks writes AND arg-scopes workflow_run via
    the HooksManager — the same path KestrelAgent.initialize uses."""
    manager = HooksManager()
    count = register_restriction_hook(
        manager, type("M", (), {"additional_constraints": fo.additional_constraints()})()
    )
    assert count == len(fo.RESTRICTED_TOOLS) + len(fo.RESTRICTED_TOOL_ARGS) + 1

    out = await manager.execute_hooks(
        HookEvent.PRE_TOOL_USE,
        HookInput(
            session_id="t",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="fs_write",
            tool_input={},
        ),
    )
    blocked = evaluate_blocking_decision(out)
    assert blocked is not None and blocked.decision == PermissionDecision.DENY

    # The arg-scope survives the reload reconstruction too: a foreign workflow
    # name is denied through the same HooksManager path.
    out = await manager.execute_hooks(
        HookEvent.PRE_TOOL_USE,
        HookInput(
            session_id="t",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="workflow_run",
            tool_input={"name": "stalled_work_rescue"},
        ),
    )
    blocked = evaluate_blocking_decision(out)
    assert blocked is not None and blocked.decision == PermissionDecision.DENY


# ---------------------------------------------------------------------------
# Constitution — the four required rules, all narrowing.
# ---------------------------------------------------------------------------


def test_constitution_encodes_required_rules():
    text = fo.constitution_text(entry_points=_workflow_provider_entry_points())
    # Dispatches ONLY via fleet_coding_pipeline workflow runs.
    assert "fleet_coding_pipeline" in text
    assert "workflow_run" in text
    # Never approves its own consent gates (approvals belong to the Falconer).
    assert "consent" in text.lower()
    assert "Falconer" in text
    # Escalates uncertainty rather than guessing.
    assert "scalat" in text or "escalate" in text.lower()
    # The restricted write tools are surfaced as not-available.
    assert "create_github_issue" in text
    # The workflow_run argument scope is surfaced too.
    assert "workflow_run" in text
    assert "fleet_coding_pipeline" in text


def test_additional_constraints_only_narrow_and_validate():
    """behavioral_rules + restricted_tools are narrowing constraints the scoped
    constitution accepts; the feature ceiling is a subset of the parent's."""
    parent_features = set(fo.FEATURE_ALLOWLIST) | {
        WORKFLOW_PROVIDER_FEATURE,
        "ComputeFeature",
        "ComputerUseFeature",
    }
    scoped = fo.build_scoped_constitution(
        base_constitution="BASE",
        parent_features=parent_features,
        entry_points=_workflow_provider_entry_points(),
    )
    ok, msg = scoped.validate_constraints()
    assert ok, msg

    constraints = fo.additional_constraints()
    assert constraints["allowed_tools"] == sorted(fo.TOOL_ALLOWLIST)
    assert isinstance(constraints["behavioral_rules"], list)
    assert isinstance(constraints["restricted_tools"], list)
    # The argument-level narrowing is carried and scopes workflow_run.
    assert isinstance(constraints["restricted_tool_args"], dict)
    assert constraints["restricted_tool_args"]["workflow_run"]["name"] == [
        "fleet_coding_pipeline"
    ]
    # No capability-granting keys.
    for forbidden in ("grant_features", "override_constitution", "remove_restrictions"):
        assert forbidden not in constraints


def test_feature_ceiling_wider_than_parent_is_rejected():
    """Sanity: a ceiling naming a feature the parent lacks fails validation —
    proves the subset check is live for this definition's shape."""
    scoped = fo.build_scoped_constitution(
        base_constitution="BASE",
        parent_features={"WorkflowsFeature"},
        entry_points=_workflow_provider_entry_points(),
    )
    ok, _ = scoped.validate_constraints()
    assert not ok


# ---------------------------------------------------------------------------
# Spawn mandate + host config materialization.
# ---------------------------------------------------------------------------


def test_spawn_mandate_carries_ceiling_and_constraints():
    mandate = fo.build_spawn_mandate(
        "did:pkh:eip155:1:0xSovereign",
        entry_points=_workflow_provider_entry_points(),
    )
    assert mandate.parent_did == "did:pkh:eip155:1:0xSovereign"
    assert mandate.purpose == "Fleet Orchestrator"
    assert set(mandate.features_allowed) == (
        set(fo.FEATURE_ALLOWLIST) | {WORKFLOW_PROVIDER_FEATURE}
    )
    assert mandate.additional_constraints["restricted_tools"] == sorted(fo.RESTRICTED_TOOLS)
    assert mandate.additional_constraints["behavioral_rules"]
    assert (
        mandate.additional_constraints["restricted_tool_args"]["workflow_run"]["name"]
        == ["fleet_coding_pipeline"]
    )
    # Long-lived (persistent orchestrator), not an ephemeral task child.
    assert mandate.ttl_seconds >= 24 * 3600


def test_local_agent_config_stamps_feature_ceiling():
    cfg = fo.build_local_agent_config(
        "agent_data/fleet-orchestrator",
        8804,
        entry_points=_workflow_provider_entry_points(),
    )
    assert cfg.port == 8804
    assert cfg.autostart is True
    assert set(cfg.features) == set(fo.FEATURE_ALLOWLIST) | {
        WORKFLOW_PROVIDER_FEATURE
    }


def test_actual_fleet_ceilings_include_resolved_workflow_provider_feature():
    """Every persisted Fleet definition carries its workflow capability owner."""

    entry_points = _workflow_provider_entry_points()
    expected = set(fo.FEATURE_ALLOWLIST) | {WORKFLOW_PROVIDER_FEATURE}
    mandate = fo.build_spawn_mandate(
        "did:pkh:eip155:1:0xSovereign",
        entry_points=entry_points,
    )
    scoped = fo.build_scoped_constitution(
        parent_features=expected,
        entry_points=entry_points,
    )
    config = fo.build_local_agent_config(
        "agent_data/fleet-orchestrator",
        8804,
        entry_points=entry_points,
    )

    assert set(mandate.features_allowed) == expected
    assert set(scoped.features_allowed) == expected
    assert set(config.features) == expected


@pytest.mark.parametrize(
    "builder",
    [
        lambda: fo.build_spawn_mandate(
            "did:pkh:eip155:1:0xSovereign", entry_points=[]
        ),
        lambda: fo.build_scoped_constitution(entry_points=[]),
        lambda: fo.build_local_agent_config(
            "agent_data/fleet-orchestrator", 8804, entry_points=[]
        ),
    ],
)
def test_persisted_ceiling_builders_fail_closed_without_workflow_provider(builder):
    with pytest.raises(WorkflowFeatureResolutionError, match="no installed provider"):
        builder()


# ---------------------------------------------------------------------------
# Scheduled reflection task → /agent/reflection/status (claws Signals tab).
# ---------------------------------------------------------------------------


def test_reflection_schedule_uses_reflect_task():
    assert fo.REFLECTION_SCHEDULE, "must schedule at least one triage sweep"
    for cron_expr, task_name, args_json in fo.REFLECTION_SCHEDULE:
        # /agent/reflection/status surfaces task_name in ("reflect", "training_cycle").
        assert task_name == "reflect"
        assert cron_expr and len(cron_expr.split()) == 5
        # args_json must be a JSON object (the shape schedule_add consumes).
        assert isinstance(json.loads(args_json), dict)
