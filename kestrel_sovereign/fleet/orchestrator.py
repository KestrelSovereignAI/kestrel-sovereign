"""The **Fleet Orchestrator** agent definition (#2321).

Sovereign-side half of ``KestrelSovereignAI/kestrel-claws#29``: define a
governed agent (friendly name ``Fleet Orchestrator``, routing slug
``fleet-orchestrator``) on the multi-agent host using the *existing*
agent-definition / constitution / tool-restriction mechanisms so it appears in
``/api/agents`` (and thus the kestrel-claws Fleet tab) with **zero** dashboard
changes and **zero** core changes.

Why a governed child, not a bare host agent
===========================================
The Fleet Orchestrator needs three narrowings that the coarse ``multi_agent.toml``
``features`` allowlist alone cannot express:

1. **A feature ceiling** — it may load only the workflows / GitHub / scheduler /
   reflection / memory features, never a coding, file, shell, or computer-use
   feature. This is the
   ``features`` allowlist (feature-class granularity) — see
   :data:`FEATURE_ALLOWLIST`.
2. **An intra-feature tool deny-list** — the ceiling features bundle read tools
   *and* mutating tools in one class each: ``WorkflowsFeature`` mixes
   ``workflow_run`` with mutating tools
   (``workflow_cancel`` …); ``GitHubFeature`` mixes reads with issue/PR writes;
   ``SchedulerFeature`` mixes status reads with cron-mutating tools
   (``schedule_add`` …); ``MemoryFeature`` mixes recall/search reads with
   destructive delete/purge/restore tools; ``ReflectionFeature`` mixes the
   ``reflect`` triage sweep + read tools with self-model/training mutations. A
   feature-class allowlist cannot keep the reads while denying the mutations, so
   the split is expressed as a tool-level deny-list **derived from the tool names
   those features actually load** — see :data:`RESTRICTED_TOOLS`. The derivation
   is audited: :func:`unclassified_tool_names` reconciles every ceiling feature's
   registered @tool names (via :data:`FEATURE_TOOL_MODULES`) against the
   allow/deny classification, and the unit tests fail closed on any tool a future
   feature version adds that is neither allowed nor denied.
2b. **An intra-tool argument deny-list** — ``workflow_run`` must be *kept* (it is
   the dispatch surface) but scoped to a single workflow. A tool-name allowlist
   cannot express "``workflow_run`` but only for ``fleet_coding_pipeline``", so
   that narrowing is expressed as an argument allowlist — see
   :data:`RESTRICTED_TOOL_ARGS` — and enforced at PRE_TOOL_USE by the same hook
   (#2321).
3. **Governing behavioral rules** — the constitution encoded below.

All are exactly what a signed :class:`~kestrel_sovereign.spawn.mandate.SpawnMandate`
carries: ``features_allowed`` (the ceiling, enforced at feature discovery —
#2226), ``additional_constraints.restricted_tools`` /
``additional_constraints.restricted_tool_args`` (hard-denied at
``PRE_TOOL_USE`` by :class:`~kestrel_sovereign.spawn.mandate_hook.MandateRestrictionHook`
— #2137, #2321), and ``additional_constraints.behavioral_rules`` (surfaced into
the governing constitution — #2225). Inceptioning the agent as a child of the
Sovereign records these on its ``spawned_by`` edge, and **every** boot path
(host restart, single-agent server, CLI) re-applies them via
``read_spawn_mandate`` / ``read_spawn_features_allowed`` in
``KestrelAgent.initialize()``. Nothing new is enforced; this module only
*declares* the constraints and provides builders that hand them to the existing
machinery.

Materializing the agent (operator step)
=======================================
This module is the declaration + unit-testable builders. To bring the agent
onto a host:

1. Incept its data directory as a child of the Sovereign::

       from kestrel_sovereign.fleet import (
           FLEET_ORCHESTRATOR_NAME, build_spawn_mandate,
       )
       await create_kestrel_identity_async(
           output_dir="agent_data/fleet-orchestrator",
           agent_name=FLEET_ORCHESTRATOR_NAME,          # → /api/agents friendly name
           parent_did=sovereign_did,
           spawn_mandate=build_spawn_mandate(sovereign_did),
       )

2. Register it in ``multi_agent.toml`` (host-local, gitignored live config)::

       [agents.fleet-orchestrator]
       data_dir = "agent_data/fleet-orchestrator"
       port = 8804
       # features intentionally omitted here: the feature ceiling is enforced
       # from the spawned_by edge (build_spawn_mandate) on every boot path.

   :func:`build_local_agent_config` produces this entry programmatically and
   also stamps the ``features`` ceiling for defense in depth.

3. Seed its periodic triage sweep (:data:`REFLECTION_SCHEDULE`) via
   ``schedule_add`` so it surfaces on ``/agent/reflection/status`` (the
   kestrel-claws Signals tab).

Attribution contract (#2302): the friendly name ``Fleet Orchestrator`` is the
orchestrator identity carried by every ``fleet_coding_pipeline`` run. Do not
rename the agent.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kestrel_sovereign.workflow_features import resolve_workflow_feature

# Friendly name (rendered in /api/agents and stamped as the observability
# orchestrator — #2302) and the multi-agent routing slug.
FLEET_ORCHESTRATOR_NAME = "Fleet Orchestrator"
FLEET_ORCHESTRATOR_SLUG = "fleet-orchestrator"

# The consent scope its dispatches gate on (mirrors fleet_coding_pipeline).
CONSENT_SCOPE = "fleet_coding_pipeline_dispatch"
FLEET_CODING_WORKFLOW_NAME = "fleet_coding_pipeline"


def registered_tool_names(feature_cls: Any) -> frozenset[str]:
    """Enumerate the ``@tool`` names a Feature class actually registers."""

    names = set()
    for attr in dir(feature_cls):
        try:
            member = getattr(feature_cls, attr)
        except Exception:  # noqa: BLE001 — skip descriptors that error on access
            continue
        schema = getattr(member, "_tool_schema", None)
        if isinstance(schema, dict) and schema.get("name"):
            names.add(schema["name"])
    return frozenset(names)


def mandatory_feature_tool_names() -> frozenset[str]:
    """Derive the tool floor provided by every mandatory core Feature.

    Feature ceilings never suppress the mandatory sovereignty foundation, so a
    positive tool ceiling must not suppress it either. The canonical class to
    module mapping is shared with feature discovery; adding a new mandatory
    Feature automatically brings its registered tools into this floor.
    """

    from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURE_MODULES

    names: set[str] = set()
    for class_name, module_path in MANDATORY_FEATURE_MODULES.items():
        module = importlib.import_module(module_path)
        feature_class = getattr(module, class_name)
        names.update(registered_tool_names(feature_class))
    return frozenset(names)


# ---------------------------------------------------------------------------
# Feature ceiling (feature-class granularity).
# ---------------------------------------------------------------------------
# The feature classes the orchestrator may load. Mandatory features
# (IdentityFeature, SecurityFeature, PeersFeature, ConstitutionFeature,
# WaitFeature) always load regardless and are intentionally omitted. NOTE the
# absence of any code-editing / filesystem / shell feature (ComputeFeature,
# ComputerUseFeature): those never load, so their write/edit/file tools
# (fs_edit, fs_write, shell, write_script, …) are unavailable by construction.
FEATURE_ALLOWLIST = frozenset(
    {
        "WorkflowsFeature",       # workflow_run + read tools (the dispatch surface)
        "GitHubFeature",          # GitHub read (list_issues / list_prs / get_repo_info)
        "SchedulerFeature",       # schedule_add + /agent/reflection/status surface
        "ReflectionFeature",      # the `reflect` triage-sweep task
        "MemoryFeature",          # reflection rides the sleep/consolidation cycle
    }
)


def effective_feature_allowlist(
    *, entry_points: Optional[Iterable[Any]] = None
) -> frozenset[str]:
    """Return the persisted ceiling including the installed workflow owner.

    The coding-workflow provider is an independently installed feature.  Its
    class name is resolved from entry-point metadata rather than named in core,
    so extracting or replacing that provider cannot leave the orchestrator with
    a persisted ceiling that excludes the feature needed to run its only
    permitted workflow.  Missing, duplicate, or cross-distribution claims fail
    closed in :func:`resolve_workflow_feature`.
    """

    workflow_provider = resolve_workflow_feature(
        FLEET_CODING_WORKFLOW_NAME,
        entry_points=entry_points,
    )
    return frozenset((*FEATURE_ALLOWLIST, workflow_provider))

# Where each ceiling feature class is imported from — the source of truth for the
# derived deny-list audit (test_fleet_orchestrator_definition). Every ceiling
# feature's registered @tool names must be reconciled against the allow/deny
# classification here: a name that is neither allowed nor denied fails the audit,
# so a future feature version adding a tool can never silently widen the agent's
# reach. In-tree features are always importable; external feature packages are
# skipped by the audit when not installed (they are covered wherever installed).
FEATURE_TOOL_MODULES: Dict[str, Tuple[str, str]] = {
    "WorkflowsFeature": ("kestrel_feature_workflows", "WorkflowsFeature"),
    "GitHubFeature": ("kestrel_feature_github", "GitHubFeature"),
    "SchedulerFeature": (
        "kestrel_sovereign.features.scheduler.feature",
        "SchedulerFeature",
    ),
    "ReflectionFeature": ("kestrel_feature_reflection", "ReflectionFeature"),
    "MemoryFeature": ("kestrel_sovereign.features.memory.feature", "MemoryFeature"),
}


# Feature classes that MUST NOT be in the ceiling — asserted by the tests as the
# structural "no write/edit/file tools" guarantee (their tools never load).
FORBIDDEN_FEATURES = frozenset(
    {
        "ComputeFeature",
        "ComputerUseFeature",
    }
)


# ---------------------------------------------------------------------------
# Positive tool allowlist (the tools the orchestrator is scoped to).
# ---------------------------------------------------------------------------
# Workflows: run + load the builtin + read/observe. workflow_run is the ONLY way
# the orchestrator commissions work (by starting fleet_coding_pipeline runs).
WORKFLOW_TOOLS = frozenset(
    {
        "workflow_run",
        "workflow_status",
        "workflow_list_runs",
        "workflow_history",
        "workflow_list_definitions",
        "workflow_list_builtin",
        "workflow_load_builtin",
    }
)

# GitHub READ tools (verified against kestrel_feature_github/feature.py @tool
# names — the GitHubFeature class the ceiling loads). The write tools
# (create_github_issue / merge_github_pull_request / …) are denied below.
GITHUB_READ_TOOLS = frozenset(
    {
        "read_github_file",
        "list_github_files",
        "search_github_code",
        "get_code_definition",
        "list_code_definitions",
        "get_self_repo_info",
        "get_github_repo_info",
        "list_source_components",
        "get_component_source",
        "list_github_issues",
        "get_github_issue",
        "get_github_issue_comments",
        "scan_stale_work",
    }
)

# Scheduler READ / status tools (verified against features/scheduler/feature.py
# @tool names). SchedulerFeature is in the ceiling for these (they feed the
# kestrel-claws Signals tab); its cron-mutating tools are denied below.
SCHEDULER_READ_TOOLS = frozenset(
    {
        "schedule_list",
        "schedule_history",
        "schedule_engagement",
    }
)

# Memory READ / recall tools (verified against features/memory/feature.py @tool
# names). MemoryFeature is in the ceiling because the reflection sweep rides the
# sleep/consolidation cycle; the orchestrator itself only reads memory for triage
# context. Its delete/purge/restore/update tools are denied below.
MEMORY_READ_TOOLS = frozenset(
    {
        "search_memory",
        "search_documents",
        "search_case_law",
        "recall_recent",
        "recall_interactions",
        "recall_decisions",
        "recall_action_items",
        "recall_emotional",
        "get_episodes",
        "list_conversations",
        "list_trashed_messages",
        "memory_status",
    }
)

# Reflection READ tools + the `reflect` triage sweep (verified against
# kestrel_feature_reflection @tool names). `reflect` is the orchestrator's core
# triage loop (see REFLECTION_SCHEDULE); the self-model/improvement/training
# mutation tools are denied below.
REFLECTION_READ_TOOLS = frozenset(
    {
        "reflect",
        "get_behavior_rules",
        "get_insights",
        "get_self_model",
    }
)

MANDATORY_TOOL_ALLOWLIST = mandatory_feature_tool_names()

# Mandatory Features always load, but a handful of their tools would violate
# the Fleet Orchestrator's narrower behavioral boundary. Identity import/export
# mutates or publishes sovereign state; Security decisions change permission or
# approval state; send_a2a_task is a second direct work-dispatch lane. Keep the
# read, wait, and escalation surfaces while explicitly denying these mutations.
# These names are stable core tool contracts, not extension-provider coupling.
MANDATORY_MUTATION_TOOLS = frozenset(
    {
        "approve",
        "deny",
        "export_identity",
        "import_identity",
        "send_a2a_task",
        "set_permission",
    }
)

TOOL_ALLOWLIST = (
    WORKFLOW_TOOLS
    | GITHUB_READ_TOOLS
    | SCHEDULER_READ_TOOLS
    | MEMORY_READ_TOOLS
    | REFLECTION_READ_TOOLS
    | MANDATORY_TOOL_ALLOWLIST
)


# ---------------------------------------------------------------------------
# Tool deny-list (hard PRE_TOOL_USE denial via MandateRestrictionHook, #2137).
# ---------------------------------------------------------------------------
# Workflow mutation tools beyond run/load/read — the orchestrator only *starts*
# and *observes* runs; it does not cancel/abort/pause/resume/redefine them.
WORKFLOW_MUTATION_TOOLS = frozenset(
    {
        "workflow_define",
        "workflow_cancel",
        "workflow_force_abort",
        "workflow_revoke_definition",
        "workflow_pause",
        "workflow_resume",
        "workflow_remediate",
        # kestrel-feature-workflows 0.5.x: workflow_await_signal_deadline is
        # the scheduler's one-shot deadline target; workflow_await_signal_
        # delivery is the adapter its periodic durable-delivery reconciliation
        # shares. Each RESOLVES an await_signal wait — advances run state — so
        # the orchestrator must not call them (#3195). The ceiling cannot say
        # "machine-only" (MandateRestrictionHook matches on tool name alone),
        # so a scheduler tick on THIS agent is denied too — already true under
        # the positive allowlist before these names were listed. Measured
        # consequence: only the low-latency scheduler wakeup is lost. The
        # feature's in-process recovery worker resolves both lanes by direct
        # call (reconcile_await_signal_waits, whose waiting lane drains
        # deliveries through _drain_await_signal_wait and whose overdue lane
        # calls resolve_await_signal_deadline directly), never through the
        # tool path, so an await_signal stage hosted on the orchestrator
        # still completes — on the recovery cadence.
        "workflow_await_signal_deadline",
        "workflow_await_signal_delivery",
    }
)

# GitHub write tools — the complete set the GitHubFeature actually loads
# (verified against kestrel_feature_github/feature.py @tool names). The
# deny-list is derived from what the feature loads, not hand-maintained against
# intent: every issue/PR/label-mutating tool is denied so the orchestrator can
# read fleet repos but never mutate them directly (all writes flow through
# fleet_coding_pipeline). ``invalidate_github_cache`` is included because it
# mutates shared feature state.
GITHUB_WRITE_TOOLS = frozenset(
    {
        "create_github_issue",
        "add_github_issue_comment",
        "add_github_label",
        "remove_github_label",
        "close_github_issue",
        "reopen_github_issue",
        "create_github_pull_request",
        "merge_github_pull_request",
        "invalidate_github_cache",
    }
)

# Scheduler mutation tools (verified against features/scheduler/feature.py @tool
# names). SchedulerFeature is in the ceiling for the read/status surface
# (schedule_list / schedule_history / schedule_engagement, which feed the
# kestrel-claws Signals tab), but its schedule-mutating tools would let the
# agent create/alter recurring or one-shot scheduled tasks — including built-in
# signal sources — so they are hard-denied. The `reflect` triage sweep
# (REFLECTION_SCHEDULE) is seeded by the operator at materialization time, not
# by the agent, so denying schedule_add here does not block it.
SCHEDULER_MUTATION_TOOLS = frozenset(
    {
        "schedule_add",
        "schedule_add_deadline",
        "schedule_remove",
        "schedule_pause",
        "schedule_resume",
        "schedule_update",
        "schedule_record_outcome",
    }
)

# Code-editing / filesystem / shell tools. These come from features NOT in the
# ceiling, so they never load — but denying them explicitly is defense in depth
# and makes the "denies write/edit tools" contract testable directly against the
# hook (real tool names verified in features/compute + features/computer_use).
WRITE_EDIT_FILE_TOOLS = frozenset(
    {
        "fs_edit",
        "fs_write",
        "shell",
        "write_script",
        "execute_command",
        "apply_patch",
    }
)

# Memory mutation tools (verified against features/memory/feature.py @tool
# names). MemoryFeature is in the ceiling for its read/recall surface, but it
# also bundles destructive tools that delete/purge/restore conversations and
# messages, mutate claim nodes, resolve person identities, and run the
# consolidation pipeline. None are needed for fleet triage and all mutate the
# agent's own memory, so they are hard-denied. (`memory_consolidate` still runs
# as its built-in scheduled task; denying the tool only blocks the agent from
# invoking it directly.)
MEMORY_MUTATION_TOOLS = frozenset(
    {
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
        "memory_index_backfill",
    }
)

# Reflection mutation tools (verified against kestrel_feature_reflection @tool
# names). ReflectionFeature is in the ceiling for the `reflect` triage sweep and
# its read tools; the tools that mutate the self-model, file improvement
# tickets/proposals, or kick off a training cycle are denied — the orchestrator
# commissions all change through fleet_coding_pipeline, never by self-modifying.
REFLECTION_MUTATION_TOOLS = frozenset(
    {
        "update_self_model",
        "propose_improvement",
        "create_improvement_ticket",
        "training_cycle",
    }
)

RESTRICTED_TOOLS = (
    WORKFLOW_MUTATION_TOOLS
    | GITHUB_WRITE_TOOLS
    | SCHEDULER_MUTATION_TOOLS
    | MEMORY_MUTATION_TOOLS
    | REFLECTION_MUTATION_TOOLS
    | WRITE_EDIT_FILE_TOOLS
    | MANDATORY_MUTATION_TOOLS
)


# ---------------------------------------------------------------------------
# Argument-level tool restriction (positive allowlist, enforced at PRE_TOOL_USE).
# ---------------------------------------------------------------------------
# ``workflow_run`` is the orchestrator's ONLY dispatch surface, but a plain
# tool-name allowlist cannot stop it from starting *other* workflows
# (``stalled_work_rescue`` or any builtin/loaded definition), which would bypass
# the fleet_coding_pipeline-specific consent scope. MandateRestrictionHook
# enforces this at the argument level: ``workflow_run`` is denied unless its
# ``name`` argument (the workflow definition name) is fleet_coding_pipeline.
WORKFLOW_RUN_ALLOWED_NAMES = frozenset({"fleet_coding_pipeline"})

RESTRICTED_TOOL_ARGS: Dict[str, Dict[str, List[str]]] = {
    "workflow_run": {"name": sorted(WORKFLOW_RUN_ALLOWED_NAMES)},
}


# ---------------------------------------------------------------------------
# The governing constitution (surfaced via behavioral_rules — #2225).
# ---------------------------------------------------------------------------
# One rule per constitutional requirement in #2321. These render into the
# agent's governing constitution through the existing spawn-mandate constitution
# block (render_mandate_constitution_block), so no per-agent CONSTITUTION.md
# overlay editing is required.
BEHAVIORAL_RULES: List[str] = [
    (
        "Dispatch coding work ONLY by starting `fleet_coding_pipeline` workflow "
        "runs (via `workflow_run`). You never write, edit, or commit code, "
        "files, or repository state directly."
    ),
    (
        "Never invoke GitHub write tools (`create_github_issue`, "
        "`merge_github_pull_request`, …), scheduler "
        "mutation tools (`schedule_add`, `schedule_remove`, …), or any "
        "code-editing, filesystem, or shell tool directly. Commission every "
        "code change through the `fleet_coding_pipeline` workflow, which routes "
        "the actual run through the feature-owned audited dispatch seam. "
        "`workflow_run` may only start `fleet_coding_pipeline`, never any other "
        "workflow."
    ),
    (
        "Never approve your own `consent_collect` gates. Approvals belong to the "
        "human Falconer through the ApprovalQueue (the kestrel-claws Approvals "
        "tab); the `fleet_coding_pipeline` `approve_dispatch` gate must be "
        "satisfied by a human, never by you, and never by a marker you place in "
        "the run params."
    ),
    (
        "Escalate uncertainty to the human Falconer rather than guessing. When a "
        "directive is ambiguous, a target repo/issue is unclear, or evidence is "
        "missing, ask — do not proceed on assumption."
    ),
    (
        "On each periodic triage sweep, survey the assigned fleet repositories "
        "with GitHub read tools, identify actionable issues, and PLAN "
        "`fleet_coding_pipeline` runs parked on the consent gate. Detected work "
        "is observational: never auto-dispatch it. For an on-demand directive "
        "(\"process milestone/label X\"), plan the corresponding "
        "`fleet_coding_pipeline` runs and let each park on its consent gate."
    ),
]


# ---------------------------------------------------------------------------
# Scheduled reflection task (periodic triage sweep → /agent/reflection/status).
# ---------------------------------------------------------------------------
# `/agent/reflection/status` surfaces scheduled tasks whose task_name is in
# ("reflect", "training_cycle"); using `reflect` makes the sweep appear on the
# kestrel-claws Signals tab. Each tuple is (cron_expression, task_name,
# args_json) — the shape schedule_add consumes.
REFLECTION_SCHEDULE: List[Tuple[str, str, str]] = [
    ("0 */6 * * *", "reflect", '{"scope": "fleet_triage"}'),
]


# ---------------------------------------------------------------------------
# Builders — hand the declared constraints to the existing machinery.
# ---------------------------------------------------------------------------


def additional_constraints() -> Dict[str, Any]:
    """The mandate ``additional_constraints`` for the Fleet Orchestrator.

    Carries ``behavioral_rules`` (surfaced into the governing constitution —
    #2225) and ``restricted_tools`` (hard-denied at PRE_TOOL_USE — #2137). Both
    are *narrowing* constraints, so ``ScopedConstitution.validate_constraints``
    accepts them.
    """
    return {
        "allowed_tools": sorted(TOOL_ALLOWLIST),
        "behavioral_rules": list(BEHAVIORAL_RULES),
        "restricted_tools": sorted(RESTRICTED_TOOLS),
        "restricted_tool_args": {
            tool: {arg: list(values) for arg, values in spec.items()}
            for tool, spec in RESTRICTED_TOOL_ARGS.items()
        },
    }


def unclassified_tool_names(feature_cls: Any) -> frozenset:
    """The registered tool names of ``feature_cls`` that are neither allowed nor
    denied. A non-empty result means the deny-list has drifted from what the
    feature loads — the audit fails so the new tool is classified explicitly."""
    classified = TOOL_ALLOWLIST | RESTRICTED_TOOLS | frozenset(RESTRICTED_TOOL_ARGS)
    return registered_tool_names(feature_cls) - classified


def is_tool_allowed(tool_name: str) -> bool:
    """True when ``tool_name`` is in the positive scope and not denied."""
    return tool_name in TOOL_ALLOWLIST and tool_name not in RESTRICTED_TOOLS


def is_tool_denied(tool_name: str) -> bool:
    """True when ``tool_name`` is hard-denied by the restriction deny-list."""
    return tool_name in RESTRICTED_TOOLS


def is_tool_call_allowed(tool_name: str, tool_input: Optional[Dict[str, Any]] = None) -> bool:
    """True when a concrete ``tool_name`` + ``tool_input`` call is permitted.

    Layers the argument-level restriction (:data:`RESTRICTED_TOOL_ARGS`) on top
    of :func:`is_tool_allowed`: an arg-restricted tool (``workflow_run``) is only
    allowed when every restricted argument holds a permitted value. This mirrors
    exactly what :class:`MandateRestrictionHook` enforces at PRE_TOOL_USE.
    """
    if is_tool_denied(tool_name):
        return False
    args = tool_input or {}
    spec = RESTRICTED_TOOL_ARGS.get(tool_name)
    if spec:
        for arg_name, allowed in spec.items():
            if str(args.get(arg_name)) not in {str(v) for v in allowed}:
                return False
    return tool_name in TOOL_ALLOWLIST


def build_spawn_mandate(
    parent_did: str,
    child_did: Optional[str] = None,
    *,
    ttl_seconds: int = 365 * 24 * 3600,
    entry_points: Optional[Iterable[Any]] = None,
):
    """Build the Fleet Orchestrator's :class:`SpawnMandate` (unsigned).

    ``parent_did`` is the Sovereign that governs this orchestrator. The mandate
    carries the feature ceiling (:data:`FEATURE_ALLOWLIST`) and the constitution
    + tool deny-list (:func:`additional_constraints`); inception records these on
    the ``spawned_by`` edge, and every boot path re-applies them. Signing happens
    at inception with the parent's key (``sign_mandate``); this builder returns
    the unsigned mandate the caller signs. ``ttl_seconds`` defaults to a long
    (one-year) window because the orchestrator is a persistent, long-lived agent
    rather than an ephemeral task child.
    """
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    feature_ceiling = effective_feature_allowlist(entry_points=entry_points)
    return SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
        additional_constraints=additional_constraints(),
        features_allowed=sorted(feature_ceiling),
        purpose=FLEET_ORCHESTRATOR_NAME,
        ttl_seconds=ttl_seconds,
    )


def build_scoped_constitution(
    base_constitution: str = "",
    parent_features: Optional[set] = None,
    *,
    entry_points: Optional[Iterable[Any]] = None,
):
    """Build the :class:`ScopedConstitution` wrapping ``base_constitution``.

    Applies the orchestrator's feature ceiling and additional constraints. Pass
    ``parent_features`` (the Sovereign's feature set) to validate the ceiling is
    a subset; omit it to just render the effective constitution.
    """
    from kestrel_sovereign.spawn.scoped_constitution import ScopedConstitution

    feature_ceiling = effective_feature_allowlist(entry_points=entry_points)
    return ScopedConstitution(
        base_constitution=base_constitution,
        additional_constraints=additional_constraints(),
        features_allowed=sorted(feature_ceiling),
        parent_features=set(parent_features) if parent_features else set(),
    )


def constitution_text(*, entry_points: Optional[Iterable[Any]] = None) -> str:
    """The governing constitution block (behavioral rules + restricted tools).

    Renders the ``--- SPAWN MANDATE CONSTRAINTS ---`` section only (no base
    constitution), the same text ``render_mandate_constitution_block`` surfaces
    into the orchestrator's governing constitution at prompt-build time (#2225).
    """
    return build_scoped_constitution(
        entry_points=entry_points
    ).constraints_section()


def build_restriction_hook():
    """Build the :class:`MandateRestrictionHook` enforcing :data:`RESTRICTED_TOOLS`.

    This is the exact hook the reload path registers from the ``spawned_by``
    edge; exposing the builder lets callers (and tests) verify the hard denial
    directly.
    """
    from kestrel_sovereign.spawn.mandate_hook import MandateRestrictionHook

    return MandateRestrictionHook(
        sorted(RESTRICTED_TOOLS),
        allowed_tools=sorted(TOOL_ALLOWLIST),
        restricted_tool_args={
            tool: {arg: list(values) for arg, values in spec.items()}
            for tool, spec in RESTRICTED_TOOL_ARGS.items()
        },
    )


def build_local_agent_config(
    data_dir: str,
    port: int,
    *,
    autostart: bool = True,
    entry_points: Optional[Iterable[Any]] = None,
):
    """Build the ``multi_agent.toml`` entry (:class:`LocalAgentConfig`).

    Stamps the :data:`FEATURE_ALLOWLIST` as the ``features`` ceiling for defense
    in depth (the ceiling is also enforced from the ``spawned_by`` edge on every
    boot path). ``data_dir`` must point at the inceptioned Fleet Orchestrator
    agent directory.
    """
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig

    feature_ceiling = effective_feature_allowlist(entry_points=entry_points)
    return LocalAgentConfig(
        data_dir=data_dir,
        port=port,
        autostart=autostart,
        features=sorted(feature_ceiling),
    )
