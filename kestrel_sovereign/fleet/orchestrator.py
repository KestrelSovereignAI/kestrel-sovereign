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

1. **A feature ceiling** — it may load only the workflows / talon / GitHub /
   reflection features, never a file/shell/computer-use feature. This is the
   ``features`` allowlist (feature-class granularity) — see
   :data:`FEATURE_ALLOWLIST`.
2. **An intra-feature tool deny-list** — ``TalonCoordinatorFeature`` bundles
   *both* read tools (``talon_status`` …) and dispatch tools (``talon_claim`` …)
   in one class, and ``WorkflowsFeature`` bundles ``workflow_run`` alongside
   mutating tools (``workflow_cancel`` …). A feature-class allowlist cannot keep
   the reads while denying the dispatch tools, so the split is expressed as a
   tool-level deny-list — see :data:`RESTRICTED_TOOLS`.
3. **Governing behavioral rules** — the constitution encoded below.

All three are exactly what a signed :class:`~kestrel_sovereign.spawn.mandate.SpawnMandate`
carries: ``features_allowed`` (the ceiling, enforced at feature discovery —
#2226), ``additional_constraints.restricted_tools`` (hard-denied at
``PRE_TOOL_USE`` by :class:`~kestrel_sovereign.spawn.mandate_hook.MandateRestrictionHook`
— #2137), and ``additional_constraints.behavioral_rules`` (surfaced into the
governing constitution — #2225). Inceptioning the agent as a child of the
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

Attribution contract (#2302): the friendly name ``Fleet Orchestrator`` is what
the talon coordinator stamps as ``KESTREL_OBSERVABILITY_ORCHESTRATOR`` on every
dispatch it drives (via ``fleet_coding_pipeline``). Do not rename the agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Friendly name (rendered in /api/agents and stamped as the observability
# orchestrator — #2302) and the multi-agent routing slug.
FLEET_ORCHESTRATOR_NAME = "Fleet Orchestrator"
FLEET_ORCHESTRATOR_SLUG = "fleet-orchestrator"

# The consent scope its dispatches gate on (mirrors fleet_coding_pipeline).
CONSENT_SCOPE = "fleet_coding_pipeline_dispatch"


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
        "TalonCoordinatorFeature",  # talon READ tools (dispatch tools denied below)
        "GitHubFeature",          # GitHub read (list_issues / list_prs / get_repo_info)
        "SchedulerFeature",       # schedule_add + /agent/reflection/status surface
        "ReflectionFeature",      # the `reflect` triage-sweep task
        "MemoryFeature",          # reflection rides the sleep/consolidation cycle
    }
)

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
        "workflow_load_builtin",
    }
)

# Talon coordinator READ tools only (verified against features/talon/coordinator.py):
# status / logs / workspace / config / health / discovery scan. No claim/dispatch.
TALON_READ_TOOLS = frozenset(
    {
        "talon_status",
        "talon_job_log",
        "talon_workspace_status",
        "talon_get_config",
        "talon_health",
        "scan_stale_work",
    }
)

# GitHub read tools (data_access skills; create_issue / create_pr are denied below).
GITHUB_READ_TOOLS = frozenset(
    {
        "list_issues",
        "list_prs",
        "get_repo_info",
    }
)

TOOL_ALLOWLIST = WORKFLOW_TOOLS | TALON_READ_TOOLS | GITHUB_READ_TOOLS


# ---------------------------------------------------------------------------
# Tool deny-list (hard PRE_TOOL_USE denial via MandateRestrictionHook, #2137).
# ---------------------------------------------------------------------------
# Talon dispatch / write tools — the orchestrator never claims or dispatches
# directly; it commissions all code changes through fleet_coding_pipeline.
TALON_DISPATCH_TOOLS = frozenset(
    {
        "talon_claim",
        "talon_file_and_claim",
        "talon_batch",
        "talon_set_config",
        "talon_verify",
        "talon_setup_workspace",
        "talon_schedule_work_rescue",
        "talon_pause",
        "talon_resume",
    }
)

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
    }
)

# GitHub write tools.
GITHUB_WRITE_TOOLS = frozenset(
    {
        "create_issue",
        "create_pr",
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

RESTRICTED_TOOLS = (
    TALON_DISPATCH_TOOLS
    | WORKFLOW_MUTATION_TOOLS
    | GITHUB_WRITE_TOOLS
    | WRITE_EDIT_FILE_TOOLS
)


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
        "Never invoke talon claim/dispatch tools (`talon_claim`, "
        "`talon_file_and_claim`, `talon_batch`, …) or any code-editing, "
        "filesystem, or shell tool directly. Commission every code change "
        "through the `fleet_coding_pipeline` workflow, which routes the actual "
        "run through the coordinator's audited dispatch seam."
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
        "(GitHub read + talon read tools), identify actionable issues, and PLAN "
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
        "behavioral_rules": list(BEHAVIORAL_RULES),
        "restricted_tools": sorted(RESTRICTED_TOOLS),
    }


def is_tool_allowed(tool_name: str) -> bool:
    """True when ``tool_name`` is in the positive scope and not denied."""
    return tool_name in TOOL_ALLOWLIST and tool_name not in RESTRICTED_TOOLS


def is_tool_denied(tool_name: str) -> bool:
    """True when ``tool_name`` is hard-denied by the restriction deny-list."""
    return tool_name in RESTRICTED_TOOLS


def build_spawn_mandate(
    parent_did: str,
    child_did: Optional[str] = None,
    *,
    ttl_seconds: int = 365 * 24 * 3600,
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

    return SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
        additional_constraints=additional_constraints(),
        features_allowed=sorted(FEATURE_ALLOWLIST),
        purpose=FLEET_ORCHESTRATOR_NAME,
        ttl_seconds=ttl_seconds,
    )


def build_scoped_constitution(
    base_constitution: str = "",
    parent_features: Optional[set] = None,
):
    """Build the :class:`ScopedConstitution` wrapping ``base_constitution``.

    Applies the orchestrator's feature ceiling and additional constraints. Pass
    ``parent_features`` (the Sovereign's feature set) to validate the ceiling is
    a subset; omit it to just render the effective constitution.
    """
    from kestrel_sovereign.spawn.scoped_constitution import ScopedConstitution

    return ScopedConstitution(
        base_constitution=base_constitution,
        additional_constraints=additional_constraints(),
        features_allowed=sorted(FEATURE_ALLOWLIST),
        parent_features=set(parent_features) if parent_features else set(),
    )


def constitution_text() -> str:
    """The governing constitution block (behavioral rules + restricted tools).

    Renders the ``--- SPAWN MANDATE CONSTRAINTS ---`` section only (no base
    constitution), the same text ``render_mandate_constitution_block`` surfaces
    into the orchestrator's governing constitution at prompt-build time (#2225).
    """
    return build_scoped_constitution().constraints_section()


def build_restriction_hook():
    """Build the :class:`MandateRestrictionHook` enforcing :data:`RESTRICTED_TOOLS`.

    This is the exact hook the reload path registers from the ``spawned_by``
    edge; exposing the builder lets callers (and tests) verify the hard denial
    directly.
    """
    from kestrel_sovereign.spawn.mandate_hook import MandateRestrictionHook

    return MandateRestrictionHook(sorted(RESTRICTED_TOOLS))


def build_local_agent_config(
    data_dir: str,
    port: int,
    *,
    autostart: bool = True,
):
    """Build the ``multi_agent.toml`` entry (:class:`LocalAgentConfig`).

    Stamps the :data:`FEATURE_ALLOWLIST` as the ``features`` ceiling for defense
    in depth (the ceiling is also enforced from the ``spawned_by`` edge on every
    boot path). ``data_dir`` must point at the inceptioned Fleet Orchestrator
    agent directory.
    """
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig

    return LocalAgentConfig(
        data_dir=data_dir,
        port=port,
        autostart=autostart,
        features=sorted(FEATURE_ALLOWLIST),
    )
