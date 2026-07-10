"""Built-in ``fleet_coding_pipeline`` workflow + its host-provided sources (#2303).

The Fleet Orchestrator's coding loop: gate a talon coding run behind human
approval, dispatch the run (write code, push a branch, open a PR), then confirm
the PR's CI went green. It is the write-side counterpart to
``stalled_work_rescue`` (which only *routes* repairs); here the pipeline
actually commissions the code change through the purpose-built
``talon_pipeline_dispatch`` source (#2302).

Why this lives in ``kestrel-sovereign`` rather than the workflows package
====================================================================
``kestrel_feature_workflows`` ships generic, agent-agnostic built-ins (see its
``library.py``). ``fleet_coding_pipeline`` is Talon/fleet-coordination domain —
it names the sovereign-native ``talon_pipeline_dispatch`` source and the two
support sources below — so the coordinator *registers* it into the workflows
package's built-in registry at ``initialize()`` time (the same place it
registers the workflow-rescue sources). No file in ``kestrel-feature-workflows``
is edited: :func:`register_fleet_coding_pipeline_builtin` injects the builder
into that package's ``BUILTIN_BUILDERS`` / ``BUILTIN_DESCRIPTIONS`` maps at
runtime, so ``workflow_list_builtin`` / ``workflow_load_builtin`` /
``workflow_run`` discover and run it like any other built-in.

Stages (manual trigger)
=======================
1. ``approve_dispatch`` — ``consent_collect`` gate. Human Falconer approval
   through the agent's existing permission / ApprovalQueue system (surfaced in
   the kestrel-claws Approvals tab) *before any code is written*. Default-closed:
   with ``{repo, issue}`` and no consent arg the gate parks the run in WAITING
   until a human approves. Fully-autonomous mode is opt-in — a caller builds the
   spec with :func:`build_fleet_coding_pipeline_spec` ``(autonomous=True)`` to
   omit this stage. The Fleet Orchestrator must NEVER be able to approve its own
   gate; default-closed is the contract.
2. ``talon_run`` — dispatches ``talon_pipeline_dispatch`` (claim mode from
   ``issue``, or iterate mode from ``pr``). ``self_review: true`` and the
   ``demo_check`` / ``eye_check`` flags thread through; gate ``signal_status_ok``.
   **Irreversible** — a run writes code, pushes a branch, and opens a PR — so it
   declares ``irreversible=True`` / ``compensate="compensate_record_only"``:
   compensation is a *record-only* notification, never a rollback (a pushed
   branch / opened PR cannot be uncommitted), mirroring ``a2a_repair_dispatch``'s
   irreversibility posture. ``KESTREL_OBSERVABILITY_WORKFLOW_RUN_ID`` / ``_STAGE``
   / ``_ORCHESTRATOR`` are stamped end-to-end by the coordinator's dispatch
   funnels (#2302) because this stage routes through them.
3. ``verify_ci`` — ``signal_status_ok`` gate over the coordinator-bound
   ``fleet_ci_probe`` source, which confirms the CI of the PR the ``talon_run``
   stage actually produced went green. The probe binds verification to the
   *dispatched run's own output* — it never trusts a caller-supplied ``branch``
   (#2303). It calls ``coordinator.verify_pipeline_ci(repo, issue|pr, mode)``,
   which locates the talon job this run dispatched, waits for it to open its PR,
   resolves that PR's head branch from GitHub, and polls the PR head's CI
   (reusing the workflows ``ci_green`` gate machinery). Fail-closed: the probe
   raises (so ``signal_status_ok`` fails) whenever the PR/branch cannot be bound
   to the dispatch or CI is not observably green.

Why the branch is NOT read from run params (#2303)
==================================================
An earlier design materialized the ``ci_green`` gate's ``branch`` from the run
params (``branch_param``). But the runner merges caller-supplied run params into
every stage payload, so a caller could pass ``branch: main`` and get a
CI-green verdict for a branch the talon PR never touched (or, omitting it, fail
closed on the common issue-only path). Verification therefore ignores any
caller ``branch`` entirely and binds to the talon PR discovered from the
dispatch. In claim mode the PR head is only known after ``talon_pipeline_dispatch``
runs, so ``verify_pipeline_ci`` correlates the dispatched job by (repo, issue)
[claim] / (repo, pr) [iterate], reads the PR the job opened from its output, and
verifies that PR's head — not run params.

Wait strategy (spec item #3 — ONE approach, documented here)
============================================================
``talon_pipeline_dispatch`` caps its held-wait at 3600s
(``MAX_HANDLE_WAIT_SECONDS``). Real coding runs routinely exceed an hour, so
holding the dispatch stage in-process would time out the common case. This
pipeline therefore dispatches with ``wait: false``: the source returns right
after dispatch, and per #2302's contract ``wait: false`` forces the durable
``cli_background`` path so the returned ``wait_ref`` (``talon:<job_id>``,
carried in the stage result) survives an agent restart. The long wait for the
run to finish (and its PR to open) then lives in the ``verify_ci`` stage:
``verify_pipeline_ci`` awaits the dispatched job's completion on the same
durable wait rail before resolving and polling its PR head CI, rather than
holding the dispatch stage on the in-process wait ceiling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from kestrel_sdk.signals import (
    RateLimit,
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Trust,
)

from kestrel_sovereign.signals.sources.talon_pipeline import (
    SOURCE_NAME as TALON_PIPELINE_SOURCE,
)

logger = logging.getLogger(__name__)

# Built-in workflow name + the consent scope its approval gate collects on.
WORKFLOW_NAME = "fleet_coding_pipeline"
CONSENT_SCOPE = "fleet_coding_pipeline_dispatch"

WORKFLOW_DESCRIPTION = (
    "Fleet coding loop (#2303): gate a talon coding run behind human approval, "
    "dispatch it via talon_pipeline_dispatch (writes code, opens a PR), then "
    "verify the PR's CI is green."
)

# Stage names (also the observability STAGE value stamped onto talon_run).
APPROVE_DISPATCH_STAGE = "approve_dispatch"
TALON_RUN_STAGE = "talon_run"
VERIFY_CI_STAGE = "verify_ci"

# Host-provided support sources this workflow names (besides talon_pipeline_dispatch).
FLEET_CODING_APPROVAL = "fleet_coding_approval"
FLEET_CI_PROBE = "fleet_ci_probe"
SOURCE_NAMES = (FLEET_CODING_APPROVAL, FLEET_CI_PROBE)

# Approval-decision fields the ``consent_collect`` gate reads off an ACTION
# result. These carry NO provenance when they arrive in caller-supplied run
# params, so the approval handler must strip them before returning — an
# orchestrator that can invoke ``workflow_run`` must never be able to satisfy
# its own consent gate by putting ``{approved: true}`` in the params (#2303).
# Real approval flows only through the runner's ``consent_collect_provider``
# (the agent's permission / ApprovalQueue system), consulted when the handler
# carries no marker of its own.
CONSENT_MARKER_FIELDS = frozenset(
    {
        "approved",
        "consent",
        "accepted",
        "decision",
        "status",
        "state",
        "outcome",
        "approval_id",
        "approved_by",
    }
)


def _quote(source: str, field: str, value: Any) -> str:
    """Format an observation the evidence-boundary way (AGENTS.md convention)."""
    return f"`{source}` reported `{field}: {value!r}`."


# --------------------------------------------------------------------------
# Support-source handlers — conservative, fail-closed, evidence-preserving.
# --------------------------------------------------------------------------


async def fleet_coding_approval_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Record the intent to dispatch a talon coding run. Consent-gated.

    Like ``governance_review`` in ``stalled_work_rescue``, this only *registers*
    the intent; it never authorizes it. Authorization is the workflow's
    ``consent_collect`` gate, evaluated separately through the agent's permission
    system (the runner's ``consent_collect_provider``, which routes to the
    ApprovalQueue / a human in the kestrel-claws Approvals tab).

    **Caller-supplied approval markers are NOT honored (#2303).** The workflow
    runner merges run params into every stage payload, so any ``approved`` /
    ``consent`` / ``status`` / ... field a caller put in the run params arrives
    here with no provenance from the approval system. Threading it into the
    result would let an orchestrator that can invoke ``workflow_run`` satisfy its
    own consent gate and approve its own irreversible ``talon_run`` dispatch —
    breaking the default-closed contract. So this handler deliberately builds a
    fresh, marker-free result: it never carries a consent decision, forcing the
    gate down the ``consent_collect_missing_approval`` path where the real
    approval provider decides (default-closed, parking the run in WAITING until a
    human approves).
    """
    scope = payload.get("scope", CONSENT_SCOPE)
    repo = payload.get("repo")
    # Fresh result with only descriptive/observational fields. No key from
    # CONSENT_MARKER_FIELDS is copied off the (caller-influenced) payload, so the
    # gate can never read a self-granted approval marker from this result.
    return {
        "source": FLEET_CODING_APPROVAL,
        "scope": scope,
        "repo": repo,
        "issue": payload.get("issue"),
        "pr": payload.get("pr"),
        "intent": "request_consent",
        # Authorization is the consent gate's job, not ours (default-closed).
        "authorized": False,
        "observation": _quote(FLEET_CODING_APPROVAL, "scope", scope),
    }


async def _run_fleet_ci_probe(
    coordinator: Any, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Verify the CI of the PR the ``talon_run`` stage produced went green.

    Bound to the coordinator so it can consult the dispatched run's own output.
    **A caller-supplied ``branch`` is deliberately never read** (#2303): the
    branch to verify is bound to the talon PR, not to run params. This delegates
    to ``coordinator.verify_pipeline_ci`` (which locates the dispatched job,
    resolves its PR head branch, and polls that head's CI) and fails closed —
    raising, so the ``signal_status_ok`` gate fails — whenever CI is not
    verified green against the talon PR head.
    """
    repo = payload.get("repo")
    issue = payload.get("issue")
    pr = payload.get("pr")
    # Derive mode the same way ``talon_pipeline_dispatch`` does: an explicit
    # ``mode`` wins, else a ``pr`` (without an ``issue``) means iterate.
    mode = payload.get("mode") or ("iterate" if pr and not issue else "claim")

    verify = getattr(coordinator, "verify_pipeline_ci", None)
    if verify is None:
        raise ValueError(
            f"{FLEET_CI_PROBE}: no coordinator bound to verify the talon PR's "
            "CI (fail closed)"
        )
    verdict = await verify(repo=repo, issue=issue, pr=pr, mode=mode)
    if not isinstance(verdict, dict) or not verdict.get("ci_green"):
        reason = verdict.get("reason") if isinstance(verdict, dict) else repr(verdict)
        raise ValueError(
            f"{FLEET_CI_PROBE}: talon PR CI not verified green "
            f"(reason={reason!r}) (fail closed)"
        )
    branch = verdict.get("branch")
    return {
        "source": FLEET_CI_PROBE,
        "repo": verdict.get("repo", repo),
        "branch": branch,
        "pr": verdict.get("pr", pr),
        "pr_url": verdict.get("pr_url"),
        "job_id": verdict.get("job_id"),
        "head_sha": verdict.get("head_sha"),
        "state": "green",
        "observation": _quote(FLEET_CI_PROBE, "branch", branch),
    }


def make_fleet_ci_probe_handler(coordinator: Any):
    """Build the ``fleet_ci_probe`` ACTION handler bound to a coordinator."""

    async def handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        return await _run_fleet_ci_probe(coordinator, payload)

    return handler


# --------------------------------------------------------------------------
# Support-source registrations.
# --------------------------------------------------------------------------


def _redaction(source: str) -> RedactionPolicy:
    return RedactionPolicy(
        summarize=lambda payload, _s=source: (
            f"{_s} repo={payload.get('repo', '?')} "
            f"issue={payload.get('issue', '?')} pr={payload.get('pr', '?')}"
        ),
        # These payloads are the bird's own orchestration data (workflow stage
        # params), not third-party UNTRUSTED content.
        store_raw_trusted=False,
        redact_caller_identifier=True,
    )


def _action_registration(name: str, handler: Any) -> SourceRegistration:
    return SourceRegistration(
        name=name,
        schema=_passthrough_schema,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        trust=Trust.TRUSTED,
        # Workflow stages fire deliberately, one at a time; a modest cap only
        # guards against a pathological retry loop, never a legitimate run.
        rate_limit=RateLimit(per_minute=30, per_hour=240),
        resources=frozenset(),
        allow_self_loops=False,
        log_redaction=_redaction(name),
        retention_days=30,
    )


def _passthrough_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept any dict payload; reject non-dicts so the audit row is stable."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"fleet-coding-pipeline payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    return payload


def build_fleet_coding_pipeline_registrations(
    coordinator: Any = None,
) -> List[SourceRegistration]:
    """The two support sources the ``fleet_coding_pipeline`` workflow names.

    ``talon_pipeline_dispatch`` is registered separately (it is bound to the
    coordinator). ``fleet_coding_approval`` is a pure function of its payload;
    ``fleet_ci_probe`` is bound to ``coordinator`` so it can verify the CI of
    the PR the dispatch produced (#2303). Without a coordinator the probe is
    still registered but fails closed on every call — it never trusts a
    caller-supplied branch as a substitute.
    """
    return [
        _action_registration(FLEET_CODING_APPROVAL, fleet_coding_approval_handler),
        _action_registration(
            FLEET_CI_PROBE, make_fleet_ci_probe_handler(coordinator)
        ),
    ]


def register_fleet_coding_pipeline_sources(
    registry: Any, coordinator: Any = None
) -> List[str]:
    """Register the support sources on ``registry``, idempotently.

    ``coordinator`` binds the ``fleet_ci_probe`` verify source so it can consult
    the dispatched run's output; the coordinator passes itself at init time.
    Returns the names newly registered (already-present sources are skipped so a
    second call — or a host that registered a richer implementation — is safe).
    A single source's failure is logged and does not abort the rest.
    """
    registered: List[str] = []
    if registry is None or not hasattr(registry, "register"):
        return registered
    has_get = hasattr(registry, "get")
    for registration in build_fleet_coding_pipeline_registrations(coordinator):
        if has_get and registry.get(registration.name) is not None:
            continue
        try:
            registry.register(registration)
            registered.append(registration.name)
        except Exception as exc:  # noqa: BLE001 - one bad source must not abort the rest
            logger.warning(
                "could not register fleet-coding-pipeline source %s: %s",
                registration.name,
                exc,
            )
    return registered


# --------------------------------------------------------------------------
# The WorkflowSpec builder + built-in registry injection.
# --------------------------------------------------------------------------


def build_fleet_coding_pipeline_spec(autonomous: bool = False):
    """Build the ``fleet_coding_pipeline`` :class:`WorkflowSpec`.

    ``autonomous=True`` omits the ``approve_dispatch`` consent stage for the
    opt-in fully-autonomous mode; the default (``False``) keeps consent
    default-closed. Imports the workflows models lazily so importing this module
    never requires ``kestrel-feature-workflows`` to be installed.
    """
    from kestrel_feature_workflows.models import (
        Edge,
        EdgeKind,
        Gate,
        Stage,
        Trigger,
        TriggerKind,
        WorkflowSpec,
    )

    stages: List[Any] = []
    edges: List[Any] = []

    if not autonomous:
        # Approve — human Falconer consent before any code is written. Like
        # stalled_work_rescue's govern_intent, the consent_collect gate makes
        # noop_idempotent eligible on its own (a rejected/pending request needs
        # no reversal). Default-closed: with no approval marker the gate parks
        # the run in WAITING until a human approves in the Approvals tab.
        stages.append(
            Stage(
                name=APPROVE_DISPATCH_STAGE,
                signal_source=FLEET_CODING_APPROVAL,
                signal_mode=SignalMode.ACTION,
                params={"scope": CONSENT_SCOPE},
                gate=Gate(
                    type="consent_collect",
                    params={"scope": CONSENT_SCOPE},
                ),
                compensate="noop_idempotent",
            )
        )

    # Dispatch — actually commission the talon coding run. Irreversible (writes
    # code, pushes a branch, opens a PR); wait: false + self_review: true.
    # demo_check / eye_check thread through from run params when supplied.
    stages.append(
        Stage(
            name=TALON_RUN_STAGE,
            signal_source=TALON_PIPELINE_SOURCE,
            signal_mode=SignalMode.ACTION,
            params={"self_review": True, "wait": False},
            gate=Gate(type="signal_status_ok"),
            irreversible=True,
            compensate="compensate_record_only",
        )
    )

    # Verify — confirm the talon PR's CI went green. The coordinator-bound
    # fleet_ci_probe binds verification to the dispatched run's own output (it
    # locates the talon job, resolves its PR head branch, and polls that head's
    # CI), so a plain signal_status_ok gate over the probe is the whole gate:
    # the probe raises unless CI is verified green against the talon PR head. A
    # caller-supplied ``branch`` is never read (#2303).
    stages.append(
        Stage(
            name=VERIFY_CI_STAGE,
            signal_source=FLEET_CI_PROBE,
            signal_mode=SignalMode.ACTION,
            params={},
            gate=Gate(type="signal_status_ok"),
            read_only=True,
            compensate="noop_idempotent",
        )
    )

    for earlier, later in zip(stages, stages[1:]):
        edges.append(
            Edge(
                kind=EdgeKind.SEQUENTIAL,
                from_stage=earlier.name,
                to_stage=later.name,
            )
        )

    return WorkflowSpec(
        name=WORKFLOW_NAME,
        version=1,
        stages=stages,
        edges=edges,
        # Runnable on demand; the Fleet Orchestrator starts it explicitly.
        triggers=[Trigger(kind=TriggerKind.MANUAL)],
        params_schema={
            "type": "object",
            "required": ["repo"],
            "properties": {
                "repo": {"type": "string"},
                "issue": {"type": "integer", "minimum": 1},
                "pr": {"type": "integer", "minimum": 1},
                "mode": {"type": "string", "enum": ["claim", "iterate"]},
                # No ``branch``: verification binds to the talon PR the dispatch
                # produces, never a caller-supplied branch (#2303).
                "self_review": {"type": "boolean"},
                "demo_check": {"type": "boolean"},
                "eye_check": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        retention_days=30,
    )


def register_fleet_coding_pipeline_builtin() -> bool:
    """Inject ``fleet_coding_pipeline`` into the workflows package built-in registry.

    Mutates ``kestrel_feature_workflows.library.BUILTIN_BUILDERS`` /
    ``BUILTIN_DESCRIPTIONS`` at runtime (no file in that package is edited) so
    ``workflow_list_builtin`` / ``workflow_load_builtin`` / ``workflow_run``
    discover and run it. The registered builder is the default consent-required
    variant; fully-autonomous callers use
    :func:`build_fleet_coding_pipeline_spec` ``(autonomous=True)`` directly.

    Returns True when newly injected; False when the workflows package is absent
    (import guarded) or the name is already registered (a host may have provided
    a richer definition — left untouched).
    """
    try:
        from kestrel_feature_workflows.library import (
            BUILTIN_BUILDERS,
            BUILTIN_DESCRIPTIONS,
        )
    except Exception as exc:  # noqa: BLE001 - workflows feature is optional
        logger.debug(
            "kestrel-feature-workflows not available; not registering the "
            "%s built-in workflow: %s",
            WORKFLOW_NAME,
            exc,
        )
        return False
    if WORKFLOW_NAME in BUILTIN_BUILDERS:
        return False
    BUILTIN_BUILDERS[WORKFLOW_NAME] = build_fleet_coding_pipeline_spec
    BUILTIN_DESCRIPTIONS[WORKFLOW_NAME] = WORKFLOW_DESCRIPTION
    return True
