"""``talon_pipeline_dispatch`` — a workflow-stage source that ACTUALLY
dispatches a full kestrel-talon pipeline run and reports the outcome.

Unlike ``a2a_repair_dispatch`` (see :mod:`workflow_rescue`), which
deliberately records intent without auto-dispatching, this source is the
purpose-built "go do it" stage: given a repo plus an issue (claim mode) or
a PR (iterate mode), it launches kestrel-talon through the coordinator's
existing invocation plumbing (A2A preferred for plain claims, CLI fallback /
CLI-only for iterate and extra quality gates) and — by default — holds the
stage until the run reaches a terminal state via the same
``talon.job_complete`` / :class:`TalonWaitable` rail the generic ``wait``
tool uses.

**IRREVERSIBLE.** A pipeline run writes code, pushes branches, and opens
PRs. Like ``a2a_repair_dispatch``, the handler fails closed on any
validation or dispatch problem (raising fails the workflow stage), and the
result records ``state: "dispatched"`` / ``"complete"`` honestly — a
dispatched run is never reported as merged/shipped.

Orchestrator identity + workflow correlation (kestrel-talon#53) are NOT
stamped here: the coordinator's dispatch funnels stamp
``KESTREL_OBSERVABILITY_ORCHESTRATOR`` / ``..._WORKFLOW_RUN_ID`` /
``..._STAGE`` onto every outgoing invocation (CLI env vars, A2A metadata),
reading the workflow run id off the in-flight Signal's ``session_id``. This
source only needs to run inside the stage's signal dispatch — which it
does, by construction.

Stage params (the workflow stage's ``params``, merged into the payload by
the workflows runner):

    repo        owner/name (or "self")                       REQUIRED
    issue       issue number  -> claim mode                  one of
    pr          PR number     -> iterate mode                issue/pr
    mode        "claim" | "iterate" (optional; inferred)
    self_review bool (optional; talon --self-review)
    demo_check  bool (optional; talon --demo-check)
    eye_check   bool (optional; talon --eye-check)
    wait        bool (default True) — hold the stage until the run
                finishes; False returns right after dispatch with a
                ``wait_ref`` (``talon:<job_id>``) for a later wait/watch.
    wait_timeout_seconds  int (default 3600). Max 3600 — the wait
                engine's held-wait ceiling (``MAX_HANDLE_WAIT_SECONDS``);
                larger values are rejected at validation time. Runs that
                need to be watched longer use ``wait: false`` plus the
                returned ``wait_ref`` on the signal-resume rail.

Detached dispatches force the CLI transport: with ``wait: false`` the
returned ``wait_ref`` (``talon:<job_id>``) is meant to outlive this stage —
possibly across an agent restart — so the dispatch is forced onto the CLI
path (``dispatch_pipeline(force_cli=True)``), whose ``cli_background`` jobs
are durably persisted. A2A jobs live only in the coordinator's in-memory
job map, so an A2A-shaped wait ref would resolve to an unknown job after a
restart. ``wait: true`` keeps the A2A-preferred path: the wait is held
in-process, and a restart simply fails that stage closed (see below).

Known limitation: an A2A claim the talon daemon accepts on the wire but
then drops (e.g. daemon crash before its task_store row reaches a terminal
state) never resolves, so a waiting stage holds until
``wait_timeout_seconds`` elapses and then fails closed, carrying the
``wait_ref`` for follow-up.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from kestrel_sdk.signals import (
    RateLimit,
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Trust,
)
from kestrel_sovereign.signals.registry import (
    RegistrationPolicy,
    RegistrationState,
)
from kestrel_sovereign.waits.engine import MAX_HANDLE_WAIT_SECONDS

logger = logging.getLogger(__name__)

SOURCE_NAME = "talon_pipeline_dispatch"

VALID_MODES = ("claim", "iterate")
_FLAG_KEYS = ("self_review", "demo_check", "eye_check")

DEFAULT_WAIT_TIMEOUT_SECONDS = 3600

_PR_URL_RE = re.compile(r"https://github\.com/[^\s\"'<>]+/pull/\d+")

# ``owner/name`` — one slash, and neither segment may start with a dash
# (dash-leading names are invalid on GitHub and a classic argv-injection
# shape). ``"self"`` is additionally accepted and resolved downstream by
# the coordinator's ``_resolve_repo``. This is a new machine-driven
# surface, so it validates strictly rather than inheriting talon_claim's
# laxity.
_REPO_RE = re.compile(
    r"^[A-Za-z0-9_.][A-Za-z0-9_.-]*/[A-Za-z0-9_.][A-Za-z0-9_.-]*$"
)


def _coerce_number(value: Any, key: str) -> Optional[int]:
    """Parse a numeric param strictly; ``None``/missing passes through.

    Accepts only actual ints (bools excluded — ``isinstance(True, int)``
    is True) or digit-only ASCII strings. Floats are rejected even when
    integral: ``int(12.9)`` would silently truncate and dispatch the
    irreversible pipeline against the wrong issue/PR (codex P2), and
    accepting 12.0 while rejecting 12.9 would make the boundary depend on
    the payload's float noise — so the whole type is refused (fail closed).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{SOURCE_NAME}: {key} must be an integer, got a bool")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        if not (text.isascii() and text.isdigit()):
            raise ValueError(
                f"{SOURCE_NAME}: {key} must be an integer or digit-only "
                f"string, got {value!r} (fail closed)"
            )
        number = int(text)
    else:
        raise ValueError(
            f"{SOURCE_NAME}: {key} must be an integer or digit-only string, "
            f"got {type(value).__name__} {value!r} — floats are rejected to "
            "prevent silent truncation (fail closed)"
        )
    if number < 1:
        raise ValueError(f"{SOURCE_NAME}: {key} must be >= 1, got {number}")
    return number


def validate_pipeline_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a stage payload into dispatch params.

    Fail closed (raise ``ValueError``) on anything ambiguous — an
    irreversible dispatch must never guess its target.
    """
    from kestrel_sovereign.features.talon.runtime import parse_talon_bool

    if not isinstance(payload, dict):
        raise ValueError(
            f"{SOURCE_NAME} payload must be a dict, got {type(payload).__name__}"
        )

    repo = payload.get("repo")
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError(
            f"{SOURCE_NAME}: repo (owner/name) is required (fail closed)"
        )
    repo = repo.strip()
    if repo != "self" and not _REPO_RE.match(repo):
        raise ValueError(
            f"{SOURCE_NAME}: repo must be 'owner/name' (segments of "
            f"[A-Za-z0-9_.-], not starting with '-') or 'self'; got "
            f"{repo!r} (fail closed)"
        )

    issue = _coerce_number(payload.get("issue"), "issue")
    pr = _coerce_number(payload.get("pr"), "pr")

    mode = payload.get("mode")
    if mode is not None:
        if not isinstance(mode, str) or mode not in VALID_MODES:
            raise ValueError(
                f"{SOURCE_NAME}: mode must be one of {VALID_MODES}, got {mode!r}"
            )
    else:
        if issue is not None and pr is not None:
            raise ValueError(
                f"{SOURCE_NAME}: both issue and pr supplied without an "
                "explicit mode; refusing to guess the target (fail closed)"
            )
        if issue is not None:
            mode = "claim"
        elif pr is not None:
            mode = "iterate"
        else:
            raise ValueError(
                f"{SOURCE_NAME}: an issue (claim) or pr (iterate) number is "
                "required (fail closed)"
            )

    if mode == "claim" and issue is None:
        raise ValueError(f"{SOURCE_NAME}: mode='claim' requires issue")
    if mode == "iterate" and pr is None:
        raise ValueError(f"{SOURCE_NAME}: mode='iterate' requires pr")

    params: Dict[str, Any] = {
        "repo": repo,
        "issue": issue if mode == "claim" else None,
        "pr": pr if mode == "iterate" else None,
        "mode": mode,
    }
    for key in _FLAG_KEYS:
        raw = payload.get(key)
        params[key] = None if raw is None else parse_talon_bool(raw, key)

    raw_wait = payload.get("wait")
    params["wait"] = True if raw_wait is None else parse_talon_bool(raw_wait, "wait")
    timeout = _coerce_number(
        payload.get("wait_timeout_seconds"), "wait_timeout_seconds"
    )
    if timeout is not None and timeout > MAX_HANDLE_WAIT_SECONDS:
        # run_wait_loop hard-rejects larger values with an immediate
        # failure carrying no job status; catch it here with an
        # actionable message instead.
        raise ValueError(
            f"{SOURCE_NAME}: wait_timeout_seconds {timeout} exceeds the "
            f"held-wait ceiling of {MAX_HANDLE_WAIT_SECONDS}s "
            "(MAX_HANDLE_WAIT_SECONDS); use wait=false and watch the "
            "returned wait_ref instead (fail closed)"
        )
    params["wait_timeout_seconds"] = timeout or DEFAULT_WAIT_TIMEOUT_SECONDS
    return params


def _find_pr_url(coordinator: Any, job_id: str) -> Optional[str]:
    """Best-effort PR URL from the job's combined log (last match wins)."""
    try:
        info = coordinator._jobs.get(job_id) or {}
        tail = coordinator._tail_job_log(info.get("log_path"), lines=400)
    except Exception:  # noqa: BLE001 - enrichment only, never fails the stage
        return None
    matches = _PR_URL_RE.findall(tail or "")
    return matches[-1] if matches else None


async def _await_completion(
    coordinator: Any, job_id: str, timeout_seconds: int
) -> Any:
    """Hold until the talon job is terminal, via the existing wait rail.

    Prefers the agent's :class:`WaitRegistry` (the same path the generic
    ``wait("talon:<job_id>")`` tool takes); falls back to driving
    :class:`TalonWaitable` through ``run_wait_loop`` directly when no
    registry is mounted (test stubs, headless agents). Returns the
    engine's ``ToolResult``.
    """
    ref = f"talon:{job_id}"
    agent = getattr(coordinator, "agent", None)
    registry = getattr(agent, "wait_registry", None) if agent is not None else None
    if registry is not None and hasattr(registry, "wait"):
        return await registry.wait(ref, timeout_seconds=timeout_seconds)

    from kestrel_sovereign.features.talon.wait_provider import TalonWaitable
    from kestrel_sovereign.waits.engine import run_wait_loop

    return await run_wait_loop(
        TalonWaitable(coordinator),
        job_id,
        timeout_seconds=timeout_seconds,
        label=ref,
    )


def make_talon_pipeline_dispatch_handler(coordinator: Any):
    """Build the ACTION handler bound to a TalonCoordinatorFeature."""

    async def handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        params = validate_pipeline_params(payload)

        dispatch = await coordinator.dispatch_pipeline(
            repo=params["repo"],
            issue=params["issue"],
            pr=params["pr"],
            mode=params["mode"],
            self_review=params["self_review"],
            demo_check=bool(params["demo_check"]),
            eye_check=bool(params["eye_check"]),
            # Detached (wait: false) hands the wait_ref back for later —
            # possibly across an agent restart — so it must land a durable
            # cli_background job, never an in-memory-only A2A task.
            force_cli=not params["wait"],
        )
        if not isinstance(dispatch, dict) or not dispatch.get("dispatched"):
            error = (
                dispatch.get("error")
                if isinstance(dispatch, dict)
                else f"unexpected dispatch result: {dispatch!r}"
            )
            raise ValueError(
                f"{SOURCE_NAME}: talon dispatch failed for "
                f"{params['repo']} ({params['mode']}): "
                f"{error or 'no error detail'} (fail closed)"
            )

        job_id = dispatch.get("job_id") or dispatch.get("task_id")
        result: Dict[str, Any] = {
            "source": SOURCE_NAME,
            "repo": params["repo"],
            "mode": params["mode"],
            "issue": params["issue"],
            "pr": params["pr"],
            "job_id": job_id,
            "method": dispatch.get("method"),
            # Dispatched is NOT merged/shipped (AGENTS.md evidence boundary).
            "state": "dispatched",
            "wait_ref": f"talon:{job_id}" if job_id else None,
            "log_path": dispatch.get("log_path"),
        }

        if not params["wait"] or not job_id:
            result["observation"] = (
                f"`{SOURCE_NAME}` reported `state: 'dispatched'` "
                f"(job_id={job_id!r})."
            )
            return result

        wait_result = await _await_completion(
            coordinator, str(job_id), params["wait_timeout_seconds"]
        )
        wait_data = dict(getattr(wait_result, "data", None) or {})
        status = wait_data.get("status")
        result.update(
            {
                "status": status,
                "returncode": wait_data.get("returncode"),
                "completed_at": wait_data.get("completed_at"),
                "waited_seconds": wait_data.get("waited_seconds"),
            }
        )
        pr_url = _find_pr_url(coordinator, str(job_id))
        if pr_url:
            result["pr_url"] = pr_url

        if status == "complete":
            result["state"] = "complete"
            result["observation"] = (
                f"`{SOURCE_NAME}` reported `status: 'complete'` "
                f"(job_id={job_id!r}"
                + (f", pr_url={pr_url!r}" if pr_url else "")
                + ")."
            )
            return result

        if wait_data.get("timed_out"):
            raise ValueError(
                f"{SOURCE_NAME}: talon job {job_id} still running after "
                f"{params['wait_timeout_seconds']}s; not claiming completion. "
                f"Watch it via wait('talon:{job_id}') (fail closed)"
            )
        raise ValueError(
            f"{SOURCE_NAME}: talon job {job_id} ended in "
            f"'{status or 'unknown'}' "
            f"(rc={wait_data.get('returncode')!r}); the pipeline run did "
            "not complete (fail closed)"
        )

    return handler


def _redact(payload: Dict[str, Any]) -> str:
    """Audit-log summary — identifiers only, no issue/PR body content."""
    return (
        f"{SOURCE_NAME} "
        f"repo={payload.get('repo', '?')} "
        f"issue={payload.get('issue', '?')} "
        f"pr={payload.get('pr', '?')} "
        f"mode={payload.get('mode', '?')}"
    )


def _schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Registry-time contract: dict payloads only.

    Full param validation happens in the handler so the stage's failure
    (raise -> stage FAIL) carries the precise fail-closed reason into the
    workflow run record rather than a generic schema drop.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"{SOURCE_NAME} payload must be a dict, got {type(payload).__name__}"
        )
    return payload


def build_talon_pipeline_dispatch_registration(
    coordinator: Any,
) -> SourceRegistration:
    """IRREVERSIBLE dispatch source — writes code, pushes branches, opens PRs.

    Mirrors ``a2a_repair_dispatch``'s registration posture (ACTION-only,
    TRUSTED, modest rate cap, identifier-only redaction); the SDK's
    ``SourceRegistration`` has no ``irreversible`` field, so — exactly like
    ``a2a_repair_dispatch`` — irreversibility is a documented property of
    the source that workflow authors must gate (consent/governance stages)
    before routing a stage here.
    """
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=make_talon_pipeline_dispatch_handler(coordinator),
        trust=Trust.TRUSTED,
        # One pipeline run is minutes-long; the cap only guards against a
        # pathological retry loop, never a legitimate workflow.
        rate_limit=RateLimit(per_minute=10, per_hour=60),
        resources=frozenset(),
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=30,
    )


def register_talon_pipeline_source(registry: Any, coordinator: Any) -> bool:
    """Register the source on ``registry`` under the OPTIONAL policy.

    Returns True only when this call *newly* registered the source. Returns
    False when the registry is absent, an equivalent registration already
    exists, validation failed, or an existing registration carries a
    *non-equivalent* contract. Under :attr:`RegistrationPolicy.OPTIONAL` (#2522)
    a non-equivalent clash is reported (logged loudly) rather than silently
    equated, and registration never raises — one bad source must not abort
    feature init.
    """
    if registry is None or not hasattr(registry, "register_with_policy"):
        return False
    outcome = registry.register_with_policy(
        build_talon_pipeline_dispatch_registration(coordinator),
        RegistrationPolicy.OPTIONAL,
    )
    return outcome.state is RegistrationState.REGISTERED
