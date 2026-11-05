"""
Individual health check functions for the Heartbeat Feature.

Each check function returns a dict with:
    name: str       - human-readable check name
    status: str     - "pass", "warn", or "fail"
    message: str    - description of the result
    duration_ms: float - wall-clock time for the check

Checks are designed to be fast and non-destructive.
Ordinary capability failures become diagnostics; original execution control
and native retirement debt propagate to the lifecycle owner without conversion.
"""

import asyncio
import logging
import shutil
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Dict, FrozenSet, List, Optional, Tuple

from kestrel_sdk.signals import ResourceLock
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.features.storage_access import (
    resolve_conversation_store,
    resolve_feature_database,
)
from kestrel_sovereign.signals import lock_manager as resource_lock_manager
from kestrel_sovereign.storage.consolidation_drain import (
    DRAIN_STATE_ISOLATED,
    HOLDER_CLOSED_CONNECTION,
    HOLDER_THIS_CONNECTION,
    reconcile_consolidation_drains,
)
from kestrel_sovereign.storage.footprint import measure_database_size
from kestrel_sovereign.storage.sync.quota import StorageQuota

logger = logging.getLogger(__name__)


# A liveness probe must return while the database operation it diagnoses is
# wedged.  Keep this below the feature's default 60-second probe cadence.
DATABASE_HEALTH_CHECK_TIMEOUT_S = 5.0


async def _run_database_bound_check(
    name: str,
    check: Awaitable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one database check within its own I/O deadline."""
    started = time.monotonic()
    deadline: asyncio.Timeout | None = None
    try:
        async with asyncio.timeout(
            DATABASE_HEALTH_CHECK_TIMEOUT_S
        ) as deadline:
            return await check
    except TimeoutError as error:
        if is_execution_control_error(error):
            raise
        # Do not relabel a TimeoutError raised by the check/provider itself as
        # this wrapper's deadline. This mirrors every other deadline boundary
        # on the consolidation and health paths.
        if deadline is None or not deadline.expired():
            raise
        label = name.replace("_", " ").capitalize()
        return {
            "name": name,
            "status": "fail",
            "message": (
                f"{label} health check timed out after "
                f"{DATABASE_HEALTH_CHECK_TIMEOUT_S:g} seconds"
            ),
            "duration_ms": _elapsed(started),
        }


def derive_overall_status(checks: List[Dict[str, Any]]) -> str:
    """Derive the shared three-state status for every detailed-health surface.

    - healthy: all checks pass
    - degraded: at least one warning or non-critical failure
    - unhealthy: a critical database, LLM, or scheduler-liveness check fails

    Both HealthFeature and the no-feature ``/health/detailed`` fallback use
    this function so installing the feature cannot change a check's severity.
    """
    statuses = [check.get("status", "pass") for check in checks]
    critical_names = {"database", "llm_service", "scheduler_liveness"}
    critical_statuses = [
        check.get("status", "pass")
        for check in checks
        if check.get("name") in critical_names
    ]

    if "fail" in critical_statuses:
        return "unhealthy"
    if "fail" in statuses or "warn" in statuses:
        return "degraded"
    return "healthy"


def worst_status(*statuses: str) -> str:
    """Combine severities within a SINGLE check, keeping the worst.

    A check whose status is written by more than one condition must never let a
    later, milder condition overwrite an earlier severe one — two facts sharing
    one variable means the last write silently downgrades the stronger. Callers
    use this instead of a bare assignment so the ordering of conditions cannot
    change the verdict.

    Unknown statuses sort as ``pass`` so a typo cannot manufacture a failure.
    """
    rank = {"pass": 0, "warn": 1, "fail": 2}
    worst = "pass"
    for status in statuses:
        if rank.get(status, 0) > rank[worst]:
            worst = status
    return worst


async def check_database(db) -> Dict[str, Any]:
    """Check database connectivity with a simple query.

    Args:
        db: AsyncDatabase instance (may be None)

    Returns:
        Check result dict
    """
    start = time.monotonic()

    if db is None:
        return {
            "name": "database",
            "status": "fail",
            "message": "No database connection available",
            "duration_ms": _elapsed(start),
        }

    backend = getattr(db, "backend", db)
    if getattr(backend, "write_connection_unavailable", False) is True:
        reconnect_required = (
            getattr(backend, "write_connection_requires_reconnect", False)
            is True
        )
        cleanup_deadline_exceeded = (
            getattr(
                backend,
                "write_connection_cleanup_deadline_exceeded",
                False,
            )
            is True
        )
        quarantined = (
            getattr(backend, "write_connection_quarantined", False) is True
        )
        cleanup_failed = reconnect_required or cleanup_deadline_exceeded
        return {
            "name": "database",
            # A live rollback handoff is transient and self-healing. It fences
            # reads and writes for correctness, but remains a warning only
            # inside its bounded cleanup window. Once that deadline expires,
            # or the owner of the abandoned work isolates the connection,
            # normal database operations are unavailable until cleanup ends
            # and the critical database check must fail even if cleanup may
            # eventually self-heal without reconnecting.
            "status": "fail" if cleanup_failed else "warn",
            "message": (
                "Database write connection is unavailable because cancellation "
                + (
                    "cleanup failed; reconnect is required"
                    if reconnect_required
                    else (
                        "cleanup is still running and the connection is "
                        "isolated until it finishes; normal reads and writes "
                        "are unavailable"
                        if quarantined
                        else (
                            "cleanup exceeded its deadline; normal reads and "
                            "writes are unavailable"
                            if cleanup_deadline_exceeded
                            else "cleanup is still pending"
                        )
                    )
                )
            ),
            "duration_ms": _elapsed(start),
        }

    try:
        # Simple read query to verify the connection is alive
        row = await db.fetchone("SELECT 1")
        if row and row[0] == 1:
            return {
                "name": "database",
                "status": "pass",
                "message": "Database connection healthy",
                "duration_ms": _elapsed(start),
            }
        return {
            "name": "database",
            "status": "warn",
            "message": "Database returned unexpected result",
            "duration_ms": _elapsed(start),
        }
    except Exception as e:
        if is_execution_control_error(e):
            raise
        return {
            "name": "database",
            "status": "fail",
            "message": f"Database query failed: {e}",
            "duration_ms": _elapsed(start),
        }


# Route evidence for ``llm_service`` and ``model_discovery`` (#3912) is the
# per-call ledger LLMService already writes for every chat, generate and
# streaming attempt (``a2a_llm_calls``, #2236). Health reads what dispatch did
# rather than re-deriving routing, and never sends a completion to probe a
# route. A (provider, model) pair with a row in the window is active, and its
# health is its newest outcome.
#
# A route is reported ok or standby, and a startup failure set aside, only on
# positive evidence. A lease-route row naming no lease or another lease, a
# capped read, a selected route with no rows, and a call that cannot be placed
# after the startup probe each leave a route unprobed or warning, never
# passing. An installed lease that refuses calls now warns whatever its rows
# say: it refuses some calls before they are recorded.
LLM_ROUTE_EVIDENCE_WINDOW = timedelta(hours=24)
# The newest rows read for each route, and for its failures when that first
# read is full (see :func:`_read_route_rows`).
LLM_ROUTE_EVIDENCE_ROW_LIMIT = 200

# What one route's evidence says.
ROUTE_OK = "ok"
ROUTE_FAILING = "failing"
ROUTE_DISABLED = "disabled"
# The installed lease route refuses calls now: its lease expired, or the
# route is not accepting calls. Such a refusal happens before the call is
# recorded, so no row shows it.
ROUTE_UNAVAILABLE = "unavailable"
# The installed lease route's newest call on a model names no lease, and
# failed: it may be this lease's. The store keeps the lease id past the
# privacy gate, so only a row written before it did names none.
ROUTE_UNATTRIBUTED_FAILING = "unattributed_failing"
# The route failed more calls in the window than were read, so a model whose
# newest call failed may lie beyond them.
ROUTE_TRUNCATED = "truncated"
# Failed its startup reachability probe, with no call known to have been made
# after that probe (see :func:`_startup_probe_seen_at`).
ROUTE_UNREACHABLE = "unreachable"
ROUTE_UNPROBED = "unprobed"
ROUTE_STANDBY = "standby"
ROUTE_UNKNOWN = "unknown"

_ROUTE_STATE_LABELS = (
    (ROUTE_DISABLED, "disabled after a failure"),
    (ROUTE_UNAVAILABLE, "refusing calls now"),
    (ROUTE_FAILING, "failing (newest call failed)"),
    (
        ROUTE_UNATTRIBUTED_FAILING,
        "failing, lease unattributed (newest call failed and names no lease)",
    ),
    (
        ROUTE_TRUNCATED,
        f"failures beyond the {LLM_ROUTE_EVIDENCE_ROW_LIMIT} rows read",
    ),
    (ROUTE_UNREACHABLE, "unreachable at startup, no call known since"),
    (ROUTE_OK, "ok"),
    (ROUTE_UNPROBED, "active route unprobed"),
    (ROUTE_STANDBY, "standby"),
    (ROUTE_UNKNOWN, "unclassified"),
)
_DEGRADING_ROUTE_STATES = frozenset({
    ROUTE_FAILING,
    ROUTE_DISABLED,
    ROUTE_UNAVAILABLE,
    ROUTE_UNATTRIBUTED_FAILING,
    ROUTE_TRUNCATED,
    ROUTE_UNREACHABLE,
})


# When a health check first saw each LLM service's startup probe result,
# with that result (see :func:`_startup_probe_seen_at`). Keyed weakly by the
# service: an agent reloaded in this process brings a new service, and so
# starts over, and a retired one is forgotten with it.
_STARTUP_PROBES_SEEN: "weakref.WeakKeyDictionary[Any, Tuple[list, datetime]]" = (
    weakref.WeakKeyDictionary()
)


def _startup_probe_seen_at(llm_service, reachability) -> Optional[datetime]:
    """When a health check first saw ``reachability``, ``llm_service``'s
    current startup probe result. ``None`` when that cannot be tracked.

    The probe stamps no time, and neither the process nor the agent records
    when it ran: an agent can be loaded, and so probed, long after this
    process started. Each run assigns a new list, so the first time a check
    sees a list, the probe that built it has finished. A call stamped after
    that was made after the probe; an earlier call, even one this same
    service made, cannot be placed after it.
    """
    if not isinstance(reachability, list):
        return None
    try:
        seen = _STARTUP_PROBES_SEEN.get(llm_service)
        if seen is None or seen[0] is not reachability:
            seen = (reachability, datetime.now(timezone.utc))
            _STARTUP_PROBES_SEEN[llm_service] = seen
    except TypeError:
        # The service cannot be held weakly, or hashed: the probe cannot be
        # placed, so it stands.
        return None
    return seen[1]


def _utc(timestamp: Any) -> Optional[datetime]:
    if not isinstance(timestamp, datetime):
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp


@dataclass(frozen=True)
class _RouteRows:
    """One route's rows in the window, newest first, as read."""

    rows: Tuple[Any, ...]
    capped: bool
    # Only when ``rows`` is capped: the route's newest failures, whether that
    # read was capped too, and the newest call of each model they name that
    # ``rows`` does not hold.
    failures: Tuple[Any, ...] = ()
    failures_capped: bool = False
    late: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class _CallEvidence:
    """The ledger rows read for one agent's routes."""

    available: bool
    # Why there is no evidence: a fixed phrase or an exception class name,
    # never exception text (#3919, #3920).
    unavailable_reason: Optional[str] = None
    # The read was attempted and failed, as opposed to there being no ledger.
    unreadable: bool = False
    by_route: Optional[Dict[str, _RouteRows]] = None

    def evidence_details(self) -> Dict[str, Any]:
        by_route = self.by_route or {}
        return {
            "source": "a2a_llm_calls",
            "window_seconds": int(LLM_ROUTE_EVIDENCE_WINDOW.total_seconds()),
            "row_limit": LLM_ROUTE_EVIDENCE_ROW_LIMIT,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "rows_read": sum(len(read.rows) for read in by_route.values()),
            "row_limit_reached": sorted(
                name for name, read in by_route.items() if read.capped
            ),
        }


async def _read_route_rows(store, agent_did: str, provider: str, since, *, relevant):
    """Read one route's newest rows; when they fill the limit, its failures too.

    A full read says nothing of the rows beyond it. A model whose newest call
    failed may be among them, so the route's newest failures are read as
    well, and for each model they name that the first read does not hold,
    that model's newest call. ``relevant`` picks the failures that can speak
    for the route (the lease route ignores another lease's).
    """
    limit = LLM_ROUTE_EVIDENCE_ROW_LIMIT
    rows = await store.query_llm_calls(
        agent_did=agent_did, provider=provider, since=since, limit=limit
    )
    if len(rows) < limit:
        return _RouteRows(rows=tuple(rows), capped=False)
    failures = await store.query_llm_calls(
        agent_did=agent_did, provider=provider, success=False, since=since,
        limit=limit,
    )
    read_models = {row.model for row in rows}
    late: List[Any] = []
    for model in dict.fromkeys(f.model for f in failures if relevant(f)):
        # The store drops an empty model filter, so such a failure is judged
        # as the model's newest call.
        if model in read_models or not model:
            continue
        late.extend(await store.query_llm_calls(
            agent_did=agent_did, provider=provider, model=model, since=since,
            limit=1,
        ))
    return _RouteRows(
        rows=tuple(rows),
        capped=True,
        failures=tuple(failures),
        failures_capped=len(failures) >= limit,
        late=tuple(late),
    )


def _lease_named_by(call) -> Optional[str]:
    metadata = call.metadata if isinstance(call.metadata, dict) else {}
    lease_id = metadata.get("inference_lease_id")
    return lease_id if isinstance(lease_id, str) and lease_id else None


def _row_owner(private_lease_id: Optional[str]):
    """``(owns, unattributed)`` predicates over one route's rows.

    A provider's rows are all its own. The lease route's rows are the
    installed lease's only when they name it exactly; a row naming another
    lease is never counted, and a row naming no lease is unattributed.
    """
    if private_lease_id is None:
        return (lambda call: True), (lambda call: False)
    return (
        lambda call: _lease_named_by(call) == private_lease_id,
        lambda call: _lease_named_by(call) is None,
    )


async def _read_call_evidence(agent, route_names, private_lease_id) -> _CallEvidence:
    """Read each route's recent ledger rows (see :func:`_read_route_rows`).

    Each route is read on its own, so one busy route's rows never crowd
    another's out of a shared limit. Only ``provider``, ``model``,
    ``success``, ``timestamp`` and ``metadata.inference_lease_id`` are judged
    off a row; prompt, response and error text never leave this module.
    """
    from kestrel_sovereign.a2a.stores.unified.observability_store import (
        ObservabilityStore,
    )
    from kestrel_sovereign.features.storage_access import (
        AgentIdentityUnavailable,
        resolve_scoped_agent_did,
    )
    from kestrel_sovereign.llm.remote_backend import REMOTE_ROUTE_NAME

    store = getattr(agent, "observability_store", None)
    if not isinstance(store, ObservabilityStore):
        return _CallEvidence(available=False, unavailable_reason="no call ledger")
    try:
        agent_did = resolve_scoped_agent_did(agent)
    except AgentIdentityUnavailable:
        return _CallEvidence(
            available=False, unavailable_reason="agent identity unavailable"
        )

    since = datetime.now(timezone.utc) - LLM_ROUTE_EVIDENCE_WINDOW
    by_route: Dict[str, _RouteRows] = {}
    deadline: asyncio.Timeout | None = None
    try:
        async with asyncio.timeout(DATABASE_HEALTH_CHECK_TIMEOUT_S) as deadline:
            for name in route_names:
                owns, unattributed = _row_owner(
                    private_lease_id if name == REMOTE_ROUTE_NAME else None
                )
                by_route[name] = await _read_route_rows(
                    store, agent_did, name, since,
                    relevant=lambda call: owns(call) or unattributed(call),
                )
    except Exception as error:
        if is_execution_control_error(error):
            raise
        expired = (
            isinstance(error, TimeoutError)
            and deadline is not None
            and deadline.expired()
        )
        return _CallEvidence(
            available=False,
            unavailable_reason=(
                f"read timed out after {DATABASE_HEALTH_CHECK_TIMEOUT_S:g} seconds"
                if expired
                else type(error).__name__
            ),
            unreadable=True,
        )
    return _CallEvidence(available=True, by_route=by_route)


def _model_stats(newest: Dict[str, Any], rows) -> List[Dict[str, Any]]:
    """Each model's newest outcome, with its calls and failures among ``rows``."""
    stats = []
    for model, call in newest.items():
        timestamp = _utc(call.timestamp)
        counted = [row for row in rows if row.model == model]
        stats.append({
            "model": model,
            "calls": len(counted),
            "failures": sum(1 for row in counted if not row.success),
            "last_call_at": timestamp.isoformat() if timestamp else None,
            "last_call_succeeded": bool(call.success),
        })
    return stats


@dataclass(frozen=True)
class _RouteJudgement:
    """What one route's rows say, before configuration and the probe."""

    state: Optional[str]  # None: no row of the route's own speaks for it
    models: List[Dict[str, Any]]
    unattributed_models: List[Dict[str, Any]]
    newest_call_at: Optional[datetime]


def _supersedes(call, current) -> bool:
    """Whether ``call`` is newer than ``current``, the same model's call.

    By timestamp, not by the order the reads returned them: a call written
    between two reads is newer than every row the first one holds. A tie, or
    a call with no timestamp, goes to the failure: neither can show the
    success came after it.
    """
    call_at, current_at = _utc(call.timestamp), _utc(current.timestamp)
    if call_at is not None and current_at is not None and call_at != current_at:
        return call_at > current_at
    return not call.success and bool(current.success)


def _keep_newest(newest: Dict[str, Any], call) -> None:
    current = newest.get(call.model)
    if current is None or _supersedes(call, current):
        newest[call.model] = call


def _judge_route_rows(read: _RouteRows, private_lease_id: Optional[str]) -> _RouteJudgement:
    """Judge each model by its newest own call, and the route by its models.

    Each model's newest call is chosen by timestamp across every read (see
    :func:`_supersedes`). A model the first read does not hold is judged by
    its newest call when that is the route's own, else by its newest own
    failure: no newer own call can be placed after it. For the lease route,
    an unattributed failure stands unless a newer call names the installed
    lease (ties stand); an unattributed success never sets it aside, nor
    speaks for the lease.
    """
    owns, unattributed = _row_owner(private_lease_id)
    newest_own: Dict[str, Any] = {}
    newest_unattributed: Dict[str, Any] = {}
    unattributed_failure: Dict[str, Any] = {}
    for call in (*read.rows, *read.failures, *read.late):
        if owns(call):
            _keep_newest(newest_own, call)
        elif unattributed(call):
            _keep_newest(newest_unattributed, call)
            if not call.success:
                _keep_newest(unattributed_failure, call)

    def stands(model: str, failure) -> bool:
        own = newest_own.get(model)
        if own is None:
            return True
        failed_at, own_at = _utc(failure.timestamp), _utc(own.timestamp)
        return failed_at is None or own_at is None or failed_at >= own_at

    seen = {row.event_id: row for row in (*read.rows, *read.failures, *read.late)}
    own_rows = [row for row in seen.values() if owns(row)]
    unattributed_rows = [row for row in seen.values() if unattributed(row)]
    own_times = [t for t in (_utc(c.timestamp) for c in newest_own.values()) if t]

    if any(not call.success for call in newest_own.values()):
        state = ROUTE_FAILING
    elif any(stands(model, f) for model, f in unattributed_failure.items()):
        state = ROUTE_UNATTRIBUTED_FAILING
    elif read.failures_capped:
        state = ROUTE_TRUNCATED
    elif newest_own:
        state = ROUTE_OK
    else:
        state = None
    return _RouteJudgement(
        state=state,
        models=_model_stats(newest_own, own_rows),
        unattributed_models=_model_stats(newest_unattributed, unattributed_rows),
        newest_call_at=max(own_times) if own_times else None,
    )


@dataclass(frozen=True)
class LlmRouteAssessment:
    """Each configured route's state, read from the call ledger (#3912)."""

    routes: Tuple[Dict[str, Any], ...]
    evidence: Dict[str, Any]
    ledger_unreadable: bool
    private_route_installed: bool

    @property
    def degraded_routes(self) -> List[str]:
        return [
            route["name"]
            for route in self.routes
            if route["state"] in _DEGRADING_ROUTE_STATES
        ]

    def standby_vendors(self) -> FrozenSet[str]:
        """Vendors every configured route of which is standby."""
        states: Dict[str, List[str]] = {}
        for route in self.routes:
            if route.get("private_inference_lease"):
                continue
            states.setdefault(route["vendor"], []).append(route["state"])
        return frozenset(
            vendor
            for vendor, vendor_states in states.items()
            if vendor_states and all(s == ROUTE_STANDBY for s in vendor_states)
        )

    def summary(self) -> str:
        groups = []
        for state, label in _ROUTE_STATE_LABELS:
            names = [
                f"{r['name']} ({r['refusal']})" if r.get("refusal") else r["name"]
                for r in self.routes
                if r["state"] == state
            ]
            if names:
                groups.append(f"{label}: {', '.join(names)}")
        text = f" — routes: {'; '.join(groups)}" if groups else ""
        reason = self.evidence.get("unavailable_reason")
        if self.ledger_unreadable:
            text += f" (call ledger unreadable: {reason})"
        elif not self.evidence.get("available"):
            text += f" (no call evidence: {reason})"
        return text


def _installed_private_lease_id(llm_service) -> Optional[str]:
    """The lease id of the private inference route, when one is installed."""
    lease = getattr(llm_service, "_remote_lease", None)
    lease_id = getattr(lease, "lease_id", None) if lease is not None else None
    return lease_id if isinstance(lease_id, str) and lease_id else None


def _private_route_refusal(llm_service) -> Optional[str]:
    """Why the installed lease route refuses every call now, or ``None``.

    The lease route refuses an expired lease, and a route that is not
    accepting calls (draining, released, or latched off after expiry),
    before the call is recorded, so the ledger never shows those refusals.
    This reads the same state: the service's backend status and the lease's
    expiry. A fixed phrase, never provider text. Availability that cannot be
    read is not shown, so it is a refusal too.
    """
    lease = getattr(llm_service, "_remote_lease", None)
    expires_at = _utc(getattr(lease, "expires_at", None))
    if expires_at is None:
        return "lease expiry unknown"
    if datetime.now(timezone.utc) >= expires_at:
        return "lease expired"
    backend_status = getattr(llm_service, "get_backend_status", None)
    try:
        status = backend_status() if callable(backend_status) else None
    except Exception as error:
        if is_execution_control_error(error):
            raise
        status = None
    if not isinstance(status, dict):
        return "route status unreadable"
    if status.get("remote_active") is not True:
        return "not accepting calls"
    return None


def _provider_vendor_and_route(provider: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    name = provider["name"]
    vendor, _, route = name.partition(":")
    provider_vendor = provider.get("vendor")
    provider_route = provider.get("route")
    return (
        provider_vendor if isinstance(provider_vendor, str) and provider_vendor else vendor,
        provider_route if isinstance(provider_route, str) and provider_route else (route or None),
    )


def _named_selections(llm_service) -> Tuple[List[Dict[str, Optional[str]]], bool]:
    """Every provider selection the agent's model configuration names, and
    whether that is all of them.

    The current selection (the mandate, else the default route), every
    selector ``model_mandate.toml`` names (``defaults.preferred``, each
    mandate, an explicit cheap model), the cheap model its hints pick, and
    each mandate fallback naming a provider. Each is read and parsed by the
    service's own helpers; nothing here resolves routing. Not complete when
    the configuration cannot be read, or a selector names no provider (a bare
    model id is tried on every route): then no route can be shown standby.
    """
    from kestrel_sovereign.llm.provider_policy import fallback_selection
    from kestrel_sovereign.llm.service import resolve_active_model_selection

    selections: List[Dict[str, Optional[str]]] = []
    try:
        current = resolve_active_model_selection(llm_service)
        if isinstance(current.get("vendor"), str) and current["vendor"]:
            selections.append({"vendor": current["vendor"], "route": current.get("route")})
        config_selectors = getattr(llm_service, "_config_selectors", None)
        select = getattr(llm_service, "_call_override_selection", None)
        resolve = getattr(llm_service, "_resolve_model_selector", None)
        if not (callable(config_selectors) and callable(select) and callable(resolve)):
            return selections, False
        complete = True
        selectors = list(config_selectors())
        cheap = getattr(llm_service, "get_cheap_model_selector", None)
        cheap_selector = cheap() if callable(cheap) else None
        if isinstance(cheap_selector, str) and cheap_selector:
            selectors.append(cheap_selector)
        for selector in selectors:
            selection = select(selector)
            if selection is None:
                resolved = resolve(selector).get("selector")
                selection = select(resolved) if resolved != selector else None
            if selection is None:
                complete = False
            else:
                selections.append(selection)
        for fallback in getattr(llm_service, "_mandate_fallbacks", None) or []:
            # A fallback naming no provider is never followed.
            selection = fallback_selection(fallback)
            if selection is not None:
                selections.append(selection)
    except Exception as error:
        if is_execution_control_error(error):
            raise
        return selections, False
    return selections, complete


def _selected_route_names(
    llm_service, providers: List[Dict[str, Any]]
) -> Tuple[set, bool]:
    """The configured routes the agent's model configuration names, and
    whether every other route is known to be named by nothing (see
    :func:`_named_selections`).
    """
    from kestrel_sovereign.llm.provider_policy import selection_covers

    selections, complete = _named_selections(llm_service)
    named = set()
    for provider in providers:
        vendor, route = _provider_vendor_and_route(provider)
        if any(selection_covers(s, vendor, route) for s in selections):
            named.add(provider["name"])
    return named, complete


def _startup_unreachable(llm_service) -> set:
    """Routes whose startup reachability probe failed."""
    reachability = getattr(llm_service, "reachability", None)
    if not isinstance(reachability, list):
        return set()
    return {
        entry["name"]
        for entry in reachability
        if isinstance(entry, dict)
        and entry.get("status") == "unreachable"
        and isinstance(entry.get("name"), str)
    }


def _route_state(
    *,
    judgement: Optional[_RouteJudgement],
    disabled: bool,
    selected: Optional[bool],
    evidence_available: bool,
    unreachable_at_startup: bool,
    probe_seen_at: Optional[datetime],
    refusing_calls: bool,
) -> str:
    """One route's state. ``selected`` is ``None`` when selection is unknown.

    ``probe_seen_at`` is when the startup probe result was first seen (see
    :func:`_startup_probe_seen_at`); ``refusing_calls`` is an installed lease
    route's current refusal (see :func:`_private_route_refusal`).
    """
    if disabled:
        # A route disabled after a failure is failing whatever its rows say.
        return ROUTE_DISABLED
    if refusing_calls:
        # So is a route refusing calls now: those refusals leave no row.
        return ROUTE_UNAVAILABLE
    state = judgement.state if judgement is not None else None
    if state in _DEGRADING_ROUTE_STATES:
        return state
    if state is None:
        if selected:
            state = ROUTE_UNPROBED
        elif not evidence_available or selected is None:
            state = ROUTE_UNKNOWN
        else:
            # Nothing selects it and it served nothing: its startup probe
            # speaks for no route this agent uses.
            return ROUTE_STANDBY
    if unreachable_at_startup:
        # Only a call stamped after the probe result was first seen is known
        # to be newer than the probe.
        newest = judgement.newest_call_at if judgement is not None else None
        if probe_seen_at is None or newest is None or newest <= probe_seen_at:
            return ROUTE_UNREACHABLE
    return state


async def assess_llm_routes(agent) -> Optional[LlmRouteAssessment]:
    """Classify every configured route, and an installed lease route, by its
    ledger evidence. ``None`` when the agent has no LLM service.

    - ``disabled``: disabled after a permanent failure; degrades.
    - ``failing``: a model's newest call in the window failed; degrades.
    - ``unattributed_failing``: the installed lease route's newest call on a
      model names no lease and failed, with no newer call of its own; degrades.
    - ``truncated``: more failures than were read; degrades.
    - ``unavailable``: the installed lease route refuses calls now (its lease
      expired, or it is not accepting calls); degrades.
    - ``unreachable``: failed its startup probe, with no call known to have
      been made after it (see :func:`_startup_probe_seen_at`), and is not
      standby; degrades.
    - ``ok``: every model's newest call in the window succeeded.
    - ``unprobed``: named by the agent's model configuration, or the installed
      private lease, with no call in the window ("active route unprobed").
    - ``standby``: configured, named by nothing, no call in the window.
    - ``unknown``: no evidence could be read, or the configuration could not
      be placed on routes, so standby cannot be claimed.
    """
    llm_service = getattr(agent, "llm_service", None)
    if llm_service is None:
        return None
    from kestrel_sovereign.llm.remote_backend import REMOTE_ROUTE_NAME

    providers = [
        provider
        for provider in (getattr(llm_service, "providers", None) or [])
        if isinstance(provider, dict)
        and isinstance(provider.get("name"), str)
        and provider["name"]
    ]
    disabled_routes = getattr(llm_service, "_disabled_routes", None)
    # The keys only: a disabled route's recorded reason is error text.
    disabled = (
        {name for name in disabled_routes if isinstance(name, str)}
        if isinstance(disabled_routes, dict)
        else set()
    )
    lease_id = _installed_private_lease_id(llm_service)
    # Read with the lease id, before any await, so both describe one lease.
    lease_refusal = (
        _private_route_refusal(llm_service) if lease_id is not None else None
    )
    selected, selection_complete = _selected_route_names(llm_service, providers)
    unreachable = _startup_unreachable(llm_service)
    # Before the ledger read: every row stamped after this was written after
    # the probe finished.
    probe_seen_at = _startup_probe_seen_at(
        llm_service, getattr(llm_service, "reachability", None)
    )
    route_names = [provider["name"] for provider in providers]
    if lease_id is not None:
        route_names.append(REMOTE_ROUTE_NAME)
    evidence = await _read_call_evidence(agent, route_names, lease_id)
    by_route = evidence.by_route or {}

    routes: List[Dict[str, Any]] = []
    for provider in providers:
        name = provider["name"]
        vendor, _route = _provider_vendor_and_route(provider)
        read = by_route.get(name)
        judgement = _judge_route_rows(read, None) if read is not None else None
        # Named, named by nothing, or not known to be named by nothing.
        named = True if name in selected else (False if selection_complete else None)
        routes.append({
            "name": name,
            "vendor": vendor,
            "state": _route_state(
                judgement=judgement,
                disabled=name in disabled,
                selected=named,
                evidence_available=evidence.available,
                unreachable_at_startup=name in unreachable,
                probe_seen_at=probe_seen_at,
                refusing_calls=False,
            ),
            "configured_route": named,
            "models": judgement.models if judgement is not None else [],
            "row_limit_reached": bool(read is not None and read.capped),
        })
    if lease_id is not None:
        # An installed lease is a configured route by definition: it is never
        # standby, and it has no startup probe.
        read = by_route.get(REMOTE_ROUTE_NAME)
        judgement = _judge_route_rows(read, lease_id) if read is not None else None
        routes.append({
            "name": REMOTE_ROUTE_NAME,
            "vendor": REMOTE_ROUTE_NAME,
            "state": _route_state(
                judgement=judgement,
                disabled=REMOTE_ROUTE_NAME in disabled,
                selected=True,
                evidence_available=evidence.available,
                unreachable_at_startup=False,
                probe_seen_at=None,
                refusing_calls=lease_refusal is not None,
            ),
            "configured_route": True,
            "private_inference_lease": True,
            # A fixed phrase (see _private_route_refusal), or None.
            "refusal": lease_refusal,
            "models": judgement.models if judgement is not None else [],
            "unattributed_models": (
                judgement.unattributed_models if judgement is not None else []
            ),
            "row_limit_reached": bool(read is not None and read.capped),
        })
    return LlmRouteAssessment(
        routes=tuple(routes),
        evidence=evidence.evidence_details(),
        ledger_unreadable=evidence.unreadable,
        private_route_installed=lease_id is not None,
    )


async def check_llm_service(agent) -> Dict[str, Any]:
    """Verify that an LLM provider is configured, and report each route's
    health from the calls it actually served.

    This does NOT make an LLM call. Route health is read from the per-call
    ledger (see :func:`assess_llm_routes`): a route whose newest call failed,
    or that was disabled after a failure, warns; a configured route nothing
    names is ``standby`` and does not degrade the check, nor does its startup
    reachability. A failed startup reachability probe warns until its route
    serves a call known to be newer than the probe. An installed private
    lease that refuses calls now warns, whatever its rows say.

    Args:
        agent: KestrelAgent instance

    Returns:
        Check result dict
    """
    start = time.monotonic()

    llm_service = getattr(agent, "llm_service", None)
    if llm_service is None:
        return {
            "name": "llm_service",
            "status": "fail",
            "message": "No LLM service configured",
            "duration_ms": _elapsed(start),
        }

    try:
        # Check that at least one provider is available
        providers = getattr(llm_service, "providers", None) or []

        # Resolve the active model via the canonical routing source so the
        # heartbeat reports what the agent will *actually* use, not just
        # what was configured or defaulted.
        active_model = None
        if hasattr(llm_service, "get_active_model_id"):
            active_model = llm_service.get_active_model_id()
            if active_model == "auto":
                active_model = None

        # Also surface the vendor (and route, if set) from mandate preference.
        active_vendor = None
        active_route = None
        if hasattr(llm_service, "get_model_preference"):
            pref = llm_service.get_model_preference()
            active_vendor = pref.get("vendor")
            active_route = pref.get("route")

        # Coverage runs before, and independently of, any provider gate: a
        # service whose only route is a private inference lease still has a
        # route to judge (#3912).
        routes = await assess_llm_routes(agent)

        if not providers and not routes.private_route_installed:
            return {
                "name": "llm_service",
                "status": "warn",
                "message": "LLM service exists but no providers initialized",
                "details": {
                    "routes": list(routes.routes),
                    "route_evidence": routes.evidence,
                },
                "duration_ms": _elapsed(start),
            }

        if providers:
            names = [p.get("name", p.get("vendor", "?")) for p in providers[:3]]
            msg = f"LLM providers available: {', '.join(names)}"
            if len(providers) > 3:
                msg += f" (+{len(providers) - 3} more)"
        else:
            msg = (
                "No configured LLM providers; a private inference route is "
                "installed"
            )
        if active_model:
            if active_vendor and active_route:
                model_label = f"{active_vendor}:{active_route}/{active_model}"
            elif active_vendor:
                model_label = f"{active_vendor}/{active_model}"
            else:
                model_label = active_model
            msg += f" (active: {model_label})"
        reachability = getattr(llm_service, "reachability", None)
        if not isinstance(reachability, list):
            reachability = None

        # Severity is written by more than one condition below, so each
        # verdict is combined through worst_status() rather than assigned —
        # a plain assignment lets a later, milder condition downgrade an
        # earlier severe one.
        status = "pass"

        details: Dict[str, Any] = {
            "routes": list(routes.routes),
            "route_evidence": routes.evidence,
        }
        if reachability:
            details["reachability"] = reachability

        # A failed startup probe is part of its route's state (#3912): it
        # stands until a call known to be newer than the probe, unless the
        # route is standby. See assess_llm_routes.
        if routes.degraded_routes or routes.ledger_unreadable:
            status = worst_status(status, "warn")
        msg += routes.summary()

        # A persisted mandate that failed to apply (#3190): the operator set
        # a model and it is NOT in effect, so this agent is running on
        # whatever route_priority picks. Previously this state existed only
        # as one WARNING line at boot.
        # Require a real string: ``getattr`` on a test double returns a
        # truthy stub, which would report every mocked service as UNPINNED
        # and then fail to JSON-serialise when the heartbeat is persisted.
        mandate_load_error = getattr(llm_service, "_mandate_load_error", None)
        if not isinstance(mandate_load_error, str):
            mandate_load_error = None
        if mandate_load_error:
            status = worst_status(status, "warn")
            details["mandate_load_error"] = mandate_load_error
            msg += (
                " — a persisted model preference failed to apply; this "
                f"agent is running UNPINNED ({mandate_load_error})"
            )

        return {
            "name": "llm_service",
            "status": status,
            "message": msg,
            "details": details,
            "duration_ms": _elapsed(start),
        }
    except Exception as e:
        if is_execution_control_error(e):
            raise
        return {
            "name": "llm_service",
            "status": "fail",
            "message": f"LLM service check failed: {e}",
            "duration_ms": _elapsed(start),
        }


async def check_memory_system(agent) -> Dict[str, Any]:
    """Verify that the memory retriever and consolidator are accessible.

    Args:
        agent: KestrelAgent instance

    Returns:
        Check result dict
    """
    start = time.monotonic()

    try:
        storage = getattr(agent, "storage", None)
        if storage is None:
            return {
                "name": "memory_system",
                "status": "fail",
                "message": "No storage system available",
                "duration_ms": _elapsed(start),
            }

        # Check for memory retriever
        retriever = getattr(storage, "retriever", None) or getattr(
            agent, "memory_retriever", None
        )
        # Check for memory consolidator
        consolidator = getattr(storage, "consolidator", None) or getattr(
            agent, "memory_consolidator", None
        )

        components = []
        if retriever:
            components.append("retriever")
        if consolidator:
            components.append("consolidator")

        db = resolve_feature_database(agent)
        if db:
            components.append("database")

        if not components:
            return {
                "name": "memory_system",
                "status": "warn",
                "message": "Storage exists but no memory components found",
                "duration_ms": _elapsed(start),
            }

        drain, settled_drain = _consolidation_drain_check(db)
        if drain is not None:
            return {
                "name": "memory_system",
                **drain,
                "duration_ms": _elapsed(start),
            }

        result = {
            "name": "memory_system",
            "status": "pass",
            "message": f"Memory system components: {', '.join(components)}",
            "duration_ms": _elapsed(start),
        }
        if settled_drain is not None:
            # History, not a fault: the last timed-out pass's work settled.
            result["details"] = {"consolidation_drain": settled_drain}
        return result
    except Exception as e:
        if is_execution_control_error(e):
            raise
        return {
            "name": "memory_system",
            "status": "fail",
            "message": f"Memory system check failed: {e}",
            "duration_ms": _elapsed(start),
        }


def _consolidation_drain_check(
    db,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Report every timed-out consolidation pass whose database work is unsettled.

    Reads the durable records timed-out passes left beside the database file
    (#2978), never the database: while that work runs the database refuses or
    delays operations, and this check must answer regardless. Each pass keeps
    its own record, and every one is read: one pass settling says nothing
    about another. A record whose owner has exited without settling it is
    reported until an operator deletes it, never cleared here.

    Returns ``(concern, settled)``: a status/message/details to report while
    any record is unsettled or unreadable, else the most recent settled
    record, kept as history. Both ``None`` when there are no records.
    """
    report = reconcile_consolidation_drains(db)
    if report is None:
        return None, None

    concerns = []
    for status in report.unsettled:
        # This agent's database refuses all work: its connection is isolated,
        # or was closed under work that only the worker thread's exit ends.
        refusing = status.holder == HOLDER_CLOSED_CONNECTION or (
            status.holder == HOLDER_THIS_CONNECTION
            and status.state == DRAIN_STATE_ISOLATED
        )
        concerns.append(("fail" if refusing else "warn", status.describe()))
    for unreadable in report.unreadable:
        concerns.append(("warn", unreadable.describe()))
    if not concerns:
        latest = report.latest_settled
        return None, None if latest is None else latest.record

    severity = worst_status(*(level for level, _ in concerns))
    # The most severe first; the rest are counted here and listed in details.
    message = next(text for level, text in concerns if level == severity)
    if len(concerns) > 1:
        message = f"{message} (and {len(concerns) - 1} more; see details)"
    return {
        "status": severity,
        "message": message,
        "details": {
            "consolidation_drains": [
                {
                    "record_path": str(status.path),
                    "holder": status.holder,
                    "record": status.record,
                }
                for status in report.unsettled
            ],
            "unreadable_drain_records": [
                {"record_path": str(unreadable.path), "error": unreadable.error}
                for unreadable in report.unreadable
            ],
        },
    }, None


async def check_disk_space(threshold_mb: int = 100) -> Dict[str, Any]:
    """Check available disk space on the root filesystem.

    Args:
        threshold_mb: Minimum free MB before warning (default 100)

    Returns:
        Check result dict
    """
    start = time.monotonic()

    try:
        usage = shutil.disk_usage("/")
        free_mb = usage.free / (1024 * 1024)
        total_mb = usage.total / (1024 * 1024)
        used_pct = (usage.used / usage.total) * 100 if usage.total > 0 else 0

        if free_mb < threshold_mb:
            return {
                "name": "disk_space",
                "status": "warn" if free_mb > threshold_mb / 2 else "fail",
                "message": (
                    f"Low disk space: {free_mb:.0f}MB free "
                    f"({used_pct:.1f}% used of {total_mb:.0f}MB)"
                ),
                "duration_ms": _elapsed(start),
            }

        return {
            "name": "disk_space",
            "status": "pass",
            "message": (
                f"Disk space OK: {free_mb:.0f}MB free "
                f"({used_pct:.1f}% used of {total_mb:.0f}MB)"
            ),
            "duration_ms": _elapsed(start),
        }
    except Exception as e:
        if is_execution_control_error(e):
            raise
        return {
            "name": "disk_space",
            "status": "fail",
            "message": f"Disk space check failed: {e}",
            "duration_ms": _elapsed(start),
        }


async def check_database_size(db) -> Dict[str, Any]:
    """Report how large the agent database is (#3785).

    Report-only: a large database neither warns nor fails the probe. It is
    measured every tick so the agent, and anyone reading its health history,
    can watch the file grow; Meridian's reached 1.89 GB before anyone looked.
    ``free pages`` is space only ``VACUUM`` returns to the filesystem.

    Args:
        db: AsyncDatabase instance (may be None)
    """
    start = time.monotonic()

    if db is None:
        # The database check already fails for a missing connection.
        return {
            "name": "database_size",
            "status": "pass",
            "message": "No database connection to measure",
            "duration_ms": _elapsed(start),
        }

    try:
        size = await measure_database_size(db)
    except Exception as e:
        if is_execution_control_error(e):
            raise
        return {
            "name": "database_size",
            "status": "warn",
            "message": f"Database size could not be measured: {e}",
            "duration_ms": _elapsed(start),
        }

    if size is None:
        return {
            "name": "database_size",
            "status": "pass",
            "message": "Database size is not measured for this database adapter",
            "duration_ms": _elapsed(start),
        }

    return {
        "name": "database_size",
        "status": "pass",
        "message": f"Database {size.describe()}",
        "duration_ms": _elapsed(start),
        "details": size.to_dict(),
    }


async def check_context_budget(agent) -> Dict[str, Any]:
    """Check context window utilization.

    Args:
        agent: KestrelAgent instance

    Returns:
        Check result dict
    """
    start = time.monotonic()

    try:
        ctx = getattr(agent, "context_manager", None)
        if ctx is not None:
            tokens_used = getattr(ctx, "tokens_used", 0) or 0
            tokens_max = getattr(ctx, "max_tokens", 0) or getattr(
                ctx, "tokens_max", 0
            ) or 0

            if tokens_max > 0:
                utilization = tokens_used / tokens_max
                status = "pass"
                if utilization > 0.9:
                    status = "fail"
                elif utilization > 0.75:
                    status = "warn"

                return {
                    "name": "context_budget",
                    "status": status,
                    "message": (
                        f"Context: {tokens_used}/{tokens_max} tokens "
                        f"({utilization * 100:.1f}% used)"
                    ),
                    "duration_ms": _elapsed(start),
                }

        # Fallback: try llm_service.token_budget
        llm = getattr(agent, "llm_service", None)
        if llm is not None:
            budget = getattr(llm, "token_budget", None)
            if budget is not None:
                used = getattr(budget, "used", 0) or 0
                total = getattr(budget, "total", 0) or 0
                if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
                    utilization = used / total
                    status = "pass"
                    if utilization > 0.9:
                        status = "fail"
                    elif utilization > 0.75:
                        status = "warn"

                    return {
                        "name": "context_budget",
                        "status": status,
                        "message": (
                            f"Token budget: {used}/{total} tokens "
                            f"({utilization * 100:.1f}% used)"
                        ),
                        "duration_ms": _elapsed(start),
                    }

        return {
            "name": "context_budget",
            "status": "pass",
            "message": "Context budget not tracked (no context manager)",
            "duration_ms": _elapsed(start),
        }
    except Exception as e:
        if is_execution_control_error(e):
            raise
        return {
            "name": "context_budget",
            "status": "fail",
            "message": f"Context budget check failed: {e}",
            "duration_ms": _elapsed(start),
        }


def _enabled_scheduler_feature(agent) -> Optional[Any]:
    """Return the agent's SchedulerFeature when it is loaded and enabled."""

    features = getattr(agent, "features", None)
    if isinstance(features, dict):
        for name, feature in features.items():
            if name == "SchedulerFeature" or type(feature).__name__ == "SchedulerFeature":
                return feature if getattr(feature, "enabled", True) else None
    return None


def _describe_disabled_schedule(schedule: Dict[str, Any]) -> str:
    """Name one scheduler-disabled schedule and the cause stored with it.

    The cause is structured (the deciding hook, its decision and the tool),
    never the hook's own reason, so this message carries nothing the call or
    the hook wrote (#3873).
    """

    from kestrel_sovereign.features.scheduler.outcome import PauseCause

    task_id = str(schedule.get("task_id") or "?")
    label = f"{schedule.get('task_name') or '?'} ({task_id[:8]}) {schedule.get('reason')}"
    cause = schedule.get("cause")
    if schedule.get("reason") != "blocked":
        return label
    if not cause:
        return f"{label}, no cause recorded"
    return f"{label}: {PauseCause.of(**cause).describe()}"


async def check_scheduler_liveness(agent, db) -> Dict[str, Any]:
    """Verify that an enabled SchedulerFeature has a current worker report."""

    start = time.monotonic()
    scheduler = _enabled_scheduler_feature(agent)
    if scheduler is None:
        return {
            "name": "scheduler_liveness",
            "status": "pass",
            "message": "SchedulerFeature is not enabled for this agent",
            "details": {"state": "not_configured"},
            "duration_ms": _elapsed(start),
        }
    if db is None:
        return {
            "name": "scheduler_liveness",
            "status": "fail",
            "message": "SchedulerFeature has no database for worker telemetry",
            "details": {"state": "storage_unavailable"},
            "duration_ms": _elapsed(start),
        }

    from kestrel_sovereign.features.storage_access import (
        AgentIdentityUnavailable,
        resolve_scoped_agent_did,
    )

    # The scheduler-status scope is the agent's DID through the shared
    # guard, resolved outside the inspection try so a missing identity is
    # its own named state rather than an empty scope that reads no reports
    # and fails liveness for the wrong reason (#3251).
    try:
        agent_did = resolve_scoped_agent_did(agent)
    except AgentIdentityUnavailable:
        return {
            "name": "scheduler_liveness",
            "status": "fail",
            "message": "Scheduler liveness inspection requires the agent's durable identity",
            "details": {"state": "identity_unavailable"},
            "duration_ms": _elapsed(start),
        }
    try:
        from kestrel_sovereign.features.scheduler.status import (
            scheduler_status,
            scheduler_status_parameters,
        )
        parameters = scheduler_status_parameters(scheduler)
        status = await scheduler_status(
            db,
            agent_id=agent_did,
            **parameters,
        )
    except Exception as error:
        if is_execution_control_error(error):
            raise
        return {
            "name": "scheduler_liveness",
            "status": "fail",
            "message": f"Scheduler liveness inspection failed: {error}",
            "details": {"state": "inspection_failed"},
            "duration_ms": _elapsed(start),
        }

    state = status["state"]
    if status["status"] == "pass":
        if status.get("executing_count"):
            message = (
                "Scheduler worker is current with "
                f"{status['enabled_count']} runnable and "
                f"{status['executing_count']} executing schedule(s)"
            )
        elif state == "running_zero_schedules":
            message = "Scheduler worker is current and reports zero schedules"
        elif state == "running_only_terminal_schedules":
            message = (
                "Scheduler worker is current; all "
                f"{status['terminal_count']} schedule(s) are terminal history"
            )
        elif state == "running_only_operator_paused_schedules":
            message = (
                "Scheduler worker is current; all "
                f"{status['disabled_count']} schedule(s) are operator-paused"
            )
        else:
            message = (
                "Scheduler worker is current with "
                f"{status['enabled_count']} runnable schedule(s)"
            )
    elif state == "awaiting_telemetry":
        message = "Scheduler is starting; no worker telemetry received yet"
    elif state == "no_telemetry":
        message = "No scheduler worker telemetry received"
    elif state == "no_current_telemetry":
        message = "No telemetry received from the current scheduler worker"
    elif state == "stale":
        message = (
            "Scheduler worker telemetry is stale "
            f"({status['report_age_seconds']}s old)"
        )
    elif state == "tick_stalled":
        tick_age = status.get("tick_age_seconds")
        age_detail = f"{tick_age}s exceeds" if tick_age is not None else "exceeded"
        message = (
            f"Scheduler polling tick is stalled ({age_detail} the "
            f"{status['tick_in_progress_limit_seconds']}s liveness bound)"
        )
    elif state == "overdue":
        message = (
            "Scheduler has unclaimed work overdue by "
            f"{status['overdue_seconds']}s"
        )
    elif state == "non_runnable_schedules":
        message = (
            f"Scheduler has {status['non_runnable_count']} enabled schedule(s) "
            "without a valid next_run_at"
        )
    elif state == "system_disabled_schedules":
        message = (
            f"Scheduler worker is current; {status['system_disabled_count']} "
            "schedule(s) were disabled by scheduler safety policy"
        )
        named = [
            _describe_disabled_schedule(schedule)
            for schedule in status.get("system_disabled_schedules") or []
        ]
        if named:
            message += ": " + "; ".join(named)
    elif state == "protocol_fenced_schedules":
        message = (
            f"Scheduler worker is current; {status['fenced_count']} "
            "schedule(s) are temporarily fenced by scheduler protocol"
        )
    else:
        message = f"Scheduler worker state is {state}"
    return {
        "name": "scheduler_liveness",
        "status": status["status"],
        "message": message,
        "details": status,
        "duration_ms": _elapsed(start),
    }


async def check_scheduled_task_failures(agent, db) -> Dict[str, Any]:
    """Flag every schedule the runner has marked as failing repeatedly (#3280).

    The runner records the marker once a schedule fails
    ``[scheduler] failure_alert_threshold`` consecutive times and clears it on
    the next success; this check only reports it. Before it existed, 23
    consecutive failures of one task showed up nowhere but the execution log.
    """

    start = time.monotonic()
    if _enabled_scheduler_feature(agent) is None:
        return {
            "name": "scheduled_task_failures",
            "status": "pass",
            "message": "SchedulerFeature is not enabled for this agent",
            "details": {"state": "not_configured"},
            "duration_ms": _elapsed(start),
        }
    if db is None:
        return {
            "name": "scheduled_task_failures",
            "status": "fail",
            "message": "SchedulerFeature has no database to read task failures from",
            "details": {"state": "storage_unavailable"},
            "duration_ms": _elapsed(start),
        }

    from kestrel_sovereign.features.storage_access import (
        AgentIdentityUnavailable,
        resolve_scoped_agent_did,
    )

    try:
        agent_did = resolve_scoped_agent_did(agent)
    except AgentIdentityUnavailable:
        return {
            "name": "scheduled_task_failures",
            "status": "fail",
            "message": "Scheduled task failure inspection requires the agent's durable identity",
            "details": {"state": "identity_unavailable"},
            "duration_ms": _elapsed(start),
        }
    try:
        from kestrel_sovereign.features.scheduler.failure_streak import (
            load_failure_markers,
        )

        markers = await load_failure_markers(db, agent_did, agent=agent)
    except Exception as error:
        if is_execution_control_error(error):
            raise
        return {
            "name": "scheduled_task_failures",
            "status": "fail",
            "message": f"Scheduled task failure inspection failed: {error}",
            "details": {"state": "inspection_failed"},
            "duration_ms": _elapsed(start),
        }

    if not markers:
        return {
            "name": "scheduled_task_failures",
            "status": "pass",
            "message": "No scheduled task is failing repeatedly",
            "details": {"state": "none_failing", "failing": []},
            "duration_ms": _elapsed(start),
        }
    summary = "; ".join(
        f"{marker['task_name']} ({marker['consecutive_failures']} consecutive "
        f"failures since {marker['first_failure_at']})"
        for marker in markers
    )
    return {
        "name": "scheduled_task_failures",
        "status": "warn",
        "message": f"{len(markers)} scheduled task(s) failing repeatedly: {summary}",
        "details": {"state": "failing", "failing": markers},
        "duration_ms": _elapsed(start),
    }


async def check_backup_quota(agent) -> Dict[str, Any]:
    """Warn when a backup target's account is near or over its quota (#3894).

    Reports what each target last read of its account (Lighthouse reads its
    ``user_data_usage`` before every snapshot upload), with when it read it;
    the check itself makes no network call. The host's Lighthouse account
    filled and every snapshot upload failed for days before anything named
    the quota.
    """
    start = time.monotonic()
    sync_service = getattr(agent, "_sync_service", None)
    reader = getattr(sync_service, "storage_quotas", None)
    if not callable(reader):
        return {
            "name": "backup_quota",
            "status": "pass",
            "message": "No backup target reports a storage quota",
            "details": {"state": "not_configured"},
            "duration_ms": _elapsed(start),
        }
    try:
        reported = reader()
    except Exception as error:
        if is_execution_control_error(error):
            raise
        return {
            "name": "backup_quota",
            "status": "warn",
            "message": f"Backup target storage quotas could not be read: {error}",
            "details": {"state": "inspection_failed"},
            "duration_ms": _elapsed(start),
        }
    quotas = {
        str(name): quota
        for name, quota in (reported.items() if isinstance(reported, dict) else ())
        if isinstance(quota, StorageQuota)
    }
    if not quotas:
        return {
            "name": "backup_quota",
            "status": "pass",
            "message": "No backup target has read its storage quota yet",
            "details": {"state": "not_observed", "targets": {}},
            "duration_ms": _elapsed(start),
        }

    targets = {
        name: {
            **quota.to_dict(),
            "near_quota": quota.near_quota,
            "over_quota": quota.over_quota,
        }
        for name, quota in sorted(quotas.items())
    }
    crowded = {name: quota for name, quota in sorted(quotas.items()) if quota.near_quota}
    if crowded:
        summary = "; ".join(
            f"{name} is {'over' if quota.over_quota else 'near'} quota: "
            f"{quota.describe()}, threshold {quota.threshold:.0%}, read "
            f"{quota.observed_at.isoformat()}"
            for name, quota in crowded.items()
        )
        return {
            "name": "backup_quota",
            "status": "warn",
            "message": f"Backup target near or over quota: {summary}",
            "details": {"state": "near_or_over_quota", "targets": targets},
            "duration_ms": _elapsed(start),
        }
    summary = "; ".join(
        f"{name}: {quota.describe()}, read {quota.observed_at.isoformat()}"
        for name, quota in sorted(quotas.items())
    )
    return {
        "name": "backup_quota",
        "status": "pass",
        "message": f"Backup target quota OK: {summary}",
        "details": {"state": "within_quota", "targets": targets},
        "duration_ms": _elapsed(start),
    }


async def check_bootstrap_state(agent, threshold_seconds: int = 3600) -> Dict[str, Any]:
    """Check whether first-contact bootstrap is stuck in PENDING."""
    start = time.monotonic()

    bootstrap_service = getattr(agent, "bootstrap_service", None)
    if bootstrap_service is None:
        return {
            "name": "bootstrap_state",
            "status": "pass",
            "message": "Bootstrap service not configured",
            "duration_ms": _elapsed(start),
        }

    storage = getattr(agent, "storage", None)
    agent_node = None
    if storage is not None:
        try:
            agent_node = await storage.get_node(getattr(agent, "agent_id", None))
        except Exception as exc:
            if is_execution_control_error(exc):
                raise
            logger.debug("Bootstrap health check could not load agent node: %s", exc)

    try:
        stale = await bootstrap_service.check_pending_timeout(
            agent_node=agent_node,
            storage=storage,
            threshold_seconds=threshold_seconds,
            mark_stale=False,
        )
    except Exception as exc:
        if is_execution_control_error(exc):
            raise
        return {
            "name": "bootstrap_state",
            "status": "warn",
            "message": f"Bootstrap state check failed: {exc}",
            "duration_ms": _elapsed(start),
        }

    if stale.is_stale:
        age_minutes = (stale.age_seconds or 0) / 60.0
        return {
            "name": "bootstrap_state",
            "status": "warn",
            "message": (
                "bootstrap_state is pending for "
                f"{age_minutes:.1f} minutes; status=stale_bootstrap"
            ),
            "duration_ms": _elapsed(start),
            "details": stale.to_dict(),
        }

    return {
        "name": "bootstrap_state",
        "status": "pass",
        "message": f"Bootstrap state OK: {stale.state.value}",
        "duration_ms": _elapsed(start),
        "details": stale.to_dict(),
    }


async def check_signal_audit_log(agent) -> Dict[str, Any]:
    """Report ``signal_log`` audit rows this process failed to persist (#2660).

    A dropped audit row is a permanent loss, but until this check existed its
    only trace was an ERROR line: 3,323 accumulated over two months of
    production before anyone looked. The dispatcher counts them; this surfaces
    the count where operators actually look.

    ``warn`` rather than ``fail`` — the agent keeps serving correctly and the
    audit trail has a hole. It does not clear when later writes succeed,
    because the rows lost earlier are still lost; an operator acknowledging
    that is the intent, not noise to be reset away.
    """
    start = time.monotonic()

    dispatcher = getattr(agent, "dispatcher", None)
    if dispatcher is None:
        return {
            "name": "signal_audit_log",
            "status": "pass",
            "message": "No signal dispatcher on this agent",
            "duration_ms": _elapsed(start),
        }

    # Duck-typed on purpose: third-party agents and test doubles implement the
    # DispatcherAgent protocol, not this accounting. A bare truthiness test
    # would warn on any stand-in whose attribute access auto-creates a value
    # (a MagicMock dispatcher reports a truthy count), so require a real
    # positive integer before claiming rows were lost. Reporting a loss that
    # did not happen is the same failure as hiding one that did.
    dropped = getattr(dispatcher, "log_write_failure_count", 0)
    if not isinstance(dropped, int) or isinstance(dropped, bool) or dropped <= 0:
        return {
            "name": "signal_audit_log",
            "status": "pass",
            "message": "No signal_log writes dropped",
            "duration_ms": _elapsed(start),
        }

    last = getattr(dispatcher, "last_log_write_failure", None)
    details: Dict[str, Any] = {"dropped": dropped}
    if last is not None:
        details["last_signal_id"] = str(getattr(last, "signal_id", ""))
        details["last_error"] = str(getattr(last, "error", ""))
        failed_at = getattr(last, "failed_at", None)
        if isinstance(failed_at, datetime):
            details["last_failed_at"] = failed_at.isoformat()

    return {
        "name": "signal_audit_log",
        "status": "warn",
        "message": (
            f"{dropped} signal_log audit row(s) dropped since start; "
            "the audit trail is incomplete"
        ),
        "duration_ms": _elapsed(start),
        "details": details,
    }


def _positive_int(value: Any) -> Optional[int]:
    """*value* when it is a positive ``int`` (not a ``bool``), else ``None``."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


async def check_conversation_embeddings(agent) -> Dict[str, Any]:
    """Report conversation embeddings dropped on write, and a width mismatch (#2913).

    A turn whose embedding cannot be stored is still persisted, without its
    vector, so semantic recall over conversation history degrades with every
    such turn. Its only trace used to be INFO lines. The store counts the
    drops and records a profile/column width disagreement, found at startup
    and re-checked whenever the embedding profile changes at runtime (#3606);
    this surfaces both, the way ``check_signal_audit_log`` surfaces dropped
    audit rows.

    ``warn`` rather than ``fail``, like that check: the agent keeps serving
    and its turns keep persisting. The count does not clear when later writes
    succeed, because the vectors dropped earlier are still missing. The
    misconfiguration does clear once the active profile fits the column.
    """
    start = time.monotonic()

    store = resolve_conversation_store(agent)
    if store is None:
        return {
            "name": "conversation_embeddings",
            "status": "pass",
            "message": "No conversation store on this agent",
            "duration_ms": _elapsed(start),
        }

    # Duck-typed and type-checked for the same reason as the signal_log
    # check: a stand-in store must not report a loss that did not happen.
    dropped = _positive_int(getattr(store, "dropped_embedding_count", None)) or 0
    mismatch = getattr(store, "embedding_column_mismatch", None)
    profile_dim = _positive_int(getattr(mismatch, "profile_dim", None))
    column_dim = _positive_int(getattr(mismatch, "column_dim", None))
    misconfigured = (
        profile_dim is not None
        and column_dim is not None
        and profile_dim != column_dim
    )

    if dropped == 0 and not misconfigured:
        return {
            "name": "conversation_embeddings",
            "status": "pass",
            "message": "No conversation embeddings dropped",
            "duration_ms": _elapsed(start),
        }

    details: Dict[str, Any] = {"dropped": dropped, "misconfigured": misconfigured}
    messages = []
    if misconfigured:
        details["misconfiguration"] = {
            "profile_id": str(getattr(mismatch, "profile_id", "")),
            "profile_dim": profile_dim,
            "column_dim": column_dim,
        }
        messages.append(
            "misconfiguration: the embedding profile produces "
            f"{profile_dim}-dimension vectors, the conversation_history "
            f"vector column is {column_dim}-dimension"
        )
    if dropped:
        last = getattr(store, "last_dropped_embedding", None)
        if last is not None:
            profile_id = getattr(last, "profile_id", None)
            details["last_profile_id"] = (
                profile_id if isinstance(profile_id, str) else None
            )
            details["last_embedding_dim"] = _positive_int(
                getattr(last, "embedding_dim", None)
            )
            details["last_column_dim"] = _positive_int(
                getattr(last, "column_dim", None)
            )
            details["last_error"] = str(getattr(last, "error", ""))
            dropped_at = getattr(last, "dropped_at", None)
            if isinstance(dropped_at, datetime):
                details["last_dropped_at"] = dropped_at.isoformat()
        messages.append(
            f"{dropped} conversation embedding(s) dropped since start; "
            "semantic recall cannot find those turns"
        )

    return {
        "name": "conversation_embeddings",
        "status": "warn",
        "message": "; ".join(messages),
        "duration_ms": _elapsed(start),
        "details": details,
    }


def _classify_resource_lock_snapshot(
    raw_holds: Any, start: float
) -> Dict[str, Any]:
    """Validate and grade one duck-typed resource-lock snapshot."""
    if not isinstance(raw_holds, list):
        return {
            "name": "resource_locks",
            "status": "pass",
            "message": "Resource lock diagnostics not available",
            "duration_ms": _elapsed(start),
        }

    slow_holds: list[dict[str, Any]] = []
    for raw in raw_holds:
        if not isinstance(raw, dict):
            continue
        if raw.get("resource") != ResourceLock.MEMORY.value:
            continue
        held_seconds = raw.get("held_seconds")
        blocked = raw.get("blocked_acquirers")
        if (
            isinstance(held_seconds, bool)
            or not isinstance(held_seconds, (int, float))
            or isinstance(blocked, bool)
            or not isinstance(blocked, int)
            or held_seconds < resource_lock_manager.SLOW_HOLD_WARN_SECONDS
            or blocked < 0
        ):
            continue
        slow_holds.append(dict(raw))

    if not slow_holds:
        return {
            "name": "resource_locks",
            "status": "pass",
            "message": "The memory lock is not held past the slow-hold threshold",
            "duration_ms": _elapsed(start),
        }

    blocked_holds = [
        hold for hold in slow_holds if hold["blocked_acquirers"] > 0
    ]
    if not blocked_holds:
        return {
            "name": "resource_locks",
            "status": "pass",
            "message": (
                f"{len(slow_holds)} resource lock(s) are long-held, but no "
                "acquirers are blocked"
            ),
            "duration_ms": _elapsed(start),
            "details": {"held_locks": slow_holds},
        }

    longest = max(blocked_holds, key=lambda hold: hold["held_seconds"])
    held_seconds = float(longest["held_seconds"])
    held_for = (
        f"{held_seconds / 3600:.1f} hours"
        if held_seconds >= 3600
        else f"{held_seconds:.0f} seconds"
    )
    escalated = any(
        hold["held_seconds"] >= resource_lock_manager.BLOCKED_HOLD_ERROR_SECONDS
        for hold in blocked_holds
    )
    return {
        "name": "resource_locks",
        # Sub-hour contention is observable in details/logs but remains within
        # the normal envelope of bounded sleep/training work. Readiness changes
        # only at the explicit escalation threshold.
        # A long-held MEMORY lock is actionable, but training and other
        # deliberately non-cancellable work can legitimately exceed the
        # escalation threshold. Keep it degraded (rather than database/LLM
        # unhealthy) consistently on both detailed-health surfaces.
        "status": "warn" if escalated else "pass",
        "message": (
            f"{longest.get('resource', 'unknown')} lock has been held for "
            f"{held_for} by {longest.get('label', 'unknown')} with "
            f"{longest['blocked_acquirers']} acquirer(s) blocked"
        ),
        "duration_ms": _elapsed(start),
        "details": {
            "held_locks": slow_holds,
            "escalation_threshold_seconds": (
                resource_lock_manager.BLOCKED_HOLD_ERROR_SECONDS
            ),
        },
    }


async def check_resource_locks(agent) -> Dict[str, Any]:
    """Surface a MEMORY lock that is both long-held and blocking work.

    MEMORY protects bounded consolidation passes as well as legitimately
    long-running work such as training cycles.  Prolonged contention is still
    an actionable operator signal, but it does not by itself prove the agent
    is unhealthy.  Other resources have different healthy hold envelopes: in
    particular, CONVERSATION spans an entire agentic turn and must not make
    routine multi-minute work degrade readiness. The snapshot is in-memory
    and cannot itself block behind the resource being diagnosed.
    """
    start = time.monotonic()
    dispatcher = getattr(agent, "dispatcher", None)
    lock_manager = (
        getattr(dispatcher, "lock_manager", None)
        if dispatcher is not None
        else None
    )
    snapshot = getattr(lock_manager, "active_hold_diagnostics", None)
    if not callable(snapshot):
        return {
            "name": "resource_locks",
            "status": "pass",
            "message": "Resource lock diagnostics not available",
            "duration_ms": _elapsed(start),
        }

    try:
        return _classify_resource_lock_snapshot(snapshot(), start)
    except Exception as exc:
        if is_execution_control_error(exc):
            raise
        return {
            "name": "resource_locks",
            "status": "warn",
            "message": f"Resource lock diagnostics failed: {exc}",
            "duration_ms": _elapsed(start),
        }


async def check_birth_record(agent) -> Dict[str, Any]:
    """Report birth-record capability the runtime database does not have (#2871).

    On a host whose runtime database is not the one inception wrote to, boot
    copies the birth record across. Whatever the local anchor cannot supply —
    a constitution it never held, chunk ownership a pre-#2649 database could
    not prove — stays missing, and no retry will change that.

    The agent still boots: refusing would convert a degraded agent into one
    that can never boot again, with no operator verb to fix it, and identity
    failures (no agent node, a fabricated placeholder) are refused at boot
    already. But the loss must not be silent — #2871's whole defect was an
    agent with zero constitution chunks while ``/health`` reported ok.

    ``warn`` rather than ``fail``: the agent serves correctly, and the thing it
    cannot do is constitutional retrieval. It does not clear on its own.
    """
    start = time.monotonic()

    # Duck-typed like the sibling checks: third-party agents and test doubles
    # implement the health protocol, not this accounting. Require a real
    # non-empty list of strings before claiming a loss — reporting one that did
    # not happen is the same failure as hiding one that did.
    shortfall = getattr(agent, "_birth_record_shortfall", None)
    if not isinstance(shortfall, list) or not shortfall:
        return {
            "name": "birth_record",
            "status": "pass",
            "message": "Birth record complete in the runtime database",
            "duration_ms": _elapsed(start),
        }
    reasons = [str(reason) for reason in shortfall]
    # A completed pass that could not supply this means the anchor does not
    # have it; a pass that DIED is usually transient. Telling an operator a
    # dropped connection is unrepairable sends them to rebuild an anchor when
    # a restart would have fixed it.
    retryable = bool(getattr(agent, "_birth_record_shortfall_retryable", False))

    return {
        "name": "birth_record",
        "status": "warn",
        "message": (
            "Birth record incomplete in the runtime database; "
            + (
                "the copy failed this pass and a restart may complete it: "
                if retryable
                else "the local anchor cannot supply it: "
            )
            + "; ".join(reasons)
        ),
        "duration_ms": _elapsed(start),
        "details": {"missing": reasons, "retryable": retryable},
    }


async def check_model_discovery(agent) -> Dict[str, Any]:
    """Surface vendors whose model discovery failed (#3190).

    A revoked or disabled API key makes ``GET /v1/models`` fail while chat calls
    on the same vendor keep working — they use a different credential and a
    different endpoint. That asymmetry is why the condition needs its own
    surface: nothing else notices.

    It matters because a vendor with no retrievable catalog cannot validate a
    pinned model. On 2026-08-31 the Anthropic key was disabled, discovery
    401ed, and the resulting empty catalog caused every agent's persisted
    ``anthropic:plan/claude-opus-5`` pin to be rejected at boot and discarded on
    a WARNING line — dropping the whole fleet onto a 1B local model with no
    health signal anywhere.

    ``warn`` when any vendor's discovery failed. ``fail`` when the failing
    vendor is the one THIS agent's mandate points at, since that is the pin
    actually at risk. Neither is critical in
    :func:`derive_overall_status`, so this reports ``degraded`` rather than
    taking the host to ``unhealthy`` for a catalog problem.

    The outcome is read from the process-wide catalog snapshot, not from the
    service that happened to run discovery (#3197): every agent on a host reads
    the same catalog, so every agent reports the same failures for it, and the
    next refresh by any of them replaces the reading for all.

    A failing vendor whose every configured route is ``standby`` for this agent
    (see :func:`assess_llm_routes`) is reported under ``standby_failed_vendors``
    and does not degrade the check (#3912): nothing this agent runs selects it.
    """
    from kestrel_sovereign.llm.model_cache import (
        DiscoveryStatus,
        get_shared_model_cache,
    )

    start = time.monotonic()

    llm_service = getattr(agent, "llm_service", None)
    if llm_service is None:
        return {
            "name": "model_discovery",
            "status": "pass",
            "message": "No LLM service configured",
            "duration_ms": _elapsed(start),
        }

    snapshot = get_shared_model_cache().snapshot()
    failures = snapshot.discovery_failures if snapshot is not None else {}
    if not failures:
        # A standing failure is never missing here: only a refresh retires
        # it, by requesting the vendor's catalog or by not covering the vendor.
        # Healthy is claimed only when some vendor's catalog request actually
        # completed — not for a catalog restored from disk, or a refresh that
        # issued no request.
        discovered = snapshot is not None and any(
            outcome.status is DiscoveryStatus.OK
            for outcome in snapshot.outcomes.values()
        )
        return {
            "name": "model_discovery",
            "status": "pass",
            "message": (
                "Model discovery healthy for every discovered vendor"
                if discovered
                else "No vendor's model catalog has been fetched"
            ),
            "details": {
                "catalog_version": snapshot.catalog_version if snapshot is not None else None,
            },
            "duration_ms": _elapsed(start),
        }

    # Is the agent's own pinned vendor among the failures? That is the case
    # that silently unpins this agent on its next restart.
    pinned_vendor = None
    try:
        if hasattr(llm_service, "get_model_preference"):
            pinned_vendor = (llm_service.get_model_preference() or {}).get("vendor")
    except Exception as error:  # pragma: no cover - ordinary preference failures are optional
        if is_execution_control_error(error):
            raise
        pinned_vendor = None

    # A vendor no configured route of this agent selects cannot unpin it or
    # serve it, so its catalog failure is reported, not counted (#3912). The
    # pinned vendor is never standby: the selection names its routes.
    try:
        routes = await assess_llm_routes(agent)
    except Exception as error:
        if is_execution_control_error(error):
            raise
        # Without an assessment no vendor can be shown to be standby, so
        # every failure counts, as it did before route evidence existed.
        routes = None
    standby = routes.standby_vendors() if routes is not None else frozenset()
    standby_vendors = sorted(
        vendor for vendor in failures
        if vendor in standby and vendor != pinned_vendor
    )
    standby_failures = {vendor: failures[vendor] for vendor in standby_vendors}
    failures = {
        vendor: error for vendor, error in failures.items()
        if vendor not in standby_failures
    }
    if not failures:
        return {
            "name": "model_discovery",
            "status": "pass",
            "message": (
                f"Model discovery failed only for standby vendor(s): "
                f"{', '.join(standby_vendors)}. No configured route this agent "
                f"selects uses them."
            ),
            "details": {
                "catalog_version": snapshot.catalog_version,
                "standby_failed_vendors": standby_failures,
                "pinned_vendor": pinned_vendor,
                "pinned_vendor_at_risk": False,
            },
            "duration_ms": _elapsed(start),
        }

    vendors = sorted(failures)
    pinned_at_risk = pinned_vendor in failures if pinned_vendor else False

    if pinned_at_risk:
        # Say what is true: the catalog needed to VALIDATE the pin is missing.
        # Whether the pin would actually be dropped on restart depends on
        # whether the model still appears in the partial catalog or is the
        # route's configured default — and the fix in this same change is
        # designed to preserve it. Asserting a guaranteed outage would be
        # crying wolf about the case we now handle (codex r2 P2).
        message = (
            f"Model discovery failed for '{pinned_vendor}', the vendor this "
            f"agent's model is pinned to — the pin cannot be validated "
            f"against a live catalog while this persists. "
            f"{failures[pinned_vendor]}"
        )
    else:
        message = (
            f"Model discovery failed for {len(vendors)} vendor(s): "
            f"{', '.join(vendors)}. Pinned models for these vendors cannot be "
            f"validated. Check the vendor API credentials."
        )

    return {
        "name": "model_discovery",
        "status": "fail" if pinned_at_risk else "warn",
        "message": message,
        "details": {
            "catalog_version": snapshot.catalog_version,
            "failed_vendors": {v: failures[v] for v in vendors},
            # The publication of this snapshot issued no request for these
            # vendors; the failure is the last one observed for each.
            "not_reattempted_vendors": [
                v for v in vendors
                if snapshot.outcomes[v].status is DiscoveryStatus.NOT_ATTEMPTED
            ],
            "standby_failed_vendors": standby_failures,
            "pinned_vendor": pinned_vendor,
            "pinned_vendor_at_risk": pinned_at_risk,
        },
        "duration_ms": _elapsed(start),
    }


async def run_standard_checks(agent, db) -> List[Dict[str, Any]]:
    """The checks every health surface runs, in one place.

    There are two callers — ``HealthFeature._run_health`` and the fallback in
    ``server.py`` for agents without the feature, which is not mandatory and so
    is reachable by design. They were two hand-maintained lists, and they had
    already drifted: a check added to one left the other reporting ``healthy``
    for a state the first calls a warning. Sharing the list is what stops the
    next one drifting.
    """
    # Snapshot resource ownership before touching the database, then bound each
    # database-touching check. A cancelled aiosqlite read may keep draining
    # internally, but it cannot prevent the in-memory diagnosis from returning.
    resource_locks = await check_resource_locks(agent)
    database = await _run_database_bound_check(
        "database", check_database(db)
    )

    return [
        resource_locks,
        database,
        await check_llm_service(agent),
        await check_memory_system(agent),
        await check_disk_space(),
        await _run_database_bound_check(
            "database_size", check_database_size(db)
        ),
        await check_context_budget(agent),
        await _run_database_bound_check(
            "scheduler_liveness",
            check_scheduler_liveness(agent, db),
        ),
        await _run_database_bound_check(
            "scheduled_task_failures",
            check_scheduled_task_failures(agent, db),
        ),
        await check_backup_quota(agent),
        await _run_database_bound_check(
            "bootstrap_state",
            check_bootstrap_state(agent),
        ),
        await check_signal_audit_log(agent),
        await check_conversation_embeddings(agent),
        await check_birth_record(agent),
        await check_model_discovery(agent),
    ]


def _elapsed(start: float) -> float:
    """Return elapsed time in milliseconds since start."""
    return round((time.monotonic() - start) * 1000, 2)
