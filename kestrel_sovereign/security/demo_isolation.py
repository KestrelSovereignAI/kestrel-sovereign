"""Server-side demo-mode isolation (#766).

Background
----------
The 2026-04-24 incident wiped Claw, Meridian, and Nellie's conversation
history because a Playwright demo harness pointed at the live multi_agent
server on ``localhost:8888`` and called destructive APIs against
whichever agents the server had mounted.

The convention layer fix is shipped (``kestrel demo run`` —
:mod:`kestrel_sovereign.cli_demo`, the canonical demo entry point;
spins an isolated server on port 8900 against ``agent_data/demo/`` and
refuses port 8888; ported from the legacy ``demos/run.sh`` in epic
#1050 tier 3). **But that's discipline, not enforcement.** A rogue
script (or a curious developer running ``cd demos/foo && npx
playwright test`` directly) still hits live.

This module is the server-side prevention layer. Even if a destructive
call reaches the live server, the rail here refuses it unless the
target is a demo-scoped agent OR the caller carries an explicit opt-in
header.

Decision matrix
---------------
================  =================  ====================================
 Server mode      Target agent       Verdict
================  =================  ====================================
 live             demo-scoped        ALLOW (demos are safe to mess with)
 live             live agent         ALLOW only if X-Kestrel-Allow-
                                     Destructive header present; else
                                     REFUSE + audit
 demo             demo-scoped        ALLOW
 demo             live agent         REFUSE + audit (catches misconfig:
                                     demo server somehow got a live agent)
================  =================  ====================================

The header carries a free-text reason (``user-initiated GDPR purge``,
``administrative cleanup before shutdown``, ``operator confirmed via
session prompt``) which lands in the audit log. Production UIs that
need to issue destructive ops attach the header automatically.

Server-mode classification happens at startup based on the loaded
agent set — see :func:`classify_server_mode`. The classification is a
single boolean stored on ``app.state.demo_mode``; endpoints don't need
to recompute it per request.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

from fastapi import HTTPException, Request

if TYPE_CHECKING:  # pragma: no cover
    from kestrel_sovereign.kestrel_agent import KestrelAgent

logger = logging.getLogger(__name__)

#: Header that opts a request into a destructive op against a live agent.
#: The value is recorded as the audit log's ``args_summary`` reason.
ALLOW_DESTRUCTIVE_HEADER = "X-Kestrel-Allow-Destructive"


def _strict_is_demo(agent) -> bool:
    """Return True only when ``agent.is_demo`` is the literal ``True``.

    Defensive against MagicMock-based tests where unset attributes
    return a truthy MagicMock. The destructive-op rail must never be
    bypassed by accident; it gates the rail decision on an explicit
    bool that the inception layer set deliberately.
    """
    return getattr(agent, "is_demo", False) is True


def classify_server_mode(agents: Mapping[str, "KestrelAgent"]) -> bool:
    """Return True when this server is running in demo mode.

    Demo mode is "every loaded agent is demo-scoped." A single live
    agent in the multi_agent flips the server back to live mode — better
    to be conservative than to mistakenly relax the rail because most
    of the multi_agent is demo.

    Empty multi_agent resolves to live mode. A server that hasn't loaded
    agents yet should not be permissive about destructive ops; the
    operator can still run demos against an empty multi_agent (no agents
    to destroy).
    """
    if not agents:
        return False
    return all(_strict_is_demo(agent) for agent in agents.values())


def _audit_summary(
    *,
    request: Request,
    agent: Optional["KestrelAgent"],
    server_demo_mode: bool,
    target_is_demo: Optional[bool],
    header_reason: Optional[str],
    decision: str,
) -> str:
    """Build the JSON blob recorded as the audit log's ``args_summary``.

    Captures everything an investigator needs to reconstruct a refusal:
    caller IP, the endpoint, the X-Kestrel-Allow-Destructive reason (or
    its absence), the server mode, and the target agent's demo flag.
    Headers carrying secrets (X-API-Key, Authorization) are dropped —
    this blob lands in a queryable table and may be exported.
    """
    redacted_headers = {
        k.lower(): v
        for k, v in request.headers.items()
        if k.lower() not in {
            "x-api-key", "authorization", "cookie", "set-cookie",
        }
    }
    payload = {
        "decision": decision,
        "endpoint": str(request.url.path),
        "method": request.method,
        "client_ip": request.client.host if request.client else None,
        "server_demo_mode": server_demo_mode,
        "target_did": getattr(agent, "did", None),
        "target_is_demo": target_is_demo,
        "header_reason": header_reason,
        "headers": redacted_headers,
    }
    return json.dumps(payload, default=str)


async def _record_audit(
    *,
    agent: Optional["KestrelAgent"],
    decision: str,
    summary: str,
) -> None:
    """Best-effort write to the agent's security_audit_log.

    Routes through the SecurityFeature's PermissionStore when available
    (every production agent loads it). Falls back to logger.warning if
    the store is missing — early-startup failures or constrained tests
    must not mask a destructive-op refusal.
    """
    permission_store = None
    try:
        # Feature registers under its class name "SecurityFeature".
        # Earlier draft used "Security" and silently dropped audit
        # writes in production — caught during smoke testing.
        # Tolerate both keys for forward-compat.
        features = getattr(agent, "features", {}) if agent else {}
        feature = features.get("SecurityFeature") or features.get("Security")
        permission_store = getattr(feature, "permission_store", None) if feature else None
    except Exception:  # pragma: no cover — defensive
        permission_store = None

    if permission_store is None:
        logger.warning(
            "[demo-isolation] %s — audit store unavailable, summary=%s",
            decision, summary,
        )
        return

    try:
        await permission_store.log_decision(
            feature_name="demo_isolation",
            tool_name="destructive_op_guard",
            action="server_side_guard",
            decision=decision,
            args_summary=summary,
        )
    except Exception as e:  # pragma: no cover — never block the request
        logger.warning(
            "[demo-isolation] failed to record audit (%s); summary=%s",
            e, summary,
        )


async def enforce_destructive_op(request: Request) -> None:
    """FastAPI dependency that gates destructive operations (#766).

    Use as ``Depends(enforce_destructive_op)`` on any route that wipes,
    purges, hard-deletes, or toggles permissions. Reads:

    * ``request.app.state.demo_mode`` — set in the app's lifespan handler
      via :func:`classify_server_mode`. Defaults to False (live) if unset.
    * ``request.state.agent`` — the target agent. ``None`` means the
      request is operating on a non-agent-scoped resource (rare for
      destructive ops); we treat it as a live target.
    * The :data:`ALLOW_DESTRUCTIVE_HEADER` header for the live-mode
      escape hatch.

    Refusal raises ``HTTPException(403)`` and writes an audit entry.
    Successful passes also leave a breadcrumb when the header was used,
    so the operator can review legitimate destructive ops after the fact.
    """
    server_demo_mode = getattr(request.app.state, "demo_mode", False) is True
    agent = getattr(request.state, "agent", None) or getattr(
        request.app.state, "agent", None
    )
    target_is_demo = _strict_is_demo(agent) if agent is not None else False
    header_reason = request.headers.get(ALLOW_DESTRUCTIVE_HEADER)

    # --- Demo server: the reverse rail ---------------------------------
    # If the operator has set up an isolated demo server, never let it
    # touch a live agent — that almost certainly indicates a misconfig
    # (wrong KESTREL_DB_PATH, bad multi_agent.toml). Refuse loud and audit.
    if server_demo_mode and not target_is_demo and agent is not None:
        summary = _audit_summary(
            request=request,
            agent=agent,
            server_demo_mode=True,
            target_is_demo=False,
            header_reason=header_reason,
            decision="refused-demo-server-vs-live-agent",
        )
        await _record_audit(
            agent=agent, decision="refused", summary=summary
        )
        logger.error(
            "[demo-isolation] refused destructive op on live agent %s "
            "from demo-mode server (likely misconfig)",
            getattr(agent, "did", "<unknown>"),
        )
        raise HTTPException(
            status_code=403,
            detail=(
                "Server is running in demo mode but the request targets a "
                "live agent. This usually means the demo server was started "
                "against a multi_agent that contains a live agent — check "
                "KESTREL_DB_PATH and multi_agent.toml."
            ),
        )

    # --- Live server: header rail --------------------------------------
    # Demo-scoped target → always allow. Live target → require the
    # opt-in header. The header value is the audit reason.
    if not server_demo_mode and not target_is_demo:
        if not header_reason:
            summary = _audit_summary(
                request=request,
                agent=agent,
                server_demo_mode=False,
                target_is_demo=False,
                header_reason=None,
                decision="refused-no-destructive-header",
            )
            await _record_audit(
                agent=agent, decision="refused", summary=summary
            )
            logger.warning(
                "[demo-isolation] refused destructive op on live agent %s "
                "(no %s header)",
                getattr(agent, "did", "<unknown>"),
                ALLOW_DESTRUCTIVE_HEADER,
            )
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Destructive operations on live agents require the "
                    f"{ALLOW_DESTRUCTIVE_HEADER} header carrying a reason. "
                    "The UI attaches this automatically; scripts must opt in "
                    "explicitly to confirm intent."
                ),
            )

        # Allowed — but record the breadcrumb so the operator can see
        # what destructive ops fired and why.
        summary = _audit_summary(
            request=request,
            agent=agent,
            server_demo_mode=False,
            target_is_demo=False,
            header_reason=header_reason,
            decision="allowed-with-header",
        )
        await _record_audit(
            agent=agent, decision="allowed", summary=summary
        )

    # demo-server + demo-target, or live-server + demo-target: silent allow.
    # No audit noise for routine demo ops.


def all_destructive_endpoints_registered(routes: Iterable) -> bool:  # pragma: no cover - debugging
    """Helper for tests to spot routes that should carry the dependency
    but don't. Not used in the production path; lives here so the rail's
    intent stays close to its enforcement.
    """
    # Names of routes that MUST be guarded. Update when adding new
    # destructive endpoints.
    expected = {
        "purge_conversation",
        "purge_message",
        "set_permission",
    }
    found = {getattr(r, "name", None) for r in routes}
    return expected.issubset(found)
