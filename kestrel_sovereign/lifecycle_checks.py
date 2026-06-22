"""
Startup lifecycle hardening checks.

Hardening rails the agent boot path was missing before #376:

- ``verify_llm_providers_initialized`` (#377): refuse to call an agent
  "ready" if zero LLM providers came up. Emma v1 ran two weeks with no
  configured keys, no provider ever initialized, and inception still
  declared success — agent was effectively mute.

- ``verify_identity_isolation`` (#381): refuse to start if the DID
  recorded in the agent's database doesn't match the DID the operator
  declared (``KESTREL_EXPECTED_DID`` env var). Catches the case where
  a runtime is accidentally pointed at another agent's data directory
  (Claw running against Emma's ``kestrel_prime.db`` — silent identity
  contamination, no validation, no refusal).

Both checks raise rather than logging a warning. They're invoked during
FastAPI lifespan startup; ``_set_startup_error`` captures the exception
so ``/health`` reports the failure and routes refuse to serve requests.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class NoLLMProvidersError(RuntimeError):
    """Raised when the agent boots with zero LLM providers initialized (#377)."""


class IdentityIsolationError(RuntimeError):
    """Raised when the DB's agent DID does not match the operator-declared DID (#381)."""


EXPECTED_DID_ENV_VAR = "KESTREL_EXPECTED_DID"


def verify_llm_providers_initialized(llm_service: Any) -> None:
    """Refuse to mark the agent as ready when no provider came up (#377).

    ``LLMService.__init__`` catches ``ProviderInitializationError`` and leaves
    ``self.providers = []``. That's deliberate — we want the server to come
    up far enough to surface a clear health-check error rather than crash
    obscurely. But once the server is up, an empty ``providers`` list means
    the agent is mute, and we should refuse to declare startup successful.

    PayerPolicy carve-out: when ``llm_service.disabled is True`` (set by
    ``KestrelAgent.initialize`` when the agent's ``PayerPolicy.llm.kind``
    is ``PayerKind.NONE``), zero providers is the *intended* state —
    operator has explicitly opted the agent out of LLM use. Skip the check
    in that case rather than rejecting a valid configuration.

    Args:
        llm_service: An ``LLMService`` instance whose ``initialize_providers``
            has already run, and (when applicable) whose PayerPolicy has been
            resolved via ``KestrelAgent.initialize``. We read both
            ``llm_service.providers`` and ``llm_service.disabled``.

    Raises:
        NoLLMProvidersError: when LLM is intended-on and ``len(providers) == 0``.
    """
    if getattr(llm_service, "disabled", False):
        # PayerKind.NONE — operator has explicitly disabled LLM use. Zero
        # providers is the intended state; don't raise.
        return
    providers = getattr(llm_service, "providers", None) or []
    if len(providers) > 0:
        return
    raise NoLLMProvidersError(
        "Inception completed but no LLM providers initialized — the agent "
        "cannot respond. Check that at least one provider credential is set "
        "(ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN for the Claude OAuth/plan "
        "route, OPENAI_API_KEY, OPENROUTER_API_KEY, "
        "GOOGLE_API_KEY) or that a local provider (Ollama, llama.cpp) is "
        "reachable. See LLM_SERVICE_ARCHITECTURE.md for the route-config "
        "shape. If this agent is intentionally LLM-disabled, configure "
        "PayerPolicy.llm.kind = NONE."
    )


def verify_identity_isolation(
    db_did: str,
    expected_did: Optional[str] = None,
    *,
    env: Optional[dict] = None,
) -> None:
    """Refuse to start if the running process is pointing at the wrong agent's data (#381).

    The operator declares the expected agent DID via ``KESTREL_EXPECTED_DID``
    (env var). If that env var is set and does not match the DID read from
    ``graph_nodes WHERE node_type = 'agent'`` in the agent's database, this
    function raises ``IdentityIsolationError`` and the lifespan handler routes
    that into a startup error visible at ``/health``.

    When the env var is unset, the check is a no-op — single-agent dev setups
    that don't pre-declare an identity continue to work unchanged. Operators
    deploying multi-agent fleets are expected to set the env var per process.

    Args:
        db_did: The DID actually stored in the agent's DB graph (returned by
            ``get_agent_did_async``).
        expected_did: Optional explicit override (for testing). When None,
            we read ``KESTREL_EXPECTED_DID`` from the environment.
        env: Optional environment dict (for testing). Defaults to os.environ.

    Raises:
        IdentityIsolationError: when both DIDs are present and disagree.
    """
    if expected_did is None:
        env_map = env if env is not None else os.environ
        expected_did = env_map.get(EXPECTED_DID_ENV_VAR, "").strip() or None

    if not expected_did:
        # No expected DID declared → operator opted out; preserve old behavior.
        return

    if expected_did == db_did:
        return

    raise IdentityIsolationError(
        f"Identity isolation check failed: this process was started with "
        f"{EXPECTED_DID_ENV_VAR}={expected_did!r}, but the agent database "
        f"records {db_did!r}. Refusing to start — running an agent against "
        "another agent's database is a security violation (cross-agent data "
        "access, false-attribution writes, potential identity contamination)."
    )


async def warn_stale_bootstrap_pending(
    agent: Any,
    *,
    threshold_seconds: int = 3600,
    context: str = "startup",
) -> Optional[dict]:
    """Log and persist a warning when bootstrap is PENDING past the timeout.

    Unlike provider/identity lifecycle checks, this does not refuse startup:
    an untouched agent can still be recovered by first contact or an explicit
    ``!skip-discovery``. The important part is making the stuck state visible
    to startup logs, heartbeat history, and the graph/metadata status.
    """
    bootstrap_service = getattr(agent, "bootstrap_service", None)
    if bootstrap_service is None:
        return None

    storage = getattr(agent, "storage", None)
    agent_node = None
    if storage is not None:
        try:
            agent_node = await storage.get_node(getattr(agent, "agent_id", None))
        except Exception as exc:
            logger.debug("Could not load agent node for bootstrap timeout check: %s", exc)

    stale = await bootstrap_service.check_pending_timeout(
        agent_node=agent_node,
        storage=storage,
        threshold_seconds=threshold_seconds,
        mark_stale=True,
    )
    if not stale.is_stale:
        return None

    agent_name = getattr(agent, "_agent_name", None) or getattr(agent, "agent_id", "unknown")
    age_minutes = (stale.age_seconds or 0) / 60.0
    logger.warning(
        "Bootstrap stale: agent %s has bootstrap_state='pending' for %.1f minutes "
        "(threshold %.1f minutes); status escalated to stale_bootstrap during %s",
        agent_name,
        age_minutes,
        threshold_seconds / 60.0,
        context,
    )
    return stale.to_dict()
