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


class NoReachableProvidersError(RuntimeError):
    """Raised when only local LLM routes are configured and none are reachable."""


class IdentityIsolationError(RuntimeError):
    """Raised when the DB's agent DID does not match the operator-declared DID (#381)."""


EXPECTED_DID_ENV_VAR = "KESTREL_EXPECTED_DID"
SKIP_REACHABILITY_PROBE_ENV_VAR = "KESTREL_SKIP_REACHABILITY_PROBE"


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


def _env_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


async def verify_llm_providers_reachable(
    llm_service: Any,
    *,
    env: Optional[dict] = None,
    timeout: float = 1.5,
) -> None:
    """Probe local LLM routes before declaring startup successful (#1265).

    Cloud routes are intentionally skipped: credential failures are caught at
    initialization and network probing cloud APIs at startup adds latency and
    transient-failure risk. Local routes are different: constructing their SDK
    clients does not prove the daemon is listening, so we perform a cheap
    adapter-owned probe and store the per-route result on ``llm_service`` for
    health endpoints.
    """
    if getattr(llm_service, "disabled", False):
        return

    env_map = env if env is not None else os.environ
    if _env_truthy(env_map.get(SKIP_REACHABILITY_PROBE_ENV_VAR)):
        providers = getattr(llm_service, "providers", None) or []
        llm_service.reachability = [
            {
                "name": p.get("name", p.get("vendor", "?")),
                "vendor": p.get("vendor"),
                "route": p.get("route"),
                "is_local": bool(p.get("is_local")),
                "status": "skipped",
                "message": f"{SKIP_REACHABILITY_PROBE_ENV_VAR}=1",
            }
            for p in providers
        ]
        return

    providers = getattr(llm_service, "providers", None) or []
    if not providers:
        return

    local_routes = [p for p in providers if p.get("is_local")]
    has_cloud_route = any(p.get("is_cloud") for p in providers)
    if not local_routes:
        llm_service.reachability = [
            {
                "name": p.get("name", p.get("vendor", "?")),
                "vendor": p.get("vendor"),
                "route": p.get("route"),
                "is_local": False,
                "status": "skipped",
                "message": "cloud route not probed at startup",
            }
            for p in providers
        ]
        return

    results = []
    reachable = []
    for provider in providers:
        name = provider.get("name", provider.get("vendor", "?"))
        result = {
            "name": name,
            "vendor": provider.get("vendor"),
            "route": provider.get("route"),
            "is_local": bool(provider.get("is_local")),
            "status": "skipped",
            "message": "cloud route not probed at startup",
        }
        if not provider.get("is_local"):
            results.append(result)
            continue

        adapter = provider.get("adapter")
        probe = getattr(adapter, "probe_reachable", None)
        if not callable(probe):
            result["message"] = "adapter has no reachability probe"
            results.append(result)
            logger.warning("Local LLM route %s has no reachability probe", name)
            continue

        try:
            ok = await probe(
                provider.get("client"),
                base_url=provider.get("base_url"),
                timeout=timeout,
            )
        except Exception as exc:  # defensive: probes should not crash startup obscurely
            ok = False
            result["error"] = str(exc)

        if ok is True:
            result["status"] = "reachable"
            result["message"] = "local route probe succeeded"
            reachable.append(name)
        elif ok is False:
            result["status"] = "unreachable"
            result["message"] = "local route probe failed"
            logger.warning("Local LLM route %s is not reachable at startup", name)
        else:
            result["status"] = "skipped"
            result["message"] = "adapter probe unavailable"
            logger.warning("Local LLM route %s did not report reachability", name)
        results.append(result)

    llm_service.reachability = results
    if reachable or has_cloud_route:
        return

    local_names = ", ".join(
        p.get("name", p.get("vendor", "?")) for p in local_routes
    )
    raise NoReachableProvidersError(
        "LLM providers initialized, but no configured local provider was "
        f"reachable at startup ({local_names}). Start the local daemon "
        "(for example `ollama serve`), configure a reachable llama.cpp/"
        "OpenAI-compatible base_url, add a cloud route, or set "
        f"{SKIP_REACHABILITY_PROBE_ENV_VAR}=1 in CI/test contexts."
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
