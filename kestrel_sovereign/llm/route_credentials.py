"""Single source of truth for which env vars satisfy an LLM route.

``kestrel setup``, ``kestrel setup --check`` and ``kestrel doctor`` all
need to answer the same question: *given a route's TOML block, which
``.env`` variables count as a valid credential for it?* Historically
each computed that independently, which let them drift — setup accepted
a management-key-only OpenRouter route that doctor then rejected
(#2245). This module centralizes the rule so they cannot disagree.

A route is considered authenticated when **any** of its accepted
credential env vars is set in ``.env``. An empty list means the route
needs no credential (e.g. a local Ollama route).
"""

from __future__ import annotations

# Per-vendor alternate credential env vars. Some vendors accept more than
# one kind of key. OpenRouter accepts either the standard inference key
# (``OPENROUTER_API_KEY``) or a management key
# (``OPENROUTER_MANAGEMENT_API_KEY``) from which the runtime mints a
# scoped bootstrap inference key. This fallback applies even when the
# route TOML omits ``management_api_key_env``, keeping setup/check/doctor
# in agreement for a management-key-only setup.
_VENDOR_ALT_API_KEY_ENV = {
    "openrouter": "OPENROUTER_MANAGEMENT_API_KEY",
}


def accepted_credential_envs(route_id: str, route: dict) -> list[str]:
    """Return the ordered env var names that satisfy ``route``'s credential.

    The route is authenticated if any returned name is set (non-empty) in
    ``.env``. An empty list means the route needs no credential.

    Resolution order:
      1. ``route["api_key_env"]`` (the primary inference key), if declared.
      2. ``route["management_api_key_env"]`` (an explicitly declared
         management key), if declared.
      3. The static per-vendor alternate for ``route_id``'s vendor, if any
         — applied even when the route TOML omits it.
    """
    envs: list[str] = []

    primary = route.get("api_key_env")
    if primary:
        envs.append(primary)

    management = route.get("management_api_key_env")
    if management and management not in envs:
        envs.append(management)

    vendor_key = route_id.partition(":")[0]
    alt = _VENDOR_ALT_API_KEY_ENV.get(vendor_key)
    if alt and alt not in envs:
        envs.append(alt)

    return envs
