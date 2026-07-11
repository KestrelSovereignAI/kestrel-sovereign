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

# Per-vendor default management-key env vars. Some vendors accept more
# than one kind of key. OpenRouter accepts either the standard inference
# key (``OPENROUTER_API_KEY``) or a management key from which the runtime
# mints a scoped bootstrap inference key. The runtime resolves that key as
# ``route["management_api_key_env"] or <this default>`` (see
# ``provider_registry._openrouter_management_key``), so the default only
# applies when the route TOML omits ``management_api_key_env`` — a route
# that declares a custom name is served *only* from that name.
_VENDOR_DEFAULT_MANAGEMENT_ENV = {
    "openrouter": "OPENROUTER_MANAGEMENT_API_KEY",
}


def accepted_credential_envs(route_id: str, route: dict) -> list[str]:
    """Return the ordered env var names that satisfy ``route``'s credential.

    The route is authenticated if any returned name is set (non-empty) in
    ``.env``. An empty list means the route needs no credential.

    Resolution order (mirrors the runtime, which reads
    ``route["management_api_key_env"] or <vendor default>``):
      1. ``route["api_key_env"]`` (the primary inference key), if declared.
      2. ``route["management_api_key_env"]`` (an explicitly declared
         management key), if declared — the vendor default does NOT
         apply in this case, because the runtime won't read it either.
      3. Otherwise the per-vendor default management env for
         ``route_id``'s vendor, if any.
    """
    # Inline credentials satisfy the route outright: the runtime resolves
    # secrets as ``env var or route["api_key"]`` (``_resolve_secret``) and
    # ``env var or route["management_api_key"]``
    # (``_openrouter_management_key``), so a route carrying an inline key
    # needs nothing from ``.env``.
    if route.get("api_key") or route.get("management_api_key"):
        return []

    envs: list[str] = []

    primary = route.get("api_key_env")
    if primary:
        envs.append(primary)

    vendor_key = route_id.partition(":")[0]
    management = route.get("management_api_key_env") or _VENDOR_DEFAULT_MANAGEMENT_ENV.get(
        vendor_key
    )
    if management and management not in envs:
        envs.append(management)

    return envs
