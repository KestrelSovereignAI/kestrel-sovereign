"""Source-host gating for the bootstrap API-key endpoint (#1724).

``/api/auth/key`` hands out the live API key for first-run frontend setup. It
previously trusted the ENTIRE Docker bridge range ``172.16.0.0/12`` (~1M
addresses), so any co-resident container on a shared host could fetch sovereign
credentials. This narrows the trust to loopback + the Docker *gateway* only,
with an explicit operator allowlist for custom network topologies — not the
whole bridge subnet.
"""
from __future__ import annotations

import os

# Loopback plus the DEFAULT Docker bridge gateway (172.17.0.1). NOT the whole
# 172.16.0.0/12 range — a sibling container must not be trusted by default.
_DEFAULT_BOOTSTRAP_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "172.17.0.1"})


def bootstrap_allowed_hosts() -> set[str]:
    """The exact set of source hosts allowed to fetch the bootstrap key.

    Operators on a non-default Docker network (custom gateway) add their gateway
    address via ``KESTREL_BOOTSTRAP_ALLOWED_HOSTS`` (comma-separated) rather than
    re-opening the whole bridge subnet.
    """
    extra = os.environ.get("KESTREL_BOOTSTRAP_ALLOWED_HOSTS", "")
    return set(_DEFAULT_BOOTSTRAP_HOSTS) | {
        h.strip() for h in extra.split(",") if h.strip()
    }


def is_bootstrap_host_allowed(client_host: str | None) -> bool:
    """Exact-match the request's socket peer against the allowed set."""
    if not client_host:
        return False
    return client_host in bootstrap_allowed_hosts()
