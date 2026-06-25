"""Coordinated request timeouts for local models (issue #1966).

The orchestrator's per-call watchdog must never fire before the LLM client
would for the route serving the call. ``LLMService.effective_request_timeout()``
reports the largest LOCAL provider timeout so the orchestrator can take
``max(its default, that)`` — making a local route's ``timeout`` the single knob.
Cloud routes are deliberately ignored so cloud-only behavior is unchanged.
"""

from types import SimpleNamespace

import httpx
import pytest

from kestrel_sovereign.llm.service import LLMService, _client_timeout_seconds


# --------------------------------------------------------------------------- #
# _client_timeout_seconds normalization
# --------------------------------------------------------------------------- #
def test_timeout_seconds_from_float():
    assert _client_timeout_seconds(1800.0) == 1800.0
    assert _client_timeout_seconds(300) == 300.0


def test_timeout_seconds_from_httpx_timeout_uses_largest_phase():
    # SDK default shape: Timeout(connect=5, read=600, write=600, pool=600)
    t = httpx.Timeout(600.0, connect=5.0)
    assert _client_timeout_seconds(t) == 600.0


def test_timeout_seconds_none():
    assert _client_timeout_seconds(None) is None
    assert _client_timeout_seconds(httpx.Timeout(None)) is None  # all phases None


# --------------------------------------------------------------------------- #
# effective_request_timeout (local-only)
# --------------------------------------------------------------------------- #
def _svc(providers, candidates=None):
    """Exercise the method without building the heavyweight LLMService.

    ``providers`` seeds ``self.providers``; ``candidates`` (when given) is the
    scoped per-call list passed as the argument.
    """
    svc = SimpleNamespace(providers=providers)
    if candidates is None:
        return LLMService.effective_request_timeout(svc)
    return LLMService.effective_request_timeout(svc, candidates)


def test_effective_timeout_scoped_to_candidates_ignores_unrelated_local():
    """The P2 case: a cloud-only candidate set must NOT be lengthened by an
    unrelated local route that exists in the deployment (#1966 review)."""
    local = {"is_local": True, "client": SimpleNamespace(timeout=1800.0)}
    cloud = {"is_local": False, "client": SimpleNamespace(timeout=httpx.Timeout(600.0))}
    # Deployment has a local route, but THIS call resolved to cloud only.
    assert _svc([local, cloud], candidates=[cloud]) is None
    # ...and a call that resolved to the local route still gets it.
    assert _svc([local, cloud], candidates=[local]) == 1800.0


def test_effective_timeout_picks_local_route():
    providers = [{"is_local": True, "client": SimpleNamespace(timeout=1800.0)}]
    assert _svc(providers) == 1800.0


def test_effective_timeout_ignores_cloud_providers():
    providers = [{"is_local": False, "client": SimpleNamespace(timeout=httpx.Timeout(600.0))}]
    assert _svc(providers) is None


def test_effective_timeout_none_for_mixed_set_to_protect_cloud_hang_detection():
    # A mixed candidate set (cloud-primary + local-fallback) must NOT lift the
    # watchdog — the cloud route may be the one tried (#1966 review round 2).
    providers = [
        {"is_local": False, "client": SimpleNamespace(timeout=httpx.Timeout(600.0))},
        {"is_local": True, "client": SimpleNamespace(timeout=1800.0)},
    ]
    assert _svc(providers) is None


def test_effective_timeout_lifts_only_when_all_candidates_local():
    providers = [
        {"is_local": True, "client": SimpleNamespace(timeout=1800.0)},
        {"is_local": True, "client": SimpleNamespace(timeout=900.0)},
    ]
    assert _svc(providers) == 1800.0


def test_effective_timeout_takes_max_across_local_routes():
    providers = [
        {"is_local": True, "client": SimpleNamespace(timeout=1800.0)},
        {"is_local": True, "client": SimpleNamespace(timeout=3600.0)},
    ]
    assert _svc(providers) == 3600.0


def test_effective_timeout_none_when_no_providers():
    assert _svc([]) is None
    assert _svc(None) is None


def test_effective_timeout_handles_local_without_explicit_timeout():
    providers = [{"is_local": True, "client": SimpleNamespace(timeout=None)}]
    assert _svc(providers) is None
