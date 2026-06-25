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
def _svc(providers):
    """Exercise the method without building the heavyweight LLMService."""
    return LLMService.effective_request_timeout(SimpleNamespace(providers=providers))


def test_effective_timeout_picks_local_route():
    providers = [{"is_local": True, "client": SimpleNamespace(timeout=1800.0)}]
    assert _svc(providers) == 1800.0


def test_effective_timeout_ignores_cloud_providers():
    providers = [{"is_local": False, "client": SimpleNamespace(timeout=httpx.Timeout(600.0))}]
    assert _svc(providers) is None


def test_effective_timeout_local_wins_over_cloud_in_mixed_deploy():
    providers = [
        {"is_local": False, "client": SimpleNamespace(timeout=httpx.Timeout(600.0))},
        {"is_local": True, "client": SimpleNamespace(timeout=1800.0)},
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
