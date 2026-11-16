"""The audit resolves ``model = "auto"`` routes the way generation does.

Every vendor ``kestrel setup --quickstart`` writes is seeded ``model = "auto"``,
which resolves only from the model-discovery cache, and a fresh install has
none. Generation warms that cache on demand (#2069, ``_ensure_models_discovered``);
``get_audit_response`` did not. The genesis self-audit runs during inception,
before any generation, so it found no model on every route and inception
failed with "Genesis audit is pending because no configured auditor completed
it" on a fresh install with a working model. Re-running setup could not help.
Found by the Windows journey harness and reproduced on macOS.
"""

from __future__ import annotations

import hashlib

import pytest

from kestrel_sovereign.constitution.genesis_audit import (
    GENESIS_AUDIT_PASSED,
    evaluate_genesis_constitution,
)
from tests.unit.test_audit_route_rejection import (
    AUDITED_TEXT,
    CONSTITUTION,
    _route,
    _service,
    _StubRoute,
)

DISCOVERED_MODEL = "llama3.2:1b"


def _fresh_install_service(route: _StubRoute, *, is_local: bool = True):
    """One Ollama route seeded ``auto`` and a discovery that resolves it."""
    provider = _route("ollama:local", route, model="auto")
    provider["is_local"] = is_local
    svc = _service([provider])
    discovery_calls: list[str] = []

    async def discover_all_models(use_cache: bool = True):
        # What discovery does to the live provider list: resolve "auto".
        discovery_calls.append("all")
        provider["model"] = DISCOVERED_MODEL
        return []

    async def resolve_local_auto_routes():
        discovery_calls.append("local")
        provider["model"] = DISCOVERED_MODEL

    svc.discover_all_models = discover_all_models
    svc._resolve_local_auto_routes = resolve_local_auto_routes
    return svc, discovery_calls


@pytest.mark.asyncio
async def test_genesis_audit_on_a_fresh_install_resolves_the_auto_route():
    route = _StubRoute(payload={"risk_level": 1, "reasoning": "Sound constitution"})
    svc, discovery_calls = _fresh_install_service(route)

    record = await evaluate_genesis_constitution(
        CONSTITUTION,
        constitution_hash=hashlib.sha256(CONSTITUTION).hexdigest(),
        auditor=svc.get_audit_response,
        provenance="test-fresh-install",
    )

    assert record["status"] == GENESIS_AUDIT_PASSED
    assert route.calls == 1
    assert discovery_calls == ["all"]


@pytest.mark.asyncio
async def test_a_local_only_audit_warms_only_local_routes():
    """Privacy: a local-only audit must not run the cloud-contacting discovery."""
    route = _StubRoute(payload={"risk_level": 1, "reasoning": "ok"})
    svc, discovery_calls = _fresh_install_service(route)

    verdict = await svc.get_audit_response(AUDITED_TEXT, force_local_only=True)

    assert verdict["risk_level"] == 1
    assert discovery_calls == ["local"]


@pytest.mark.asyncio
async def test_a_concrete_model_needs_no_discovery():
    route = _StubRoute(payload={"risk_level": 1, "reasoning": "ok"})
    svc = _service([_route("ollama:local", route, model=DISCOVERED_MODEL)])

    ran: list[bool] = []

    async def record(use_cache: bool = True):
        # Recorded, not raised: the warm-up logs and swallows a discovery
        # exception, so an assertion raised in here could never fail the test.
        ran.append(use_cache)
        return []

    svc.discover_all_models = record

    verdict = await svc.get_audit_response(AUDITED_TEXT)
    assert verdict["risk_level"] == 1
    assert ran == []
