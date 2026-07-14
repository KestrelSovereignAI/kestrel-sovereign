"""Tests for the identity→tenant resolver at the host auth edge (issue #2444)."""

import uuid
from types import SimpleNamespace

import pytest

from kestrel_sovereign.auth import CallerContext, AuthMethod
from kestrel_sovereign.security.tenant_resolver import (
    DEFAULT_PERSONAL_TENANT_ID,
    HOST_CONFIG_KEY,
    build_tenant_resolver,
    resolve_tenant,
    tenant_id_for_identity,
)


def _request(caller=None, session=None):
    """Minimal stand-in for a Starlette Request the resolver reads."""
    state = SimpleNamespace()
    if caller is not None:
        state.caller = caller
    req = SimpleNamespace(state=state)
    if session is not None:
        req.session = session
    return req


def test_default_tenant_is_stable_uuid():
    assert isinstance(DEFAULT_PERSONAL_TENANT_ID, uuid.UUID)
    # Stable across calls — derived from a fixed namespace, not random.
    assert tenant_id_for_identity(None) == DEFAULT_PERSONAL_TENANT_ID
    assert tenant_id_for_identity("") == DEFAULT_PERSONAL_TENANT_ID
    assert tenant_id_for_identity("   ") == DEFAULT_PERSONAL_TENANT_ID


def test_solo_owner_resolves_to_default_tenant():
    """INV-SOLO: sovereign API key / anonymous / no-caller → one default tenant."""
    # Sovereign API-key caller (the solo owner / machine emitter).
    sovereign = _request(caller=CallerContext.sovereign(AuthMethod.API_KEY))
    assert resolve_tenant(sovereign) == DEFAULT_PERSONAL_TENANT_ID

    # Anonymous / internal caller.
    anon = _request(caller=CallerContext.anonymous())
    assert resolve_tenant(anon) == DEFAULT_PERSONAL_TENANT_ID

    # No caller attached at all, no session.
    bare = _request()
    assert resolve_tenant(bare) == DEFAULT_PERSONAL_TENANT_ID


def test_distinct_authenticated_users_resolve_to_distinct_tenants():
    """Acceptance: two distinct principals → two tenants (store isolates them)."""
    alice = _request(
        caller=CallerContext.authenticated("alice@example.com", AuthMethod.OAUTH_SESSION)
    )
    bob = _request(
        caller=CallerContext.authenticated("bob@example.com", AuthMethod.JWT)
    )

    t_alice = resolve_tenant(alice)
    t_bob = resolve_tenant(bob)

    assert isinstance(t_alice, uuid.UUID)
    assert isinstance(t_bob, uuid.UUID)
    assert t_alice != t_bob
    # Neither distinct user is the solo default tenant.
    assert t_alice != DEFAULT_PERSONAL_TENANT_ID
    assert t_bob != DEFAULT_PERSONAL_TENANT_ID


def test_same_user_is_deterministic_and_case_insensitive():
    a = _request(caller=CallerContext.authenticated("User@Example.com"))
    b = _request(caller=CallerContext.authenticated("user@example.com"))
    assert resolve_tenant(a) == resolve_tenant(b)


def test_session_cookie_fallback_when_no_caller():
    """Cookie session (no CallerContext yet) still resolves the user's tenant."""
    req = _request(session={"user_email": "carol@example.com"})
    tenant = resolve_tenant(req)
    assert tenant == tenant_id_for_identity("carol@example.com")
    assert tenant != DEFAULT_PERSONAL_TENANT_ID


def test_build_tenant_resolver_returns_callable_matching_seam():
    resolver = build_tenant_resolver()
    assert callable(resolver)
    result = resolver(_request(caller=CallerContext.anonymous()))
    assert result == DEFAULT_PERSONAL_TENANT_ID


def test_host_config_mapping_injects_resolver():
    """server._host_config_mapping wires the resolver under the seam key."""
    from kestrel_sovereign.server import _host_config_mapping

    # No config (single-agent boot) still injects the resolver.
    mapping = _host_config_mapping(None)
    assert HOST_CONFIG_KEY in mapping
    assert callable(mapping[HOST_CONFIG_KEY])
    assert mapping[HOST_CONFIG_KEY](_request(caller=CallerContext.anonymous())) == (
        DEFAULT_PERSONAL_TENANT_ID
    )

    # With a multi-agent-style config the resolver is still present alongside
    # the host bind/port/agents keys.
    cfg = SimpleNamespace(
        host=SimpleNamespace(bind="0.0.0.0", port=8888),
        agents={"Kestrel": object()},
    )
    mapping2 = _host_config_mapping(cfg)
    assert HOST_CONFIG_KEY in mapping2
    assert mapping2["host_port"] == 8888
    assert mapping2["agents"] == ["Kestrel"]
