"""Tests for the startup lifecycle hardening rails (#377, #381, #406).

These are the three checks that prevent the failure modes Emma v1/v2 and
Meridian hit during the Feb-Mar 2026 incidents:

- #377: agent boots with zero providers, runs mute for 2 weeks
- #381: agent process opens another agent's database, no validation
- #406: brand-new agent paralyzed in approval-modal loop on first turn
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.lifecycle_checks import (
    EXPECTED_DID_ENV_VAR,
    IdentityIsolationError,
    NoLLMProvidersError,
    verify_identity_isolation,
    verify_llm_providers_initialized,
)


# ---------------------------------------------------------------------------
# #377 — refuse to declare startup success when zero LLM providers came up
# ---------------------------------------------------------------------------


class _FakeLLMService:
    def __init__(self, providers):
        self.providers = providers


def test_provider_check_passes_when_at_least_one_provider_initialized():
    svc = _FakeLLMService(providers=["anthropic:api"])
    # Must not raise.
    verify_llm_providers_initialized(svc)


def test_provider_check_passes_with_multiple_providers():
    svc = _FakeLLMService(providers=["anthropic:api", "openai:api", "ollama:local"])
    verify_llm_providers_initialized(svc)


def test_provider_check_raises_when_zero_providers():
    svc = _FakeLLMService(providers=[])
    with pytest.raises(NoLLMProvidersError) as exc:
        verify_llm_providers_initialized(svc)
    # Error message must name the concrete env vars an operator can set,
    # so a fresh deployer sees the fix path rather than just "agent broken."
    msg = str(exc.value)
    assert "ANTHROPIC_API_KEY" in msg
    assert "OPENAI_API_KEY" in msg


def test_provider_check_raises_when_providers_attr_missing():
    """A service that never ran initialize_providers (no attr) is also broken."""
    class _Empty:
        pass

    with pytest.raises(NoLLMProvidersError):
        verify_llm_providers_initialized(_Empty())


def test_provider_check_raises_when_providers_is_none():
    svc = _FakeLLMService(providers=None)
    with pytest.raises(NoLLMProvidersError):
        verify_llm_providers_initialized(svc)


def test_provider_check_skipped_when_llm_service_disabled():
    """PayerKind.NONE: zero providers is the intended state — don't raise."""
    svc = _FakeLLMService(providers=[])
    svc.disabled = True
    # Must not raise — operator has explicitly disabled LLM use for this agent.
    verify_llm_providers_initialized(svc)


def test_provider_check_still_raises_when_disabled_false_and_no_providers():
    """Explicitly-enabled LLM with no providers is still a failure."""
    svc = _FakeLLMService(providers=[])
    svc.disabled = False
    with pytest.raises(NoLLMProvidersError):
        verify_llm_providers_initialized(svc)


def test_provider_check_message_mentions_payer_policy_escape_hatch():
    svc = _FakeLLMService(providers=[])
    with pytest.raises(NoLLMProvidersError) as exc:
        verify_llm_providers_initialized(svc)
    # Operators reading the error should see the path to a deliberate-disable
    # configuration, not just "missing key" diagnostics.
    assert "PayerPolicy.llm.kind = NONE" in str(exc.value)


# ---------------------------------------------------------------------------
# #381 — refuse to start if the DB's agent DID differs from the operator's
# declaration (KESTREL_EXPECTED_DID env var)
# ---------------------------------------------------------------------------


def test_identity_check_noop_when_env_var_unset(monkeypatch):
    """Single-agent dev setups don't pre-declare an identity → no-op."""
    monkeypatch.delenv(EXPECTED_DID_ENV_VAR, raising=False)
    # Even with an unusual DB DID, no exception when operator didn't declare one.
    verify_identity_isolation("did:web:emma.example/agent")


def test_identity_check_noop_when_env_var_empty(monkeypatch):
    monkeypatch.setenv(EXPECTED_DID_ENV_VAR, "")
    verify_identity_isolation("did:web:emma.example/agent")


def test_identity_check_passes_when_db_matches_expected(monkeypatch):
    monkeypatch.setenv(EXPECTED_DID_ENV_VAR, "did:web:emma.example/agent")
    verify_identity_isolation("did:web:emma.example/agent")


def test_identity_check_raises_on_mismatch(monkeypatch):
    """Claw process pointed at Emma's database → refuse to start."""
    monkeypatch.setenv(EXPECTED_DID_ENV_VAR, "did:web:claw.example/agent")
    with pytest.raises(IdentityIsolationError) as exc:
        verify_identity_isolation("did:web:emma.example/agent")
    msg = str(exc.value)
    assert "did:web:emma.example/agent" in msg
    assert "did:web:claw.example/agent" in msg
    assert "Refusing to start" in msg


def test_identity_check_explicit_expected_overrides_env(monkeypatch):
    """The explicit kwarg wins (used by tests + multi-agent loader)."""
    monkeypatch.setenv(EXPECTED_DID_ENV_VAR, "did:web:from-env/agent")
    # Match: explicit expected agrees with db_did even when env disagrees.
    verify_identity_isolation(
        "did:web:override/agent",
        expected_did="did:web:override/agent",
    )
    # Mismatch via explicit expected raises.
    with pytest.raises(IdentityIsolationError):
        verify_identity_isolation(
            "did:web:db-says/agent",
            expected_did="did:web:caller-says/agent",
        )


def test_identity_check_strips_whitespace(monkeypatch):
    """Trailing whitespace in env var must not cause spurious mismatches."""
    monkeypatch.setenv(EXPECTED_DID_ENV_VAR, "  did:web:agent  ")
    verify_identity_isolation("did:web:agent")


def test_identity_check_treats_whitespace_only_env_as_unset(monkeypatch):
    monkeypatch.setenv(EXPECTED_DID_ENV_VAR, "   ")
    # Empty after strip → no-op.
    verify_identity_isolation("anything")


# ---------------------------------------------------------------------------
# #406 — per-feature default permission map
# ---------------------------------------------------------------------------


def test_core_features_default_to_allow():
    """Boot-critical features must not paralyze a fresh agent (Meridian #406).

    Names MUST match actual ``Feature.name`` (Python class name) values —
    aspirational names would silently fall through to ASK. Codex review on
    the first revision caught five name mismatches; this test pins the names.
    """
    from kestrel_sovereign.features.security.feature import (
        default_permission_for_feature,
    )
    from kestrel_sovereign.features.security.permissions import PermissionLevel

    core_must_allow = [
        "BootstrapFeature",
        "IdentityFeature",
        "ConstitutionFeature",
        "MemoryFeature",
        "MemoryAgencyFeature",
        "StrategicMemoryFeature",
        "SovereigntyFeature",
        "ContextFeature",
        "HealthFeature",
        "ModelAgent",            # NOT ModelFeature
        "SaveFeature",
        "KeyManagementFeature",  # NOT KeysFeature
        "TaskFeature",           # NOT TasksFeature
        "ChannelFeature",        # NOT ChannelsFeature
        "CliFeature",
        "WorkflowsFeature",
        "FeatureFeaturesFeature",
    ]
    for name in core_must_allow:
        assert default_permission_for_feature(name) == PermissionLevel.ALLOW, (
            f"{name} should default to ALLOW (core boot feature) so a fresh "
            "agent isn't paralyzed in an approval loop."
        )


def test_risky_features_default_to_ask():
    """Externally-visible / irreversible features stay on ASK."""
    from kestrel_sovereign.features.security.feature import (
        default_permission_for_feature,
    )
    from kestrel_sovereign.features.security.permissions import PermissionLevel

    must_ask = [
        "ComputeFeature",          # arbitrary code execution
        "ComputerUseFeature",      # arbitrary screen/keyboard actions
        "SpawnFeature",            # creates new agents
        "DeliveryFeature",         # sends external messages
        "WebhookFeature",          # external network egress; NOT WebhooksFeature
        "BridgeFeature",           # cross-agent escalation
        "DeployFeature",           # deploys infrastructure
        "TalonCoordinatorFeature", # spawns workspaces and runs codex jobs
    ]
    for name in must_ask:
        assert default_permission_for_feature(name) == PermissionLevel.ASK, (
            f"{name} should default to ASK — it has external side effects."
        )


def test_default_map_keys_match_real_feature_class_names():
    """Every key in the default-permission map must match an actual class name
    in kestrel_sovereign.features. If a key drifts (e.g. someone writes
    "ModelFeature" when the class is "ModelAgent"), this fails — far better
    than silently falling through to ASK on production agents.
    """
    import importlib
    import pkgutil
    from pathlib import Path

    from kestrel_sovereign.features.security.feature import (
        _DEFAULT_PERMISSION_BY_FEATURE,
    )

    # Walk the features package and collect every class name defined in
    # kestrel_sovereign/features/**/feature.py (and top-level modules).
    features_pkg = importlib.import_module("kestrel_sovereign.features")
    features_root = Path(features_pkg.__file__).parent

    known_class_names: set[str] = set()
    for py_file in features_root.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.lstrip()
            if not stripped.startswith("class "):
                continue
            head = stripped[len("class "):]
            name = head.split("(", 1)[0].split(":", 1)[0].strip()
            if name and (name.endswith("Feature") or name.endswith("Agent")):
                known_class_names.add(name)

    map_keys = set(_DEFAULT_PERMISSION_BY_FEATURE.keys())
    missing = map_keys - known_class_names
    assert not missing, (
        f"Default-permission map has keys with no matching Feature/Agent "
        f"class in the source tree: {sorted(missing)}. Either fix the typo "
        f"(common case: 'ModelFeature' should be 'ModelAgent') or remove the "
        f"entry. A mismatched key silently falls through to ASK."
    )


def test_unmapped_features_fall_back_to_ask():
    """Adding a new feature must NOT silently grant it ALLOW."""
    from kestrel_sovereign.features.security.feature import (
        default_permission_for_feature,
    )
    from kestrel_sovereign.features.security.permissions import PermissionLevel

    assert (
        default_permission_for_feature("SomeBrandNewFutureFeature")
        == PermissionLevel.ASK
    )


def test_explicit_fallback_can_override():
    from kestrel_sovereign.features.security.feature import (
        default_permission_for_feature,
    )
    from kestrel_sovereign.features.security.permissions import PermissionLevel

    # An unmapped feature with a caller-supplied fallback uses the fallback.
    assert (
        default_permission_for_feature("Unmapped", fallback=PermissionLevel.DENY)
        == PermissionLevel.DENY
    )
    # A mapped feature ignores the fallback (mapping is authoritative).
    assert (
        default_permission_for_feature(
            "BootstrapFeature", fallback=PermissionLevel.DENY
        )
        == PermissionLevel.ALLOW
    )
