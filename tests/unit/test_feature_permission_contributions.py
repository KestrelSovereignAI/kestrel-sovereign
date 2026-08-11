"""SecurityFeature consumption of SDK permission-default contributions."""

from enum import Enum
from types import SimpleNamespace

import pytest
from kestrel_sdk.features import (
    ContributionContractError,
    FeaturePermissionDefaults,
    PermissionLevel as SDKPermissionLevel,
)

from kestrel_sovereign.features.contribution_runtime import (
    FeatureContributionCollectionError,
    FeatureContributionRuntime,
    PermissionDefaultRegistration,
    PermissionDefaultsRegistry,
)
from kestrel_sovereign.features.security import feature as security_feature_module
from kestrel_sovereign.features.security.feature import SecurityFeature
from kestrel_sovereign.features.security.permissions import (
    PermissionLevel,
    PermissionStore,
    _LEVEL_RANK,
    assert_sdk_permission_level_parity,
    compose_restrictive_permission,
)
from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature


class _Feature:
    def __init__(self, *tool_names: str) -> None:
        self._tools = tuple(SimpleNamespace(name=name) for name in tool_names)

    def get_tools(self):
        return self._tools


async def _security(
    tmp_path,
    *,
    feature_name: str,
    feature: _Feature,
    defaults: FeaturePermissionDefaults | None,
) -> tuple[SecurityFeature, PermissionStore]:
    registry = PermissionDefaultsRegistry()
    if defaults is not None:
        registry.register(
            PermissionDefaultRegistration(
                owner="tests:permission-owner",
                feature_name=feature_name,
                defaults=defaults,
            )
        )
    agent = SimpleNamespace(
        features={feature_name: feature},
        permission_defaults_registry=registry,
    )
    security = SecurityFeature(agent)
    store = PermissionStore(str(tmp_path / "permissions.db"))
    await store.initialize()
    security.permission_store = store
    return security, store


def test_sdk_and_sovereign_permission_enforcement_vocabularies_are_closed():
    assert {level.name: level.value for level in SDKPermissionLevel} == {
        level.name: level.value for level in PermissionLevel
    }
    assert set(_LEVEL_RANK) == set(PermissionLevel)
    assert_sdk_permission_level_parity()


@pytest.mark.parametrize(
    ("static_level", "expected_by_declaration"),
    [
        (
            PermissionLevel.ALLOW,
            (PermissionLevel.ALLOW,) * 4
            + (PermissionLevel.ALWAYS_ASK, PermissionLevel.DENY),
        ),
        (
            PermissionLevel.AUTO,
            (PermissionLevel.AUTO,) * 4
            + (PermissionLevel.ALWAYS_ASK, PermissionLevel.DENY),
        ),
        (
            PermissionLevel.DENY,
            (PermissionLevel.DENY,) * 6,
        ),
        (
            PermissionLevel.ALWAYS_ASK,
            (PermissionLevel.ALWAYS_ASK,) * 5 + (PermissionLevel.DENY,),
        ),
        (
            PermissionLevel.ASK,
            (PermissionLevel.ASK,) * 4
            + (PermissionLevel.ALWAYS_ASK, PermissionLevel.DENY),
        ),
        (
            PermissionLevel.SESSION,
            (PermissionLevel.SESSION,) * 4
            + (PermissionLevel.ALWAYS_ASK, PermissionLevel.DENY),
        ),
    ],
)
def test_static_and_declared_defaults_compose_by_restrictiveness(
    static_level,
    expected_by_declaration,
):
    declared_levels = (
        PermissionLevel.ALLOW,
        PermissionLevel.AUTO,
        PermissionLevel.SESSION,
        PermissionLevel.ASK,
        PermissionLevel.ALWAYS_ASK,
        PermissionLevel.DENY,
    )
    assert tuple(
        compose_restrictive_permission(static_level, declared_level)
        for declared_level in declared_levels
    ) == expected_by_declaration


def test_permission_parity_fails_closed_on_unknown_or_unenforced_values(
    monkeypatch,
):
    import kestrel_sdk.features as sdk_features

    class FutureSDKPermissionLevel(Enum):
        ALLOW = "allow"
        AUTO = "auto"
        DENY = "deny"
        ALWAYS_ASK = "always_ask"
        ASK = "ask"
        SESSION = "session"
        FUTURE = "future"

    monkeypatch.setattr(sdk_features, "PermissionLevel", FutureSDKPermissionLevel)
    with pytest.raises(RuntimeError, match="vocabulary mismatch"):
        assert_sdk_permission_level_parity()

    monkeypatch.setattr(sdk_features, "PermissionLevel", SDKPermissionLevel)
    monkeypatch.delitem(_LEVEL_RANK, PermissionLevel.SESSION)
    with pytest.raises(RuntimeError, match="vocabulary mismatch"):
        assert_sdk_permission_level_parity()


def test_unknown_declared_tool_override_is_rejected_by_sdk_validation():
    feature = SDKFixtureFeature(
        SimpleNamespace(did="did:test:permissions", agent_id="did:test:permissions")
    )
    permission_defaults = FeaturePermissionDefaults(
        tool_overrides={"not_a_real_tool": SDKPermissionLevel.ALLOW}
    )
    feature.permission_defaults = permission_defaults

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        FeatureContributionRuntime._collect(feature)

    error = exc_info.value
    assert error.feature is feature
    assert error.stage == "contribution validation"
    assert error.getter == "validate_feature_contributions"
    assert str(error) == (
        "feature contribution failure during contribution validation "
        "(validate_feature_contributions)"
    )
    assert "not_a_real_tool" not in str(error)
    assert type(error.__cause__) is ContributionContractError
    assert str(error.__cause__) == (
        "permission overrides reference unknown feature tools: not_a_real_tool"
    )
    assert feature.permission_defaults is permission_defaults
    assert not feature.initialized
    assert not feature.disabled


@pytest.mark.asyncio
async def test_contributed_defaults_cannot_weaken_static_destructive_rail(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_DEMO_SERVER", "1")
    feature = _Feature("purge_conversation", "ordinary_tool")
    security, store = await _security(
        tmp_path,
        feature_name="MemoryFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(
            feature_default=SDKPermissionLevel.ASK,
            tool_overrides={
                # An extracted declaration cannot weaken Sovereign's static
                # destructive-tool floor.
                "purge_conversation": SDKPermissionLevel.ALLOW,
            },
        ),
    )

    await security._register_all_tools()

    assert (
        await store.get_permission("MemoryFeature", "purge_conversation")
        is PermissionLevel.ALWAYS_ASK
    )
    assert (
        await store.get_permission("MemoryFeature", "ordinary_tool")
        is PermissionLevel.ALLOW
    )


@pytest.mark.asyncio
async def test_demo_allow_baseline_composes_with_sdk_permission_declarations(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("KESTREL_DEMO_SERVER", "1")
    feature = _Feature(
        "ordinary",
        "explicit_ask",
        "prompt_every_time",
        "blocked",
    )
    security, store = await _security(
        tmp_path,
        feature_name="ExtractedFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(
            # The SDK's conservative feature default must not erase the demo
            # contract. Explicit tool declarations still take precedence.
            feature_default=SDKPermissionLevel.ASK,
            tool_overrides={
                "explicit_ask": SDKPermissionLevel.ASK,
                "prompt_every_time": SDKPermissionLevel.ALWAYS_ASK,
                "blocked": SDKPermissionLevel.DENY,
            },
        ),
    )

    with caplog.at_level("INFO"):
        await security._register_all_tools()

    assert (
        await store.get_permission("ExtractedFeature", "ordinary")
        is PermissionLevel.ALLOW
    )
    assert (
        await store.get_permission("ExtractedFeature", "explicit_ask")
        is PermissionLevel.ASK
    )
    assert (
        await store.get_permission("ExtractedFeature", "prompt_every_time")
        is PermissionLevel.ALWAYS_ASK
    )
    assert (
        await store.get_permission("ExtractedFeature", "blocked")
        is PermissionLevel.DENY
    )
    assert "demo permission baseline ALLOW" in caplog.text
    assert "may tighten it" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("declared_level", "expected_level"),
    [
        (SDKPermissionLevel.ALWAYS_ASK, PermissionLevel.ALWAYS_ASK),
        (SDKPermissionLevel.DENY, PermissionLevel.DENY),
    ],
)
async def test_demo_baseline_accepts_feature_wide_hard_tightening(
    tmp_path, monkeypatch, declared_level, expected_level
):
    monkeypatch.setenv("KESTREL_DEMO_SERVER", "1")
    feature = _Feature("ordinary")
    security, store = await _security(
        tmp_path,
        feature_name="ExtractedFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(feature_default=declared_level),
    )

    await security._register_all_tools()

    assert (
        await store.get_permission("ExtractedFeature", "ordinary")
        is expected_level
    )


@pytest.mark.asyncio
async def test_extracted_talon_allow_cannot_weaken_static_feature_ask(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("KESTREL_DEMO_SERVER", raising=False)
    feature = _Feature("talon_claim")
    security, store = await _security(
        tmp_path,
        feature_name="TalonCoordinatorFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(
            feature_default=SDKPermissionLevel.ALLOW,
        ),
    )

    await security._register_all_tools()

    assert (
        await store.get_permission("TalonCoordinatorFeature", "talon_claim")
        is PermissionLevel.ASK
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature_name", "tool_name"),
    [
        ("TalonCoordinatorFeature", "talon_claim"),
        ("KeyManagementFeature", "list_service_keys"),
    ],
)
async def test_mapped_tool_allow_cannot_bypass_feature_ask(
    tmp_path,
    monkeypatch,
    feature_name,
    tool_name,
):
    """A contributed tool override is still bounded by a mapped feature rail."""
    monkeypatch.delenv("KESTREL_DEMO_SERVER", raising=False)
    feature = _Feature(tool_name)
    security, store = await _security(
        tmp_path,
        feature_name=feature_name,
        feature=feature,
        defaults=FeaturePermissionDefaults(
            tool_overrides={tool_name: SDKPermissionLevel.ALLOW},
        ),
    )

    await security._register_all_tools()

    assert (
        await store.get_permission(feature_name, tool_name)
        is PermissionLevel.ASK
    )


@pytest.mark.asyncio
async def test_declared_feature_default_can_tighten_static_feature_ask(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("KESTREL_DEMO_SERVER", raising=False)
    feature = _Feature("talon_claim")
    security, store = await _security(
        tmp_path,
        feature_name="TalonCoordinatorFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(
            feature_default=SDKPermissionLevel.ALWAYS_ASK,
        ),
    )

    await security._register_all_tools()

    assert (
        await store.get_permission("TalonCoordinatorFeature", "talon_claim")
        is PermissionLevel.ALWAYS_ASK
    )


@pytest.mark.asyncio
async def test_unmapped_feature_retains_its_declared_default(tmp_path, monkeypatch):
    monkeypatch.delenv("KESTREL_DEMO_SERVER", raising=False)
    feature = _Feature("ordinary")
    security, store = await _security(
        tmp_path,
        feature_name="UnmappedExtractedFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(
            feature_default=SDKPermissionLevel.ALLOW,
        ),
    )

    await security._register_all_tools()

    assert (
        await store.get_permission("UnmappedExtractedFeature", "ordinary")
        is PermissionLevel.ALLOW
    )


@pytest.mark.asyncio
async def test_unmapped_feature_retains_declared_tool_allow(tmp_path, monkeypatch):
    monkeypatch.delenv("KESTREL_DEMO_SERVER", raising=False)
    feature = _Feature("ordinary")
    security, store = await _security(
        tmp_path,
        feature_name="UnmappedExtractedFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(
            tool_overrides={"ordinary": SDKPermissionLevel.ALLOW},
        ),
    )

    await security._register_all_tools()

    assert (
        await store.get_permission("UnmappedExtractedFeature", "ordinary")
        is PermissionLevel.ALLOW
    )


@pytest.mark.asyncio
async def test_future_static_ask_rail_rejects_and_migrates_declared_allow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(
        security_feature_module._DEFAULT_PERMISSION_BY_TOOL,
        "ExtractedFeature",
        {"guarded": PermissionLevel.ASK},
    )
    feature = _Feature("guarded")
    security, store = await _security(
        tmp_path,
        feature_name="ExtractedFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(
            feature_default=SDKPermissionLevel.ALLOW,
            tool_overrides={"guarded": SDKPermissionLevel.ALLOW},
        ),
    )
    await store.register_tool(
        "ExtractedFeature", "guarded", PermissionLevel.ALLOW
    )

    await security._register_all_tools()

    assert (
        await store.get_permission("ExtractedFeature", "guarded")
        is PermissionLevel.ASK
    )


@pytest.mark.asyncio
async def test_contributed_hard_rails_upgrade_permissive_rows_and_preserve_deny(
    tmp_path,
):
    feature = _Feature("prompt_each_time", "blocked", "ordinary")
    security, store = await _security(
        tmp_path,
        feature_name="ExtractedFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(
            feature_default=SDKPermissionLevel.ALLOW,
            tool_overrides={
                "prompt_each_time": SDKPermissionLevel.ALWAYS_ASK,
                "blocked": SDKPermissionLevel.DENY,
            },
        ),
    )
    await store.register_tool(
        "ExtractedFeature", "prompt_each_time", PermissionLevel.ALLOW
    )
    await store.register_tool(
        "ExtractedFeature", "blocked", PermissionLevel.ALWAYS_ASK
    )
    await store.register_tool("ExtractedFeature", "ordinary", PermissionLevel.ASK)

    await security._register_all_tools()

    assert (
        await store.get_permission("ExtractedFeature", "prompt_each_time")
        is PermissionLevel.ALWAYS_ASK
    )
    assert (
        await store.get_permission("ExtractedFeature", "blocked")
        is PermissionLevel.DENY
    )
    # ALLOW is a normal first-registration default, not a hardening request.
    assert (
        await store.get_permission("ExtractedFeature", "ordinary")
        is PermissionLevel.ASK
    )


@pytest.mark.asyncio
async def test_contributed_always_ask_preserves_persisted_stricter_deny(tmp_path):
    feature = _Feature("dangerous")
    security, store = await _security(
        tmp_path,
        feature_name="ExtractedFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(
            tool_overrides={"dangerous": SDKPermissionLevel.ALWAYS_ASK}
        ),
    )
    await store.register_tool("ExtractedFeature", "dangerous", PermissionLevel.DENY)

    await security._register_all_tools()

    assert (
        await store.get_permission("ExtractedFeature", "dangerous")
        is PermissionLevel.DENY
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("declared_level", "expected_level"),
    [
        (SDKPermissionLevel.ALWAYS_ASK, PermissionLevel.ALWAYS_ASK),
        (SDKPermissionLevel.DENY, PermissionLevel.DENY),
    ],
)
async def test_contributed_hard_feature_default_hardens_dispatch_row(
    tmp_path,
    declared_level,
    expected_level,
):
    feature = _Feature("ordinary")
    feature.tool_name = "dispatch_feature"
    security, store = await _security(
        tmp_path,
        feature_name="ExtractedFeature",
        feature=feature,
        defaults=FeaturePermissionDefaults(feature_default=declared_level),
    )
    await store.register_tool(
        "ExtractedFeature", "dispatch_feature", PermissionLevel.ALLOW
    )

    await security._register_all_tools()

    assert (
        await store.get_permission("ExtractedFeature", "dispatch_feature")
        is expected_level
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature_name", "tool_name"),
    [
        ("TalonCoordinatorFeature", "scan_stale_work"),
        ("InferenceLeaseFeature", "inference_lease_status"),
        ("InferenceLeaseFeature", "inference_lease_release"),
    ],
)
async def test_static_allow_migrations_upgrade_existing_ask_rows(
    tmp_path, feature_name, tool_name
):
    feature = _Feature(tool_name)
    security, store = await _security(
        tmp_path,
        feature_name=feature_name,
        feature=feature,
        defaults=FeaturePermissionDefaults(
            feature_default=SDKPermissionLevel.ASK,
        ),
    )
    await store.register_tool(feature_name, tool_name, PermissionLevel.ASK)

    await security._register_all_tools()

    assert (
        await store.get_permission(feature_name, tool_name)
        is PermissionLevel.ALLOW
    )
