"""Operational push policy is explicit, validated and independent of old pins."""

from dataclasses import fields

import pytest

from kestrel_sovereign.a2a.native_publication import NativeHostingConfiguration
from kestrel_sovereign.a2a.push_policy import NativePushPolicy


def test_public_host_enables_push_with_one_canonical_default_policy():
    configured = NativeHostingConfiguration.from_environment(
        {"KESTREL_BASE_URL": "https://host.test"}
    )
    assert configured.push == NativePushPolicy()
    assert (
        NativePushPolicy.from_environment({"KESTREL_A2A_PUSH_ENABLED": "false"}) is None
    )


@pytest.mark.parametrize("field", [field.name for field in fields(NativePushPolicy)])
def test_each_policy_setting_is_configurable_by_environment(field):
    value = getattr(NativePushPolicy(), field) + 1
    configured = NativePushPolicy.from_environment(
        {"KESTREL_A2A_PUSH_" + field.upper(): str(value)}
    )
    assert getattr(configured, field) == value


@pytest.mark.parametrize("value", [0, -1, True, "1", 1.5])
@pytest.mark.parametrize("field", [field.name for field in fields(NativePushPolicy)])
def test_nonpositive_or_wrong_typed_settings_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        NativePushPolicy(**{field: value})


@pytest.mark.parametrize(
    "environment",
    [
        {"KESTREL_A2A_PUSH_ENABLED": "maybe"},
        {"KESTREL_A2A_PUSH_TIMEOUT_SECONDS": "forever"},
        {"KESTREL_A2A_PUSH_RENEWAL_SECONDS": "60"},
    ],
)
def test_invalid_operator_policy_fails_before_publication(environment):
    with pytest.raises(ValueError):
        NativePushPolicy.from_environment(environment)


def test_embedding_requires_a_typed_push_policy():
    with pytest.raises(TypeError, match="push policy"):
        NativeHostingConfiguration("https://host.test", push=True)
