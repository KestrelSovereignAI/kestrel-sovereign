"""Regression test for the security-feature-lookup bug.

Features including code_edit (extracted), keys, and computer_use historically asked
``self.agent.features.get("security")``, but ``KestrelAgent`` stores
features keyed by class name (``"SecurityFeature"``). The lowercase
key always missed → tools emitted ``"Security feature not available"``
even when the feature was registered.

The code_edit case is now covered by kestrel-feature-code's own test
suite. computer_use exercises the same lookup path here.

The original tests for these features stubbed ``_request_approval``
entirely, so the buggy lookup never ran in CI. This test wires a
realistic agent stub (a dict keyed exactly the way ``KestrelAgent``
keys it) and asserts the lookups resolve.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _StubSecurity:
    """Minimal SecurityFeature stand-in.

    Mirrors the real Feature surface: ``name = __class__.__name__``,
    ``tool_name`` is the auto-derived snake-case (``security_feature``).
    Earlier versions of this stub hand-set ``tool_name = "security"``,
    which made tests pass against a lookup that didn't actually work
    in production. Don't reintroduce that.
    """

    def __init__(self):
        self.name = "SecurityFeature"
        self.approval_queue = MagicMock()

    @property
    def tool_name(self) -> str:
        return "security_feature"


def _agent_with_security():
    """Build an agent whose feature dict mirrors KestrelAgent's layout.

    ``KestrelAgent._register_feature`` does
    ``self.features[feature.name] = feature`` and
    ``Feature.name = self.__class__.__name__`` — so the dict key is the
    class name, not the lowercase tool name.
    """
    sec = _StubSecurity()
    return SimpleNamespace(features={"SecurityFeature": sec}), sec


def test_kestrel_agent_get_feature_resolves_class_name():
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    sec = _StubSecurity()
    agent = SimpleNamespace.__new__(SimpleNamespace)
    agent.features = {"SecurityFeature": sec}

    assert KestrelAgent.get_feature(agent, "SecurityFeature") is sec


def test_kestrel_agent_get_feature_resolves_lowercase_alias():
    """The lowercase ``"security"`` shorthand the broken sites used
    must resolve to the registered SecurityFeature, even though the
    feature's ``tool_name`` is the auto-derived ``"security_feature"``
    (not ``"security"``). This is the case the original fix missed.
    """
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    sec = _StubSecurity()
    agent = SimpleNamespace.__new__(SimpleNamespace)
    agent.features = {"SecurityFeature": sec}

    assert KestrelAgent.get_feature(agent, "security") is sec
    assert KestrelAgent.get_feature(agent, "Security") is sec
    assert KestrelAgent.get_feature(agent, "security_feature") is sec


def test_kestrel_agent_get_feature_resolves_against_real_security_feature():
    """End-to-end with the actual SecurityFeature class. If the lookup
    can't find the real thing, no synthetic stub-based test counts.
    """
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.features.security.feature import SecurityFeature

    sec = SecurityFeature.__new__(SecurityFeature)
    sec.name = "SecurityFeature"
    agent = SimpleNamespace.__new__(SimpleNamespace)
    agent.features = {"SecurityFeature": sec}

    assert KestrelAgent.get_feature(agent, "security") is sec
    assert KestrelAgent.get_feature(agent, "SecurityFeature") is sec


def test_kestrel_agent_get_feature_resolves_tool_name():
    """If a feature defines ``tool_name`` distinct from class name,
    that should also resolve.
    """
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    feat = SimpleNamespace(name="WeirdName", tool_name="custom_tool")
    agent = SimpleNamespace.__new__(SimpleNamespace)
    agent.features = {"WeirdName": feat}

    assert KestrelAgent.get_feature(agent, "custom_tool") is feat


def test_kestrel_agent_get_feature_returns_none_when_missing():
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    agent = SimpleNamespace.__new__(SimpleNamespace)
    agent.features = {}

    assert KestrelAgent.get_feature(agent, "SecurityFeature") is None
    assert KestrelAgent.get_feature(agent, "security") is None
    assert KestrelAgent.get_feature(agent, "") is None


@pytest.mark.parametrize(
    "module_path,handler_attr,handler_kwargs",
    [
        # Each tuple resolves a security lookup that historically returned
        # None against an agent registered the way KestrelAgent registers.
        (
            "kestrel_sovereign.features.computer_use.feature",
            "ComputerUseFeature",
            {},
        ),
    ],
)
def test_feature_security_lookup_finds_registered_security_feature(
    module_path, handler_attr, handler_kwargs, tmp_path,
):
    """Construct each feature with an agent whose features dict is keyed
    the way KestrelAgent keys it, and assert ``_get_security_feature``
    returns the registered SecurityFeature instead of None.
    """
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, handler_attr)

    agent, sec = _agent_with_security()
    # Bolt the canonical lookup helper onto the stub so the
    # ``hasattr(agent, "get_feature")`` branch is exercised — this is
    # the path KestrelAgent provides in production.
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    agent.get_feature = lambda name: KestrelAgent.get_feature(agent, name)

    feat = cls(agent=agent, **handler_kwargs)

    found = feat._get_security_feature()
    assert found is sec, (
        f"{handler_attr}._get_security_feature() returned {found!r}, "
        "expected the registered SecurityFeature"
    )
