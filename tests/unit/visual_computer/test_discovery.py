from types import SimpleNamespace

import pytest
from kestrel_sdk.computer_use import COMPUTER_DRIVER_ENTRY_POINT_GROUP, ENVIRONMENT_PROVIDER_ENTRY_POINT_GROUP
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound
from kestrel_sovereign.visual_computer import discovery


def test_distinct_groups_and_private_factory_settings(monkeypatch):
    seen = []
    provider = SimpleNamespace(capabilities=lambda:None, quote=lambda:None, acquire=lambda:None,
        inspect=lambda:None, release=lambda:None, connect=lambda:None)
    driver = SimpleNamespace(driver_id="driver", open=lambda:None, status=lambda:None, observe=lambda:None,
        act=lambda:None, action_status=lambda:None, control=lambda:None)
    def points(*, group):
        seen.append(group)
        value = provider if group == ENVIRONMENT_PROVIDER_ENTRY_POINT_GROUP else driver
        def factory(*, settings):
            assert settings == {"private":group}
            return value
        return [SimpleNamespace(name="adapter", load=lambda:factory)]
    monkeypatch.setattr(discovery, "entry_points", points)
    providers, drivers = discovery.construct_adapters(providers={"adapter":{"private":ENVIRONMENT_PROVIDER_ENTRY_POINT_GROUP}},
        drivers={"adapter":{"private":COMPUTER_DRIVER_ENTRY_POINT_GROUP}})
    assert providers == {"adapter":provider} and drivers == {"adapter":driver}
    assert seen == [ENVIRONMENT_PROVIDER_ENTRY_POINT_GROUP, COMPUTER_DRIVER_ENTRY_POINT_GROUP]


def test_duplicate_or_missing_factories_fail_without_fallback(monkeypatch):
    point = SimpleNamespace(name="duplicate", load=lambda:lambda **_:None)
    monkeypatch.setattr(discovery, "entry_points", lambda **_:[point,point])
    with pytest.raises(ResourceConflict):
        discovery.discover_factories(COMPUTER_DRIVER_ENTRY_POINT_GROUP)
    monkeypatch.setattr(discovery, "entry_points", lambda **_:[])
    with pytest.raises(ResourceNotFound):
        discovery.construct_adapters(providers={"missing":{}}, drivers={})
    with pytest.raises(ValueError):
        discovery.discover_factories("kestrel_sovereign.cloud_providers")
