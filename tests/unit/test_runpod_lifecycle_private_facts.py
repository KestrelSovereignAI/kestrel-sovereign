"""Private fixed lifecycle facts do not imply a released lease or authority."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from kestrel_sovereign.visual_computer.runpod_quote_storage import (
    _RunpodInspection,
    _RunpodReleaseFact,
)


def fact(**change):
    return _RunpodReleaseFact(intent_id="a"*32, provider_resource_id="owned", requested_at=datetime.now(UTC),
        state="intent", **change)


def test_private_release_fact_hides_identity_and_time():
    value = fact()
    assert value.model_dump() == {"state": "intent", "native_status": None}
    assert "owned" not in repr(value)


@pytest.mark.parametrize("change", [
    {"native_status":204}, {"acknowledged_at":datetime.now(UTC)},
    {"requested_at":datetime.now(UTC).replace(tzinfo=None)}, {"provider_resource_id":""}, {"intent_id":"changed"},
])
def test_malformed_release_fact_refuses(change):
    values={"intent_id":"a"*32, "provider_resource_id":"owned", "requested_at":datetime.now(UTC), "state":"intent"}
    with pytest.raises(ValidationError):
        _RunpodReleaseFact(**(values|change))


@pytest.mark.parametrize("change", [
    {}, {"native_status":204}, {"acknowledged_at":datetime.now(UTC)},
    {"native_status":204,"acknowledged_at":datetime.now(UTC).replace(tzinfo=None)},
    {"native_status":204,"acknowledged_at":datetime.now(UTC)-timedelta(days=1)},
])
def test_partial_or_reordered_terminal_fact_refuses(change):
    values={"intent_id":"a"*32,"provider_resource_id":"owned","requested_at":datetime.now(UTC),"state":"acknowledged"}
    with pytest.raises(ValidationError):
        _RunpodReleaseFact(**(values|change))


def test_inspection_requires_aware_time():
    with pytest.raises(ValidationError):
        _RunpodInspection(provider_resource_id="owned", observed_at=datetime.now(UTC).replace(tzinfo=None), state="running")
