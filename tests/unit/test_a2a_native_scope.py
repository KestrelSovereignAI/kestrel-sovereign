"""Server configuration provides one immutable recipient/tenant value."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from kestrel_sovereign.a2a.native_context import NativeRecipientBinding
from kestrel_sovereign.a2a.scope import NativeRecipientScope


@pytest.mark.parametrize("recipient", [None, False, "", " ", 7])
def test_scope_requires_a_concrete_recipient(recipient):
    with pytest.raises(ValueError, match="recipient_id"):
        NativeRecipientScope(recipient)


@pytest.mark.parametrize(
    "tenant",
    [None, False, 7, ".", "..", "a/b", "a%2Fb", "a?b", "a#b", "with space", "café"],
)
def test_scope_rejects_tenants_that_cannot_be_advertised_as_one_opaque_segment(tenant):
    with pytest.raises(ValueError, match="tenant"):
        NativeRecipientScope("did:kestrel:recipient", tenant)


@pytest.mark.parametrize("tenant", ["", "tenant-2", "~abc_1.0"])
def test_native_bindings_share_the_same_storage_scope(tenant):
    expected = NativeRecipientScope("did:kestrel:recipient", tenant)
    rpc = NativeRecipientBinding(
        "routed-recipient", expected.recipient_id, tenant, "JSONRPC"
    )
    rest = NativeRecipientBinding(
        "routed-recipient", expected.recipient_id, tenant, "HTTP+JSON"
    )
    assert rpc.scope == rest.scope == expected
    assert hash(rpc.scope) == hash(rest.scope)


@pytest.mark.parametrize("name,value", [("recipient_id", "other"), ("tenant", "other")])
def test_scope_cannot_change_between_authorization_and_storage(name, value):
    scope = NativeRecipientScope("did:kestrel:recipient", "tenant")
    with pytest.raises(FrozenInstanceError):
        setattr(scope, name, value)
