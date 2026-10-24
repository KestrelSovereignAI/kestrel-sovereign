"""Configured native interfaces use the same stable host route geometry."""

import pytest

from kestrel_sovereign.a2a.native_routes import (
    native_http_interfaces,
    native_route_prefix,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.multi_agent.route_name import decode_agent_route_name


def test_interfaces_preserve_registration_and_reverse_proxy_root():
    name = "Team/鳥 ? Display"
    scope = NativeRecipientScope("did:kestrel:recipient", "tenant-alpha")
    prefix = native_route_prefix(name)
    assert decode_agent_route_name(prefix.split("/")[3]) == name
    interfaces = native_http_interfaces(
        "https://host.test/reverse-proxy/", routing_key=name, scope=scope
    )
    assert [item.url for item in interfaces] == [
        "https://host.test/reverse-proxy" + prefix + "/rpc",
        "https://host.test/reverse-proxy" + prefix,
    ]
    assert [item.protocol_binding for item in interfaces] == ["JSONRPC", "HTTP+JSON"]
    assert all(
        item.tenant == scope.tenant and item.protocol_version == "1.0"
        for item in interfaces
    )


@pytest.mark.parametrize(
    "url",
    [
        "",
        "/relative",
        "ftp://host.test",
        "https://",
        "https://user:password@host.test",
        "https://host.test?query=1",
        "https://host.test#fragment",
        "https://host.test/with space",
        "https://host.test:invalid",
        "https://host.test:65536",
    ],
)
def test_invalid_public_urls_fail_before_a_card_or_mount_can_be_published(url):
    with pytest.raises(ValueError):
        native_http_interfaces(
            url, routing_key="registered", scope=NativeRecipientScope("did:recipient")
        )
