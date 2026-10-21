"""The actual server onboarding hook retains scoped standalone/process policy."""

from types import SimpleNamespace

from kestrel_sovereign.a2a.did_registry import (
    A2A_PEER_STABLE_AGENT_ID_FIELD,
    ProcessA2ADidResolver,
)
from kestrel_sovereign.features.peers.directory import LocalHostPeerDirectory
from kestrel_sovereign.features.peers.feature import PeersFeature
from tests.unit.test_a2a_native_standalone import (
    single_candidate as single_candidate,  # noqa: PLC0414 - fixture re-export
)
from tests.unit.test_a2a_native_standalone import (
    single_identity as single_identity,  # noqa: PLC0414 - fixture dependency
)
from tests.utils.native_a2a_security import PRINCIPAL, SENDER


async def test_actual_single_onboarding_publishes_with_the_installed_scoped_policy(
    single_candidate,
):
    from kestrel_sovereign.server import _onboard_single_agent_native

    p = single_candidate
    original = p.recipient.a2a_inbound_sender_authorizer
    publication = await _onboard_single_agent_native(
        p.app,
        p.recipient,
        {
            "KESTREL_BASE_URL": "https://single.native.test",
            "KESTREL_A2A_PUSH_ENABLED": "false",
            "KESTREL_A2A_DISCOVERY_AGENT": "standalone-key",
        },
    )
    owner = p.recipient._native_a2a_runtime_authority
    try:
        assert publication.is_published and publication.routing_key == "standalone-key"
        assert p.recipient.a2a_inbound_sender_authorizer is original
        assert owner.router is p.recipient.peer_directory_router
    finally:
        await owner.close()


async def test_operator_only_server_does_not_require_an_implicit_peer_identity(
    single_candidate,
):
    from kestrel_sovereign.server import _onboard_single_agent_native

    p = single_candidate
    assert await _onboard_single_agent_native(p.app, object(), {}) is None
    assert p.app.routes == []


async def test_process_onboarding_keeps_the_attested_signing_did_to_principal_mapping(
    single_candidate,
):
    from kestrel_sovereign.server import _onboard_single_agent_native

    p = single_candidate
    resolver = ProcessA2ADidResolver(
        (p.admission.signing[1] | {A2A_PEER_STABLE_AGENT_ID_FIELD: PRINCIPAL},)
    )
    p.recipient.a2a_did_resolver = resolver.resolve
    p.recipient._a2a_process_did_resolver = resolver
    p.recipient.a2a_inbound_sender_authorizer = None
    await _onboard_single_agent_native(
        p.app,
        p.recipient,
        {
            "KESTREL_BASE_URL": "https://single.native.test",
            "KESTREL_A2A_PUSH_ENABLED": "false",
        },
    )
    try:
        principal = await p.recipient.a2a_inbound_sender_authorizer.authorize_principal_with_policy(
            SENDER,
            router=p.recipient.peer_directory_router,
            requester=p.recipient.peer_requester,
        )
        assert principal == PRINCIPAL
    finally:
        await p.recipient._native_a2a_runtime_authority.close()


async def test_single_host_preserves_its_configured_parent_roster_and_native_factory(
    single_candidate,
):
    from kestrel_sovereign.server import _hosted_peer_directory_context

    p = single_candidate
    router = LocalHostPeerDirectory("http://parent.native.test:8888")
    feature = PeersFeature(p.recipient)
    feature._peer_router = router
    feature._peer_requester = p.recipient.peer_requester
    feature._host_url = "http://parent.native.test:8888"
    p.recipient.features = {"PeersFeature": feature}
    p.app.state.multi_agent_config = SimpleNamespace(
        host=SimpleNamespace(port=9999, bind="127.0.0.1")
    )
    selected, requester = _hosted_peer_directory_context(p.app, p.recipient)
    assert selected is router and requester is p.recipient.peer_requester


async def test_process_onboarding_preserves_an_explicitly_owned_native_factory(
    single_candidate,
):
    from kestrel_sovereign.server import _onboard_single_agent_native

    p = single_candidate
    factory = lambda requester, peer: None
    router = LocalHostPeerDirectory(
        "https://parent.native.test", native_session_factory=factory
    )
    resolver = ProcessA2ADidResolver(())
    p.recipient.peer_directory_router = router
    p.recipient.a2a_did_resolver = resolver.resolve
    p.recipient._a2a_process_did_resolver = resolver
    await _onboard_single_agent_native(p.app, p.recipient, {
        "KESTREL_BASE_URL": "https://single.native.test",
        "KESTREL_A2A_PUSH_ENABLED": "false",
    })
    try:
        assert router._native_session_factory is factory
        assert p.recipient.peer_directory_router is router
    finally:
        await p.recipient._native_a2a_runtime_authority.close()
