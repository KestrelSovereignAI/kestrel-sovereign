"""Exercise Core's production fetch callers over real verified TLS and HTTP."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.unit.test_identity_constitution_endpoint_contracts import (
    _prepare_app,
    _restore_app,
)
from tests.utils.outbound_tls import outbound_tls as outbound_tls  # noqa: PLC0414 - pytest fixture re-export
from tests.utils.tls import make_tls_contexts


def test_did_fetch_preserves_idna2008_host_sni_and_proxy_isolation(outbound_tls):
    from kestrel_sovereign.identity.did_web import _default_fetcher

    peer = outbound_tls
    assert (
        json.loads(_default_fetcher(peer.url + "/did.json"))["id"]
        == "did:web:straße.example"
    )
    assert peer.dns == [peer.host]
    assert peer.dials == [("8.8.8.8", peer.port)]
    assert peer.sni == [peer.host]
    assert peer.requests == [("/did.json", f"{peer.host}:{peer.port}")]


@pytest.mark.parametrize("outbound_tls", ["wrong.example"], indirect=True)
def test_did_fetch_enforces_the_real_peer_certificate(outbound_tls):
    from kestrel_sovereign.identity.did_web import DidWebError, _default_fetcher

    peer = outbound_tls
    with pytest.raises(DidWebError, match="failed"):
        _default_fetcher(peer.url + "/did.json")
    assert peer.dials == [("8.8.8.8", peer.port)]
    assert peer.sni == [peer.host]
    assert peer.requests == []


@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize(
    "outbound_tls,path,status,served",
    [
        pytest.param(
            "xn--strae-oqa.example", "/avatar.png", 200, True, id="verified-peer"
        ),
        pytest.param(
            "xn--strae-oqa.example", "/redirect", 400, True, id="redirect-refused"
        ),
        pytest.param(
            "wrong.example", "/avatar.png", 400, False, id="wrong-certificate"
        ),
    ],
    indirect=["outbound_tls"],
)
def test_avatar_uses_real_tls_and_refuses_http_redirects(
    outbound_tls, path, status, served
):
    peer = outbound_tls
    store = AsyncMock(return_value="avatar-hash")
    agent = SimpleNamespace(
        agent_id="did:test:avatar",
        storage=SimpleNamespace(files=SimpleNamespace(store_avatar=store)),
    )
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/identity/avatar",
                    json={"url": peer.url + path},
                    headers={"X-API-Key": "test-key"},
                )
        assert response.status_code == status
        assert peer.dns == [peer.host]
        assert peer.dials == [("8.8.8.8", peer.port)]
        assert peer.sni == [peer.host]
        assert peer.requests == ([(path, f"{peer.host}:{peer.port}")] if served else [])
        if status == 400:
            assert (
                "302" if served else "CERTIFICATE_VERIFY_FAILED"
            ) in response.json()["detail"]
            store.assert_not_awaited()
        else:
            store.assert_awaited_once()
            assert store.await_args.kwargs["source_url"] == peer.url + path
    finally:
        _restore_app(app, original)


def test_extracted_tls_fixture_still_supports_the_load_test_memory_bio(tmp_path):
    from tests.load.test_lighthouse_tls_write_ceiling import _memory_bio_pair

    server, client, _ = make_tls_contexts(tmp_path)
    assert _memory_bio_pair(server, client).write(b"small payload") == len(
        b"small payload"
    )
