"""Real loopback HTTP plus gateway guards; Docker/browser proof is separate."""
import asyncio
from contextlib import asynccontextmanager
import json
import socket

import pytest
from kestrel_sdk.resources import ResourceUnsupported
from kestrel_sovereign.visual_computer.docker_egress import (
    DockerEgressPolicy, HTTPGateway, EgressDenied, EgressUnknown, OwnedTestServer,
    OWNED_SERVER_LABEL, public_address, MAX_BODY, MAX_HEADER,
)


@pytest.mark.parametrize("value", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "172.17.0.1", "192.168.1.1",
    "0.0.0.0", "255.255.255.255", "224.0.0.1", "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1", "2002:7f00:1::"])
def test_no_private_special_multicast_or_transition_destination(value):
    with pytest.raises(EgressDenied):
        public_address(value)


@pytest.mark.parametrize("configuration", [
    {"origins": ["http://127.0.0.1"], "methods": ["GET"]},
    {"origins": ["http://localhost"], "methods": ["GET"]},
    {"origins": ["http://metadata.google.internal"], "methods": ["GET"]},
    {"origins": ["https://example.com/path"], "methods": ["GET"]},
    {"origins": ["https://example.com"], "methods": ["POST"]},
    {"origins": ["https://example.com"], "methods": ["GET"], "test_server": {}},
])
def test_normal_settings_cannot_admit_test_exception_or_implicit_authority(configuration):
    with pytest.raises(ResourceUnsupported):
        DockerEgressPolicy.from_settings(configuration)


@asynccontextmanager
async def controlled_server(response):
    requests, connections = [], []
    async def handle(reader, writer):
        connections.append(writer)
        try:
            data = await reader.readuntil(b"\r\n\r\n")
            requests.append(data)
            writer.write(response)
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    class Boundary:
        async def inspect(self, cid):
            return {"Id": cid, "State": {"Running": True}, "Config": {"Labels": {OWNED_SERVER_LABEL: "fixture"}},
                    "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]}}}
    # This synthetic ownership record tests HTTP behavior; it does not establish
    # Docker isolation. Real conformance checks actual daemon-owned identities.
    owned = OwnedTestServer(Boundary(), "c" * 64, "fixture", port)
    policy = DockerEgressPolicy((f"http://127.0.0.1:{port}",), ("GET", "HEAD"), owned)
    async with server:
        yield policy, requests, connections


def job(url, **changes):
    return {"job_id": "job", "action_id": "action", "session_id": "session", "control_epoch": 1,
            "url": url, "method": "GET", **changes}


@pytest.mark.asyncio
async def test_raw_gateway_uses_exact_server_and_never_host_proxy_credentials(monkeypatch,caplog):
    async with controlled_server(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nContent-Type: text/plain\r\n\r\nBODY") as (policy, requests, _):
        monkeypatch.setenv("HTTP_PROXY", "http://credential:PRIVATE@127.0.0.1:1")
        caplog.set_level("DEBUG")
        checks = []
        result = await HTTPGateway(policy).fetch(job(policy.origins[0]+"/private?token=SYNTHETIC"), lambda: checks.append(True))
        assert result["body"] == b"BODY" and len(checks) >= 5
        assert len(requests) == 1 and requests[0].startswith(b"GET /private?token=SYNTHETIC HTTP/1.1")
        assert b"Authorization" not in requests[0] and b"Cookie" not in requests[0] and b"PRIVATE" not in requests[0]
        assert "SYNTHETIC" not in caplog.text and "/private" not in caplog.text


@pytest.mark.asyncio
async def test_redirect_is_explicitly_refused_without_following_new_destination():
    async with controlled_server(b"HTTP/1.1 302 Found\r\nContent-Length: 0\r\nLocation: http://169.254.169.254/\r\n\r\n") as (policy, requests, _):
        with pytest.raises(EgressUnknown, match="redirect"):
            await HTTPGateway(policy).fetch(job(policy.origins[0]+"/redirect"), lambda: None)
        assert len(requests) == 1
        with pytest.raises(EgressDenied):
            await HTTPGateway(policy).fetch(job("http://169.254.169.254/"), lambda: None)
        assert len(requests) == 1


@pytest.mark.asyncio
async def test_pinned_hostname_never_schedules_delayed_libc_resolver(monkeypatch):
    policy = DockerEgressPolicy.from_settings({"origins":["https://example.com"],"methods":["GET"],
        "address_pins":{"https://example.com":"93.184.216.34"}})
    loop = asyncio.get_running_loop()
    calls = []
    async def delayed(*args,**kwargs):
        calls.append(args)
        await asyncio.Event().wait()
    monkeypatch.setattr(loop,"getaddrinfo",delayed)
    async with asyncio.timeout(.2):
        result = await policy.addresses("https://example.com/","GET",lambda:None)
    assert result == ("https","example.com",443,[(socket.AF_INET,"93.184.216.34")]) and calls == []


@pytest.mark.parametrize("pins",[{}, {"https://example.com":"127.0.0.1"},
    {"https://example.com":"169.254.169.254"},{"https://example.com":"2606:4700:4700::1111"},
    {"https://other.example.com":"93.184.216.34"}])
def test_production_origin_requires_exact_public_ipv4_pin(pins):
    with pytest.raises(ResourceUnsupported):
        DockerEgressPolicy.from_settings({"origins":["https://example.com"],"methods":["GET"],"address_pins":pins})


def test_profile_digest_binds_original_hostname_and_exact_pin():
    def policy(pin):
        return DockerEgressPolicy.from_settings({"origins":["https://example.com"],"methods":["GET"],
            "address_pins":{"https://example.com":pin}})
    assert policy("93.184.216.34").profile_id != policy("1.1.1.1").profile_id


@pytest.mark.asyncio
async def test_normal_pinned_transport_uses_literal_target_and_original_host_sni_with_no_dns(monkeypatch):
    async with controlled_server(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nBODY") as (local,requests,_):
        target = "https://example.com:"+str(local.test_server.port)
        policy = DockerEgressPolicy.from_settings({"origins":[target],"methods":["GET"],
            "address_pins":{target:"93.184.216.34"}})
        original = asyncio.open_connection
        connected = []
        async def connect(address,port,**kwargs):
            connected.append((address,port,kwargs))
            # API-boundary check only: actual isolation is measured separately.
            return await original("127.0.0.1",port,family=socket.AF_INET)
        async def delayed(*args,**kwargs):
            raise AssertionError("runtime libc DNS scheduled")
        monkeypatch.setattr(asyncio,"open_connection",connect)
        monkeypatch.setattr(asyncio.get_running_loop(),"getaddrinfo",delayed)
        assert (await HTTPGateway(policy).fetch(job(target+"/"),lambda:None))["body"] == b"BODY"
        address,port,options = connected[0]
        assert address == "93.184.216.34" and options["server_hostname"] == "example.com"
        assert options["ssl"].check_hostname and options["ssl"].verify_mode.name == "CERT_REQUIRED"
        assert b"Host: example.com:" in requests[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("headers,body", [(b"Content-Encoding: gzip\r\n", b"compressed"),
    (b"Set-Cookie: session=secret\r\n", b"cookie"), (b"Content-Disposition: attachment\r\n", b"download"),
    (b"", b"x"*(MAX_BODY+1)),(b"X-Padding: "+b"x"*MAX_HEADER+b"\r\n",b"")],
    ids=["encoding", "cookie", "download", "body_limit", "header_limit"])
async def test_response_limits_and_unsupported_semantics_remain_unknown_after_request(headers, body):
    response = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n"+headers+b"\r\n"+body
    async with controlled_server(response) as (policy, requests, _):
        with pytest.raises(EgressUnknown):
            await HTTPGateway(policy).fetch(job(policy.origins[0]+"/"), lambda: None)
        assert len(requests) == 1
