"""Active-call HTTP gateway; the browser itself retains Docker network=none.

Only trusted host configuration selects origins/methods. The exact original
host operation supplies the current authority check and owns cancellation.
There is no ambient proxy, cookie jar, credential profile or connection pool.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import ipaddress
import json
import re
import socket
import ssl
from urllib.parse import urlsplit

import h11
from kestrel_sdk.resources import ResourceUnsupported

MAX_BODY = 128 * 1024
MAX_HEADER = 16384
MAX_JOBS = 16
OWNED_SERVER_LABEL = "io.kestrel.visual-computer.egress-proof"


class EgressDenied(Exception):
    """Known refusal before an HTTP request effect, with a bounded public code."""


class EgressUnknown(Exception):
    """The admitted server may have accepted a request; never retry it blindly."""


def origin(value):
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or not parsed.hostname.isascii() or "\\" in value
                or any(character.isspace() or ord(character) < 32 for character in value)):
            raise ValueError
        host = parsed.hostname.lower().rstrip(".")
        if not re.fullmatch(r"[a-z0-9.:-]+", host):
            raise ValueError
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535:
            raise ValueError
        authority = f"[{host}]" if ":" in host else host
        return f"{parsed.scheme}://{authority}:{port}", parsed.scheme, host, port
    except (TypeError, ValueError, AttributeError):
        raise EgressDenied("invalid_http_destination") from None


def public_address(value):
    if "%" in value:
        raise EgressDenied("scoped_address_refused")
    address = ipaddress.ip_address(value)
    if (not address.is_global or address.is_multicast or address.is_reserved
            or (isinstance(address, ipaddress.IPv6Address)
                and (address.ipv4_mapped is not None or address.sixtofour is not None or address.teredo is not None))):
        raise EgressDenied("private_or_special_destination_refused")
    return address


@dataclass(frozen=True)
class OwnedTestServer:
    """Test-only exception proven against an exact owned Docker port binding.

This object cannot be constructed through the normal mapping parser. It grants
one synthetic server endpoint, not a loopback/LAN/private-network category.
"""
    boundary: object
    container_id: str
    ownership: str
    port: int
    tls: bool = False

    @property
    def origin(self):
        # The TLS fixture name is derived from the exact ownership label, never
        # an arbitrary hostname supplied through production settings.
        if type(self.tls) is not bool or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,61}[a-z0-9]",self.ownership):
            raise EgressDenied("invalid_owned_test_server")
        return (f"https://{self.ownership}.invalid" if self.tls else "http://127.0.0.1") + f":{self.port}"

    async def verify(self):
        record = await self.boundary.inspect(self.container_id)
        ports = record.get("NetworkSettings", {}).get("Ports", {}).get("8080/tcp")
        if (record.get("Id") != self.container_id or not record.get("State", {}).get("Running")
                or record.get("Config", {}).get("Labels", {}).get(OWNED_SERVER_LABEL) != self.ownership
                or ports != [{"HostIp": "127.0.0.1", "HostPort": str(self.port)}]):
            raise EgressDenied("owned_test_server_unavailable")


@dataclass(frozen=True)
class DockerEgressPolicy:
    origins: tuple[str, ...]
    methods: tuple[str, ...] = ("GET", "HEAD")
    test_server: OwnedTestServer | None = None
    address_pins: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if not 1 <= len(self.origins) <= 8 or not self.methods or not set(self.methods) <= {"GET", "HEAD"}:
            raise ResourceUnsupported("unsupported_http_policy")
        pins = {}
        for name, value in self.address_pins:
            canonical, _, _, _ = origin(name)
            parsed = urlsplit(name)
            if parsed.path not in {"","/"} or parsed.query or parsed.fragment:
                raise ResourceUnsupported("http_pin_requires_exact_origin")
            if canonical in pins:
                raise ResourceUnsupported("duplicate_http_origin_pin")
            pins[canonical] = value
        normalized = []
        for value in self.origins:
            canonical, _, host, _ = origin(value)
            parsed = urlsplit(value)
            if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
                raise ResourceUnsupported("http_policy_requires_exact_origins")
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                if host in {"localhost", "metadata.google.internal", "metadata"} or host.endswith((".localhost", ".local", ".internal")):
                    raise ResourceUnsupported("private_http_origin_refused")
                if canonical not in pins:
                    raise ResourceUnsupported("explicit_public_ipv4_origin_pin_required")
            else:
                if canonical in pins and pins[canonical] != str(address):
                    raise ResourceUnsupported("literal_http_origin_pin_conflict")
                pins[canonical] = str(address)
            if self.test_server is None:
                try:
                    pin = public_address(pins[canonical])
                    if pin.version != 4:
                        raise EgressDenied("unmeasured_ipv6_egress_refused")
                except (EgressDenied,ValueError,TypeError):
                    raise ResourceUnsupported("public_ipv4_origin_pin_required") from None
            else:
                if canonical != self.test_server.origin or pins[canonical] != "127.0.0.1":
                    raise ResourceUnsupported("test_policy_requires_exact_owned_server")
            normalized.append(canonical)
        if set(pins) != set(normalized):
            raise ResourceUnsupported("http_origin_pin_set_mismatch")
        object.__setattr__(self, "origins", tuple(sorted(set(normalized))))
        object.__setattr__(self, "methods", tuple(sorted(set(self.methods))))
        object.__setattr__(self, "address_pins", tuple(sorted(pins.items())))

    @property
    def profile_id(self):
        payload = {"origins": self.origins, "methods": self.methods, "address_pins": self.address_pins,
                   "owned_test_container": self.test_server.container_id if self.test_server else None}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return "http-get-pinned-v1-" + digest

    @classmethod
    def from_settings(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if (not isinstance(value, dict) or set(value) not in ({"origins", "methods"},{"origins","methods","address_pins"})
                or not isinstance(value["origins"], (list, tuple)) or not isinstance(value["methods"], (list, tuple))
                or not all(isinstance(item, str) for item in (*value["origins"], *value["methods"]))
                or not isinstance(value.get("address_pins",{}),dict)
                or not all(isinstance(name,str) and isinstance(pin,str) for name,pin in value.get("address_pins",{}).items())):
            raise ResourceUnsupported("invalid_private_http_policy")
        try:
            return cls(tuple(value["origins"]), tuple(value["methods"]), address_pins=tuple(value.get("address_pins",{}).items()))
        except EgressDenied:
            raise ResourceUnsupported("invalid_private_http_policy") from None

    async def addresses(self, url, method, current):
        canonical, scheme, host, port = origin(url)
        if canonical not in self.origins or method not in self.methods:
            raise EgressDenied("http_destination_or_method_refused")
        current()
        if self.test_server is not None:
            await self.test_server.verify()
            current()
        # Static operator pins are immutable admission facts. Runtime libc DNS
        # cannot be physically cancelled/joined, so this profile never calls it.
        address = dict(self.address_pins)[canonical]
        current()
        return scheme, host, port, [(socket.AF_INET, address)]


class HTTPGateway:
    def __init__(self, policy):
        self.policy = policy

    async def fetch(self, job, current):
        if (set(job) != {"job_id", "action_id", "session_id", "control_epoch", "url", "method"}
                or not isinstance(job["url"], str) or len(job["url"]) > 4096):
            raise EgressDenied("invalid_http_job")
        writer, sent = None, False
        try:
            async with asyncio.timeout(8):
                scheme, host, port, addresses = await self.policy.addresses(job["url"], job["method"], current)
                # No retries/fallback after a failed connect or request. Even a
                # timeout can have reached an admitted consequential endpoint.
                family, address = addresses[0]
                current()
                reader, writer = await asyncio.open_connection(address, port, family=family,
                    ssl=ssl.create_default_context() if scheme == "https" else None,
                    server_hostname=host if scheme == "https" else None, limit=MAX_HEADER + 1)
                current()
                parsed = urlsplit(job["url"])
                target = parsed.path or "/"
                if parsed.query:
                    target += "?" + parsed.query
                authority = f"[{host}]" if ":" in host else host
                if port != (443 if scheme == "https" else 80):
                    authority += ":" + str(port)
                connection = h11.Connection(h11.CLIENT, max_incomplete_event_size=MAX_HEADER)
                message = connection.send(h11.Request(method=job["method"], target=target.encode("ascii"),
                    headers=[(b"Host", authority.encode("ascii")), (b"Accept", b"*/*"),
                             (b"Accept-Encoding", b"identity"), (b"Connection", b"close")]))
                message += connection.send(h11.EndOfMessage())
                current()
                sent = True
                writer.write(message)
                await writer.drain()
                current()
                status, headers, chunks, size = None, {}, [], 0
                while True:
                    event = connection.next_event()
                    if event is h11.NEED_DATA:
                        data = await reader.read(65536)
                        current()
                        connection.receive_data(data)
                        continue
                    if isinstance(event, h11.InformationalResponse):
                        raise EgressUnknown("http_upgrade_or_informational_response_refused")
                    if isinstance(event, h11.Response):
                        status = event.status_code
                        if len(event.reason) + 32 + sum(len(name)+len(value)+4 for name,value in event.headers) > MAX_HEADER:
                            raise EgressUnknown("http_response_header_limit")
                        # Playwright routes only the first URL of a redirect
                        # chain. Forwarding 3xx would leave its next hop outside
                        # this relay. network=none still denies that socket,
                        # but honest negotiated support refuses the response.
                        if 300 <= status < 400:
                            raise EgressUnknown("http_redirect_response_unsupported")
                        for raw_name, raw_value in event.headers:
                            name, value = raw_name.decode("ascii").lower(), raw_value.decode("latin-1")
                            if name in headers:
                                raise EgressUnknown("duplicate_http_response_header_refused")
                            headers[name] = value
                        if (headers.get("content-encoding", "identity") != "identity"
                                or "content-disposition" in headers or "set-cookie" in headers):
                            raise EgressUnknown("unsupported_http_response_semantics")
                        continue
                    if isinstance(event, h11.Data):
                        current()
                        size += len(event.data)
                        if size > MAX_BODY:
                            raise EgressUnknown("http_response_body_limit")
                        chunks.append(bytes(event.data))
                        continue
                    if isinstance(event, h11.EndOfMessage):
                        if status is None:
                            raise EgressUnknown("http_response_missing_status")
                        current()
                        allowed = {name: value for name, value in headers.items()
                                   if name in {"content-type", "cache-control", "content-security-policy", "referrer-policy"}}
                        return {"status": status, "headers": allowed, "body": b"".join(chunks)}
                    raise EgressUnknown("http_response_protocol_refused")
        except EgressDenied:
            raise
        except (OSError, TimeoutError, ValueError, UnicodeError, h11.ProtocolError):
            if sent:
                raise EgressUnknown("http_request_outcome_unknown") from None
            raise EgressDenied("http_connection_unavailable") from None
        finally:
            if writer is not None:
                writer.close()
                # Socket close is synchronous; bounded TLS shutdown does not
                # allow another application request or escape task custody.
                try:
                    async with asyncio.timeout(1):
                        await writer.wait_closed()
                except (OSError, TimeoutError):
                    pass
