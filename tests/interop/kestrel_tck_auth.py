"""TCK-only sender extension client using the TCK's official protobuf schema.

Loaded in the independent TCK environment as an HTTPX auth provider. It never
imports Core or changes protocol payloads, responses or compatibility assertions.
The parent test supplies temporary keys, a trusted recipient and an owned signer.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx
from google.api import annotations_pb2
from google.protobuf.json_format import MessageToDict, ParseDict, ParseError
from google.protobuf.message_factory import GetMessageClass
from specification.generated import a2a_pb2
from tck.requirements import base as requirement_bindings


class KestrelTckAuth(httpx.Auth):
    """Authenticate typed service requests without rewriting negative probes."""

    requires_request_body = True

    def __init__(self, *, transport, agent_card):
        self._config_path = Path(os.environ["KESTREL_TCK_PEER_CONFIG"])
        self._binary = Path(os.environ["KESTREL_TCK_GO_PEER"])
        self._config = json.loads(self._config_path.read_text())
        self._transport = transport
        binding = {"jsonrpc": "JSONRPC", "http_json": "HTTP+JSON"}[transport]
        endpoint = next(item["url"] for item in agent_card["supportedInterfaces"] if item["protocolBinding"] == binding)
        self._endpoint = urlsplit(endpoint)
        self._endpoint_path = httpx.URL(endpoint).raw_path.split(b"?", 1)[0].decode("ascii").rstrip("/")
        trusted = urlsplit(self._config["coreCardUrl"])
        if (self._endpoint.scheme, self._endpoint.netloc) != (trusted.scheme, trusted.netloc):
            raise ValueError("TCK card interface is outside the parent-owned recipient origin")
        self._routes = []
        # The TCK's binding table follows the specification's documented POST
        # subscription, while the proto annotation declares GET. The native
        # SDK serves both. Authenticate the original probe in either form;
        # this changes no method, URL, payload or response.
        def normalized_template(template):
            return re.sub(r"\{(\w+)(?:=[^}]+)?\}", r"{\1}", template)

        documented_methods = {
            (binding.jsonrpc_method, normalized_template(binding.http_json_path)): binding.http_json_method
            for binding in vars(requirement_bindings).values()
            if isinstance(binding, requirement_bindings.TransportBinding)
        }
        service = a2a_pb2.DESCRIPTOR.services_by_name["A2AService"]
        for method in service.methods:
            rule = method.GetOptions().Extensions[annotations_pb2.http]
            for binding_rule in [rule, *rule.additional_bindings]:
                verb = binding_rule.WhichOneof("pattern")
                if verb is None:
                    continue
                template = getattr(binding_rule, verb)
                pattern, offset = "", 0
                for field in re.finditer(r"\{(\w+)(?:=[^}]+)?\}", template):
                    pattern += re.escape(template[offset:field.start()])
                    pattern += f"(?P<{field.group(1)}>[^/]+)"
                    offset = field.end()
                pattern += re.escape(template[offset:])
                verbs = {verb.upper()}
                documented = documented_methods.get((method.name, normalized_template(template)))
                if documented is not None:
                    verbs.add(documented)
                for http_method in sorted(verbs):
                    self._routes.append((http_method, re.compile(pattern), method))

    def _service_input(self, request):
        try:
            body = json.loads(request.content) if request.content else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            # An invalid envelope has no service request to authenticate. Emit
            # the original probe so the SUT's protocol parser must reject it.
            return None
        if not isinstance(body, dict):
            return None
        if self._transport == "jsonrpc":
            operation, params = body.get("method"), body.get("params", {})
        else:
            # Match encoded segments, then decode each captured native field
            # once. HTTPX's .path has already decoded escapes and can turn an
            # opaque %2F into a separator or decode a literal %252F twice.
            prefix = self._endpoint_path
            request_path = request.url.raw_path.split(b"?", 1)[0].decode("ascii")
            if not request_path.startswith(prefix + "/") and request_path != prefix:
                raise ValueError("TCK request left its declared native interface")
            path = request_path[len(prefix):] or "/"
            operation, params = None, None
            for verb, pattern, descriptor in self._routes:
                match = pattern.fullmatch(path)
                if request.method == verb and match:
                    operation = descriptor.name
                    fields = descriptor.input_type.fields_by_name
                    path_fields = {fields[key].json_name: unquote(value) for key, value in match.groupdict().items()}
                    params = {**body, **path_fields, **dict(request.url.params)}
                    for field in fields.values():
                        value = params.get(field.json_name)
                        if field.type == field.TYPE_BOOL and isinstance(value, str) and value in {"true", "false"}:
                            params[field.json_name] = value == "true"
                    break
        service = a2a_pb2.DESCRIPTOR.services_by_name["A2AService"]
        descriptor = service.methods_by_name.get(operation)
        if descriptor is None or not isinstance(params, dict):
            return None
        proto = GetMessageClass(descriptor.input_type)()
        try:
            ParseDict(params, proto, ignore_unknown_fields=True)
        except ParseError:
            # Preserve deliberate invalid native fields for server validation;
            # they cannot establish the sender of a valid service operation.
            return None
        return operation, MessageToDict(proto)

    def auth_flow(self, request):
        origin = urlsplit(str(request.url))
        if (origin.scheme, origin.netloc) != (self._endpoint.scheme, self._endpoint.netloc):
            raise ValueError("TCK refused to sign outside its parent-owned origin")
        native = self._service_input(request)
        if native is not None:
            operation, params = native
            extensions = sorted(set(self._config["extensions"]) | {
                item.strip() for item in request.headers.get("A2A-Extensions", "").split(",") if item.strip()
            })
            result = subprocess.run(
                [str(self._binary), "proof", str(self._config_path)],
                input=json.dumps({"operation": operation, "params": params, "extensions": extensions}),
                capture_output=True, text=True, check=True, timeout=10,
            )
            request.headers["Kestrel-A2A-Sender-Proof"] = json.loads(result.stdout)["proof"]
            request.headers["A2A-Extensions"] = ",".join(extensions)
        yield request


def create_auth(*, transport, agent_card):
    """The implementor factory required by the TCK's explicit HTTP auth API."""
    return KestrelTckAuth(transport=transport, agent_card=agent_card)
