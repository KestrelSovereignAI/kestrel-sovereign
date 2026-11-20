"""Private immutable Task submission evidence; none of these DTOs grant authority."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Literal

from kestrel_sovereign.a2a.media import encode_inline_json


def _binding_text(value: str) -> None:
    if (not isinstance(value, str) or not value or len(value.encode('utf-8')) > 1024
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError('MCP Task bindings require bounded control-free strings')


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def task_connection_id(agent_did: str, owner: str, connection_name: str) -> str:
    return hashlib.sha256(encode_inline_json(
        ['mcp-task-connection-v1', agent_did, owner, connection_name])).hexdigest()


def _conversation(value: str) -> None:
    from kestrel_sovereign.features.security.hooks import reserved_session_id

    _binding_text(value)
    if not value.strip() or reserved_session_id(value) is not None:
        raise ValueError('MCP Task delivery requires an original conversation')


@dataclass(frozen=True, slots=True, repr=False)
class MCPTaskInteractiveOrigin:
    principal_digest: str
    session_id: str
    kind: Literal['interactive'] = field(default='interactive', init=False)

    def __post_init__(self):
        if not _is_digest(self.principal_digest):
            raise ValueError('MCP Task interactive attribution requires a principal digest')
        _conversation(self.session_id)


@dataclass(frozen=True, slots=True, repr=False)
class MCPTaskServiceOrigin:
    service_kind: str
    namespace_digest: str
    session_id: str | None = None
    kind: Literal['service'] = field(default='service', init=False)

    def __post_init__(self):
        from kestrel_sovereign.tools.effect_invocation import SERVICE_OPERATION_KINDS

        if self.service_kind not in SERVICE_OPERATION_KINDS or not _is_digest(self.namespace_digest):
            raise ValueError('MCP Task service attribution requires its bounded kind and digest')
        if self.session_id is not None:
            _conversation(self.session_id)


def decode_origin(value):
    if type(value) is not dict:
        raise ValueError('Invalid MCP Task origin')
    if set(value) == {'kind', 'principal_digest', 'session_id'} and value['kind'] == 'interactive':
        return MCPTaskInteractiveOrigin(value['principal_digest'], value['session_id'])
    if (set(value) == {'kind', 'service_kind', 'namespace_digest', 'session_id'}
            and value['kind'] == 'service'):
        return MCPTaskServiceOrigin(value['service_kind'], value['namespace_digest'], value['session_id'])
    raise ValueError('Invalid MCP Task origin')


@dataclass(frozen=True, slots=True, repr=False)
class MCPTaskDelivery:
    kind: Literal['chat', 'app', 'internal']
    session_id: str | None = None
    view_token: str | None = None

    def __post_init__(self):
        if self.kind == 'internal':
            if self.session_id is not None or self.view_token is not None:
                raise ValueError('Internal MCP Task delivery cannot invent a conversation')
        elif self.kind in {'chat', 'app'}:
            _conversation(self.session_id)
            if self.kind == 'app':
                _binding_text(self.view_token)
            elif self.view_token is not None:
                raise ValueError('Chat MCP Task delivery cannot identify an App View')
        else:
            raise ValueError('Invalid MCP Task delivery kind')


@dataclass(frozen=True, slots=True, repr=False)
class MCPTaskOAuthLocator:
    """Restored custody can select this exact grant slot without granting access."""

    owner: str
    connection_name: str
    principal_digest: str
    slot_id: str
    kind: Literal['oauth'] = field(default='oauth', init=False)

    def __post_init__(self):
        _binding_text(self.owner)
        _binding_text(self.connection_name)
        if not _is_digest(self.principal_digest) or not _is_digest(self.slot_id):
            raise ValueError('MCP Task OAuth attribution requires principal and slot digests')

    @classmethod
    def from_json(cls, value):
        if (type(value) is not dict
                or set(value) != {'kind', 'owner', 'connection_name', 'principal_digest', 'slot_id'}
                or value['kind'] != 'oauth'):
            raise ValueError('Invalid MCP Task OAuth locator')
        return cls(value['owner'], value['connection_name'], value['principal_digest'], value['slot_id'])


@dataclass(frozen=True, slots=True, repr=False)
class MCPTaskSubmission:
    """Original arguments, descriptor, delivery and grant locator, sealed once."""

    canonical_json: bytes

    def __post_init__(self):
        if type(self.canonical_json) is not bytes:
            raise TypeError('MCP Task submission requires immutable bytes')
        try:
            value = json.loads(self.canonical_json)
            if (type(value) is not dict or set(value) != {'arguments', 'contract', 'delivery', 'authorization'}
                    or type(value['arguments']) is not dict or type(value['contract']) is not str
                    or not value['contract'] or type(value['delivery']) is not dict
                    or set(value['delivery']) != {'kind', 'session_id', 'view_token'}
                    or encode_inline_json(value, sort_keys=True) != self.canonical_json):
                raise ValueError('Invalid submission')
            value['contract'].encode('utf-8')
            MCPTaskDelivery(**value['delivery'])
            MCPTaskOAuthLocator.from_json(value['authorization'])
        except (TypeError, ValueError, RecursionError):
            raise ValueError('Invalid private MCP Task submission') from None

    @classmethod
    def seal(cls, *, arguments: dict, contract: bytes, delivery: MCPTaskDelivery,
             authorization: MCPTaskOAuthLocator):
        if (type(contract) is not bytes or type(delivery) is not MCPTaskDelivery
                or type(authorization) is not MCPTaskOAuthLocator):
            raise TypeError('MCP Task submission requires original bytes, typed delivery and OAuth attribution')
        try:
            return cls(encode_inline_json({'arguments': arguments, 'contract': contract.decode('utf-8'),
                'delivery': asdict(delivery), 'authorization': asdict(authorization)}, sort_keys=True))
        except (TypeError, ValueError, RecursionError):
            raise ValueError('Invalid private MCP Task submission') from None

    @property
    def arguments(self) -> dict:
        return json.loads(self.canonical_json)['arguments']

    @property
    def contract(self) -> bytes:
        return json.loads(self.canonical_json)['contract'].encode('utf-8')

    @property
    def delivery(self) -> MCPTaskDelivery:
        return MCPTaskDelivery(**json.loads(self.canonical_json)['delivery'])

    @property
    def authorization(self) -> MCPTaskOAuthLocator:
        return MCPTaskOAuthLocator.from_json(json.loads(self.canonical_json)['authorization'])
