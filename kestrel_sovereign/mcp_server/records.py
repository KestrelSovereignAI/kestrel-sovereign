"""One SDK cipher and record-domain policy for durable MCP authorization data."""

import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from enum import StrEnum

from kestrel_sdk.security.aead import AEADCipher
from kestrel_sdk.security.encryption import get_agent_key


class MCPRecordBudgetExceeded(ValueError):
    """A new authorization record exceeds its explicit encoding budget."""


class MCPRecordKind(StrEnum):
    PENDING = "pending"
    GRANT = "grant"
    CODE = "code"
    ACCESS = "access"
    REFRESH = "refresh"


def new_mcp_secret() -> str:
    """Return an opaque 256-bit handle; persist only its hash."""
    return secrets.token_urlsafe(32)


def hash_mcp_secret(secret: str) -> str:
    if not isinstance(secret, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", secret) is None:
        raise ValueError("Invalid MCP authorization secret")
    return hashlib.sha256(secret.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class MCPAuthorizationRecords:
    """Encrypt bounded JSON without borrowing database or authorization authority."""

    agent_id: str
    max_bytes: int

    def __post_init__(self) -> None:
        if (not isinstance(self.agent_id, str) or not self.agent_id
                or any(ord(char) < 32 or ord(char) == 127 for char in self.agent_id)):
            raise ValueError("MCP authorization records require the original agent identity")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("MCP authorization records require a positive byte budget")

    def _aad(self, kind: MCPRecordKind, record_hash: str, expires_at: int) -> bytes:
        if not isinstance(kind, MCPRecordKind):
            raise TypeError("An explicit MCP authorization record domain is required")
        if not isinstance(record_hash, str) or re.fullmatch(r"[a-f0-9]{64}", record_hash) is None:
            raise ValueError("MCP authorization records require a native secret hash")
        if type(expires_at) is not int or expires_at <= 0:
            raise ValueError("MCP authorization records require a positive Unix expiry")
        return json.dumps(["kestrel-mcp-server-" + kind.value + "-v1", self.agent_id,
            record_hash, expires_at], separators=(",", ":")).encode()

    def seal(self, kind: MCPRecordKind, record_hash: str, expires_at: int, value: dict) -> str:
        if type(value) is not dict:
            raise TypeError("MCP authorization records require a JSON object")
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > self.max_bytes:
            raise MCPRecordBudgetExceeded("MCP authorization record exceeds its storage budget")
        return AEADCipher(get_agent_key(self.agent_id, "service-keys")).encrypt(
            raw, aad=self._aad(kind, record_hash, expires_at)).decode("ascii")

    def open(self, kind: MCPRecordKind, record_hash: str, expires_at: int, ciphertext: str) -> dict:
        raw = AEADCipher(get_agent_key(self.agent_id, "service-keys")).decrypt(
            ciphertext.encode("ascii"), aad=self._aad(kind, record_hash, expires_at))
        if len(raw) > self.max_bytes:
            raise ValueError("MCP authorization record exceeds its current storage budget")
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("Stored MCP authorization record is not a JSON object")
        return value
