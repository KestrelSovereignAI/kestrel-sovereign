"""Bounded, encrypted pending OAuth requests; none of these records grants access."""

import hashlib
import json
import math
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from kestrel_sdk.security.aead import AEADCipher
from kestrel_sdk.security.encryption import get_agent_key
from kestrel_sdk.storage.database import DatabaseBackend
from mcp.server.auth.provider import AuthorizationParams

from .policy import MCPServerPolicy


class PendingMCPQuotaExceeded(ValueError):
    """The target agent already has its configured pending-request quota."""


@dataclass(frozen=True)
class PendingMCPAuthorization:
    """An original OAuth request, never a token, authorization code or consent."""

    client_id: str
    expires_at: int
    parameters: AuthorizationParams = field(repr=False)


class PendingMCPAuthorizations:
    """Borrow the original agent backend; its owner retains physical retirement.

    A native row lock serializes capacity accounting across processes. Handles
    are stored only as hashes; request contents use the shared SDK cipher with
    agent, record identity and expiry bound as authenticated associated data.
    """

    def __init__(self, backend: DatabaseBackend, agent_id: str, resource_url: str,
                 policy: MCPServerPolicy, *, clock: Callable[[], float] = time.time):
        if (not isinstance(agent_id, str) or not agent_id
                or any(ord(char) < 32 or ord(char) == 127 for char in agent_id)):
            raise ValueError("Pending MCP authorization requires the original agent identity")
        if (not isinstance(resource_url, str) or "#" in resource_url
                or not resource_url.startswith(policy.console_origin + "/")):
            raise ValueError("MCP resource must belong to the configured host origin")
        self.backend, self.agent_id = backend, agent_id
        self.resource_url, self.policy, self.clock = resource_url, policy, clock

    def _now(self) -> int:
        value = self.clock()
        if not math.isfinite(value) or value < 0:
            raise ValueError("MCP authorization clock must supply a finite Unix time")
        return math.floor(value)

    def _handle_hash(self, handle: str) -> str:
        if not isinstance(handle, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", handle) is None:
            raise ValueError("Invalid pending MCP authorization handle")
        return hashlib.sha256(handle.encode("utf-8")).hexdigest()

    def _aad(self, handle_hash: str, expires_at: int) -> bytes:
        return json.dumps(["kestrel-mcp-server-pending-v1", self.agent_id,
            handle_hash, expires_at], separators=(",", ":")).encode()

    def _cipher(self) -> AEADCipher:
        return AEADCipher(get_agent_key(self.agent_id, "service-keys"))

    async def initialize(self) -> None:
        async with self.backend.transaction():
            await self.backend.execute("""CREATE TABLE IF NOT EXISTS mcp_server_oauth_capacity (
                agent_id TEXT PRIMARY KEY, revision BIGINT NOT NULL
            )""")
            await self.backend.execute("""CREATE TABLE IF NOT EXISTS mcp_server_oauth_pending (
                agent_id TEXT NOT NULL, handle_hash TEXT NOT NULL,
                ciphertext TEXT NOT NULL, expires_at BIGINT NOT NULL,
                consumed INTEGER NOT NULL DEFAULT 0 CHECK (consumed IN (0, 1)),
                PRIMARY KEY (agent_id, handle_hash)
            )""")
            await self.backend.execute("""INSERT INTO mcp_server_oauth_capacity (agent_id, revision)
                VALUES (?, 0) ON CONFLICT (agent_id) DO NOTHING""", (self.agent_id,))

    async def create(self, client_id: str, parameters: AuthorizationParams) -> str:
        client = next((row for row in self.policy.clients if row.client_id == client_id), None)
        if client is None or not isinstance(parameters, AuthorizationParams):
            raise ValueError("Pending MCP authorization requires a registered client and native parameters")
        if parameters.resource != self.resource_url:
            raise ValueError("Pending MCP authorization has a different resource audience")
        if not parameters.scopes or not set(parameters.scopes) <= set(client.scopes):
            raise ValueError("Pending MCP authorization requires explicit registered scopes")
        handle = secrets.token_urlsafe(32)
        handle_hash = self._handle_hash(handle)
        now = self._now()
        expires = now + math.ceil(self.policy.authorization_seconds)
        raw = json.dumps({"client_id": client_id, "parameters": parameters.model_dump(mode="json")},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > self.policy.max_request_bytes:
            raise ValueError("Pending MCP authorization exceeds its storage budget")
        ciphertext = self._cipher().encrypt(raw, aad=self._aad(handle_hash, expires)).decode("ascii")
        async with self.backend.transaction():
            # UPDATE takes the original native row lock until this transaction
            # exits; a COUNT followed by INSERT alone races on PostgreSQL.
            capacity = await self.backend.fetch_one("UPDATE mcp_server_oauth_capacity SET revision = revision + 1 WHERE agent_id = ? RETURNING revision",
                (self.agent_id,))
            if capacity is None:
                raise RuntimeError("The original MCP authorization capacity record is unavailable")
            await self.backend.execute("DELETE FROM mcp_server_oauth_pending WHERE agent_id = ? AND (expires_at <= ? OR consumed = 1)",
                (self.agent_id, now))
            row = await self.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_pending WHERE agent_id = ?",
                (self.agent_id,))
            quota_full = row[0] >= self.policy.max_pending_per_agent
            if not quota_full:
                await self.backend.execute("""INSERT INTO mcp_server_oauth_pending
                    (agent_id, handle_hash, ciphertext, expires_at) VALUES (?, ?, ?, ?)""",
                    (self.agent_id, handle_hash, ciphertext, expires))
        if quota_full:
            raise PendingMCPQuotaExceeded("The agent's pending MCP authorization quota is full")
        return handle

    def _decode(self, handle_hash: str, row) -> PendingMCPAuthorization:
        ciphertext, expires = row
        raw = self._cipher().decrypt(ciphertext.encode("ascii"), aad=self._aad(handle_hash, expires))
        value = json.loads(raw)
        return PendingMCPAuthorization(value["client_id"], expires,
            AuthorizationParams.model_validate(value["parameters"]))

    async def read(self, handle: str) -> PendingMCPAuthorization | None:
        handle_hash = self._handle_hash(handle)
        row = await self.backend.fetch_one("""SELECT ciphertext, expires_at FROM mcp_server_oauth_pending
            WHERE agent_id = ? AND handle_hash = ? AND consumed = 0 AND expires_at > ?""",
            (self.agent_id, handle_hash, self._now()))
        return None if row is None else self._decode(handle_hash, row)

    async def claim(self, handle: str) -> PendingMCPAuthorization | None:
        """Consume once inside the consent owner's transaction; this mints no grant."""
        handle_hash = self._handle_hash(handle)
        async with self.backend.transaction():
            row = await self.backend.fetch_one("""UPDATE mcp_server_oauth_pending SET consumed = 1
                WHERE agent_id = ? AND handle_hash = ? AND consumed = 0 AND expires_at > ?
                RETURNING ciphertext, expires_at""", (self.agent_id, handle_hash, self._now()))
            pending = None if row is None else self._decode(handle_hash, row)
        return pending
