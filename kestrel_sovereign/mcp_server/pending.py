"""Bounded, encrypted pending OAuth requests; none of these records grants access."""

import math
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from kestrel_sdk.storage.database import DatabaseBackend
from mcp.server.auth.provider import AuthorizationParams

from .policy import MCPServerClient, MCPServerPolicy
from .records import (
    MCPAuthorizationRecords,
    MCPRecordKind,
    hash_mcp_secret,
    new_mcp_secret,
)


class PendingMCPQuotaExceeded(ValueError):
    """The target agent already has its configured pending-request quota."""


class PendingMCPAuthorizationExpired(ValueError):
    """Native admission settled after the pending authorization deadline."""


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
        self._records = MCPAuthorizationRecords(agent_id, policy.max_request_bytes)
        if (not isinstance(resource_url, str) or "#" in resource_url
                or not resource_url.startswith(policy.console_origin + "/")):
            raise ValueError("MCP resource must belong to the configured host origin")
        self.backend, self.agent_id = backend, agent_id
        self.resource_url, self.policy, self.clock = resource_url, policy, clock

    def now(self, *, fractional: bool = False) -> int | float:
        """Read validated time; records use seconds, deadlines retain fractions."""
        value = self.clock()
        if not math.isfinite(value) or value < 0:
            raise ValueError("MCP authorization clock must supply a finite Unix time")
        return value if fractional else math.floor(value)

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

    def _currently_registered(self, client_id: str, parameters: AuthorizationParams) -> bool:
        """The consuming host's current policy controls admission and durable reads."""
        client = self._registered_client(client_id)
        return (client is not None and isinstance(parameters, AuthorizationParams)
            and parameters.resource == self.resource_url
            and client.native_redirect_uri(str(parameters.redirect_uri)) is not None
            and bool(parameters.scopes) and set(parameters.scopes) <= set(client.scopes))

    def _registered_client(self, client_id: str) -> MCPServerClient | None:
        return self.policy.registered_client(client_id)

    async def _lock_capacity(self) -> None:
        capacity = await self.backend.fetch_one("UPDATE mcp_server_oauth_capacity SET revision = revision + 1 WHERE agent_id = ? RETURNING revision",
            (self.agent_id,))
        if capacity is None:
            raise RuntimeError("The original MCP authorization capacity record is unavailable")

    @asynccontextmanager
    async def serialized(self):
        """Retain the original transaction and per-agent OAuth state lock."""
        async with self.backend.transaction():
            await self._lock_capacity()
            yield

    async def create(self, client_id: str, parameters: AuthorizationParams) -> str:
        if not self._currently_registered(client_id, parameters):
            raise ValueError("Pending MCP authorization requires a registered client, redirect, scopes and resource")
        handle = new_mcp_secret()
        handle_hash = hash_mcp_secret(handle)
        now = self.now()
        expires = now + math.ceil(self.policy.authorization_seconds)
        ciphertext = self._records.seal(MCPRecordKind.PENDING, handle_hash, expires,
            {"client_id": client_id, "parameters": parameters.model_dump(mode="json")})
        async with self.serialized():
            now = self.now()
            expired = expires <= now
            await self.backend.execute("DELETE FROM mcp_server_oauth_pending WHERE agent_id = ? AND (expires_at <= ? OR consumed = 1)",
                (self.agent_id, now))
            row = await self.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_pending WHERE agent_id = ?",
                (self.agent_id,))
            quota_full = row[0] >= self.policy.max_pending_per_agent
            if not quota_full and not expired:
                await self.backend.execute("""INSERT INTO mcp_server_oauth_pending
                    (agent_id, handle_hash, ciphertext, expires_at) VALUES (?, ?, ?, ?)""",
                    (self.agent_id, handle_hash, ciphertext, expires))
        # Publication follows the original COMMIT, which may itself wait.
        # No successful handle may outlive its deadline while SQL settles.
        if expired or expires <= self.now():
            raise PendingMCPAuthorizationExpired("The pending MCP authorization request expired during admission")
        if quota_full:
            raise PendingMCPQuotaExceeded("The agent's pending MCP authorization quota is full")
        return handle

    def _decode(self, handle_hash: str, row) -> PendingMCPAuthorization | None:
        ciphertext, expires = row
        value = self._records.open(MCPRecordKind.PENDING, handle_hash, expires, ciphertext)
        client = self._registered_client(value["client_id"])
        if client is None:
            return None
        # The SDK's registration model preserves an empty URL path, while
        # AuthorizationParams reparses a serialized string with a slash.
        # Re-present the exact registered native URL object, as the official
        # authorization handler does when selecting a client's sole callback.
        redirect = client.native_redirect_uri(value["parameters"]["redirect_uri"])
        if redirect is None:
            return None
        parameters = AuthorizationParams.model_validate({**value["parameters"], "redirect_uri": redirect})
        if not self._currently_registered(value["client_id"], parameters) or expires <= self.now():
            return None
        return PendingMCPAuthorization(value["client_id"], expires, parameters)

    async def read(self, handle: str) -> PendingMCPAuthorization | None:
        handle_hash = hash_mcp_secret(handle)
        row = await self.backend.fetch_one("""SELECT ciphertext, expires_at FROM mcp_server_oauth_pending
            WHERE agent_id = ? AND handle_hash = ? AND consumed = 0 AND expires_at > ?""",
            (self.agent_id, handle_hash, self.now()))
        return None if row is None else self._decode(handle_hash, row)

    async def claim(self, handle: str) -> PendingMCPAuthorization | None:
        """Consume once inside the consent owner's transaction; this mints no grant."""
        handle_hash = hash_mcp_secret(handle)
        async with self.serialized():
            row = await self.backend.fetch_one("""SELECT ciphertext, expires_at FROM mcp_server_oauth_pending
                WHERE agent_id = ? AND handle_hash = ? AND consumed = 0 AND expires_at > ?""",
                (self.agent_id, handle_hash, self.now()))
            pending = None if row is None else self._decode(handle_hash, row)
            if pending is not None:
                claimed = await self.backend.fetch_one("""UPDATE mcp_server_oauth_pending SET consumed = 1
                WHERE agent_id = ? AND handle_hash = ? AND consumed = 0 AND expires_at > ?
                RETURNING ciphertext, expires_at""", (self.agent_id, handle_hash, self.now()))
                if claimed is None or pending.expires_at <= self.now():
                    pending = None
        return pending if pending is not None and pending.expires_at > self.now() else None
