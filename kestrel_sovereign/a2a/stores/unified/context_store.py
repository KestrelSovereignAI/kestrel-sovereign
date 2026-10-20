"""Durable ownership of native A2A contexts, separate from chat projections.

The native handler must verify the sender and recipient authorization before
reserving a context, and reserve before admitting a task. An existing context
cannot acquire a different creator through a later SendMessage. JSON-RPC and
HTTP+JSON share this ledger; the selected recipient and tenant define its scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase


class ContextOwnershipError(PermissionError):
    """The authenticated caller does not own the requested native context."""


class ContextPersistenceError(RuntimeError):
    """A context reservation was not visible after its write completed."""


@dataclass(frozen=True)
class ContextOwnership:
    """One authoritative owner within a recipient's native tenant."""

    context_id: str
    tenant: str
    recipient_agent_id: str
    creator_agent_id: str


class ContextStore(UnifiedStoreBase):
    """Reserve and check native contexts through the canonical database backend."""

    async def initialize(self) -> None:
        """Create the ownership ledger without changing existing session data."""
        await self.backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_contexts (
                context_id TEXT NOT NULL,
                tenant TEXT NOT NULL,
                recipient_agent_id TEXT NOT NULL,
                creator_agent_id TEXT NOT NULL,
                created_at {self.timestamp_type()} {self.now_default()},
                PRIMARY KEY (recipient_agent_id, tenant, context_id)
            )
        """)

    @staticmethod
    def _validate_identity(
        context_id: str, creator_agent_id: str, scope: NativeRecipientScope
    ) -> None:
        if not isinstance(scope, NativeRecipientScope):
            raise TypeError(
                "Context ownership requires a server-owned native recipient scope"
            )
        if not isinstance(context_id, str) or not context_id:
            raise ValueError("Native A2A context_id must be a non-empty string")
        if not isinstance(creator_agent_id, str) or not creator_agent_id.strip():
            raise ValueError("Native A2A creator_agent_id must be a verified identity")

    async def _read_owned(
        self,
        context_id: str,
        scope: NativeRecipientScope,
        creator_agent_id: str,
    ) -> ContextOwnership | None:
        row = await self.backend.fetch_one(
            """
            SELECT creator_agent_id FROM a2a_contexts
            WHERE context_id = ? AND tenant = ? AND recipient_agent_id = ?
            """,
            (context_id, scope.tenant, scope.recipient_id),
        )
        if row is None:
            return None
        if row[0] != creator_agent_id:
            raise ContextOwnershipError(
                "Native A2A context is not owned by this sender"
            )
        return ContextOwnership(
            context_id, scope.tenant, scope.recipient_id, creator_agent_id
        )

    async def reserve(
        self,
        context_id: str,
        scope: NativeRecipientScope,
        *,
        creator_agent_id: str,
    ) -> ContextOwnership:
        """Atomically claim a new context or confirm the same existing owner.

        A lost response or cancellation may leave a reservation for this sender.
        Retrying this ownership operation has no task or message side effect.
        It does not authorize replaying SendMessage after a lost acknowledgement.
        """
        self._validate_identity(context_id, creator_agent_id, scope)
        await self.backend.execute(
            """
            INSERT INTO a2a_contexts (context_id, tenant, recipient_agent_id, creator_agent_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (recipient_agent_id, tenant, context_id) DO NOTHING
            """,
            (context_id, scope.tenant, scope.recipient_id, creator_agent_id),
        )
        ownership = await self._read_owned(context_id, scope, creator_agent_id)
        if ownership is None:
            raise ContextPersistenceError(
                "Native A2A context reservation was not persisted"
            )
        return ownership

    async def require_owned(
        self,
        context_id: str,
        scope: NativeRecipientScope,
        *,
        creator_agent_id: str,
    ) -> ContextOwnership:
        """Check an existing context without creating one or exposing another owner."""
        self._validate_identity(context_id, creator_agent_id, scope)
        ownership = await self._read_owned(context_id, scope, creator_agent_id)
        if ownership is None:
            raise ContextOwnershipError(
                "Native A2A context is not owned by this sender"
            )
        return ownership
