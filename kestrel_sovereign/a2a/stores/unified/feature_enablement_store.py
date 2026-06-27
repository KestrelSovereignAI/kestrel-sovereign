"""Per-agent feature / MCP-server **enablement** deltas, persisted in the DB.

Two distinct concerns must not be conflated (see the capability-management
design ruling):

* **Provisioning** — which feature *packages* are installed in the host venv —
  stays in config (``.kestrel-host-features.toml`` + ``feature_reconcile``).
* **Enablement** — which installed features a given agent *activates*, and which
  MCP servers it has enabled — is **per-agent runtime state**, so it lives here
  in the agent's DB rather than in ``multi_agent.toml``.

``multi_agent.toml``'s ``[agents.*].features`` allowlist is the operator's
**bootstrap** set (what loads at birth). When the agent itself adds/removes a
feature (``FeatureFeaturesFeature``) or enables/disables an MCP server
(``MCPAgent``), it records a **delta** here so the change survives restart — the
"installed = config, activated = DB" pattern (cf. WordPress ``active_plugins``;
mirrors per-agent permissions, which already live in the DB). One table serves
both kinds; startup computes ``effective = (bootstrap ∪ enabled) − disabled``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase

KIND_FEATURE = "feature"
KIND_MCP_SERVER = "mcp_server"
STATE_ENABLED = "enabled"
STATE_DISABLED = "disabled"


class FeatureEnablementStore(UnifiedStoreBase):
    """CRUD over the ``feature_enablement`` delta table (SQLite + Postgres)."""

    async def initialize(self) -> None:
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()
        int_pk = self.integer_primary_key_type()

        await self._backend.execute_script(
            f"""
            CREATE TABLE IF NOT EXISTS feature_enablement (
                id {int_pk},
                agent_did TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('feature', 'mcp_server')),
                name TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('enabled', 'disabled')),
                actor TEXT,
                metadata {json_type},
                updated_at {ts_type} {ts_default},
                UNIQUE(agent_did, kind, name)
            )
            """
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_feature_enablement_agent_kind "
            "ON feature_enablement(agent_did, kind)"
        )

    async def set_state(
        self,
        *,
        agent_did: str,
        kind: str,
        name: str,
        state: str,
        actor: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert the latest enablement state for ``(agent_did, kind, name)``."""
        meta = json.dumps(metadata) if metadata is not None else None
        await self._backend.execute(
            """
            INSERT INTO feature_enablement
                (agent_did, kind, name, state, actor, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_did, kind, name) DO UPDATE SET
                state = excluded.state,
                actor = excluded.actor,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (agent_did, kind, name, state, actor, meta, self.now_utc_param()),
        )

    async def get_deltas(
        self, agent_did: str, kind: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return enablement deltas for an agent (optionally filtered by kind)."""
        if kind is None:
            rows = await self._backend.fetch_all(
                "SELECT kind, name, state, actor, metadata, updated_at "
                "FROM feature_enablement WHERE agent_did = ?",
                (agent_did,),
            )
        else:
            rows = await self._backend.fetch_all(
                "SELECT kind, name, state, actor, metadata, updated_at "
                "FROM feature_enablement WHERE agent_did = ? AND kind = ?",
                (agent_did, kind),
            )
        deltas: List[Dict[str, Any]] = []
        for r in rows or []:
            deltas.append(
                {
                    "kind": r[0],
                    "name": r[1],
                    "state": r[2],
                    "actor": r[3],
                    "metadata": json.loads(r[4]) if r[4] else None,
                    "updated_at": r[5],
                }
            )
        return deltas

    async def clear(self, agent_did: str, kind: str, name: str) -> None:
        """Drop a delta entirely (revert to the bootstrap default for it)."""
        await self._backend.execute(
            "DELETE FROM feature_enablement "
            "WHERE agent_did = ? AND kind = ? AND name = ?",
            (agent_did, kind, name),
        )
