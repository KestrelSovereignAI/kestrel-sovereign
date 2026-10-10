"""Field-only identity updates under native graph custody.

An earlier identity snapshot is never authority to replace governance receipts.
Use the caller's storage surface so privacy policy still governs each write.
"""

from copy import deepcopy
import json
from collections.abc import Mapping


def _merged_properties(properties, updates):
    if not isinstance(properties, Mapping):
        raise ValueError("Identity metadata requires readable existing properties")
    return {**deepcopy(properties), **deepcopy(updates)}


async def merge_identity_metadata_in_database(db, agent_id, updates):
    """Host-owned field update using a borrowed native database, never an upsert.

    This surface is for existing control-plane producers that own a database
    rather than a privacy storage facade. It keeps the same graph reservation
    order and fresh-read merge as the facade operation, without constructing
    another database owner or granting an identity that does not exist.
    """
    from kestrel_sovereign.storage.async_graph_store import lock_graph_nodes_for_update

    async with db.transaction(immediate=db.backend_type == "sqlite"):
        await lock_graph_nodes_for_update(db, [agent_id])
        row = await db.fetchone(
            "SELECT node_type, properties FROM graph_nodes WHERE node_id = ?",
            (agent_id,),
        )
        if row is None:
            return False
        if row[0] != "agent":
            raise ValueError("Identity metadata requires an existing agent root")
        properties = row[1]
        if isinstance(properties, (str, bytes, bytearray)):
            properties = json.loads(properties)
        merged = _merged_properties(properties, updates)
        affected = await db.execute(
            "UPDATE graph_nodes SET properties = ? WHERE node_id = ? AND node_type = 'agent'",
            (json.dumps(merged), agent_id),
        )
        return affected != 0


async def merge_identity_metadata(storage, agent_id, updates, *, label=None, capability=None):
    """Return the updated root, or None if absent; never create an identity."""
    async with storage.transaction():
        await storage.lock_nodes_for_update([agent_id])
        fresh = await storage.get_node(agent_id)
        if fresh is None:
            return None
        if fresh.node_type != "agent":
            raise ValueError("Identity metadata requires an existing agent root")
        # Do not mutate a caller/cache-owned read object before the actual
        # writer succeeds (a failed rename must leave its old live label).
        fresh = deepcopy(fresh)
        fresh.properties = _merged_properties(fresh.properties, updates)
        if label is not None:
            fresh.label = label
        if capability is None:
            await storage.add_node(fresh)
        else:
            await storage.add_node(fresh, capability=capability)
        return fresh
