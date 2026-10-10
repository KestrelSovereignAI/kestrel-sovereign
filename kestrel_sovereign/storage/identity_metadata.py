"""Field-only identity updates under native graph custody.

An earlier identity snapshot is never authority to replace governance receipts.
Use the caller's storage surface so privacy policy still governs each write.
"""

from copy import deepcopy


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
        fresh.properties = {**fresh.properties, **deepcopy(updates)}
        if label is not None:
            fresh.label = label
        if capability is None:
            await storage.add_node(fresh)
        else:
            await storage.add_node(fresh, capability=capability)
        return fresh
