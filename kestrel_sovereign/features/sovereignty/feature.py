import asyncio
import logging
import os
from typing import Dict, Any, Optional
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier
from decimal import Decimal
from datetime import datetime
from kestrel_sovereign.storage import GraphNode

logger = logging.getLogger(__name__)

class SovereigntyFeature(Feature):
    """
    Feature for managing Agent Sovereignty (Backups, Exports, Imports).
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage data sovereignty - export agent state to IPFS/Filecoin for backup, "
            "import and restore from backup CIDs, check sovereignty status and export history. "
            "CALL THIS TOOL when the user asks about: backing up, exporting their data, IPFS exports, "
            "sovereignty status, previous backups, or any question about data portability. "
            "Do NOT answer sovereignty questions from memory - always call this tool to get real data."
        )

    async def initialize(self):
        logger.info("Initializing SovereigntyFeature")
        # Ensure dependencies are available
        if not hasattr(self.agent, 'storage') or not hasattr(self.agent, 'wallet'):
            logger.warning("SovereigntyFeature requires storage and wallet on agent.")

    @tool(
        name="export_sovereignty",
        description="Export the agent's entire state to IPFS/Filecoin for sovereignty backup.",
        category=ToolCategory.SYSTEM,
        command_prefix="!export-sovereignty"
    )
    async def export_sovereignty(self, storage_tier: str = "ipfs", encrypt: bool = True) -> ToolResult:
        """
        Export agent state to IPFS/Filecoin.

        Args:
            storage_tier: 'local', 'ipfs', or 'filecoin' (default: 'ipfs')
            encrypt: Whether to encrypt the backup (default: True)

        Returns:
            ToolResult.ok with CID + tier + size on a clean export, PARTIAL
            when a non-local tier produced no IPFS CID (backup hashed
            locally but not actually pushed to the network), or
            ToolResult.failed when the wallet cannot afford the storage
            fee.
        """
        # Map tier string
        tier_map = {
            "local": StorageTier.LOCAL_ONLY,
            "ipfs": StorageTier.IPFS,
            "filecoin": StorageTier.FILECOIN,
        }
        tier_enum = tier_map.get(storage_tier.lower(), StorageTier.IPFS)

        # Don't encrypt for local storage (no point, and complicates retrieval)
        encrypt = encrypt and tier_enum != StorageTier.LOCAL_ONLY

        # Budget check
        fee_main = Decimal('1.0') if tier_enum != StorageTier.LOCAL_ONLY else Decimal('0.0')
        if fee_main > 0 and not self.agent.wallet.can_afford(fee_main):
            return ToolResult.failed(error="Insufficient funds for backup.")

        # Create backup blob
        backup_blob = await self.agent.storage.create_backup_blob(include_db=True)

        # Store via adapter
        adapter = FilecoinAdapter()
        result = await asyncio.to_thread(
            adapter.store_content,
            backup_blob,
            storage_tier=tier_enum,
            encrypt=encrypt,
            metadata={"agent": self.agent.agent_id}
        )

        # Record graph receipt
        node_id = await self.agent.storage.record_backup_artifact(self.agent.agent_id, result)

        # Deduct funds
        if fee_main > 0:
            await self.agent.wallet.transfer(fee_main, memo=f"backup:{tier_enum.value}:{node_id}")

        audit_anchors = None
        try:
            for feature in getattr(self.agent, 'features', {}).values():
                if type(feature).__name__ == 'AuditAnchorFeature':
                    status_envelope = await feature.anchor_status()
                    # anchor_status returns a ToolResult envelope
                    # (#1061 wave 17); the legacy dict lives under .data.
                    if hasattr(status_envelope, "data") and status_envelope.data is not None:
                        audit_anchors = status_envelope.data
                    else:
                        audit_anchors = status_envelope
                    break
        except Exception:
            logger.debug("Failed to attach audit anchors to sovereignty receipt", exc_info=True)

        receipt_properties = {
            "cid": result.ipfs_cid,
            "ipfs_cid": result.ipfs_cid,
            "content_hash": result.content_hash,
            "storage_tier": result.storage_tier.value,
            "provider": getattr(result, "provider", None),
            "encrypted": result.encrypted,
            "encryption_key_hash": result.encryption_key_hash,
            "size_bytes": getattr(result, "size_bytes", 0) or len(backup_blob),
            "created_at": datetime.now().isoformat(),
            "node_id": node_id,
        }
        if audit_anchors is not None:
            receipt_properties["audit_anchors"] = audit_anchors

        # Create receipt node
        receipt_node = GraphNode(
            node_id=f"sovereignty_receipt_{result.ipfs_cid or datetime.now().timestamp()}",
            node_type="sovereignty_receipt",
            label="Sovereignty Export Receipt",
            properties=receipt_properties
        )
        await self.agent.storage.add_node(receipt_node)

        cid = result.ipfs_cid or result.content_hash
        size_bytes = getattr(result, "size_bytes", 0) or len(backup_blob)
        confirmation = (
            "✅ Sovereignty Export Complete.\n"
            f"CID: {cid}\n"
            f"Tier: {tier_enum.value}\n"
            f"Encrypted: {encrypt}\n"
            f"Size: {len(backup_blob)} bytes\n"
        )
        data = {
            "cid": result.ipfs_cid,
            "content_hash": result.content_hash,
            "tier": tier_enum.value,
            "encrypted": encrypt,
            "size_bytes": size_bytes,
            "node_id": node_id,
        }

        # Honesty: a non-local tier that returned no IPFS CID means the
        # blob is sitting on the local content-addressed store but was
        # not actually published to IPFS/Filecoin. The receipt records
        # that fact, but the LLM should speak it explicitly so the
        # sovereign doesn't believe the backup is durably off-host.
        if tier_enum != StorageTier.LOCAL_ONLY and not result.ipfs_cid:
            return ToolResult.partial(
                confirmation,
                (
                    f"requested tier '{tier_enum.value}' but no IPFS CID was "
                    "returned; backup is hashed locally and is NOT durable "
                    "off-host (content_hash is content-addressed only)."
                ),
                data=data,
            )

        return ToolResult.ok(confirmation, data=data)

    @tool(
        name="import_sovereignty",
        description="Restore the agent's state from an IPFS CID.",
        category=ToolCategory.SYSTEM,
        command_prefix="!import-sovereignty"
    )
    async def import_sovereignty(self, cid: str) -> ToolResult:
        """
        Import agent state from IPFS CID.

        The CID should correspond to a backup artifact that was previously
        exported. If the backup was encrypted, the key will be looked up
        from the backup artifact record.
        """
        try:
            adapter = FilecoinAdapter()

            # Look up the backup artifact to check if it was encrypted
            # The CID maps to a backup_artifact node with content_hash = cid
            backup_nodes = await self.agent.storage.get_nodes_by_type("backup_artifact")
            key_hash = None

            for node in backup_nodes:
                if node.properties.get("ipfs_cid") == cid or node.node_id == cid:
                    key_hash = node.properties.get("encryption_key_hash")
                    logger.info(f"Found backup artifact node for CID {cid}, encrypted: {node.properties.get('encrypted')}")
                    break

            # Retrieve content (will decrypt if key_hash provided)
            content = await asyncio.to_thread(
                adapter.retrieve_content, cid, ipfs_cid=cid, key_hash=key_hash
            )

            if not content:
                return ToolResult.failed(
                    error=f"Could not retrieve content for CID {cid}"
                )

            logger.info(f"Retrieved content size: {len(content)}")

            # Restore from backup blob
            stats = await self.agent.storage.restore_from_backup_blob(content)
            messages_restored = stats.get("messages_restored", 0)

            return ToolResult.ok(
                f"✅ Sovereignty Import Complete. Restored {messages_restored} messages.",
                data={
                    "cid": cid,
                    "messages_restored": messages_restored,
                    "stats": dict(stats) if stats else {},
                },
            )
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return ToolResult.failed(error=f"❌ Error during import: {str(e)}")

    @tool(
        name="check_sovereignty_status",
        description="Check the status of sovereignty backups.",
        category=ToolCategory.SYSTEM,
        command_prefix="!check-sovereignty-status"
    )
    async def check_sovereignty_status(self) -> ToolResult:
        """Check sovereignty status."""
        try:
            receipts = await self.agent.storage.get_nodes_by_type("sovereignty_receipt")
            if not receipts:
                return ToolResult.ok(
                    "No sovereignty exports found.",
                    data={"latest_cid": None, "latest_created_at": None, "total_exports": 0},
                )

            receipts_sorted = sorted(
                receipts,
                key=lambda r: r.properties.get('created_at', ''),
                reverse=True
            )
            latest = receipts_sorted[0]
            latest_cid = latest.properties.get('cid')
            latest_created_at = latest.properties.get('created_at')

            confirmation = (
                "Sovereignty Status:\n"
                f"Latest Export: {latest_created_at}\n"
                f"CID: {latest_cid}\n"
                f"Total Exports: {len(receipts)}\n"
            )
            return ToolResult.ok(
                confirmation,
                data={
                    "latest_cid": latest_cid,
                    "latest_created_at": latest_created_at,
                    "total_exports": len(receipts),
                },
            )
        except Exception as e:
            return ToolResult.failed(error=f"Error checking status: {e}")
