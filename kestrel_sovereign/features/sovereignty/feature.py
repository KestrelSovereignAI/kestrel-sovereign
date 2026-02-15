import asyncio
import logging
import os
from typing import Dict, Any, Optional
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory
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
            "import and restore from backup CIDs, check sovereignty status and export history"
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
    async def export_sovereignty(self, storage_tier: str = "ipfs", encrypt: bool = True) -> str:
        """
        Export agent state to IPFS/Filecoin.
        
        Args:
            storage_tier: 'local', 'ipfs', or 'filecoin' (default: 'ipfs')
            encrypt: Whether to encrypt the backup (default: True)
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
            return "Insufficient funds for backup."

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

        # Create receipt node
        receipt_node = GraphNode(
            node_id=f"sovereignty_receipt_{result.ipfs_cid or datetime.now().timestamp()}",
            node_type="sovereignty_receipt",
            label="Sovereignty Export Receipt",
            properties={
                "cid": result.ipfs_cid,
                "storage_tier": tier_enum.value,
                "encrypted": encrypt,
                "size_bytes": len(backup_blob),
                "created_at": datetime.now().isoformat(),
                "node_id": node_id
            }
        )
        await self.agent.storage.add_node(receipt_node)

        return f"""✅ Sovereignty Export Complete.
CID: {result.ipfs_cid or result.content_hash}
Tier: {tier_enum.value}
Encrypted: {encrypt}
Size: {len(backup_blob)} bytes
"""

    @tool(
        name="import_sovereignty",
        description="Restore the agent's state from an IPFS CID.",
        category=ToolCategory.SYSTEM,
        command_prefix="!import-sovereignty"
    )
    async def import_sovereignty(self, cid: str) -> str:
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
                return f"❌ Error: Could not retrieve content for CID {cid}"
            
            logger.info(f"Retrieved content size: {len(content)}")
            
            # Restore from backup blob
            stats = await self.agent.storage.restore_from_backup_blob(content)
            
            return f"✅ Sovereignty Import Complete. Restored {stats.get('messages_restored', 0)} messages."
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return f"❌ Error during import: {str(e)}"

    @tool(
        name="check_sovereignty_status",
        description="Check the status of sovereignty backups.",
        category=ToolCategory.SYSTEM,
        command_prefix="!check-sovereignty-status"
    )
    async def check_sovereignty_status(self) -> str:
        """Check sovereignty status."""
        try:
            receipts = await self.agent.storage.get_nodes_by_type("sovereignty_receipt")
            if not receipts:
                return "No sovereignty exports found."
                
            receipts_sorted = sorted(
                receipts,
                key=lambda r: r.properties.get('created_at', ''),
                reverse=True
            )
            latest = receipts_sorted[0]
            
            return f"""Sovereignty Status:
Latest Export: {latest.properties.get('created_at')}
CID: {latest.properties.get('cid')}
Total Exports: {len(receipts)}
"""
        except Exception as e:
            return f"Error checking status: {e}"
