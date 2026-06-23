import asyncio
import inspect
import logging
import os
from typing import Callable, Dict, Any, Optional
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
        # Ensure dependencies are available. Wallet is optional for local
        # exports, but required for paid non-local storage tiers.
        if not hasattr(self.agent, 'storage'):
            logger.warning("SovereigntyFeature requires storage on agent.")

    @tool(
        name="export_sovereignty",
        description="Export the agent's entire state to IPFS/Filecoin for sovereignty backup.",
        category=ToolCategory.SYSTEM,
        command_prefix="!export-sovereignty"
    )
    async def export_sovereignty(
        self,
        storage_tier: str = "ipfs",
        encrypt: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> ToolResult:
        """
        Export agent state to IPFS/Filecoin.

        Args:
            storage_tier: 'local', 'ipfs', or 'filecoin' (default: 'ipfs')
            encrypt: Whether to encrypt the backup (default: True)
            on_progress: Optional callback(bytes_sent, total_bytes)

        Returns:
            ToolResult.ok with CID + tier + size on a clean export, PARTIAL
            when a non-local tier produced no IPFS CID (backup hashed
            locally but not actually pushed to the network), or
            ToolResult.failed when a paid storage tier cannot be accounted
            for through a wallet.
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

        wallet = getattr(self.agent, "wallet", None)

        # Budget check. Wallet is an optional feature package; core-only
        # agents can export locally, while paid tiers still require it.
        fee_main = Decimal('1.0') if tier_enum != StorageTier.LOCAL_ONLY else Decimal('0.0')
        if fee_main > 0:
            if wallet is None:
                return ToolResult.failed(
                    error=(
                        f"{tier_enum.value} sovereignty export requires the "
                        "wallet feature. Use storage_tier='local' for a "
                        "core-only export."
                    )
                )
            if not wallet.can_afford(fee_main):
                return ToolResult.failed(error="Insufficient funds for backup.")

        # Create backup blob
        backup_blob = await self.agent.storage.create_backup_blob(include_db=True)
        progress = self._build_progress_reporter(
            on_progress=on_progress,
            tier=tier_enum.value,
            total_bytes=len(backup_blob),
        )
        progress(0, len(backup_blob))

        # Store via adapter
        adapter = FilecoinAdapter()
        result = await asyncio.to_thread(
            adapter.store_content,
            backup_blob,
            storage_tier=tier_enum,
            encrypt=encrypt,
            metadata={"agent": self.agent.agent_id},
            on_progress=progress,
        )
        progress(len(backup_blob), len(backup_blob))

        # Record graph receipt
        node_id = await self.agent.storage.record_backup_artifact(self.agent.agent_id, result)

        # Deduct funds
        if fee_main > 0 and wallet is not None:
            await wallet.transfer(fee_main, memo=f"backup:{tier_enum.value}:{node_id}")

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
        # ``FilecoinAdapter.store_content`` downgrades ``result.tier``
        # to LOCAL_ONLY when the provider stack (Lotus/IPFS) is
        # unreachable. We report the *actual* tier here, not the
        # tier the caller asked for, so a fallback to local doesn't
        # surface as "Tier: ipfs" while the receipt and the real
        # storage are local-only. The requested tier is also exposed
        # in ``data`` so callers can detect the downgrade.
        actual_tier = result.storage_tier
        confirmation = (
            "✅ Sovereignty Export Complete.\n"
            f"CID: {cid}\n"
            f"Tier: {actual_tier.value}\n"
            f"Encrypted: {encrypt}\n"
            f"Size: {len(backup_blob)} bytes\n"
        )
        data = {
            "cid": result.ipfs_cid,
            "content_hash": result.content_hash,
            "tier": actual_tier.value,
            "tier_requested": tier_enum.value,
            "encrypted": encrypt,
            "size_bytes": size_bytes,
            "node_id": node_id,
        }

        # Honesty: a non-local request that ended up local (or that
        # returned no IPFS CID) means the blob is sitting on the local
        # content-addressed store but was not actually published to
        # IPFS/Filecoin. The receipt records that fact, but the LLM
        # should speak it explicitly so the sovereign doesn't believe
        # the backup is durably off-host.
        downgraded = (
            tier_enum != StorageTier.LOCAL_ONLY
            and (actual_tier == StorageTier.LOCAL_ONLY or not result.ipfs_cid)
        )
        if downgraded:
            return ToolResult.partial(
                confirmation,
                (
                    f"requested tier '{tier_enum.value}' but actual tier is "
                    f"'{actual_tier.value}' and no IPFS CID was returned; "
                    "backup is hashed locally and is NOT durable off-host "
                    "(content_hash is content-addressed only)."
                ),
                data=data,
            )

        return ToolResult.ok(confirmation, data=data)

    def _build_progress_reporter(
        self,
        *,
        on_progress: Optional[Callable[[int, int], None]],
        tier: str,
        total_bytes: int,
    ) -> Callable[[int, int], None]:
        """Create a thread-safe progress reporter for export uploads."""
        loop = asyncio.get_running_loop()
        agent_id = getattr(self.agent, "agent_id", None)
        last_logged_percent = -10
        last_emitted_percent = -1

        def _call_user_callback(sent: int, total: int) -> None:
            if not callable(on_progress):
                return
            try:
                maybe_awaitable = on_progress(sent, total)
                if inspect.isawaitable(maybe_awaitable):
                    loop.call_soon_threadsafe(
                        asyncio.create_task, maybe_awaitable
                    )
            except Exception:
                logger.debug("Sovereignty export progress callback failed", exc_info=True)

        async def _emit(sent: int, total: int, percent: int) -> None:
            emit_event = getattr(self.agent, "emit_event", None)
            if emit_event is None:
                return
            await emit_event(
                "sovereignty_export_progress",
                {
                    "agent_id": agent_id,
                    "tier": tier,
                    "bytes_sent": sent,
                    "total_bytes": total,
                    "percent": percent,
                },
            )

        def _report(sent: int, upload_total: int) -> None:
            nonlocal last_logged_percent, last_emitted_percent
            total = upload_total or total_bytes or 0
            if total <= 0:
                normalized_sent = sent
                percent = 0
            else:
                normalized_sent = min(max(sent, 0), total)
                percent = int((normalized_sent / total) * 100)

            _call_user_callback(normalized_sent, total)

            if percent >= last_logged_percent + 10 or percent in (0, 100):
                last_logged_percent = percent
                logger.info(
                    "Sovereignty export upload progress: %s/%s bytes (%s%%)",
                    normalized_sent,
                    total,
                    percent,
                )

            if percent != last_emitted_percent:
                last_emitted_percent = percent
                loop.call_soon_threadsafe(
                    asyncio.create_task, _emit(normalized_sent, total, percent)
                )

        return _report

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
