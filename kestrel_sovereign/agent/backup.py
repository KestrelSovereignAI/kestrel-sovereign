"""Backup and restore functionality for Kestrel Agent."""
import asyncio
import logging
from decimal import Decimal


class BackupMixin:
    """Mixin class providing backup methods."""

    async def _command_backup(self, user_input: str) -> str:
        """Create a backup according to privacy and configured storage tier."""
        if self.privacy_agent.privacy_config.is_ephemeral():
            return "Backups are disabled in ephemeral mode."

        tier = "local"
        encrypt = True
        parts = user_input.split()
        for i, part in enumerate(parts[1:], start=1):
            if part == '--tier' and i + 1 < len(parts):
                tier = parts[i + 1]
            elif part.startswith("tier="):
                tier = part.split("=", 1)[1]
            if part == '--no-encrypt':
                encrypt = False
            elif part == "noenc":
                encrypt = False

        from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier

        tier_map = {
            "local": StorageTier.LOCAL_ONLY,
            "ipfs": StorageTier.IPFS,
            "filecoin": StorageTier.FILECOIN,
        }
        storage_tier = tier_map.get(tier, StorageTier.LOCAL_ONLY)

        if self.privacy_agent.privacy_config.uses_temp_storage() and storage_tier != StorageTier.LOCAL_ONLY:
            return "In isolated mode, backups are cached locally only."
        if self.privacy_agent.privacy_config.requires_anonymization() and storage_tier == StorageTier.FILECOIN:
            encrypt = True

        from kestrel_sovereign.storage.sync.service import (
            RemoteTierPolicyContext,
            _remote_tiers_allowed,
        )
        from kestrel_sovereign.storage.tiered_manager import REMOTE_STORAGE_TIERS

        if storage_tier in REMOTE_STORAGE_TIERS:
            has_constitution_anchor = False
            if hasattr(self, "get_constitution_hash"):
                try:
                    has_constitution_anchor = bool(await self.get_constitution_hash())
                except Exception:  # noqa: BLE001
                    has_constitution_anchor = False
            privacy_mode = getattr(getattr(self, "_privacy_mode", None), "value", None)
            decision = _remote_tiers_allowed(
                RemoteTierPolicyContext(
                    identity=getattr(self, "agent_id", None) or getattr(self, "did", None),
                    db_path=getattr(self, "storage_path", None),
                    is_test_instance=bool(getattr(self, "is_test_instance", False)),
                    has_constitution_anchor=has_constitution_anchor,
                    is_sovereign_identity=not str(
                        getattr(self, "agent_id", None) or getattr(self, "did", "")
                    ).lower().startswith("did:test:"),
                    privacy_mode=privacy_mode,
                )
            )
            if not decision.allowed:
                return f"Remote backup skipped by policy: {decision.reason}. Use tier=local."

        fee_main = Decimal('1.0') if storage_tier != StorageTier.LOCAL_ONLY else Decimal('0.0')
        if fee_main > 0 and not self.wallet.can_afford(fee_main):
            return "Insufficient funds for backup."

        backup_blob = await self.storage.create_backup_blob(include_db=True)

        adapter = FilecoinAdapter()
        result = await asyncio.to_thread(
            adapter.store_content, backup_blob, storage_tier=storage_tier, encrypt=encrypt, metadata={"agent": self.agent_id}
        )

        node_id = await self.storage.record_backup_artifact(self.agent_id, result)

        if fee_main > 0:
            await self.wallet.transfer(fee_main, memo=f"backup:{storage_tier.value}:{node_id}")

        msg = f"Backup created: node={node_id} tier={storage_tier.value}"
        if result.ipfs_cid:
            msg += f" cid={result.ipfs_cid}"
        if result.filecoin_deal_id:
            msg += f" deal={result.filecoin_deal_id}"
        return msg

    async def _command_promote_backup(self, user_input: str) -> str:
        """Promote an isolated session by saving it, then create a backup."""
        if self.privacy_agent.privacy_config.uses_temp_storage():
            save_msg = await self.privacy_agent.save_isolated_session()
            if save_msg.startswith("Error"):
                return save_msg
            # Switch to normal mode so the backup can proceed through the privacy wrapper
            from kestrel_sovereign.privacy import PrivacyMode
            await self.set_privacy_mode(PrivacyMode.NORMAL)
        return await self._command_backup(user_input.replace("!promote-backup", "!backup", 1))

    async def anchor_memory_state(self):
        """Creates a cryptographic anchor of current conversation history."""
        if self.privacy_agent.privacy_config.is_ephemeral():
            return "Error: Cannot anchor memory in ephemeral mode."

        history_hash = self.storage.conversation.get_conversation_history_hash()

        cost = self.notary_service.estimate_cost(history_hash)
        if not self.wallet.can_afford(cost):
            return f"Error: Insufficient funds. Need {cost} FIL."

        try:
            await self.wallet.transfer(cost, "Notary Service for memory anchor")
            tx_id = self.notary_service.publish_anchor(history_hash)

            self.storage.conversation.add_log_anchor(history_hash, tx_id)

            return f"Successfully anchored memory state. Hash: {history_hash}, TX: {tx_id}"
        except Exception as e:
            logging.error(f"Failed to publish anchor: {e}")
            return "Error: Failed to publish memory anchor."
