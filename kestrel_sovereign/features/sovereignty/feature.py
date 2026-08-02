import asyncio
import contextlib
import inspect
import logging
from typing import Callable, Optional
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier
from decimal import Decimal
from datetime import datetime
from kestrel_sovereign.storage import GraphNode
from kestrel_sovereign.storage.privacy_wrapper import (
    PrivacyEnforcingStorage,
    PrivacyPolicy,
    PrivacyViolationError,
    optional_transition_lock,
)

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
        description="Export the agent's entire state to IPFS/Filecoin for sovereignty backup. "
                    "storage_tier must be one of 'local', 'ipfs' (default), or 'filecoin'; an "
                    "unrecognized value is rejected (it is NOT silently defaulted to ipfs).",
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
            on_progress: Optional callback(bytes_sent, total_bytes). Not
                LLM-supplied (an LLM can't pass a callable); retained for
                internal callers and the export-progress SSE pipeline,
                which has dedicated test coverage.

        Returns:
            ToolResult.ok on a clean, durable, correctly-encrypted export;
            ToolResult.partial when the backup was produced but the requested
            durability tier could not be honoured (the blob is
            content-addressed on THIS host only, not a durable off-host
            replica); or ToolResult.failed when the requested encryption
            cannot be honoured, the privacy policy forbids the export, or a
            paid storage tier cannot be accounted for through a wallet.

        Contract (#2872): a requested security/durability parameter is NEVER
        silently downgraded under an OK envelope. ``encrypt=True`` that does
        not yield an encrypted blob is a hard failure, and a requested
        off-host tier that resolves to a local-only copy is surfaced as
        PARTIAL with an explicit reason — validated by durability provenance,
        not by the mere presence of a CID.
        """
        # Validate storage_tier explicitly. The old code resolved this via
        # ``tier_map.get(storage_tier.lower(), StorageTier.IPFS)``, so a
        # typo'd / unknown tier SILENTLY fell through to IPFS — the agent
        # could believe it kept the backup local-only while it was actually
        # (attempted to be) published to the network, or vice-versa. Reject
        # unknown / wrong-type tiers loudly. Only an explicitly omitted value
        # (None or empty/whitespace string) keeps the documented default
        # ('ipfs'); a falsy non-string such as ``false``/``0``/``[]`` is a
        # wrong type, NOT an omission, and must be rejected.
        # Only the tiers FilecoinAdapter.store_content actually persists are
        # exposed. The cloud_hot / cloud_cold StorageTier members exist in
        # the enum but the sovereignty export adapter has no storage path for
        # them — passing them through silently produced a receipt + wallet
        # charge for a blob that was never written. They are kept OUT of this
        # map (and out of the endpoint allowlist) until an implementation
        # lands. The old ``.get(..., IPFS)`` collapsed cloud_hot/cloud_cold to
        # IPFS silently; rejecting is the honest behavior.
        tier_map = {
            "local": StorageTier.LOCAL_ONLY,
            "ipfs": StorageTier.IPFS,
            "filecoin": StorageTier.FILECOIN,
        }
        if storage_tier is None or (isinstance(storage_tier, str) and not storage_tier.strip()):
            tier_key = "ipfs"
        elif isinstance(storage_tier, str):
            tier_key = storage_tier.strip().lower()
        else:
            tier_key = None  # wrong type → fall into the rejection below
        if tier_key not in tier_map:
            return ToolResult.failed(
                error=(
                    f"storage_tier must be one of {', '.join(tier_map)}; "
                    f"got {storage_tier!r}. The tier was NOT silently "
                    "defaulted to ipfs — re-run with a valid tier."
                )
            )
        tier_enum = tier_map[tier_key]

        # Honour the caller's encryption request verbatim (#2872 Defect 1).
        # The old code silently cleared ``encrypt`` for LOCAL_ONLY
        # (``encrypt = encrypt and tier_enum != LOCAL_ONLY``), so an explicit
        # ``encrypt=True`` returned ``encrypted: False`` with no error — a
        # full-state export written to disk in the clear while the caller was
        # told nothing. ``FilecoinAdapter.store_content`` encrypts BEFORE it
        # branches on tier, so encryption is tier-independent (a local
        # fallback stays encrypted); we pass the request through unchanged and
        # then VERIFY the adapter actually honoured it below.
        requested_encrypt = bool(encrypt)

        # Policy denial BEFORE building the blob, inside the error boundary
        # (#2872 in-scope #3). A privacy configuration that forbids cloud
        # backups (EPHEMERAL / ISOLATED / DEIDENTIFIED) must refuse the export
        # loudly and must NOT first read the agent's private state into a blob.
        denial = self._backup_denied_reason()
        if denial:
            return ToolResult.failed(error=denial)

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

        # Build + store under the privacy-transition lock so a concurrent mode
        # flip cannot land between the denial re-check and the storage writes.
        # Any ``PrivacyViolationError`` raised by the real
        # ``PrivacyEnforcingStorage`` is caught here and reported as a loud
        # refusal rather than an unhandled 500 (#2872 in-scope #3).
        try:
            async with optional_transition_lock(self._resolve_privacy_lock()):
                denial = self._backup_denied_reason()
                if denial:
                    return ToolResult.failed(error=denial)

                # Create backup blob
                backup_blob = await self.agent.storage.create_backup_blob(include_db=True)
                progress = self._build_progress_reporter(
                    on_progress=on_progress,
                    tier=tier_enum.value,
                    total_bytes=len(backup_blob),
                )
                progress(0, len(backup_blob))

                # Store via adapter. ``asyncio.to_thread`` runs the synchronous
                # ``store_content`` on an executor thread that CANNOT be
                # interrupted mid-write. If THIS coroutine is cancelled (client
                # disconnect, request timeout) while the write is in flight, we
                # must NOT let the CancelledError unwind through the enclosing
                # ``async with`` and release the privacy-transition lock while
                # the backup is still being written — a concurrent flip to
                # EPHEMERAL/ISOLATED could then land mid-write, defeating the
                # whole point of holding the lock. So shield the store from the
                # outer cancellation, and on cancellation DRAIN the thread to
                # completion (holding the lock the entire time) before
                # re-raising the cancellation, which then releases the lock.
                adapter = FilecoinAdapter()
                store_future = asyncio.ensure_future(
                    asyncio.to_thread(
                        adapter.store_content,
                        backup_blob,
                        storage_tier=tier_enum,
                        encrypt=requested_encrypt,
                        metadata={"agent": self.agent.agent_id},
                        on_progress=progress,
                    )
                )
                try:
                    result = await asyncio.shield(store_future)
                except asyncio.CancelledError:
                    # Join the un-interruptible adapter thread before the lock
                    # is released. ``shield`` keeps ``store_future`` alive
                    # despite the outer cancellation; wait for it, suppressing
                    # any error (we are aborting anyway), then re-raise so the
                    # lock is only released once the write has finished.
                    with contextlib.suppress(BaseException):
                        await asyncio.shield(store_future)
                    raise
                progress(len(backup_blob), len(backup_blob))

                # Encryption honesty gate (#2872 Defect 1). Act on the ACTUAL
                # result, never the request: if encryption was asked for but
                # the stored blob is not encrypted, refuse LOUDLY and do NOT
                # record a success receipt for it.
                if requested_encrypt and not result.encrypted:
                    return ToolResult.failed(
                        error=(
                            "Requested encrypt=True but the stored backup is "
                            f"NOT encrypted (tier '{result.storage_tier.value}'). "
                            "Encryption is a security contract and is never "
                            "silently downgraded, so the export was refused. "
                            "Ensure KESTREL_DATA_KEY is set (required for backup "
                            "encryption) and retry, or pass encrypt=False to "
                            "accept an unencrypted backup."
                        )
                    )

                # Evaluate achieved durability BEFORE charging (#2872 review).
                # A requested off-host tier that downgraded to a weaker one did
                # NOT deliver the paid service, so it must not be billed. This
                # is validated by durability provenance (a Filecoin deal or a
                # non-local pinning provider), not by CID presence alone, and is
                # reused for the PARTIAL envelope below.
                downgrade_reason = self._tier_downgrade_reason(tier_enum, result)

                # Record graph receipt. The blob IS physically stored (even on a
                # local-only downgrade), so the receipt reflects storage truth
                # regardless of tier or billing outcome.
                node_id = await self.agent.storage.record_backup_artifact(self.agent.agent_id, result)

                # Deduct funds only when the requested paid tier was actually
                # achieved, and only when the wallet CONFIRMS the charge.
                # ``wallet.transfer`` returns ``False`` on failure (e.g. funds
                # spent concurrently since ``can_afford``); an ignored ``False``
                # silently gives away paid storage, so a failed charge fails the
                # whole export loudly (#2872 review).
                if fee_main > 0 and wallet is not None and downgrade_reason is None:
                    transferred = await wallet.transfer(
                        fee_main, memo=f"backup:{tier_enum.value}:{node_id}"
                    )
                    if not transferred:
                        return ToolResult.failed(
                            error=(
                                f"{tier_enum.value} sovereignty export was stored "
                                f"but the {fee_main} storage fee could not be "
                                "charged (wallet.transfer returned failure). The "
                                "export is reported as failed so storage and "
                                "accounting stay consistent — retry once the "
                                "wallet can settle the charge."
                            )
                        )

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
        except PrivacyViolationError as exc:
            logger.warning("Sovereignty export refused by privacy policy: %s", exc)
            return ToolResult.failed(
                error=(
                    "Export refused: the current privacy configuration forbids "
                    f"creating a backup ({exc})."
                )
            )
        except Exception as exc:
            logger.error("Sovereignty export failed: %s", exc, exc_info=True)
            return ToolResult.failed(error=f"Sovereignty export failed: {exc}")

        cid = result.ipfs_cid or result.content_hash
        size_bytes = getattr(result, "size_bytes", 0) or len(backup_blob)
        # ``FilecoinAdapter.store_content`` downgrades ``result.tier``
        # to LOCAL_ONLY when the provider stack (Lotus/IPFS) is
        # unreachable. We report the *actual* tier and the *actual*
        # ``result.encrypted`` here, not the values the caller asked for,
        # so a fallback to local doesn't surface as "Tier: ipfs" while the
        # receipt and the real storage are local-only. The requested tier is
        # also exposed in ``data`` so callers can detect the downgrade.
        actual_tier = result.storage_tier
        confirmation = (
            "✅ Sovereignty Export Complete.\n"
            f"CID: {cid}\n"
            f"Tier: {actual_tier.value}\n"
            f"Encrypted: {result.encrypted}\n"
            f"Size: {len(backup_blob)} bytes\n"
        )
        data = {
            "cid": result.ipfs_cid,
            "content_hash": result.content_hash,
            "tier": actual_tier.value,
            "tier_requested": tier_enum.value,
            "encrypted": result.encrypted,
            "size_bytes": size_bytes,
            "node_id": node_id,
        }

        # Tier durability honesty (#2872 in-scope #2). ``downgrade_reason`` was
        # computed above BEFORE the paid-tier charge, so a request that did not
        # achieve durable off-host storage is both left unbilled and surfaced
        # here as PARTIAL — validated by durability provenance (a Filecoin deal
        # or a remote pinning provider), NOT by CID presence alone. A CID pinned
        # to a local Kubo node is content-addressed on this host only and dies
        # with it, so it is not a durable off-host replica.
        if downgrade_reason:
            return ToolResult.partial(confirmation, downgrade_reason, data=data)

        return ToolResult.ok(confirmation, data=data)

    # Durability ranks for the tiers export actually persists. The rank
    # measures OFF-HOST durability (surviving loss of THIS host), so a local
    # pin and a local-only copy are both rank 0.
    _TIER_DURABILITY_RANK = {
        StorageTier.BROWSER: 0,
        StorageTier.LOCAL: 0,
        StorageTier.LOCAL_ONLY: 0,
        StorageTier.IPFS: 1,
        StorageTier.CLOUD_HOT: 1,
        StorageTier.FILECOIN: 2,
        StorageTier.ENCRYPTED_FILECOIN: 2,
        StorageTier.CLOUD_COLD: 2,
    }

    def _requested_tier_rank(self, tier: StorageTier) -> int:
        return self._TIER_DURABILITY_RANK.get(tier, 0)

    def _achieved_durability_rank(self, result) -> int:
        """Rank the durability a storage result actually ATTESTS, by provenance.

        A bare CID is not proof of off-host durability: ``FilecoinAdapter``
        pins through a local Kubo node by default and returns
        ``provider="local"`` even when it produced a CID (#2872). Off-host
        durability is proven by a Filecoin deal (cold storage) or a non-local
        pinning provider (hot storage).
        """
        if getattr(result, "filecoin_deal_id", None):
            return 2
        provider = (getattr(result, "provider", None) or "").strip().lower()
        cid = getattr(result, "ipfs_cid", None)
        if cid and provider and provider != "local":
            return 1
        return 0

    def _tier_downgrade_reason(self, requested_tier: StorageTier, result) -> Optional[str]:
        """Explain a durability shortfall, or return None when the request held.

        Returns a caller-facing reason when the requested off-host tier was not
        durably achieved (#2872 in-scope #2), so the export is reported as
        PARTIAL instead of a silent local downgrade under an OK envelope.
        """
        requested = self._requested_tier_rank(requested_tier)
        if requested <= 0:
            return None
        if self._achieved_durability_rank(result) >= requested:
            return None
        actual_tier = getattr(result, "storage_tier", None)
        actual_tier_value = actual_tier.value if actual_tier is not None else "unknown"
        return (
            f"requested tier '{requested_tier.value}' but actual tier is "
            f"'{actual_tier_value}' (provider "
            f"'{getattr(result, 'provider', None)}', cid "
            f"'{getattr(result, 'ipfs_cid', None)}', filecoin_deal_id "
            f"'{getattr(result, 'filecoin_deal_id', None)}'): the backup is "
            "content-addressed on this host only and is NOT a durable off-host "
            "replica. Durable off-host storage requires a remote pinning/deal "
            "provider (wiring tracked separately in #2873/#2874)."
        )

    def _backup_denied_reason(self) -> Optional[str]:
        """Return a refusal reason when privacy policy forbids any backup.

        Mirrors ``PrivacyEnforcingStorage.create_backup_blob``'s own guard via
        the wrapper's PUBLIC surface (#2872 in-scope #3), so the export refuses
        BEFORE the agent's private state is read into a blob. EPHEMERAL /
        ISOLATED / DEIDENTIFIED configs set ``allow_cloud_backup=False``.
        """
        storage = getattr(self.agent, "storage", None)
        if isinstance(storage, PrivacyEnforcingStorage):
            policy = PrivacyPolicy.from_config(storage.privacy_config)
            if not policy.allow_cloud_backup:
                return (
                    "Export refused: the current privacy configuration "
                    f"(storage={storage.privacy_config.storage}) disables "
                    "backups. Switch to a mode that permits cloud backups "
                    "(e.g. NORMAL) before exporting."
                )
        return None

    def _resolve_privacy_lock(self):
        """Best-effort resolve the agent's privacy-transition lock, or None.

        Test / CLI agent shapes have no ``_get_privacy_transition_lock``; those
        run unguarded via ``optional_transition_lock(None)``.
        """
        getter = getattr(self.agent, "_get_privacy_transition_lock", None)
        if callable(getter):
            try:
                return getter()
            except Exception:  # noqa: BLE001 - never let lock resolution block an export
                return None
        return None

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
        description=(
            "Restore this agent's CONVERSATION HISTORY from a prior backup "
            "(IPFS CID). Faithfully preserves message timestamps and trash "
            "state. NOTE: this currently restores conversation history only — "
            "NOT full agent state (memories, knowledge graph, saved items, "
            "files, settings). Full-state restore is tracked separately."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!import-sovereignty"
    )
    async def import_sovereignty(self, cid: str) -> ToolResult:
        """
        Restore conversation history from an IPFS CID backup.

        The CID should correspond to a backup artifact that was previously
        exported. If the backup was encrypted, the key will be looked up
        from the backup artifact record.

        Scope (#F265): restores ``conversation_history`` faithfully —
        ``created_at`` (ordering) and ``deleted_at`` (trash) are preserved, so
        the restore neither reorders history nor un-deletes trashed messages.
        Other agent state (memories, graph_nodes, saved_items, files, settings)
        is NOT restored yet; full multi-table restore is a tracked follow-up.
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
                f"✅ Restored {messages_restored} conversation messages "
                f"(conversation history only — other agent state is not "
                f"restored yet).",
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
