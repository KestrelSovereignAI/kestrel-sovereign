import asyncio
import functools
import inspect
import logging
from typing import Callable, Optional

import httpx

from kestrel_sovereign._async_ownership import await_owned_task, raise_owned_outcome
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.filecoin_adapter import (
    ContentRetrievalLimitError,
    DecryptionKeyUnavailableError,
    EncryptedEnvelopeError,
    EnvelopeKeyMismatchError,
    EnvelopeOwnerMismatchError,
    EnvelopePolicy,
    FilecoinAdapter,
    LegacyDecryptionUnavailableError,
    StorageTier,
    UnencryptedContentError,
    is_cache_name_of,
)
from decimal import Decimal
from datetime import datetime, timezone
from kestrel_sovereign.audit_time import utc_now_iso
from kestrel_sovereign.storage import GraphNode
from kestrel_sovereign.storage.privacy_wrapper import (
    BackupPolicyUnavailableError,
    PrivacyTransitionGuardUnavailableError,
    PrivacyViolationError,
    guarded_backup_store,
    require_backup_permitted,
    require_durable_persistence_gate,
)
from kestrel_sovereign.storage.sync.health import check_lighthouse_health
from kestrel_sovereign.storage.sync.retention import load_retention_policy
from kestrel_sovereign.storage.sync.sovereignty_exports import (
    sovereignty_export_deletions,
    sovereignty_export_tag,
    sovereignty_export_upload_name,
)

logger = logging.getLogger(__name__)

# Typed refusal codes carried on ``ToolResult.failed(data={"refusal": ...})``.
# A refusal is the caller asking for something this host cannot honour — NOT a
# server fault — so HTTP callers map these codes to a 4xx instead of
# substring-sniffing the human-readable message (#2918). Endpoint mapping lives
# in ``kestrel_sovereign/endpoints/sovereignty.py``.
REFUSAL_ENCRYPTION_KEY_UNAVAILABLE = "encryption_key_unavailable"
REFUSAL_INSUFFICIENT_FUNDS = "insufficient_funds"
# The agent's privacy mode does not permit a backup, read before the backup
# blob or, after a transition, immediately before the store (#2875).
REFUSAL_PRIVACY_MODE_FORBIDS_BACKUP = "privacy_mode_forbids_backup"
# Import refusals (#2873): an export that cannot be decrypted, or cannot be
# trusted, is refused before any of it reaches the restore.
REFUSAL_ENCRYPTION_KEY_MISMATCH = "encryption_key_mismatch"
REFUSAL_EXPORT_OF_ANOTHER_AGENT = "export_of_another_agent"
REFUSAL_DECRYPTION_FAILED = "decryption_failed"
REFUSAL_LEGACY_DECRYPTION_METADATA_UNAVAILABLE = "legacy_decryption_metadata_unavailable"
REFUSAL_UNENCRYPTED_WITHOUT_RECEIPT = "unencrypted_export_without_receipt"
REFUSAL_EXPORT_EXCEEDS_IMPORT_LIMIT = "export_exceeds_import_limit"
REFUSAL_BACKUP_RECEIPT_INVALID = "backup_receipt_invalid"

# The largest backup blob an import restores. A fresh host restores whatever
# a caller-supplied CID returns, before anything has authenticated it (#2873),
# so the IPFS read and the decompression stop at a ceiling derived from this
# one (twice it, plus an allowance for ciphertext overhead), and a decrypted
# blob larger than it is refused. The largest export observed, a long-lived
# agent's whole database, was 105 MB; 1 GiB leaves ten times that.
MAX_SOVEREIGNTY_IMPORT_BYTES = 1024 * 1024 * 1024


class _LighthousePinning:
    """The agent's Lighthouse provider as the IPFS tier's pinning service.

    ``FilecoinAdapter.store_content`` runs in a worker thread and calls
    ``pin`` there. The provider's client belongs to the agent's event loop,
    so the upload runs on that loop while the worker waits for its CID.
    """

    def __init__(self, provider, *, tag: str, loop: asyncio.AbstractEventLoop):
        self.provider = provider
        self._tag = tag
        self._loop = loop
        self.provider_name = provider.provider_name

    def pin(
        self,
        payload: bytes,
        *,
        filename: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            raise RuntimeError(
                "Lighthouse pinning waits on the agent's event loop and must "
                "be called from a worker thread"
            )
        upload = self.provider.publish(
            payload, filename=filename, tag=self._tag, on_progress=on_progress
        )
        return asyncio.run_coroutine_threadsafe(upload, self._loop).result()


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
            encrypt: Whether to encrypt the backup (default: True). Honoured
                on every tier, local included; requires KESTREL_DATA_KEY.
            on_progress: Optional callback(bytes_sent, total_bytes). Not
                LLM-supplied (an LLM can't pass a callable); retained for
                internal callers and the export-progress SSE pipeline,
                which has dedicated test coverage.

        Returns:
            ToolResult.ok with CID + tier + size on a clean export, PARTIAL
            when a non-local tier produced no IPFS CID (backup hashed
            locally but not actually pushed to the network), or
            ToolResult.failed when requested encryption cannot be honoured,
            a paid storage tier cannot be accounted for through a wallet, or
            the agent's privacy mode does not permit a backup when the store
            would begin. Refusals carry a typed ``data["refusal"]`` code.
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

        # Encryption is honoured on EVERY tier, local included. The old code
        # coerced ``encrypt`` to False for LOCAL_ONLY ("no point, and
        # complicates retrieval") and silently wrote a plaintext backup of the
        # whole database (#2872). Both halves of that rationale were wrong:
        # ``FilecoinAdapter.store_content`` encrypts BEFORE it branches on
        # tier, and the export is sealed in a self-describing envelope whose
        # authenticated header names the content hash the key derives from
        # (#2873), so any export decrypts from its payload and
        # KESTREL_DATA_KEY alone, on this host or a fresh one.
        #
        # The one request that cannot be honoured is encryption with no master
        # key configured. Refuse it BEFORE the backup blob, the adapter, and
        # any wallet debit, and carry a typed refusal code so HTTP callers can
        # tell a refusal from a server fault (#2918).
        if encrypt:
            from kestrel_sovereign.security.encryption import get_master_key_bytes

            if not get_master_key_bytes():
                return ToolResult.failed(
                    error=(
                        "encrypt=True requires KESTREL_DATA_KEY, but no master "
                        "encryption key is configured; no sovereignty backup "
                        "was created. Set KESTREL_DATA_KEY, or re-run with "
                        "encrypt=False to accept a plaintext backup."
                    ),
                    data={"refusal": REFUSAL_ENCRYPTION_KEY_UNAVAILABLE},
                )

        # The privacy policy that authorizes this export is read here, before
        # the backup blob, and again immediately before the store, under the
        # durable persistence gate (#2875). Without the gate a transition could
        # land between the two and the export would write under a policy that
        # no longer permits it, so an agent that exposes none is refused.
        try:
            require_durable_persistence_gate(self.agent)
            require_backup_permitted(getattr(self.agent, "storage", None))
        except (
            PrivacyTransitionGuardUnavailableError,
            BackupPolicyUnavailableError,
            PrivacyViolationError,
        ) as e:
            return self._backup_guard_refusal(e)

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
                return ToolResult.failed(
                    error="Insufficient funds for backup.",
                    data={"refusal": REFUSAL_INSUFFICIENT_FUNDS},
                )

        # Create backup blob. A transition that landed since the check above
        # makes the storage wrapper refuse it; that is the same refusal.
        try:
            backup_blob = await self.agent.storage.create_backup_blob(include_db=True)
        except PrivacyViolationError as e:
            return self._backup_refused(str(e))
        progress = self._build_progress_reporter(
            on_progress=on_progress,
            tier=tier_enum.value,
            total_bytes=len(backup_blob),
        )
        progress(0, len(backup_blob))

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

        # Store via adapter. The upload is named for this agent, so the
        # Lighthouse health check and retention attribute it to this agent
        # alone (#2874); IPFS ignores the name, so the CID is unchanged.
        adapter = FilecoinAdapter()
        pinning = self._lighthouse_pinning(tier_enum)
        store = functools.partial(
            adapter.store_content,
            backup_blob,
            storage_tier=tier_enum,
            encrypt=encrypt,
            metadata={"agent": self.agent.agent_id},
            on_progress=progress,
            envelope_owner=self.agent.agent_id if encrypt else None,
            upload_name=sovereignty_export_upload_name(
                self.agent.agent_id, datetime.now(timezone.utc)
            ),
            ipfs_pinning=pinning,
        )
        # ``privacy_transition()`` takes the persistence gate exclusive, so
        # while the guard holds it shared no transition completes: one that
        # landed before it is seen by the guard's check, and one that arrives
        # later waits until the store and its receipts are written (#2875).
        # Concurrent exports share the gate, and it is not the
        # privacy-transition lock a turn holds, so an export waits on neither.
        # The store is held across, not re-checked after: an IPFS upload or
        # Filecoin deal cannot be taken back once it lands, and its receipts
        # are durable writes the same policy governs.
        #
        # Once the store begins, it, its receipts and its fee settle as one
        # owned operation. A cancellation waits for them and is raised after,
        # so the gate is never released under a write in flight, and a backup
        # that landed is never left without its receipts or its charge.
        try:
            async with guarded_backup_store(
                self.agent, getattr(self.agent, "storage", None)
            ):
                stored = await await_owned_task(
                    asyncio.create_task(
                        self._store_and_record(store, backup_blob, audit_anchors)
                    )
                )
        except (
            PrivacyTransitionGuardUnavailableError,
            BackupPolicyUnavailableError,
            PrivacyViolationError,
        ) as e:
            return self._backup_guard_refusal(e)

        # Deduct funds
        if stored.error is None and fee_main > 0 and wallet is not None:
            _, node_id = stored.result
            charged = await await_owned_task(
                asyncio.create_task(
                    self._charge_export_fee(
                        wallet, fee_main, memo=f"backup:{tier_enum.value}:{node_id}"
                    )
                ),
                stored.cancellation,
            )
            raise_owned_outcome(charged, operation="sovereignty export fee")
        result, node_id = raise_owned_outcome(stored, operation="sovereignty export store")
        progress(len(backup_blob), len(backup_blob))

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
        if (
            pinning is not None
            and result.ipfs_cid
            and getattr(result, "provider", None) == pinning.provider_name
        ):
            # Published through the agent's Lighthouse provider: that
            # provider thins the agent's older exports on the same account.
            data["export_retention"] = await self._thin_lighthouse_exports(
                pinning.provider, keep_cid=result.ipfs_cid
            )

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

    def _lighthouse_pinning(self, tier: StorageTier) -> Optional[_LighthousePinning]:
        """The IPFS tier's publisher when the adapter's IPFS node is unreachable.

        The agent's Lighthouse provider exists only when its remote-tier
        policy and PayerPolicy permitted one at boot, and resolves its own
        key, so publishing through it decides neither (#2874). A node the
        adapter reaches still takes the upload: a sovereign-operated node
        outranks a delegated service. Without either, the export stays local
        and reports the downgrade.
        """
        provider = getattr(self.agent, "lighthouse_provider", None)
        if tier != StorageTier.IPFS or provider is None:
            return None
        return _LighthousePinning(
            provider,
            tag=sovereignty_export_tag(self.agent.agent_id),
            loop=asyncio.get_running_loop(),
        )

    async def _thin_lighthouse_exports(self, provider, *, keep_cid: str) -> dict:
        """Thin this agent's older exports on the account ``provider`` just
        published ``keep_cid`` to, by the backup retention policy (#2874).

        The provider that published the export does it, so the account is
        the one the agent's PayerPolicy resolved, whichever key the host
        environment or the snapshot target holds. Only uploads that attribute
        to this agent are considered. The export has landed whatever happens
        here: a failure leaves the remaining exports in place and is
        reported, not raised.
        """
        deleted: list[str] = []
        skipped_missing_id = 0
        try:
            uploads = await provider.list_uploads()
            deletions = sovereignty_export_deletions(
                uploads,
                self.agent.agent_id,
                load_retention_policy(),
                keep_cids={keep_cid},
            )
            for upload in deletions:
                file_id = upload.get("id")
                if not file_id:
                    skipped_missing_id += 1
                    continue
                await provider.delete_upload(str(file_id))
                deleted.append(str(upload.get("cid") or file_id))
        except (httpx.HTTPError, OSError, ValueError) as exc:
            logger.warning(
                "Thinning older sovereignty exports on Lighthouse failed: %s", exc
            )
            return {
                "deleted": len(deleted),
                "cids": deleted,
                "skipped_missing_id": skipped_missing_id,
                "error": str(exc),
            }
        return {
            "deleted": len(deleted),
            "cids": deleted,
            "skipped_missing_id": skipped_missing_id,
        }

    async def _store_and_record(self, store, backup_blob: bytes, audit_anchors):
        """Store the backup, then record it. Returns ``(result, node_id)``."""
        result = await asyncio.to_thread(store)
        node_id = await self._record_export(result, backup_blob, audit_anchors)
        return result, node_id

    @staticmethod
    async def _charge_export_fee(wallet, fee: Decimal, *, memo: str) -> None:
        await wallet.transfer(fee, memo=memo)

    async def _record_export(self, result, backup_blob: bytes, audit_anchors) -> str:
        """Record a stored export: its ``backup_artifact`` and its receipt."""
        node_id = await self.agent.storage.record_backup_artifact(self.agent.agent_id, result)

        receipt_properties = {
            "cid": result.ipfs_cid,
            "ipfs_cid": result.ipfs_cid,
            "content_hash": result.content_hash,
            "storage_tier": result.storage_tier.value,
            "provider": getattr(result, "provider", None),
            "encrypted": result.encrypted,
            "encryption_key_hash": result.encryption_key_hash,
            # The host-cache name, as the ``backup_artifact`` records it. A
            # receipt that records it owns that cache entry only, never the
            # bare content hash, which another agent's export of the same
            # plaintext may be cached under (#2873).
            "cache_name": getattr(result, "cache_name", None),
            "size_bytes": getattr(result, "size_bytes", 0) or len(backup_blob),
            # Graph created_at is a UTC ISO-8601 contract (#3256).
            "created_at": utc_now_iso(),
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
        return node_id

    @classmethod
    def _backup_guard_refusal(cls, error: Exception) -> ToolResult:
        """The refusal for an export its privacy guard would not admit (#2875).

        A privacy mode that forbids backups is a typed refusal; an agent with
        no persistence gate or no backup policy is a host fault, untyped.
        """
        if isinstance(error, PrivacyViolationError):
            return cls._backup_refused(str(error))
        return ToolResult.failed(
            error=(
                f"Sovereignty export refused: {error}; no sovereignty backup "
                "was created."
            )
        )

    @staticmethod
    def _backup_refused(reason: str) -> ToolResult:
        logger.warning(f"Refused sovereignty export: {reason}")
        return ToolResult.failed(
            error=f"Sovereignty export refused: {reason}. No sovereignty backup was stored.",
            data={"refusal": REFUSAL_PRIVACY_MODE_FORBIDS_BACKUP},
        )

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

        An encrypted export carries its decryption metadata in a
        self-describing envelope and decrypts with the configured
        KESTREL_DATA_KEY alone, so it restores on a fresh host whose storage
        holds no record of it (#2873). The envelope is sealed for the agent
        that exported it, and an export of another agent is refused: its
        conversation rows are encrypted at rest under that agent's identity.
        Without a backup receipt only IPFS is read, never the host cache
        co-hosted agents share (#3225), and only an encrypted export is
        restored. A legacy encrypted export, made before
        the envelope, decrypts with the key hash in this agent's backup
        receipt; without one the configured key is tried under the name it
        was requested by, which opens it only when that name is its content
        hash. Without a receipt that name must also be one IPFS answers for:
        an export known only by its content hash (a local-tier export) is in
        a host cache, which that path does not read. With a receipt, the
        host cache is read under the name the receipt records (its content
        hash, for a receipt that records none), then under the name it was
        requested by. An envelope's recorded name derives from the key it
        was sealed under, so an export made under another key, or read with
        none configured, is still found and refused naming the key rather
        than reported missing. A copy fetched from IPFS is cached again
        under its content hash, with its owner for an envelope, as the
        export was, never under the CID. The receipt
        also decides whether the export is
        encrypted: only one recording a plaintext export accepts a payload
        that is not ciphertext. The IPFS read and the decompression stop at a
        ceiling derived from ``MAX_SOVEREIGNTY_IMPORT_BYTES`` rather than
        after it, and a larger export is refused. An export that cannot be
        decrypted is refused, and none of its bytes reach the restore.

        Scope (#F265): restores ``conversation_history`` faithfully —
        ``created_at`` (ordering) and ``deleted_at`` (trash) are preserved, so
        the restore neither reorders history nor un-deletes trashed messages.
        Other agent state (memories, graph_nodes, saved_items, files, settings)
        is NOT restored yet; full multi-table restore is a tracked follow-up.
        """
        receipt = None
        try:
            adapter = FilecoinAdapter()

            # The receipt is the gate to the host cache (#3225).
            # ``retrieve_content`` reads ``storage_cache/{name}.cache`` by
            # whatever name it is given, and the cache is one directory per
            # host: without this, any co-hosted agent could name another
            # agent's 64-hex content hash (which passes the CID pattern) and
            # restore that agent's conversation history into its own
            # database, or probe which hashes exist by the error text.
            backup_nodes = await self.agent.storage.get_nodes_by_type("backup_artifact")
            receipt = next(
                (
                    node for node in backup_nodes
                    if node.properties.get("ipfs_cid") == cid or node.node_id == cid
                ),
                None,
            )
            if receipt is None:
                # A fresh host, or a database lost with its records (#2873).
                # The host cache still may not answer for a name this agent
                # holds no receipt for, so only IPFS can, and only an export
                # the configured key decrypts is restored. That key is shared
                # by every agent on the host, so an envelope must also have
                # been sealed for this agent's DID, which stands in for the
                # ownership a receipt would otherwise show. An envelope names
                # its own key hash; a legacy export is tried under the name
                # it was requested by. A plaintext payload proves nothing.
                logger.info(
                    f"No backup receipt for CID {cid} in this agent's storage; "
                    "restoring an encrypted export from IPFS only"
                )
                content = await asyncio.to_thread(
                    adapter.retrieve_content,
                    cid,
                    ipfs_cid=cid,
                    max_output_bytes=MAX_SOVEREIGNTY_IMPORT_BYTES,
                    allow_local_cache=False,
                    envelope=EnvelopePolicy.ENCRYPTED_ONLY,
                    envelope_owner=self.agent.agent_id,
                )
            else:
                # The envelope names its own key hash, and the receipt's must
                # equal it; a legacy export without an envelope decrypts with
                # the receipt's, or under its name if the receipt has none.
                # A receipt that lost its key hash still names its export:
                # a ``backup_artifact`` node's id is the content hash, so an
                # envelope of another export is refused under it too.
                key_hash = receipt.properties.get("encryption_key_hash")
                # The receipt, not the payload, says whether the export is
                # encrypted. An envelope whose magic bytes were damaged no
                # longer looks like ciphertext, and a CID lookup skips the
                # content-hash check, so trusting the payload would hand the
                # damaged bytes to the restore as plaintext. Only a receipt
                # that records a plaintext export, and no key hash, accepts
                # one.
                receipt_encrypted = (
                    receipt.properties.get("encrypted") is not False or bool(key_hash)
                )
                # Where the host cache holds the export, which the export
                # determines and the CID it is requested by does not. An
                # envelope's name derives from the master key it was sealed
                # under, so a reader under another key, or none, cannot
                # derive it: the receipt's record of it is read instead, and
                # the envelope then names the key it needs. A receipt written
                # before the name was recorded holds none; ``store_content``
                # cached every export then under its content hash, which is
                # the receipt's node id. ``retrieve_content`` also reads the
                # requested name, under which a restore by CID before #2873
                # cached what IPFS returned.
                cache_name = receipt.properties.get("cache_name")
                if cache_name is None:
                    if is_cache_name_of(receipt.node_id, receipt.node_id):
                        cache_name = receipt.node_id
                elif not is_cache_name_of(cache_name, receipt.node_id):
                    return self._import_refused(
                        cid,
                        REFUSAL_BACKUP_RECEIPT_INVALID,
                        "this agent's backup receipt records a cache name "
                        "that is not one of the export it records",
                    )
                logger.info(f"Found backup artifact node for CID {cid}, encrypted: {receipt.properties.get('encrypted')}")
                content = await asyncio.to_thread(
                    adapter.retrieve_content,
                    cid,
                    ipfs_cid=cid,
                    key_hash=key_hash,
                    max_output_bytes=MAX_SOVEREIGNTY_IMPORT_BYTES,
                    envelope=(
                        EnvelopePolicy.ENCRYPTED_ONLY
                        if receipt_encrypted
                        else EnvelopePolicy.OPEN
                    ),
                    envelope_owner=self.agent.agent_id,
                    envelope_key_hash=key_hash or receipt.node_id,
                    cache_name=cache_name,
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
        except ContentRetrievalLimitError:
            return self._import_refused(
                cid,
                REFUSAL_EXPORT_EXCEEDS_IMPORT_LIMIT,
                f"it exceeds the {MAX_SOVEREIGNTY_IMPORT_BYTES}-byte limit on "
                "what an import retrieves and decodes",
                limit_bytes=MAX_SOVEREIGNTY_IMPORT_BYTES,
            )
        except DecryptionKeyUnavailableError:
            return self._import_refused(
                cid,
                REFUSAL_ENCRYPTION_KEY_UNAVAILABLE,
                "it is an encrypted export and no KESTREL_DATA_KEY is "
                "configured to decrypt it",
            )
        except EnvelopeKeyMismatchError as e:
            return self._import_refused(
                cid,
                REFUSAL_ENCRYPTION_KEY_MISMATCH,
                f"{e}; configure the KESTREL_DATA_KEY the export was made with",
                export_key_id=e.envelope_key_id,
                configured_key_id=e.configured_key_id,
            )
        except EnvelopeOwnerMismatchError as e:
            return self._import_refused(
                cid,
                REFUSAL_EXPORT_OF_ANOTHER_AGENT,
                f"{e}. An export restores only into the agent that made it: "
                "its conversation history is encrypted at rest under that "
                "agent's identity",
                export_owner_id=e.envelope_owner_id,
                agent_owner_id=e.owner_id,
            )
        except EncryptedEnvelopeError as e:
            return self._import_refused(
                cid,
                REFUSAL_DECRYPTION_FAILED,
                f"it could not be decrypted: {e}",
            )
        except LegacyDecryptionUnavailableError as e:
            return self._import_refused(
                cid,
                REFUSAL_LEGACY_DECRYPTION_METADATA_UNAVAILABLE,
                "it is a legacy encrypted export and its decryption metadata "
                "is unavailable: it predates the self-describing envelope, "
                f"this agent's storage records no key hash for it, and {e}",
                decryption_attempted=e.attempted,
            )
        except UnencryptedContentError:
            if receipt is not None:
                return self._import_refused(
                    cid,
                    REFUSAL_DECRYPTION_FAILED,
                    "this agent's backup receipt records an encrypted export, "
                    "but the retrieved payload is not ciphertext this host "
                    "can decrypt; it may be damaged or substituted",
                )
            return self._import_refused(
                cid,
                REFUSAL_UNENCRYPTED_WITHOUT_RECEIPT,
                "it is an unencrypted export and this agent holds no backup "
                "receipt for it; without a receipt only an export encrypted "
                "under the configured KESTREL_DATA_KEY can be restored",
            )
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return ToolResult.failed(error=f"❌ Error during import: {str(e)}")

    @staticmethod
    def _import_refused(cid: str, refusal: str, reason: str, **details) -> ToolResult:
        logger.warning(f"Refused sovereignty import of {cid}: {reason}")
        return ToolResult.failed(
            error=f"❌ Import refused for CID {cid}: {reason}. Nothing was restored.",
            data={"refusal": refusal, "cid": cid, **details},
        )

    @tool(
        name="check_sovereignty_status",
        description="Check the status of sovereignty backups.",
        category=ToolCategory.SYSTEM,
        command_prefix="!check-sovereignty-status"
    )
    async def check_sovereignty_status(self) -> ToolResult:
        """Check sovereignty status: the exports this host recorded, and this
        agent's latest export on the Lighthouse account it publishes to."""
        try:
            receipts = await self.agent.storage.get_nodes_by_type("sovereignty_receipt")
            lighthouse = await self._lighthouse_export_health()
            lighthouse_line = self._lighthouse_status_line(lighthouse)
            if not receipts:
                return ToolResult.ok(
                    "No sovereignty exports found on this host.\n" + lighthouse_line,
                    data={
                        "latest_cid": None,
                        "latest_created_at": None,
                        "total_exports": 0,
                        "lighthouse": lighthouse,
                    },
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
                f"{lighthouse_line}"
            )
            return ToolResult.ok(
                confirmation,
                data={
                    "latest_cid": latest_cid,
                    "latest_created_at": latest_created_at,
                    "total_exports": len(receipts),
                    "lighthouse": lighthouse,
                },
            )
        except Exception as e:
            return ToolResult.failed(error=f"Error checking status: {e}")

    async def _lighthouse_export_health(self) -> dict:
        """This agent's latest sovereignty export on its Lighthouse account.

        Read with the agent's Lighthouse provider's key, its PayerPolicy
        resolution: the account its exports were published to, which neither
        the host environment's key nor the snapshot target's need name
        (#2874). The export is attributed to this agent by its upload record.
        """
        provider = getattr(self.agent, "lighthouse_provider", None)
        api_key = await provider.resolve_api_key() if provider is not None else None
        if not api_key:
            return {
                "status": "not_configured",
                "message": (
                    "This agent has no Lighthouse credential, so it publishes "
                    "no sovereignty export to Lighthouse."
                ),
            }
        health = await check_lighthouse_health(
            api_key=api_key, agent_id=self.agent.agent_id
        )
        if health.status == "error":
            return {"status": "error", "message": health.message}
        export = (health.details or {}).get("sovereignty_export")
        if export is None:
            return {
                "status": "not_found",
                "message": (
                    "No sovereignty export of this agent on its Lighthouse account."
                ),
            }
        return {
            "status": export["status"],
            "message": f"Latest Lighthouse sovereignty export: {export['status']}",
            "agent_id": export["agent_id"],
            "cid": export["cid"],
            "file_name": export["file_name"],
            "uploaded_at": export["uploaded_at"],
            "age_seconds": export["age_seconds"],
            "deal_count": export["deal_count"],
        }

    @staticmethod
    def _lighthouse_status_line(lighthouse: dict) -> str:
        if "cid" not in lighthouse:
            return f"Lighthouse: {lighthouse['status']}: {lighthouse['message']}\n"
        return (
            f"Lighthouse: {lighthouse['status']}: {lighthouse['cid']} "
            f"(deals: {lighthouse['deal_count']})\n"
        )
