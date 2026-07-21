#!/usr/bin/env python3
"""
Identity Importer: Import and restore agent identity from a package.

This module provides the IdentityImporter class which takes a portable
AgentIdentityPackage and restores the agent's identity, memories, and
relationships to a new substrate.

Usage:
    importer = IdentityImporter(db, target_agent_id)
    result = await importer.import_package(package)
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING

from kestrel_sovereign.storage.async_graph_store import (
    record_graph_edge_owner,
    record_graph_node_owner,
    release_graph_node_owners,
)

from .access_grant import (
    DataAccessGrant,
    REJECT_HOST_POLICY,
    verify_import_consent,
)
from .identity_package import (
    AgentIdentityPackage,
    SubstrateType,
    create_migration_id,
)
from .graph_namespace import namespace_imported_record

if TYPE_CHECKING:
    from kestrel_sovereign.identity.portable_trust import IdentityTrustPolicy
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Result of an identity import operation."""
    success: bool
    migration_id: str
    agent_id: str
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "migration_id": self.migration_id,
            "agent_id": self.agent_id,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


class IdentityImporter:
    """
    Import agent identity from a portable package.

    Restores:
    - Core identity verification
    - Memory episodes
    - Saved items
    - Temporal patterns
    - Reflection insights
    - Relationships (as graph nodes/edges)
    - Skills
    - Wallet state
    - Migration record

    The importer verifies package integrity before importing and creates
    a migration record for audit trail.
    """

    # F186: node types the importer must NEVER overwrite. Package-supplied
    # graph nodes (users, skills) are namespaced under the importing agent's
    # id, but a defense-in-depth guard additionally refuses to clobber the
    # agent's own identity node or any existing reserved-type node, so a
    # crafted id can't hijack governance/identity/lineage rows.
    RESERVED_NODE_TYPES = frozenset({
        "agent",
        "constitution",
        "migration_record",
        "lifecycle_event",
        "retirement_event",
        "audit_anchor",
        "sovereignty_receipt",
        "agent_identity_resource",
    })

    # Exact agent-scoped row inventory owned by replace mode. Graph-backed
    # relationships and skills are cleared separately so shared nodes survive.
    REPLACE_ROW_TABLES = (
        "memory_episodes",
        "saved_items",
        "temporal_patterns",
        "reflection_insights",
        "wallet_transactions",
        "wallet_state",
    )
    MERGE_MODES = frozenset({"replace", "merge", "skip_existing"})

    def __init__(
        self,
        db: "AsyncDatabase",
        target_agent_id: Optional[str] = None,
        target_substrate: Optional[str] = None,
        storage_dir: Optional[Path] = None,
    ):
        """
        Initialize the importer.

        Args:
            db: Database connection
            target_agent_id: The agent ID to import into. If None, uses DID from package.
            target_substrate: The substrate being imported to
            storage_dir: Runtime agent-data directory containing the trusted
                identity keys used to verify package signatures.
        """
        self.db = db
        self.target_agent_id = target_agent_id
        self.target_substrate = target_substrate or SubstrateType.UNKNOWN.value
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {}

    async def import_serialized(
        self,
        serialized: str,
        *,
        kem_keypair: Optional[Any] = None,
        kem_slug: Optional[str] = None,
        kem_storage_dir: Optional[Path] = None,
        **import_kwargs: Any,
    ) -> ImportResult:
        """Import from a serialized export — sealed capsule or plaintext JSON.

        Counterpart to ``IdentityExporter.export_sealed`` (#2398).
        Detects a ``kestrel-sealed-capsule-v1`` envelope by its
        ``format`` field and unseals it with the local hybrid KEM
        private keys before the existing :meth:`import_package` path
        runs. Legacy plaintext-JSON exports are parsed exactly as
        before.

        Fail-loud contract: a sealed capsule with no local KEM keys
        available, a capsule sealed for a different recipient, or a
        tampered capsule raises
        :class:`~kestrel_sovereign.identity.sealed_export.SealedExportError`
        — nothing is imported and there is no plaintext fallback.

        Args:
            serialized: the export string (sealed capsule or package JSON).
            kem_keypair: this agent's ``HybridKEMKeypair``, if already
                loaded. Takes precedence over ``kem_slug``.
            kem_slug: key-file slug used to load the local KEM keypair
                from encrypted storage (``<slug>_x25519`` +
                ``<slug>_mlkem768``). Required for sealed input when
                ``kem_keypair`` is not passed.
            kem_storage_dir: storage dir for the key files (defaults to
                the agent data dir).
            **import_kwargs: forwarded to :meth:`import_package`
                (verify_signature, merge_mode, grant, ...).
        """
        from .sealed_export import open_identity_export

        package = open_identity_export(
            serialized,
            kem_keypair=kem_keypair,
            slug=kem_slug,
            storage_dir=kem_storage_dir,
        )
        return await self.import_package(package, **import_kwargs)

    async def import_package(
        self,
        package: AgentIdentityPackage,
        verify_signature: bool = True,
        verify_constitution: bool = True,
        merge_mode: str = "replace",  # replace, merge, skip_existing
        allow_unsigned: bool = False,
        grant: Optional[DataAccessGrant] = None,
        host_policy: Optional[
            Callable[[DataAccessGrant, str], bool]
        ] = None,
        grant_did_web_resolver: Optional[Callable[[str], Any]] = None,
        revoked_grant_ids: Optional[Iterable[str]] = None,
        identity_trust_policy: Optional["IdentityTrustPolicy"] = None,
    ) -> ImportResult:
        """
        Import agent identity from a package.

        Args:
            package: The identity package to import
            verify_signature: Whether to verify DID signature
            verify_constitution: Whether to verify constitution integrity
            merge_mode: How to handle existing data:
                - "replace": Clear existing data and import all
                - "merge": Add new data, keep existing
                - "skip_existing": Only import if no existing data
            allow_unsigned: If True, allow importing unsigned packages.
                Defaults to False for security. Set to True only for
                development/testing use cases.
            grant: Optional owner-signed :class:`DataAccessGrant`
                authorizing this import. When provided, consent is
                verified BEFORE signature verification — a valid
                package signature without owner consent is still
                unauthorized. Default ``None`` preserves the pre-#1273
                behavior (no consent gate).
            host_policy: Optional host-side filter callable evaluated
                AFTER consent verification returns ``ok=True``. The
                callable receives ``(grant, canonical_grant_id)`` —
                policies that allowlist or audit by id MUST use
                ``canonical_grant_id`` (the verifier-recomputed,
                trustworthy value), not the ``grant.grant_id`` field
                on the dataclass (which is unsigned and spoofable).
                If the callable returns ``False`` the import is
                refused with a distinct ``host_policy_rejected``
                reason. Never a substitute for the grant — runs only
                when the grant already verifies. Ignored when
                ``grant`` is ``None``.
            grant_did_web_resolver: Optional ``did:web:`` resolver
                forwarded to :func:`verify_import_consent` so hybrid
                owners on ``did:web:`` DIDs can have their grant
                signatures verified. The binding helper refuses-by-
                default for ``did:web:`` without a resolver; pass one
                here when accepting hybrid-owner grants. ``did:pkh:``
                / ``did:key:`` owners need no resolver. Ignored when
                ``grant`` is ``None``.
            revoked_grant_ids: Optional iterable of canonical grant
                ids (as returned by
                :func:`access_grant.compute_grant_id`) that are
                currently revoked. The recomputed canonical id of
                ``grant`` is compared against this set; matches are
                rejected with ``grant_expired_or_revoked``. Sourced
                from a trusted revocation registry — never from the
                grant payload itself (an in-grant flag would be
                unsigned and spoofable). Ignored when ``grant`` is
                ``None``.
            identity_trust_policy: Receiver-owned root-key pins, succession
                revocations, and optional archival requirements for portable
                verification. Self-certifying did:pkh/did:key roots need no
                explicit pin; did:web roots always require pinned methods.

        Returns:
            ImportResult with success status and statistics
        """
        self.errors = []
        self.warnings = []
        self.stats = {}

        # Use package DID if no target specified
        agent_id = self.target_agent_id or package.did

        logger.info(f"Importing identity package for {agent_id[:20]}...")
        logger.info(f"Package version: {package.package_version}")
        logger.info(f"Source substrate: {package.source_substrate}")

        # 1. Verify package integrity
        if verify_constitution and package.constitution_text:
            if not package.verify_constitution():
                self.errors.append("Constitution hash verification failed")
                return self._build_result(False, agent_id)

        if package.content_hash:
            if not package.verify_content_hash():
                self.errors.append("Content hash verification failed")
                return self._build_result(False, agent_id)

        # Consent gate (#1273) — when a grant is provided, owner
        # authorization is checked BEFORE signature verification. A
        # valid signature without owner consent is still unauthorized.
        # An optional ``host_policy`` callable runs only AFTER consent
        # verifies; it's a filter on top of a valid grant, never a
        # substitute for one.
        if grant is not None:
            if not self.target_agent_id:
                self.errors.append(
                    "consent grant requires target_agent_id (host_did) "
                    "to be set on the IdentityImporter"
                )
                return self._build_result(False, agent_id)
            consent = await verify_import_consent(
                package, grant, host_did=self.target_agent_id,
                revoked_grant_ids=revoked_grant_ids,
                did_web_resolver=grant_did_web_resolver,
            )
            if not consent.ok:
                self.errors.append(
                    f"consent verification failed: {consent.reason}"
                )
                return self._build_result(False, agent_id)
            if host_policy is not None and not host_policy(
                grant, consent.canonical_grant_id
            ):
                self.errors.append(
                    f"{REJECT_HOST_POLICY}: host policy rejected an "
                    f"otherwise-valid grant "
                    f"{consent.canonical_grant_id[:16]}"
                )
                return self._build_result(False, agent_id)

        # Hybrid packages carry sigs only on package.signatures
        # (the v2 array); the legacy package.signature field is
        # empty by design for post-ceremony agents because that
        # field can't be made byte-compatible with v1 readers
        # over a v2 canonical hash. Treat either carrier as
        # "signed" and let _verify_signature route by alg.
        has_signature = bool(package.signature) or bool(package.signatures)

        # F185: the unsigned-package policy is enforced UNCONDITIONALLY,
        # independent of verify_signature. ``verify_signature`` gates only
        # the cryptographic validation of a *present* signature — it must
        # never relax the requirement that a signature be present. Hoisting
        # this out of the ``if verify_signature:`` block closes the bypass
        # where a model-controlled ``verify_signature=False`` silently
        # imported an unsigned package despite ``allow_unsigned=False``.
        if not has_signature and not allow_unsigned:
            self.errors.append(
                "Package is not signed (unsigned). "
                "Set allow_unsigned=True to import unsigned packages."
            )
            return self._build_result(False, agent_id)

        if has_signature:
            if verify_signature:
                if identity_trust_policy is None:
                    # Preserve the historical bound-method call shape for
                    # integrations that override or instrument this hook.
                    sig_valid = await self._verify_signature(package)
                else:
                    sig_valid = await self._verify_signature(
                        package, identity_trust_policy
                    )
                if not sig_valid:
                    self.errors.append("DID signature verification failed")
                    return self._build_result(False, agent_id)
        else:
            # Reached only when allow_unsigned=True (else we returned above).
            self.warnings.append("Importing unsigned package (allow_unsigned=True)")

        # 2. Check merge mode and package shape before any mutation.
        if merge_mode not in self.MERGE_MODES:
            self.errors.append(
                "Invalid merge mode; expected replace, merge, or skip_existing"
            )
            return self._build_result(False, agent_id)

        if merge_mode == "skip_existing":
            existing = await self._check_existing_data(agent_id)
            if existing:
                self.warnings.append("Skipping import - existing data found")
                return self._build_result(True, agent_id)

        try:
            self._validate_package_components(package)
        except ValueError as e:
            self.errors.append(f"Identity package validation failed: {e}")
            return self._build_result(False, agent_id)

        # 3. Begin import
        migration_id = create_migration_id()

        component = "transaction setup"
        try:
            # Both backends bind every write in this block to one transaction.
            # Helpers must not commit or suppress required-row failures.
            async with self.db.transaction():
                if merge_mode == "replace":
                    component = "replace cleanup"
                    await self._clear_existing_data(agent_id)

                component = "memory episodes"
                await self._import_episodes(agent_id, package.episodes)
                component = "saved items"
                await self._import_saved_items(agent_id, package.saved_items)
                component = "temporal patterns"
                await self._import_temporal_patterns(
                    agent_id, package.temporal_patterns
                )
                component = "reflection insights"
                await self._import_reflection_insights(
                    agent_id, package.reflection_insights
                )
                component = "relationships"
                await self._import_relationships(agent_id, package.relationships)
                component = "skills"
                await self._import_skills(agent_id, package.skills)
                component = "wallet"
                await self._import_wallet_state(agent_id, package)
                component = "migration evidence"
                await self._record_migration(agent_id, package, migration_id)

            logger.info(f"Import complete: {self.stats}")
            return self._build_result(True, agent_id, migration_id)

        except Exception as e:
            # Attempted counts describe rolled-back writes, not evidence.
            self.stats = {}
            self.errors.append(
                f"Identity import failed during {component}; "
                "all database changes were rolled back"
            )
            logger.error(
                "Identity import failed during %s: %s",
                component,
                e,
                exc_info=True,
            )
            return self._build_result(False, agent_id)

    def _build_result(self, success: bool, agent_id: str, migration_id: str = "") -> ImportResult:
        """Build an ImportResult."""
        return ImportResult(
            success=success,
            migration_id=migration_id,
            agent_id=agent_id,
            errors=self.errors.copy(),
            warnings=self.warnings.copy(),
            stats=self.stats.copy(),
        )

    async def _verify_signature(
        self,
        package: AgentIdentityPackage,
        trust_policy: Optional["IdentityTrustPolicy"] = None,
    ) -> bool:
        """Verify the package's signature via the canonical verifier.

        Routes through ``verify_package_signature`` so both legacy
        ``signature`` (single ECDSA hex) and the v2 ``signatures``
        array (hybrid Ed25519 + ML-DSA-65) are handled by their
        appropriate paths. The verifier loads the trust anchor from
        the receiver's local agent_data dir.
        """
        try:
            from kestrel_sovereign.identity.signing import verify_package_signature
            if trust_policy is None:
                ok, msg = verify_package_signature(package, self.storage_dir)
            else:
                ok, msg = verify_package_signature(
                    package, self.storage_dir, trust_policy=trust_policy
                )
            if not ok:
                self.warnings.append(f"Signature verification failed: {msg}")
                return False
            self.stats["signature_verification"] = msg
            return True
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            self.warnings.append(f"Signature verification failed: {str(e)}")
            return False

    async def _check_existing_data(self, agent_id: str) -> bool:
        """Check the importer-owned inventory for existing agent state."""
        for table in self.REPLACE_ROW_TABLES:
            row = await self.db.fetchone(
                f"SELECT COUNT(*) FROM {table} WHERE agent_id = ?",
                (agent_id,),
            )
            if row and row[0] > 0:
                return True

        row = await self.db.fetchone(
            """SELECT COUNT(*)
               FROM graph_edges ge
               JOIN graph_nodes gn ON gn.node_id = ge.target_id
               JOIN graph_edge_owners geo
                 ON geo.source_id = ge.source_id
                AND geo.target_id = ge.target_id
                AND geo.label = ge.label
                AND geo.agent_id = ?
               JOIN graph_node_owners gno
                 ON gno.node_id = gn.node_id
                AND gno.agent_id = ?
               WHERE ge.source_id = ?
                 AND (gn.node_type = 'user'
                      OR (gn.node_type = 'skill' AND ge.label = 'has_skill'))""",
            (agent_id, agent_id, agent_id),
        )
        return bool(row and row[0] > 0)

    @staticmethod
    def _record_dict(record: Any, component: str, index: int) -> Dict[str, Any]:
        if hasattr(record, "to_dict"):
            record = record.to_dict()
        if not isinstance(record, dict):
            raise ValueError(f"{component}[{index}] must be an object")
        return record

    def _validate_records(
        self,
        component: str,
        records: List[Any],
        required_fields: tuple[str, ...],
        unique_fields: tuple[str, ...],
    ) -> None:
        seen: set[tuple[Any, ...]] = set()
        for index, raw_record in enumerate(records):
            record = self._record_dict(raw_record, component, index)
            for field in required_fields:
                value = record.get(field)
                empty_required_string = (
                    isinstance(value, str)
                    and not value.strip()
                    and not (component == "saved_items" and field == "content")
                )
                if value is None or empty_required_string:
                    raise ValueError(
                        f"{component}[{index}] is missing required field {field!r}"
                    )
            if unique_fields:
                identity = tuple(record.get(field) for field in unique_fields)
                if identity in seen:
                    fields = ", ".join(unique_fields)
                    raise ValueError(
                        f"{component}[{index}] duplicates an earlier {fields} value"
                    )
                seen.add(identity)

    def _validate_package_components(self, package: AgentIdentityPackage) -> None:
        """Reject malformed required records before destructive work begins."""
        specifications = (
            ("episodes", package.episodes, ("id", "title"), ("id",)),
            (
                "saved_items",
                package.saved_items,
                ("id", "item_type", "name", "content"),
                ("id",),
            ),
            (
                "temporal_patterns",
                package.temporal_patterns,
                ("id", "pattern_type", "description"),
                ("id",),
            ),
            (
                "reflection_insights",
                package.reflection_insights,
                ("id", "insight_type", "title"),
                ("id",),
            ),
            (
                "relationships",
                package.relationships,
                ("user_id",),
                ("user_id", "relationship_type"),
            ),
            (
                "skills",
                package.skills,
                ("skill_id", "skill_name"),
                ("skill_id",),
            ),
            (
                "wallet_transaction_history",
                package.wallet_transaction_history,
                ("amount",),
                (),
            ),
        )
        for component, records, required, unique in specifications:
            if not isinstance(records, list):
                raise ValueError(f"{component} must be a list")
            self._validate_records(component, records, required, unique)

    def _timestamp_param(self, value: Any, field: str) -> Any:
        """Validate a package timestamp and shape it for the active backend."""
        if value in (None, ""):
            return None
        original = value
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValueError(
                    f"{field} is not a valid ISO-8601 timestamp"
                ) from e
        else:
            raise ValueError(f"{field} is not a valid timestamp")

        if self.db.backend_type == "postgres":
            return parsed
        return original if isinstance(original, str) else parsed.isoformat()

    async def _clear_existing_data(self, agent_id: str):
        """Clear the exact importer-owned inventory for replace mode."""
        for table in self.REPLACE_ROW_TABLES:
            await self.db.execute(
                f"DELETE FROM {table} WHERE agent_id = ?",
                (agent_id,),
            )

        await self._clear_graph_component(agent_id, "user")
        await self._clear_graph_component(agent_id, "skill", label="has_skill")

    async def _clear_graph_component(
        self,
        agent_id: str,
        node_type: str,
        *,
        label: Optional[str] = None,
    ) -> None:
        """Remove this agent's component edges and reclaim orphaned nodes."""
        query = (
            "SELECT DISTINCT gn.node_id FROM graph_nodes gn "
            "JOIN graph_edges ge ON gn.node_id = ge.target_id "
            "JOIN graph_node_owners gno "
            "  ON gno.node_id = gn.node_id AND gno.agent_id = ? "
            "JOIN graph_edge_owners geo "
            "  ON geo.source_id = ge.source_id "
            " AND geo.target_id = ge.target_id "
            " AND geo.label = ge.label AND geo.agent_id = ? "
            "WHERE ge.source_id = ? AND gn.node_type = ?"
        )
        query_params: tuple[Any, ...] = (
            agent_id,
            agent_id,
            agent_id,
            node_type,
        )
        if label is not None:
            query += " AND ge.label = ?"
            query_params += (label,)
        rows = await self.db.fetchall(query, query_params)

        for row in rows:
            node_id = row[0]
            delete_condition = "source_id = ? AND target_id = ?"
            delete_params: tuple[Any, ...] = (agent_id, node_id)
            if label is not None:
                delete_condition += " AND label = ?"
                delete_params += (label,)
            await self.db.execute(
                f"DELETE FROM graph_edge_owners WHERE {delete_condition} "
                "AND agent_id = ?",
                delete_params + (agent_id,),
            )
            await self.db.execute(
                f"DELETE FROM graph_edges WHERE {delete_condition} "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM graph_edge_owners AS remaining_owner "
                "  WHERE remaining_owner.source_id = graph_edges.source_id "
                "  AND remaining_owner.target_id = graph_edges.target_id "
                "  AND remaining_owner.label = graph_edges.label"
                ")",
                delete_params,
            )

            remaining_owned = await self.db.fetchone(
                "SELECT COUNT(*) FROM graph_edge_owners "
                "WHERE agent_id = ? AND (source_id = ? OR target_id = ?)",
                (agent_id, node_id, node_id),
            )
            if not remaining_owned or remaining_owned[0] == 0:
                await release_graph_node_owners(
                    self.db, [node_id], agent_id
                )

    async def _import_episodes(self, agent_id: str, episodes: List[Dict[str, Any]]):
        """Import memory episodes."""
        count = 0
        for index, ep in enumerate(episodes):
            new_id = namespace_imported_record(agent_id, ep["id"])
            await self.db.execute(
                """INSERT OR REPLACE INTO memory_episodes
                   (id, agent_id, title, summary, timespan_start, timespan_end,
                    key_message_ids, emotional_arc, created_at, importance,
                    access_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_id,
                    agent_id,
                    ep["title"],
                    ep.get("summary"),
                    self._timestamp_param(
                        ep.get("timespan_start"),
                        f"episodes[{index}].timespan_start",
                    ),
                    self._timestamp_param(
                        ep.get("timespan_end"),
                        f"episodes[{index}].timespan_end",
                    ),
                    json.dumps(ep.get("key_message_ids", [])),
                    ep.get("emotional_arc"),
                    self._timestamp_param(
                        ep.get("created_at"),
                        f"episodes[{index}].created_at",
                    ),
                    ep.get("importance", 0.5),
                    ep.get("access_count", 0),
                )
            )
            count += 1

        self.stats["episodes_imported"] = count
        logger.info(f"Imported {count} memory episodes")

    async def _import_saved_items(self, agent_id: str, items: List[Dict[str, Any]]):
        """Import saved items."""
        count = 0
        for index, item in enumerate(items):
            new_id = namespace_imported_record(agent_id, item["id"])
            await self.db.execute(
                """INSERT OR REPLACE INTO saved_items
                   (id, agent_id, item_type, name, summary, content, content_hash,
                    ipfs_cid, source_type, source_ref, schema_id, tags, metadata,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_id,
                    agent_id,
                    item["item_type"],
                    item["name"],
                    item.get("summary"),
                    item["content"],
                    item.get("content_hash"),
                    item.get("ipfs_cid"),
                    item.get("source_type"),
                    item.get("source_ref"),
                    item.get("schema_id"),
                    json.dumps(item.get("tags", [])),
                    json.dumps(item.get("metadata", {})),
                    self._timestamp_param(
                        item.get("created_at"),
                        f"saved_items[{index}].created_at",
                    ),
                    self._timestamp_param(
                        item.get("updated_at"),
                        f"saved_items[{index}].updated_at",
                    ),
                )
            )
            count += 1

        self.stats["saved_items_imported"] = count
        logger.info(f"Imported {count} saved items")

    async def _import_temporal_patterns(self, agent_id: str, patterns: List[Dict[str, Any]]):
        """Import temporal patterns."""
        count = 0
        for index, pattern in enumerate(patterns):
            new_id = namespace_imported_record(agent_id, pattern["id"])
            await self.db.execute(
                """INSERT OR REPLACE INTO temporal_patterns
                   (id, agent_id, pattern_type, description, trigger_conditions,
                    confidence, observations, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_id,
                    agent_id,
                    pattern["pattern_type"],
                    pattern["description"],
                    json.dumps(pattern.get("trigger_conditions", {})),
                    pattern.get("confidence", 0.0),
                    pattern.get("observations", 0),
                    self._timestamp_param(
                        pattern.get("created_at"),
                        f"temporal_patterns[{index}].created_at",
                    ),
                    self._timestamp_param(
                        pattern.get("updated_at"),
                        f"temporal_patterns[{index}].updated_at",
                    ),
                )
            )
            count += 1

        self.stats["temporal_patterns_imported"] = count
        logger.info(f"Imported {count} temporal patterns")

    async def _import_reflection_insights(self, agent_id: str, insights: List[Dict[str, Any]]):
        """Import reflection insights."""
        count = 0
        for index, insight in enumerate(insights):
            new_id = namespace_imported_record(agent_id, insight["id"])
            await self.db.execute(
                """INSERT OR REPLACE INTO reflection_insights
                   (id, agent_id, type, title, description, evidence, confidence,
                    actionable, suggested_action, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_id,
                    agent_id,
                    insight["insight_type"],  # maps to 'type' column
                    insight["title"],
                    insight.get("description", ""),
                    json.dumps(insight.get("evidence", [])),
                    insight.get("confidence", 0.5),
                    1 if insight.get("actionable") else 0,
                    insight.get("suggested_action"),
                    self._timestamp_param(
                        insight.get("created_at"),
                        f"reflection_insights[{index}].created_at",
                    ),
                )
            )
            count += 1

        self.stats["reflection_insights_imported"] = count
        logger.info(f"Imported {count} reflection insights")

    def _namespace_node_id(self, agent_id: str, raw_id: str) -> str:
        """Place a package-supplied graph id in this full agent's namespace."""
        return namespace_imported_record(agent_id, raw_id)

    async def _upsert_owned_graph_node(
        self,
        agent_id: str,
        *,
        node_id: str,
        node_type: str,
        label: str,
        properties: str,
    ) -> None:
        """Insert or update one import-owned node without claiming collisions."""
        existing = await self.db.fetchone(
            "SELECT 1 FROM graph_nodes WHERE node_id = ?",
            (node_id,),
        )
        if existing:
            owner_rows = await self.db.fetchall(
                "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
                (node_id,),
            )
            if {row[0] for row in owner_rows} != {agent_id}:
                raise ValueError(
                    "import graph node id is unowned or owned by another agent"
                )
            await self.db.execute(
                "UPDATE graph_nodes SET node_type = ?, label = ?, properties = ? "
                "WHERE node_id = ?",
                (node_type, label, properties, node_id),
            )
        else:
            await self.db.execute(
                "INSERT INTO graph_nodes "
                "(node_id, node_type, label, properties) VALUES (?, ?, ?, ?)",
                (node_id, node_type, label, properties),
            )
        await record_graph_node_owner(self.db, node_id, agent_id)

    async def _upsert_owned_graph_edge(
        self,
        agent_id: str,
        *,
        source_id: str,
        target_id: str,
        label: str,
        properties: str = "{}",
    ) -> None:
        """Insert or update one import-owned edge without replacing a peer's."""
        existing = await self.db.fetchone(
            "SELECT 1 FROM graph_edges "
            "WHERE source_id = ? AND target_id = ? AND label = ?",
            (source_id, target_id, label),
        )
        if existing:
            owner_rows = await self.db.fetchall(
                "SELECT agent_id FROM graph_edge_owners "
                "WHERE source_id = ? AND target_id = ? AND label = ?",
                (source_id, target_id, label),
            )
            if {row[0] for row in owner_rows} != {agent_id}:
                raise ValueError(
                    "import graph edge is unowned or owned by another agent"
                )
            await self.db.execute(
                "UPDATE graph_edges SET properties = ? "
                "WHERE source_id = ? AND target_id = ? AND label = ?",
                (properties, source_id, target_id, label),
            )
        else:
            await self.db.execute(
                "INSERT INTO graph_edges "
                "(source_id, target_id, label, properties) VALUES (?, ?, ?, ?)",
                (source_id, target_id, label, properties),
            )
        await record_graph_edge_owner(
            self.db, source_id, target_id, label, agent_id
        )

    async def _node_is_protected(self, agent_id: str, node_id: str) -> bool:
        """True if ``node_id`` must not be overwritten by an import.

        F186 defense-in-depth on top of namespacing: refuse to upsert any
        node that IS the importing agent's identity node, or that collides
        with an EXISTING node whose type is reserved (governance, identity,
        lineage). Prevents a crafted package from clobbering those rows even
        if the namespacing prefix were ever bypassed.
        """
        if node_id == agent_id:
            return True
        try:
            row = await self.db.fetchone(
                "SELECT node_type FROM graph_nodes WHERE node_id = ?",
                (node_id,),
            )
        except Exception as e:
            # Fail safe: if we can't determine the existing type, refuse the
            # overwrite rather than risk clobbering a protected row.
            logger.warning(f"Reserved-node check failed for {node_id}: {e}")
            return True
        return bool(row) and row[0] in self.RESERVED_NODE_TYPES

    async def _import_relationships(self, agent_id: str, relationships: List[Any]):
        """Import relationships as graph nodes and edges."""
        count = 0
        for rel in relationships:
            rel_dict = rel.to_dict() if hasattr(rel, 'to_dict') else rel
            # Create user node. F186: namespace the package-supplied id.
            raw_user_id = rel_dict["user_id"]
            user_id = self._namespace_node_id(agent_id, raw_user_id)
            if await self._node_is_protected(agent_id, user_id):
                raise ValueError(
                    "relationship node collides with a reserved identity node"
                )
            properties = json.dumps({
                "first_interaction": rel_dict.get("first_interaction"),
                "last_interaction": rel_dict.get("last_interaction"),
                "interaction_count": rel_dict.get("interaction_count", 0),
                "notes": rel_dict.get("relationship_notes", ""),
                "trust_level": rel_dict.get("trust_level", 0.5),
                "preferences": rel_dict.get("preferences_learned", {}),
            })

            await self._upsert_owned_graph_node(
                agent_id,
                node_id=user_id,
                node_type="user",
                label=f"User {str(raw_user_id)[:8]}",
                properties=properties,
            )

            relationship_type = rel_dict.get("relationship_type", "knows")
            await self._upsert_owned_graph_edge(
                agent_id,
                source_id=agent_id,
                target_id=user_id,
                label=relationship_type,
            )
            count += 1

        self.stats["relationships_imported"] = count
        logger.info(f"Imported {count} relationships")

    async def _import_skills(self, agent_id: str, skills: List[Any]):
        """Import skills as graph nodes."""
        count = 0
        for skill in skills:
            skill_dict = skill.to_dict() if hasattr(skill, 'to_dict') else skill
            raw_skill_id = skill_dict["skill_id"]
            skill_id = self._namespace_node_id(agent_id, raw_skill_id)
            if await self._node_is_protected(agent_id, skill_id):
                raise ValueError(
                    "skill node collides with a reserved identity node"
                )
            properties = json.dumps({
                "type": skill_dict.get("skill_type"),
                "proficiency": skill_dict.get("proficiency", 0.5),
                "times_used": skill_dict.get("times_used", 0),
                "last_used": skill_dict.get("last_used"),
                "config": skill_dict.get("configuration", {}),
            })

            await self._upsert_owned_graph_node(
                agent_id,
                node_id=skill_id,
                node_type="skill",
                label=skill_dict["skill_name"],
                properties=properties,
            )

            await self._upsert_owned_graph_edge(
                agent_id,
                source_id=agent_id,
                target_id=skill_id,
                label="has_skill",
            )
            count += 1

        self.stats["skills_imported"] = count
        logger.info(f"Imported {count} skills")

    async def _import_wallet_state(self, agent_id: str, package: AgentIdentityPackage):
        """Import wallet state."""
        # Transaction ids are backend-local surrogate keys; signed fields are
        # portable and receive fresh row ids on the target backend.
        await self.db.execute(
            """INSERT OR REPLACE INTO wallet_state
               (agent_id, main_balance, audit_balance, updated_at)
               VALUES (?, ?, ?, ?)""",
            (
                agent_id,
                package.wallet_balance,
                "0.0",
                self._timestamp_param(
                    datetime.now(timezone.utc), "wallet_state.updated_at"
                ),
            )
        )

        imported = 0
        skipped = 0
        for index, tx in enumerate(package.wallet_transaction_history):
            created_at = self._timestamp_param(
                tx.get("created_at"),
                f"wallet_transaction_history[{index}].created_at",
            )
            memo = tx.get("memo")
            duplicate_sql = (
                "SELECT COUNT(*) FROM wallet_transactions "
                "WHERE agent_id = ? AND amount = ? "
                "AND COALESCE(memo, '') = ?"
            )
            duplicate_params: tuple[Any, ...] = (
                agent_id,
                str(tx["amount"]),
                "" if memo is None else str(memo),
            )
            if created_at is None:
                duplicate_sql += " AND created_at IS NULL"
            else:
                duplicate_sql += " AND created_at = ?"
                duplicate_params += (created_at,)
            existing = await self.db.fetchone(
                duplicate_sql,
                duplicate_params,
            )
            if existing and existing[0] > 0:
                skipped += 1
                continue

            await self.db.execute(
                """INSERT INTO wallet_transactions
                   (agent_id, transaction_type, currency, amount, memo,
                    new_balance, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    agent_id,
                    tx.get("transaction_type", "imported"),
                    tx.get("currency", "FIL"),
                    str(tx["amount"]),
                    memo,
                    str(tx.get("new_balance", package.wallet_balance)),
                    created_at,
                )
            )
            imported += 1

        if skipped:
            self.warnings.append(
                f"Skipped {skipped} wallet transaction(s) already present"
            )
        self.stats["wallet_imported"] = True
        self.stats["wallet_transactions_imported"] = imported
        self.stats["wallet_transactions_skipped_existing"] = skipped
        logger.info(f"Imported wallet state (balance: {package.wallet_balance})")

    async def _record_migration(
        self,
        agent_id: str,
        package: AgentIdentityPackage,
        migration_id: str
    ):
        """Record the migration in the graph."""
        timestamp = datetime.now(timezone.utc).isoformat()

        properties = json.dumps({
            "timestamp": timestamp,
            "source_substrate": package.source_substrate,
            "target_substrate": self.target_substrate,
            "source_package_hash": package.content_hash,
            "package_version": package.package_version,
            "reason": "identity_import",
            "stats": self.stats,
        })

        # Audit evidence is load-bearing: failure aborts the whole import.
        await self.db.execute(
            """INSERT INTO graph_nodes
               (node_id, node_type, label, properties)
               VALUES (?, 'migration_record', ?, ?)""",
            (migration_id, f"Migration {migration_id[:8]}", properties)
        )
        await record_graph_node_owner(self.db, migration_id, agent_id)

        await self.db.execute(
            """INSERT INTO graph_edges
               (source_id, target_id, label, properties)
               VALUES (?, ?, 'migrated_via', '{}')""",
            (agent_id, migration_id)
        )
        await record_graph_edge_owner(
            self.db, agent_id, migration_id, "migrated_via", agent_id
        )
        logger.info(f"Recorded migration: {migration_id}")


async def import_identity(
    db: "AsyncDatabase",
    package: AgentIdentityPackage,
    target_agent_id: Optional[str] = None,
    target_substrate: Optional[str] = None,
    **kwargs
) -> ImportResult:
    """
    Convenience function for importing agent identity.

    Args:
        db: Database connection
        package: The identity package to import
        target_agent_id: Override the agent ID (uses package.did if None)
        target_substrate: The substrate being imported to
        **kwargs: Additional arguments passed to IdentityImporter.import_package()

    Returns:
        ImportResult with success status and statistics
    """
    importer = IdentityImporter(db, target_agent_id, target_substrate)
    return await importer.import_package(package, **kwargs)
