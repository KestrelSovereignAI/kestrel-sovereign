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
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .identity_package import (
    AgentIdentityPackage,
    MigrationRecord,
    SubstrateType,
    create_migration_id,
)

if TYPE_CHECKING:
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
    stats: Dict[str, int]

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

    def __init__(
        self,
        db: "AsyncDatabase",
        target_agent_id: Optional[str] = None,
        target_substrate: Optional[str] = None,
    ):
        """
        Initialize the importer.

        Args:
            db: Database connection
            target_agent_id: The agent ID to import into. If None, uses DID from package.
            target_substrate: The substrate being imported to
        """
        self.db = db
        self.target_agent_id = target_agent_id
        self.target_substrate = target_substrate or SubstrateType.UNKNOWN.value
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, int] = {}

    async def import_package(
        self,
        package: AgentIdentityPackage,
        verify_signature: bool = True,
        verify_constitution: bool = True,
        merge_mode: str = "replace",  # replace, merge, skip_existing
        allow_unsigned: bool = False,
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

        if verify_signature:
            # Hybrid packages carry sigs only on package.signatures
            # (the v2 array); the legacy package.signature field is
            # empty by design for post-ceremony agents because that
            # field can't be made byte-compatible with v1 readers
            # over a v2 canonical hash. Treat either carrier as
            # "signed" and let _verify_signature route by alg.
            has_signature = bool(package.signature) or bool(package.signatures)
            if has_signature:
                sig_valid = await self._verify_signature(package)
                if not sig_valid:
                    self.errors.append("DID signature verification failed")
                    return self._build_result(False, agent_id)
            elif not allow_unsigned:
                self.errors.append(
                    "Package is not signed (unsigned). "
                    "Set allow_unsigned=True to import unsigned packages."
                )
                return self._build_result(False, agent_id)
            else:
                self.warnings.append("Importing unsigned package (allow_unsigned=True)")

        # 2. Check merge mode
        if merge_mode == "skip_existing":
            existing = await self._check_existing_data(agent_id)
            if existing:
                self.warnings.append("Skipping import - existing data found")
                return self._build_result(True, agent_id)

        # 3. Begin import
        migration_id = create_migration_id()

        try:
            # Clear existing data if replace mode
            if merge_mode == "replace":
                await self._clear_existing_data(agent_id)

            # Import each component
            await self._import_episodes(agent_id, package.episodes)
            await self._import_saved_items(agent_id, package.saved_items)
            await self._import_temporal_patterns(agent_id, package.temporal_patterns)
            await self._import_reflection_insights(agent_id, package.reflection_insights)
            await self._import_relationships(agent_id, package.relationships)
            await self._import_skills(agent_id, package.skills)
            await self._import_wallet_state(agent_id, package)

            # Record the migration
            await self._record_migration(agent_id, package, migration_id)

            logger.info(f"Import complete: {self.stats}")
            return self._build_result(True, agent_id, migration_id)

        except Exception as e:
            self.errors.append(f"Import failed: {str(e)}")
            logger.error(f"Import failed: {e}", exc_info=True)
            return self._build_result(False, agent_id, migration_id)

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

    async def _verify_signature(self, package: AgentIdentityPackage) -> bool:
        """Verify the package's signature via the canonical verifier.

        Routes through ``verify_package_signature`` so both legacy
        ``signature`` (single ECDSA hex) and the v2 ``signatures``
        array (hybrid Ed25519 + ML-DSA-65) are handled by their
        appropriate paths. The verifier loads the trust anchor from
        the receiver's local agent_data dir.
        """
        try:
            from kestrel_sovereign.identity.signing import verify_package_signature
            ok, msg = verify_package_signature(package)
            if not ok:
                self.warnings.append(f"Signature verification failed: {msg}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            self.warnings.append(f"Signature verification failed: {str(e)}")
            return False

    async def _check_existing_data(self, agent_id: str) -> bool:
        """Check if there's existing data for this agent."""
        row = await self.db.fetchone(
            "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?",
            (agent_id,)
        )
        return row and row[0] > 0

    async def _clear_existing_data(self, agent_id: str):
        """Clear existing data for this agent (replace mode)."""
        tables = [
            "memory_episodes",
            "saved_items",
            "temporal_patterns",
            "reflection_insights",
        ]
        for table in tables:
            try:
                await self.db.execute(
                    f"DELETE FROM {table} WHERE agent_id = ?",
                    (agent_id,)
                )
            except Exception as e:
                logger.warning(f"Could not clear {table}: {e}")

    async def _import_episodes(self, agent_id: str, episodes: List[Dict[str, Any]]):
        """Import memory episodes."""
        count = 0
        for ep in episodes:
            try:
                # Generate new ID to avoid conflicts when importing to same DB
                new_id = f"{agent_id[:20]}_{ep.get('id', '')}"
                await self.db.execute(
                    """INSERT OR REPLACE INTO memory_episodes
                       (id, agent_id, title, summary, timespan_start, timespan_end,
                        key_message_ids, emotional_arc, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_id,
                        agent_id,
                        ep.get("title"),
                        ep.get("summary"),
                        ep.get("timespan_start"),
                        ep.get("timespan_end"),
                        json.dumps(ep.get("key_message_ids", [])),
                        ep.get("emotional_arc"),
                        ep.get("created_at"),
                    )
                )
                count += 1
            except Exception as e:
                self.warnings.append(f"Failed to import episode {ep.get('id')}: {e}")

        await self.db.commit()
        self.stats["episodes_imported"] = count
        logger.info(f"Imported {count} memory episodes")

    async def _import_saved_items(self, agent_id: str, items: List[Dict[str, Any]]):
        """Import saved items."""
        count = 0
        for item in items:
            try:
                # Generate new ID to avoid conflicts when importing to same DB
                new_id = f"{agent_id[:20]}_{item.get('id', '')}"
                await self.db.execute(
                    """INSERT OR REPLACE INTO saved_items
                       (id, agent_id, item_type, name, summary, content, content_hash,
                        ipfs_cid, source_type, source_ref, schema_id, tags, metadata,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_id,
                        agent_id,
                        item.get("item_type"),
                        item.get("name"),
                        item.get("summary"),
                        item.get("content"),
                        item.get("content_hash"),
                        item.get("ipfs_cid"),
                        item.get("source_type"),
                        item.get("source_ref"),
                        item.get("schema_id"),
                        json.dumps(item.get("tags", [])),
                        json.dumps(item.get("metadata", {})),
                        item.get("created_at"),
                        item.get("updated_at"),
                    )
                )
                count += 1
            except Exception as e:
                self.warnings.append(f"Failed to import saved item {item.get('id')}: {e}")

        await self.db.commit()
        self.stats["saved_items_imported"] = count
        logger.info(f"Imported {count} saved items")

    async def _import_temporal_patterns(self, agent_id: str, patterns: List[Dict[str, Any]]):
        """Import temporal patterns."""
        count = 0
        for pattern in patterns:
            try:
                # Generate new ID to avoid conflicts when importing to same DB
                new_id = f"{agent_id[:20]}_{pattern.get('id', '')}"
                await self.db.execute(
                    """INSERT OR REPLACE INTO temporal_patterns
                       (id, agent_id, pattern_type, description, trigger_conditions,
                        confidence, observations, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_id,
                        agent_id,
                        pattern.get("pattern_type"),
                        pattern.get("description"),
                        json.dumps(pattern.get("trigger_conditions", {})),
                        pattern.get("confidence", 0.0),
                        pattern.get("observations", 0),
                        pattern.get("created_at"),
                        pattern.get("updated_at"),
                    )
                )
                count += 1
            except Exception as e:
                self.warnings.append(f"Failed to import pattern {pattern.get('id')}: {e}")

        await self.db.commit()
        self.stats["temporal_patterns_imported"] = count
        logger.info(f"Imported {count} temporal patterns")

    async def _import_reflection_insights(self, agent_id: str, insights: List[Dict[str, Any]]):
        """Import reflection insights."""
        count = 0
        for insight in insights:
            try:
                # Generate new ID to avoid conflicts when importing to same DB
                new_id = f"{agent_id[:20]}_{insight.get('id', '')}"
                await self.db.execute(
                    """INSERT OR REPLACE INTO reflection_insights
                       (id, agent_id, type, title, description, evidence, confidence,
                        actionable, suggested_action, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_id,
                        agent_id,
                        insight.get("insight_type"),  # maps to 'type' column
                        insight.get("title", "Imported Insight"),
                        insight.get("description", ""),
                        json.dumps(insight.get("evidence", [])),
                        insight.get("confidence", 0.5),
                        1 if insight.get("actionable") else 0,
                        insight.get("suggested_action"),
                        insight.get("created_at"),
                    )
                )
                count += 1
            except Exception as e:
                self.warnings.append(f"Failed to import insight {insight.get('id')}: {e}")

        await self.db.commit()
        self.stats["reflection_insights_imported"] = count
        logger.info(f"Imported {count} reflection insights")

    async def _import_relationships(self, agent_id: str, relationships: List[Any]):
        """Import relationships as graph nodes and edges."""
        count = 0
        for rel in relationships:
            rel_dict = rel.to_dict() if hasattr(rel, 'to_dict') else rel
            try:
                # Create user node
                user_id = rel_dict.get("user_id")
                properties = json.dumps({
                    "first_interaction": rel_dict.get("first_interaction"),
                    "last_interaction": rel_dict.get("last_interaction"),
                    "interaction_count": rel_dict.get("interaction_count", 0),
                    "notes": rel_dict.get("relationship_notes", ""),
                    "trust_level": rel_dict.get("trust_level", 0.5),
                    "preferences": rel_dict.get("preferences_learned", {}),
                })

                await self.db.execute(
                    """INSERT OR REPLACE INTO graph_nodes
                       (node_id, node_type, label, properties)
                       VALUES (?, 'user', ?, ?)""",
                    (user_id, f"User {user_id[:8]}", properties)
                )

                # Create edge
                await self.db.execute(
                    """INSERT OR REPLACE INTO graph_edges
                       (source_id, target_id, label, properties)
                       VALUES (?, ?, ?, '{}')""",
                    (agent_id, user_id, rel_dict.get("relationship_type", "knows"))
                )
                count += 1
            except Exception as e:
                self.warnings.append(f"Failed to import relationship: {e}")

        await self.db.commit()
        self.stats["relationships_imported"] = count
        logger.info(f"Imported {count} relationships")

    async def _import_skills(self, agent_id: str, skills: List[Any]):
        """Import skills as graph nodes."""
        count = 0
        for skill in skills:
            skill_dict = skill.to_dict() if hasattr(skill, 'to_dict') else skill
            try:
                skill_id = skill_dict.get("skill_id")
                properties = json.dumps({
                    "type": skill_dict.get("skill_type"),
                    "proficiency": skill_dict.get("proficiency", 0.5),
                    "times_used": skill_dict.get("times_used", 0),
                    "last_used": skill_dict.get("last_used"),
                    "config": skill_dict.get("configuration", {}),
                })

                await self.db.execute(
                    """INSERT OR REPLACE INTO graph_nodes
                       (node_id, node_type, label, properties)
                       VALUES (?, 'skill', ?, ?)""",
                    (skill_id, skill_dict.get("skill_name"), properties)
                )

                # Create edge
                await self.db.execute(
                    """INSERT OR REPLACE INTO graph_edges
                       (source_id, target_id, label, properties)
                       VALUES (?, ?, 'has_skill', '{}')""",
                    (agent_id, skill_id)
                )
                count += 1
            except Exception as e:
                self.warnings.append(f"Failed to import skill: {e}")

        await self.db.commit()
        self.stats["skills_imported"] = count
        logger.info(f"Imported {count} skills")

    async def _import_wallet_state(self, agent_id: str, package: AgentIdentityPackage):
        """Import wallet state."""
        try:
            # Insert or update wallet balance (main_balance column in schema)
            await self.db.execute(
                """INSERT OR REPLACE INTO wallet_state
                   (agent_id, main_balance, audit_balance, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (agent_id, package.wallet_balance, "0.0", datetime.now(timezone.utc).isoformat())
            )

            # Import transaction history
            for tx in package.wallet_transaction_history:
                try:
                    await self.db.execute(
                        """INSERT INTO wallet_transactions
                           (id, agent_id, amount, memo, created_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            tx.get("id"),
                            agent_id,
                            tx.get("amount"),
                            tx.get("memo"),
                            tx.get("created_at"),
                        )
                    )
                except Exception:
                    pass  # Skip duplicate transactions

            await self.db.commit()
            self.stats["wallet_imported"] = True
            logger.info(f"Imported wallet state (balance: {package.wallet_balance})")
        except Exception as e:
            self.warnings.append(f"Failed to import wallet state: {e}")

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

        try:
            # Create migration record node
            await self.db.execute(
                """INSERT INTO graph_nodes
                   (node_id, node_type, label, properties)
                   VALUES (?, 'migration_record', ?, ?)""",
                (migration_id, f"Migration {migration_id[:8]}", properties)
            )

            # Link to agent
            await self.db.execute(
                """INSERT INTO graph_edges
                   (source_id, target_id, label, properties)
                   VALUES (?, ?, 'migrated_via', '{}')""",
                (agent_id, migration_id)
            )

            await self.db.commit()
            logger.info(f"Recorded migration: {migration_id}")
        except Exception as e:
            self.warnings.append(f"Failed to record migration: {e}")


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
