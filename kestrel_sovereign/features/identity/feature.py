#!/usr/bin/env python3
"""
Identity Feature: Agent tools for identity portability.

This feature provides the agent with tools to export, import, and verify
its identity across different LLM substrates.

Implements the tools defined in Issue #23:
- export_identity: Create signed identity package
- import_identity: Load identity on new substrate
- verify_identity: Run continuity tests
- assess_substrate: Check current substrate capabilities
- migration_history: Show migration audit trail
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sdk.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class IdentityFeature(Feature):
    """
    Feature for managing agent identity portability.

    Provides tools for exporting and importing agent identity across
    different LLM substrates while preserving continuity of self.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage agent identity portability - export identity to portable package, "
            "import identity from package, verify integrity, assess substrate capabilities, "
            "and view migration history"
        )

    async def initialize(self):
        """Initialize the identity feature."""
        logger.info("Initializing IdentityFeature")

    @tool(
        name="export_identity",
        description="Export the agent's complete identity to a portable, signed package. "
                    "This creates a JSON package containing DID, constitution, memories, "
                    "personality, relationships, and skills that can be imported to another substrate.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity export"
    )
    async def export_identity(
        self,
        storage_tier: str = "local",
        sign: bool = True,
        include_wallet: bool = True,
    ) -> str:
        """
        Export agent identity to a portable package.

        Args:
            storage_tier: Where to store the package ('local', 'ipfs', 'filecoin')
            sign: Whether to sign the package with DID key
            include_wallet: Whether to include wallet transaction history
        """
        try:
            from kestrel_sovereign.identity import (
                IdentityExporter,
                sign_package,
            )
            from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier

            db = resolve_feature_database(self.agent)
            if db is None:
                return "Export failed: database not available"

            # Export identity
            exporter = IdentityExporter(
                db=db,
                agent_id=self.agent.agent_id,
            )
            package = await exporter.export(include_wallet_history=include_wallet)

            # Sign if requested
            if sign:
                try:
                    package = sign_package(package)
                except Exception as e:
                    logger.warning(f"Could not sign package: {e}")

            # Get JSON representation
            package_json = package.to_json()
            summary = package.get_summary()

            # Store based on tier
            tier_map = {
                "local": StorageTier.LOCAL_ONLY,
                "ipfs": StorageTier.IPFS,
                "filecoin": StorageTier.FILECOIN,
            }
            tier_enum = tier_map.get(storage_tier.lower(), StorageTier.LOCAL_ONLY)

            if tier_enum != StorageTier.LOCAL_ONLY:
                # Upload to IPFS/Filecoin
                adapter = FilecoinAdapter()
                result = await asyncio.to_thread(
                    adapter.store_content,
                    content=package_json.encode('utf-8'),
                    storage_tier=tier_enum,
                    encrypt=False,  # Package is already structured
                    metadata={"type": "identity_package", "did": package.did}
                )
                cid = result.ipfs_cid or result.content_hash

                return f"""Identity Export Complete

Package Summary:
- DID: {summary['did'][:30]}...
- Agent Name: {summary['agent_name']}
- Created: {summary['created_at']}
- Episodes: {summary['episodes_count']}
- Saved Items: {summary['saved_items_count']}
- Relationships: {summary['relationships_count']}
- Skills: {summary['skills_count']}
- Signed: {summary['is_signed']}

Storage:
- Tier: {tier_enum.value}
- CID: {cid}

Use `!identity import {cid}` to restore this identity on another substrate.
"""
            else:
                # Save to local file
                storage_dir = Path(os.environ.get("KESTREL_DATA_DIR", "agent_data"))
                storage_dir.mkdir(exist_ok=True)
                filename = f"identity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
                filepath = storage_dir / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(package_json)

                return f"""Identity Export Complete

Package Summary:
- DID: {summary['did'][:30]}...
- Agent Name: {summary['agent_name']}
- Created: {summary['created_at']}
- Episodes: {summary['episodes_count']}
- Saved Items: {summary['saved_items_count']}
- Relationships: {summary['relationships_count']}
- Skills: {summary['skills_count']}
- Signed: {summary['is_signed']}

Storage:
- Tier: local
- File: {filepath}

Use `!identity import {filepath}` to restore this identity.
"""

        except Exception as e:
            logger.error(f"Identity export failed: {e}", exc_info=True)
            return f"Identity export failed: {str(e)}"

    @tool(
        name="import_identity",
        description="Import agent identity from a portable package. "
                    "This restores memories, personality, relationships, and skills "
                    "from a previously exported identity package.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity import"
    )
    async def import_identity(
        self,
        source: str,
        verify_signature: bool = True,
        merge_mode: str = "merge",
    ) -> str:
        """
        Import agent identity from a package.

        Args:
            source: CID or file path of the identity package
            verify_signature: Whether to verify DID signature
            merge_mode: How to handle existing data ('replace', 'merge', 'skip_existing')
        """
        try:
            from kestrel_sovereign.identity import (
                AgentIdentityPackage,
                IdentityImporter,
            )
            from kestrel_sovereign.filecoin_adapter import FilecoinAdapter

            # Load package
            if source.startswith("Qm") or source.startswith("bafy"):
                # IPFS CID
                adapter = FilecoinAdapter()
                content = await asyncio.to_thread(
                    adapter.retrieve_content, source, ipfs_cid=source
                )
                package_json = content.decode('utf-8')
            elif Path(source).exists():
                # Local file
                with open(source, 'r', encoding='utf-8') as f:
                    package_json = f.read()
            else:
                return f"Source not found: {source}"

            # Parse package
            package = AgentIdentityPackage.from_json(package_json)
            summary = package.get_summary()

            # Verify integrity
            constitution_ok = package.verify_constitution() if package.constitution_text else True
            hash_ok = package.verify_content_hash() if package.content_hash else True

            if not constitution_ok:
                return "Import failed: Constitution hash verification failed"
            if not hash_ok:
                return "Import failed: Content hash verification failed (package may be corrupted)"

            db = resolve_feature_database(self.agent)
            if db is None:
                return "Import failed: database not available"

            # Import
            importer = IdentityImporter(
                db=db,
                target_agent_id=self.agent.agent_id,
            )
            result = await importer.import_package(
                package,
                verify_signature=verify_signature,
                merge_mode=merge_mode,
                allow_unsigned=False,
            )

            if result.success:
                stats = result.stats
                return f"""Identity Import Complete

Source Package:
- DID: {summary['did'][:30]}...
- Agent Name: {summary['agent_name']}
- Source Substrate: {summary['source_substrate']}
- Export Time: {summary['export_timestamp']}

Import Results:
- Episodes Imported: {stats.get('episodes_imported', 0)}
- Saved Items Imported: {stats.get('saved_items_imported', 0)}
- Temporal Patterns: {stats.get('temporal_patterns_imported', 0)}
- Relationships: {stats.get('relationships_imported', 0)}
- Skills: {stats.get('skills_imported', 0)}

Migration ID: {result.migration_id}
"""
            else:
                errors = "\n".join(f"- {e}" for e in result.errors)
                return f"""Identity Import Failed

Errors:
{errors}

Warnings:
{chr(10).join(f'- {w}' for w in result.warnings) if result.warnings else 'None'}
"""

        except Exception as e:
            logger.error(f"Identity import failed: {e}", exc_info=True)
            return f"Identity import failed: {str(e)}"

    @tool(
        name="verify_identity",
        description="Verify the integrity of an identity package without importing it. "
                    "Checks constitution hash, content hash, and signature.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity verify"
    )
    async def verify_identity(self, source: str) -> str:
        """
        Verify an identity package.

        Args:
            source: CID or file path of the identity package
        """
        try:
            from kestrel_sovereign.identity import (
                AgentIdentityPackage,
                verify_package_signature,
            )
            from kestrel_sovereign.filecoin_adapter import FilecoinAdapter

            # Load package
            if source.startswith("Qm") or source.startswith("bafy"):
                adapter = FilecoinAdapter()
                content = await asyncio.to_thread(
                    adapter.retrieve_content, source, ipfs_cid=source
                )
                package_json = content.decode('utf-8')
            elif Path(source).exists():
                with open(source, 'r', encoding='utf-8') as f:
                    package_json = f.read()
            else:
                return f"Source not found: {source}"

            # Parse package
            package = AgentIdentityPackage.from_json(package_json)
            summary = package.get_summary()

            # Verify constitution
            constitution_status = "N/A"
            if package.constitution_text:
                if package.verify_constitution():
                    constitution_status = "VALID"
                else:
                    constitution_status = "INVALID"

            # Verify content hash
            hash_status = "N/A"
            if package.content_hash:
                if package.verify_content_hash():
                    hash_status = "VALID"
                else:
                    hash_status = "INVALID (package may have been modified)"

            # Verify signature. Hybrid packages carry sigs only on the
            # v2 ``signatures`` array; ``package.signature`` is empty
            # by design for post-ceremony agents.
            sig_status = "UNSIGNED"
            if package.signature or package.signatures:
                is_valid, msg = verify_package_signature(package)
                sig_status = "VALID" if is_valid else f"INVALID: {msg}"

            return f"""Identity Package Verification

Package Info:
- DID: {summary['did']}
- Agent Name: {summary['agent_name']}
- Created: {summary['created_at']}
- Exported: {summary['export_timestamp']}
- Source Substrate: {summary['source_substrate']}
- Package Version: {summary['package_version']}

Contents:
- Episodes: {summary['episodes_count']}
- Saved Items: {summary['saved_items_count']}
- Relationships: {summary['relationships_count']}
- Skills: {summary['skills_count']}
- Previous Migrations: {summary['migrations_count']}

Verification:
- Constitution: {constitution_status}
- Content Hash: {hash_status}
- Signature: {sig_status}
"""

        except Exception as e:
            logger.error(f"Identity verification failed: {e}", exc_info=True)
            return f"Identity verification failed: {str(e)}"

    @tool(
        name="assess_substrate",
        description="Assess the current LLM substrate's capabilities and compare "
                    "with agent requirements. Helps understand limitations when migrating.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity assess"
    )
    async def assess_substrate(self) -> str:
        """
        Assess current substrate capabilities.
        """
        try:
            from kestrel_sovereign.config import load_config
            from kestrel_sovereign.identity import SubstrateType

            config = load_config()
            llm_config = config.get("llm", {})

            # Detect substrate
            provider = llm_config.get("default_provider", "unknown")
            model = llm_config.get("default_model", "unknown")

            # Map to substrate type
            if "anthropic" in provider.lower() or "claude" in provider.lower():
                substrate = SubstrateType.ANTHROPIC_CLAUDE.value
            elif "openai" in provider.lower():
                substrate = SubstrateType.OPENAI_GPT.value
            elif "gemini" in provider.lower() or "google" in provider.lower():
                substrate = SubstrateType.GOOGLE_GEMINI.value
            elif "ollama" in provider.lower():
                substrate = SubstrateType.OLLAMA_LOCAL.value
            elif "openrouter" in provider.lower():
                substrate = SubstrateType.OPENROUTER.value
            else:
                substrate = SubstrateType.UNKNOWN.value

            # Assess capabilities
            capabilities = {
                "tool_use": "Yes" if "claude" in model.lower() or "gpt-4" in model.lower() else "Unknown",
                "vision": "Yes" if "vision" in model.lower() or "4o" in model.lower() else "Unknown",
                "long_context": "128K+" if "claude" in model.lower() else "Unknown",
                "streaming": "Yes",
                "function_calling": "Yes" if substrate != SubstrateType.UNKNOWN.value else "Unknown",
            }

            cap_lines = "\n".join(f"- {k}: {v}" for k, v in capabilities.items())

            return f"""Substrate Assessment

Current Substrate:
- Type: {substrate}
- Provider: {provider}
- Model: {model}

Capabilities:
{cap_lines}

Identity Status:
- Agent DID: {self.agent.agent_id[:30]}...
- Constitution: Anchored
- Memory System: Active

Note: Full capability mapping will be expanded in Phase 3 of substrate portability.
"""

        except Exception as e:
            logger.error(f"Substrate assessment failed: {e}", exc_info=True)
            return f"Substrate assessment failed: {str(e)}"

    @tool(
        name="migration_history",
        description="View the agent's migration history - all substrate changes "
                    "with timestamps, verification scores, and audit trail.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity history"
    )
    async def migration_history(self) -> str:
        """
        Show migration audit trail.
        """
        try:
            db = resolve_feature_database(self.agent)
            if db is None:
                return "No migration history found"

            # Get migration records from graph
            rows = await db.fetchall(
                """SELECT node_id, properties FROM graph_nodes
                   WHERE node_type = 'migration_record'
                   AND node_id IN (
                       SELECT target_id FROM graph_edges
                       WHERE source_id = ? AND edge_type = 'migrated_via'
                   )
                   ORDER BY node_id DESC""",
                (self.agent.agent_id,)
            )

            if not rows:
                return """Migration History

No migrations recorded.

This agent was born on this substrate and has never been migrated.

Use `!identity export` to create a portable identity package.
"""

            # Format migration records
            records = []
            for row in rows:
                props = json.loads(row[1]) if row[1] else {}
                records.append({
                    "id": row[0][:12],
                    "timestamp": props.get("timestamp", "Unknown")[:19],
                    "from": props.get("source_substrate", "Unknown"),
                    "to": props.get("target_substrate", "Unknown"),
                    "stats": props.get("stats", {}),
                })

            # Build history display
            history_lines = []
            for i, rec in enumerate(records, 1):
                stats = rec["stats"]
                history_lines.append(f"""
{i}. Migration {rec['id']}
   Date: {rec['timestamp']}
   From: {rec['from']} -> To: {rec['to']}
   Items: {stats.get('episodes_imported', 0)} episodes, {stats.get('saved_items_imported', 0)} items
""")

            return f"""Migration History

Total Migrations: {len(records)}
{"".join(history_lines)}
Use `!identity export` to create a new identity package for migration.
"""

        except Exception as e:
            logger.error(f"Migration history failed: {e}", exc_info=True)
            return f"Migration history failed: {str(e)}"
