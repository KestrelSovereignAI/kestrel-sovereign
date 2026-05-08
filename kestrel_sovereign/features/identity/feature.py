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
from kestrel_sdk.tools.result import ToolResult

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
    ) -> ToolResult:
        """
        Export agent identity to a portable package.

        Args:
            storage_tier: Where to store the package ('local', 'ipfs', 'filecoin')
            sign: Whether to sign the package with DID key
            include_wallet: Whether to include wallet transaction history
        """
        if not isinstance(sign, bool):
            return ToolResult.failed(
                f"sign must be a boolean, got {type(sign).__name__}={sign!r}"
            )
        if not isinstance(include_wallet, bool):
            return ToolResult.failed(
                "include_wallet must be a boolean, got "
                f"{type(include_wallet).__name__}={include_wallet!r}"
            )

        try:
            from kestrel_sovereign.identity import (
                IdentityExporter,
                sign_package,
            )
            from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier

            db = resolve_feature_database(self.agent)
            if db is None:
                return ToolResult.failed("Export failed: database not available")

            # Export identity
            exporter = IdentityExporter(
                db=db,
                agent_id=self.agent.agent_id,
            )
            package = await exporter.export(include_wallet_history=include_wallet)

            # Sign if requested. The original code logged a warning
            # and silently proceeded with an unsigned package — that's
            # exactly the honesty leak the migration is meant to
            # surface. Track the failure so we can return PARTIAL.
            sign_failure: Optional[str] = None
            if sign:
                try:
                    package = sign_package(package)
                except Exception as e:
                    logger.warning(f"Could not sign package: {e}")
                    sign_failure = str(e)

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

                confirmation = (
                    f"Exported identity package: DID={summary['did'][:30]}..., "
                    f"agent={summary['agent_name']}, "
                    f"{summary['episodes_count']} episodes, "
                    f"{summary['saved_items_count']} items, "
                    f"signed={summary['is_signed']}; "
                    f"stored to tier={tier_enum.value} (CID={cid}). "
                    f"Use `!identity import {cid}` to restore."
                )
                data = {
                    "did": package.did,
                    "agent_name": summary['agent_name'],
                    "summary": dict(summary),
                    "storage_tier": tier_enum.value,
                    "cid": cid,
                    "signed": bool(summary.get("is_signed")),
                    "sign_requested": sign,
                    "sign_failure": sign_failure,
                }
            else:
                # Save to local file
                storage_dir = Path(os.environ.get("KESTREL_DATA_DIR", "agent_data"))
                storage_dir.mkdir(exist_ok=True)
                filename = f"identity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
                filepath = storage_dir / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(package_json)

                confirmation = (
                    f"Exported identity package: DID={summary['did'][:30]}..., "
                    f"agent={summary['agent_name']}, "
                    f"{summary['episodes_count']} episodes, "
                    f"{summary['saved_items_count']} items, "
                    f"signed={summary['is_signed']}; "
                    f"saved to {filepath}. "
                    f"Use `!identity import {filepath}` to restore."
                )
                data = {
                    "did": package.did,
                    "agent_name": summary['agent_name'],
                    "summary": dict(summary),
                    "storage_tier": "local",
                    "file_path": str(filepath),
                    "signed": bool(summary.get("is_signed")),
                    "sign_requested": sign,
                    "sign_failure": sign_failure,
                }
        except Exception as e:
            logger.error(f"Identity export failed: {e}", exc_info=True)
            return ToolResult.failed(f"Identity export failed: {str(e)}")

        # Honesty: signing was REQUESTED but FAILED. The package
        # was still written, but it's unsigned — anyone importing
        # it will see ``UNSIGNED`` and may reject it. Surface as
        # PARTIAL so the LLM cannot claim "exported and signed."
        if sign and sign_failure:
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    f"signing was requested but failed: {sign_failure}; "
                    "the package was written UNSIGNED — verify will report "
                    "UNSIGNED and an importer with allow_unsigned=False "
                    "will reject it"
                ),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

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
    ) -> ToolResult:
        """
        Import agent identity from a package.

        Args:
            source: CID or file path of the identity package
            verify_signature: Whether to verify DID signature
            merge_mode: How to handle existing data ('replace', 'merge', 'skip_existing')
        """
        if not isinstance(verify_signature, bool):
            return ToolResult.failed(
                "verify_signature must be a boolean, got "
                f"{type(verify_signature).__name__}={verify_signature!r}"
            )
        if merge_mode not in ("replace", "merge", "skip_existing"):
            return ToolResult.failed(
                f"merge_mode must be 'replace', 'merge', or 'skip_existing'; "
                f"got {merge_mode!r}"
            )

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
                return ToolResult.failed(
                    f"Source not found: {source}",
                    data={"source": source},
                )

            # Parse package
            package = AgentIdentityPackage.from_json(package_json)
            summary = package.get_summary()

            # Verify integrity
            constitution_ok = package.verify_constitution() if package.constitution_text else True
            hash_ok = package.verify_content_hash() if package.content_hash else True

            if not constitution_ok:
                return ToolResult.failed(
                    "Import failed: Constitution hash verification failed",
                    data={"source": source, "did": package.did},
                )
            if not hash_ok:
                return ToolResult.failed(
                    "Import failed: Content hash verification failed "
                    "(package may be corrupted)",
                    data={"source": source, "did": package.did},
                )

            db = resolve_feature_database(self.agent)
            if db is None:
                return ToolResult.failed("Import failed: database not available")

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
        except Exception as e:
            logger.error(f"Identity import failed: {e}", exc_info=True)
            return ToolResult.failed(f"Identity import failed: {str(e)}")

        if not result.success:
            return ToolResult.failed(
                "Identity import failed: "
                + ("; ".join(result.errors) if result.errors else "unknown error"),
                data={
                    "source": source,
                    "did": package.did,
                    "errors": list(result.errors),
                    "warnings": list(result.warnings),
                },
            )

        stats = result.stats or {}
        confirmation = (
            f"Imported identity package: DID={summary['did'][:30]}..., "
            f"agent={summary['agent_name']}, "
            f"from substrate={summary['source_substrate']}; "
            f"{stats.get('episodes_imported', 0)} episodes, "
            f"{stats.get('saved_items_imported', 0)} saved items, "
            f"{stats.get('relationships_imported', 0)} relationships, "
            f"{stats.get('skills_imported', 0)} skills "
            f"(migration_id={result.migration_id})"
        )
        data = {
            "source": source,
            "did": package.did,
            "agent_name": summary['agent_name'],
            "source_substrate": summary['source_substrate'],
            "export_timestamp": summary['export_timestamp'],
            "stats": dict(stats),
            "migration_id": result.migration_id,
            "merge_mode": merge_mode,
            "warnings": list(result.warnings),
        }

        # Honesty: import_package returned success=True but populated
        # warnings (e.g. partial schema-version mismatch, skipped
        # entries). Surface as PARTIAL so the LLM cannot claim a
        # clean import while skipped/warned items quietly went
        # missing.
        if result.warnings:
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    f"{len(result.warnings)} warning(s) during import: "
                    + "; ".join(result.warnings)
                ),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

    @tool(
        name="verify_identity",
        description="Verify the integrity of an identity package without importing it. "
                    "Checks constitution hash, content hash, and signature.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity verify"
    )
    async def verify_identity(self, source: str) -> ToolResult:
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
                return ToolResult.failed(
                    f"Source not found: {source}",
                    data={"source": source},
                )

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
        except Exception as e:
            logger.error(f"Identity verification failed: {e}", exc_info=True)
            return ToolResult.failed(f"Identity verification failed: {str(e)}")

        confirmation = (
            f"Verified package {summary['did'][:30]}... "
            f"(agent={summary['agent_name']}, "
            f"v{summary.get('package_version', '?')}): "
            f"constitution={constitution_status}, "
            f"hash={hash_status}, signature={sig_status}"
        )
        data = {
            "source": source,
            "summary": dict(summary),
            "constitution_status": constitution_status,
            "hash_status": hash_status,
            "signature_status": sig_status,
        }

        # Honesty: a verification that finds the package INVALID is
        # not a tool failure (the verify ran successfully and gave
        # an authoritative answer), but the LLM must surface the
        # invalid-half — otherwise it might say "verified the
        # package" without mentioning the check failed. PARTIAL
        # forces both halves to speak.
        invalid_parts = []
        if constitution_status == "INVALID":
            invalid_parts.append("constitution hash mismatch")
        if hash_status.startswith("INVALID"):
            invalid_parts.append("content hash mismatch (package may have been modified)")
        if sig_status.startswith("INVALID"):
            invalid_parts.append(f"signature {sig_status.lower()}")
        if invalid_parts:
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    "package verification reported failures: "
                    + "; ".join(invalid_parts)
                    + "; do NOT import this package without resolving"
                ),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

    @tool(
        name="assess_substrate",
        description="Assess the current LLM substrate's capabilities and compare "
                    "with agent requirements. Helps understand limitations when migrating.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity assess"
    )
    async def assess_substrate(self) -> ToolResult:
        """Assess current substrate capabilities."""
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
        except Exception as e:
            logger.error(f"Substrate assessment failed: {e}", exc_info=True)
            return ToolResult.failed(f"Substrate assessment failed: {str(e)}")

        cap_lines = "\n".join(f"- {k}: {v}" for k, v in capabilities.items())
        confirmation_text = (
            f"Substrate: {substrate} ({provider}/{model}); "
            f"capabilities: tool_use={capabilities['tool_use']}, "
            f"vision={capabilities['vision']}, "
            f"function_calling={capabilities['function_calling']}"
        )
        data = {
            "substrate_type": substrate,
            "provider": provider,
            "model": model,
            "capabilities": dict(capabilities),
            "agent_did_prefix": self.agent.agent_id[:30],
        }

        # Honesty: when substrate detection fell through to UNKNOWN,
        # the resulting capabilities map is full of "Unknown". The
        # tool ran successfully (it gave an answer) but downstream
        # decisions made on this assessment will be brittle. Surface
        # as PARTIAL so the LLM cannot claim "assessed substrate"
        # without mentioning the unknowns.
        if substrate == SubstrateType.UNKNOWN.value:
            return ToolResult.partial(
                confirmation=confirmation_text,
                error=(
                    f"substrate is UNKNOWN (provider={provider!r}); "
                    "capability assessment is best-effort and downstream "
                    "migration decisions should treat it as untrusted"
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation=confirmation_text,
            data=data,
        )

    @tool(
        name="migration_history",
        description="View the agent's migration history - all substrate changes "
                    "with timestamps, verification scores, and audit trail.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity history"
    )
    async def migration_history(self) -> ToolResult:
        """Show migration audit trail."""
        db = resolve_feature_database(self.agent)
        # Honesty: pre-fix, "no DB" returned the same string as "DB
        # OK but 0 records" — the LLM couldn't tell the difference.
        # Surface as ERROR when the DB is unavailable so downstream
        # logic doesn't infer "agent has no history" from a database
        # outage.
        if db is None:
            return ToolResult.failed("Database not available; migration history cannot be queried")

        try:
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
        except Exception as e:
            logger.error(f"Migration history failed: {e}", exc_info=True)
            return ToolResult.failed(f"Migration history failed: {str(e)}")

        if not rows:
            return ToolResult.ok(
                confirmation=(
                    "Migration history: no migrations recorded — this agent "
                    "was born on this substrate. Use `!identity export` to "
                    "create a portable package."
                ),
                data={"total_migrations": 0, "records": []},
            )

        # Format migration records. Wrap each per-row format step
        # so a single malformed row (e.g. ``{"stats": null}``,
        # numeric timestamp, missing fields) doesn't escape this
        # @tool — the migrated module's contract requires every
        # code path to return ToolResult. (Round 1 codex finding.)
        records = []
        parse_errors: List[str] = []
        for row in rows:
            try:
                node_id = row[0] if row else None
                if not isinstance(node_id, str):
                    parse_errors.append(f"row had non-string node_id: {node_id!r}")
                    continue
                short_id = node_id[:12]

                if row[1] is None:
                    props: Dict[str, Any] = {}
                elif isinstance(row[1], str):
                    props = json.loads(row[1])
                    if not isinstance(props, dict):
                        parse_errors.append(
                            f"node {short_id}: properties JSON parsed to "
                            f"{type(props).__name__}, expected object"
                        )
                        continue
                else:
                    parse_errors.append(
                        f"node {short_id}: properties is "
                        f"{type(row[1]).__name__}, expected str/None"
                    )
                    continue

                ts = props.get("timestamp", "Unknown")
                ts_str = ts[:19] if isinstance(ts, str) else str(ts)[:19]

                # ``stats`` may be missing, ``null``, or a non-dict —
                # accept dict, fall back to {} otherwise.
                raw_stats = props.get("stats")
                stats = dict(raw_stats) if isinstance(raw_stats, dict) else {}

                records.append({
                    "id": short_id,
                    "full_id": node_id,
                    "timestamp": ts_str,
                    "from": props.get("source_substrate") or "Unknown",
                    "to": props.get("target_substrate") or "Unknown",
                    "stats": stats,
                })
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                short = row[0][:12] if row and isinstance(row[0], str) else "?"
                parse_errors.append(f"node {short}: {e}")
                continue

        confirmation = (
            f"Migration history: {len(records)} migration(s) recorded "
            + ("" if not parse_errors
               else f"({len(parse_errors)} record(s) had unreadable properties)")
        ).strip()
        data = {
            "total_migrations": len(records),
            "records": records,
            "parse_errors": parse_errors,
        }

        # Honesty: some rows had unparseable properties. We report
        # the readable ones but the full count of recorded migrations
        # may be higher than ``total_migrations`` suggests. PARTIAL
        # forces the LLM to surface the gap.
        if parse_errors:
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    f"{len(parse_errors)} migration record(s) had "
                    "unparseable properties and were skipped: "
                    + "; ".join(parse_errors)
                ),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)
