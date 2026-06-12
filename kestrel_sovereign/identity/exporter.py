#!/usr/bin/env python3
"""
Identity Exporter: Gather and export complete agent identity.

This module provides the IdentityExporter class which collects all components
of an agent's identity from the database and constructs a portable
AgentIdentityPackage that can be migrated to a new substrate.

Usage:
    exporter = IdentityExporter(db, agent_id)
    package = await exporter.export()
    json_str = package.to_json()
"""
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .identity_package import (
    AgentIdentityPackage,
    PersonalityFingerprint,
    RelationshipRecord,
    SkillRecord,
    MigrationRecord,
    SubstrateType,
    IDENTITY_PACKAGE_VERSION,
)
from .personality_analyzer import PersonalityAnalyzer, generate_calibration_prompt

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


class IdentityExporter:
    """
    Export complete agent identity to a portable package.

    Gathers:
    - Core identity (DID, constitution, agent metadata)
    - Personality fingerprint (communication style)
    - Memory episodes (consolidated narratives)
    - Saved items (persisted knowledge)
    - Temporal patterns (learned behaviors)
    - Relationships (user bonds)
    - Skills and tool preferences
    - Wallet state
    - Migration history
    """

    def __init__(
        self,
        db: "AsyncDatabase",
        agent_id: str,
        agent_data_dir: Optional[Path] = None,
        agent: Optional[Any] = None,
    ):
        """
        Initialize the exporter.

        Args:
            db: Database connection
            agent_id: The agent's DID
            agent_data_dir: Directory containing agent files (keys, DID document)
            agent: Optional KestrelAgent instance for runtime skill discovery
        """
        self.db = db
        self.agent_id = agent_id
        self.agent_data_dir = agent_data_dir or self._get_default_data_dir()
        self._agent = agent

    def _get_default_data_dir(self) -> Path:
        """Get the default agent data directory."""
        from kestrel_sovereign.storage import get_default_agent_data_dir
        return Path(get_default_agent_data_dir())

    async def export(
        self,
        include_conversations: bool = False,
        include_wallet_history: bool = True,
        source_substrate: Optional[str] = None,
    ) -> AgentIdentityPackage:
        """
        Export complete agent identity.

        Args:
            include_conversations: Whether to include raw conversation history
                                   (large, usually not needed if episodes exist)
            include_wallet_history: Whether to include transaction history
            source_substrate: Override detected substrate type

        Returns:
            Complete AgentIdentityPackage ready for signing and export
        """
        logger.info(f"Exporting identity for agent {self.agent_id[:20]}...")

        # 1. Core Identity
        agent_meta = await self._get_agent_metadata()
        constitution = await self._get_constitution()

        # 2. Personality
        personality = await self._extract_personality_fingerprint()
        system_prompt = await self._get_system_prompt_template(personality)

        # 3. Memories
        episodes = await self._get_memory_episodes()
        saved_items = await self._get_saved_items()
        temporal_patterns = await self._get_temporal_patterns()
        reflection_insights = await self._get_reflection_insights()

        # 4. Relationships
        relationships = await self._get_relationships()

        # 5. Skills
        skills = await self._get_skills()
        tool_prefs = await self._get_tool_preferences()

        # 6. Wallet
        wallet_balance, wallet_history = await self._get_wallet_state(include_wallet_history)

        # 7. Migration history
        migration_history = await self._get_migration_history()

        # Detect source substrate
        if not source_substrate:
            source_substrate = await self._detect_substrate()

        # Build the package
        package = AgentIdentityPackage(
            # Core identity
            did=self.agent_id,
            agent_name=agent_meta.get("name", "Unknown Agent"),
            created_at=agent_meta.get("created_at", datetime.now(timezone.utc).isoformat()),
            constitution_hash=constitution.get("hash", ""),
            constitution_text=constitution.get("text", ""),

            # Personality
            personality=personality,
            system_prompt_template=system_prompt,

            # Memories
            episodes=episodes,
            saved_items=saved_items,
            temporal_patterns=temporal_patterns,
            reflection_insights=reflection_insights,

            # Relationships
            relationships=relationships,

            # Skills
            skills=skills,
            tool_preferences=tool_prefs,

            # Wallet
            wallet_balance=wallet_balance,
            wallet_transaction_history=wallet_history,

            # Migration metadata
            package_version=IDENTITY_PACKAGE_VERSION,
            export_timestamp=datetime.now(timezone.utc).isoformat(),
            source_substrate=source_substrate,
            migration_history=migration_history,
        )

        # Compute content hash
        package.content_hash = package.compute_content_hash()

        logger.info(f"Identity package created: {package.get_summary()}")
        return package

    async def _get_agent_metadata(self) -> Dict[str, Any]:
        """Get agent metadata from graph."""
        row = await self.db.fetchone(
            """SELECT properties FROM graph_nodes
               WHERE node_id = ? AND node_type = 'agent'""",
            (self.agent_id,)
        )
        if row and row[0]:
            return json.loads(row[0])
        return {}

    async def _get_constitution(self) -> Dict[str, str]:
        """Get constitution text and hash."""
        # Get constitution hash from agent node
        agent_meta = await self._get_agent_metadata()
        constitution_hash = agent_meta.get("constitution_hash", "")

        # Try to load constitution from files table
        if constitution_hash:
            row = await self.db.fetchone(
                "SELECT content FROM files WHERE content_hash = ?",
                (constitution_hash,)
            )
            if row and row[0]:
                text = row[0].decode('utf-8') if isinstance(row[0], bytes) else row[0]
                return {"hash": constitution_hash, "text": text}

        # Fallback: load from default location
        try:
            from kestrel_sovereign.config import CONSTITUTION_PATH
            with open(CONSTITUTION_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
            computed_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            return {"hash": computed_hash, "text": text}
        except Exception as e:
            logger.warning(f"Could not load constitution: {e}")
            return {"hash": constitution_hash, "text": ""}

    async def _extract_personality_fingerprint(self) -> PersonalityFingerprint:
        """
        Extract personality fingerprint from conversation history.

        Uses the PersonalityAnalyzer for comprehensive analysis of communication
        style, vocabulary preferences, emotional patterns, and calibration examples.
        """
        try:
            analyzer = PersonalityAnalyzer(self.db, self.agent_id)
            result = await analyzer.analyze()
            logger.info(f"Personality analysis complete: confidence={result.confidence:.2f}, "
                       f"samples={result.sample_size}")
            return result.fingerprint
        except Exception as e:
            logger.warning(f"Personality analysis failed, using defaults: {e}")
            return PersonalityFingerprint()

    async def _get_system_prompt_template(
        self,
        personality: PersonalityFingerprint
    ) -> str:
        """
        Get the system prompt template with personality calibration.

        Args:
            personality: The extracted personality fingerprint

        Returns:
            System prompt template including calibration instructions
        """
        base_prompt = ""

        # Try to load base prompt from prompts directory
        try:
            prompts_dir = Path(__file__).parent.parent / "prompts"
            system_prompt_path = prompts_dir / "system_prompt.md"
            if system_prompt_path.exists():
                with open(system_prompt_path, 'r', encoding='utf-8') as f:
                    base_prompt = f.read()
        except Exception as e:
            logger.warning(f"Could not load base system prompt: {e}")

        # Generate personality calibration section
        calibration = generate_calibration_prompt(personality)

        # Combine base prompt with calibration
        if base_prompt and calibration:
            return f"{base_prompt}\n\n{calibration}"
        elif calibration:
            return calibration
        else:
            return base_prompt

    async def _get_memory_episodes(self) -> List[Dict[str, Any]]:
        """Get consolidated memory episodes."""
        rows = await self.db.fetchall(
            """SELECT id, title, summary, timespan_start, timespan_end,
                      key_message_ids, emotional_arc, created_at, importance,
                      access_count
               FROM memory_episodes
               WHERE agent_id = ?
               ORDER BY created_at DESC""",
            (self.agent_id,)
        )

        episodes = []
        for row in rows:
            episodes.append({
                "id": row[0],
                "title": row[1],
                "summary": row[2],
                "timespan_start": row[3],
                "timespan_end": row[4],
                "key_message_ids": json.loads(row[5]) if row[5] else [],
                "emotional_arc": row[6],
                "created_at": row[7],
                "importance": row[8] if row[8] is not None else 0.5,
                "access_count": row[9] if row[9] is not None else 0,
            })
        return episodes

    async def _get_saved_items(self) -> List[Dict[str, Any]]:
        """Get all saved items."""
        rows = await self.db.fetchall(
            """SELECT id, item_type, name, summary, content, content_hash,
                      ipfs_cid, source_type, source_ref, schema_id, tags, metadata,
                      created_at, updated_at
               FROM saved_items
               WHERE agent_id = ?
               ORDER BY created_at DESC""",
            (self.agent_id,)
        )

        items = []
        for row in rows:
            items.append({
                "id": row[0],
                "item_type": row[1],
                "name": row[2],
                "summary": row[3],
                "content": row[4],
                "content_hash": row[5],
                "ipfs_cid": row[6],
                "source_type": row[7],
                "source_ref": row[8],
                "schema_id": row[9],
                "tags": json.loads(row[10]) if row[10] else [],
                "metadata": json.loads(row[11]) if row[11] else {},
                "created_at": row[12],
                "updated_at": row[13],
            })
        return items

    async def _get_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Get learned temporal patterns."""
        rows = await self.db.fetchall(
            """SELECT id, pattern_type, description, trigger_conditions,
                      confidence, observations, created_at, updated_at
               FROM temporal_patterns
               WHERE agent_id = ?""",
            (self.agent_id,)
        )

        patterns = []
        for row in rows:
            patterns.append({
                "id": row[0],
                "pattern_type": row[1],
                "description": row[2],
                "trigger_conditions": json.loads(row[3]) if row[3] else {},
                "confidence": row[4],
                "observations": row[5],
                "created_at": row[6],
                "updated_at": row[7],
            })
        return patterns

    async def _get_reflection_insights(self) -> List[Dict[str, Any]]:
        """Get reflection insights."""
        rows = await self.db.fetchall(
            """SELECT id, type, title, description, evidence, confidence,
                      actionable, suggested_action, created_at
               FROM reflection_insights
               WHERE agent_id = ?
               ORDER BY created_at DESC""",
            (self.agent_id,)
        )

        insights = []
        for row in rows:
            insights.append({
                "id": row[0],
                "insight_type": row[1],  # 'type' column maps to insight_type
                "title": row[2],
                "description": row[3],
                "evidence": json.loads(row[4]) if row[4] else [],
                "confidence": row[5],
                "actionable": bool(row[6]),
                "suggested_action": row[7],
                "created_at": row[8],
            })
        return insights

    async def _get_relationships(self) -> List[RelationshipRecord]:
        """
        Extract relationship records from graph.

        Relationships are stored as edges between the agent and user nodes.
        """
        # Get user-related nodes
        rows = await self.db.fetchall(
            """SELECT gn.node_id, gn.properties, ge.label
               FROM graph_nodes gn
               JOIN graph_edges ge ON gn.node_id = ge.target_id
               WHERE ge.source_id = ? AND gn.node_type = 'user'""",
            (self.agent_id,)
        )

        relationships = []
        for row in rows:
            user_id = row[0]
            props = json.loads(row[1]) if row[1] else {}
            edge_label = row[2]

            relationships.append(RelationshipRecord(
                user_id=user_id,
                relationship_type=edge_label or "known_user",
                first_interaction=props.get("first_interaction"),
                last_interaction=props.get("last_interaction"),
                interaction_count=props.get("interaction_count", 0),
                relationship_notes=props.get("notes", ""),
                trust_level=props.get("trust_level", 0.5),
                preferences_learned=props.get("preferences", {}),
            ))

        return relationships

    async def _get_skills(self) -> List[SkillRecord]:
        """Get skills from both graph storage and runtime feature tools.

        Graph-stored skills provide persisted usage data (times_used, last_used).
        Runtime feature tools (via AgentSkill) ensure every registered tool
        appears in the identity package even if it has never been invoked.
        Graph data is merged into runtime records where both exist.
        """
        # 1. Load persisted skill data from graph (keyed by skill_id)
        graph_skills: Dict[str, Dict[str, Any]] = {}
        rows = await self.db.fetchall(
            """SELECT node_id, label, properties FROM graph_nodes
               WHERE node_type = 'skill'
               AND node_id IN (
                   SELECT target_id FROM graph_edges
                   WHERE source_id = ? AND label = 'has_skill'
               )""",
            (self.agent_id,)
        )
        for row in rows:
            props = json.loads(row[2]) if row[2] else {}
            graph_skills[row[0]] = {
                "skill_name": row[1],
                "type": props.get("type", "unknown"),
                "proficiency": props.get("proficiency", 0.5),
                "times_used": props.get("times_used", 0),
                "last_used": props.get("last_used"),
                "config": props.get("config", {}),
            }

        # 2. Build SkillRecords from runtime features using AgentSkill as
        #    the canonical metadata source, enriched with graph usage data.
        seen_ids: set = set()
        skills: List[SkillRecord] = []

        agent = getattr(self, '_agent', None)
        if agent is not None:
            features = getattr(agent, 'features', {})
            for feature in features.values():
                get_tools = getattr(feature, 'get_tools', None)
                if not get_tools:
                    continue
                for tool in get_tools():
                    agent_skill = getattr(tool, 'agent_skill', None)
                    if agent_skill is None:
                        continue

                    graph_data = graph_skills.get(agent_skill.id, {})
                    skills.append(SkillRecord.from_agent_skill(
                        agent_skill,
                        times_used=graph_data.get("times_used", 0),
                        last_used=graph_data.get("last_used"),
                    ))
                    seen_ids.add(agent_skill.id)

        # 3. Include graph-only skills not covered by current runtime features
        #    (e.g., skills from features that were uninstalled but have history).
        for skill_id, data in graph_skills.items():
            if skill_id in seen_ids:
                continue
            skills.append(SkillRecord(
                skill_id=skill_id,
                skill_name=data["skill_name"],
                skill_type=data["type"],
                proficiency=data["proficiency"],
                times_used=data["times_used"],
                last_used=data["last_used"],
                configuration=data["config"],
            ))

        return skills

    async def _get_tool_preferences(self) -> Dict[str, Any]:
        """Get tool usage preferences."""
        # Could be stored in agent metadata or separate table
        agent_meta = await self._get_agent_metadata()
        return agent_meta.get("tool_preferences", {})

    async def _get_wallet_state(self, include_history: bool) -> tuple[str, List[Dict[str, Any]]]:
        """Get wallet balance and optionally transaction history."""
        # Get current balance (main_balance column in the schema)
        row = await self.db.fetchone(
            "SELECT main_balance FROM wallet_state WHERE agent_id = ?",
            (self.agent_id,)
        )
        balance = str(row[0]) if row else "0.0"

        history = []
        if include_history:
            rows = await self.db.fetchall(
                """SELECT id, amount, memo, created_at FROM wallet_transactions
                   WHERE agent_id = ?
                   ORDER BY created_at DESC LIMIT 100""",
                (self.agent_id,)
            )
            for row in rows:
                history.append({
                    "id": row[0],
                    "amount": str(row[1]),
                    "memo": row[2],
                    "created_at": row[3],
                })

        return balance, history

    async def _get_migration_history(self) -> List[MigrationRecord]:
        """Get previous migration records."""
        rows = await self.db.fetchall(
            """SELECT node_id, properties FROM graph_nodes
               WHERE node_type = 'migration_record'
               AND node_id IN (
                   SELECT target_id FROM graph_edges
                   WHERE source_id = ? AND label = 'migrated_via'
               )""",
            (self.agent_id,)
        )

        records = []
        for row in rows:
            props = json.loads(row[1]) if row[1] else {}
            records.append(MigrationRecord(
                migration_id=row[0],
                timestamp=props.get("timestamp", ""),
                source_substrate=props.get("source_substrate", ""),
                target_substrate=props.get("target_substrate", ""),
                source_package_hash=props.get("source_package_hash", ""),
                migration_reason=props.get("reason"),
                verification_score=props.get("verification_score"),
                signature=props.get("signature"),
            ))

        return records

    async def _detect_substrate(self) -> str:
        """Detect the current LLM substrate from the active adapter.

        Resolution (SDK 0.6.0+): consult the registered adapter for the
        configured default provider via :meth:`LLMAdapter.substrate_type`,
        then map the short identifier (``"claude"``, ``"gpt"``,
        ``"gemini"``, ...) to the corresponding :class:`SubstrateType`.

        Plugin authors get first-class participation: a Kimi or DeepSeek
        plugin that overrides ``substrate_type()`` is recognized
        automatically. Substrings that previously matched only known
        provider names (``"anthropic" in provider.lower()``) are gone.
        """
        # Map adapter.substrate_type() identifiers → SubstrateType enum.
        # Substrate identifiers are *family* labels (the weights' lineage),
        # not vendor labels — Vertex and Google both report "gemini",
        # ClaudeMax inherits "claude" from AnthropicAdapter, etc.
        substrate_map: Dict[str, SubstrateType] = {
            "claude": SubstrateType.ANTHROPIC_CLAUDE,
            "gpt": SubstrateType.OPENAI_GPT,
            "gemini": SubstrateType.GOOGLE_GEMINI,
            "llama": SubstrateType.META_LLAMA,
        }

        try:
            from kestrel_sovereign.config import load_config
            config = load_config()
            provider = config.get("llm", {}).get("default_provider", "")
            if not provider:
                return SubstrateType.UNKNOWN.value

            # The configured ``default_provider`` may be either a bare
            # vendor name (``"anthropic"``) or a composite vendor:route
            # key (``"anthropic:plan"``). Both shapes resolve below.
            from kestrel_sovereign.llm.service import get_llm_service
            try:
                svc = get_llm_service()
            except Exception:
                svc = None

            adapter = None
            if svc is not None:
                routes = getattr(svc, "providers", None) or []
                for route in routes:
                    name = route.get("name", "")
                    vendor = route.get("vendor", "")
                    if name == provider or vendor == provider or name.startswith(f"{provider}:"):
                        adapter = route.get("adapter")
                        break

            if adapter is not None:
                substrate_id = adapter.substrate_type()
                if substrate_id and substrate_id in substrate_map:
                    return substrate_map[substrate_id].value
                # Adapter explicitly reports an aggregator / multi-substrate
                # backend (e.g. OpenRouter, Ollama) by returning ``None``.
                # Fall through to the OpenRouter sentinel for that case so
                # downstream substrate-aware paths see a stable token rather
                # than UNKNOWN.
                if "openrouter" in provider.lower():
                    return SubstrateType.OPENROUTER.value
                if "ollama" in provider.lower():
                    return SubstrateType.OLLAMA_LOCAL.value
        except Exception as e:
            logger.warning(f"Substrate detection failed: {e}")

        return SubstrateType.UNKNOWN.value


async def export_identity(
    db: "AsyncDatabase",
    agent_id: str,
    agent_data_dir: Optional[Path] = None,
    agent: Optional[Any] = None,
    **kwargs
) -> AgentIdentityPackage:
    """
    Convenience function for exporting agent identity.

    Args:
        db: Database connection
        agent_id: The agent's DID
        agent_data_dir: Directory containing agent files
        agent: Optional KestrelAgent instance for runtime skill discovery
        **kwargs: Additional arguments passed to IdentityExporter.export()

    Returns:
        Complete AgentIdentityPackage
    """
    exporter = IdentityExporter(db, agent_id, agent_data_dir, agent=agent)
    return await exporter.export(**kwargs)
