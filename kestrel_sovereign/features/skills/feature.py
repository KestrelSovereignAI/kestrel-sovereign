"""Skills Feature — extract, store, and recall reusable procedural knowledge.

Inspired by the OB1 "Claudeception" pattern. This feature watches the
reflection feature's insights table for non-obvious discoveries
(actionable patterns/improvements with high confidence) and promotes
them into structured skill files that future sessions can find.

## Storage authority

Skills have **one primary record** and **one best-effort secondary index**:

- **Primary (authoritative): markdown file** at `<agent_data>/skills/<id>.md`.
  Every read operation (`skill_list`, `skill_show`) reads from disk. Loss of
  the file is loss of the skill; loss of the graph node is not. Writes are
  atomic via tmp + rename so crash/interruption cannot leave truncated files.
- **Secondary (best-effort): graph node** of type "skill". Enables associative
  recall when a related concept comes up, but the feature never relies on
  the node's presence for correctness. A failed graph write does not fail
  the save — the file is still the source of truth.

This asymmetry is intentional. Dual-write transactional semantics across a
filesystem and a DB would need real 2PC; we would rather have honest drift
than pretended consistency.

## Uniqueness

Dedup has three layers:
1. Preflight normalized-title check in `skill_extract_candidates` (advisory).
2. Hard collision guard at write time in `skill_save` — if the target file
   already exists, the save fails rather than silently overwriting.
3. ``os.link()`` claim file — the first concurrent writer to hardlink its
   tmp file to ``<id>.md.claim`` wins; the second gets EEXIST. This
   eliminates the last-writer-wins race where two writers both pass the
   ``path.exists()`` check before either has written the final file.

Per the Incubator Principle, extraction is never automatic — it is
triggered by an explicit tool call. Governed agents without this feature
installed cannot extract skills at all.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.tools.base import ToolCategory

from .models import Skill, normalize_title, skill_id_from_title

logger = logging.getLogger(__name__)


SKILL_NODE_TYPE = "skill"
SKILLS_SUBDIR = "skills"

# Default filter for what counts as a "non-obvious discovery" worth extracting.
MIN_CANDIDATE_CONFIDENCE = 0.7
CANDIDATE_INSIGHT_TYPES = ("pattern", "improvement")


class SkillsFeature(Feature):
    """Extract, store, and recall reusable skills from agent work sessions."""

    @property
    def tool_description(self) -> str:
        return (
            "Extract reusable procedural skills from reflection insights and "
            "save them as searchable skill files. Lists, shows, and deletes "
            "skills that survive across sessions."
        )

    async def initialize(self):
        self._db = resolve_feature_database(self.agent)
        self._skills_dir: Optional[Path] = self._resolve_skills_dir()
        if self._skills_dir:
            self._skills_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "SkillsFeature initialized (db=%s, skills_dir=%s)",
            bool(self._db), self._skills_dir,
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(
        "skill_list",
        "List all extracted skills for this agent",
        category=ToolCategory.UTILITY,
        command_prefix="!skill list",
    )
    async def skill_list(self) -> Dict[str, Any]:
        """List all skills this agent has extracted."""
        skills = await self._load_all_skills()
        return {
            "skills": [
                {
                    "id": s.id,
                    "title": s.title,
                    "tags": s.tags,
                    "confidence": s.confidence,
                    "created_at": s.created_at,
                }
                for s in skills
            ],
            "count": len(skills),
        }

    @tool(
        "skill_show",
        "Show the full content of an extracted skill",
        category=ToolCategory.UTILITY,
        command_prefix="!skill show",
    )
    async def skill_show(self, skill_id: str) -> Dict[str, Any]:
        """Show a skill's full content by ID."""
        skill = await self._load_skill(skill_id)
        if skill is None:
            return {"success": False, "error": f"Skill {skill_id} not found"}
        return {"success": True, "skill": skill.to_dict()}

    @tool(
        "skill_extract_candidates",
        "List reflection insights that are candidates for skill extraction",
        category=ToolCategory.UTILITY,
        command_prefix="!skill candidates",
    )
    async def skill_extract_candidates(
        self,
        min_confidence: float = MIN_CANDIDATE_CONFIDENCE,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Find actionable, high-confidence insights that aren't already skills.

        Deduplication: skips insights whose `suggested_action` would normalize
        to the same title as an existing skill (local dir + graph). Dedup is
        a preflight check — the authoritative uniqueness gate is in skill_save
        where a collision on the target file path hard-fails.

        Args:
            min_confidence: Lower bound on insight confidence, in [0.0, 1.0]
            limit: Maximum number of candidates to return, in [1, 500]
        """
        try:
            min_confidence = float(min_confidence)
        except (TypeError, ValueError):
            return {"success": False, "error": f"min_confidence must be numeric, got {min_confidence!r}"}
        if not (0.0 <= min_confidence <= 1.0):
            return {"success": False, "error": "min_confidence must be in [0.0, 1.0]"}

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return {"success": False, "error": f"limit must be an integer, got {limit!r}"}
        if limit < 1 or limit > 500:
            return {"success": False, "error": "limit must be in [1, 500]"}

        if not self._db:
            return {"success": False, "error": "Database not available"}

        try:
            candidates = await self._fetch_candidates(min_confidence=min_confidence, limit=limit)
        except Exception as e:
            logger.error("Candidate query failed: %s", e)
            return {"success": False, "error": str(e)}

        existing_titles = await self._existing_normalized_titles()

        out: List[Dict[str, Any]] = []
        for row in candidates:
            suggested = (row.get("suggested_action") or "").strip()
            if not suggested:
                continue
            norm = normalize_title(suggested)
            if not norm or norm in existing_titles:
                continue
            out.append({
                "insight_id": row["id"],
                "title": suggested,
                "description": row.get("description"),
                "confidence": row.get("confidence"),
                "type": row.get("type"),
                "would_become_skill_id": skill_id_from_title(suggested),
            })

        return {"candidates": out, "count": len(out)}

    @tool(
        "skill_save",
        "Promote a reflection insight into a saved skill",
        category=ToolCategory.UTILITY,
        command_prefix="!skill save",
    )
    async def skill_save(
        self,
        insight_id: str,
        steps_json: str,
        verification: str,
        tags_json: str = "[]",
    ) -> Dict[str, Any]:
        """Promote a candidate insight into a stored skill.

        The LLM typically calls this after reviewing `skill_extract_candidates`
        output. Steps and verification must be supplied because the insight
        itself only captures an observation — the operational detail of how
        to reproduce the fix has to be articulated explicitly.

        Args:
            insight_id: ID of the insight to promote
            steps_json: JSON array of procedural steps
            verification: How to know the skill worked
            tags_json: Optional JSON array of tag strings
        """
        if not self._db:
            return {"success": False, "error": "Database not available"}

        try:
            steps = json.loads(steps_json)
            if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
                return {"success": False, "error": "steps_json must be a JSON array of strings"}
            if len(steps) == 0:
                return {"success": False, "error": "at least one step is required"}

            tags = json.loads(tags_json)
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                return {"success": False, "error": "tags_json must be a JSON array of strings"}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON: {e}"}

        insight = await self._fetch_insight(insight_id)
        if insight is None:
            return {"success": False, "error": f"Insight {insight_id} not found"}

        title = (insight.get("suggested_action") or "").strip()
        trigger = (insight.get("description") or title).strip()
        if not title:
            return {"success": False, "error": "Insight has no suggested_action"}

        norm = normalize_title(title)
        existing = await self._existing_normalized_titles()
        if norm in existing:
            return {"success": False, "error": f"A skill with a similar title already exists"}

        skill = Skill(
            id=skill_id_from_title(title),
            title=title,
            trigger=trigger,
            steps=steps,
            verification=verification.strip() or "Reproduce the trigger scenario and confirm the steps resolve it.",
            tags=tags,
            source_insight_id=insight_id,
            source_session_id=insight.get("session_id"),
            confidence=float(insight.get("confidence", MIN_CANDIDATE_CONFIDENCE)),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            await self._save_skill(skill)
        except Exception as e:
            logger.error("Failed to save skill: %s", e)
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "skill_id": skill.id,
            "title": skill.title,
            "path": str(self._skill_path(skill.id)) if self._skills_dir else None,
        }

    @tool(
        "skill_delete",
        "Remove a saved skill (file + graph node)",
        category=ToolCategory.UTILITY,
        command_prefix="!skill delete",
    )
    async def skill_delete(self, skill_id: str) -> Dict[str, Any]:
        """Remove a skill from disk and the graph."""
        removed_file = False
        removed_node = False

        if self._skills_dir:
            path = self._skill_path(skill_id)
            if path.exists():
                try:
                    path.unlink()
                    removed_file = True
                except OSError as e:
                    logger.warning("Could not remove skill file %s: %s", path, e)

        storage = getattr(self.agent, "storage", None)
        if storage is not None and hasattr(storage, "delete_node"):
            try:
                await storage.delete_node(skill_id)
                removed_node = True
            except Exception as e:
                logger.warning("Could not remove skill node %s: %s", skill_id, e)

        if not (removed_file or removed_node):
            return {"success": False, "error": f"Skill {skill_id} not found"}
        return {
            "success": True,
            "skill_id": skill_id,
            "removed_file": removed_file,
            "removed_node": removed_node,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_skills_dir(self) -> Optional[Path]:
        """Locate the per-agent skills directory."""
        for attr in ("bootstrap_service", "context_builder"):
            svc = getattr(self.agent, attr, None)
            base = getattr(svc, "agent_data_path", None) if svc else None
            if base:
                return Path(base) / SKILLS_SUBDIR
        storage_path = getattr(self.agent, "storage_path", None)
        if storage_path:
            return Path(storage_path).parent / SKILLS_SUBDIR
        return None

    def _skill_path(self, skill_id: str) -> Path:
        assert self._skills_dir is not None
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", skill_id)
        return self._skills_dir / f"{safe}.md"

    async def _fetch_candidates(self, min_confidence: float, limit: int) -> List[Dict[str, Any]]:
        """Query reflection_insights for actionable high-confidence rows."""
        placeholders = ",".join("?" for _ in CANDIDATE_INSIGHT_TYPES)
        rows = await self._db.fetchall(
            f"""
            SELECT id, session_id, type, title, description, confidence,
                   actionable, suggested_action
            FROM reflection_insights
            WHERE agent_id = ?
              AND actionable = 1
              AND confidence >= ?
              AND type IN ({placeholders})
              AND suggested_action IS NOT NULL
              AND suggested_action != ''
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            """,
            (self._agent_id(), min_confidence, *CANDIDATE_INSIGHT_TYPES, limit),
        )
        return [
            {
                "id": r[0],
                "session_id": r[1],
                "type": r[2],
                "title": r[3],
                "description": r[4],
                "confidence": r[5],
                "actionable": bool(r[6]),
                "suggested_action": r[7],
            }
            for r in rows
        ]

    async def _fetch_insight(self, insight_id: str) -> Optional[Dict[str, Any]]:
        row = await self._db.fetchone(
            """
            SELECT id, session_id, type, title, description, confidence,
                   actionable, suggested_action
            FROM reflection_insights
            WHERE agent_id = ? AND id = ?
            """,
            (self._agent_id(), insight_id),
        )
        if row is None:
            return None
        return {
            "id": row[0],
            "session_id": row[1],
            "type": row[2],
            "title": row[3],
            "description": row[4],
            "confidence": row[5],
            "actionable": bool(row[6]),
            "suggested_action": row[7],
        }

    async def _existing_normalized_titles(self) -> set[str]:
        """Collect normalized titles of already-saved skills from disk + graph."""
        titles: set[str] = set()

        if self._skills_dir and self._skills_dir.exists():
            for path in self._skills_dir.glob("*.md"):
                try:
                    skill = Skill.from_markdown(path.read_text(encoding="utf-8"))
                    if skill.title:
                        titles.add(normalize_title(skill.title))
                except Exception as e:
                    logger.debug("Could not parse %s: %s", path, e)

        storage = getattr(self.agent, "storage", None)
        if storage is not None and hasattr(storage, "get_nodes_by_type"):
            try:
                nodes = await storage.get_nodes_by_type(SKILL_NODE_TYPE)
                for node in nodes:
                    label = getattr(node, "label", "") or ""
                    if label:
                        titles.add(normalize_title(label))
            except Exception as e:
                logger.debug("get_nodes_by_type failed: %s", e)

        return titles

    async def _load_all_skills(self) -> List[Skill]:
        """Load all skills from disk (graph-only skills are ignored for listing)."""
        if not self._skills_dir or not self._skills_dir.exists():
            return []
        out: List[Skill] = []
        for path in sorted(self._skills_dir.glob("*.md")):
            try:
                out.append(Skill.from_markdown(path.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning("Failed to load skill from %s: %s", path, e)
        return out

    async def _load_skill(self, skill_id: str) -> Optional[Skill]:
        if not self._skills_dir:
            return None
        path = self._skill_path(skill_id)
        if not path.exists():
            return None
        try:
            return Skill.from_markdown(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load skill %s: %s", skill_id, e)
            return None

    async def _save_skill(self, skill: Skill) -> None:
        """Persist skill to disk atomically, then update the graph.

        File is primary: if the file write fails the save fails. The graph
        node is best-effort — a graph failure is logged but does not fail
        the save, because losing an associative index entry is recoverable
        while losing the skill content is not.

        Concurrency: two writers targeting the same skill ID are serialized
        via an ``os.link()`` claim file. The first writer to hardlink its
        tmp file to ``<id>.md.claim`` wins; the second gets EEXIST and
        receives a structured conflict error. No lock file, no contention
        window — ``os.link()`` is atomic on POSIX.
        """
        if self._skills_dir is None:
            raise RuntimeError("No skills directory configured — agent has no data path")

        path = self._skill_path(skill.id)
        if path.exists():
            raise FileExistsError(f"skill file already exists: {path.name}")

        # Write content to a per-writer tmp file (uuid suffix avoids tmp-
        # file clobber between concurrent writers).
        tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
        claim_path = path.with_suffix(path.suffix + ".claim")
        try:
            tmp_path.write_text(skill.to_markdown(), encoding="utf-8")

            # Claim: hardlink is atomic — second writer raises FileExistsError.
            try:
                os.link(tmp_path, claim_path)
            except FileExistsError:
                raise FileExistsError(
                    f"skill already claimed by concurrent writer: {path.name}"
                )

            # We won the claim — promote to final path.
            os.replace(claim_path, path)
        except Exception:
            # Clean up the tmp file if it exists so we don't accumulate cruft.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                claim_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        finally:
            # The winner's tmp is now a dangling hardlink (same inode as
            # the claim that was renamed to final). Remove it quietly.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "add_node"):
            return
        try:
            from kestrel_sovereign.storage.async_graph_store import GraphNode
            node = GraphNode(
                node_id=skill.id,
                node_type=SKILL_NODE_TYPE,
                label=skill.title,
                properties={
                    "trigger": skill.trigger,
                    "tags": skill.tags,
                    "confidence": skill.confidence,
                    "source_insight_id": skill.source_insight_id,
                    "created_at": skill.created_at,
                },
            )
            await storage.add_node(node)
        except Exception as e:
            # Graph persistence is best-effort — the file is the primary record.
            logger.warning("Could not persist graph node for %s: %s", skill.id, e)

    def _agent_id(self) -> str:
        did = getattr(self.agent, "did", None)
        return did or getattr(self.agent, "agent_id", "")
