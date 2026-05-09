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
2. Atomic claim via ``os.link()`` — first concurrent writer to hardlink a
   ``.claim`` file wins; the second gets ``FileExistsError``.
3. Non-overwriting finalization via ``os.link()`` to the final path — if the
   target file already exists, the save fails rather than silently overwriting.

Stale claim files (orphaned by a crashed process) are automatically reclaimed
after ``CLAIM_STALENESS_SECONDS`` (default 60 s).

Per the Incubator Principle, extraction is never automatic — it is
triggered by an explicit tool call. Governed agents without this feature
installed cannot extract skills at all.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from .models import Skill, normalize_title, skill_id_from_title

logger = logging.getLogger(__name__)


SKILL_NODE_TYPE = "skill"
SKILLS_SUBDIR = "skills"

# Default filter for what counts as a "non-obvious discovery" worth extracting.
MIN_CANDIDATE_CONFIDENCE = 0.7
CANDIDATE_INSIGHT_TYPES = ("pattern", "improvement")

# Claim files older than this are assumed to be from a crashed process and
# eligible for reclamation.  See issue #667 item 4.
CLAIM_STALENESS_SECONDS = 60


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
    async def skill_list(self) -> ToolResult:
        """List all skills this agent has extracted."""
        skills = await self._load_all_skills()
        rows = [
            {
                "id": s.id,
                "title": s.title,
                "tags": s.tags,
                "confidence": s.confidence,
                "created_at": s.created_at,
            }
            for s in skills
        ]
        if not rows:
            return ToolResult.ok(
                "No extracted skills yet.",
                data={"skills": [], "count": 0},
            )
        # Render the title list inside the confirmation so the !skill list
        # CLI surface stays useful — the command-handler envelope formatter
        # appends ``data`` only when it carries structural payload (a list
        # value on a dict triggers the heuristic), so the JSON block does
        # render here, but a one-line summary up top makes the LLM-facing
        # confirmation honest about what's there.
        titles = ", ".join(r["title"] for r in rows)
        return ToolResult.ok(
            f"{len(rows)} skill(s): {titles}",
            data={"skills": rows, "count": len(rows)},
        )

    @tool(
        "skill_show",
        "Show the full content of an extracted skill",
        category=ToolCategory.UTILITY,
        command_prefix="!skill show",
    )
    async def skill_show(self, skill_id: str) -> ToolResult:
        """Show a skill's full content by ID."""
        skill = await self._load_skill(skill_id)
        if skill is None:
            return ToolResult.failed(error=f"Skill {skill_id} not found")
        # Render the skill body inside the confirmation so !skill show
        # surfaces what the user came for. ``data`` keeps the structured
        # form for programmatic callers.
        body = skill.to_markdown()
        return ToolResult.ok(
            f"Skill {skill_id} — {skill.title}:\n{body}",
            data={"skill": skill.to_dict()},
        )

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
    ) -> ToolResult:
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
            return ToolResult.failed(
                error=f"min_confidence must be numeric, got {min_confidence!r}"
            )
        if not (0.0 <= min_confidence <= 1.0):
            return ToolResult.failed(error="min_confidence must be in [0.0, 1.0]")

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(error=f"limit must be an integer, got {limit!r}")
        if limit < 1 or limit > 500:
            return ToolResult.failed(error="limit must be in [1, 500]")

        if not self._db:
            return ToolResult.failed(error="Database not available")

        try:
            candidates = await self._fetch_candidates(min_confidence=min_confidence, limit=limit)
        except Exception as e:
            logger.error("Candidate query failed: %s", e)
            return ToolResult.failed(error=str(e))

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

        return ToolResult.ok(
            f"Found {len(out)} candidate insight(s) for skill extraction.",
            data={"candidates": out, "count": len(out)},
        )

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
    ) -> ToolResult:
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
            return ToolResult.failed(error="Database not available")

        try:
            steps = json.loads(steps_json)
            if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
                return ToolResult.failed(error="steps_json must be a JSON array of strings")
            if len(steps) == 0:
                return ToolResult.failed(error="at least one step is required")

            tags = json.loads(tags_json)
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                return ToolResult.failed(error="tags_json must be a JSON array of strings")
        except json.JSONDecodeError as e:
            return ToolResult.failed(error=f"Invalid JSON: {e}")

        insight = await self._fetch_insight(insight_id)
        if insight is None:
            return ToolResult.failed(error=f"Insight {insight_id} not found")

        title = (insight.get("suggested_action") or "").strip()
        trigger = (insight.get("description") or title).strip()
        if not title:
            return ToolResult.failed(error="Insight has no suggested_action")

        norm = normalize_title(title)
        existing = await self._existing_normalized_titles()
        if norm in existing:
            return ToolResult.failed(error="A skill with a similar title already exists")

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
            return ToolResult.failed(error=str(e))

        return ToolResult.ok(
            f"Saved skill '{skill.title}' (id={skill.id}).",
            data={
                "skill_id": skill.id,
                "title": skill.title,
                "path": str(self._skill_path(skill.id)) if self._skills_dir else None,
            },
        )

    @tool(
        "skill_delete",
        "Remove a saved skill (file + graph node)",
        category=ToolCategory.UTILITY,
        command_prefix="!skill delete",
    )
    async def skill_delete(self, skill_id: str) -> ToolResult:
        """Remove a skill from disk and the graph.

        Returns:
            ToolResult.ok when every layer that *could* apply did its
            job (both file + graph deleted, or only one layer was
            available and it succeeded); PARTIAL when both layers
            were available but only one succeeded — the asymmetric
            outcome must be spoken because a stale graph node still
            surfaces in associative recall and a stale file
            resurrects on the next list/save; ERROR when neither
            layer found anything to remove.
        """
        removed_file = False
        removed_node = False
        file_attempted = self._skills_dir is not None
        graph_attempted = False

        if file_attempted:
            path = self._skill_path(skill_id)
            if path.exists():
                try:
                    path.unlink()
                    removed_file = True
                except OSError as e:
                    logger.warning("Could not remove skill file %s: %s", path, e)

        storage = getattr(self.agent, "storage", None)
        if storage is not None and hasattr(storage, "delete_node"):
            graph_attempted = True
            try:
                await storage.delete_node(skill_id)
                removed_node = True
            except Exception as e:
                logger.warning("Could not remove skill node %s: %s", skill_id, e)

        if not (removed_file or removed_node):
            return ToolResult.failed(error=f"Skill {skill_id} not found")

        data = {
            "skill_id": skill_id,
            "removed_file": removed_file,
            "removed_node": removed_node,
        }
        # Asymmetric outcomes only matter when *both* layers were
        # available to try. If only one layer applies (graph-only
        # agent, or no graph backend), the single-layer outcome is
        # symmetric by construction — no PARTIAL needed.
        # File is the source of truth — if file went but graph didn't,
        # the skill is "deleted" semantically but a stale graph node
        # may still surface in recall. If graph went but file didn't,
        # the next save targeting this id will hit FileExistsError.
        if file_attempted and graph_attempted and removed_file != removed_node:
            if removed_file and not removed_node:
                caveat = (
                    f"file removed but graph node {skill_id} could not be "
                    "deleted — it will still surface in associative recall "
                    "until the graph is reachable again."
                )
            else:
                # Only safe to call _skill_path here because file_attempted
                # implies _skills_dir is not None (codex round 1 of #1130).
                caveat = (
                    f"graph node removed but file at {self._skill_path(skill_id)} "
                    "could not be unlinked — the skill will reload on next "
                    "list/save attempt."
                )
            return ToolResult.partial(
                f"Skill {skill_id} partially removed.",
                caveat,
                data=data,
            )
        return ToolResult.ok(
            f"Skill {skill_id} removed.",
            data=data,
        )

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

        Concurrency: uses an ``os.link()`` claim-and-swap protocol so that
        two concurrent writers targeting the same skill ID are serialized
        atomically — the first writer to hardlink wins, the second gets
        EEXIST. Both preflight existence check and finalization use
        ``os.link`` (not ``os.replace``), so save fails rather than
        silently overwriting an existing file at any stage.

        Claim ownership is tracked per-writer: a losing writer never deletes
        the winning writer's claim file.  Stale claim files (from a crashed
        process) are reclaimed after ``CLAIM_STALENESS_SECONDS``.
        """
        if self._skills_dir is None:
            raise RuntimeError("No skills directory configured — agent has no data path")

        path = self._skill_path(skill.id)
        if path.exists():
            raise FileExistsError(f"skill file already exists: {path.name}")

        # -- Phase 1: write content to a per-writer tmp file ---------------
        tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
        i_own_claim = False
        claim_path = path.with_suffix(path.suffix + ".claim")

        try:
            tmp_path.write_text(skill.to_markdown(), encoding="utf-8")

            # -- Phase 2: atomic claim via os.link -------------------------
            # First writer to link wins; second gets FileExistsError.
            try:
                os.link(tmp_path, claim_path)
                i_own_claim = True
            except FileExistsError:
                # A claim file exists — check if it's stale (orphaned by a
                # crashed process).
                if self._is_stale_claim(claim_path):
                    logger.info("Reclaiming stale claim file: %s", claim_path)
                    try:
                        claim_path.unlink()
                    except OSError:
                        pass
                    # Retry the link after removing the stale claim.
                    os.link(tmp_path, claim_path)
                    i_own_claim = True
                else:
                    raise FileExistsError(
                        f"concurrent write in progress for {path.name}"
                    )

            # tmp is now redundant (claim holds the same content via hardlink).
            try:
                tmp_path.unlink()
            except OSError:
                pass

            # -- Phase 3: finalize — promote claim to final path -----------
            # Uses os.link so finalization fails if the final file appeared
            # between our preflight check and now (no silent overwrite).
            os.link(claim_path, path)

        except Exception:
            # Only clean up files this writer owns.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            if i_own_claim:
                try:
                    claim_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

        # Claim is no longer needed — the final file is the record.
        if i_own_claim:
            try:
                claim_path.unlink(missing_ok=True)
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

    @staticmethod
    def _is_stale_claim(claim_path: Path) -> bool:
        """Return True if *claim_path* is older than the staleness threshold.

        A claim file left behind by a process that crashed after ``os.link``
        but before finalization would block future saves for that skill_id
        forever.  This check lets the next writer reclaim the slot.
        """
        try:
            age = time.time() - claim_path.stat().st_mtime
        except OSError:
            return False
        return age > CLAIM_STALENESS_SECONDS

    def _agent_id(self) -> str:
        did = getattr(self.agent, "did", None)
        return did or getattr(self.agent, "agent_id", "")
