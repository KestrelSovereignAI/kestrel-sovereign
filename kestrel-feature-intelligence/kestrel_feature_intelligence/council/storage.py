"""
Council Session Storage

Stores council sessions in the knowledge graph for:
- Permanent audit trail
- Historical decision lookup
- Evidence linking
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import CouncilSession, Evidence

logger = logging.getLogger(__name__)

# Default storage location
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "council_sessions"


class CouncilStorage:
    """Storage interface for council sessions."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        graph_store: Optional[Any] = None,
    ):
        """
        Initialize council storage.

        Args:
            data_dir: Directory for JSON session files
            graph_store: Optional knowledge graph store for richer querying
        """
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.graph_store = graph_store

    async def save_session(self, session: CouncilSession) -> str:
        """
        Save a council session.

        Returns:
            Session ID
        """
        # Save to JSON file
        session_file = self.data_dir / f"{session.id}.json"
        session_data = session.to_dict()

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=str)

        logger.info(f"Saved council session {session.id} to {session_file}")

        # Save transcript as markdown
        transcript_file = self.data_dir / f"{session.id}.md"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(session.to_transcript())

        # Optionally save to knowledge graph
        if self.graph_store:
            await self._save_to_graph(session)

        return session.id

    async def load_session(self, session_id: str) -> Optional[CouncilSession]:
        """Load a council session by ID."""
        session_file = self.data_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file, encoding="utf-8") as f:
                data = json.load(f)
            return self._session_from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def list_sessions(
        self,
        limit: int = 20,
        outcome: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List recent council sessions.

        Args:
            limit: Maximum sessions to return
            outcome: Filter by outcome (APPROVED, REJECTED, DEADLOCK)

        Returns:
            List of session summaries
        """
        sessions = []

        for session_file in sorted(
            self.data_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit * 2]:  # Load extra in case of filtering
            try:
                with open(session_file, encoding="utf-8") as f:
                    data = json.load(f)

                if outcome and data.get("outcome") != outcome:
                    continue

                sessions.append({
                    "id": data.get("id"),
                    "question": data.get("question", "")[:100],
                    "outcome": data.get("outcome"),
                    "created_at": data.get("created_at"),
                    "completed_at": data.get("completed_at"),
                    "member_count": len(data.get("members", [])),
                    "verdict_count": len(data.get("verdicts", [])),
                })

                if len(sessions) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Could not read {session_file}: {e}")

        return sessions

    async def get_previous_decisions(
        self,
        limit: int = 5,
        approved_only: bool = False,
    ) -> List[str]:
        """
        Get summaries of previous council decisions.

        Used for evidence compilation.
        """
        decisions = []

        filter_outcome = "APPROVED" if approved_only else None
        sessions = await self.list_sessions(limit=limit, outcome=filter_outcome)

        for s in sessions:
            decisions.append(
                f"{s['created_at']}: {s['question']} -> {s['outcome']}"
            )

        return decisions

    async def _save_to_graph(self, session: CouncilSession) -> None:
        """Save session to knowledge graph."""
        try:
            # Create session node
            await self.graph_store.add_node(
                node_id=f"council_session:{session.id}",
                node_type="council_session",
                properties={
                    "question": session.question,
                    "outcome": session.outcome.value,
                    "consensus_rule": session.consensus_rule.value,
                    "created_at": session.created_at.isoformat(),
                    "completed_at": (
                        session.completed_at.isoformat()
                        if session.completed_at else None
                    ),
                    "member_count": len(session.members),
                    "round_count": len(session.rounds),
                    "human_override": session.human_override,
                },
            )

            # Create verdict nodes and link to session
            for verdict in session.verdicts:
                verdict_id = f"verdict:{session.id}:{verdict.member_name}"
                await self.graph_store.add_node(
                    node_id=verdict_id,
                    node_type="council_verdict",
                    properties={
                        "member_name": verdict.member_name,
                        "model": verdict.model,
                        "decision": verdict.decision.value,
                        "confidence": verdict.confidence,
                        "reasoning": verdict.reasoning[:500],  # Truncate
                        "concerns": verdict.concerns,
                        "conditions": verdict.conditions,
                    },
                )
                await self.graph_store.add_edge(
                    source_id=verdict_id,
                    target_id=f"council_session:{session.id}",
                    edge_type="verdict_for",
                )

            # Link to evidence if available
            if session.evidence:
                evidence_id = f"evidence:{session.evidence.content_hash()}"
                await self.graph_store.add_node(
                    node_id=evidence_id,
                    node_type="council_evidence",
                    properties={
                        "target": session.evidence.target,
                        "test_count": session.evidence.test_count,
                        "test_passed": session.evidence.test_passed,
                        "risk_count": len(session.evidence.risks),
                        "compiled_at": session.evidence.compiled_at.isoformat(),
                    },
                )
                await self.graph_store.add_edge(
                    source_id=f"council_session:{session.id}",
                    target_id=evidence_id,
                    edge_type="reviewed_evidence",
                )

            logger.info(f"Saved session {session.id} to knowledge graph")

        except Exception as e:
            logger.error(f"Failed to save session to graph: {e}")

    def _session_from_dict(self, data: Dict[str, Any]) -> CouncilSession:
        """Reconstruct a CouncilSession from dictionary."""
        from .models import (
            CouncilMember,
            ConsensusRule,
            Decision,
            DeliberationMessage,
            DeliberationRound,
            SessionOutcome,
            Verdict,
        )

        members = [
            CouncilMember(
                name=m["name"],
                provider=m["provider"],
                model=m["model"],
                role=m.get("role", "reviewer"),
            )
            for m in data.get("members", [])
        ]

        rounds = []
        for r in data.get("rounds", []):
            dr = DeliberationRound(
                round_number=r["round_number"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
            )
            for m in r.get("messages", []):
                dr.messages.append(DeliberationMessage(
                    member_name=m["member_name"],
                    model=m["model"],
                    content=m["content"],
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                ))
            rounds.append(dr)

        verdicts = []
        for v in data.get("verdicts", []):
            verdicts.append(Verdict(
                member_name=v["member_name"],
                model=v["model"],
                decision=Decision(v["decision"]),
                confidence=v["confidence"],
                reasoning=v["reasoning"],
                concerns=v.get("concerns", []),
                conditions=v.get("conditions", []),
                dissent=v.get("dissent"),
                timestamp=datetime.fromisoformat(v["timestamp"]),
            ))

        session = CouncilSession(
            id=data["id"],
            question=data["question"],
            members=members,
            rounds=rounds,
            verdicts=verdicts,
            outcome=SessionOutcome(data["outcome"]),
            consensus_rule=ConsensusRule(data["consensus_rule"]),
            human_override=data.get("human_override"),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at") else None
            ),
        )

        return session


# Convenience function for default storage
_default_storage: Optional[CouncilStorage] = None


def get_storage(
    data_dir: Optional[Path] = None,
    graph_store: Optional[Any] = None
) -> CouncilStorage:
    """
    Get the default council storage instance.

    Args:
        data_dir: Directory for JSON session files (only used on first call)
        graph_store: Optional knowledge graph store (only used on first call)

    Returns:
        CouncilStorage instance
    """
    global _default_storage
    if _default_storage is None:
        _default_storage = CouncilStorage(data_dir=data_dir, graph_store=graph_store)
    else:
        # Warn if trying to initialize with different params
        normalized_data_dir = data_dir or DATA_DIR
        if (_default_storage.data_dir != normalized_data_dir or
            _default_storage.graph_store != graph_store):
            logger.warning(
                f"Attempted to re-initialize council storage with different params. "
                f"Existing: data_dir={_default_storage.data_dir}, graph_store={_default_storage.graph_store}. "
                f"Requested: data_dir={normalized_data_dir}, graph_store={graph_store}. "
                f"Ignoring new params and returning existing instance."
            )
    return _default_storage


def reset_storage() -> None:
    """
    Reset the default council storage singleton.

    This is primarily for testing purposes to allow re-initialization
    with different parameters.
    """
    global _default_storage
    _default_storage = None
