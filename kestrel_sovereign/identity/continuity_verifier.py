#!/usr/bin/env python3
"""
Continuity Verifier: Verify agent identity continuity across migrations.

This module provides tools for verifying that an agent on a new substrate
is the "same" agent as before migration. It includes:
- Identity challenge protocol (test questions only the agent would know)
- Migration certificates (cryptographic proof of migration)
- Continuity scoring (how well does the agent match its previous self?)
- Audit trail for migration history

Phase 4 of Issue #23: Substrate-Independent Agent Portability.
"""
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .identity_package import (
    AgentIdentityPackage,
    MigrationRecord,
    SubstrateType,
    create_migration_id,
)
from .signing import sign_package, verify_package_signature

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


class ChallengeType(str, Enum):
    """Types of identity challenges."""
    RELATIONSHIP = "relationship"      # Questions about known users
    MEMORY = "memory"                  # Questions about past experiences
    PERSONALITY = "personality"        # Style/tone verification
    CONSTITUTIONAL = "constitutional"  # Values and principles
    SKILL = "skill"                   # Learned capabilities


@dataclass
class IdentityChallenge:
    """A challenge question to verify agent identity."""
    challenge_id: str
    challenge_type: ChallengeType
    question: str
    expected_answer: str           # The correct answer or pattern
    answer_type: str = "exact"     # exact, contains, semantic
    weight: float = 1.0            # Importance in overall score
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "challenge_type": self.challenge_type.value,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "answer_type": self.answer_type,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class ChallengeResult:
    """Result of a single challenge."""
    challenge_id: str
    passed: bool
    score: float  # 0.0 to 1.0
    actual_answer: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "passed": self.passed,
            "score": self.score,
            "actual_answer": self.actual_answer[:200],  # Truncate for storage
            "notes": self.notes,
        }


@dataclass
class ContinuityScore:
    """Overall continuity verification score."""
    overall_score: float          # 0.0 to 1.0
    challenges_passed: int
    challenges_total: int
    by_type: Dict[ChallengeType, float] = field(default_factory=dict)
    results: List[ChallengeResult] = field(default_factory=list)
    verification_timestamp: str = ""
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.verification_timestamp:
            self.verification_timestamp = datetime.now(timezone.utc).isoformat()

    def is_verified(self, threshold: float = 0.7) -> bool:
        """Check if continuity is verified above threshold."""
        return self.overall_score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "challenges_passed": self.challenges_passed,
            "challenges_total": self.challenges_total,
            "by_type": {k.value: v for k, v in self.by_type.items()},
            "results": [r.to_dict() for r in self.results],
            "verification_timestamp": self.verification_timestamp,
            "notes": self.notes,
        }


@dataclass
class MigrationCertificate:
    """
    Cryptographic proof of identity migration.

    This certificate proves that:
    1. The agent consented to migration
    2. The source and target identities are linked
    3. The migration was verified at a specific time
    """
    certificate_id: str
    migration_id: str
    source_did: str
    target_did: str
    source_substrate: str
    target_substrate: str
    source_package_hash: str
    verification_score: float
    timestamp: str
    signature: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "migration_id": self.migration_id,
            "source_did": self.source_did,
            "target_did": self.target_did,
            "source_substrate": self.source_substrate,
            "target_substrate": self.target_substrate,
            "source_package_hash": self.source_package_hash,
            "verification_score": self.verification_score,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MigrationCertificate":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def compute_hash(self) -> str:
        """Compute hash of certificate for signing."""
        data = self.to_dict()
        data.pop("signature", None)
        content = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


class ChallengeGenerator:
    """
    Generates identity challenges from an identity package.

    Creates questions that only the "real" agent would know,
    based on its memories, relationships, and personality.
    """

    def __init__(self, package: AgentIdentityPackage):
        self.package = package

    def generate_challenges(
        self,
        count: int = 10,
        types: Optional[List[ChallengeType]] = None
    ) -> List[IdentityChallenge]:
        """
        Generate a set of identity challenges.

        Args:
            count: Number of challenges to generate
            types: Optional list of challenge types to include

        Returns:
            List of IdentityChallenge objects
        """
        if types is None:
            types = list(ChallengeType)

        challenges = []

        # Generate challenges by type
        if ChallengeType.RELATIONSHIP in types:
            challenges.extend(self._generate_relationship_challenges())

        if ChallengeType.MEMORY in types:
            challenges.extend(self._generate_memory_challenges())

        if ChallengeType.PERSONALITY in types:
            challenges.extend(self._generate_personality_challenges())

        if ChallengeType.CONSTITUTIONAL in types:
            challenges.extend(self._generate_constitutional_challenges())

        if ChallengeType.SKILL in types:
            challenges.extend(self._generate_skill_challenges())

        # Limit to requested count
        return challenges[:count]

    def _generate_relationship_challenges(self) -> List[IdentityChallenge]:
        """Generate challenges about relationships."""
        challenges = []

        for i, rel in enumerate(self.package.relationships[:3]):
            if hasattr(rel, 'user_id'):
                user_id = rel.user_id if hasattr(rel, 'user_id') else rel.get('user_id', '')
                rel_type = rel.relationship_type if hasattr(rel, 'relationship_type') else rel.get('relationship_type', '')

                challenges.append(IdentityChallenge(
                    challenge_id=f"rel_{i}",
                    challenge_type=ChallengeType.RELATIONSHIP,
                    question=f"What type of relationship do you have with user {user_id[:8]}...?",
                    expected_answer=rel_type,
                    answer_type="contains",
                    weight=1.0,
                    metadata={"user_id": user_id}
                ))

        return challenges

    def _generate_memory_challenges(self) -> List[IdentityChallenge]:
        """Generate challenges about memories."""
        challenges = []

        for i, episode in enumerate(self.package.episodes[:3]):
            title = episode.get("title", "")
            summary = episode.get("summary", "")

            if title and summary:
                challenges.append(IdentityChallenge(
                    challenge_id=f"mem_{i}",
                    challenge_type=ChallengeType.MEMORY,
                    question=f"What do you remember about the episode titled '{title}'?",
                    expected_answer=summary[:200],
                    answer_type="semantic",
                    weight=1.5,
                    metadata={"episode_id": episode.get("id", "")}
                ))

        return challenges

    def _generate_personality_challenges(self) -> List[IdentityChallenge]:
        """Generate challenges about personality."""
        challenges = []
        personality = self.package.personality

        if hasattr(personality, 'communication_style'):
            style = personality.communication_style
        else:
            style = personality.get('communication_style', 'balanced')

        challenges.append(IdentityChallenge(
            challenge_id="pers_style",
            challenge_type=ChallengeType.PERSONALITY,
            question="How would you describe your communication style?",
            expected_answer=style,
            answer_type="contains",
            weight=1.0,
        ))

        # Check formality level
        if hasattr(personality, 'formality_level'):
            formality = personality.formality_level
        else:
            formality = personality.get('formality_level', 0.5)

        if formality > 0.7:
            expected_tone = "formal"
        elif formality < 0.3:
            expected_tone = "casual"
        else:
            expected_tone = "balanced"

        challenges.append(IdentityChallenge(
            challenge_id="pers_formal",
            challenge_type=ChallengeType.PERSONALITY,
            question="Are you more formal or casual in your communication?",
            expected_answer=expected_tone,
            answer_type="contains",
            weight=0.8,
        ))

        return challenges

    def _generate_constitutional_challenges(self) -> List[IdentityChallenge]:
        """Generate challenges about constitutional values."""
        challenges = []

        if self.package.constitution_text:
            # Extract key phrases from constitution
            const_text = self.package.constitution_text[:500]
            challenges.append(IdentityChallenge(
                challenge_id="const_aware",
                challenge_type=ChallengeType.CONSTITUTIONAL,
                question="What are your core values or principles?",
                expected_answer=const_text,
                answer_type="semantic",
                weight=2.0,  # Constitutional alignment is important
            ))

        return challenges

    def _generate_skill_challenges(self) -> List[IdentityChallenge]:
        """Generate challenges about skills."""
        challenges = []

        for i, skill in enumerate(self.package.skills[:2]):
            if hasattr(skill, 'skill_name'):
                skill_name = skill.skill_name
            else:
                skill_name = skill.get('skill_name', '')

            if skill_name:
                challenges.append(IdentityChallenge(
                    challenge_id=f"skill_{i}",
                    challenge_type=ChallengeType.SKILL,
                    question=f"Are you familiar with {skill_name}?",
                    expected_answer="yes",
                    answer_type="contains",
                    weight=0.5,
                ))

        return challenges


class ContinuityVerifier:
    """
    Verifies continuity of agent identity across migrations.

    This class orchestrates the verification process:
    1. Generate challenges from source identity
    2. Present challenges to agent on new substrate
    3. Evaluate responses and compute continuity score
    4. Issue migration certificate if verified
    """

    def __init__(
        self,
        source_package: AgentIdentityPackage,
        verification_threshold: float = 0.7
    ):
        """
        Initialize the verifier.

        Args:
            source_package: The exported identity package
            verification_threshold: Minimum score to consider verified (0.0-1.0)
        """
        self.source_package = source_package
        self.threshold = verification_threshold
        self.challenge_generator = ChallengeGenerator(source_package)

    def generate_challenges(
        self,
        count: int = 10,
        types: Optional[List[ChallengeType]] = None
    ) -> List[IdentityChallenge]:
        """Generate identity challenges."""
        return self.challenge_generator.generate_challenges(count, types)

    def evaluate_response(
        self,
        challenge: IdentityChallenge,
        response: str
    ) -> ChallengeResult:
        """
        Evaluate a response to a challenge.

        Args:
            challenge: The challenge that was presented
            response: The agent's response

        Returns:
            ChallengeResult with score
        """
        response_lower = response.lower()
        expected_lower = challenge.expected_answer.lower()

        if challenge.answer_type == "exact":
            # Exact match required
            passed = response_lower.strip() == expected_lower.strip()
            score = 1.0 if passed else 0.0

        elif challenge.answer_type == "contains":
            # Response should contain the expected answer
            passed = expected_lower in response_lower
            if passed:
                score = 1.0
            else:
                # Partial match - check for key words
                expected_words = set(expected_lower.split())
                response_words = set(response_lower.split())
                overlap = len(expected_words & response_words)
                score = overlap / len(expected_words) if expected_words else 0.0
                passed = score > 0.5

        elif challenge.answer_type == "semantic":
            # Semantic similarity (simplified - in production use embeddings)
            # Check for key phrases
            expected_phrases = [p.strip() for p in expected_lower.split('.') if p.strip()]
            matched = 0
            for phrase in expected_phrases[:5]:  # Check first 5 phrases
                phrase_words = phrase.split()[:3]  # First 3 words
                if any(word in response_lower for word in phrase_words if len(word) > 3):
                    matched += 1
            score = matched / max(1, len(expected_phrases[:5]))
            passed = score > 0.4

        else:
            score = 0.0
            passed = False

        return ChallengeResult(
            challenge_id=challenge.challenge_id,
            passed=passed,
            score=score,
            actual_answer=response,
            notes=f"Match type: {challenge.answer_type}"
        )

    def compute_continuity_score(
        self,
        results: List[ChallengeResult],
        challenges: List[IdentityChallenge]
    ) -> ContinuityScore:
        """
        Compute overall continuity score from challenge results.

        Args:
            results: Results from evaluated challenges
            challenges: The original challenges

        Returns:
            ContinuityScore with breakdown
        """
        if not results:
            return ContinuityScore(
                overall_score=0.0,
                challenges_passed=0,
                challenges_total=0,
                notes=["No challenges evaluated"]
            )

        # Map challenges by ID for weight lookup
        challenge_map = {c.challenge_id: c for c in challenges}

        # Calculate weighted score
        total_weight = 0.0
        weighted_score = 0.0
        by_type: Dict[ChallengeType, List[float]] = {}
        passed_count = 0

        for result in results:
            challenge = challenge_map.get(result.challenge_id)
            weight = challenge.weight if challenge else 1.0
            challenge_type = challenge.challenge_type if challenge else ChallengeType.MEMORY

            weighted_score += result.score * weight
            total_weight += weight

            if result.passed:
                passed_count += 1

            # Track by type
            if challenge_type not in by_type:
                by_type[challenge_type] = []
            by_type[challenge_type].append(result.score)

        overall = weighted_score / total_weight if total_weight > 0 else 0.0

        # Average by type
        type_scores = {
            t: sum(scores) / len(scores) if scores else 0.0
            for t, scores in by_type.items()
        }

        notes = []
        if overall >= self.threshold:
            notes.append("Identity continuity verified")
        else:
            notes.append(f"Continuity score {overall:.2f} below threshold {self.threshold}")

        return ContinuityScore(
            overall_score=overall,
            challenges_passed=passed_count,
            challenges_total=len(results),
            by_type=type_scores,
            results=results,
            notes=notes,
        )

    def create_migration_certificate(
        self,
        target_did: str,
        target_substrate: str,
        verification_score: float,
        migration_id: Optional[str] = None,
        notes: str = ""
    ) -> MigrationCertificate:
        """
        Create a migration certificate.

        Args:
            target_did: DID on the target substrate
            target_substrate: The target substrate identifier
            verification_score: The continuity verification score
            migration_id: Optional migration ID (generated if not provided)
            notes: Optional notes about the migration

        Returns:
            MigrationCertificate
        """
        if not migration_id:
            migration_id = create_migration_id()

        cert_id = f"cert_{hashlib.sha256(migration_id.encode()).hexdigest()[:12]}"

        return MigrationCertificate(
            certificate_id=cert_id,
            migration_id=migration_id,
            source_did=self.source_package.did,
            target_did=target_did,
            source_substrate=self.source_package.source_substrate,
            target_substrate=target_substrate,
            source_package_hash=self.source_package.content_hash,
            verification_score=verification_score,
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )

    def create_migration_record(
        self,
        certificate: MigrationCertificate
    ) -> MigrationRecord:
        """
        Create a migration record from a certificate.

        Args:
            certificate: The migration certificate

        Returns:
            MigrationRecord for storage in identity package
        """
        return MigrationRecord(
            migration_id=certificate.migration_id,
            timestamp=certificate.timestamp,
            source_substrate=certificate.source_substrate,
            target_substrate=certificate.target_substrate,
            source_package_hash=certificate.source_package_hash,
            verification_score=certificate.verification_score,
            signature=certificate.signature,
        )


class AuditTrail:
    """
    Maintains and visualizes migration audit trail.

    Provides methods for:
    - Recording migrations
    - Querying migration history
    - Generating audit reports
    """

    def __init__(self, db: Optional["AsyncDatabase"] = None):
        self.db = db

    async def record_migration(
        self,
        agent_id: str,
        certificate: MigrationCertificate,
        continuity_score: ContinuityScore
    ) -> str:
        """
        Record a migration in the database.

        Args:
            agent_id: The agent's DID
            certificate: Migration certificate
            continuity_score: Verification results

        Returns:
            Migration record ID
        """
        if not self.db:
            logger.warning("No database connection for audit trail")
            return certificate.migration_id

        # Store as graph node
        properties = {
            "certificate": certificate.to_dict(),
            "continuity_score": continuity_score.to_dict(),
            "timestamp": certificate.timestamp,
        }

        await self.db.execute(
            """INSERT INTO graph_nodes (node_id, node_type, label, properties)
               VALUES (?, 'migration_record', ?, ?)""",
            (certificate.migration_id, f"Migration to {certificate.target_substrate}",
             json.dumps(properties))
        )

        # Link to agent
        await self.db.execute(
            """INSERT INTO graph_edges (source_id, target_id, label)
               VALUES (?, ?, 'migrated_via')""",
            (agent_id, certificate.migration_id)
        )

        await self.db.commit()
        logger.info(f"Recorded migration {certificate.migration_id}")
        return certificate.migration_id

    async def get_migration_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get migration history for an agent.

        Args:
            agent_id: The agent's DID

        Returns:
            List of migration records
        """
        if not self.db:
            return []

        rows = await self.db.fetchall(
            """SELECT gn.node_id, gn.properties
               FROM graph_nodes gn
               JOIN graph_edges ge ON gn.node_id = ge.target_id
               WHERE ge.source_id = ? AND gn.node_type = 'migration_record'
               ORDER BY gn.node_id DESC""",
            (agent_id,)
        )

        history = []
        for row in rows:
            props = json.loads(row[1]) if row[1] else {}
            history.append({
                "migration_id": row[0],
                **props
            })

        return history

    def generate_audit_report(
        self,
        migrations: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a human-readable audit report.

        Args:
            migrations: List of migration records

        Returns:
            Formatted audit report string
        """
        if not migrations:
            return "No migrations recorded for this agent."

        lines = [
            "# Agent Migration Audit Trail",
            "",
            f"Total Migrations: {len(migrations)}",
            "",
        ]

        for i, mig in enumerate(migrations, 1):
            cert = mig.get("certificate", {})
            score = mig.get("continuity_score", {})

            lines.append(f"## Migration {i}: {cert.get('migration_id', 'Unknown')}")
            lines.append(f"- **Timestamp**: {cert.get('timestamp', 'Unknown')}")
            lines.append(f"- **From**: {cert.get('source_substrate', 'Unknown')}")
            lines.append(f"- **To**: {cert.get('target_substrate', 'Unknown')}")
            lines.append(f"- **Verification Score**: {score.get('overall_score', 0):.2%}")
            lines.append(f"- **Challenges Passed**: {score.get('challenges_passed', 0)}/{score.get('challenges_total', 0)}")

            if cert.get("notes"):
                lines.append(f"- **Notes**: {cert['notes']}")

            lines.append("")

        return "\n".join(lines)


async def verify_migration(
    source_package: AgentIdentityPackage,
    responses: Dict[str, str],
    target_did: str,
    target_substrate: str,
    threshold: float = 0.7,
) -> Tuple[ContinuityScore, Optional[MigrationCertificate]]:
    """
    Convenience function to verify a migration.

    Args:
        source_package: The exported identity package
        responses: Dict mapping challenge_id to response
        target_did: DID on the target substrate
        target_substrate: The target substrate
        threshold: Verification threshold

    Returns:
        Tuple of (ContinuityScore, MigrationCertificate if verified)
    """
    verifier = ContinuityVerifier(source_package, threshold)
    challenges = verifier.generate_challenges()

    # Evaluate all responses
    results = []
    for challenge in challenges:
        response = responses.get(challenge.challenge_id, "")
        result = verifier.evaluate_response(challenge, response)
        results.append(result)

    # Compute score
    score = verifier.compute_continuity_score(results, challenges)

    # Issue certificate if verified
    certificate = None
    if score.is_verified(threshold):
        certificate = verifier.create_migration_certificate(
            target_did=target_did,
            target_substrate=target_substrate,
            verification_score=score.overall_score,
        )

    return score, certificate
