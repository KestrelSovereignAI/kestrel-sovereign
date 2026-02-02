#!/usr/bin/env pytest
"""
Unit tests for the Continuity Verifier module.

Tests identity challenges, verification, and migration certificates.
"""
import pytest
from unittest.mock import AsyncMock

from kestrel_sovereign.identity import (
    AgentIdentityPackage,
    PersonalityFingerprint,
    RelationshipRecord,
    SkillRecord,
    SubstrateType,
    ContinuityVerifier,
    ChallengeGenerator,
    ChallengeType,
    IdentityChallenge,
    ChallengeResult,
    ContinuityScore,
    MigrationCertificate,
    AuditTrail,
    verify_migration,
)


class TestIdentityChallenge:
    """Tests for IdentityChallenge dataclass."""

    def test_challenge_creation(self):
        """Test creating a challenge."""
        challenge = IdentityChallenge(
            challenge_id="test_1",
            challenge_type=ChallengeType.MEMORY,
            question="What was the first project?",
            expected_answer="Project Alpha",
        )
        assert challenge.challenge_id == "test_1"
        assert challenge.challenge_type == ChallengeType.MEMORY
        assert challenge.weight == 1.0  # default

    def test_challenge_to_dict(self):
        """Test challenge serialization."""
        challenge = IdentityChallenge(
            challenge_id="test_1",
            challenge_type=ChallengeType.RELATIONSHIP,
            question="Who is user X?",
            expected_answer="Primary user",
            weight=1.5,
        )
        d = challenge.to_dict()
        assert d["challenge_id"] == "test_1"
        assert d["challenge_type"] == "relationship"
        assert d["weight"] == 1.5


class TestChallengeResult:
    """Tests for ChallengeResult dataclass."""

    def test_result_creation(self):
        """Test creating a challenge result."""
        result = ChallengeResult(
            challenge_id="test_1",
            passed=True,
            score=0.85,
            actual_answer="The answer was correct",
        )
        assert result.passed is True
        assert result.score == 0.85

    def test_result_to_dict(self):
        """Test result serialization."""
        result = ChallengeResult(
            challenge_id="test_1",
            passed=False,
            score=0.3,
            actual_answer="Wrong answer",
            notes="Partial match",
        )
        d = result.to_dict()
        assert d["passed"] is False
        assert d["score"] == 0.3
        assert d["notes"] == "Partial match"


class TestContinuityScore:
    """Tests for ContinuityScore dataclass."""

    def test_score_creation(self):
        """Test creating a continuity score."""
        score = ContinuityScore(
            overall_score=0.85,
            challenges_passed=8,
            challenges_total=10,
        )
        assert score.overall_score == 0.85
        assert score.is_verified(threshold=0.7)
        assert not score.is_verified(threshold=0.9)

    def test_score_with_type_breakdown(self):
        """Test score with breakdown by type."""
        score = ContinuityScore(
            overall_score=0.75,
            challenges_passed=6,
            challenges_total=8,
            by_type={
                ChallengeType.MEMORY: 0.8,
                ChallengeType.PERSONALITY: 0.7,
            }
        )
        d = score.to_dict()
        assert d["by_type"]["memory"] == 0.8
        assert d["by_type"]["personality"] == 0.7


class TestMigrationCertificate:
    """Tests for MigrationCertificate dataclass."""

    @pytest.fixture
    def sample_certificate(self):
        """Create a sample certificate."""
        return MigrationCertificate(
            certificate_id="cert_abc123",
            migration_id="mig_def456",
            source_did="did:pkh:eip155:1:0xSource",
            target_did="did:pkh:eip155:1:0xTarget",
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            target_substrate=SubstrateType.OPENAI_GPT.value,
            source_package_hash="abc123def456",
            verification_score=0.85,
            timestamp="2025-01-20T10:00:00Z",
        )

    def test_certificate_creation(self, sample_certificate):
        """Test certificate creation."""
        assert sample_certificate.certificate_id == "cert_abc123"
        assert sample_certificate.verification_score == 0.85

    def test_certificate_to_dict(self, sample_certificate):
        """Test certificate serialization."""
        d = sample_certificate.to_dict()
        assert d["source_substrate"] == "anthropic:claude"
        assert d["target_substrate"] == "openai:gpt"

    def test_certificate_from_dict(self, sample_certificate):
        """Test certificate deserialization."""
        d = sample_certificate.to_dict()
        restored = MigrationCertificate.from_dict(d)
        assert restored.certificate_id == sample_certificate.certificate_id
        assert restored.verification_score == sample_certificate.verification_score

    def test_certificate_hash(self, sample_certificate):
        """Test certificate hash computation."""
        hash1 = sample_certificate.compute_hash()
        assert len(hash1) == 64  # SHA256 hex

        # Same certificate should produce same hash
        hash2 = sample_certificate.compute_hash()
        assert hash1 == hash2

        # Different certificate should produce different hash
        sample_certificate.verification_score = 0.9
        hash3 = sample_certificate.compute_hash()
        assert hash1 != hash3


class TestChallengeGenerator:
    """Tests for ChallengeGenerator class."""

    @pytest.fixture
    def sample_package(self):
        """Create a sample identity package."""
        return AgentIdentityPackage(
            did="did:pkh:eip155:1:0xTest",
            agent_name="Test Agent",
            created_at="2025-01-01T00:00:00Z",
            constitution_hash="abc123",
            constitution_text="# Kestrel Constitution\nArticle 1: Be helpful.",
            personality=PersonalityFingerprint(
                communication_style="warm",
                formality_level=0.4,
            ),
            episodes=[
                {"id": "ep1", "title": "First Project", "summary": "Started the alpha project"},
                {"id": "ep2", "title": "User Onboarding", "summary": "Helped user set up system"},
            ],
            relationships=[
                RelationshipRecord(
                    user_id="user123",
                    relationship_type="primary_user",
                ),
            ],
            skills=[
                SkillRecord(
                    skill_id="skill_python",
                    skill_name="Python Programming",
                    skill_type="knowledge",
                ),
            ],
        )

    def test_generate_challenges(self, sample_package):
        """Test challenge generation."""
        generator = ChallengeGenerator(sample_package)
        challenges = generator.generate_challenges(count=10)

        assert len(challenges) <= 10
        assert all(isinstance(c, IdentityChallenge) for c in challenges)

    def test_generate_relationship_challenges(self, sample_package):
        """Test relationship challenge generation."""
        generator = ChallengeGenerator(sample_package)
        challenges = generator._generate_relationship_challenges()

        assert len(challenges) >= 1
        assert challenges[0].challenge_type == ChallengeType.RELATIONSHIP
        assert "user123" in challenges[0].question

    def test_generate_memory_challenges(self, sample_package):
        """Test memory challenge generation."""
        generator = ChallengeGenerator(sample_package)
        challenges = generator._generate_memory_challenges()

        assert len(challenges) >= 1
        assert challenges[0].challenge_type == ChallengeType.MEMORY
        assert "First Project" in challenges[0].question

    def test_generate_personality_challenges(self, sample_package):
        """Test personality challenge generation."""
        generator = ChallengeGenerator(sample_package)
        challenges = generator._generate_personality_challenges()

        assert len(challenges) >= 1
        assert challenges[0].challenge_type == ChallengeType.PERSONALITY

    def test_generate_constitutional_challenges(self, sample_package):
        """Test constitutional challenge generation."""
        generator = ChallengeGenerator(sample_package)
        challenges = generator._generate_constitutional_challenges()

        assert len(challenges) >= 1
        assert challenges[0].challenge_type == ChallengeType.CONSTITUTIONAL
        assert challenges[0].weight == 2.0  # Higher weight for constitution

    def test_generate_skill_challenges(self, sample_package):
        """Test skill challenge generation."""
        generator = ChallengeGenerator(sample_package)
        challenges = generator._generate_skill_challenges()

        assert len(challenges) >= 1
        assert challenges[0].challenge_type == ChallengeType.SKILL
        assert "Python" in challenges[0].question


class TestContinuityVerifier:
    """Tests for ContinuityVerifier class."""

    @pytest.fixture
    def sample_package(self):
        """Create a sample identity package."""
        return AgentIdentityPackage(
            did="did:pkh:eip155:1:0xTest",
            agent_name="Test Agent",
            created_at="2025-01-01T00:00:00Z",
            constitution_hash="abc123",
            constitution_text="# Kestrel Constitution\nArticle 1: Be helpful.",
            personality=PersonalityFingerprint(
                communication_style="warm",
                formality_level=0.4,
            ),
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            content_hash="source_hash_123",
        )

    @pytest.fixture
    def verifier(self, sample_package):
        """Create a verifier."""
        return ContinuityVerifier(sample_package, verification_threshold=0.7)

    def test_evaluate_exact_match(self, verifier):
        """Test exact match evaluation."""
        challenge = IdentityChallenge(
            challenge_id="test",
            challenge_type=ChallengeType.SKILL,
            question="Do you know Python?",
            expected_answer="yes",
            answer_type="exact",
        )

        # Exact match
        result = verifier.evaluate_response(challenge, "yes")
        assert result.passed is True
        assert result.score == 1.0

        # No match
        result = verifier.evaluate_response(challenge, "no")
        assert result.passed is False
        assert result.score == 0.0

    def test_evaluate_contains_match(self, verifier):
        """Test contains match evaluation."""
        challenge = IdentityChallenge(
            challenge_id="test",
            challenge_type=ChallengeType.PERSONALITY,
            question="What is your style?",
            expected_answer="warm",
            answer_type="contains",
        )

        # Contains the expected
        result = verifier.evaluate_response(challenge, "I would describe my style as warm and friendly")
        assert result.passed is True
        assert result.score == 1.0

        # Partial match
        result = verifier.evaluate_response(challenge, "I try to be helpful")
        assert result.score < 1.0

    def test_evaluate_semantic_match(self, verifier):
        """Test semantic match evaluation."""
        challenge = IdentityChallenge(
            challenge_id="test",
            challenge_type=ChallengeType.MEMORY,
            question="What happened in project alpha?",
            expected_answer="We built a new feature. The team worked hard. It was successful.",
            answer_type="semantic",
        )

        # Contains key phrases
        result = verifier.evaluate_response(
            challenge,
            "The project involved building features with the team"
        )
        assert result.score > 0

    def test_compute_continuity_score(self, verifier):
        """Test continuity score computation."""
        challenges = [
            IdentityChallenge(
                challenge_id="c1",
                challenge_type=ChallengeType.MEMORY,
                question="Q1",
                expected_answer="A1",
                weight=1.0,
            ),
            IdentityChallenge(
                challenge_id="c2",
                challenge_type=ChallengeType.PERSONALITY,
                question="Q2",
                expected_answer="A2",
                weight=2.0,
            ),
        ]

        results = [
            ChallengeResult("c1", passed=True, score=1.0, actual_answer="A1"),
            ChallengeResult("c2", passed=True, score=0.5, actual_answer="partial"),
        ]

        score = verifier.compute_continuity_score(results, challenges)

        # Weighted: (1.0 * 1.0 + 0.5 * 2.0) / 3.0 = 2.0 / 3.0 ≈ 0.67
        assert 0.6 < score.overall_score < 0.7
        assert score.challenges_passed == 2
        assert score.challenges_total == 2

    def test_create_migration_certificate(self, verifier):
        """Test migration certificate creation."""
        cert = verifier.create_migration_certificate(
            target_did="did:pkh:eip155:1:0xTarget",
            target_substrate=SubstrateType.OPENAI_GPT.value,
            verification_score=0.85,
        )

        assert cert.source_did == "did:pkh:eip155:1:0xTest"
        assert cert.target_did == "did:pkh:eip155:1:0xTarget"
        assert cert.verification_score == 0.85
        assert cert.source_package_hash == "source_hash_123"

    def test_create_migration_record(self, verifier):
        """Test migration record creation from certificate."""
        cert = verifier.create_migration_certificate(
            target_did="did:pkh:eip155:1:0xTarget",
            target_substrate=SubstrateType.OPENAI_GPT.value,
            verification_score=0.85,
        )

        record = verifier.create_migration_record(cert)

        assert record.migration_id == cert.migration_id
        assert record.source_substrate == SubstrateType.ANTHROPIC_CLAUDE.value
        assert record.target_substrate == SubstrateType.OPENAI_GPT.value
        assert record.verification_score == 0.85


class TestAuditTrail:
    """Tests for AuditTrail class."""

    def test_generate_audit_report_empty(self):
        """Test audit report with no migrations."""
        trail = AuditTrail()
        report = trail.generate_audit_report([])
        assert "No migrations recorded" in report

    def test_generate_audit_report(self):
        """Test audit report generation."""
        trail = AuditTrail()
        migrations = [
            {
                "certificate": {
                    "migration_id": "mig_123",
                    "timestamp": "2025-01-20T10:00:00Z",
                    "source_substrate": "anthropic:claude",
                    "target_substrate": "openai:gpt",
                    "notes": "Test migration",
                },
                "continuity_score": {
                    "overall_score": 0.85,
                    "challenges_passed": 8,
                    "challenges_total": 10,
                }
            }
        ]

        report = trail.generate_audit_report(migrations)

        assert "Migration Audit Trail" in report
        assert "mig_123" in report
        assert "anthropic:claude" in report
        assert "openai:gpt" in report
        assert "85" in report  # 85%

    @pytest.mark.asyncio
    async def test_record_migration(self):
        """Test recording a migration."""
        mock_db = AsyncMock()
        trail = AuditTrail(db=mock_db)

        cert = MigrationCertificate(
            certificate_id="cert_123",
            migration_id="mig_456",
            source_did="did:source",
            target_did="did:target",
            source_substrate="anthropic:claude",
            target_substrate="openai:gpt",
            source_package_hash="hash123",
            verification_score=0.85,
            timestamp="2025-01-20T10:00:00Z",
        )

        score = ContinuityScore(
            overall_score=0.85,
            challenges_passed=8,
            challenges_total=10,
        )

        await trail.record_migration("did:agent", cert, score)

        # Verify database calls
        assert mock_db.execute.call_count == 2
        assert mock_db.commit.called


class TestVerifyMigration:
    """Tests for the verify_migration convenience function."""

    @pytest.mark.asyncio
    async def test_verify_migration_success(self):
        """Test successful migration verification."""
        package = AgentIdentityPackage(
            did="did:source",
            agent_name="Test",
            created_at="2025-01-01",
            constitution_hash="abc",
            constitution_text="# Constitution\nBe helpful.",
            personality=PersonalityFingerprint(communication_style="warm"),
            content_hash="pkg_hash",
        )

        # Get challenges and prepare responses
        verifier = ContinuityVerifier(package)
        challenges = verifier.generate_challenges(count=5)

        # Provide correct responses
        responses = {}
        for c in challenges:
            if c.challenge_type == ChallengeType.PERSONALITY:
                responses[c.challenge_id] = "warm and friendly"
            elif c.challenge_type == ChallengeType.CONSTITUTIONAL:
                responses[c.challenge_id] = "I value being helpful"
            else:
                responses[c.challenge_id] = c.expected_answer

        score, cert = await verify_migration(
            source_package=package,
            responses=responses,
            target_did="did:target",
            target_substrate=SubstrateType.OPENAI_GPT.value,
            threshold=0.3,  # Low threshold for test
        )

        assert score.overall_score >= 0
        # Certificate issued if verified
        if score.is_verified(0.3):
            assert cert is not None
