"""
Kestrel Identity Module: Substrate-Independent Agent Portability.

This module provides the infrastructure for exporting and importing complete
agent identities across different LLM substrates while preserving continuity
of self, memories, and relationships.

Key Components:
- AgentIdentityPackage: Complete portable agent identity
- IdentityExporter: Export agent state to portable package
- IdentityImporter: Import and verify identity packages
- PersonalityFingerprint: Communication style preservation
- Signing: DID-based cryptographic signing

See Issue #23 for the full design: Substrate-Independent Agent Portability.
"""
from .identity_package import (
    AgentIdentityPackage,
    PersonalityFingerprint,
    RelationshipRecord,
    SkillRecord,
    MigrationRecord,
    SubstrateType,
    IDENTITY_PACKAGE_VERSION,
    create_package_hash,
    create_migration_id,
)

from .exporter import (
    IdentityExporter,
    export_identity,
)

from .importer import (
    IdentityImporter,
    ImportResult,
    import_identity,
)

from .signing import (
    sign_package,
    verify_package_signature,
    sign_and_export,
    verify_and_load,
    PackageSigner,
    SigningError,
    VerificationError,
)

from .personality_analyzer import (
    PersonalityAnalyzer,
    CalibrationPromptGenerator,
    AnalysisResult,
    analyze_personality,
    generate_calibration_prompt,
)

from .substrate_adapter import (
    SubstrateAdapter,
    Capability,
    CapabilityMap,
    CapabilityGap,
    discover_substrate_capabilities,
    generate_migration_prompt,
)

from .continuity_verifier import (
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

from .graceful_degradation import (
    GracefulDegradationHandler,
    SeverityLevel,
    CapabilityLoss,
    DegradationReport,
    assess_migration_impact,
    generate_limitation_disclosure,
)

__all__ = [
    # Package schema
    "AgentIdentityPackage",
    "PersonalityFingerprint",
    "RelationshipRecord",
    "SkillRecord",
    "MigrationRecord",
    "SubstrateType",
    "IDENTITY_PACKAGE_VERSION",
    "create_package_hash",
    "create_migration_id",
    # Export
    "IdentityExporter",
    "export_identity",
    # Import
    "IdentityImporter",
    "ImportResult",
    "import_identity",
    # Signing
    "sign_package",
    "verify_package_signature",
    "sign_and_export",
    "verify_and_load",
    "PackageSigner",
    "SigningError",
    "VerificationError",
    # Personality Analysis
    "PersonalityAnalyzer",
    "CalibrationPromptGenerator",
    "AnalysisResult",
    "analyze_personality",
    "generate_calibration_prompt",
    # Substrate Adaptation
    "SubstrateAdapter",
    "Capability",
    "CapabilityMap",
    "CapabilityGap",
    "discover_substrate_capabilities",
    "generate_migration_prompt",
    # Continuity Verification
    "ContinuityVerifier",
    "ChallengeGenerator",
    "ChallengeType",
    "IdentityChallenge",
    "ChallengeResult",
    "ContinuityScore",
    "MigrationCertificate",
    "AuditTrail",
    "verify_migration",
    # Graceful Degradation
    "GracefulDegradationHandler",
    "SeverityLevel",
    "CapabilityLoss",
    "DegradationReport",
    "assess_migration_impact",
    "generate_limitation_disclosure",
]
