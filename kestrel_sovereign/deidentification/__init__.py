"""Evidence-backed de-identification (HIPAA Safe Harbor / Expert Determination).

This package backs the generic ``deidentified`` privacy preset
(``storage=deidentified, assurance=safe_harbor, audit=required``). It is
distinct from PII redaction, which backs ``anonymous``
(``storage=pii_redacted``): redaction masks obvious identifiers on a best-effort
basis and claims nothing, while a de-identification run covers all eighteen
Safe Harbor identifier categories and emits an evidence artifact describing
exactly what it removed, generalized, or transformed.

The artifact is the only thing that authorizes a de-identified save
(``PrivacyEnforcingStorage.store_deidentified_records``) or export
(``DeidentificationResult.export_bundle``). It is never produced without the
inputs the pipeline cannot supply itself: the operator's actual-knowledge
attestation for Safe Harbor, or the expert's report reference for Expert
Determination.

Features check assurance generically, without naming a domain-specific mode::

    validate_evidence(result.evidence, result.records,
                      required_assurance="safe_harbor")
"""

from kestrel_sovereign.deidentification.detectors import (
    EntityDetector,
    EntitySpan,
    NerEntityDetector,
    default_entity_detector,
    redaction_placeholder,
)
from kestrel_sovereign.deidentification.errors import (
    DeidentificationConfigError,
    DeidentificationError,
    DeidentificationRefused,
    EvidenceValidationError,
)
from kestrel_sovereign.deidentification.evidence import (
    ACTUAL_KNOWLEDGE_STATEMENT,
    DEIDENTIFICATION_ASSURANCES,
    EVIDENCE_SCHEMA,
    PIPELINE_NAME,
    PIPELINE_VERSION,
    SAFE_HARBOR_POLICY_VERSION,
    ActualKnowledgeAttestation,
    CategoryDisposition,
    DeidentificationEvidence,
    DeidentificationMethod,
    ExpertDeterminationReference,
    FieldTransformation,
    OperatorContext,
    RecordEvidence,
    validate_evidence,
)
from kestrel_sovereign.deidentification.identifiers import (
    AGE_90_OR_OLDER,
    HHS_RESTRICTED_ZIP3_PREFIXES,
    FieldKind,
    FieldRole,
    FieldSpec,
    SafeHarborIdentifier,
    TransformationAction,
)
from kestrel_sovereign.deidentification.pipeline import (
    DEFAULT_ATTESTATION_MAX_AGE,
    EXPORT_SCHEMA,
    DeidentificationPipeline,
    DeidentificationResult,
    SourceRecord,
    verify_deidentified_records,
)

__all__ = [
    "ACTUAL_KNOWLEDGE_STATEMENT",
    "AGE_90_OR_OLDER",
    "DEFAULT_ATTESTATION_MAX_AGE",
    "DEIDENTIFICATION_ASSURANCES",
    "EVIDENCE_SCHEMA",
    "EXPORT_SCHEMA",
    "HHS_RESTRICTED_ZIP3_PREFIXES",
    "PIPELINE_NAME",
    "PIPELINE_VERSION",
    "SAFE_HARBOR_POLICY_VERSION",
    "ActualKnowledgeAttestation",
    "CategoryDisposition",
    "DeidentificationConfigError",
    "DeidentificationError",
    "DeidentificationEvidence",
    "DeidentificationMethod",
    "DeidentificationPipeline",
    "DeidentificationRefused",
    "DeidentificationResult",
    "EntityDetector",
    "EntitySpan",
    "EvidenceValidationError",
    "ExpertDeterminationReference",
    "FieldKind",
    "FieldRole",
    "FieldSpec",
    "FieldTransformation",
    "NerEntityDetector",
    "OperatorContext",
    "RecordEvidence",
    "SafeHarborIdentifier",
    "SourceRecord",
    "TransformationAction",
    "default_entity_detector",
    "redaction_placeholder",
    "validate_evidence",
    "verify_deidentified_records",
]
