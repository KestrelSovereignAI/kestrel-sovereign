"""Errors raised by the de-identification pipeline and its evidence contract."""


class DeidentificationError(ValueError):
    """A de-identification run or artifact cannot be trusted.

    Messages name fields, identifier categories, and digests only. They never
    echo a record value, because the value is the thing being protected.
    """


class DeidentificationConfigError(DeidentificationError):
    """The pipeline configuration (field schema, detectors, key) is unusable."""


class DeidentificationRefused(DeidentificationError):
    """A run was refused before any record was transformed.

    Raised when a precondition of the requested method is missing: the
    operator's Safe Harbor actual-knowledge attestation, or the expert report
    reference an Expert Determination run must cite. No artifact is produced,
    so the evidence-gated write it would have authorized stays blocked.
    """


class EvidenceValidationError(DeidentificationError):
    """An evidence artifact is incomplete, inconsistent, or does not bind the
    records it is presented with."""
