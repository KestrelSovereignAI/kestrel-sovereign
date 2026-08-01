"""Typed contracts for controlled semantic export and corpus artifacts.

The value objects in this module deliberately carry *exact* assertion and
revision lineage while an artifact is active.  They do not model the exported
bytes themselves: a registry receipt is an accountability record, never a
second content store.  The storage owner removes lineage before retaining an
erasure/revocation receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence
from uuid import UUID


ARTIFACT_LIFECYCLE_SCHEMA_VERSION = 1


class GovernedArtifactError(ValueError):
    """A controlled semantic artifact request is malformed or unsafe."""


class GovernedArtifactKind(str, Enum):
    EXPORT_SNAPSHOT = "export_snapshot"
    CORPUS_MANIFEST = "corpus_manifest"
    FUTURE_CORPUS_CANDIDATE = "future_corpus_candidate"


class GovernedArtifactState(str, Enum):
    ACTIVE = "active"
    REVOCATION_PENDING = "revocation_pending"
    REVOKED = "revoked"
    EXPIRED = "expired"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise GovernedArtifactError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, field: str) -> str:
    value = _text(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise GovernedArtifactError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _utc_timestamp(value: object, field: str) -> str:
    value = _text(value, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GovernedArtifactError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise GovernedArtifactError(f"{field} must carry an explicit UTC offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class GovernedArtifactLineage:
    """One exact current assertion revision included in a controlled artifact."""

    assertion_id: str
    revision_id: str

    def __post_init__(self) -> None:
        _text(self.assertion_id, "assertion_id")
        _text(self.revision_id, "revision_id")


@dataclass(frozen=True, slots=True)
class GovernedArtifactRegistration:
    """A complete, tenant-bound request to register an artifact before use."""

    artifact_id: str
    kind: GovernedArtifactKind | str
    tenant_id: str
    consumer_id: str
    checkpoint_generation: int
    policy_pin: str
    capability_pins: Mapping[str, str]
    lineage: Sequence[GovernedArtifactLineage]
    retention_expires_at: str
    artifact_digest: str

    def __post_init__(self) -> None:
        artifact_id = _text(self.artifact_id, "artifact_id")
        try:
            parsed_id = UUID(artifact_id)
        except ValueError as error:
            raise GovernedArtifactError("artifact_id must be a UUIDv4") from error
        if parsed_id.version != 4 or str(parsed_id) != artifact_id:
            raise GovernedArtifactError("artifact_id must be a UUIDv4")
        _text(self.tenant_id, "tenant_id")
        _text(self.consumer_id, "consumer_id")
        _sha256(self.policy_pin, "policy_pin")
        retention_expires_at = _utc_timestamp(self.retention_expires_at, "retention_expires_at")
        _sha256(self.artifact_digest, "artifact_digest")
        if type(self.checkpoint_generation) is not int or self.checkpoint_generation < 0:
            raise GovernedArtifactError("checkpoint_generation must be a non-negative integer")
        kind = GovernedArtifactKind(self.kind)
        pins = dict(self.capability_pins)
        if not pins or any(not isinstance(key, str) or not key for key in pins):
            raise GovernedArtifactError("capability_pins must be a non-empty string mapping")
        for value in pins.values():
            _sha256(value, "capability pin")
        lineage = tuple(self.lineage)
        if not lineage or any(not isinstance(item, GovernedArtifactLineage) for item in lineage):
            raise GovernedArtifactError("lineage must contain GovernedArtifactLineage values")
        if len({(item.assertion_id, item.revision_id) for item in lineage}) != len(lineage):
            raise GovernedArtifactError("lineage must not contain duplicate assertion/revision pairs")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "retention_expires_at", retention_expires_at)
        object.__setattr__(self, "capability_pins", MappingProxyType(dict(sorted(pins.items()))))
        object.__setattr__(self, "lineage", tuple(sorted(lineage, key=lambda item: (item.assertion_id, item.revision_id))))

    @property
    def capability_digest(self) -> str:
        return _digest(dict(self.capability_pins))


@dataclass(frozen=True, slots=True)
class GovernedArtifactReceipt:
    """Immutable, content-free lifecycle observation for one artifact action."""

    artifact_key: str
    kind: GovernedArtifactKind
    state: GovernedArtifactState
    generation: int
    receipt_digest: str


@dataclass(frozen=True, slots=True)
class GovernedArtifactRevocationLease:
    """Opaque, single-attempt acknowledgement authority for one consumer."""

    revocation_id: str
    lease_token: str
    attempt: int


@dataclass(frozen=True, slots=True)
class GovernedArtifactErasureObservation:
    """Tenant aggregate, generation-fenced and deliberately identifier-free."""

    generation: int
    export_snapshots: int
    governed_corpus: int
    future_corpus: int
    pending_revocations: int
    completed_revocations: int

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise GovernedArtifactError("generation must be a non-negative integer")
        for field in (
            "export_snapshots", "governed_corpus", "future_corpus",
            "pending_revocations", "completed_revocations",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise GovernedArtifactError(f"{field} must be a non-negative integer")


__all__ = [
    "ARTIFACT_LIFECYCLE_SCHEMA_VERSION", "GovernedArtifactErasureObservation",
    "GovernedArtifactError", "GovernedArtifactKind", "GovernedArtifactLineage",
    "GovernedArtifactReceipt", "GovernedArtifactRegistration", "GovernedArtifactRevocationLease",
    "GovernedArtifactState",
]
