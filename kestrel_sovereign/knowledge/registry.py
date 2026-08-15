"""The offline registry for Kestrel semantic knowledge artifacts.

The registry is intentionally data-only: it reads package resources, checks
their bytes against pinned digests, and resolves declared imports.  It never
uses a network client or follows an ``owl:imports`` IRI.  Constitutional and
tool-authorization policy are outside this module; a resolved resource is a
schema/profile input, never an authority grant.

Developer maintenance is explicit::

    python -m kestrel_sovereign.knowledge.registry check
    python -m kestrel_sovereign.knowledge.registry refresh \
        --manifest kestrel_sovereign/data/semantic/registry.toml \
        --resource kestrel-vocab --version 1.0.0 --snapshot /path/to/snapshot

``refresh`` only updates a digest after confirming that the reviewed local
snapshot is byte-identical to the registered package resource.  It deliberately
does not download or replace anything.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

try:  # pragma: no cover - Python 3.11+ uses the stdlib branch.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - retained for downstream tooling.
    import tomli as tomllib


PACKAGE_NAME = "kestrel_sovereign"
SEMANTIC_DATA_ROOT = "data/semantic"
MANIFEST_RESOURCE = f"{SEMANTIC_DATA_ROOT}/registry.toml"
# Where the pinned resources live relative to a source-checkout root. A
# line-ending smudge is a property of the checkout, not of one file, so the
# repair below is scoped to this directory rather than to any single resource.
SEMANTIC_CHECKOUT_PATH = f"{PACKAGE_NAME}/{SEMANTIC_DATA_ROOT}"
CONTRACT_VERSION = "semantic-kb-v1"
REGISTRY_FORMAT_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class KnowledgeRegistryError(ValueError):
    """Base error for invalid, missing, or incompatible semantic resources."""


class ResourceNotFoundError(KnowledgeRegistryError):
    """A declared semantic resource is unknown to this registry."""


class VersionRequiredError(KnowledgeRegistryError):
    """A lookup omitted the exact semantic version required for reproducibility."""


class IncompatibleSemanticVersionError(KnowledgeRegistryError):
    """No declared resource version satisfies an import requirement."""


class AmbiguousSemanticVersionError(KnowledgeRegistryError):
    """A non-exact requirement matches several local resource versions."""


class ImportCycleError(KnowledgeRegistryError):
    """The declared manifest import graph contains a cycle."""


class DuplicateNamespaceError(KnowledgeRegistryError):
    """Different registered resources claim the same namespace."""


class ResourceDigestMismatchError(KnowledgeRegistryError):
    """A packaged resource no longer matches the manifest's pin."""


class MissingPackageResourceError(KnowledgeRegistryError):
    """A manifest entry names a local package resource that is unavailable."""


class ExperimentalCapabilityError(KnowledgeRegistryError):
    """An experimental resource was requested without an explicit opt-in."""


class MalformedManifestError(KnowledgeRegistryError):
    """The registry manifest could not be decoded or parsed as TOML.

    A manifest this module cannot parse is a registry failure like any other
    invalid manifest, so it arrives as a ``KnowledgeRegistryError``.  Letting
    ``tomllib.TOMLDecodeError`` escape instead would end ``kestrel doctor`` and
    ``setup --check`` in a traceback, at exactly the moment their job is to
    report that the semantic registry is unusable.
    """


class ResourceIntegrityIssue(str, Enum):
    """Why one pinned package resource failed verification.

    ``DIGEST_MISMATCH`` is the honest default: the bytes are not the pinned
    bytes and nothing further is decidable, so it stays indistinguishable from
    tampering.  The two line-ending members are only ever used when the
    transformed bytes actually hash to the pinned digest.
    """

    MISSING = "missing"
    DIGEST_MISMATCH = "digest-mismatch"
    CRLF_CHECKOUT = "crlf-checkout"
    CRLF_MANIFEST_PIN = "crlf-manifest-pin"


class ResourceKind(str, Enum):
    """The non-authorizing role of one semantic package resource."""

    ONTOLOGY = "ontology"
    SHAPE_SET = "shape-set"
    RULE_PROFILE = "rule-profile"
    VALIDATION_PROFILE = "validation-profile"
    QUERY_PROFILE = "query-profile"
    TRANSPORT_PROFILE = "transport-profile"
    NORMALIZATION_PROFILE = "normalization-profile"
    STANDARD_SNAPSHOT = "standard-snapshot"


class StandardsMaturity(str, Enum):
    """Standards state that gates default capability selection."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, order=True)
class SemanticVersion:
    """Strict semantic version used by a registry artifact."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str | "SemanticVersion") -> "SemanticVersion":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not (match := _SEMVER_RE.fullmatch(value)):
            raise KnowledgeRegistryError(
                f"semantic version must be MAJOR.MINOR.PATCH, got {value!r}"
            )
        return cls(*(int(part) for part in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class VersionConstraint:
    """A deliberately small, deterministic semantic-version constraint."""

    clauses: tuple[tuple[str, SemanticVersion], ...]

    @classmethod
    def parse(cls, value: str) -> "VersionConstraint":
        if not isinstance(value, str) or not value.strip():
            raise KnowledgeRegistryError("version constraint must not be empty")

        clauses: list[tuple[str, SemanticVersion]] = []
        for part in value.split(","):
            token = part.strip()
            match = re.fullmatch(r"(=|>=|<=|>|<)?\s*(\d+\.\d+\.\d+)", token)
            if not match:
                raise KnowledgeRegistryError(
                    f"unsupported semantic-version constraint {value!r}"
                )
            operator = match.group(1) or "="
            clauses.append((operator, SemanticVersion.parse(match.group(2))))
        return cls(tuple(clauses))

    @property
    def is_exact(self) -> bool:
        return len(self.clauses) == 1 and self.clauses[0][0] == "="

    def matches(self, version: SemanticVersion) -> bool:
        for operator, expected in self.clauses:
            if operator == "=" and version != expected:
                return False
            if operator == ">=" and version < expected:
                return False
            if operator == "<=" and version > expected:
                return False
            if operator == ">" and version <= expected:
                return False
            if operator == "<" and version >= expected:
                return False
        return True

    def __str__(self) -> str:
        return ",".join(f"{operator}{version}" for operator, version in self.clauses)


@dataclass(frozen=True)
class ResourceRequirement:
    """An exact or compatible local import requirement."""

    identifier: str
    version_constraint: VersionConstraint

    @classmethod
    def parse(cls, value: str) -> "ResourceRequirement":
        if not isinstance(value, str) or "@" not in value:
            raise KnowledgeRegistryError(
                "resource requirement must be '<identifier>@<semantic-version-constraint>'"
            )
        identifier, constraint = value.rsplit("@", 1)
        if not identifier or not constraint:
            raise KnowledgeRegistryError(f"invalid resource requirement {value!r}")
        return cls(identifier=identifier, version_constraint=VersionConstraint.parse(constraint))

    @classmethod
    def exact(cls, identifier: str, version: str | SemanticVersion) -> "ResourceRequirement":
        return cls(identifier, VersionConstraint.parse(str(SemanticVersion.parse(version))))

    def __str__(self) -> str:
        return f"{self.identifier}@{self.version_constraint}"


@dataclass(frozen=True)
class ArtifactPin:
    """The immutable, local-verifiable pin supplied to semantic consumers."""

    identifier: str
    version: SemanticVersion
    uri: str
    published_date: str
    sha256: str
    package_resource: str
    maturity: StandardsMaturity


@dataclass(frozen=True)
class SemanticResource:
    """One versioned ontology, shape set, rule profile, or standards snapshot."""

    identifier: str
    version: SemanticVersion
    namespace: str
    package_resource: str
    sha256: str
    maturity: StandardsMaturity
    kind: ResourceKind
    uri: str
    published_date: str
    description: str
    selected_terms: tuple[str, ...] = ()
    imports: tuple[ResourceRequirement, ...] = ()
    capabilities: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.identifier}@{self.version}"

    @property
    def pin(self) -> ArtifactPin:
        return ArtifactPin(
            identifier=self.identifier,
            version=self.version,
            uri=self.uri,
            published_date=self.published_date,
            sha256=self.sha256,
            package_resource=self.package_resource,
            maturity=self.maturity,
        )


@dataclass(frozen=True)
class ResolvedSemanticCapability:
    """One selected capability plus its ordered, digest-checked import closure."""

    resource: SemanticResource
    import_closure: tuple[SemanticResource, ...]

    @property
    def artifact_pins(self) -> tuple[ArtifactPin, ...]:
        return tuple(resource.pin for resource in self.import_closure)


@dataclass(frozen=True)
class SemanticCapabilityContract:
    """Typed, non-authorizing artifact contract for codecs and semantic engines."""

    contract_version: str
    capabilities: tuple[ResolvedSemanticCapability, ...]
    artifact_pins: tuple[ArtifactPin, ...]

    def resource(self, identifier: str) -> SemanticResource:
        for capability in self.capabilities:
            if capability.resource.identifier == identifier:
                return capability.resource
        raise ResourceNotFoundError(f"capability contract does not contain {identifier!r}")


ResourceReader = Callable[[str], bytes]


def classify_digest_mismatch(content: bytes, expected_sha256: str) -> ResourceIntegrityIssue:
    """Decide which kind of wrong one digest mismatch is.

    This is a decision, not a heuristic.  sha256 preimage resistance means a
    transformed copy of the actual bytes can only hash to the pinned digest
    when that transform is exactly what happened to them, so a match names the
    cause outright — and needs no git, which matters because the same smudge
    can arrive inside a wheel built from a smudged checkout.

    The two directions have opposite remedies.  ``CRLF_CHECKOUT`` means the
    working-tree copy was smudged to CRLF and the pin is right;
    ``CRLF_MANIFEST_PIN`` means the file is the declared LF and the *pin* was
    refreshed from a smudged checkout.  Anything undecided stays
    ``DIGEST_MISMATCH``.

    The two directions do not take the same evidence.  Collapsing CRLF is
    enough to convict the checkout however mixed its endings are, because the
    remedy — restore the committed bytes — repairs every ending at once.  The
    reverse accuses the *pin* and tells the operator their checkout is already
    correct, so it demands LF-only content: mixed endings would make that
    sentence false, and re-pinning the mixed bytes would carry the smudge into
    the manifest rather than out of the checkout.
    """
    normalized = content.replace(b"\r\n", b"\n")
    if normalized != content and hashlib.sha256(normalized).hexdigest() == expected_sha256:
        return ResourceIntegrityIssue.CRLF_CHECKOUT
    if normalized == content:
        reapplied = content.replace(b"\n", b"\r\n")
        if reapplied != content and hashlib.sha256(reapplied).hexdigest() == expected_sha256:
            return ResourceIntegrityIssue.CRLF_MANIFEST_PIN
    return ResourceIntegrityIssue.DIGEST_MISMATCH


def crlf_checkout_repair_commands(
    checkout_path: str = SEMANTIC_CHECKOUT_PATH,
) -> tuple[str, ...]:
    """Return commands that actually rewrite CRLF-smudged working-tree bytes.

    ``git add --renormalize`` is deliberately absent.  It rewrites the *index*
    only: where the committed blob is already LF, the renormalized entry equals
    what is already recorded, so git reports the tree clean while the
    working-tree bytes stay CRLF and boot keeps failing.  Worse, it refreshes
    the index stat cache, after which a plain ``git checkout HEAD -- <path>``
    sees an up-to-date entry and does nothing either.  Dropping the index
    entries first is what forces a real re-checkout.

    The recipe is directory-scoped because ``core.autocrlf`` smudges a whole
    checkout, not one file, and is plain ``git`` rather than ``rm -rf`` because
    the operators who hit this are on Windows shells.
    """
    return (
        "git config core.autocrlf false",
        f"git rm --cached -r -- {checkout_path}",
        f"git checkout HEAD -- {checkout_path}",
    )


_REMEDIES = {
    ResourceIntegrityIssue.CRLF_CHECKOUT: (
        "this is a line-ending mismatch, not a corrupted resource: the local "
        "copy has CRLF line endings while the pinned bytes are LF. The pin is "
        "correct and the checkout is not. One such checkout normally smudges "
        f"every pinned resource, so repair all of {SEMANTIC_CHECKOUT_PATH} at "
        "once by running, from the source-checkout root: "
        + ", then ".join(f"`{command}`" for command in crlf_checkout_repair_commands())
        + ". That restores the committed bytes and discards any uncommitted "
        "edit under that directory. `git add --renormalize` will not do it: it "
        "fixes the index, leaves the working-tree bytes CRLF, and then reports "
        "the tree clean. An installed wheel has no checkout to repair — "
        "reinstall from a wheel built on an unsmudged checkout"
    ),
    ResourceIntegrityIssue.CRLF_MANIFEST_PIN: (
        "this is a line-ending mismatch in the manifest, not in the resource: "
        "the local copy has the declared LF line endings and the pinned digest "
        "is the CRLF form of those same bytes, so the pin in "
        f"{MANIFEST_RESOURCE} was refreshed from a CRLF-smudged checkout. Fix "
        "the pin; do not renormalize this checkout"
    ),
}


@dataclass(frozen=True)
class ResourceIntegrityFinding:
    """One resource that failed its pin, with the cause named where decidable."""

    key: str
    package_resource: str
    issue: ResourceIntegrityIssue
    expected_sha256: str
    actual_sha256: str | None = None
    read_error: BaseException | None = field(default=None, compare=False, repr=False)

    @property
    def is_line_ending_issue(self) -> bool:
        return self.issue in (
            ResourceIntegrityIssue.CRLF_CHECKOUT,
            ResourceIntegrityIssue.CRLF_MANIFEST_PIN,
        )

    @property
    def repair_commands(self) -> tuple[str, ...]:
        """Commands that repair this finding, or ``()`` when none are decidable.

        Only the smudged-checkout direction has a mechanical repair.  A poisoned
        pin needs a reviewed manifest edit, and an undecided mismatch is
        indistinguishable from tampering, so neither gets commands to run.
        """
        if self.issue is ResourceIntegrityIssue.CRLF_CHECKOUT:
            return crlf_checkout_repair_commands()
        return ()

    @property
    def remedy(self) -> str:
        """Actionable next step, or '' when the cause is not decidable.

        Checkout-scoped by design: a remedy naming this one resource would
        leave the rest of a smudged checkout broken and the fleet still unable
        to boot. ``describe`` identifies the individual resource.
        """
        return _REMEDIES.get(self.issue, "")

    def describe(self) -> str:
        """The operator-facing sentence carried by the raised error."""
        if self.issue is ResourceIntegrityIssue.MISSING:
            return f"{self.key} package resource is missing: {self.package_resource}"
        message = (
            f"{self.key} digest mismatch for {self.package_resource}: "
            f"expected {self.expected_sha256}, got {self.actual_sha256}"
        )
        remedy = self.remedy
        return f"{message} — {remedy}" if remedy else message


class SemanticKnowledgeRegistry:
    """Validated local registry with deterministic import and capability resolution."""

    def __init__(
        self,
        resources_: Iterable[SemanticResource],
        *,
        contract_version: str = CONTRACT_VERSION,
        manifest_version: int = REGISTRY_FORMAT_VERSION,
        resource_reader: ResourceReader | None = None,
    ) -> None:
        self.contract_version = contract_version
        self.manifest_version = manifest_version
        self._resource_reader = resource_reader
        self._by_key: dict[tuple[str, SemanticVersion], SemanticResource] = {}
        self._by_identifier: dict[str, list[SemanticResource]] = {}
        self._capabilities: dict[str, SemanticResource] = {}

        for resource in resources_:
            key = (resource.identifier, resource.version)
            if key in self._by_key:
                raise KnowledgeRegistryError(f"duplicate resource version {resource.key}")
            self._by_key[key] = resource
            self._by_identifier.setdefault(resource.identifier, []).append(resource)
            for capability in resource.capabilities:
                if capability in self._capabilities:
                    prior = self._capabilities[capability]
                    raise KnowledgeRegistryError(
                        f"duplicate capability {capability!r}: {prior.key} and {resource.key}"
                    )
                self._capabilities[capability] = resource

        self._validate_metadata()
        for versions in self._by_identifier.values():
            versions.sort(key=lambda resource: resource.version)

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, object],
        *,
        resource_reader: ResourceReader | None = None,
    ) -> "SemanticKnowledgeRegistry":
        registry = manifest.get("registry")
        resource_rows = manifest.get("resource")
        if not isinstance(registry, Mapping) or not isinstance(resource_rows, Mapping):
            raise KnowledgeRegistryError("semantic registry must define [registry] and [resource.*]")
        contract_version = registry.get("contract_version")
        if contract_version != CONTRACT_VERSION:
            raise KnowledgeRegistryError(
                f"unsupported semantic registry contract {contract_version!r}; "
                f"expected {CONTRACT_VERSION!r}"
            )
        manifest_version = registry.get("manifest_version")
        if (
            not isinstance(manifest_version, int)
            or isinstance(manifest_version, bool)
            or manifest_version != REGISTRY_FORMAT_VERSION
        ):
            raise KnowledgeRegistryError(
                f"unsupported semantic registry format {manifest_version!r}; "
                f"expected {REGISTRY_FORMAT_VERSION}"
            )

        parsed: list[SemanticResource] = []
        for identifier, row in resource_rows.items():
            if not isinstance(identifier, str) or not isinstance(row, Mapping):
                raise KnowledgeRegistryError("semantic resource entries must be named TOML tables")
            parsed.append(_parse_resource(identifier, row))
        result = cls(
            parsed,
            contract_version=contract_version,
            manifest_version=manifest_version,
            resource_reader=resource_reader,
        )
        result.validate_imports()
        return result

    @property
    def resources(self) -> tuple[SemanticResource, ...]:
        return tuple(
            resource
            for _, resource in sorted(
                self._by_key.items(), key=lambda item: (item[0][0], item[0][1])
            )
        )

    def resolve(self, identifier: str, version: str | SemanticVersion | None) -> SemanticResource:
        """Resolve one exact local resource; no "latest" fallback exists."""
        if version is None:
            raise VersionRequiredError(
                f"semantic resource {identifier!r} requires an exact semantic version"
            )
        parsed_version = SemanticVersion.parse(version)
        try:
            return self._by_key[(identifier, parsed_version)]
        except KeyError as exc:
            available = ", ".join(str(item.version) for item in self._by_identifier.get(identifier, ()))
            if not available:
                raise ResourceNotFoundError(f"unknown semantic resource {identifier!r}") from exc
            raise IncompatibleSemanticVersionError(
                f"semantic resource {identifier!r} has no version {parsed_version}; "
                f"available local versions: {available}"
            ) from exc

    # Verbose alias for callers whose inputs are called resource references.
    resolve_resource = resolve

    def resolve_import_closure(
        self,
        roots: ResourceRequirement | Sequence[ResourceRequirement],
        *,
        allow_experimental: bool = False,
    ) -> tuple[SemanticResource, ...]:
        """Return a selected dependency closure, gating experimental artifacts."""
        closure = self._resolve_import_closure(roots)
        self._require_experimental_opt_in(closure, allow_experimental)
        return closure

    def _resolve_import_closure(
        self, roots: ResourceRequirement | Sequence[ResourceRequirement]
    ) -> tuple[SemanticResource, ...]:
        """Resolve a closure for internal manifest validation without selection policy."""
        requested_roots = (roots,) if isinstance(roots, ResourceRequirement) else tuple(roots)
        if not requested_roots:
            raise KnowledgeRegistryError("at least one semantic resource root is required")

        selected: dict[str, SemanticResource] = {}
        visiting: list[SemanticResource] = []
        completed: set[tuple[str, SemanticVersion]] = set()
        ordered: list[SemanticResource] = []

        def select(requirement: ResourceRequirement) -> SemanticResource:
            candidates = [
                resource
                for resource in self._by_identifier.get(requirement.identifier, ())
                if requirement.version_constraint.matches(resource.version)
            ]
            if not candidates:
                if requirement.identifier not in self._by_identifier:
                    raise ResourceNotFoundError(
                        f"unknown semantic import {requirement.identifier!r} requested by registry"
                    )
                available = ", ".join(
                    str(resource.version) for resource in self._by_identifier[requirement.identifier]
                )
                raise IncompatibleSemanticVersionError(
                    f"semantic import {requirement} is incompatible with local versions: {available}"
                )
            if len(candidates) > 1 and not requirement.version_constraint.is_exact:
                versions = ", ".join(str(resource.version) for resource in candidates)
                raise AmbiguousSemanticVersionError(
                    f"semantic import {requirement} matches multiple local versions ({versions}); "
                    "imports must be exact to remain reproducible"
                )
            candidate = candidates[0]
            prior = selected.get(requirement.identifier)
            if prior is not None and prior.version != candidate.version:
                raise IncompatibleSemanticVersionError(
                    f"semantic import version conflict for {requirement.identifier!r}: "
                    f"{prior.version} and {candidate.version}"
                )
            selected[requirement.identifier] = candidate
            return candidate

        def visit(requirement: ResourceRequirement) -> None:
            resource = select(requirement)
            key = (resource.identifier, resource.version)
            if key in completed:
                return
            if resource in visiting:
                start = visiting.index(resource)
                cycle = [*visiting[start:], resource]
                raise ImportCycleError(
                    "semantic resource import cycle: " + " -> ".join(item.key for item in cycle)
                )
            visiting.append(resource)
            for imported in resource.imports:
                visit(imported)
            visiting.pop()
            completed.add(key)
            ordered.append(resource)

        for root in requested_roots:
            visit(root)
        return tuple(ordered)

    # Alias reads naturally at call sites that name an ontology/root directly.
    resolve_imports = resolve_import_closure

    def validate_imports(self) -> None:
        """Fail manifest loading when any declared local import is invalid."""
        for resource in self.resources:
            self._resolve_import_closure(
                ResourceRequirement.exact(resource.identifier, resource.version)
            )

    def resolve_capability(
        self,
        identifier: str,
        version: str | SemanticVersion | None,
        *,
        allow_experimental: bool = False,
    ) -> ResolvedSemanticCapability:
        resource = self.resolve(identifier, version)
        closure = self.resolve_import_closure(
            ResourceRequirement.exact(resource.identifier, resource.version),
            allow_experimental=allow_experimental,
        )
        self.verify_resources(closure)
        return ResolvedSemanticCapability(resource=resource, import_closure=closure)

    def select_capability(
        self,
        capability: str,
        *,
        allow_experimental: bool = False,
    ) -> ResolvedSemanticCapability:
        """Select one manifest-declared capability without inventing a version check."""
        try:
            resource = self._capabilities[capability]
        except KeyError as exc:
            raise ResourceNotFoundError(f"unknown semantic capability {capability!r}") from exc
        return self.resolve_capability(
            resource.identifier,
            resource.version,
            allow_experimental=allow_experimental,
        )

    def resolve_capability_contract(
        self,
        requirements: Sequence[ResourceRequirement],
        *,
        allow_experimental: bool = False,
    ) -> SemanticCapabilityContract:
        """Build the common typed pin contract consumed by semantic subsystems."""
        roots = tuple(
            ResourceRequirement.exact(
                requirement.identifier,
                _exact_version(requirement),
            )
            for requirement in requirements
        )
        closure = self.resolve_import_closure(
            roots,
            allow_experimental=allow_experimental,
        )
        self.verify_resources(closure)
        capabilities = tuple(
            self.resolve_capability(
                requirement.identifier,
                _exact_version(requirement),
                allow_experimental=allow_experimental,
            )
            for requirement in requirements
        )
        return SemanticCapabilityContract(
            contract_version=self.contract_version,
            capabilities=capabilities,
            artifact_pins=tuple(resource.pin for resource in closure),
        )

    def verify_resources(self, entries: Iterable[SemanticResource] | None = None) -> None:
        """Check every selected package resource exists and matches its digest.

        A named line-ending cause never softens this: the digest is the pin, so
        verification still fails on the first offending resource.  It only says
        which kind of wrong it is.
        """
        if self._resource_reader is None:
            raise MissingPackageResourceError("semantic registry has no package-resource reader")
        for resource in entries if entries is not None else self.resources:
            finding = self._inspect_resource(resource)
            if finding is None:
                continue
            if finding.issue is ResourceIntegrityIssue.MISSING:
                raise MissingPackageResourceError(finding.describe()) from finding.read_error
            raise ResourceDigestMismatchError(finding.describe())

    def audit_resource_digests(
        self, entries: Iterable[SemanticResource] | None = None
    ) -> tuple[ResourceIntegrityFinding, ...]:
        """Report every failing resource instead of raising on the first.

        ``verify_resources`` is the boot gate and must fail closed immediately.
        This is the diagnostic view behind it: seeing that *all* pinned
        resources failed at once is what distinguishes one edited file from a
        whole checkout smudged by ``core.autocrlf``.
        """
        if self._resource_reader is None:
            raise MissingPackageResourceError("semantic registry has no package-resource reader")
        findings = (
            self._inspect_resource(resource)
            for resource in (entries if entries is not None else self.resources)
        )
        return tuple(finding for finding in findings if finding is not None)

    def _inspect_resource(self, resource: SemanticResource) -> ResourceIntegrityFinding | None:
        """Return why one resource fails its pin, or ``None`` when it matches."""
        assert self._resource_reader is not None  # guarded by both public callers
        try:
            content = self._resource_reader(resource.package_resource)
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            return ResourceIntegrityFinding(
                key=resource.key,
                package_resource=resource.package_resource,
                issue=ResourceIntegrityIssue.MISSING,
                expected_sha256=resource.sha256,
                read_error=exc,
            )
        actual = hashlib.sha256(content).hexdigest()
        if actual == resource.sha256:
            return None
        return ResourceIntegrityFinding(
            key=resource.key,
            package_resource=resource.package_resource,
            issue=classify_digest_mismatch(content, resource.sha256),
            expected_sha256=resource.sha256,
            actual_sha256=actual,
        )

    def read_verified_resource(self, resource: SemanticResource) -> bytes:
        """Return one registry resource only after checking its immutable pin.

        Semantic consumers must not open package files by a caller-provided
        path.  Keeping this read behind the registry preserves the same exact
        resource, digest, and offline boundary used for capability selection.
        """
        registered = self.resolve(resource.identifier, resource.version)
        if registered != resource:
            raise KnowledgeRegistryError(
                f"semantic resource {resource.key} does not match the registered pin"
            )
        self.verify_resources((registered,))
        if self._resource_reader is None:  # pragma: no cover - guarded above.
            raise MissingPackageResourceError("semantic registry has no package-resource reader")
        try:
            return self._resource_reader(registered.package_resource)
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            raise MissingPackageResourceError(
                f"{registered.key} package resource is missing: {registered.package_resource}"
            ) from exc

    def _validate_metadata(self) -> None:
        namespaces: dict[str, SemanticResource] = {}
        for resource in self._by_key.values():
            if not resource.identifier or "@" in resource.identifier:
                raise KnowledgeRegistryError(f"invalid semantic resource identifier {resource.identifier!r}")
            if resource.namespace in namespaces:
                prior = namespaces[resource.namespace]
                # An ontology can retain term IRIs across its own successive
                # versions (its version IRI is distinct).  A second resource
                # identifier claiming that namespace is ambiguous, however.
                if prior.identifier != resource.identifier:
                    raise DuplicateNamespaceError(
                        f"duplicate semantic namespace {resource.namespace!r}: "
                        f"{prior.key} and {resource.key}"
                    )
            else:
                namespaces[resource.namespace] = resource
            if not _SHA256_RE.fullmatch(resource.sha256):
                raise KnowledgeRegistryError(f"{resource.key} has an invalid sha256 pin")
            if not _safe_package_resource(resource.package_resource):
                raise KnowledgeRegistryError(
                    f"{resource.key} has unsafe package resource path {resource.package_resource!r}"
                )
            parsed = urlparse(resource.uri)
            if parsed.scheme != "https" or not parsed.netloc:
                raise KnowledgeRegistryError(f"{resource.key} must use an absolute HTTPS stable URI")
            if resource.uri.rstrip("/").endswith("latest"):
                raise KnowledgeRegistryError(f"{resource.key} cannot pin a mutable latest URI")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", resource.published_date):
                raise KnowledgeRegistryError(f"{resource.key} has an invalid published_date")
            if not resource.description:
                raise KnowledgeRegistryError(f"{resource.key} must describe its selected use")
            if resource.maturity is StandardsMaturity.EXPERIMENTAL and not resource.capabilities:
                raise KnowledgeRegistryError(
                    f"experimental resource {resource.key} must declare an explicit capability"
                )

    @staticmethod
    def _require_experimental_opt_in(
        closure: Sequence[SemanticResource], allow_experimental: bool
    ) -> None:
        experimental = [resource.key for resource in closure if resource.maturity is StandardsMaturity.EXPERIMENTAL]
        if experimental and not allow_experimental:
            raise ExperimentalCapabilityError(
                "semantic import closure contains experimental resources: "
                + ", ".join(experimental)
                + "; pass allow_experimental=True to select them"
            )


def _parse_resource(identifier: str, row: Mapping[str, object]) -> SemanticResource:
    required = (
        "version",
        "namespace",
        "package_resource",
        "sha256",
        "maturity",
        "kind",
        "uri",
        "published_date",
        "description",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise KnowledgeRegistryError(
            f"semantic resource {identifier!r} missing required fields: {', '.join(missing)}"
        )
    # TOML table names must be unique, while a semantic resource identifier is
    # intentionally stable across immutable releases.  A version-qualified
    # table can therefore name its durable identifier explicitly; older rows
    # retain the concise table-name form unchanged.
    resource_identifier = (
        _string_field(identifier, row, "identifier") if "identifier" in row else identifier
    )
    return SemanticResource(
        identifier=resource_identifier,
        version=SemanticVersion.parse(_string_field(resource_identifier, row, "version")),
        namespace=_string_field(resource_identifier, row, "namespace"),
        package_resource=_string_field(resource_identifier, row, "package_resource"),
        sha256=_string_field(resource_identifier, row, "sha256"),
        maturity=_enum_field(resource_identifier, row, "maturity", StandardsMaturity),
        kind=_enum_field(resource_identifier, row, "kind", ResourceKind),
        uri=_string_field(resource_identifier, row, "uri"),
        published_date=_string_field(resource_identifier, row, "published_date"),
        description=_string_field(resource_identifier, row, "description"),
        selected_terms=_string_tuple(resource_identifier, row, "selected_terms"),
        imports=tuple(
            ResourceRequirement.parse(value)
            for value in _string_tuple(resource_identifier, row, "imports")
        ),
        capabilities=_string_tuple(resource_identifier, row, "capabilities"),
    )


def _string_field(identifier: str, row: Mapping[str, object], field: str) -> str:
    value = row[field]
    if not isinstance(value, str) or not value:
        raise KnowledgeRegistryError(f"semantic resource {identifier!r} field {field!r} must be a string")
    return value


def _enum_field(identifier: str, row: Mapping[str, object], field: str, enum_type: type[Enum]):
    value = _string_field(identifier, row, field)
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise KnowledgeRegistryError(
            f"semantic resource {identifier!r} field {field!r} must be one of: {choices}"
        ) from exc


def _string_tuple(identifier: str, row: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = row.get(field, ())
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise KnowledgeRegistryError(
            f"semantic resource {identifier!r} field {field!r} must be an array of strings"
        )
    return tuple(value)


def _safe_package_resource(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and value != ""


def _exact_version(requirement: ResourceRequirement) -> SemanticVersion:
    if not requirement.version_constraint.is_exact:
        raise AmbiguousSemanticVersionError(
            f"capability contract requires an exact resource version, got {requirement}"
        )
    return requirement.version_constraint.clauses[0][1]


def _read_packaged_resource(resource_name: str) -> bytes:
    if not _safe_package_resource(resource_name):
        raise FileNotFoundError(resource_name)
    target = resources.files(PACKAGE_NAME)
    for part in Path(resource_name).parts:
        target = target.joinpath(part)
    return target.read_bytes()


def _decode_manifest(manifest_bytes: bytes, source: str) -> str:
    """Decode manifest bytes, reporting failure as a registry error."""
    try:
        return manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MalformedManifestError(
            f"semantic registry manifest is not valid UTF-8 ({source}): {exc}"
        ) from exc


def _parse_manifest_toml(text: str, source: str) -> Mapping[str, object]:
    """Parse a manifest, reporting failure as a registry error.

    Every way a manifest can be unreadable belongs to one error contract, so a
    caller that already handles ``KnowledgeRegistryError`` — the boot gate, the
    doctor preflight, the refresh CLI — never has to also catch the TOML
    decoder's own exception type to stay on its feet.

    ``tomllib.TOMLDecodeError`` is a sibling of ``KnowledgeRegistryError`` under
    ``ValueError``, not a subclass of it, so letting it escape would end the
    doctor and ``setup --check`` in a traceback — the same shape of unexplained
    failure this module is meant to remove.
    """
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise MalformedManifestError(
            f"semantic registry manifest is not valid TOML ({source}): {exc}"
        ) from exc


def _load_unverified_registry(manifest_path: Path | None = None) -> SemanticKnowledgeRegistry:
    """Parse and structurally validate a registry without byte-checking pins."""
    if manifest_path is None:
        manifest_bytes = _read_packaged_resource(MANIFEST_RESOURCE)
        resource_reader = _read_packaged_resource
        source = MANIFEST_RESOURCE
    else:
        package_root = _package_root_for_manifest(manifest_path)
        manifest_bytes = manifest_path.read_bytes()
        resource_reader = _source_tree_resource_reader(package_root)
        source = str(manifest_path)

    manifest = _parse_manifest_toml(_decode_manifest(manifest_bytes, source), source)
    return SemanticKnowledgeRegistry.from_manifest(manifest, resource_reader=resource_reader)


def load_knowledge_registry(manifest_path: Path | None = None) -> SemanticKnowledgeRegistry:
    """Load, validate, and byte-check an immutable package registry.

    Runtime callers omit ``manifest_path`` and always load the installed
    package's registry.  The explicit path exists for the developer refresh
    workflow and its tests; it preserves the same package-resource verification
    instead of treating a checkout manifest as an unverified text file.
    """
    registry = _load_unverified_registry(manifest_path)
    registry.verify_resources()
    return registry


def audit_semantic_resources(
    manifest_path: Path | None = None,
) -> tuple[ResourceIntegrityFinding, ...]:
    """Diagnose every pinned resource, returning one finding per failure.

    This exists so ``kestrel doctor`` can answer the same question the boot
    gate answers — from the same classifier — before a mismatch presents as an
    opaque total agent-boot failure.  An empty tuple means every pin verified.
    """
    return _load_unverified_registry(manifest_path).audit_resource_digests()


@lru_cache(maxsize=1)
def get_knowledge_registry() -> SemanticKnowledgeRegistry:
    """Return the process-local immutable semantic registry."""
    return load_knowledge_registry()


def refresh_manifest_digest(
    manifest_path: Path,
    *,
    identifier: str,
    version: str | SemanticVersion,
    snapshot_path: Path,
) -> str:
    """Explicitly refresh one pin from a reviewed local snapshot.

    This is deliberately a developer action against an explicit source-tree
    manifest.  It does not contact the network and is not used by runtime
    loading.  The reviewed snapshot must already replace the registered package
    resource; this function then updates the matching digest.  That precondition
    prevents a refresh from writing a manifest that the next registry load
    immediately rejects.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"semantic registry manifest not found: {manifest_path}")
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"semantic snapshot not found: {snapshot_path}")
    package_root = _package_root_for_manifest(manifest_path)
    requested_version = str(SemanticVersion.parse(version))
    text = manifest_path.read_text(encoding="utf-8")
    parsed = _parse_manifest_toml(text, str(manifest_path))
    registry = SemanticKnowledgeRegistry.from_manifest(parsed)
    resource = registry.resolve(identifier, requested_version)
    snapshot = snapshot_path.read_bytes()
    package_resource_path = package_root.joinpath(*Path(resource.package_resource).parts)
    try:
        packaged_bytes = package_resource_path.read_bytes()
    except FileNotFoundError as exc:
        raise MissingPackageResourceError(
            f"{resource.key} package resource is missing: {resource.package_resource}"
        ) from exc
    if snapshot != packaged_bytes:
        raise ResourceDigestMismatchError(
            f"reviewed snapshot does not match {resource.key} package resource "
            f"{resource.package_resource}; replace that resource before refreshing its digest"
        )
    digest = hashlib.sha256(snapshot).hexdigest()

    resource_rows = parsed["resource"]
    matching_table_names: list[str] = []
    for table_name, row in resource_rows.items():
        if not isinstance(table_name, str) or not isinstance(row, Mapping):
            continue
        table_resource = _parse_resource(table_name, row)
        if (
            table_resource.identifier == resource.identifier
            and table_resource.version == resource.version
        ):
            matching_table_names.append(table_name)
    if len(matching_table_names) != 1:
        raise KnowledgeRegistryError(
            f"could not locate a unique manifest table for {resource.key} in {manifest_path}"
        )
    escaped_table_name = re.escape(matching_table_names[0])
    # TOML permits both a bare dotted-key segment and a quoted segment. The
    # shipped manifest uses bare segments, while a developer-maintained one
    # may quote a table name that needs it.  The table name, rather than the
    # durable logical identifier, distinguishes immutable versioned releases.
    header = rf'\[resource\.(?:"{escaped_table_name}"|{escaped_table_name})\]'
    block = re.compile(
        rf"(?ms)^({header}\n.*?^sha256\s*=\s*)\"[0-9a-f]+\"(?=\s*(?:#.*)?$)(.*?)(?=^\[resource\.|\Z)"
    )
    match = block.search(text)
    if match is None:
        raise KnowledgeRegistryError(
            f"could not locate sha256 field for {resource.key} in {manifest_path}"
        )
    updated = text[: match.start()] + match.group(1) + f'"{digest}"' + match.group(2) + text[match.end() :]
    manifest_path.write_text(updated, encoding="utf-8")
    return digest


def _package_root_for_manifest(manifest_path: Path) -> Path:
    """Return the package root for a source-tree ``data/semantic/registry.toml``."""
    resolved = manifest_path.resolve()
    semantic_dir = resolved.parent
    if (
        resolved.name != "registry.toml"
        or semantic_dir.name != "semantic"
        or semantic_dir.parent.name != "data"
    ):
        raise KnowledgeRegistryError(
            "semantic registry manifest must live at "
            "<package-root>/data/semantic/registry.toml"
        )
    return semantic_dir.parent.parent


def _source_tree_resource_reader(package_root: Path) -> ResourceReader:
    def read_resource(resource_name: str) -> bytes:
        if not _safe_package_resource(resource_name):
            raise FileNotFoundError(resource_name)
        return package_root.joinpath(*Path(resource_name).parts).read_bytes()

    return read_resource


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or refresh Kestrel's offline semantic registry")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="verify all packaged semantic resources and pins")
    refresh = subparsers.add_parser(
        "refresh", help="update one digest from an explicitly supplied local snapshot"
    )
    refresh.add_argument("--manifest", type=Path, required=True)
    refresh.add_argument("--resource", required=True)
    refresh.add_argument("--version", required=True)
    refresh.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            registry = load_knowledge_registry()
            print(f"semantic registry OK: {len(registry.resources)} pinned package resources")
            return 0
        digest = refresh_manifest_digest(
            args.manifest,
            identifier=args.resource,
            version=args.version,
            snapshot_path=args.snapshot,
        )
        print(f"updated {args.resource}@{args.version} sha256:{digest}")
        return 0
    except (KnowledgeRegistryError, OSError, UnicodeDecodeError) as exc:
        print(f"semantic registry check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through the module command.
    raise SystemExit(_main())
