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
from dataclasses import dataclass
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
MANIFEST_RESOURCE = "data/semantic/registry.toml"
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
        """Check every selected package resource exists and matches its digest."""
        if self._resource_reader is None:
            raise MissingPackageResourceError("semantic registry has no package-resource reader")
        for resource in entries if entries is not None else self.resources:
            try:
                content = self._resource_reader(resource.package_resource)
            except (FileNotFoundError, ModuleNotFoundError) as exc:
                raise MissingPackageResourceError(
                    f"{resource.key} package resource is missing: {resource.package_resource}"
                ) from exc
            actual = hashlib.sha256(content).hexdigest()
            if actual != resource.sha256:
                raise ResourceDigestMismatchError(
                    f"{resource.key} digest mismatch for {resource.package_resource}: "
                    f"expected {resource.sha256}, got {actual}"
                )

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


def load_knowledge_registry(manifest_path: Path | None = None) -> SemanticKnowledgeRegistry:
    """Load, validate, and byte-check an immutable package registry.

    Runtime callers omit ``manifest_path`` and always load the installed
    package's registry.  The explicit path exists for the developer refresh
    workflow and its tests; it preserves the same package-resource verification
    instead of treating a checkout manifest as an unverified text file.
    """
    if manifest_path is None:
        manifest_bytes = _read_packaged_resource(MANIFEST_RESOURCE)
        resource_reader = _read_packaged_resource
    else:
        package_root = _package_root_for_manifest(manifest_path)
        manifest_bytes = manifest_path.read_bytes()
        resource_reader = _source_tree_resource_reader(package_root)

    manifest = tomllib.loads(manifest_bytes.decode("utf-8"))
    registry = SemanticKnowledgeRegistry.from_manifest(manifest, resource_reader=resource_reader)
    registry.verify_resources()
    return registry


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
    parsed = tomllib.loads(text)
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
