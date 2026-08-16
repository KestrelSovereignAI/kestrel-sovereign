"""Exact per-agent selection of stable or experimental semantic capabilities.

The registry describes what this build *can* support.  This module describes
what one agent has explicitly enabled.  It is deliberately independent from
assertion content and is parsed before an agent becomes ready, so a typo or a
partial draft selection cannot silently select a nearby capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .rdf_codec import (
    RdfAssertionCodec,
    RdfCodecConfiguration,
    UnsupportedRdfCapabilityError,
)
from .registry import (
    KnowledgeRegistryError,
    ResourceKind,
    SemanticKnowledgeRegistry,
    get_knowledge_registry,
)
from .shacl_validation import ShapeSetReference


class SemanticCapabilityConfigurationError(ValueError):
    """A per-agent semantic capability selection is incomplete or unavailable."""


_STABLE_VALIDATION_CAPABILITY = "validation-profile:shacl-core-20170720"
_STABLE_SHAPE_SET = ShapeSetReference("kestrel-assertion-shapes", "1.0.0")
_EXPERIMENTAL_REQUIRED_KEYS = frozenset(
    {"mode", "rdf12", "sparql12", "shacl12", "shape_set"}
)


@dataclass(frozen=True, slots=True)
class CapabilitySelection:
    """One exact registry selector and its immutable version pin."""

    capability: str
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.capability, str) or not self.capability:
            raise SemanticCapabilityConfigurationError(
                "semantic capability must be non-empty"
            )
        if not isinstance(self.version, str) or not self.version:
            raise SemanticCapabilityConfigurationError(
                "semantic capability version must be non-empty"
            )

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "CapabilitySelection":
        if not isinstance(value, Mapping) or set(value) != {"capability", "version"}:
            raise SemanticCapabilityConfigurationError(
                f"{label} must contain exactly capability and version"
            )
        return cls(value["capability"], value["version"])

    def to_mapping(self) -> dict[str, str]:
        return {"capability": self.capability, "version": self.version}


@dataclass(frozen=True, slots=True)
class SemanticRuntimeCapabilities:
    """Runtime-owned stable/default or all-or-nothing experimental selection."""

    mode: str
    rdf12: CapabilitySelection | None = None
    sparql12: CapabilitySelection | None = None
    shacl12: CapabilitySelection | None = None
    shape_set: ShapeSetReference = _STABLE_SHAPE_SET

    def __post_init__(self) -> None:
        if not isinstance(self.shape_set, ShapeSetReference):
            raise SemanticCapabilityConfigurationError(
                "semantic capability shape_set must be ShapeSetReference"
            )
        if self.mode == "stable":
            if any(
                item is not None for item in (self.rdf12, self.sparql12, self.shacl12)
            ):
                raise SemanticCapabilityConfigurationError(
                    "stable semantic capabilities cannot select experimental drafts"
                )
            if self.shape_set != _STABLE_SHAPE_SET:
                raise SemanticCapabilityConfigurationError(
                    "stable semantic capabilities require the stable assertion "
                    "shape set"
                )
            return
        if self.mode != "experimental":
            raise SemanticCapabilityConfigurationError(
                "semantic capability mode must be stable or experimental"
            )
        if any(
            item is not None and not isinstance(item, CapabilitySelection)
            for item in (self.rdf12, self.sparql12, self.shacl12)
        ):
            raise SemanticCapabilityConfigurationError(
                "experimental semantic capability selections must be "
                "CapabilitySelection values"
            )
        if any(item is None for item in (self.rdf12, self.sparql12, self.shacl12)):
            raise SemanticCapabilityConfigurationError(
                "experimental semantic capabilities require exact rdf12, "
                "sparql12, and shacl12 selections"
            )

    @property
    def allow_experimental(self) -> bool:
        return self.mode == "experimental"

    @classmethod
    def stable(cls) -> "SemanticRuntimeCapabilities":
        return cls("stable")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        *,
        registry: SemanticKnowledgeRegistry | None = None,
    ) -> "SemanticRuntimeCapabilities":
        if not isinstance(config, Mapping):
            raise SemanticCapabilityConfigurationError(
                "[semantic_capabilities] must be a table"
            )
        mode = config.get("mode")
        if mode == "stable":
            if set(config) != {"mode"}:
                raise SemanticCapabilityConfigurationError(
                    "stable semantic capabilities must contain only mode = 'stable'"
                )
            result = cls.stable()
        elif mode == "experimental":
            if set(config) != _EXPERIMENTAL_REQUIRED_KEYS:
                missing = sorted(_EXPERIMENTAL_REQUIRED_KEYS.difference(config))
                unexpected = sorted(set(config).difference(_EXPERIMENTAL_REQUIRED_KEYS))
                detail = []
                if missing:
                    detail.append("missing " + ", ".join(missing))
                if unexpected:
                    detail.append("unsupported " + ", ".join(map(str, unexpected)))
                raise SemanticCapabilityConfigurationError(
                    "experimental semantic capabilities require all exact selections ("
                    + "; ".join(detail)
                    + ")"
                )
            shape_value = config["shape_set"]
            if not isinstance(shape_value, Mapping) or set(shape_value) != {
                "identifier",
                "version",
            }:
                raise SemanticCapabilityConfigurationError(
                    "experimental shape_set must contain exactly identifier and version"
                )
            result = cls(
                "experimental",
                rdf12=CapabilitySelection.from_mapping(config["rdf12"], label="rdf12"),
                sparql12=CapabilitySelection.from_mapping(
                    config["sparql12"], label="sparql12"
                ),
                shacl12=CapabilitySelection.from_mapping(
                    config["shacl12"], label="shacl12"
                ),
                shape_set=ShapeSetReference(
                    shape_value["identifier"], shape_value["version"]
                ),
            )
        else:
            raise SemanticCapabilityConfigurationError(
                "semantic capability mode must be stable or experimental"
            )
        result.validate(registry=registry)
        return result

    def validate(self, *, registry: SemanticKnowledgeRegistry | None = None) -> None:
        """Verify every selected pin and its local resource bytes before startup."""
        registry = registry or get_knowledge_registry()
        if not self.allow_experimental:
            try:
                profile = registry.select_capability(_STABLE_VALIDATION_CAPABILITY)
                shapes = registry.resolve_capability(
                    self.shape_set.identifier, self.shape_set.version
                )
                registry.verify_resources(
                    (*profile.import_closure, *shapes.import_closure)
                )
            except KnowledgeRegistryError as exc:
                # Carry the registry's own sentence forward. The traceback is
                # not what an operator reads in a boot log; this message is.
                raise SemanticCapabilityConfigurationError(
                    f"stable semantic capability resources are unavailable: {exc}"
                ) from exc
            return
        assert (
            self.rdf12 is not None
            and self.sparql12 is not None
            and self.shacl12 is not None
        )
        required = (
            (self.rdf12, "rdf-profile:rdf12", "rdf12"),
            (self.sparql12, "query-profile:sparql12", "sparql12"),
            (self.shacl12, "validation-profile:shacl12", "shacl12"),
        )
        resolved = []
        try:
            for selection, prefix, label in required:
                if not selection.capability.startswith(prefix):
                    raise SemanticCapabilityConfigurationError(
                        f"{label} capability must begin {prefix!r}"
                    )
                selected = registry.select_capability(
                    selection.capability, allow_experimental=True
                )
                if str(selected.resource.version) != selection.version:
                    raise SemanticCapabilityConfigurationError(
                        f"{label} capability version does not match its registry pin"
                    )
                resolved.extend(selected.import_closure)
            shapes = registry.resolve_capability(
                self.shape_set.identifier,
                self.shape_set.version,
                allow_experimental=True,
            )
            if shapes.resource.kind is not ResourceKind.SHAPE_SET:
                raise SemanticCapabilityConfigurationError(
                    "experimental shape_set is not a SHACL shape set"
                )
            shacl = registry.select_capability(
                self.shacl12.capability, allow_experimental=True
            )
            closure_keys = {
                (item.identifier, str(item.version)) for item in shapes.import_closure
            }
            if (
                shacl.resource.identifier,
                str(shacl.resource.version),
            ) not in closure_keys:
                raise SemanticCapabilityConfigurationError(
                    "experimental shape_set does not import the selected "
                    "SHACL 1.2 profile"
                )
            registry.verify_resources((*resolved, *shapes.import_closure))
            # Construct through the same factory owned by the live storage
            # boundary.  This catches an RDF/SPARQL implementation mismatch at
            # boot rather than treating draft selections as diagnostics only.
            self.create_rdf_codec(registry=registry)
        except (KnowledgeRegistryError, UnsupportedRdfCapabilityError) as exc:
            raise SemanticCapabilityConfigurationError(
                "experimental semantic capability is unavailable or has an "
                f"invalid pin: {exc}"
            ) from exc

    def create_rdf_codec(
        self, *, registry: SemanticKnowledgeRegistry | None = None
    ) -> RdfAssertionCodec:
        """Create the RDF/SPARQL implementation bound to this exact selection.

        Callers deliberately own the returned instance for their lifetime.  A
        runtime must not re-parse draft pins on each operation or silently
        substitute a stable codec after boot.
        """
        configuration = RdfCodecConfiguration()
        if self.allow_experimental:
            assert self.rdf12 is not None and self.sparql12 is not None
            configuration = RdfCodecConfiguration(
                rdf12_capability=self.rdf12.capability,
                rdf12_version=self.rdf12.version,
                sparql12_capability=self.sparql12.capability,
                sparql12_version=self.sparql12.version,
            )
        return RdfAssertionCodec(registry=registry, configuration=configuration)

    def rdf_runtime_matches(
        self, report, *, registry: SemanticKnowledgeRegistry | None = None
    ) -> bool:
        """Whether a codec report is the exact runtime selected for this agent."""
        if report.rdf12 is None or report.sparql12 is None:
            return not self.allow_experimental and (
                report.rdf12 is None and report.sparql12 is None
            )
        if not self.allow_experimental:
            return False
        assert self.rdf12 is not None and self.sparql12 is not None
        registry = registry or get_knowledge_registry()
        try:
            rdf12 = registry.select_capability(
                self.rdf12.capability, allow_experimental=True
            ).resource.pin
            sparql12 = registry.select_capability(
                self.sparql12.capability, allow_experimental=True
            ).resource.pin
        except KnowledgeRegistryError:
            return False
        return report.rdf12 == rdf12 and report.sparql12 == sparql12

    @property
    def validation_capability(self) -> str:
        return (
            self.shacl12.capability
            if self.shacl12 is not None
            else _STABLE_VALIDATION_CAPABILITY
        )

    @property
    def validation_profile_version(self) -> str:
        return self.shacl12.version if self.shacl12 is not None else "1.0.0"

    def capability_versions(self) -> dict[str, str]:
        values = {
            "semantic_capability_mode": self.mode,
            "shape_set": f"{self.shape_set.identifier}@{self.shape_set.version}",
            "validation_capability": self.validation_capability,
            "validation_profile_version": self.validation_profile_version,
        }
        if self.rdf12 is not None and self.sparql12 is not None:
            values.update(
                {
                    "rdf12_capability": self.rdf12.capability,
                    "rdf12_version": self.rdf12.version,
                    "sparql12_capability": self.sparql12.capability,
                    "sparql12_version": self.sparql12.version,
                }
            )
        return values

    def to_mapping(self) -> dict[str, object]:
        if not self.allow_experimental:
            return {"mode": "stable"}
        assert (
            self.rdf12 is not None
            and self.sparql12 is not None
            and self.shacl12 is not None
        )
        return {
            "mode": "experimental",
            "rdf12": self.rdf12.to_mapping(),
            "sparql12": self.sparql12.to_mapping(),
            "shacl12": self.shacl12.to_mapping(),
            "shape_set": {
                "identifier": self.shape_set.identifier,
                "version": self.shape_set.version,
            },
        }


def semantic_capabilities_from_config(
    config: Mapping[str, object], *, registry: SemanticKnowledgeRegistry | None = None
) -> SemanticRuntimeCapabilities:
    """Parse the one per-agent semantic capability table without fallbacks."""
    return SemanticRuntimeCapabilities.from_config(config, registry=registry)


__all__ = [
    "CapabilitySelection",
    "SemanticCapabilityConfigurationError",
    "SemanticRuntimeCapabilities",
    "semantic_capabilities_from_config",
]
