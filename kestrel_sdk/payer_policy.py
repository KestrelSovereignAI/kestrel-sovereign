"""PayerPolicy primitives.

This local overlay keeps Kestrel's payer-policy contract forward-compatible
with the external SDK while adding ``USER_BYOK`` before the next SDK release.

``USER_BYOK`` is zero-knowledge BYOK: the user supplies a passphrase per
request, Kestrel derives a wrapping key from that passphrase, decrypts the
provider credential for that request only, and never stores a
platform-decryptable copy. Because the platform cannot read the user's key
without the passphrase, this kind explicitly forgoes platform-side child-key
minting, caps, and rotation.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Iterator, Optional, Protocol


class PayerPolicyError(Exception):
    """Base error for payer-policy validation and resolution failures."""


class SupportStatus(StrEnum):
    READY = "ready"
    NOT_IMPLEMENTED = "not_implemented"
    OUT_OF_SCOPE = "out_of_scope"
    NOT_APPLICABLE = "not_applicable"


class ResourceClass(StrEnum):
    LLM = "llm"
    STORAGE = "storage"
    COMPUTE = "compute"
    TOOLS = "tools"
    COMMS = "comms"


class PayerKind(StrEnum):
    NONE = "none"
    HOST_ENV = "host_env"
    HOST_MASTER_PROVISIONED = "host_master_provisioned"
    USER_MASTER_PROVISIONED = "user_master_provisioned"
    USER_BYOK = "user_byok"
    SELF_WALLET = "self_wallet"
    SPONSOR = "sponsor"


@dataclass(frozen=True)
class PayerSpec:
    vendor: str
    kind: PayerKind
    master_did: Optional[str] = None
    monthly_cap_usd: Optional[Decimal] = None

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", PayerKind(self.kind))
        if self.monthly_cap_usd is not None and not isinstance(
            self.monthly_cap_usd, Decimal
        ):
            object.__setattr__(self, "monthly_cap_usd", Decimal(str(self.monthly_cap_usd)))
        if self.kind in (PayerKind.USER_MASTER_PROVISIONED, PayerKind.SPONSOR):
            if not self.master_did:
                raise ValueError(f"{self.kind.value} requires master_did")
        elif self.master_did is not None:
            raise ValueError(f"{self.kind.value} must not set master_did")
        if self.kind is PayerKind.USER_BYOK and self.monthly_cap_usd is not None:
            raise ValueError("user_byok cannot set monthly_cap_usd; no platform caps")

    def to_toml_section(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "vendor": self.vendor,
            "kind": self.kind.value,
        }
        if self.master_did is not None:
            data["master_did"] = self.master_did
        if self.monthly_cap_usd is not None:
            data["monthly_cap_usd"] = str(self.monthly_cap_usd)
        return data

    @classmethod
    def from_toml_section(cls, section: dict[str, Any]) -> "PayerSpec":
        return cls(
            vendor=str(section["vendor"]),
            kind=PayerKind(str(section["kind"])),
            master_did=section.get("master_did"),
            monthly_cap_usd=section.get("monthly_cap_usd"),
        )


@dataclass(frozen=True)
class PayerPolicy:
    llm: PayerSpec
    storage: PayerSpec
    compute: PayerSpec
    tools: PayerSpec
    comms: PayerSpec

    @classmethod
    def host_env_default(cls) -> "PayerPolicy":
        return cls(
            llm=PayerSpec(vendor="openrouter", kind=PayerKind.HOST_ENV),
            storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
            compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        )

    def _iter_specs(self) -> Iterator[tuple[ResourceClass, PayerSpec]]:
        yield ResourceClass.LLM, self.llm
        yield ResourceClass.STORAGE, self.storage
        yield ResourceClass.COMPUTE, self.compute
        yield ResourceClass.TOOLS, self.tools
        yield ResourceClass.COMMS, self.comms

    def validate_against_matrix(self) -> None:
        for resource_class, spec in self._iter_specs():
            status = status_for(resource_class, spec.vendor, spec.kind)
            if status is not SupportStatus.READY:
                raise UnsupportedCombinationError(
                    resource_class=resource_class,
                    vendor=spec.vendor,
                    kind=spec.kind,
                    status=status,
                )

    def to_toml_section(self) -> dict[str, Any]:
        return {
            resource_class.value: spec.to_toml_section()
            for resource_class, spec in self._iter_specs()
        }

    @classmethod
    def from_toml_section(cls, section: dict[str, Any]) -> "PayerPolicy":
        return cls(
            llm=PayerSpec.from_toml_section(section["llm"]),
            storage=PayerSpec.from_toml_section(section["storage"]),
            compute=PayerSpec.from_toml_section(section["compute"]),
            tools=PayerSpec.from_toml_section(section["tools"]),
            comms=PayerSpec.from_toml_section(section["comms"]),
        )


@dataclass(frozen=True)
class ResolvedResource:
    enabled: bool
    key_resolver: Any = None

    @classmethod
    def disabled(cls) -> "ResolvedResource":
        return cls(enabled=False, key_resolver=None)


class PayerResolver(Protocol):
    async def resolve_for(
        self,
        agent_did: str,
        resource_class: ResourceClass,
        *,
        user_passphrase: str | None = None,
    ) -> ResolvedResource: ...


@dataclass(frozen=True)
class SupportMatrixEntry:
    resource_class: ResourceClass
    vendor: str
    kind: PayerKind
    status: SupportStatus
    notes: str = ""


class UnsupportedCombinationError(PayerPolicyError):
    def __init__(
        self,
        *,
        resource_class: ResourceClass,
        vendor: str,
        kind: PayerKind,
        status: SupportStatus,
    ) -> None:
        self.resource_class = resource_class
        self.vendor = vendor
        self.kind = kind
        self.status = status
        super().__init__(
            f"Unsupported payer combination: resource={resource_class.value}, "
            f"vendor={vendor}, kind={kind.value}, status={status.value}"
        )


def _key(
    resource_class: ResourceClass,
    vendor: str,
    kind: PayerKind,
) -> tuple[ResourceClass, str, PayerKind]:
    return (resource_class, vendor, kind)


SUPPORT_MATRIX: dict[tuple[ResourceClass, str, PayerKind], SupportStatus] = {}
SUPPORT_MATRIX_NOTES: dict[tuple[ResourceClass, str, PayerKind], str] = {}


def _set(
    resource_class: ResourceClass,
    vendor: str,
    kind: PayerKind,
    status: SupportStatus,
    notes: str = "",
) -> None:
    matrix_key = _key(resource_class, vendor, kind)
    SUPPORT_MATRIX[matrix_key] = status
    if notes:
        SUPPORT_MATRIX_NOTES[matrix_key] = notes


for _resource in ResourceClass:
    for _kind in PayerKind:
        _set(_resource, "*", _kind, SupportStatus.NOT_IMPLEMENTED)
    _set(_resource, "*", PayerKind.NONE, SupportStatus.READY)
    _set(_resource, "*", PayerKind.HOST_ENV, SupportStatus.READY)

_set(ResourceClass.LLM, "openrouter", PayerKind.NONE, SupportStatus.READY)
_set(ResourceClass.LLM, "openrouter", PayerKind.HOST_ENV, SupportStatus.READY)
_set(ResourceClass.LLM, "openrouter", PayerKind.HOST_MASTER_PROVISIONED, SupportStatus.READY)
_set(ResourceClass.LLM, "openrouter", PayerKind.USER_MASTER_PROVISIONED, SupportStatus.READY)
_set(
    ResourceClass.LLM,
    "openrouter",
    PayerKind.USER_BYOK,
    SupportStatus.READY,
    "Zero-knowledge passphrase BYOK; no platform minting, caps, or rotation.",
)
_set(ResourceClass.LLM, "openrouter", PayerKind.SPONSOR, SupportStatus.READY)
_set(ResourceClass.LLM, "openrouter", PayerKind.SELF_WALLET, SupportStatus.NOT_IMPLEMENTED)

_set(ResourceClass.LLM, "local", PayerKind.HOST_ENV, SupportStatus.READY)
_set(ResourceClass.LLM, "local", PayerKind.NONE, SupportStatus.READY)

_set(ResourceClass.STORAGE, "lighthouse", PayerKind.HOST_ENV, SupportStatus.READY)
_set(ResourceClass.STORAGE, "lighthouse", PayerKind.SELF_WALLET, SupportStatus.READY)
_set(ResourceClass.STORAGE, "lighthouse", PayerKind.NONE, SupportStatus.READY)
_set(ResourceClass.STORAGE, "local-disk", PayerKind.HOST_ENV, SupportStatus.READY)
_set(ResourceClass.STORAGE, "local-disk", PayerKind.NONE, SupportStatus.READY)


def status_for(
    resource_class: ResourceClass,
    vendor: str,
    kind: PayerKind,
) -> SupportStatus:
    """Return the support status for a (resource, vendor, kind) triple.

    Looks up the exact triple in SUPPORT_MATRIX. If not found, returns
    NOT_IMPLEMENTED. The wildcard vendor "*" is only matched when
    explicitly requested in the policy (e.g., compute/tools/comms), not
    as a fallback for unknown vendors.
    """
    if isinstance(resource_class, str):
        resource_class = ResourceClass(resource_class)
    if isinstance(kind, str):
        kind = PayerKind(kind)
    return SUPPORT_MATRIX.get(
        _key(resource_class, vendor, kind),
        SupportStatus.NOT_IMPLEMENTED,
    )


def is_offerable(
    resource_class: ResourceClass,
    vendor: str,
    kind: PayerKind,
) -> bool:
    return status_for(resource_class, vendor, kind) is SupportStatus.READY


def supported_kinds_for(
    resource_class: ResourceClass,
    vendor: str,
) -> list[PayerKind]:
    return [kind for kind in PayerKind if is_offerable(resource_class, vendor, kind)]


SUPPORTED_PAYER_COMBINATIONS = SUPPORT_MATRIX


__all__ = [
    "PayerKind",
    "PayerPolicy",
    "PayerPolicyError",
    "PayerResolver",
    "PayerSpec",
    "ResolvedResource",
    "ResourceClass",
    "SUPPORTED_PAYER_COMBINATIONS",
    "SUPPORT_MATRIX",
    "SUPPORT_MATRIX_NOTES",
    "SupportMatrixEntry",
    "SupportStatus",
    "UnsupportedCombinationError",
    "is_offerable",
    "status_for",
    "supported_kinds_for",
]
