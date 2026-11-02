"""Private quote facts on the canonical encrypted request row, never grants."""

import base64
import hashlib
import hmac
import json
from datetime import datetime
from typing import Literal

from kestrel_sdk.computer_use import EnvironmentCapabilities
from pydantic import BaseModel, ConfigDict, Field, SecretBytes, model_validator

MAX_RUNPOD_QUOTE_PRIVATE_BYTES = 128 * 1024
MAX_RUNPOD_PLAN_PRIVATE_BYTES = 32768


def _bytes(value):
    pending, nodes, budget = [(value, 0)], 2048, MAX_RUNPOD_QUOTE_PRIVATE_BYTES
    while pending:
        item, depth = pending.pop()
        nodes -= 1
        if nodes < 0 or depth > 16:
            raise ValueError("private Runpod quote exceeds structural bounds")
        kind = type(item)
        if kind is str:
            budget -= 2
            for char in item:
                code = ord(char)
                budget -= (
                    2
                    if char in {'"', "\\"}
                    else 1
                    if 32 <= code < 127
                    else 6
                    if code <= 65535
                    else 12
                )
                if budget < 0:
                    raise ValueError("private Runpod quote exceeds byte bounds")
        elif kind is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("invalid private Runpod quote key")
            budget -= 2 + 2 * len(item)
            pending.extend((part, depth + 1) for key, part in item.items())
            pending.extend((key, depth + 1) for key in item)
        elif kind in (list, tuple):
            budget -= 2 + len(item)
            pending.extend((part, depth + 1) for part in item)
        elif item is None or kind is bool:
            budget -= 5
        elif kind is int and item.bit_length() <= 64:
            budget -= 21
        else:
            raise ValueError("invalid private Runpod quote scalar")
        if budget < 0:
            raise ValueError("private Runpod quote exceeds byte bounds")
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    if not 0 < len(raw) <= MAX_RUNPOD_QUOTE_PRIVATE_BYTES:
        raise ValueError("private Runpod quote exceeds byte bounds")
    return raw


def _caps(value):
    result = value.model_dump(mode="json")
    result["operations"] = sorted(result["operations"])
    result["surfaces"] = sorted(result["surfaces"])
    return result


class _RunpodControlPlaneBinding(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, hide_input_in_errors=True
    )
    account: str = Field(min_length=1, max_length=1024, repr=False, exclude=True)
    base_url: str = Field(min_length=1, max_length=2048, repr=False, exclude=True)
    credential_selector: str = Field(
        min_length=1, max_length=256, repr=False, exclude=True
    )
    credential_fingerprint: str = Field(
        pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True
    )
    tls_sha256: str = Field(pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True)


class _RunpodPreparationCorrelation(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, hide_input_in_errors=True
    )
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True)
    binding: str = Field(pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True)
    create_sha256: str = Field(pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True)
    worker_identity_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True
    )


class _RunpodInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)
    provider_resource_id: str = Field(min_length=1, max_length=512, repr=False, exclude=True)
    observed_at: datetime = Field(repr=False, exclude=True)
    state: Literal["pending", "running"]

    @model_validator(mode="after")
    def _aware(self):
        if self.observed_at.utcoffset() is None:
            raise ValueError("original inspection time must be aware")
        return self


class _RunpodReleaseFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)
    intent_id: str = Field(pattern=r"^[a-f0-9]{32}$", repr=False, exclude=True)
    provider_resource_id: str = Field(min_length=1, max_length=512, repr=False, exclude=True)
    requested_at: datetime = Field(repr=False, exclude=True)
    state: Literal["intent", "acknowledged"]
    acknowledged_at: datetime | None = Field(default=None, repr=False, exclude=True)
    native_status: Literal[204] | None = None

    @model_validator(mode="after")
    def _complete(self):
        if self.requested_at.utcoffset() is None:
            raise ValueError("original release time must be aware")
        if self.state == "intent":
            if self.acknowledged_at is not None or self.native_status is not None:
                raise ValueError("original release intent has terminal facts")
        elif (self.acknowledged_at is None or self.acknowledged_at.utcoffset() is None
                or self.acknowledged_at < self.requested_at or type(self.native_status) is not int
                or self.native_status != 204):
            raise ValueError("original release acknowledgement is incomplete")
        return self


class _RunpodEnvironmentEnvelope(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, hide_input_in_errors=True
    )
    schema_version: Literal[1] = 1
    plan_bytes: SecretBytes = Field(exclude=True, repr=False)
    plan_storage_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True
    )
    plan_digest: str = Field(pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True)
    capabilities: EnvironmentCapabilities = Field(repr=False, exclude=True)
    capabilities_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True
    )
    quote_binding_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$", repr=False, exclude=True
    )
    binding: _RunpodControlPlaneBinding = Field(repr=False, exclude=True)
    prepared: _RunpodPreparationCorrelation | None = Field(
        default=None, repr=False, exclude=True
    )

    inspection: _RunpodInspection | None = Field(default=None, repr=False, exclude=True)
    release: _RunpodReleaseFact | None = Field(default=None, repr=False, exclude=True)

    def private_value(self):
        raw = self.plan_bytes.get_secret_value()
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_RUNPOD_PLAN_PRIVATE_BYTES:
            raise ValueError("private Runpod plan exceeds byte bounds")
        if not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(), self.plan_storage_sha256
        ):
            raise ValueError("original private Runpod plan storage differs")
        caps = _caps(self.capabilities)
        if len(_bytes(caps)) > 8192 or not hmac.compare_digest(
            hashlib.sha256(_bytes(caps)).hexdigest(), self.capabilities_sha256
        ):
            raise ValueError("original independent Runpod capabilities differ")
        result = {
            "schema_version": self.schema_version,
            "plan_bytes": base64.b64encode(raw).decode("ascii"),
            "plan_storage_sha256": self.plan_storage_sha256,
            "plan_digest": self.plan_digest,
            "capabilities": caps,
            "capabilities_sha256": self.capabilities_sha256,
            "quote_binding_sha256": self.quote_binding_sha256,
            "binding": {
                name: getattr(self.binding, name)
                for name in type(self.binding).model_fields
            },
            "prepared": (
                {
                    name: getattr(self.prepared, name)
                    for name in type(self.prepared).model_fields
                }
                if self.prepared is not None
                else None
            ),
        }
        # Only bounded canonical fields, never raw Pod JSON or billing content.
        for name in ("inspection", "release"):
            fact = getattr(self, name)
            if fact is not None:
                expected = _RunpodInspection if name == "inspection" else _RunpodReleaseFact
                if type(fact) is not expected:
                    raise ValueError("original lifecycle fact type differs")
                fact._aware() if name == "inspection" else fact._complete()
                value = {}
                for field in type(fact).model_fields:
                    part = getattr(fact, field)
                    if isinstance(part, datetime):
                        if part.utcoffset() is None:
                            raise ValueError("original lifecycle time must be aware")
                        part = part.isoformat()
                    value[field] = part
                if name == "release" and ((fact.state == "acknowledged") !=
                        (fact.acknowledged_at is not None and fact.native_status == 204)):
                    raise ValueError("original release fact is incomplete")
                result[name] = value
        _bytes(result)
        return result

    def require_lifecycle_reservation(self):
        # Reserve the maximum fixed lifecycle facts before any native effects.
        # No smaller post-response cap may discard a valid acknowledgement.
        value = self.private_value()
        value.update(inspection={"provider_resource_id": "\U0010ffff" * 512,
            "observed_at": "2" * 48, "state": "running"},
            release={"intent_id": "a" * 32, "provider_resource_id": "\U0010ffff" * 512,
                "requested_at": "2" * 48, "state": "acknowledged",
                "acknowledged_at": "2" * 48, "native_status": 204})
        _bytes(value)

    @classmethod
    def from_private(cls, value):
        if (
            type(value) is not dict
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or len(_bytes(value)) > MAX_RUNPOD_QUOTE_PRIVATE_BYTES
        ):
            raise ValueError("invalid private Runpod quote bounds")
        raw = value.get("plan_bytes")
        if type(raw) is not str or not 0 < len(raw) <= 43692:
            raise ValueError("invalid private Runpod plan bounds")
        values = dict(value)
        values["plan_bytes"] = SecretBytes(base64.b64decode(raw, validate=True))
        values["capabilities"] = EnvironmentCapabilities.model_validate_json(
            json.dumps(value["capabilities"])
        )
        for name, kind in (("inspection", _RunpodInspection), ("release", _RunpodReleaseFact)):
            if values.get(name) is not None:
                values[name] = kind.model_validate_json(json.dumps(values[name]))
        result = cls.model_validate(values)
        result.private_value()
        return result
