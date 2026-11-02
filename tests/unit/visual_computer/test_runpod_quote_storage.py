"""Private SAME-row quote envelope bounds; these facts confer no IO grant."""

import base64
import hashlib

import pytest
from kestrel_sdk.computer_use import EnvironmentCapabilities
from pydantic import SecretBytes, ValidationError

from kestrel_sovereign.visual_computer.runpod_quote_storage import (
    MAX_RUNPOD_PLAN_PRIVATE_BYTES,
    _bytes,
    _caps,
    _RunpodControlPlaneBinding,
    _RunpodEnvironmentEnvelope,
)


@pytest.fixture
def envelope():
    caps = EnvironmentCapabilities(
        provider_id="runpod-cpu",
        location="remote",
        surfaces=frozenset({"browser"}),
        operations=frozenset({"screenshot"}),
    )
    raw = b"PRIVATE-ORIGINAL-PLAN-FACTS"
    binding = _RunpodControlPlaneBinding(
        account="PRIVATE-ACCOUNT",
        base_url="https://owned.invalid/v2",
        credential_selector="PRIVATE-CREDENTIAL-SELECTOR",
        credential_fingerprint="a" * 64,
        tls_sha256="b" * 64,
        source_sha256="c" * 64,
    )
    return _RunpodEnvironmentEnvelope(
        plan_bytes=SecretBytes(raw),
        plan_storage_sha256=hashlib.sha256(raw).hexdigest(),
        plan_digest="d" * 64,
        capabilities=caps,
        capabilities_sha256=hashlib.sha256(_bytes(_caps(caps))).hexdigest(),
        quote_binding_sha256="e" * 64,
        binding=binding,
    )


def test_exact_private_envelope_roundtrip_and_public_omission(envelope):
    private = envelope.private_value()
    restored = _RunpodEnvironmentEnvelope.from_private(private)
    assert restored.private_value() == private
    assert restored.plan_bytes.get_secret_value() == b"PRIVATE-ORIGINAL-PLAN-FACTS"
    assert (
        envelope.model_dump() == {"schema_version": 1}
        and envelope.binding.model_dump() == {}
    )
    assert (
        "PRIVATE-"
        not in repr(envelope) + repr(envelope.binding) + envelope.model_dump_json()
    )
    with pytest.raises(ValidationError) as invalid:
        _RunpodControlPlaneBinding.model_validate(
            {**private["binding"], "account": {"PRIVATE-INPUT": 1}}
        )
    assert "PRIVATE-INPUT" not in str(invalid.value)


@pytest.mark.parametrize("extra", [0, 1], ids=["exact", "plus-one"])
def test_original_plan_byte_limit(envelope, extra):
    raw = b"x" * (MAX_RUNPOD_PLAN_PRIVATE_BYTES + extra)
    value = envelope.model_copy(
        update={
            "plan_bytes": SecretBytes(raw),
            "plan_storage_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    if extra:
        with pytest.raises(ValueError, match="plan exceeds byte bounds"):
            value.private_value()
    else:
        assert (
            _RunpodEnvironmentEnvelope.from_private(
                value.private_value()
            ).plan_bytes.get_secret_value()
            == raw
        )


@pytest.mark.parametrize(
    "change", ["plan", "caps", "version", "extra", "boolean-version"]
)
def test_private_envelope_changed_facts_refuse(envelope, change):
    value = envelope.private_value()
    if change == "plan":
        value["plan_bytes"] = base64.b64encode(b"changed valid bytes").decode()
    elif change == "caps":
        value["capabilities"]["operations"] = ["click"]
    elif change == "version":
        value["schema_version"] = 2
    elif change == "boolean-version":
        value["schema_version"] = True
    elif change == "extra":
        value["grant"] = True
    with pytest.raises(ValueError):
        _RunpodEnvironmentEnvelope.from_private(value)


def test_bounded_private_fields_refuse_before_base64_allocation(envelope, monkeypatch):
    value = envelope.private_value()
    value["plan_bytes"] = "x" * 43693

    def forbidden(*args, **kwargs):
        pytest.fail("oversize plan reached base64 allocator")

    monkeypatch.setattr(base64, "b64decode", forbidden)
    with pytest.raises(ValueError, match="plan bounds"):
        _RunpodEnvironmentEnvelope.from_private(value)


@pytest.mark.parametrize(
    "value",
    [
        {"a": 1 << 10000},
        {"a": object()},
        {"a": ["x"] * 2049},
        {"a": "x" * (128 * 1024)},
    ],
    ids=["integer", "object", "nodes", "bytes"],
)
def test_private_envelope_preallocation_bounds(value):
    with pytest.raises(ValueError):
        _bytes(value)
