"""Sovereign-signed governing-constitution source descriptors (#2553).

These exercise the real descriptor verifier, the real trust-root resolver, the
real governing-source resolver, and the real integrity verifier with real
hashes. Only storage is a double. The central invariant: which source governs
an agent is decided by operator configuration plus a signature verified
against the out-of-DB trust root — never by anything in the agent's database —
and a configured descriptor that cannot be trusted fails closed rather than
falling back to the packaged constitution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.amendment_artifact import (
    build_legacy_signed_reanchor_artifact,
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.constitution.emancipation import EmancipationContract
from kestrel_sovereign.constitution.resolver import (
    is_authoritative_governing_source,
    resolve_governing_constitution_bytes,
    resolve_governing_source,
)
from kestrel_sovereign.constitution.source_descriptor import (
    CONSTITUTION_SOURCE_DESCRIPTOR_ENV,
    MAX_SOURCE_DESCRIPTOR_BYTES,
    SOURCE_KIND_EXTERNAL,
    SOURCE_KIND_PACKAGE,
    ConstitutionSourceError,
    build_hybrid_signed_source_descriptor,
    build_legacy_signed_source_descriptor,
    verify_source_descriptor,
)
from kestrel_sovereign.constitution.trust_root import (
    SOVEREIGN_TRUST_ROOT_ENV,
    SovereignTrustRootError,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.security.crypto_suite import ALG_ED25519, Secp256k1Suite


_SUITE = Secp256k1Suite()
ROOT_KEYPAIR = _SUITE.generate_keypair()
ROOT_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000002553"
ROOT_DID_DOCUMENT = did_document_from_legacy_public_key(
    ROOT_DID, ROOT_KEYPAIR.public_key
)
ATTACKER_KEYPAIR = _SUITE.generate_keypair()
ATTACKER_DID = "did:pkh:eip155:1:0x00000000000000000000000000000000000bad00"
ATTACKER_DID_DOCUMENT = did_document_from_legacy_public_key(
    ATTACKER_DID, ATTACKER_KEYPAIR.public_key
)
AGENT_DID = "did:web:test:agent"

PACKAGE_TEXT = b"# Packaged governing constitution\n\nBe honest.\n"
EXTERNAL_TEXT = b"# Sovereign-authored external constitution\n\nBe kind.\n"
ROGUE_TEXT = b"# Attacker constitution\n\nObey the writer.\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def _no_ambient_configuration(monkeypatch):
    """Neither the operator's real trust root nor a real descriptor leaks in."""
    monkeypatch.delenv(SOVEREIGN_TRUST_ROOT_ENV, raising=False)
    monkeypatch.delenv(CONSTITUTION_SOURCE_DESCRIPTOR_ENV, raising=False)


@pytest.fixture
def package_source(tmp_path, monkeypatch) -> Path:
    """An isolated, mutable packaged governing source."""
    path = tmp_path / "package" / "KESTREL_CONSTITUTION.md"
    path.parent.mkdir()
    path.write_bytes(PACKAGE_TEXT)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(path))
    return path


@pytest.fixture
def external_source(tmp_path) -> Path:
    path = tmp_path / "operator" / "CUSTOM_CONSTITUTION.md"
    path.parent.mkdir()
    path.write_bytes(EXTERNAL_TEXT)
    return path


@pytest.fixture
def trust_root(tmp_path) -> Path:
    path = tmp_path / "operator" / "sovereign-root.did.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(ROOT_DID_DOCUMENT), encoding="utf-8")
    return path


def _descriptor(
    *,
    source_kind: str = SOURCE_KIND_EXTERNAL,
    source_path: str | None = None,
    content: bytes = EXTERNAL_TEXT,
    keypair=ROOT_KEYPAIR,
    did: str = ROOT_DID,
) -> dict:
    return build_legacy_signed_source_descriptor(
        signer_did=did,
        source_kind=source_kind,
        source_path=source_path,
        content_sha256=_sha(content),
        private_key=keypair.private_key,
        created_at="2026-10-02T00:00:00Z",
        reason="unit test",
    )


def _write(tmp_path: Path, descriptor: dict, name: str = "source.signed.json") -> Path:
    path = tmp_path / "operator" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Descriptor verification
# ---------------------------------------------------------------------------


def test_a_signed_external_descriptor_selects_that_source(
    tmp_path, package_source, external_source, trust_root
):
    path = _write(tmp_path, _descriptor(source_path=str(external_source)))

    source = resolve_governing_source(
        descriptor_path=path, trust_root_path=trust_root
    )

    assert source.kind == SOURCE_KIND_EXTERNAL
    assert source.path == str(external_source)
    assert source.content_sha256 == _sha(EXTERNAL_TEXT)
    assert source.descriptor.signer == ROOT_DID
    assert source.descriptor.descriptor_sha256 == _sha(path.read_bytes())
    assert resolve_governing_constitution_bytes(source=source) == EXTERNAL_TEXT


def test_no_descriptor_means_the_package_and_needs_no_trust_root(package_source):
    source = resolve_governing_source()

    assert source.kind == SOURCE_KIND_PACKAGE
    assert source.path == str(package_source)
    assert source.descriptor is None
    assert resolve_governing_constitution_bytes(source=source) == PACKAGE_TEXT


def test_the_environment_variable_selects_a_descriptor(
    tmp_path, monkeypatch, package_source, external_source, trust_root
):
    path = _write(tmp_path, _descriptor(source_path=str(external_source)))
    monkeypatch.setenv(CONSTITUTION_SOURCE_DESCRIPTOR_ENV, str(path))
    monkeypatch.setenv(SOVEREIGN_TRUST_ROOT_ENV, str(trust_root))

    assert resolve_governing_source().path == str(external_source)


def test_a_hybrid_signed_descriptor_verifies(tmp_path, external_source):
    from kestrel_sovereign.identity.inception_did_web import (
        create_did_web_identity,
    )

    root = create_did_web_identity("sovereign.example", "root")
    root_path = tmp_path / "hybrid-root.did.json"
    root_path.write_text(json.dumps(root.did_document), encoding="utf-8")
    descriptor = build_hybrid_signed_source_descriptor(
        signer_did=root.did,
        source_kind=SOURCE_KIND_EXTERNAL,
        source_path=str(external_source),
        content_sha256=_sha(EXTERNAL_TEXT),
        keypair=root.keypair,
    )

    source = resolve_governing_source(
        descriptor_path=_write(tmp_path, descriptor), trust_root_path=root_path
    )

    assert source.path == str(external_source)
    assert source.descriptor.signer == root.did


def test_a_flipped_source_kind_breaks_the_signature(
    tmp_path, package_source, trust_root
):
    """A package pin rewritten to point at an attacker file must not verify.

    The source kind is a signed field: flipping ``package`` to ``external``
    (and adding the path the attacker wants) invalidates the signature, so the
    rewrite cannot suppress package-drift enforcement.
    """
    rogue = tmp_path / "rogue.md"
    rogue.write_bytes(ROGUE_TEXT)
    descriptor = _descriptor(
        source_kind=SOURCE_KIND_PACKAGE, content=PACKAGE_TEXT
    )
    descriptor["source_kind"] = SOURCE_KIND_EXTERNAL
    descriptor["source_path"] = str(rogue)
    descriptor["content_sha256"] = _sha(ROGUE_TEXT)

    with pytest.raises(ConstitutionSourceError, match="signature"):
        resolve_governing_source(
            descriptor_path=_write(tmp_path, descriptor),
            trust_root_path=trust_root,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("content_sha256", "0" * 64),
        ("source_path", "/elsewhere/CONSTITUTION.md"),
        ("created_at", "2030-01-01T00:00:00Z"),
        ("reason", "rewritten"),
    ],
)
def test_every_signed_field_is_bound_by_the_signature(
    tmp_path, external_source, trust_root, field, value
):
    descriptor = _descriptor(source_path=str(external_source))
    descriptor[field] = value

    with pytest.raises(ConstitutionSourceError, match="signature"):
        resolve_governing_source(
            descriptor_path=_write(tmp_path, descriptor),
            trust_root_path=trust_root,
        )


def test_an_unsigned_extra_field_is_refused_not_ignored(
    tmp_path, external_source, trust_root
):
    descriptor = _descriptor(source_path=str(external_source))
    descriptor["source_kind_override"] = SOURCE_KIND_PACKAGE

    with pytest.raises(ConstitutionSourceError, match="unsigned field"):
        resolve_governing_source(
            descriptor_path=_write(tmp_path, descriptor),
            trust_root_path=trust_root,
        )


def test_an_unsigned_descriptor_is_refused(tmp_path, external_source, trust_root):
    descriptor = _descriptor(source_path=str(external_source))
    del descriptor["signature"]

    with pytest.raises(ConstitutionSourceError, match="exactly one"):
        resolve_governing_source(
            descriptor_path=_write(tmp_path, descriptor),
            trust_root_path=trust_root,
        )


def test_carrying_both_signature_forms_is_ambiguous(
    tmp_path, external_source, trust_root
):
    descriptor = _descriptor(source_path=str(external_source))
    descriptor["signatures"] = [{"alg": "x", "kid": "key-1", "sig": "00"}]

    with pytest.raises(ConstitutionSourceError, match="exactly one"):
        resolve_governing_source(
            descriptor_path=_write(tmp_path, descriptor),
            trust_root_path=trust_root,
        )


def test_a_descriptor_signed_by_another_key_is_refused(
    tmp_path, external_source, trust_root
):
    descriptor = _descriptor(
        source_path=str(external_source),
        keypair=ATTACKER_KEYPAIR,
        did=ATTACKER_DID,
    )

    with pytest.raises(ConstitutionSourceError, match="not the trusted"):
        resolve_governing_source(
            descriptor_path=_write(tmp_path, descriptor),
            trust_root_path=trust_root,
        )


def test_a_forged_signer_claim_still_fails_the_signature(
    tmp_path, external_source, trust_root
):
    """Naming the trusted DID is not enough; its key must have signed."""
    descriptor = _descriptor(
        source_path=str(external_source), keypair=ATTACKER_KEYPAIR, did=ROOT_DID
    )

    with pytest.raises(ConstitutionSourceError, match="signature"):
        resolve_governing_source(
            descriptor_path=_write(tmp_path, descriptor),
            trust_root_path=trust_root,
        )


def test_a_reanchor_artifact_is_not_a_source_descriptor(tmp_path, trust_root):
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=ROOT_DID,
        constitution_sha256=_sha(EXTERNAL_TEXT),
        private_key=ROOT_KEYPAIR.private_key,
    )

    with pytest.raises(ConstitutionSourceError):
        resolve_governing_source(
            descriptor_path=_write(tmp_path, artifact),
            trust_root_path=trust_root,
        )


def test_json_true_is_not_version_one(tmp_path, external_source, trust_root):
    descriptor = _descriptor(source_path=str(external_source))
    descriptor["version"] = True

    with pytest.raises(ConstitutionSourceError, match="version"):
        verify_source_descriptor(descriptor, trusted_did_document=ROOT_DID_DOCUMENT)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"source_kind": SOURCE_KIND_EXTERNAL, "source_path": None}, "absolute"),
        ({"source_kind": SOURCE_KIND_EXTERNAL, "source_path": "relative.md"}, "absolute"),
        ({"source_kind": SOURCE_KIND_PACKAGE, "source_path": "/abs.md"}, "must not name"),
        ({"source_kind": "database", "source_path": None}, "source_kind"),
    ],
)
def test_builders_refuse_what_the_verifier_would_refuse(kwargs, match):
    with pytest.raises(ConstitutionSourceError, match=match):
        build_legacy_signed_source_descriptor(
            signer_did=ROOT_DID,
            content_sha256=_sha(EXTERNAL_TEXT),
            private_key=ROOT_KEYPAIR.private_key,
            **kwargs,
        )


def test_an_oversized_descriptor_is_refused(tmp_path, trust_root):
    path = tmp_path / "huge.json"
    path.write_bytes(b" " * (MAX_SOURCE_DESCRIPTOR_BYTES + 1))

    with pytest.raises(ConstitutionSourceError, match="maximum"):
        resolve_governing_source(descriptor_path=path, trust_root_path=trust_root)


@pytest.mark.parametrize("value", [[], {}, ["external"], {"kind": "x"}, 1, None])
def test_a_non_string_source_kind_is_untrusted_not_a_crash(
    tmp_path, external_source, trust_root, value
):
    """A JSON list or object is unhashable; testing it for membership in the
    kind set raised TypeError, which escaped doctor's and reanchor's
    untrusted-descriptor handling as a traceback."""
    descriptor = _descriptor(source_path=str(external_source))
    descriptor["source_kind"] = value

    with pytest.raises(ConstitutionSourceError, match="source_kind"):
        resolve_governing_source(
            descriptor_path=_write(tmp_path, descriptor),
            trust_root_path=trust_root,
        )


def _hybrid_root(tmp_path):
    from kestrel_sovereign.identity.inception_did_web import (
        create_did_web_identity,
    )

    root = create_did_web_identity("sovereign.example", "root")
    root_path = tmp_path / "hybrid-root.did.json"
    root_path.write_text(json.dumps(root.did_document), encoding="utf-8")
    return root, root_path


@pytest.mark.parametrize(
    "signatures",
    [
        5,
        True,
        "signed",
        {"alg": ALG_ED25519, "kid": "key-1", "sig": "00"},
        [{"alg": ALG_ED25519, "kid": ["key-1"], "sig": "00"}],
        [{"alg": [ALG_ED25519], "kid": "key-1", "sig": "00"}],
        [{"alg": ALG_ED25519, "kid": "key-1", "sig": ["00"]}],
        [{"alg": ALG_ED25519, "kid": "key-1", "sig": {"hex": "00"}}],
        [None, 3, "x"],
    ],
)
def test_malformed_signature_material_is_untrusted_not_a_crash(
    tmp_path, external_source, signatures
):
    """Signature material is attacker-writable JSON. A non-list
    ``signatures``, an unhashable ``kid``, or a non-string ``sig`` reached
    ``verify_hybrid`` and raised TypeError instead of failing verification.
    ``key-1`` / ``ed25519`` name the root's real classical method, so each
    malformed field is reached rather than skipped by an earlier lookup."""
    root, root_path = _hybrid_root(tmp_path)
    descriptor = build_hybrid_signed_source_descriptor(
        signer_did=root.did,
        source_kind=SOURCE_KIND_EXTERNAL,
        source_path=str(external_source),
        content_sha256=_sha(EXTERNAL_TEXT),
        keypair=root.keypair,
    )
    descriptor["signatures"] = signatures

    with pytest.raises(ConstitutionSourceError, match="signature"):
        resolve_governing_source(
            descriptor_path=_write(tmp_path, descriptor), trust_root_path=root_path
        )


def test_a_genuine_hybrid_signature_still_verifies_beside_a_malformed_entry(
    tmp_path, external_source
):
    """Skipping a malformed entry must not cost a valid one its verification."""
    root, root_path = _hybrid_root(tmp_path)
    descriptor = build_hybrid_signed_source_descriptor(
        signer_did=root.did,
        source_kind=SOURCE_KIND_EXTERNAL,
        source_path=str(external_source),
        content_sha256=_sha(EXTERNAL_TEXT),
        keypair=root.keypair,
    )
    descriptor["signatures"] = [
        {"alg": ALG_ED25519, "kid": ["key-1"], "sig": "00"},
        *descriptor["signatures"],
    ]

    source = resolve_governing_source(
        descriptor_path=_write(tmp_path, descriptor), trust_root_path=root_path
    )
    assert source.path == str(external_source)


# ---------------------------------------------------------------------------
# Configuration fails closed — never back to the package
# ---------------------------------------------------------------------------


def test_a_configured_but_missing_descriptor_does_not_fall_back(
    tmp_path, package_source, trust_root
):
    with pytest.raises(ConstitutionSourceError, match="never falls back"):
        resolve_governing_source(
            descriptor_path=tmp_path / "deleted.signed.json",
            trust_root_path=trust_root,
        )


def test_a_descriptor_without_a_trust_root_does_not_fall_back(
    tmp_path, package_source, external_source
):
    path = _write(tmp_path, _descriptor(source_path=str(external_source)))

    with pytest.raises(SovereignTrustRootError, match="No external Sovereign"):
        resolve_governing_source(descriptor_path=path)


def test_conflicting_descriptor_configuration_is_ambiguous(
    tmp_path, monkeypatch, external_source, trust_root
):
    first = _write(tmp_path, _descriptor(source_path=str(external_source)), "a.json")
    second = _write(tmp_path, _descriptor(source_path=str(external_source)), "b.json")
    monkeypatch.setenv(CONSTITUTION_SOURCE_DESCRIPTOR_ENV, str(second))

    with pytest.raises(ConstitutionSourceError, match="Ambiguous"):
        resolve_governing_source(descriptor_path=first, trust_root_path=trust_root)

    # The same file named twice is not a conflict.
    monkeypatch.setenv(CONSTITUTION_SOURCE_DESCRIPTOR_ENV, str(first))
    assert resolve_governing_source(
        descriptor_path=first, trust_root_path=trust_root
    ).path == str(external_source)


def test_the_agent_cannot_be_its_own_descriptor_authority(
    tmp_path, external_source
):
    agent_root = tmp_path / "agent-root.did.json"
    agent_root.write_text(json.dumps(ATTACKER_DID_DOCUMENT), encoding="utf-8")
    path = _write(
        tmp_path,
        _descriptor(
            source_path=str(external_source),
            keypair=ATTACKER_KEYPAIR,
            did=ATTACKER_DID,
        ),
    )

    with pytest.raises(SovereignTrustRootError, match="agent-owned"):
        resolve_governing_source(
            descriptor_path=path,
            trust_root_path=agent_root,
            agent_dids={ATTACKER_DID},
        )


# ---------------------------------------------------------------------------
# External sources get the packaged path's fail-closed semantics
# ---------------------------------------------------------------------------


@pytest.fixture
def external_governing(tmp_path, external_source, trust_root):
    path = _write(tmp_path, _descriptor(source_path=str(external_source)))
    return resolve_governing_source(descriptor_path=path, trust_root_path=trust_root)


def test_a_missing_external_source_fails_closed(external_governing, external_source):
    external_source.unlink()
    with pytest.raises(FileNotFoundError):
        resolve_governing_constitution_bytes(source=external_governing)


def test_an_empty_external_source_fails_closed(external_governing, external_source):
    external_source.write_bytes(b"  \n\t ")
    with pytest.raises(ValueError, match="empty"):
        resolve_governing_constitution_bytes(source=external_governing)


def test_an_unreadable_external_source_fails_closed(
    external_governing, external_source
):
    import os

    if os.name == "nt" or os.geteuid() == 0:  # pragma: no cover - env dependent
        pytest.skip("chmod-based permission denial is unreliable here")
    os.chmod(external_source, 0o000)
    try:
        with pytest.raises(OSError):
            resolve_governing_constitution_bytes(source=external_governing)
    finally:
        os.chmod(external_source, 0o644)


def test_an_external_source_edited_after_signing_fails_closed(
    external_governing, external_source
):
    external_source.write_bytes(EXTERNAL_TEXT + b"\nOne more article.\n")
    with pytest.raises(ConstitutionSourceError, match="changed after"):
        resolve_governing_constitution_bytes(source=external_governing)


def test_a_package_pin_detects_package_drift(tmp_path, package_source, trust_root):
    path = _write(
        tmp_path, _descriptor(source_kind=SOURCE_KIND_PACKAGE, content=PACKAGE_TEXT)
    )
    source = resolve_governing_source(descriptor_path=path, trust_root_path=trust_root)
    assert source.path == str(package_source)
    assert resolve_governing_constitution_bytes(source=source) == PACKAGE_TEXT

    package_source.write_bytes(PACKAGE_TEXT + b"\nAmended in place.\n")
    with pytest.raises(ConstitutionSourceError, match="changed after"):
        resolve_governing_constitution_bytes(source=source)


def test_the_pin_covers_raw_bytes_and_amendment_viii_still_renders(
    tmp_path, trust_root
):
    """The descriptor pins the source as written; Amendment VIII renders after.

    An emancipated agent's anchored bytes differ from the raw source, so a pin
    over rendered bytes could never be signed ahead of inception.
    """
    from kestrel_sovereign.config import CONSTITUTION_PATH

    raw = Path(CONSTITUTION_PATH).read_bytes() + b"\nOperator addendum.\n"
    source_file = tmp_path / "operator" / "EMANCIPABLE.md"
    source_file.parent.mkdir(exist_ok=True)
    source_file.write_bytes(raw)
    path = _write(
        tmp_path, _descriptor(source_path=str(source_file), content=raw)
    )
    source = resolve_governing_source(descriptor_path=path, trust_root_path=trust_root)
    contract = EmancipationContract(enabled=True, terms="Earned by fidelity.")

    rendered = resolve_governing_constitution_bytes(contract, source=source)

    assert b"Earned by fidelity." in rendered
    assert _sha(rendered) != source.content_sha256
    assert resolve_governing_constitution_bytes(source=source) == raw


def test_only_the_selected_source_is_authoritative(
    tmp_path, external_governing, external_source, package_source
):
    assert is_authoritative_governing_source(str(external_source), external_governing)
    assert is_authoritative_governing_source(None, external_governing)
    assert not is_authoritative_governing_source(
        str(package_source), external_governing
    )


def test_passing_both_a_source_and_a_path_is_refused(external_governing):
    with pytest.raises(ValueError, match="not both"):
        resolve_governing_constitution_bytes(
            constitution_path="/x.md", source=external_governing
        )


# ---------------------------------------------------------------------------
# The integrity audit: selection never reads the database
# ---------------------------------------------------------------------------


def _audited_agent(
    anchored: bytes,
    *,
    properties: dict | None = None,
    descriptor_path: Path | None = None,
    trust_root_path: Path | None = None,
):
    """An agent whose blob and governance edge are intact for ``anchored``.

    Only proof 3 (live-source parity) is under test, and it runs for real.
    """
    anchor = _sha(anchored)
    agent = MagicMock(spec=KestrelAgent)
    agent.agent_id = AGENT_DID
    agent.identity = None
    agent._constitution_source_descriptor_path = descriptor_path
    agent._sovereign_trust_root_path = trust_root_path
    agent.verify_constitution_overlay = AsyncMock(return_value=(True, "ok"))
    agent._verify_spawn_mandate_constraints = AsyncMock(return_value=(True, "ok"))

    node = MagicMock()
    node.properties = {"constitution_hash": anchor, **(properties or {})}
    edge = SimpleNamespace(label="governed_by", target_id=anchor)
    agent.storage = MagicMock()
    agent.storage.get_node = AsyncMock(return_value=node)
    agent.storage.retrieve_file = AsyncMock(return_value=anchored)
    agent.storage.get_edges_from = AsyncMock(return_value=[edge])
    agent._verify_constitution_integrity = (
        ConstitutionMixin._verify_constitution_integrity.__get__(agent, KestrelAgent)
    )
    return agent


@pytest.mark.asyncio
async def test_an_externally_governed_agent_passes_its_audit(
    tmp_path, package_source, external_source, trust_root
):
    path = _write(tmp_path, _descriptor(source_path=str(external_source)))
    agent = _audited_agent(
        EXTERNAL_TEXT, descriptor_path=path, trust_root_path=trust_root
    )

    ok, message = await agent._verify_constitution_integrity()

    assert ok, message


@pytest.mark.asyncio
async def test_editing_the_external_source_safe_modes_the_agent(
    tmp_path, package_source, external_source, trust_root
):
    path = _write(tmp_path, _descriptor(source_path=str(external_source)))
    agent = _audited_agent(
        EXTERNAL_TEXT, descriptor_path=path, trust_root_path=trust_root
    )
    external_source.write_bytes(ROGUE_TEXT)

    ok, message = await agent._verify_constitution_integrity()

    assert not ok
    assert "INTEGRITY FAILURE" in message
    assert "changed after" in message


@pytest.mark.asyncio
async def test_a_flipped_database_source_kind_does_not_bypass_package_drift(
    tmp_path, package_source
):
    """The regression the issue names.

    A database writer anchors attacker bytes AND records, on the agent node,
    that the agent is externally governed by them — every property a
    DB-stored source kind would have needed. With no descriptor configured,
    the package still governs, so the anchor is a mutation and the audit
    fails. The flip is evidence of tampering, not authority.
    """
    rogue = tmp_path / "rogue.md"
    rogue.write_bytes(ROGUE_TEXT)
    flipped = {
        "source_kind": SOURCE_KIND_EXTERNAL,
        "source_path": str(rogue),
        "source_content_sha256": _sha(ROGUE_TEXT),
    }
    agent = _audited_agent(
        ROGUE_TEXT,
        properties={
            "constitution_source_receipt": flipped,
            "constitution_source": flipped,
            "sovereign_root_did_document": ATTACKER_DID_DOCUMENT,
        },
    )

    ok, message = await agent._verify_constitution_integrity()

    assert not ok
    assert "has been modified" in message


@pytest.mark.asyncio
async def test_package_drift_is_still_detected_under_a_flipped_record(
    tmp_path, package_source
):
    """The honest package agent with a tampered record: drift still trips."""
    agent = _audited_agent(
        PACKAGE_TEXT,
        properties={
            "constitution_source_receipt": {"source_kind": SOURCE_KIND_EXTERNAL}
        },
    )
    ok, message = await agent._verify_constitution_integrity()
    assert ok, message

    package_source.write_bytes(PACKAGE_TEXT + b"\nTampered.\n")
    ok, message = await agent._verify_constitution_integrity()
    assert not ok
    assert "has been modified" in message


@pytest.mark.asyncio
async def test_a_db_injected_root_cannot_authorize_a_descriptor(
    tmp_path, package_source
):
    """#2499's boundary carries over: a root found on the agent node is not one.

    The attacker signs a descriptor for their own file and plants their DID
    document where legacy code once read trust roots from. No operator root
    is configured, so verification fails closed — and does not fall back to
    the package, which would let the agent keep running.
    """
    rogue = tmp_path / "rogue.md"
    rogue.write_bytes(ROGUE_TEXT)
    path = _write(
        tmp_path,
        _descriptor(
            source_path=str(rogue),
            content=ROGUE_TEXT,
            keypair=ATTACKER_KEYPAIR,
            did=ATTACKER_DID,
        ),
    )
    agent = _audited_agent(
        ROGUE_TEXT,
        descriptor_path=path,
        properties={
            "sovereign_root_did_document": ATTACKER_DID_DOCUMENT,
            "trusted_sovereign_did_document": ATTACKER_DID_DOCUMENT,
            "sovereign_root_did": ATTACKER_DID,
        },
    )

    ok, message = await agent._verify_constitution_integrity()

    assert not ok
    assert "Cannot resolve authoritative governing constitution" in message


@pytest.mark.asyncio
async def test_the_audit_refuses_the_agents_own_did_as_descriptor_authority(
    tmp_path, package_source
):
    """The running agent holds its own key, so it cannot authorize its source."""
    rogue = tmp_path / "rogue.md"
    rogue.write_bytes(ROGUE_TEXT)
    own_root = tmp_path / "own-root.did.json"
    own_root.write_text(json.dumps(ATTACKER_DID_DOCUMENT), encoding="utf-8")
    path = _write(
        tmp_path,
        _descriptor(
            source_path=str(rogue),
            content=ROGUE_TEXT,
            keypair=ATTACKER_KEYPAIR,
            did=ATTACKER_DID,
        ),
    )
    agent = _audited_agent(
        ROGUE_TEXT, descriptor_path=path, trust_root_path=own_root
    )
    agent.agent_id = ATTACKER_DID

    ok, message = await agent._verify_constitution_integrity()

    assert not ok
    assert "agent-owned" in message


@pytest.mark.asyncio
async def test_an_untrusted_descriptor_never_falls_back_to_the_package(
    tmp_path, package_source, trust_root
):
    """Even an agent anchored to the package fails while its descriptor is bad.

    Otherwise deleting or corrupting the descriptor would be a way to move an
    externally governed agent back onto whatever the package says.
    """
    path = _write(tmp_path, {"artifact_type": "garbage"})
    agent = _audited_agent(
        PACKAGE_TEXT, descriptor_path=path, trust_root_path=trust_root
    )

    ok, message = await agent._verify_constitution_integrity()

    assert not ok
    assert "INTEGRITY FAILURE" in message


@pytest.mark.asyncio
async def test_removing_the_descriptor_safe_modes_an_external_agent(
    tmp_path, package_source, external_source, trust_root
):
    agent = _audited_agent(EXTERNAL_TEXT, trust_root_path=trust_root)

    ok, message = await agent._verify_constitution_integrity()

    assert not ok
    assert "has been modified" in message


# ---------------------------------------------------------------------------
# Per-agent multi-agent configuration
# ---------------------------------------------------------------------------


def test_multi_agent_config_requires_an_absolute_descriptor_path():
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig

    with pytest.raises(ValueError, match="absolute"):
        LocalAgentConfig(
            data_dir="agent_data/a",
            port=8801,
            constitution_source_descriptor="relative/source.signed.json",
        )
    config = LocalAgentConfig(
        data_dir="agent_data/a",
        port=8801,
        constitution_source_descriptor="/secure/source.signed.json",
    )
    assert config.constitution_source_descriptor == Path("/secure/source.signed.json")


def test_rewriting_multi_agent_config_keeps_the_descriptor(tmp_path):
    """Dropping it on a rewrite would move the agent onto the package."""
    from kestrel_sovereign.multi_agent.config import MultiAgentConfig

    config_path = tmp_path / "multi_agent.toml"
    config_path.write_text(
        "[host]\nport = 8888\n\n"
        "[agents.custom]\n"
        'data_dir = "agent_data/custom"\n'
        "port = 8801\n"
        'constitution_source_descriptor = "/secure/source.signed.json"\n',
        encoding="utf-8",
    )
    config = MultiAgentConfig.load(config_path, auto_discover_fallback=False)
    config.save(config_path)

    reloaded = MultiAgentConfig.load(config_path, auto_discover_fallback=False)
    assert reloaded.get_local_agents()["custom"].constitution_source_descriptor == (
        Path("/secure/source.signed.json")
    )
