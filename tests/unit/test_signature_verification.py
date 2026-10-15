"""The shared signature-entry verifier, and every public surface built on it (#2487).

Signature arrays and verification methods are external material: a peer's
DID document, an archived succession statement, an identity package. Before
#2487 the hybrid, succession and succession-chain verifiers each carried a
copy of the kid-indexing / multikey / hex loop, and an integer ``sig``, an
integer or unhashable ``id``/``kid``, or a non-base58btc
``publicKeyMultibase`` escaped from them as ``TypeError``/``ValueError``.

The vectors below are applied to each public surface. Every one must come
back as a rejection, never an exception, and the rejection must come from
the signature side of the verdict, not some unrelated check. A valid
baseline on each surface must still verify.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.hybrid_keypair import (
    generate_hybrid_keypair,
    sign_hybrid,
    verify_hybrid,
)
from kestrel_sovereign.identity.succession import (
    SuccessionStatement,
    finalize,
    sign_predecessor,
    sign_successor,
    verify_succession,
)
from kestrel_sovereign.identity.succession_chain import (
    build_chain,
    verify_artifact_against_chain,
)
from kestrel_sovereign.inception_service import public_key_to_ethereum_address
from kestrel_sovereign.security.crypto_suite import (
    ALG_ED25519,
    ALG_ML_DSA_65,
    CryptoSuiteError,
    Secp256k1Suite,
)
from kestrel_sovereign.security.kem_suite import KEMSuiteError
from kestrel_sovereign.security.multikey import (
    base58btc_encode,
    encode_varint,
    multibase_to_kem_public_key,
    multibase_to_public_key,
)
from kestrel_sovereign.security.signature_verification import (
    decode_verification_method_key,
    index_verification_methods,
    verification_method_kid,
    verify_signature_entries,
)
from kestrel_sovereign.security.verify_policy import VerifyPolicy


DATA = b"#2487 signed artifact"
DID = "did:web:example.com:agent:kite"
EFFECTIVE_FROM = "2026-01-01T00:00:00+00:00"
ARTIFACT_TIMESTAMP = "2026-02-01T00:00:00+00:00"


@pytest.fixture(scope="module")
def material():
    """A hybrid did:web identity plus a legacy did:pkh predecessor for it."""
    keypair = generate_hybrid_keypair()
    vms = build_verification_methods(DID, keypair.public_keys())
    secp = Secp256k1Suite()
    legacy = secp.generate_keypair()
    pred_did = (
        f"did:pkh:eip155:1:{public_key_to_ethereum_address(legacy.public_key)}"
    )
    pred_vms = build_verification_methods(pred_did, [(secp, legacy.public_key)])
    return SimpleNamespace(
        keypair=keypair,
        vms=vms,
        classical_kid=verification_method_kid(vms[0]["id"]),
        pq_kid=verification_method_kid(vms[1]["id"]),
        legacy=legacy,
        pred_did=pred_did,
        pred_vms=pred_vms,
        pred_kid=verification_method_kid(pred_vms[0]["id"]),
    )


def _vms(m):
    return copy.deepcopy(m.vms)


def _artifact_signatures(m):
    return sign_hybrid(
        DATA, m.keypair, classical_kid=m.classical_kid, pq_kid=m.pq_kid,
    )


# ---------------------------------------------------------------------------
# The public surfaces. Each takes a signature-array mutator and a
# verification-method mutator and returns (overall ok, signature-side ok).
# ---------------------------------------------------------------------------

def _publishing(did, vms):
    """A did:web resolver that publishes ``vms`` (malformed or not) for ``did``.

    Publishing the same VMs the statement embeds keeps DID binding from
    being the thing that rejects a key-material vector: the signature side
    has to do it.
    """
    def resolve(requested):
        if requested == did:
            return {"id": did, "verificationMethod": vms}
        raise LookupError(requested)
    return resolve


def _successor_statement(m, mutate_sigs, mutate_vms):
    successor_vms = mutate_vms(_vms(m))
    statement = SuccessionStatement(
        predecessor_did=m.pred_did,
        successor_did=DID,
        effective_from=EFFECTIVE_FROM,
        reason="#2487 malformed-material vector",
        predecessor_verification_methods=copy.deepcopy(m.pred_vms),
        successor_verification_methods=successor_vms,
    )
    statement = sign_predecessor(statement, [(m.legacy, m.pred_kid)])
    statement = sign_successor(
        statement,
        [(m.keypair.classical, m.classical_kid), (m.keypair.pq, m.pq_kid)],
    )
    statement = replace(
        statement,
        successor_signatures=mutate_sigs(statement.successor_signatures),
    )
    return finalize(statement), successor_vms


def _surface_hybrid(m, mutate_sigs, mutate_vms):
    result = verify_hybrid(DATA, mutate_sigs(_artifact_signatures(m)), mutate_vms(_vms(m)))
    return result.ok, result.ok


def _surface_succession(m, mutate_sigs, mutate_vms):
    statement, successor_vms = _successor_statement(m, mutate_sigs, mutate_vms)
    result = verify_succession(
        statement, did_web_resolver=_publishing(DID, successor_vms),
    )
    return result.ok, result.successor.ok


def _surface_chain_root(m, mutate_sigs, mutate_vms):
    """No succession yet: the root identity is active."""
    result = verify_artifact_against_chain(
        root_did=DID,
        root_verification_methods=mutate_vms(_vms(m)),
        chain=build_chain([]),
        artifact_timestamp=ARTIFACT_TIMESTAMP,
        artifact_payload=DATA,
        artifact_signatures=mutate_sigs(_artifact_signatures(m)),
    )
    return result.ok, result.policy_result.ok


def _surface_chain_successor(m, mutate_sigs, mutate_vms):
    """After a legacy → hybrid succession: the successor is active."""
    statement, successor_vms = _successor_statement(m, lambda s: s, mutate_vms)
    result = verify_artifact_against_chain(
        root_did=m.pred_did,
        root_verification_methods=copy.deepcopy(m.pred_vms),
        chain=build_chain([statement]),
        artifact_timestamp=ARTIFACT_TIMESTAMP,
        artifact_payload=DATA,
        artifact_signatures=mutate_sigs(_artifact_signatures(m)),
        did_web_resolver=_publishing(DID, successor_vms),
    )
    return result.ok, result.policy_result.ok


SURFACES = {
    "verify_hybrid": _surface_hybrid,
    "verify_succession": _surface_succession,
    "verify_artifact_against_chain[root]": _surface_chain_root,
    "verify_artifact_against_chain[successor]": _surface_chain_successor,
}


def _unchanged(value):
    return value


# ---------------------------------------------------------------------------
# Vectors. Signature-entry vectors corrupt the classical entry and leave the
# PQ entry valid, so HYBRID_REQUIRED fails only if the corrupted entry was
# not counted. Key-material vectors corrupt the classical method.
# ---------------------------------------------------------------------------

_MISSING = object()


def _classical_entry(**fields):
    def mutate(sigs):
        entry = dict(sigs[0])
        for name, value in fields.items():
            if value is _MISSING:
                entry.pop(name, None)
            else:
                entry[name] = value
        return [entry, sigs[1]]
    return mutate


def _replace_classical_entry(value):
    return lambda sigs: [value, sigs[1]]


def _flip_last_byte(sigs):
    entry = dict(sigs[0])
    raw = bytearray(bytes.fromhex(entry["sig"]))
    raw[-1] ^= 0x01
    entry["sig"] = raw.hex()
    return [entry, sigs[1]]


SIGNATURE_VECTORS = {
    "entry is a string": _replace_classical_entry("junk"),
    "entry is an int": _replace_classical_entry(5),
    "entry is null": _replace_classical_entry(None),
    "entry is a list": _replace_classical_entry(["alg", "kid", "sig"]),
    "alg missing": _classical_entry(alg=_MISSING),
    "alg null": _classical_entry(alg=None),
    "alg int": _classical_entry(alg=5),
    "alg bool": _classical_entry(alg=True),
    "alg empty": _classical_entry(alg=""),
    "alg list (unhashable)": _classical_entry(alg=[ALG_ED25519]),
    "alg dict (unhashable)": _classical_entry(alg={"alg": ALG_ED25519}),
    "alg names the PQ suite": _classical_entry(alg=ALG_ML_DSA_65),
    "alg unregistered": _classical_entry(alg="rot13"),
    "kid missing": _classical_entry(kid=_MISSING),
    "kid null": _classical_entry(kid=None),
    "kid int": _classical_entry(kid=7),
    "kid float": _classical_entry(kid=1.5),
    "kid empty": _classical_entry(kid=""),
    "kid list (unhashable)": _classical_entry(kid=["key-1"]),
    "kid dict (unhashable)": _classical_entry(kid={"kid": "key-1"}),
    "kid unknown": _classical_entry(kid="key-99"),
    "sig missing": _classical_entry(sig=_MISSING),
    "sig null": _classical_entry(sig=None),
    "sig int": _classical_entry(sig=123),
    "sig bool": _classical_entry(sig=True),
    "sig bytes": _classical_entry(sig=b"\x00\x01"),
    "sig list": _classical_entry(sig=["00"]),
    "sig empty": _classical_entry(sig=""),
    "sig non-hex": _classical_entry(sig="zz"),
    "sig odd-length hex": _classical_entry(sig="abc"),
    "sig non-ascii": _classical_entry(sig="éé"),
    "sig does not verify": _flip_last_byte,
    "signatures null": lambda sigs: None,
    "signatures int": lambda sigs: 5,
    "signatures string": lambda sigs: "signatures",
    "signatures single object": lambda sigs: sigs[0],
}


def _classical_method(**fields):
    def mutate(vms):
        vm = dict(vms[0])
        for name, value in fields.items():
            if value is _MISSING:
                vm.pop(name, None)
            else:
                vm[name] = value
        return [vm] + vms[1:]
    return mutate


def _multibase(payload: bytes) -> str:
    return "z" + base58btc_encode(payload)


VM_VECTORS = {
    "method is a string": lambda vms: ["junk"] + vms[1:],
    "method is an int": lambda vms: [5] + vms[1:],
    "method is null": lambda vms: [None] + vms[1:],
    "method is a list": lambda vms: [["id"]] + vms[1:],
    "extra junk method": lambda vms: vms + ["junk"],
    "id missing": _classical_method(id=_MISSING),
    "id int": _classical_method(id=5),
    "id float": _classical_method(id=1.5),
    "id bool": _classical_method(id=True),
    "id list (unhashable)": lambda vms: _classical_method(id=[vms[0]["id"]])(vms),
    "id dict (unhashable)": _classical_method(id={"id": "key-1"}),
    "duplicate kid": lambda vms: [vms[0], {**vms[1], "id": vms[0]["id"]}],
    # Both legitimate methods stay; a later method under another DID reuses
    # the classical kid fragment (the duplicate-kid takeover).
    "duplicate kid fragment from another DID": lambda vms: vms + [{
        **vms[1],
        "id": "did:web:attacker.example#" + verification_method_kid(vms[0]["id"]),
    }],
    "multibase missing": _classical_method(publicKeyMultibase=_MISSING),
    "multibase null": _classical_method(publicKeyMultibase=None),
    "multibase int": _classical_method(publicKeyMultibase=5),
    "multibase list": _classical_method(publicKeyMultibase=["z6Mk"]),
    "multibase empty": _classical_method(publicKeyMultibase=""),
    "multibase wrong prefix": _classical_method(publicKeyMultibase="mAAAA"),
    "multibase non-base58btc": _classical_method(publicKeyMultibase="z0OIl"),
    "multibase non-ascii": _classical_method(publicKeyMultibase="zé"),
    "multibase empty payload": _classical_method(publicKeyMultibase="z"),
    "multicodec varint truncated": _classical_method(
        publicKeyMultibase=_multibase(b"\x80"),
    ),
    "multicodec varint overlong": _classical_method(
        publicKeyMultibase=_multibase(b"\xff" * 10 + b"\x01"),
    ),
    "multicodec unregistered": _classical_method(
        publicKeyMultibase=_multibase(encode_varint(0x1234) + b"\x00" * 32),
    ),
    "ed25519 key wrong length": _classical_method(
        publicKeyMultibase=_multibase(b"\xed\x01" + b"\x00" * 5),
    ),
    "methods null": lambda vms: None,
    "methods int": lambda vms: 5,
    "methods string": lambda vms: "verificationMethod",
    "methods single object": lambda vms: vms[0],
}


def _malformed_cases():
    for surface in SURFACES:
        for name in SIGNATURE_VECTORS:
            yield pytest.param(surface, "signature", name, id=f"{surface}-{name}")
        for name in VM_VECTORS:
            yield pytest.param(surface, "method", name, id=f"{surface}-{name}")


def _mutators(kind, name):
    if kind == "signature":
        return SIGNATURE_VECTORS[name], _unchanged
    return _unchanged, VM_VECTORS[name]


@pytest.mark.parametrize("surface", list(SURFACES))
def test_valid_signatures_still_verify(material, surface):
    ok, signature_side_ok = SURFACES[surface](material, _unchanged, _unchanged)
    assert ok is True
    assert signature_side_ok is True


@pytest.mark.parametrize("surface,kind,name", list(_malformed_cases()))
def test_malformed_material_is_a_rejection_not_an_exception(
    material, surface, kind, name,
):
    mutate_sigs, mutate_vms = _mutators(kind, name)
    ok, signature_side_ok = SURFACES[surface](material, mutate_sigs, mutate_vms)
    assert ok is False
    assert signature_side_ok is False


@pytest.mark.parametrize("surface", list(SURFACES))
def test_malformed_extra_entries_do_not_hide_valid_signatures(material, surface):
    """Junk signature entries are dropped, not fatal: the valid pair still
    satisfies HYBRID_REQUIRED wherever it sits in the array."""
    def with_junk(sigs):
        return [
            "junk",
            {"alg": ALG_ED25519, "kid": ["unhashable"], "sig": 123},
            sigs[1],
            {"alg": 5, "kid": None, "sig": None},
            sigs[0],
        ]

    ok, signature_side_ok = SURFACES[surface](material, with_junk, _unchanged)
    assert ok is True
    assert signature_side_ok is True


# ---------------------------------------------------------------------------
# The primitive itself: structured results, diagnostics, input order.
# ---------------------------------------------------------------------------

def test_verified_entries_and_rejections_keep_input_order(material):
    classical, pq = _artifact_signatures(material)
    result = verify_signature_entries(
        DATA, ["junk", pq, {"kid": "key-99"}, classical], _vms(material),
    )
    assert result.ok
    assert [e["alg"] for e in result.verified] == [ALG_ML_DSA_65, ALG_ED25519]
    assert [r.index for r in result.rejected] == [0, 2]
    assert result.rejected[0].reason == "entry is not an object (got str)"


def test_verified_entries_are_copies(material):
    signatures = _artifact_signatures(material)
    result = verify_signature_entries(DATA, signatures, _vms(material))
    result.verified[0]["sig"] = "tampered"
    assert signatures[0]["sig"] != "tampered"


def test_duplicate_kid_is_fatal_and_names_both_ids(material):
    vms = _vms(material)
    duplicate = {**vms[1], "id": f"did:web:attacker.example#{material.classical_kid}"}
    result = verify_signature_entries(
        DATA, _artifact_signatures(material), vms + [duplicate],
    )
    assert not result.ok
    assert result.verified == ()
    assert len(result.fatal) == 1
    assert "duplicate kid" in result.fatal[0]
    assert vms[0]["id"] in result.fatal[0]
    assert duplicate["id"] in result.fatal[0]


def test_unknown_kid_is_named(material):
    result = verify_signature_entries(
        DATA,
        [{"alg": ALG_ED25519, "kid": "key-99", "sig": "00"}],
        _vms(material),
    )
    assert result.rejected[0].reason == "no verification method for kid 'key-99'"


def test_alg_must_match_the_method_key_type(material):
    classical, pq = _artifact_signatures(material)
    relabelled = {**classical, "alg": ALG_ML_DSA_65}
    result = verify_signature_entries(DATA, [relabelled, pq], _vms(material))
    assert [e["alg"] for e in result.verified] == [ALG_ML_DSA_65]
    assert result.verified[0]["kid"] == material.pq_kid
    assert "does not match verification method" in result.rejected[0].reason
    policy = result.evaluate(VerifyPolicy.HYBRID_REQUIRED)
    assert not policy.ok


def test_method_without_id_is_unaddressable_not_fatal(material):
    vms = _vms(material)
    index = index_verification_methods([{"publicKeyMultibase": "z"}, *vms])
    assert index.ok
    assert list(index.methods) == [material.classical_kid, material.pq_kid]


@pytest.mark.parametrize("bad_id", [5, 1.5, True, ["x#key-1"], {"id": "x"}])
def test_non_string_method_id_is_fatal(material, bad_id):
    index = index_verification_methods([{**_vms(material)[0], "id": bad_id}])
    assert not index.ok
    assert "id must be a string" in index.fatal[0]


@pytest.mark.parametrize("container", [None, 5, "vms", b"vms", {"id": "x#key-1"}])
def test_non_list_containers_are_fatal(material, container):
    as_methods = verify_signature_entries(DATA, _artifact_signatures(material), container)
    assert as_methods.fatal == (
        f"verification methods must be a list; got {type(container).__name__}",
    )
    as_signatures = verify_signature_entries(DATA, container, _vms(material))
    assert as_signatures.fatal == (
        f"signatures must be a list; got {type(container).__name__}",
    )


def test_evaluate_reports_fatal_reason(material):
    result = verify_signature_entries(DATA, _artifact_signatures(material), 5)
    policy = result.evaluate(VerifyPolicy.LEGACY_ALLOWED)
    assert not policy.ok
    assert policy.reason == "verification methods must be a list; got int"
    assert policy.alg_ids_seen == frozenset()


def test_evaluate_names_rejected_entries_when_policy_fails(material):
    _classical, pq = _artifact_signatures(material)
    result = verify_signature_entries(
        DATA, [{"alg": ALG_ED25519, "kid": "key-99", "sig": "00"}, pq], _vms(material),
    )
    policy = result.evaluate(VerifyPolicy.HYBRID_REQUIRED)
    assert not policy.ok
    assert "missing: classical" in policy.reason
    assert (
        "rejected signature entries: signature[0]: no verification method "
        "for kid 'key-99'"
    ) in policy.reason


def test_evaluate_passing_policy_keeps_its_reason(material):
    result = verify_signature_entries(
        DATA, ["junk", *_artifact_signatures(material)], _vms(material),
    )
    policy = result.evaluate(VerifyPolicy.HYBRID_REQUIRED)
    assert policy.ok
    assert policy.reason == "HYBRID_REQUIRED satisfied (classical + PQ both present)"


def test_evaluate_applies_the_callers_temporal_cutoff(material):
    classical, _pq = _artifact_signatures(material)
    result = verify_signature_entries(DATA, [classical], _vms(material))
    assert result.evaluate(VerifyPolicy.LEGACY_ALLOWED).ok
    cut_off = result.evaluate(
        VerifyPolicy.LEGACY_ALLOWED, post_cutoff_classical_allowed=False,
    )
    assert not cut_off.ok
    assert "post-cutoff" in cut_off.reason


def test_diagnostics_are_bounded(material):
    """Attacker-chosen kids and entry counts cannot inflate a reason string
    that ends up in logs and HTTP error details."""
    huge_kid = "k" * 100_000
    junk = [{"alg": ALG_ED25519, "kid": huge_kid, "sig": "00"}] * 50
    result = verify_signature_entries(DATA, junk, _vms(material))
    assert len(result.rejected) == 50
    assert all(len(r.reason) < 200 for r in result.rejected)
    reason = result.evaluate(VerifyPolicy.LEGACY_ALLOWED).reason
    assert "... and 42 more" in reason
    assert len(reason) < 2_000

    duplicates = [{**_vms(material)[0], "id": f"x#{huge_kid}"}] * 50
    fatal = verify_signature_entries(DATA, [], duplicates)
    assert len(fatal.fatal) == 49
    assert len(fatal.evaluate(VerifyPolicy.LEGACY_ALLOWED).reason) < 5_000


@pytest.mark.parametrize("field_name", ["alg", "kid", "sig"])
def test_unrenderable_field_value_is_rejected_not_raised(material, field_name):
    """``repr`` of a >4300-digit int raises ValueError; a diagnostic must
    never render an attacker value that way."""
    classical, pq = _artifact_signatures(material)
    result = verify_signature_entries(
        DATA, [{**classical, field_name: 10 ** 5000}, pq], _vms(material),
    )
    assert result.rejected[0].reason == (
        f"{field_name!r} must be a non-empty string; got <int>"
    )


def test_entry_is_checked_and_counted_as_one_snapshot(material):
    """A mapping that reports one alg through ``get`` and another through
    ``[]`` must not verify as Ed25519 and then count as ML-DSA-65."""
    from collections.abc import Mapping

    class TwoFaced(Mapping):
        def __init__(self, real):
            self._real = real

        def get(self, key, default=None):
            return self._real.get(key, default)

        def __getitem__(self, key):
            return ALG_ML_DSA_65 if key == "alg" else self._real[key]

        def __iter__(self):
            return iter(self._real)

        def __len__(self):
            return len(self._real)

    classical, _pq = _artifact_signatures(material)
    result = verify_signature_entries(DATA, [TwoFaced(classical)], _vms(material))
    assert result.verified == ()
    assert not result.evaluate(VerifyPolicy.PQ_REQUIRED).ok


def test_decode_error_text_is_bounded(material, monkeypatch):
    from kestrel_sovereign.security import signature_verification

    def fail(_multibase):
        raise CryptoSuiteError("x" * 100_000)

    monkeypatch.setattr(signature_verification, "multibase_to_public_key", fail)
    result = verify_signature_entries(DATA, _artifact_signatures(material), _vms(material))
    assert len(result.rejected) == 2
    assert all(len(r.reason) < 300 for r in result.rejected)


def test_decode_verification_method_key(material):
    vm = _vms(material)[0]
    key, reason = decode_verification_method_key(vm)
    assert reason == ""
    assert key[0].alg_id == ALG_ED25519

    assert decode_verification_method_key({}) == (
        None, "has no string publicKeyMultibase",
    )
    key, reason = decode_verification_method_key({"publicKeyMultibase": "z0OIl"})
    assert key is None
    assert reason.startswith("multikey decode failed: malformed multibase payload")


# ---------------------------------------------------------------------------
# Multikey decoding raises its typed error for every malformation.
# ---------------------------------------------------------------------------

MALFORMED_MULTIBASE = [
    pytest.param(5, id="int"),
    pytest.param(None, id="null"),
    pytest.param(["z6Mk"], id="list"),
    pytest.param("", id="empty"),
    pytest.param("mAAAA", id="wrong-prefix"),
    pytest.param("z0OIl", id="non-base58btc"),
    pytest.param("zé", id="non-ascii"),
    pytest.param("z", id="empty-payload"),
    pytest.param(_multibase(b"\x80"), id="varint-truncated"),
    pytest.param(_multibase(b"\xff" * 10 + b"\x01"), id="varint-overlong"),
    pytest.param(_multibase(encode_varint(0x1234) + b"\x00" * 32), id="unregistered"),
]


@pytest.mark.parametrize("value", MALFORMED_MULTIBASE)
def test_signing_multikey_decode_raises_crypto_suite_error(value):
    with pytest.raises(CryptoSuiteError):
        multibase_to_public_key(value)


@pytest.mark.parametrize("value", MALFORMED_MULTIBASE)
def test_kem_multikey_decode_raises_kem_suite_error(value):
    with pytest.raises(KEMSuiteError):
        multibase_to_kem_public_key(value)
