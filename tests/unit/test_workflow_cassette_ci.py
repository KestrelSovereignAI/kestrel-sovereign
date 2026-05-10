"""Tests for the workflow cassette CI bootstrap helper."""

from __future__ import annotations

import json

import pytest

from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)
from scripts.workflow_cassette_ci import (
    DEFAULT_CI_RETENTION_SECONDS,
    DEFAULT_TRUSTED_REF,
    OPERATOR_PUBLIC_KEY_ENV,
    canonical_record_approval_payload,
    encrypt_payload_file_from_env,
    validate_record_gate_from_env,
    verify_operator_record_signature,
)
from tests.fixtures.workflow_harness import (
    CASSETTE_KEY_ENV,
    CASSETTE_SECRET_STORE_ENV,
    WorkflowCassetteError,
)


def _signed_env(**overrides):
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    keypair = suite.generate_keypair()
    payload = canonical_record_approval_payload(
        repository="KestrelSovereignAI/kestrel-sovereign",
        ref=DEFAULT_TRUSTED_REF,
        sha="a" * 40,
        owner_did="did:web:k.example",
        cassette_id="red-team/review-1",
    )
    env = {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": "KestrelSovereignAI/kestrel-sovereign",
        "GITHUB_REF": DEFAULT_TRUSTED_REF,
        "GITHUB_SHA": "a" * 40,
        CASSETTE_SECRET_STORE_ENV: "approved",
        CASSETTE_KEY_ENV: "test cassette root key",
        "WORKFLOW_CASSETTE_OPERATOR_DID": "did:web:operator.example",
        "WORKFLOW_CASSETTE_OPERATOR_SIGNATURE": suite.sign(
            payload,
            keypair.private_key,
        ).hex(),
        OPERATOR_PUBLIC_KEY_ENV: suite.serialize_public_key(
            keypair.public_key,
        ).hex(),
        "WORKFLOW_CASSETTE_OWNER_DID": "did:web:k.example",
        "WORKFLOW_CASSETTE_ID": "red-team/review-1",
    }
    env.update(overrides)
    return env


def test_validate_record_gate_requires_manual_dispatch():
    env = _signed_env(GITHUB_EVENT_NAME="pull_request")

    with pytest.raises(WorkflowCassetteError, match="workflow_dispatch"):
        validate_record_gate_from_env(env)


def test_validate_record_gate_requires_manual_secret_store_approval():
    env = _signed_env(**{CASSETTE_SECRET_STORE_ENV: ""})

    with pytest.raises(WorkflowCassetteError, match="manual approval"):
        validate_record_gate_from_env(env)


def test_validate_record_gate_requires_trusted_ref():
    env = _signed_env(GITHUB_REF="refs/heads/feature")

    with pytest.raises(WorkflowCassetteError, match="trusted ref"):
        validate_record_gate_from_env(env)


def test_validate_record_gate_does_not_require_cassette_key():
    env = _signed_env()
    env.pop(CASSETTE_KEY_ENV)

    payload = validate_record_gate_from_env(env)

    assert json.loads(payload)["ref"] == DEFAULT_TRUSTED_REF


def test_verify_operator_record_signature_binds_ref_and_cassette():
    env = _signed_env()

    payload = validate_record_gate_from_env(env)

    assert json.loads(payload) == {
        "cassette_id": "red-team/review-1",
        "mode": "record",
        "owner_did": "did:web:k.example",
        "ref": DEFAULT_TRUSTED_REF,
        "repository": "KestrelSovereignAI/kestrel-sovereign",
        "sha": "a" * 40,
    }
    with pytest.raises(WorkflowCassetteError, match="did not verify"):
        verify_operator_record_signature(
            operator_did=env["WORKFLOW_CASSETTE_OPERATOR_DID"],
            operator_signature_hex=env["WORKFLOW_CASSETTE_OPERATOR_SIGNATURE"],
            operator_public_key_hex=env[OPERATOR_PUBLIC_KEY_ENV],
            repository=env["GITHUB_REPOSITORY"],
            ref=env["GITHUB_REF"],
            sha="b" * 40,
            owner_did=env["WORKFLOW_CASSETTE_OWNER_DID"],
            cassette_id=env["WORKFLOW_CASSETTE_ID"],
        )


def test_encrypt_payload_file_writes_only_encrypted_envelope(tmp_path):
    env = _signed_env()
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(
        json.dumps(
            {
                "request": {"pr_diff": "SECRET_DIFF"},
                "response": {"blockers": []},
            }
        ),
        encoding="utf-8",
    )

    path = encrypt_payload_file_from_env(
        payload_file=payload_file,
        output_dir=tmp_path / "tests" / "cassettes",
        env=env,
    )

    on_disk = path.read_text(encoding="utf-8")
    assert path.suffix == ".enc"
    assert "SECRET_DIFF" not in on_disk
    assert "ciphertext" in on_disk
    envelope = json.loads(on_disk)
    assert envelope["expires_at"] - envelope["created_at"] == DEFAULT_CI_RETENTION_SECONDS


def test_encrypt_payload_file_allows_explicit_retention(tmp_path):
    env = _signed_env(KESTREL_WORKFLOW_CASSETTE_RETENTION_SECONDS="86400")
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps({"response": {"ok": True}}), encoding="utf-8")

    path = encrypt_payload_file_from_env(
        payload_file=payload_file,
        output_dir=tmp_path / "tests" / "cassettes",
        env=env,
    )

    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["expires_at"] - envelope["created_at"] == 86400
