"""Signed native cards, independent DID trust, and official JSON round trips."""

import base64
from copy import deepcopy

import pytest
import rfc8785
from kestrel_sdk.native_a2a.server.request_handlers.response_helpers import agent_card_to_dict
from kestrel_sdk.native_a2a.types import AgentCard, AgentSkill
from kestrel_sdk.native_a2a.utils.signing import InvalidSignaturesError, create_agent_card_signer
from google.protobuf.json_format import ParseDict

from kestrel_sovereign.a2a.extensions import RECIPIENT_POLICY_URI, SIGNED_SENDER_URI
from kestrel_sovereign.a2a.native_card import (
    REQUIRED_NATIVE_EXTENSIONS,
    create_native_agent_card,
    verify_native_agent_card,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from tests.utils.native_a2a import signing_identity


@pytest.fixture(scope="module")
def card_identity():
    pair, document = signing_identity("did:web:cards.test:recipient")
    return AgentIdentity(
        hybrid_keypair=pair,
        new_did=document["id"],
        new_verification_methods=document["verificationMethod"],
    ), document


@pytest.fixture
def card_arguments(card_identity):
    return {
        "identity": card_identity[0],
        "scope": NativeRecipientScope("did:recipient", "tenant"),
        "routing_key": "Stable Key",
        "public_base_url": "https://host.test/proxy",
        "name": "Live Name",
        "description": "Native recipient",
        "version": "1.0.0",
        "skills": [
            AgentSkill(
                id="ask", name="Ask", description="Ask a question", tags=["chat"]
            )
        ],
    }


def verify(card, card_identity):
    identity, document = card_identity
    verify_native_agent_card(
        card, signing_did=identity.signing_did, trusted_document=document
    )


def test_native_card_uses_sdk_json_and_independent_trusted_did(
    card_arguments, card_identity
):
    card = create_native_agent_card(**card_arguments)
    verify(card, card_identity)
    wire = agent_card_to_dict(card)
    assert wire["description"] == "Native recipient"
    assert wire["skills"][0]["tags"] == ["chat"]
    assert "url" not in wire and "authentication" not in wire
    verify(ParseDict(wire, AgentCard()), card_identity)
    assert {
        ext.uri for ext in card.capabilities.extensions if ext.required
    } == REQUIRED_NATIVE_EXTENSIONS
    assert REQUIRED_NATIVE_EXTENSIONS == {SIGNED_SENDER_URI, RECIPIENT_POLICY_URI}
    assert all(interface.tenant == "tenant" for interface in card.supported_interfaces)
    assert all(
        "Live Name" not in interface.url for interface in card.supported_interfaces
    )
    assert not card.capabilities.push_notifications
    assert not card.capabilities.extended_agent_card
    assert (
        card.security_schemes["signedSender"].api_key_security_scheme.location
        == "header"
    )
    assert card_arguments["skills"][0] == card.skills[0]
    card.skills[0].name = "Changed"
    assert card_arguments["skills"][0].name == "Ask"


def test_public_wire_card_verifies_with_independent_jcs_and_ed25519(
    card_arguments, card_identity
):
    card = create_native_agent_card(**card_arguments)
    wire = agent_card_to_dict(card)
    signatures = wire.pop("signatures")
    # This independent oracle covers this builder's valid public JSON contract.
    # Upstream's field-presence vectors separately cover optional/required empties.
    payload = base64.urlsafe_b64encode(rfc8785.dumps(wire)).rstrip(b"=")
    signature = signatures[0]
    signing_input = signature["protected"].encode() + b"." + payload
    raw_signature = base64.urlsafe_b64decode(signature["signature"] + "==")
    card_identity[0].hybrid_keypair.classical.public_key.verify(
        raw_signature, signing_input
    )


def test_push_capability_is_part_of_the_signed_native_contract(
    card_arguments, card_identity
):
    card = create_native_agent_card(**card_arguments, push_notifications=True)
    assert card.capabilities.push_notifications
    verify(card, card_identity)
    card.capabilities.push_notifications = False
    with pytest.raises(InvalidSignaturesError):
        verify(card, card_identity)


@pytest.mark.parametrize("value", [1, "true", None])
def test_push_capability_requires_an_explicit_boolean(card_arguments, value):
    with pytest.raises(TypeError, match="boolean"):
        create_native_agent_card(**card_arguments, push_notifications=value)


@pytest.mark.parametrize(
    "field", ["name", "description", "tenant", "recipient", "required", "skill"]
)
def test_signed_contract_mutations_fail(card_arguments, card_identity, field):
    card = create_native_agent_card(**card_arguments)
    if field in {"name", "description"}:
        setattr(card, field, "tampered")
    elif field == "tenant":
        card.supported_interfaces[0].tenant = "other"
    elif field == "recipient":
        card.capabilities.extensions[0].params["recipient"] = "other"
    elif field == "required":
        card.capabilities.extensions[0].required = False
    else:
        card.skills[0].description = "tampered"
    with pytest.raises(InvalidSignaturesError):
        verify(card, card_identity)


@pytest.mark.parametrize(
    "fault",
    ["other-document", "duplicate-kid", "foreign-method", "bad-key", "no-ed25519"],
)
def test_card_cannot_introduce_or_substitute_trust(
    card_arguments, card_identity, fault
):
    card = create_native_agent_card(**card_arguments)
    document = deepcopy(card_identity[1])
    if fault == "other-document":
        document["id"] = "did:web:attacker.test"
    elif fault == "duplicate-kid":
        document["verificationMethod"].append(document["verificationMethod"][0])
    elif fault == "foreign-method":
        document["verificationMethod"][0]["id"] = "did:web:attacker.test#key-1"
    elif fault == "bad-key":
        document["verificationMethod"][0]["publicKeyMultibase"] = "invalid"
    else:
        document["verificationMethod"] = document["verificationMethod"][1:]
    with pytest.raises(ValueError):
        verify_native_agent_card(
            card, signing_did=card_identity[0].signing_did, trusted_document=document
        )


@pytest.mark.parametrize(
    "jku,kid",
    [("https://attacker.test/keys", None), (None, "did:web:attacker.test#key-1")],
)
def test_card_signature_never_fetches_keys_from_jws(
    card_arguments, card_identity, jku, kid
):
    card = create_native_agent_card(**card_arguments)
    card.ClearField("signatures")
    header = {
        "alg": "EdDSA",
        "kid": kid or card_identity[1]["verificationMethod"][0]["id"],
    }
    if jku:
        header["jku"] = jku
    create_agent_card_signer(
        card_identity[0].hybrid_keypair.classical.private_key, header
    )(card)
    with pytest.raises(InvalidSignaturesError):
        verify(card, card_identity)


@pytest.mark.parametrize("untrusted", ["jku", "kid"])
def test_untrusted_signature_cannot_hide_a_later_independently_trusted_signature(
    card_arguments, card_identity, untrusted
):
    card = create_native_agent_card(**card_arguments)
    trusted = deepcopy(card.signatures[0])
    card.ClearField("signatures")
    header = {"alg": "EdDSA", "kid": card_identity[1]["verificationMethod"][0]["id"]}
    if untrusted == "jku":
        header["jku"] = "https://attacker.test/keys"
    else:
        header["kid"] = "did:web:attacker.test#key-1"
    create_agent_card_signer(
        card_identity[0].hybrid_keypair.classical.private_key, header
    )(card)
    card.signatures.append(trusted)
    verify(card, card_identity)


def test_card_keys_must_match_current_recipient_identity(card_arguments):
    other, _ = signing_identity("did:web:other.test")
    card_arguments["identity"] = AgentIdentity(
        hybrid_keypair=other,
        new_did=card_arguments["identity"].signing_did,
        new_verification_methods=card_arguments["identity"].new_verification_methods,
    )
    with pytest.raises(ValueError, match="do not match"):
        create_native_agent_card(**card_arguments)


@pytest.mark.parametrize(
    "field,value",
    [("public_base_url", ""), ("name", " "), ("version", ""), ("identity", None)],
)
def test_missing_host_configuration_fails_before_card_publication(
    card_arguments, field, value
):
    card_arguments[field] = value
    with pytest.raises(ValueError):
        create_native_agent_card(**card_arguments)


def test_duplicate_skills_are_not_an_ambiguous_public_contract(card_arguments):
    card_arguments["skills"] *= 2
    with pytest.raises(ValueError, match="unique"):
        create_native_agent_card(**card_arguments)
