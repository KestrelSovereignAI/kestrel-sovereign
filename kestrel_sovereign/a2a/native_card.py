"""Native signed discovery using official Agent Cards and SDK JWS services."""

from collections.abc import Sequence

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentProvider,
    AgentSkill,
    APIKeySecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from a2a.utils.proto_utils import validate_proto_required_fields
from a2a.utils.signing import create_agent_card_signer, create_signature_verifier
from jwt.exceptions import InvalidKeyError

from kestrel_sovereign.a2a.extensions import (
    NATIVE_EXTENSION_URIS,
    PEER_STOP_URI,
    RECIPIENT_POLICY_URI,
    SIGNED_SENDER_URI,
)
from kestrel_sovereign.a2a.media import DEFAULT_NATIVE_MEDIA_TYPES
from kestrel_sovereign.a2a.native_routes import native_http_interfaces
from kestrel_sovereign.a2a.native_sender import SENDER_PROOF_HEADER
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.security.crypto_suite import ALG_ED25519, get_suite
from kestrel_sovereign.security.signature_verification import (
    decode_verification_method_key,
    index_verification_methods,
)

REQUIRED_NATIVE_EXTENSIONS = frozenset({SIGNED_SENDER_URI, RECIPIENT_POLICY_URI})


def _card_keys(document: dict, signing_did: str) -> dict:
    """Use a trusted resolver's document, never keys or URLs supplied by a card."""
    if not signing_did or document.get("id") != signing_did:
        raise ValueError("Agent Card verification requires the expected signing DID")
    indexed = index_verification_methods(document.get("verificationMethod"))
    if indexed.fatal:
        raise ValueError("Agent Card DID document has ambiguous verification methods")
    keys = {}
    for method in indexed.methods.values():
        key_id = method["id"]
        if not key_id.startswith(signing_did + "#"):
            raise ValueError("Agent Card verification method belongs to another DID")
        decoded, reason = decode_verification_method_key(method)
        if decoded is None:
            raise ValueError(f"Agent Card verification method {reason}")
        suite, key = decoded
        if suite.alg_id == ALG_ED25519:
            keys[key_id] = key
    if not keys:
        raise ValueError("Agent Card signing requires an Ed25519 verification method")
    return keys


def verify_native_agent_card(
    card: AgentCard, *, signing_did: str, trusted_document: dict
) -> None:
    """Verify SDK JWS against independently resolved, explicitly selected identity.

    A JWS ``jku`` cannot introduce a key-fetch side effect. Standard EdDSA card
    signing is distinct from the hybrid proof required on operational requests.
    """
    keys = _card_keys(trusted_document, signing_did)

    def key_provider(kid, jku):
        if jku is not None or kid not in keys:
            raise InvalidKeyError("Agent Card key is not in the trusted DID document")
        return keys[kid]

    create_signature_verifier(key_provider, algorithms=["EdDSA"])(card)


def create_native_agent_card(
    *,
    identity: AgentIdentity,
    scope: NativeRecipientScope,
    routing_key: str,
    public_base_url: str,
    name: str,
    description: str,
    version: str,
    skills: Sequence[AgentSkill],
    peer_stop: bool = False,
) -> AgentCard:
    """Build and sign the one public contract for a hosted native recipient."""
    if not isinstance(identity, AgentIdentity) or not identity.is_hybrid:
        raise ValueError("Native discovery requires a hybrid recipient identity")
    pair = identity.hybrid_keypair
    if pair.classical.suite_id != ALG_ED25519:
        raise ValueError("Native Agent Card JWS requires an Ed25519 identity")
    document = {
        "id": identity.signing_did,
        "verificationMethod": identity.new_verification_methods,
    }
    keys = _card_keys(document, identity.signing_did)
    suite = get_suite(ALG_ED25519)
    public_bytes = suite.serialize_public_key(pair.classical.public_key)
    matching = [
        kid
        for kid, key in keys.items()
        if suite.serialize_public_key(key) == public_bytes
    ]
    if len(matching) != 1:
        raise ValueError("Agent Card identity keys do not match its DID document")
    if not name.strip() or not version.strip():
        raise ValueError("Agent Card requires an explicit name and version")
    if len({skill.id for skill in skills}) != len(skills):
        raise ValueError("Agent Card skill identifiers must be unique")
    card = AgentCard(
        name=name,
        description=description,
        version=version,
        supported_interfaces=native_http_interfaces(
            public_base_url, routing_key=routing_key, scope=scope
        ),
        provider=AgentProvider(
            organization="Kestrel Sovereign AI",
            url="https://github.com/KestrelSovereignAI/kestrel-sovereign",
        ),
        capabilities=AgentCapabilities(
            streaming=True,
            extensions=[
                AgentExtension(
                    uri=uri,
                    required=uri in REQUIRED_NATIVE_EXTENSIONS,
                    params=(
                        {
                            "version": 1,
                            "recipient": scope.recipient_id,
                            "signing_did": identity.signing_did,
                            "header": SENDER_PROOF_HEADER,
                        }
                        if uri == SIGNED_SENDER_URI
                        else {"version": 1}
                    ),
                )
                for uri in NATIVE_EXTENSION_URIS
                if uri != PEER_STOP_URI or peer_stop
            ],
        ),
        security_schemes={
            "signedSender": SecurityScheme(
                api_key_security_scheme=APIKeySecurityScheme(
                    location="header", name=SENDER_PROOF_HEADER
                )
            )
        },
        security_requirements=[
            SecurityRequirement(schemes={"signedSender": StringList()})
        ],
        default_input_modes=DEFAULT_NATIVE_MEDIA_TYPES,
        default_output_modes=DEFAULT_NATIVE_MEDIA_TYPES,
        skills=skills,
    )
    validate_proto_required_fields(card)
    return create_agent_card_signer(
        pair.classical.private_key,
        {"kid": matching[0], "alg": "EdDSA", "typ": "JOSE"},
    )(card)
