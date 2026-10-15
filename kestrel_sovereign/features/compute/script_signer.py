"""
Kestrel Compute Feature - Script Signer.

Cryptographic signing of scripts using the agent's secp256k1 DID key.
Ensures scripts cannot be tampered with after agent creates them.

Sign-or-fail: if the agent's signing keys are unavailable, signing raises
``ScriptSigningKeysUnavailable``. There is no fallback. Verification rejects
the historical ``hmac:`` prefix, which used a public DID as the HMAC key and
was forgeable by anyone who could read the script. See
``docs/architecture/security/CRYPTO_INVENTORY.md`` and the Wave 0B section of
the Quantum Hardening epic.

Signature versions (#2484). A signature string names the version of the
bytes it signs:

- **v2** (``v2:ecdsa:…`` / ``v2:hybrid:…``): signs
  :func:`canonical_script_bytes_v2`, a sequence of length-framed byte strings
  carrying a domain tag, the version, and each signed field's name and exact
  UTF-8 value. Every new signature is v2.
- **v1** (untagged ``ecdsa:…`` / ``hybrid:…``): signs a hash of
  ``name|language|content|purpose``. The ``|`` is neither framed nor escaped,
  so moving one between adjacent fields leaves the digest unchanged and a v1
  signature over one script verifies for another. v1 is verified only during
  the compatibility window that :data:`ACCEPT_LEGACY_SIGNATURES_ENV` closes,
  and each v1 acceptance logs a WARNING naming the script.

Any other ``v<N>:`` tag is an unknown version and is refused; it is never
verified as v1. The removal plan for v1 is in
``docs/architecture/COMPUTE_FEATURE_DESIGN.md`` §3.6.
"""

import hashlib
import logging
import os
import re
import struct
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Tuple

from .models import ComputeScript

logger = logging.getLogger(__name__)

#: Version of every signature :meth:`ScriptSigner.sign` produces.
CURRENT_SIGNATURE_VERSION = 2
#: Version of the untagged legacy signatures, verified only while the
#: compatibility window is open.
LEGACY_SIGNATURE_VERSION = 1

#: Environment switch for the v1 compatibility window. Unset means open.
ACCEPT_LEGACY_SIGNATURES_ENV = "KESTREL_COMPUTE_ACCEPT_LEGACY_SCRIPT_SIGNATURES"

_V2_TAG = f"v{CURRENT_SIGNATURE_VERSION}:"
# Matches any version tag. A tag other than _V2_TAG ("v3:", "v02:", "v1:")
# is an unknown version: v1 signatures never carried a tag.
_VERSION_TAG = re.compile(r"v[0-9]+:")
_SIGNATURE_SCHEMES = frozenset({"ecdsa", "hybrid"})

# The v2 canonical form. Changing any of these changes the signed bytes and
# therefore requires a new signature version.
_V2_DOMAIN = b"kestrel.compute.script-signature"
_V2_FIELDS = ("name", "language", "content", "purpose")
_V2_FRAME_LENGTH = struct.Struct(">Q")

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class ScriptSigningKeysUnavailable(Exception):
    """Raised when ScriptSigner.sign is invoked without usable signing keys."""


class ScriptSignatureEncodingError(ValueError):
    """Raised when a script's signed fields cannot be encoded canonically:
    a field is not a ``str``, or it holds a lone surrogate that has no UTF-8
    encoding."""


def accept_legacy_signatures_from_env(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Read the v1 compatibility-window switch.

    Unset or empty means the window is open (v1 accepted). An unrecognised
    value raises rather than guessing which way a security switch was meant.
    """
    env = os.environ if environ is None else environ
    raw = env.get(ACCEPT_LEGACY_SIGNATURES_ENV, "").strip().lower()
    if not raw:
        return True
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{ACCEPT_LEGACY_SIGNATURES_ENV}={raw!r} is not a boolean; use one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}."
    )


def _frame(data: bytes) -> bytes:
    return _V2_FRAME_LENGTH.pack(len(data)) + data


def _encode_signed_field(script: ComputeScript, field_name: str) -> bytes:
    value = getattr(script, field_name)
    if not isinstance(value, str):
        raise ScriptSignatureEncodingError(
            f"Script field {field_name!r} must be str to be signed, "
            f"got {type(value).__name__}"
        )
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ScriptSignatureEncodingError(
            f"Script field {field_name!r} has no UTF-8 encoding: {e.reason}"
        ) from e


def canonical_script_bytes_v2(script: ComputeScript) -> bytes:
    """Return the bytes a v2 script signature signs.

    The layout is a sequence of frames, each an unsigned 64-bit big-endian
    byte length followed by that many bytes::

        frame(b"kestrel.compute.script-signature")
        frame(b"2")
        frame(b"name")     frame(utf8(name))
        frame(b"language") frame(utf8(language))
        frame(b"content")  frame(utf8(content))
        frame(b"purpose")  frame(utf8(purpose))

    Length framing makes the encoding injective: no choice of field values
    can move bytes from one field into another. Values are bound exactly as
    stored, with no Unicode normalisation, so two strings that render alike
    but differ in code points sign differently. The leading domain tag names
    what the bytes are, so a signature over them is not also a signature over
    another kind of artifact unless that artifact's encoding begins with the
    same frame. The version frame binds the version inside the signed bytes.

    Raises:
        ScriptSignatureEncodingError: a signed field is not a ``str`` or
            cannot be encoded as UTF-8.
    """
    frames = [_V2_DOMAIN, str(CURRENT_SIGNATURE_VERSION).encode("ascii")]
    for field_name in _V2_FIELDS:
        frames.append(field_name.encode("ascii"))
        frames.append(_encode_signed_field(script, field_name))
    return b"".join(_frame(frame) for frame in frames)


def _legacy_v1_signing_message(script: ComputeScript) -> bytes:
    """Return the bytes a legacy v1 signature signs.

    Kept byte-for-byte as it was signed so existing signatures verify during
    the compatibility window. Ambiguous by construction (see module
    docstring); never used to sign.
    """
    canonical = f"{script.name}|{script.language}|{script.content}|{script.purpose}"
    try:
        content_hash = hashlib.sha256(canonical.encode()).hexdigest()
    except UnicodeEncodeError as e:
        raise ScriptSignatureEncodingError(
            f"Script fields have no UTF-8 encoding: {e.reason}"
        ) from e
    return hashlib.sha256(content_hash.encode()).digest()


class ScriptSigner:
    """
    Sign and verify scripts using secp256k1 ECDSA cryptography.
    
    Uses the agent's DID key (same as used for Ethereum identity) to sign 
    script content, creating a tamper-evident record of script creation.
    Verification ensures the script hasn't been modified since signing.
    
    Example:
        signer = ScriptSigner(agent_did="did:ethr:0x123...", db_path="kestrel.db")
        
        # Sign a script
        signature = await signer.sign(script)
        
        # Later, verify it hasn't been tampered with
        is_valid = await signer.verify(script)
    """
    
    def __init__(
        self,
        agent_did: Optional[str],
        db_path: str,
        *,
        accept_legacy_signatures: bool = True,
    ):
        """
        Initialize the script signer.

        Args:
            agent_did: The agent's DID (did:ethr:0x...)
            db_path: Path to the database (for key retrieval)
            accept_legacy_signatures: Whether untagged v1 signatures still
                verify. False closes the compatibility window.
        """
        self.agent_did = agent_did
        self.db_path = db_path
        self.accept_legacy_signatures = accept_legacy_signatures
        self._private_key = None
        self._public_key = None
        self._key_id: Optional[str] = None
        # Set when the agent has completed a hybrid-rotation ceremony.
        # When non-None, sign() emits a ``hybrid:`` token combining
        # Ed25519 + ML-DSA-65 signatures; verify() accepts both
        # ``hybrid:`` and the legacy ``ecdsa:`` formats so existing
        # signed scripts on disk continue to verify.
        self._agent_identity = None
    
    def _extract_key_id(self) -> Optional[str]:
        """Extract key_id from DID for key lookup."""
        if not self.agent_did:
            return None
        
        # Format: did:ethr:0x123... -> kestrel_0x123...
        if self.agent_did.startswith("did:ethr:"):
            address = self.agent_did.replace("did:ethr:", "")
            return f"kestrel_{address}"
        
        # Other DID formats - try extracting address-like suffix
        parts = self.agent_did.split(":")
        if len(parts) >= 3 and parts[-1].startswith("0x"):
            return f"kestrel_{parts[-1]}"

        return None

    def _is_born_hybrid_did(self) -> bool:
        """did:web agents (#2397) have no legacy key id — their keys are
        discovered from the <slug>_did.json document on disk."""
        return bool(self.agent_did) and self.agent_did.startswith("did:web:")
    
    async def _load_keys(self) -> bool:
        """
        Load the agent's secp256k1 keys from Kestrel's key storage.
        
        Returns:
            True if keys loaded successfully, False otherwise
        """
        if self._private_key is not None:
            return True
        
        try:
            # did:web is checked FIRST: _extract_key_id's address-suffix
            # heuristic would misroute a did:web slug like "0xcafe" to a
            # nonexistent legacy key path and strand the hybrid keys.
            key_id = None if self._is_born_hybrid_did() else self._extract_key_id()
            if not key_id and not self._is_born_hybrid_did():
                logger.warning(f"Cannot extract key_id from DID: {self.agent_did}")
                return False
            
            self._key_id = key_id
            
            # Try to use Kestrel's inception_service to load keys
            from kestrel_sovereign.inception_service import load_kestrel_identity
            from kestrel_sovereign.storage import get_default_agent_data_dir
            
            # Determine storage directory from db_path
            db_dir = Path(self.db_path).parent
            if db_dir.name == "":
                db_dir = Path(get_default_agent_data_dir())
            
            # Try the hybrid-aware loader first. If a succession statement
            # is on disk, this returns an AgentIdentity carrying BOTH the
            # legacy ECDSA keypair (for verifying pre-rotation artifacts)
            # AND the new Ed25519 + ML-DSA-65 hybrid keypair (for signing
            # NEW artifacts). FileNotFoundError → pre-ceremony agent
            # without any key on disk yet → fall through to legacy.
            # RuntimeIdentityError → INCONSISTENT post-ceremony state
            # (succession statement present but hybrid keys missing or
            # corrupt) → propagate; silently downgrading to legacy would
            # mask a security-critical key-state problem (codex P2 catch).
            from kestrel_sovereign.identity.runtime_identity import (
                RuntimeIdentityError, load_agent_identity,
            )
            try:
                self._agent_identity = load_agent_identity(key_id, storage_dir=db_dir)
                # Bind the loaded born-hybrid identity to THIS agent's
                # DID — the slug-based loader returns whatever identity
                # is in the dir (same principle as
                # signing._load_local_identity_for_did).
                if key_id is None and self._agent_identity.new_did != self.agent_did:
                    logger.warning(
                        f"Identity on disk is {self._agent_identity.new_did!r} "
                        f"but this signer's agent DID is {self.agent_did!r} — "
                        f"refusing the mis-bound identity."
                    )
                    self._agent_identity = None
                    return False
                # Born-hybrid agents (#2397) carry no legacy keypair at
                # all; hybrid signing below covers them. Only populate
                # the legacy ECDSA slots when legacy material exists.
                if self._agent_identity.legacy_keypair is not None:
                    self._private_key = self._agent_identity.legacy_keypair.private_key
                    self._public_key = self._agent_identity.legacy_keypair.public_key
                if self._agent_identity.is_born_hybrid:
                    logger.info(
                        f"Loaded BORN-HYBRID signing keys: "
                        f"{self._agent_identity.new_did}"
                    )
                elif self._agent_identity.is_hybrid:
                    logger.info(
                        f"Loaded HYBRID signing keys for {key_id}: "
                        f"legacy={self._agent_identity.legacy_did} -> "
                        f"new={self._agent_identity.new_did}"
                    )
                else:
                    logger.info(f"Loaded signing keys for {key_id} (legacy-only)")
                return True
            except FileNotFoundError as e:
                if key_id is None:
                    # Born-hybrid DID with no identity material on disk:
                    # there is no legacy key file to fall through to.
                    logger.warning(
                        f"No born-hybrid identity on disk for "
                        f"{self.agent_did}: {e}"
                    )
                    return False
                logger.debug(
                    f"No identity on disk for {key_id}; falling through to "
                    f"legacy load: {e}"
                )
            except RuntimeIdentityError:
                # Inconsistent post-ceremony state — propagate. Do NOT
                # silently downgrade to legacy single-key signing.
                raise

            private_key, did_document = load_kestrel_identity(key_id, db_dir)

            self._private_key = private_key
            self._public_key = private_key.public_key()
            logger.info(f"Loaded signing keys for {key_id}")
            return True
            
        except FileNotFoundError:
            logger.warning(f"No keys found for key_id {self._key_id}")
            return False
        except ImportError as e:
            logger.warning(f"Required module not available for key loading: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to load signing keys: {e}")
            return False
    
    async def sign(self, script: ComputeScript) -> str:
        """
        Sign a script's content.

        Sign-or-fail. If the agent's keys cannot be loaded, raises
        ``ScriptSigningKeysUnavailable``. There is no fallback.

        Every signature is version 2: it signs
        :func:`canonical_script_bytes_v2` and carries the ``v2:`` tag.

        Wire format:

        - **Hybrid agent** (post-rotation ceremony, ``self._agent_identity.is_hybrid``):
          returns ``v2:hybrid:<base64>`` where the base64 wraps a JSON list of
          ``{alg, kid, sig}`` objects (Ed25519 + ML-DSA-65).
        - **Legacy-key agent** (pre-ceremony): returns ``v2:ecdsa:<base64>``
          (single secp256k1 ECDSA signature).

        Args:
            script: The script to sign

        Returns:
            Signature string with a ``v2:hybrid:`` or ``v2:ecdsa:`` prefix.

        Raises:
            ScriptSigningKeysUnavailable: when keys cannot be loaded or the
                signing operation fails.
            ScriptSignatureEncodingError: when a signed field cannot be
                encoded canonically.
        """
        import base64
        import json

        message = canonical_script_bytes_v2(script)

        # Either a legacy ECDSA private key OR a hybrid identity
        # capable of signing must be available. Post-destruction
        # agents (legacy key zapped per quantum_destroy_legacy_key.py)
        # have ``self._private_key is None`` but
        # ``self._agent_identity.is_hybrid`` True — that's a fully
        # functional signing state.
        loaded = await self._load_keys()
        has_hybrid = (
            self._agent_identity is not None
            and self._agent_identity.is_hybrid
        )
        if not loaded or (self._private_key is None and not has_hybrid):
            raise ScriptSigningKeysUnavailable(
                f"Cannot sign script {script.id[:8]}…: signing keys for "
                f"DID {self.agent_did!r} are not available. Refusing to "
                "produce a signature; the historical HMAC fallback was "
                "forgeable and has been removed."
            )

        # Hybrid path: emit a hybrid token if the agent has completed
        # the rotation ceremony.
        identity = self._agent_identity
        if identity is not None and identity.is_hybrid:
            from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid
            try:
                vms = identity.new_verification_methods or []
                classical_kid = next(
                    (
                        vm["id"].rsplit("#", 1)[-1]
                        for vm in vms
                        if "ed25519" in (vm.get("publicKeyMultibase") or "").lower()
                        or vm.get("type") == "Ed25519VerificationKey2020"
                    ),
                    vms[0]["id"].rsplit("#", 1)[-1] if vms else "key-1",
                )
                pq_kid = next(
                    (
                        vm["id"].rsplit("#", 1)[-1]
                        for vm in vms
                        if vm["id"].rsplit("#", 1)[-1] != classical_kid
                    ),
                    vms[1]["id"].rsplit("#", 1)[-1] if len(vms) > 1 else "key-2",
                )
                sigs = sign_hybrid(
                    message,
                    identity.hybrid_keypair,
                    classical_kid=classical_kid,
                    pq_kid=pq_kid,
                )
                payload = base64.b64encode(json.dumps(sigs).encode()).decode()
                return f"{_V2_TAG}hybrid:{payload}"
            except Exception as e:
                raise ScriptSigningKeysUnavailable(
                    f"Hybrid signing failed for script {script.id[:8]}…: {e}"
                ) from e

        # Legacy-key path: route through Secp256k1Suite. The suite call is
        # byte-identical to the previous direct ec.ECDSA(SHA256) call;
        # pinned by the CryptoSuite behavior-preservation pair test.
        from kestrel_sovereign.security.crypto_suite import (
            ALG_ECDSA_SECP256K1_SHA256, CryptoSuiteError, get_suite,
        )
        suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
        try:
            signature_bytes = suite.sign(message, self._private_key)
            return f"{_V2_TAG}ecdsa:" + base64.b64encode(signature_bytes).decode()
        except (CryptoSuiteError, Exception) as e:
            raise ScriptSigningKeysUnavailable(
                f"ECDSA signing failed for script {script.id[:8]}…: {e}"
            ) from e
    
    async def verify(self, script: ComputeScript) -> bool:
        """
        Verify a script's signature.

        The signature's version selects the bytes it is checked against:

        - ``v2:<scheme>:<payload>`` — :func:`canonical_script_bytes_v2`.
        - untagged ``<scheme>:<payload>`` — the legacy v1 message, only while
          ``accept_legacy_signatures`` is True. Each v1 acceptance logs a
          WARNING naming the script so it can be re-signed.

        Schemes:

        - ``hybrid`` — base64 of a JSON list of ``{alg, kid, sig}`` objects.
          Every entry must verify against the agent's loaded hybrid identity,
          and both Ed25519 and ML-DSA-65 must be present.
        - ``ecdsa`` — base64 of a single secp256k1 signature, verified against
          the agent's legacy ECDSA key, so scripts signed before the rotation
          ceremony stay valid.

        Rejects:

        - ``hmac:`` — the removed fallback that used the public DID
          as the HMAC key (forgeable by any reader).
        - Any version tag other than ``v2:`` — unknown versions are refused,
          never verified as v1.
        - Anything else — unknown format, refuse rather than guess.

        Args:
            script: The script to verify

        Returns:
            True only if a valid signature verifies; False otherwise.
        """
        if not script.signature:
            logger.warning(f"Script {script.id[:8]}… has no signature")
            return False

        if script.signature.startswith("hmac:"):
            logger.critical(
                f"Script {script.id[:8]}… carries an 'hmac:' signature. "
                "These were produced by a removed fallback that used the "
                "public DID as the HMAC key (forgeable by any reader). "
                "Rejecting; re-sign with current ECDSA keys to restore."
            )
            return False

        parsed = self._parse_signature(script)
        if parsed is None:
            return False
        version, scheme, payload = parsed

        if version == LEGACY_SIGNATURE_VERSION and not self.accept_legacy_signatures:
            logger.warning(
                f"Script {script.id} carries a legacy v1 signature and the v1 "
                f"compatibility window is closed (see "
                f"{ACCEPT_LEGACY_SIGNATURES_ENV}). Rejecting; re-sign the "
                "script to give it a v2 signature."
            )
            return False

        try:
            if version == CURRENT_SIGNATURE_VERSION:
                message = canonical_script_bytes_v2(script)
            else:
                message = _legacy_v1_signing_message(script)
        except ScriptSignatureEncodingError as e:
            logger.warning(f"Script {script.id[:8]}… cannot be verified: {e}")
            return False

        if not await self._load_keys():
            logger.warning("Cannot verify signature: failed to load keys")
            return False

        if scheme == "hybrid":
            valid = self._verify_hybrid(payload, message)
        else:
            valid = self._verify_ecdsa(payload, message)

        if valid and version == LEGACY_SIGNATURE_VERSION:
            logger.warning(
                f"Script {script.id} verified with a legacy v1 signature. v1 "
                "signatures hash an unframed 'name|language|content|purpose' "
                "string, so one can verify for a different script. Re-sign "
                "the script to give it a v2 signature before the v1 "
                f"compatibility window ({ACCEPT_LEGACY_SIGNATURES_ENV}) closes."
            )
        return valid

    @staticmethod
    def _parse_signature(script: ComputeScript) -> Optional[Tuple[int, str, str]]:
        """Split a signature into ``(version, scheme, payload)``.

        Returns None, after logging why, for an unknown version tag or an
        unknown scheme.
        """
        signature = script.signature
        if signature.startswith(_V2_TAG):
            version, body = CURRENT_SIGNATURE_VERSION, signature[len(_V2_TAG):]
        else:
            tag = _VERSION_TAG.match(signature)
            if tag is not None:
                logger.warning(
                    f"Script {script.id[:8]}… carries signature version tag "
                    f"{tag.group(0)!r}, which this verifier does not know. "
                    "Rejecting; an unknown version is never verified as v1."
                )
                return None
            version, body = LEGACY_SIGNATURE_VERSION, signature
        scheme, separator, payload = body.partition(":")
        if not separator or scheme not in _SIGNATURE_SCHEMES:
            logger.warning(f"Unknown signature format: {signature[:10]}…")
            return None
        return version, scheme, payload

    def _verify_ecdsa(self, payload: str, message: bytes) -> bool:
        """Verify a base64 secp256k1 ECDSA signature over ``message``
        against the agent's legacy public key."""
        import base64
        if self._public_key is None:
            logger.warning("Cannot verify ECDSA signature without public key")
            return False
        from kestrel_sovereign.security.crypto_suite import (
            ALG_ECDSA_SECP256K1_SHA256, get_suite,
        )
        try:
            signature_bytes = base64.b64decode(payload)
            suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
            if suite.verify(message, signature_bytes, self._public_key):
                return True
            logger.warning("ECDSA signature verification failed")
            return False
        except Exception as e:
            logger.warning(f"ECDSA signature verification failed: {e}")
            return False

    def _verify_hybrid(self, payload: str, message: bytes) -> bool:
        """Verify a base64-wrapped JSON hybrid signature array over
        ``message`` against the agent's loaded hybrid keypair. Every
        entry must verify, and both halves must be present."""
        import base64, json
        identity = self._agent_identity
        if identity is None or not identity.is_hybrid:
            logger.warning(
                "Script carries a hybrid: signature but the agent's "
                "runtime identity is not hybrid (no successions/<slug>.json "
                "on disk, or load failed). Rejecting."
            )
            return False
        try:
            sigs_json = base64.b64decode(payload).decode("utf-8")
            sigs = json.loads(sigs_json)
            if not isinstance(sigs, list) or not sigs:
                logger.warning("Hybrid signature payload is empty or not a list")
                return False
        except Exception as e:
            logger.warning(f"Hybrid signature parse failed: {e}")
            return False

        from kestrel_sovereign.security.crypto_suite import get_suite
        # Map kid (the fragment after #) → public key from the loaded
        # hybrid keypair. We pull pubkeys directly off the in-memory
        # AgentIdentity rather than re-resolving the did:web document
        # over HTTPS — this is the same agent's runtime, so its loaded
        # keys ARE authoritative.
        kid_to_pub = {}
        if identity.new_verification_methods and identity.hybrid_keypair:
            vms = identity.new_verification_methods
            # Convention from rotation_ceremony: VM[0] is classical, VM[1] is PQ
            if len(vms) >= 1:
                kid_to_pub[vms[0]["id"].rsplit("#", 1)[-1]] = (
                    identity.hybrid_keypair.classical
                )
            if len(vms) >= 2:
                kid_to_pub[vms[1]["id"].rsplit("#", 1)[-1]] = (
                    identity.hybrid_keypair.pq
                )

        # Hybrid verify rules (HYBRID_REQUIRED on script signing):
        # 1. EVERY entry in the payload must verify. A corrupted entry
        #    fails the whole signature — flipping one signature byte
        #    must not pass just because the other half still works.
        # 2. Both ed25519 AND ml-dsa-65 must be present. Stripping the
        #    PQ half to leave a classical-only payload is rejected;
        #    that's the attack hybrid is here to prevent.
        algs_seen: set[str] = set()
        for entry in sigs:
            # Every field is used as a dict key or parsed as hex below, so a
            # non-string value (e.g. a list kid) must be refused here rather
            # than raise out of verify().
            fields = (
                (entry.get("alg"), entry.get("kid"), entry.get("sig"))
                if isinstance(entry, dict) else (None, None, None)
            )
            if not all(isinstance(value, str) for value in fields):
                logger.warning("Hybrid signature entry malformed; rejecting")
                return False
            alg, kid, sig_hex = fields
            kp = kid_to_pub.get(kid)
            if kp is None or kp.suite_id != alg:
                logger.warning(
                    f"Hybrid signature kid={kid!r} alg={alg!r} doesn't map "
                    f"to a known verification method; rejecting"
                )
                return False
            try:
                # sign_hybrid encodes signatures as hex (matches the
                # identity-package v2 ``signatures`` array shape).
                sig_bytes = bytes.fromhex(sig_hex)
            except ValueError:
                logger.warning(f"Hybrid signature {kid} has malformed hex; rejecting")
                return False
            suite = get_suite(alg)
            if not suite.verify(message, sig_bytes, kp.public_key):
                logger.warning(
                    f"Hybrid signature {kid} ({alg}) failed crypto verify; rejecting"
                )
                return False
            algs_seen.add(alg)

        # Require both halves of the hybrid identity. Removing the PQ
        # half is the canonical attack the hybrid format defends against.
        from kestrel_sovereign.security.crypto_suite import (
            ALG_ED25519, ALG_ML_DSA_65,
        )
        required = {ALG_ED25519, ALG_ML_DSA_65}
        if not required.issubset(algs_seen):
            missing = required - algs_seen
            logger.warning(
                f"Hybrid signature missing required algs: {sorted(missing)}; "
                f"rejecting (HYBRID_REQUIRED on script signing)"
            )
            return False
        return True
    
    async def sign_and_update(self, script: ComputeScript) -> ComputeScript:
        """
        Sign a script and update its signature fields.
        
        Convenience method that signs and updates the script in place.
        Re-signing a script that carries a legacy v1 signature replaces it
        with a v2 signature over the fields as they are now. Only re-sign a
        script whose fields have been reviewed: a v1 signature does not prove
        they are the fields that were originally signed.
        
        Args:
            script: The script to sign
            
        Returns:
            The updated script with signature fields set
        """
        script.signature = await self.sign(script)
        script.signed_by = self.agent_did
        script.signed_at = datetime.now()
        return script
