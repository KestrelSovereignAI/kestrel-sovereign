"""
Kestrel Compute Feature - Script Signer.

Cryptographic signing of scripts using the agent's secp256k1 DID key.
Ensures scripts cannot be tampered with after agent creates them.

Sign-or-fail: if the agent's signing keys are unavailable, signing raises
``ScriptSigningKeysUnavailable``. There is no fallback. Verification rejects
any signature whose prefix is not ``ecdsa:``; the historical ``hmac:`` prefix
used a public DID as the HMAC key and was forgeable by anyone who could read
the script. See ``docs/architecture/security/CRYPTO_INVENTORY.md`` and the
Wave 0B section of the Quantum Hardening epic.
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ComputeScript

logger = logging.getLogger(__name__)


class ScriptSigningKeysUnavailable(Exception):
    """Raised when ScriptSigner.sign is invoked without usable signing keys."""


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
    
    def __init__(self, agent_did: Optional[str], db_path: str):
        """
        Initialize the script signer.

        Args:
            agent_did: The agent's DID (did:ethr:0x...)
            db_path: Path to the database (for key retrieval)
        """
        self.agent_did = agent_did
        self.db_path = db_path
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
    
    async def _load_keys(self) -> bool:
        """
        Load the agent's secp256k1 keys from Kestrel's key storage.
        
        Returns:
            True if keys loaded successfully, False otherwise
        """
        if self._private_key is not None:
            return True
        
        try:
            key_id = self._extract_key_id()
            if not key_id:
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
                self._private_key = self._agent_identity.legacy_keypair.private_key
                self._public_key = self._agent_identity.legacy_keypair.public_key
                if self._agent_identity.is_hybrid:
                    logger.info(
                        f"Loaded HYBRID signing keys for {key_id}: "
                        f"legacy={self._agent_identity.legacy_did} -> "
                        f"new={self._agent_identity.new_did}"
                    )
                else:
                    logger.info(f"Loaded signing keys for {key_id} (legacy-only)")
                return True
            except FileNotFoundError as e:
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
    
    def _content_hash(self, script: ComputeScript) -> str:
        """
        Create a deterministic hash of script content for signing.
        
        The hash includes:
        - Script name
        - Language
        - Content
        - Purpose
        
        This ensures any modification invalidates the signature.
        
        Args:
            script: The script to hash
            
        Returns:
            Hex-encoded SHA-256 hash
        """
        # Canonical representation for hashing
        canonical = f"{script.name}|{script.language}|{script.content}|{script.purpose}"
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    async def sign(self, script: ComputeScript) -> str:
        """
        Sign a script's content.

        Sign-or-fail. If the agent's keys cannot be loaded, raises
        ``ScriptSigningKeysUnavailable``. There is no fallback.

        Wire format:

        - **Hybrid agent** (post-rotation ceremony, ``self._agent_identity.is_hybrid``):
          returns ``hybrid:<base64>`` where the base64 wraps a JSON list of
          ``{alg, kid, sig}`` objects (Ed25519 + ML-DSA-65). Verifiers that
          understand the new format check both signatures; older verifiers
          will see a non-``ecdsa:`` prefix and reject — which is the correct
          behavior for a tamper-aware system.
        - **Legacy agent** (pre-ceremony or fallback): returns
          ``ecdsa:<base64>`` (single secp256k1 ECDSA signature, unchanged).

        Args:
            script: The script to sign

        Returns:
            Signature string with either ``hybrid:`` or ``ecdsa:`` prefix.

        Raises:
            ScriptSigningKeysUnavailable: when keys cannot be loaded or the
                signing operation fails.
        """
        import base64
        import json

        content_hash = self._content_hash(script)
        content_hash_bytes = hashlib.sha256(content_hash.encode()).digest()

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
                    content_hash_bytes,
                    identity.hybrid_keypair,
                    classical_kid=classical_kid,
                    pq_kid=pq_kid,
                )
                payload = base64.b64encode(json.dumps(sigs).encode()).decode()
                return f"hybrid:{payload}"
            except Exception as e:
                raise ScriptSigningKeysUnavailable(
                    f"Hybrid signing failed for script {script.id[:8]}…: {e}"
                ) from e

        # Legacy path: route through Secp256k1Suite. Byte-identical to
        # the previous direct ec.ECDSA(SHA256) call; pinned by the
        # CryptoSuite behavior-preservation pair test.
        from kestrel_sovereign.security.crypto_suite import (
            ALG_ECDSA_SECP256K1_SHA256, CryptoSuiteError, get_suite,
        )
        suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
        try:
            signature_bytes = suite.sign(content_hash_bytes, self._private_key)
            return "ecdsa:" + base64.b64encode(signature_bytes).decode()
        except (CryptoSuiteError, Exception) as e:
            raise ScriptSigningKeysUnavailable(
                f"ECDSA signing failed for script {script.id[:8]}…: {e}"
            ) from e
    
    async def verify(self, script: ComputeScript) -> bool:
        """
        Verify a script's signature.

        Accepts either:

        - ``hybrid:<base64-of-json-array>`` — the post-ceremony format.
          The base64 wraps a JSON list of ``{alg, kid, sig}`` objects.
          Returns True if at least one signature in the list verifies
          against the agent's loaded hybrid identity. (We don't require
          BOTH halves to verify here — script signing is "did this agent
          produce this byte sequence?" and either half answering yes is
          a definitive yes. The chain walker enforces hybrid policy on
          the artifacts where it matters — live identity assertion.)

        - ``ecdsa:<base64>`` — the legacy single-signature format.
          Continues to verify against the agent's legacy ECDSA key, so
          scripts signed before the rotation ceremony stay valid.

        Rejects:

        - ``hmac:`` — the removed fallback that used the public DID
          as the HMAC key (forgeable by any reader).
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

        import base64
        content_hash = self._content_hash(script)
        content_hash_bytes = hashlib.sha256(content_hash.encode()).digest()

        if not await self._load_keys():
            logger.warning("Cannot verify signature: failed to load keys")
            return False

        # Hybrid format
        if script.signature.startswith("hybrid:"):
            return self._verify_hybrid(script, content_hash_bytes)

        # Legacy ECDSA format
        if script.signature.startswith("ecdsa:"):
            return self._verify_legacy_ecdsa(script, content_hash_bytes)

        logger.warning(f"Unknown signature format: {script.signature[:10]}…")
        return False

    def _verify_legacy_ecdsa(
        self, script: ComputeScript, content_hash_bytes: bytes,
    ) -> bool:
        """Verify the ``ecdsa:<base64>`` legacy format against the
        agent's secp256k1 public key."""
        import base64
        if self._public_key is None:
            logger.warning("Cannot verify ECDSA signature without public key")
            return False
        from kestrel_sovereign.security.crypto_suite import (
            ALG_ECDSA_SECP256K1_SHA256, get_suite,
        )
        try:
            signature_bytes = base64.b64decode(script.signature[len("ecdsa:"):])
            suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
            if suite.verify(content_hash_bytes, signature_bytes, self._public_key):
                return True
            logger.warning("ECDSA signature verification failed")
            return False
        except Exception as e:
            logger.warning(f"ECDSA signature verification failed: {e}")
            return False

    def _verify_hybrid(
        self, script: ComputeScript, content_hash_bytes: bytes,
    ) -> bool:
        """Verify the ``hybrid:<base64-json>`` format against the
        agent's loaded hybrid keypair. At least one signature in the
        embedded array must verify."""
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
            payload_b64 = script.signature[len("hybrid:"):]
            sigs_json = base64.b64decode(payload_b64).decode("utf-8")
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
            try:
                alg = entry["alg"]
                kid = entry["kid"]
                sig_hex = entry["sig"]
            except (TypeError, KeyError):
                logger.warning("Hybrid signature entry malformed; rejecting")
                return False
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
            if not suite.verify(content_hash_bytes, sig_bytes, kp.public_key):
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
        
        Args:
            script: The script to sign
            
        Returns:
            The updated script with signature fields set
        """
        script.signature = await self.sign(script)
        script.signed_by = self.agent_did
        script.signed_at = datetime.now()
        return script
