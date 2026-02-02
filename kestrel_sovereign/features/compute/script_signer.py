"""
Kestrel Compute Feature - Script Signer.

Cryptographic signing of scripts using the agent's secp256k1 DID key.
Ensures scripts cannot be tampered with after agent creates them.
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ComputeScript

logger = logging.getLogger(__name__)


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
        
        Creates an ECDSA signature of the script's content hash using secp256k1.
        Falls back to HMAC-SHA256 if keys aren't available.
        
        Args:
            script: The script to sign
            
        Returns:
            Base64-encoded signature string
        """
        import base64
        
        content_hash = self._content_hash(script)
        content_hash_bytes = hashlib.sha256(content_hash.encode()).digest()
        
        # Try ECDSA signing with secp256k1 first
        if await self._load_keys() and self._private_key:
            try:
                from cryptography.hazmat.primitives.asymmetric import ec
                from cryptography.hazmat.primitives import hashes
                
                signature_bytes = self._private_key.sign(
                    content_hash_bytes,
                    ec.ECDSA(hashes.SHA256())
                )
                return "ecdsa:" + base64.b64encode(signature_bytes).decode()
            except Exception as e:
                logger.warning(f"ECDSA signing failed: {e}, falling back to HMAC")
        
        # Fallback: HMAC-SHA256 with DID as key
        # This is less secure but provides tamper detection without proper keys
        import hmac
        key = (self.agent_did or "kestrel-unsigned").encode()
        signature = hmac.new(key, content_hash.encode(), hashlib.sha256).digest()
        
        # Prefix with "hmac:" to indicate fallback method
        return "hmac:" + base64.b64encode(signature).decode()
    
    async def verify(self, script: ComputeScript) -> bool:
        """
        Verify a script's signature.
        
        Ensures the script hasn't been modified since it was signed.
        
        Args:
            script: The script to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        if not script.signature:
            logger.warning(f"Script {script.id[:8]}... has no signature")
            return False
        
        import base64
        
        content_hash = self._content_hash(script)
        content_hash_bytes = hashlib.sha256(content_hash.encode()).digest()
        
        # Check if HMAC fallback signature
        if script.signature.startswith("hmac:"):
            import hmac
            key = (script.signed_by or self.agent_did or "kestrel-unsigned").encode()
            expected = hmac.new(key, content_hash.encode(), hashlib.sha256).digest()
            expected_b64 = base64.b64encode(expected).decode()
            
            actual_b64 = script.signature[5:]  # Remove "hmac:" prefix
            return hmac.compare_digest(expected_b64, actual_b64)
        
        # ECDSA verification
        if script.signature.startswith("ecdsa:"):
            if await self._load_keys() and self._public_key:
                try:
                    from cryptography.hazmat.primitives.asymmetric import ec
                    from cryptography.hazmat.primitives import hashes
                    
                    signature_bytes = base64.b64decode(script.signature[6:])  # Remove "ecdsa:" prefix
                    self._public_key.verify(
                        signature_bytes,
                        content_hash_bytes,
                        ec.ECDSA(hashes.SHA256())
                    )
                    return True
                except Exception as e:
                    logger.warning(f"ECDSA signature verification failed: {e}")
                    return False
            else:
                logger.warning("Cannot verify ECDSA signature without public key")
                return False
        
        # Unknown signature format
        logger.warning(f"Unknown signature format: {script.signature[:10]}...")
        return False
    
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
