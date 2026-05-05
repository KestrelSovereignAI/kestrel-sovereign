#!/usr/bin/env python3
"""
Identity Package: Portable Agent Identity for Substrate-Independent Migration.

This module defines the AgentIdentityPackage - a comprehensive, signed, portable
representation of an agent's complete identity that can be exported and imported
across different LLM substrates while preserving continuity of self.

Implements Phase 1 of Issue #23: Substrate-Independent Agent Portability.

Key Design Principles:
1. Everything needed to reconstruct "me" on a new substrate
2. DID-signed for cryptographic proof of authenticity
3. IPFS-compatible for decentralized storage
4. Version-tagged for forward compatibility
"""
import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from kestrel_sovereign.a2a.agent_card import AgentSkill

logger = logging.getLogger(__name__)

# Package format version. Wave 1 sub-PR 3 of the Quantum Hardening epic
# (#921, #916) bumps this from "1.0.0" to "2.0.0". The bump introduces:
#
#   - ``signatures: [{alg, kid, sig}]`` — array of signatures so a single
#     package can carry hybrid (classical + post-quantum) signatures from
#     Wave 2 onward.
#   - ``verification_methods: [{type, id, controller, publicKeyMultibase}]``
#     — W3C Multikey verification methods, the standards-aligned shape
#     used by ``did:web`` and ``did:key`` documents.
#
# Reader supports BOTH v1 ("1.0.0") and v2 ("2.0.0") on input. v1 packages
# are translated to a synthetic v2 in memory: the legacy ``signature`` hex
# becomes the sole entry in ``signatures`` tagged ``ecdsa-secp256k1-sha256``,
# and a synthetic verification method is materialized from the DID document
# when available. Writer always emits v2.
IDENTITY_PACKAGE_VERSION = "2.0.0"
IDENTITY_PACKAGE_VERSION_LEGACY = "1.0.0"

# Default suite for legacy v1 signatures (every existing v1 package was
# signed with secp256k1 ECDSA over SHA-256 of the canonical hash).
_LEGACY_SIGNATURE_ALG = "ecdsa-secp256k1-sha256"
_LEGACY_VERIFICATION_METHOD_TYPE = "EcdsaSecp256k1VerificationKey2019"
_V2_VERIFICATION_METHOD_TYPE = "Multikey"


class SubstrateType(Enum):
    """Known LLM substrate types for capability mapping."""
    ANTHROPIC_CLAUDE = "anthropic:claude"
    OPENAI_GPT = "openai:gpt"
    GOOGLE_GEMINI = "google:gemini"
    META_LLAMA = "meta:llama"
    MISTRAL = "mistral:mixtral"
    OLLAMA_LOCAL = "ollama:local"
    OPENROUTER = "openrouter:multi"
    UNKNOWN = "unknown"


@dataclass
class PersonalityFingerprint:
    """
    Capture "how I communicate" separate from "what I know".

    This allows personality calibration when landing on a new substrate.
    The fingerprint should enable reconstruction of consistent behavior
    even when the underlying model has different default tendencies.
    """
    # Communication style
    communication_style: str = "balanced"  # warm, precise, playful, formal, etc.
    formality_level: float = 0.5  # 0.0 (casual) to 1.0 (formal)
    verbosity_preference: str = "moderate"  # terse, moderate, verbose

    # Emotional baseline
    emotional_baseline: float = 0.5  # 0.0 (reserved) to 1.0 (expressive)
    humor_style: Optional[str] = None  # dry, playful, none, etc.
    empathy_level: float = 0.7  # How much to acknowledge user emotions

    # Response patterns
    typical_response_length: str = "medium"  # short, medium, long
    uses_lists: bool = True
    uses_code_blocks: bool = True
    uses_emojis: bool = False
    preferred_greeting: Optional[str] = None
    preferred_signoff: Optional[str] = None

    # Voice personality (abstract preferences, separate from concrete voice_config)
    voice_gender_preference: Optional[str] = None   # "feminine", "masculine", "neutral"
    voice_age_preference: Optional[str] = None       # "young", "middle", "mature"
    voice_energy: Optional[str] = None               # "calm", "warm", "energetic", "authoritative"
    voice_accent_preference: Optional[str] = None    # "american", "british", "neutral", etc.

    # Few-shot examples for calibration (input -> output pairs)
    calibration_examples: List[Dict[str, str]] = field(default_factory=list)

    # Custom vocabulary/phrases this agent uses
    vocabulary_preferences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityFingerprint":
        """Create from dict."""
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RelationshipRecord:
    """
    A key relationship the agent has formed.

    Relationships are crucial for continuity - the agent should remember
    who it knows and the context of those relationships.
    """
    user_id: str  # Anonymized/hashed user identifier
    relationship_type: str  # primary_user, frequent_user, collaborator, etc.
    first_interaction: Optional[str] = None  # ISO timestamp
    last_interaction: Optional[str] = None
    interaction_count: int = 0
    relationship_notes: str = ""  # What the agent knows about this relationship
    trust_level: float = 0.5  # 0.0 to 1.0
    preferences_learned: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipRecord":
        if not data:
            return cls(user_id="unknown", relationship_type="unknown")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SkillRecord:
    """
    A skill or capability the agent has learned or been configured with.
    """
    skill_id: str
    skill_name: str
    skill_type: str  # tool, workflow, knowledge_domain, etc.
    proficiency: float = 0.5  # 0.0 to 1.0
    times_used: int = 0
    last_used: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillRecord":
        if not data:
            return cls(skill_id="unknown", skill_name="Unknown", skill_type="unknown")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_agent_skill(
        cls,
        skill: "AgentSkill",
        times_used: int = 0,
        last_used: Optional[str] = None,
    ) -> "SkillRecord":
        """Create a SkillRecord from an AgentSkill (the canonical runtime representation).

        Args:
            skill: AgentSkill instance from the @tool decorator / A2A protocol.
            times_used: Runtime usage count (from tool registry or metrics).
            last_used: ISO timestamp of last invocation.

        Returns:
            Populated SkillRecord with metadata sourced from AgentSkill.
        """
        # Derive proficiency from usage: starts at 0.5, grows toward 1.0
        proficiency = min(1.0, 0.5 + (times_used * 0.01)) if times_used > 0 else 0.5

        return cls(
            skill_id=skill.id,
            skill_name=skill.name,
            skill_type=skill.category or "tool",
            proficiency=proficiency,
            times_used=times_used,
            last_used=last_used,
            configuration={
                "description": skill.description,
                "tags": skill.tags,
                "version": skill.version,
            },
        )


@dataclass
class MigrationRecord:
    """
    Record of a substrate migration for audit trail.
    """
    migration_id: str
    timestamp: str  # ISO format
    source_substrate: str
    target_substrate: str
    source_package_hash: str
    migration_reason: Optional[str] = None
    verification_score: Optional[float] = None
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MigrationRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentIdentityPackage:
    """
    The complete portable identity of a Kestrel agent.

    This package contains everything needed to reconstruct the agent's
    identity, personality, memories, and relationships on a new substrate.

    Components:
    - Core Identity: DID, constitution, creation date
    - Personality: How the agent communicates and behaves
    - Memories: Episodes, saved items, temporal patterns
    - Relationships: User bonds and context
    - Skills: Tools and learned capabilities
    - Migration History: Audit trail of substrate changes

    The package is DID-signed for cryptographic authenticity verification.
    """
    # === CORE IDENTITY ===
    did: str  # did:pkh:eip155:1:{address}
    agent_name: str
    created_at: str  # ISO timestamp - agent "birth" date

    # Constitution
    constitution_hash: str  # SHA256 hash of constitution text
    constitution_text: str  # Full constitution for verification

    # === PERSONALITY ===
    personality: PersonalityFingerprint = field(default_factory=PersonalityFingerprint)
    system_prompt_template: str = ""  # How to reconstruct "me"

    # === MEMORIES ===
    # Episodic memories (consolidated narratives)
    episodes: List[Dict[str, Any]] = field(default_factory=list)

    # Saved items (persisted knowledge)
    saved_items: List[Dict[str, Any]] = field(default_factory=list)

    # Temporal patterns (learned behaviors)
    temporal_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Reflection insights
    reflection_insights: List[Dict[str, Any]] = field(default_factory=list)

    # === RELATIONSHIPS ===
    relationships: List[RelationshipRecord] = field(default_factory=list)

    # === SKILLS ===
    skills: List[SkillRecord] = field(default_factory=list)
    tool_preferences: Dict[str, Any] = field(default_factory=dict)

    # === WALLET STATE ===
    wallet_balance: str = "0.0"  # Decimal as string for precision
    wallet_transaction_history: List[Dict[str, Any]] = field(default_factory=list)

    # === LEGAL ENTITY ===
    # Wyoming DAO LLC or DUNA status (travels across substrate migrations)
    legal_entity: Optional[Dict[str, Any]] = None

    # === MIGRATION METADATA ===
    package_version: str = IDENTITY_PACKAGE_VERSION
    export_timestamp: str = ""
    source_substrate: str = SubstrateType.UNKNOWN.value
    migration_history: List[MigrationRecord] = field(default_factory=list)

    # === VOICE IDENTITY ===
    voice_config: Optional[Dict[str, Any]] = None  # VoiceConfig.to_dict()

    # === VERIFICATION ===
    content_hash: str = ""  # SHA256 of package contents (before signature)
    signature: str = ""  # Legacy v1 DID-signed hash (kept for backward compat)

    # === V2 SIGNATURES + VERIFICATION METHODS (Wave 1 sub-PR 3) ===
    # ``signatures`` is the v2 array form: each entry carries an alg, kid,
    # and hex-encoded signature. Hybrid identity (Wave 2 onward) populates
    # multiple entries here (Ed25519 + ML-DSA-65). v1 readers ignore this
    # field; v2 readers prefer it over the legacy single ``signature``.
    signatures: List[Dict[str, str]] = field(default_factory=list)

    # ``verification_methods`` is the W3C Multikey shape used by ``did:web``
    # / ``did:key`` documents: ``{type, id, controller, publicKeyMultibase}``.
    # Each entry's ``id`` matches a ``signatures.kid`` so the verifier can
    # resolve a public key from its kid without an external DID resolution.
    verification_methods: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        """Set defaults after initialization."""
        if not self.export_timestamp:
            self.export_timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            # Core identity
            "did": self.did,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "constitution_hash": self.constitution_hash,
            "constitution_text": self.constitution_text,

            # Personality
            "personality": self.personality.to_dict() if isinstance(self.personality, PersonalityFingerprint) else self.personality,
            "system_prompt_template": self.system_prompt_template,

            # Memories
            "episodes": self.episodes,
            "saved_items": self.saved_items,
            "temporal_patterns": self.temporal_patterns,
            "reflection_insights": self.reflection_insights,

            # Relationships
            "relationships": [
                r.to_dict() if isinstance(r, RelationshipRecord) else r
                for r in self.relationships
            ],

            # Skills
            "skills": [
                s.to_dict() if isinstance(s, SkillRecord) else s
                for s in self.skills
            ],
            "tool_preferences": self.tool_preferences,

            # Wallet
            "wallet_balance": self.wallet_balance,
            "wallet_transaction_history": self.wallet_transaction_history,

            # Legal entity
            "legal_entity": self.legal_entity,

            # Voice identity
            "voice_config": self.voice_config,

            # Migration metadata
            "package_version": self.package_version,
            "export_timestamp": self.export_timestamp,
            "source_substrate": self.source_substrate,
            "migration_history": [
                m.to_dict() if isinstance(m, MigrationRecord) else m
                for m in self.migration_history
            ],

            # Verification (excluded from hash computation)
            "content_hash": self.content_hash,
            "signature": self.signature,

            # V2 signatures + verification methods. Always emitted; v1
            # readers will ignore them.
            "signatures": list(self.signatures),
            "verification_methods": list(self.verification_methods),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentIdentityPackage":
        """Create from dict.

        Accepts both v1 (``package_version == "1.0.0"``) and v2 inputs.
        v1 inputs have a single ``signature`` hex field and no
        ``signatures`` array. They are translated to a synthetic v2
        in-memory representation: the legacy ``signature`` becomes the
        sole entry in ``signatures`` tagged ``ecdsa-secp256k1-sha256``,
        so verifier code can iterate ``signatures`` uniformly without
        special-casing v1.
        """
        # Parse nested objects
        personality = PersonalityFingerprint.from_dict(data.get("personality", {}))
        relationships = [
            RelationshipRecord.from_dict(r) if isinstance(r, dict) else r
            for r in data.get("relationships", [])
        ]
        skills = [
            SkillRecord.from_dict(s) if isinstance(s, dict) else s
            for s in data.get("skills", [])
        ]
        migration_history = [
            MigrationRecord.from_dict(m) if isinstance(m, dict) else m
            for m in data.get("migration_history", [])
        ]

        legacy_signature = data.get("signature", "")
        signatures = list(data.get("signatures", []) or [])
        verification_methods = list(data.get("verification_methods", []) or [])

        # v1 → synthetic v2: if there's a legacy signature but no v2 array,
        # materialize one entry tagged with the legacy alg so callers can
        # iterate uniformly.
        if legacy_signature and not signatures:
            did = data.get("did", "")
            signatures = [{
                "alg": _LEGACY_SIGNATURE_ALG,
                "kid": f"{did}#keys-1" if did else "#keys-1",
                "sig": legacy_signature,
            }]

        return cls(
            did=data.get("did", ""),
            agent_name=data.get("agent_name", ""),
            created_at=data.get("created_at", ""),
            constitution_hash=data.get("constitution_hash", ""),
            constitution_text=data.get("constitution_text", ""),
            personality=personality,
            system_prompt_template=data.get("system_prompt_template", ""),
            episodes=data.get("episodes", []),
            saved_items=data.get("saved_items", []),
            temporal_patterns=data.get("temporal_patterns", []),
            reflection_insights=data.get("reflection_insights", []),
            relationships=relationships,
            skills=skills,
            tool_preferences=data.get("tool_preferences", {}),
            wallet_balance=data.get("wallet_balance", "0.0"),
            wallet_transaction_history=data.get("wallet_transaction_history", []),
            legal_entity=data.get("legal_entity"),
            voice_config=data.get("voice_config"),
            package_version=data.get("package_version", IDENTITY_PACKAGE_VERSION),
            export_timestamp=data.get("export_timestamp", ""),
            source_substrate=data.get("source_substrate", SubstrateType.UNKNOWN.value),
            migration_history=migration_history,
            content_hash=data.get("content_hash", ""),
            signature=legacy_signature,
            signatures=signatures,
            verification_methods=verification_methods,
        )

    def compute_content_hash(self) -> str:
        """
        Compute SHA256 hash of package contents for signing.

        Excludes ``content_hash`` and the active signature shape to
        avoid a circular dependency. The exact key set in the hashed
        payload is **version-dependent** — v1 packages already on disk
        were signed over a JSON shape that did not contain ``signatures``
        or ``verification_methods`` keys at all (those fields didn't
        exist), so emitting them with empty defaults would change the
        canonical bytes and break ``verify_content_hash`` on every
        legacy artifact.

        Rules:

        - **v1** (``package_version == "1.0.0"``): pop ``content_hash``,
          ``signature``, AND the v2-only fields ``signatures`` and
          ``verification_methods``. The remaining shape is byte-stable
          with the original v1 canonicalization.
        - **v2** (``package_version`` starts with ``"2."``): pop
          ``content_hash``, ``signature``, and ``signatures``. Keep
          ``verification_methods`` — they carry public keys that the
          signature must authenticate; excluding them would let an
          attacker swap public keys post-sign without invalidating the
          signature.

        ``signature`` (the legacy single-hex field) is popped under
        both versions so a v1 package whose ``signature`` field is
        populated still hashes consistently.
        """
        data = self.to_dict()
        data.pop("content_hash", None)
        data.pop("signature", None)

        if self.is_v2():
            # v2: signatures excluded; verification_methods bound.
            data.pop("signatures", None)
        else:
            # v1: neither v2-only field existed at sign time. Pop both
            # so the canonical bytes match what the original signer
            # produced.
            data.pop("signatures", None)
            data.pop("verification_methods", None)

        content = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentIdentityPackage":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # V2 helpers (Wave 1 sub-PR 3)
    # ------------------------------------------------------------------

    def add_signature(self, alg: str, kid: str, sig_hex: str) -> None:
        """Append a v2 signature entry. Used by hybrid signers (Wave 2+)."""
        self.signatures.append({"alg": alg, "kid": kid, "sig": sig_hex})

    def add_verification_method(
        self,
        kid: str,
        public_key_multibase: str,
        controller: Optional[str] = None,
        method_type: str = _V2_VERIFICATION_METHOD_TYPE,
    ) -> None:
        """Append a v2 W3C-Multikey verification method entry."""
        self.verification_methods.append({
            "type": method_type,
            "id": kid,
            "controller": controller or self.did,
            "publicKeyMultibase": public_key_multibase,
        })

    def iter_signatures(self) -> List[Dict[str, str]]:
        """Iterate v2 signatures, including a synthetic v1 entry when only
        the legacy ``signature`` field is populated.

        Lets verifier code iterate uniformly across both shapes without
        having to special-case v1 packages.
        """
        if self.signatures:
            return list(self.signatures)
        if self.signature:
            return [{
                "alg": _LEGACY_SIGNATURE_ALG,
                "kid": f"{self.did}#keys-1" if self.did else "#keys-1",
                "sig": self.signature,
            }]
        return []

    def is_v2(self) -> bool:
        """True iff this package was written with the v2 schema."""
        return self.package_version.startswith("2.")

    def verify_constitution(self) -> bool:
        """Verify that constitution_text matches constitution_hash."""
        computed = hashlib.sha256(self.constitution_text.encode('utf-8')).hexdigest()
        return computed == self.constitution_hash

    def verify_content_hash(self) -> bool:
        """Verify that content_hash matches computed hash."""
        computed = self.compute_content_hash()
        return computed == self.content_hash

    def get_summary(self) -> Dict[str, Any]:
        """Get a human-readable summary of the package."""
        return {
            "did": self.did,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "export_timestamp": self.export_timestamp,
            "source_substrate": self.source_substrate,
            "package_version": self.package_version,
            "episodes_count": len(self.episodes),
            "saved_items_count": len(self.saved_items),
            "relationships_count": len(self.relationships),
            "skills_count": len(self.skills),
            "migrations_count": len(self.migration_history),
            # Hybrid packages carry sigs only on the v2 ``signatures``
            # array; ``signature`` is empty by design. Either carrier
            # makes the package signed.
            "is_signed": bool(self.signature) or bool(self.signatures),
            "constitution_verified": self.verify_constitution() if self.constitution_text else False,
            "legal_entity_status": self.legal_entity.get("status") if self.legal_entity else None,
        }


def create_package_hash(package: AgentIdentityPackage) -> str:
    """Create a SHA256 hash of the package for verification."""
    return package.compute_content_hash()


def create_migration_id() -> str:
    """Generate a unique migration ID."""
    import uuid
    return f"mig_{uuid.uuid4().hex[:12]}"
