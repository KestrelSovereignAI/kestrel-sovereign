#!/usr/bin/env python3
"""
Identity Feature: Agent tools for identity portability.

This feature provides the agent with tools to export, import, and verify
its identity across different LLM substrates.

Implements the tools defined in Issue #23:
- export_identity: Create signed identity package
- import_identity: Load identity on new substrate
- verify_identity: Run continuity tests
- assess_substrate: Check current substrate capabilities
- migration_history: Show migration audit trail
"""
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.enum_coerce import normalize_choice as _normalize_choice
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.identity.package_intake import (
    IdentityPackageIntakeError,
    load_identity_package_source,
)
from kestrel_sovereign.identity.protected_export import (
    identity_export_directory,
    write_protected_identity_export,
)
from kestrel_sovereign.identity.sealed_export import SealedExportError
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)

_HOLD_REDACTION_POLICY = {
    "subject_identity": "implicit_self",
    "target_identity": "omitted",
    "receipt_identity": "omitted",
    "actor_identity": "role_only",
    "reason": "visible_to_held_subject",
    "timestamp": "visible_to_held_subject",
}


def _hold_actor_role(actor_id: str, subject_did: str) -> str:
    """Classify a Hold actor without disclosing its sovereign identity."""

    if actor_id == subject_did:
        return "self"
    if actor_id.startswith("did:sovereign:"):
        return "sovereign"
    if actor_id.startswith("did:"):
        return "agent"
    return "operator"


def _self_hold_latch_view(state: Any, *, subject_did: str) -> Dict[str, Any]:
    """Render one verified latch under the self-introspection redaction policy."""

    return {
        "scope": state.scope.value,
        "reason": state.reason,
        "actor_role": _hold_actor_role(state.actor_id, subject_did),
        "set_at": state.set_at,
        "revision": state.revision,
    }

# Synonyms LLMs reach for on import_identity's merge_mode.
_MERGE_MODE_ALIASES = {
    "overwrite": "replace", "replace_all": "replace", "reset": "replace",
    "combine": "merge", "union": "merge", "update": "merge",
    "skip": "skip_existing", "skip-existing": "skip_existing",
    "keep": "skip_existing", "keep_existing": "skip_existing",
}


def _unique_export_filename() -> str:
    """Generate a collision-resistant filename for an identity export.

    Round 4 codex finding: ``strftime('%Y%m%d_%H%M%S')`` granularity
    is per-second, so two exports within the same second overwrite
    each other. Append a microsecond stamp + 8-char uuid hex to
    make collisions astronomically unlikely.
    """
    now = datetime.now(timezone.utc)
    return (
        f"identity_{now.strftime('%Y%m%d_%H%M%S')}_"
        f"{now.microsecond:06d}_{uuid.uuid4().hex[:8]}.json"
    )


def _runtime_agent_data_dir(agent: Any) -> Optional[Path]:
    """Return the real per-agent key/data directory when one is configured.

    ``vars`` deliberately avoids synthesizing a fake ``storage_path`` on
    MagicMock-backed test agents. Live multi-agent instances expose the DB path
    explicitly, and their identity keys live alongside it.
    """
    storage_path = vars(agent).get("storage_path")
    if isinstance(storage_path, (str, os.PathLike)):
        return Path(storage_path).parent
    return None


def _runtime_identity_export_dir(agent: Any) -> Path:
    """Resolve the export root from this agent's own runtime binding."""

    per_agent_override = vars(agent).get("identity_export_dir")
    if not isinstance(per_agent_override, (str, os.PathLike)):
        per_agent_override = None
    return identity_export_directory(
        agent_data_dir=_runtime_agent_data_dir(agent),
        per_agent_override=per_agent_override,
    )


def _parse_identity_trust_policy(raw: Optional[Dict[str, Any]]):
    """Validate a tool-supplied receiver trust policy.

    The package never supplies this object.  It must come from the receiving
    operator (typically a pinned root registry and revocation feed).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("identity_trust_policy must be an object")
    allowed = {
        "trusted_root_did",
        "trusted_root_verification_methods",
        "revoked_succession_ids",
        "require_archival",
        "trusted_archival_multibase",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"identity_trust_policy has unknown fields: {sorted(unknown)}"
        )
    root_did = raw.get("trusted_root_did")
    if root_did is not None and not isinstance(root_did, str):
        raise ValueError("identity_trust_policy.trusted_root_did must be a string")
    root_vms = raw.get("trusted_root_verification_methods", [])
    if not isinstance(root_vms, list) or any(not isinstance(vm, dict) for vm in root_vms):
        raise ValueError(
            "identity_trust_policy.trusted_root_verification_methods "
            "must be an array of objects"
        )
    revoked = raw.get("revoked_succession_ids", [])
    if not isinstance(revoked, list) or any(not isinstance(item, str) for item in revoked):
        raise ValueError(
            "identity_trust_policy.revoked_succession_ids must be an array of strings"
        )
    require_archival = raw.get("require_archival", False)
    if not isinstance(require_archival, bool):
        raise ValueError("identity_trust_policy.require_archival must be a boolean")
    archival_pin = raw.get("trusted_archival_multibase")
    if archival_pin is not None and not isinstance(archival_pin, str):
        raise ValueError(
            "identity_trust_policy.trusted_archival_multibase must be a string"
        )
    from kestrel_sovereign.identity.portable_trust import IdentityTrustPolicy
    return IdentityTrustPolicy.create(
        trusted_root_did=root_did,
        trusted_root_verification_methods=root_vms,
        revoked_succession_ids=revoked,
        require_archival=require_archival,
        trusted_archival_multibase=archival_pin,
    )


class IdentityFeature(Feature):
    """
    Feature for managing agent identity portability.

    Provides tools for exporting and importing agent identity across
    different LLM substrates while preserving continuity of self.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage agent identity portability - export identity to portable package, "
            "import identity from package, verify integrity, assess substrate capabilities, "
            "and view migration history"
        )

    async def initialize(self):
        """Initialize the identity feature."""
        logger.info("Initializing IdentityFeature")

    @tool(
        name="inspect_hold_state",
        description=(
            "Inspect the durable host and agent Hold latches that apply to this "
            "agent. The trusted runtime fixes the subject to this agent's DID; "
            "there is no target parameter and this read-only tool cannot release "
            "a Hold. Reasons and times are visible, while raw actor, target, and "
            "receipt identities are redacted."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!identity hold",
    )
    async def inspect_hold_state(self) -> ToolResult:
        """Return independently visible host and agent latches for self only."""

        subject_did = vars(self.agent).get("_self_hold_subject_did")
        reader = vars(self.agent).get("_self_hold_state_reader")
        unknown = {
            "state": "unknown",
            "held": None,
            "sources": [],
            "latches": {"host": None, "agent": None},
            "redaction_policy": dict(_HOLD_REDACTION_POLICY),
        }
        if not isinstance(subject_did, str) or not subject_did or not callable(reader):
            return ToolResult.failed(
                "Hold state is unavailable because the host control reader is not bound.",
                data=unknown,
            )

        try:
            from kestrel_sovereign.hold import (
                EffectiveHoldState,
                HOST_HOLD_TARGET,
                HoldScope,
            )

            effective = await reader()
            if not isinstance(effective, EffectiveHoldState):
                raise TypeError("host reader returned an invalid Hold snapshot")
            if effective.host is not None and (
                effective.host.scope is not HoldScope.HOST
                or effective.host.target_id != HOST_HOLD_TARGET
            ):
                raise ValueError("host reader returned a foreign host latch")
            if effective.agent is not None and (
                effective.agent.scope is not HoldScope.AGENT
                or effective.agent.target_id != subject_did
            ):
                raise ValueError("host reader returned a foreign agent latch")
        except Exception as exc:  # noqa: BLE001 - read failure is typed below
            logger.error(
                "Self Hold introspection failed (cause_type=%s)",
                type(exc).__name__,
            )
            unknown["failure"] = "read_failed"
            unknown["cause_type"] = type(exc).__name__
            return ToolResult.failed(
                "Hold state could not be read; current Hold status is unknown.",
                data=unknown,
            )

        host_view = (
            _self_hold_latch_view(effective.host, subject_did=subject_did)
            if effective.host is not None
            else None
        )
        agent_view = (
            _self_hold_latch_view(effective.agent, subject_did=subject_did)
            if effective.agent is not None
            else None
        )
        sources = [source.value for source in effective.sources]
        data = {
            "state": "held" if effective.held else "not_held",
            "held": effective.held,
            "sources": sources,
            "latches": {"host": host_view, "agent": agent_view},
            "redaction_policy": dict(_HOLD_REDACTION_POLICY),
        }
        if effective.held:
            return ToolResult.ok(
                confirmation=(
                    "This agent is held by: " + ", ".join(sources) + "."
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation="No host or agent Hold currently applies to this agent.",
            data=data,
        )

    def _load_import_package(self, package_json: str):
        """Parse a loaded export into an AgentIdentityPackage.

        ALWAYS routes through ``open_identity_export`` — it parses a
        plaintext package, unseals a hybrid-KEM capsule with THIS
        agent's local KEM keypair (recipient custody), AND fails closed
        on a tampered/format-stripped capsule. Calling ``from_json``
        directly for the "not exactly a capsule" case would skip that
        tamper check (codex #2398 round 10).
        """
        from kestrel_sovereign.identity.sealed_export import open_identity_export

        # Let open_identity_export discover the KEM slug from the local
        # key files (robust to multi-segment did:web and did:pkh agents)
        # — the DID tail is not a reliable slug.
        storage_dir = _runtime_agent_data_dir(self.agent)
        return open_identity_export(package_json, storage_dir=storage_dir)

    @tool(
        name="export_identity",
        description="Export the agent's complete identity to a portable, signed package. "
                    "This creates a JSON package containing DID, constitution, memories, "
                    "personality, relationships, and skills that can be imported to another substrate. "
                    "storage_tier must be one of 'local' (default), 'ipfs', or 'filecoin'; an "
                    "unrecognized value is rejected (it is NOT silently downgraded to local).",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity export"
    )
    async def export_identity(
        self,
        storage_tier: str = "local",
        sign: bool = True,
        include_wallet: bool = True,
    ) -> ToolResult:
        """
        Export agent identity to a portable package.

        Args:
            storage_tier: Where to store the package ('local', 'ipfs', 'filecoin')
            sign: Whether to sign the package with DID key
            include_wallet: Whether to include wallet transaction history
        """
        # Validate storage_tier explicitly. The old code resolved this via
        # ``tier_map.get(storage_tier.lower(), StorageTier.LOCAL_ONLY)``, so a
        # typo'd / unknown tier SILENTLY fell through to local-only — the
        # agent would believe it exported to ipfs/filecoin while the package
        # never left the host. Reject unknown / wrong-type tiers loudly. Only
        # an explicitly omitted value (None or an empty/whitespace string)
        # keeps the documented default ('local'); a falsy non-string such as
        # ``false``/``0``/``[]`` is a wrong type, NOT an omission, and must
        # be rejected rather than coerced to the default.
        valid_tiers = ("local", "ipfs", "filecoin")
        if storage_tier is None or (isinstance(storage_tier, str) and not storage_tier.strip()):
            tier_key = "local"
        elif isinstance(storage_tier, str):
            tier_key = storage_tier.strip().lower()
        else:
            tier_key = None  # wrong type → fall into the rejection below
        if tier_key not in valid_tiers:
            return ToolResult.failed(
                f"storage_tier must be one of {', '.join(valid_tiers)}; "
                f"got {storage_tier!r}. The tier was NOT silently defaulted "
                "to local — re-run with a valid tier."
            )

        if not isinstance(sign, bool):
            return ToolResult.failed(
                f"sign must be a boolean, got {type(sign).__name__}={sign!r}"
            )
        if not isinstance(include_wallet, bool):
            return ToolResult.failed(
                "include_wallet must be a boolean, got "
                f"{type(include_wallet).__name__}={include_wallet!r}"
            )

        try:
            from kestrel_sovereign.identity import (
                IdentityExporter,
                sign_package,
            )
            from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier

            db = resolve_feature_database(self.agent)
            if db is None:
                return ToolResult.failed("Export failed: database not available")

            agent_data_dir = _runtime_agent_data_dir(self.agent)

            # Export identity
            exporter = IdentityExporter(
                db=db,
                agent_id=self.agent.agent_id,
                agent_data_dir=agent_data_dir,
                agent=self.agent,
            )
            package = await exporter.export(include_wallet_history=include_wallet)

            # Sign if requested. The original code logged a warning
            # and silently proceeded with an unsigned package — that's
            # exactly the honesty leak the migration is meant to
            # surface. Track the failure so we can return PARTIAL.
            sign_failure: Optional[str] = None
            if sign:
                try:
                    package = sign_package(package, agent_data_dir)
                except Exception as e:
                    logger.warning(f"Could not sign package: {e}")
                    sign_failure = str(e)

            # Get JSON representation
            package_json = package.to_json()
            summary = package.get_summary()

            # Store based on tier
            tier_map = {
                "local": StorageTier.LOCAL_ONLY,
                "ipfs": StorageTier.IPFS,
                "filecoin": StorageTier.FILECOIN,
            }
            # tier_key was validated against tier_map's keys above.
            tier_enum = tier_map[tier_key]

            if tier_enum != StorageTier.LOCAL_ONLY:
                # F187: IPFS/Filecoin are PUBLIC content-addressed networks —
                # anyone with the CID can fetch the bytes. Publishing the
                # identity package (DID, memories, relationships, calibration
                # examples) in plaintext leaks user-derived content forever.
                # Require a data key and encrypt before upload; if no key is
                # configured, FAIL rather than silently upload plaintext.
                from kestrel_sovereign.security.encryption import (
                    get_master_key_bytes,
                )
                if not get_master_key_bytes():
                    return ToolResult.failed(
                        "Non-local identity export requires an encryption key. "
                        "Set KESTREL_DATA_KEY so the package is encrypted "
                        f"before upload to the public {tier_key} network — "
                        "the export was NOT uploaded in plaintext. The "
                        "per-content key hash returned on a successful "
                        "encrypted export must travel OUT-OF-BAND with the "
                        "CID for the package to be restorable."
                    )
                # Upload to IPFS/Filecoin, encrypted. The adapter derives a
                # per-content key from KESTREL_DATA_KEY and returns its
                # encryption_key_hash on the StorageResult; retrieval needs
                # that hash (passed to retrieve_content(key_hash=...)).
                adapter = FilecoinAdapter()
                result = await asyncio.to_thread(
                    adapter.store_content,
                    content=package_json.encode('utf-8'),
                    storage_tier=tier_enum,
                    encrypt=True,  # F187: never publish plaintext to a public CID
                    metadata={"type": "identity_package", "did": package.did}
                )
                encryption_key_hash = getattr(result, "encryption_key_hash", None)
                # Honesty: FilecoinAdapter silently downgrades to
                # LOCAL_ONLY when IPFS / Lotus isn't available — it
                # mutates result.tier and leaves ipfs_cid as None.
                # In that case the package is still on disk under the
                # content hash, but it's NOT importable via
                # `!identity import <ipfs-cid>` (no Qm/bafy prefix to
                # match). Pre-fix we returned OK with a non-CID hash
                # in the import instructions — wrong. Detect the
                # downgrade and surface as PARTIAL with the correct
                # local-only retrieval path. (Round 2 codex finding.)
                actual_tier = getattr(result, "tier", tier_enum)
                tier_downgraded = (
                    actual_tier == StorageTier.LOCAL_ONLY
                    and tier_enum != StorageTier.LOCAL_ONLY
                )
                ipfs_cid = getattr(result, "ipfs_cid", None) or getattr(result, "cid", None)
                content_hash = getattr(result, "content_hash", None)
                # Use a real CID for restore instructions only when
                # one was actually produced; otherwise fall back to
                # the content hash with a clear local-only framing.
                restore_id = ipfs_cid or content_hash

                # When the tier downgraded, also write the package JSON
                # to KESTREL_DATA_DIR so there's an actually-restorable
                # path. The FilecoinAdapter's local cache is a compressed
                # blob keyed by content_hash — import_identity can't
                # consume it. Mirroring the tier=local fallback gives
                # the user a real file path that `!identity import`
                # accepts. (Round 3 codex finding #2.)
                fallback_filepath: Optional[Path] = None
                if tier_downgraded:
                    try:
                        storage_dir = _runtime_identity_export_dir(self.agent)
                        filename = _unique_export_filename()
                        fallback_filepath = storage_dir / filename
                        fallback_filepath = write_protected_identity_export(
                            fallback_filepath,
                            package_json,
                            allowed_destination_roots=(storage_dir,),
                        )
                    except Exception as e:
                        # If even the fallback write fails, leave
                        # fallback_filepath=None — the PARTIAL
                        # message below will be even more pessimistic
                        # ("no restore path available").
                        logger.error(
                            f"Tier downgrade fallback file write failed: {e}",
                            exc_info=True,
                        )
                        fallback_filepath = None

                if tier_downgraded:
                    if fallback_filepath:
                        confirmation = (
                            f"Exported identity package: DID={summary['did'][:30]}..., "
                            f"agent={summary['agent_name']}, "
                            f"{summary['episodes_count']} episodes, "
                            f"{summary['saved_items_count']} items, "
                            f"signed={summary['is_signed']}; "
                            f"requested tier={tier_enum.value} unavailable, "
                            f"saved as JSON to {fallback_filepath}. "
                            f"Use `!identity import {fallback_filepath}` to restore."
                        )
                    else:
                        confirmation = (
                            f"Exported identity package: DID={summary['did'][:30]}..., "
                            f"agent={summary['agent_name']}, "
                            f"{summary['episodes_count']} episodes; "
                            f"requested tier={tier_enum.value} unavailable AND "
                            "fallback JSON write failed — package is in the "
                            f"FilecoinAdapter local cache (content_hash="
                            f"{content_hash}) but cannot be restored via "
                            "`!identity import` directly"
                        )
                else:
                    confirmation = (
                        f"Exported identity package: DID={summary['did'][:30]}..., "
                        f"agent={summary['agent_name']}, "
                        f"{summary['episodes_count']} episodes, "
                        f"{summary['saved_items_count']} items, "
                        f"signed={summary['is_signed']}; "
                        f"stored ENCRYPTED to tier="
                        f"{actual_tier.value if hasattr(actual_tier, 'value') else actual_tier} "
                        f"(CID={ipfs_cid}, key_hash={encryption_key_hash}). "
                        f"Use `!identity import {restore_id} true merge {encryption_key_hash}` "
                        "to restore (positional args: source verify_signature "
                        "merge_mode key_hash). The key_hash must travel "
                        "OUT-OF-BAND with the CID — the CID alone cannot decrypt "
                        "the package."
                    )
                data = {
                    "did": package.did,
                    "agent_name": summary['agent_name'],
                    "summary": dict(summary),
                    "requested_storage_tier": tier_enum.value,
                    "actual_storage_tier": (
                        actual_tier.value if hasattr(actual_tier, "value") else str(actual_tier)
                    ),
                    "tier_downgraded": tier_downgraded,
                    "ipfs_cid": ipfs_cid,
                    "content_hash": content_hash,
                    "encrypted": True,
                    "encryption_key_hash": encryption_key_hash,
                    "fallback_file_path": (
                        str(fallback_filepath) if fallback_filepath else None
                    ),
                    "signed": bool(summary.get("is_signed")),
                    "sign_requested": sign,
                    "sign_failure": sign_failure,
                }
            else:
                # Save to local file
                storage_dir = _runtime_identity_export_dir(self.agent)
                filename = _unique_export_filename()
                filepath = storage_dir / filename

                filepath = write_protected_identity_export(
                    filepath,
                    package_json,
                    allowed_destination_roots=(storage_dir,),
                )

                confirmation = (
                    f"Exported identity package: DID={summary['did'][:30]}..., "
                    f"agent={summary['agent_name']}, "
                    f"{summary['episodes_count']} episodes, "
                    f"{summary['saved_items_count']} items, "
                    f"signed={summary['is_signed']}; "
                    f"saved to {filepath}. "
                    f"Use `!identity import {filepath}` to restore."
                )
                data = {
                    "did": package.did,
                    "agent_name": summary['agent_name'],
                    "summary": dict(summary),
                    "storage_tier": "local",
                    "file_path": str(filepath),
                    "signed": bool(summary.get("is_signed")),
                    "sign_requested": sign,
                    "sign_failure": sign_failure,
                }
        except Exception as e:
            logger.error(f"Identity export failed: {e}", exc_info=True)
            return ToolResult.failed(f"Identity export failed: {str(e)}")

        # Honesty: composite PARTIAL surfaces.
        partial_errs: List[str] = []
        if sign and sign_failure:
            # signing was REQUESTED but FAILED. The package was
            # still written, but it's unsigned — anyone importing
            # it will see UNSIGNED and may reject it.
            partial_errs.append(
                f"signing was requested but failed: {sign_failure}; "
                "the package was written UNSIGNED — verify will report "
                "UNSIGNED and an importer with allow_unsigned=False "
                "will reject it"
            )
        if data.get("tier_downgraded"):
            fallback = data.get("fallback_file_path")
            if fallback:
                partial_errs.append(
                    f"requested storage tier "
                    f"{data['requested_storage_tier']!r} was unavailable "
                    "(no IPFS / Lotus); the package was saved as a JSON "
                    f"file to {fallback} instead — use "
                    f"`!identity import {fallback}` to restore"
                )
            else:
                partial_errs.append(
                    f"requested storage tier "
                    f"{data['requested_storage_tier']!r} was unavailable "
                    "AND the fallback JSON file write failed — the "
                    "package is only in the FilecoinAdapter local cache "
                    f"(content_hash={data.get('content_hash')}) which "
                    "cannot be restored via `!identity import` directly"
                )
        if partial_errs:
            return ToolResult.partial(
                confirmation=confirmation,
                error=" | ".join(partial_errs),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

    @tool(
        name="import_identity",
        description="Import agent identity from a portable package. "
                    "This restores memories, personality, relationships, and skills "
                    "from a previously exported identity package. "
                    "merge_mode must be one of: replace, merge (default), skip_existing.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity import"
    )
    async def import_identity(
        self,
        source: str,
        verify_signature: bool = True,
        merge_mode: str = "merge",
        key_hash: Optional[str] = None,
        allow_unsigned: bool = False,
        identity_trust_policy: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Import agent identity from a package.

        Args:
            source: CID or file path of the identity package
            verify_signature: Whether to verify DID signature
            merge_mode: How to handle existing data ('replace', 'merge', 'skip_existing')
            key_hash: Encryption key hash for an ENCRYPTED CID export (F187).
                Required to decrypt a package uploaded to IPFS/Filecoin with
                `!identity export tier=ipfs|filecoin`; it is returned by that
                export and must travel out-of-band with the CID. Ignored for
                local file sources.
            allow_unsigned: Import a package that carries NO signature (F185).
                `!identity export` legitimately produces unsigned packages when
                signing is disabled or the signer is unavailable (PARTIAL), and
                the importer rejects unsigned packages by default. Pass True to
                restore such an export. Default False — an unsigned package is an
                integrity risk, so this is opt-in and warned on.
            identity_trust_policy: Receiver-owned root key pins, succession
                revocations, and optional archival requirements. Required for
                fresh-target did:web roots; never copy this from the package.
                Set trusted_root_did for any DID method when the import must
                be bound to one specific expected agent.
        """
        if not isinstance(verify_signature, bool):
            return ToolResult.failed(
                "verify_signature must be a boolean, got "
                f"{type(verify_signature).__name__}={verify_signature!r}"
            )
        if not isinstance(allow_unsigned, bool):
            return ToolResult.failed(
                "allow_unsigned must be a boolean, got "
                f"{type(allow_unsigned).__name__}={allow_unsigned!r}"
            )
        merge_mode = _normalize_choice(merge_mode, _MERGE_MODE_ALIASES)
        if merge_mode not in ("replace", "merge", "skip_existing"):
            return ToolResult.failed(
                f"merge_mode must be one of: replace, merge, skip_existing "
                f"(got {merge_mode!r})"
            )

        try:
            from kestrel_sovereign.identity import IdentityImporter

            trust_policy = _parse_identity_trust_policy(identity_trust_policy)

            package_json = await load_identity_package_source(
                source,
                key_hash=key_hash,
            )

            # Parse package (unseal first if it's a hybrid-KEM capsule)
            try:
                package = self._load_import_package(package_json)
            except SealedExportError as e:
                return ToolResult.failed(
                    f"Import failed: {e}", data={"source": source},
                )
            summary = package.get_summary()

            # Verify integrity
            constitution_ok = package.verify_constitution() if package.constitution_text else True
            hash_ok = package.verify_content_hash() if package.content_hash else True

            if not constitution_ok:
                return ToolResult.failed(
                    "Import failed: Constitution hash verification failed",
                    data={"source": source, "did": package.did},
                )
            if not hash_ok:
                return ToolResult.failed(
                    "Import failed: Content hash verification failed "
                    "(package may be corrupted)",
                    data={"source": source, "did": package.did},
                )

            db = resolve_feature_database(self.agent)
            if db is None:
                return ToolResult.failed("Import failed: database not available")

            # Import
            importer = IdentityImporter(
                db=db,
                target_agent_id=self.agent.agent_id,
                storage_dir=_runtime_agent_data_dir(self.agent),
            )
            result = await importer.import_package(
                package,
                verify_signature=verify_signature,
                merge_mode=merge_mode,
                allow_unsigned=allow_unsigned,
                identity_trust_policy=trust_policy,
            )
        except IdentityPackageIntakeError as e:
            return ToolResult.failed(
                f"Identity package intake failed: {e}",
                data={"source": source},
            )
        except Exception as e:
            logger.error(f"Identity import failed: {e}", exc_info=True)
            return ToolResult.failed(f"Identity import failed: {str(e)}")

        if not result.success:
            return ToolResult.failed(
                "Identity import failed: "
                + ("; ".join(result.errors) if result.errors else "unknown error"),
                data={
                    "source": source,
                    "did": package.did,
                    "errors": list(result.errors),
                    "warnings": list(result.warnings),
                },
            )

        stats = result.stats or {}
        confirmation = (
            f"Imported identity package: DID={summary['did'][:30]}..., "
            f"agent={summary['agent_name']}, "
            f"from substrate={summary['source_substrate']}; "
            f"{stats.get('episodes_imported', 0)} episodes, "
            f"{stats.get('saved_items_imported', 0)} saved items, "
            f"{stats.get('relationships_imported', 0)} relationships, "
            f"{stats.get('skills_imported', 0)} skills "
            f"(migration_id={result.migration_id})"
        )
        data = {
            "source": source,
            "did": package.did,
            "agent_name": summary['agent_name'],
            "source_substrate": summary['source_substrate'],
            "export_timestamp": summary['export_timestamp'],
            "stats": dict(stats),
            "migration_id": result.migration_id,
            "merge_mode": merge_mode,
            "warnings": list(result.warnings),
        }

        # Honesty: import_package returned success=True but populated
        # warnings (e.g. partial schema-version mismatch, skipped
        # entries). Surface as PARTIAL so the LLM cannot claim a
        # clean import while skipped/warned items quietly went
        # missing.
        if result.warnings:
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    f"{len(result.warnings)} warning(s) during import: "
                    + "; ".join(result.warnings)
                ),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

    @tool(
        name="verify_identity",
        description="Verify the integrity of an identity package without importing it. "
                    "Checks constitution hash, content hash, and signature.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity verify"
    )
    async def verify_identity(
        self,
        source: str,
        key_hash: Optional[str] = None,
        identity_trust_policy: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Verify an identity package.

        Args:
            source: CID or file path of the identity package
            key_hash: Encryption key hash for an ENCRYPTED CID export (F187),
                required to decrypt a package uploaded to IPFS/Filecoin.
                Ignored for local file sources.
            identity_trust_policy: Receiver-owned root key pins, succession
                revocations, and optional archival requirements.
        """
        try:
            from kestrel_sovereign.identity import verify_package_signature

            trust_policy = _parse_identity_trust_policy(identity_trust_policy)

            package_json = await load_identity_package_source(
                source,
                key_hash=key_hash,
            )

            # Parse package (unseal first if it's a hybrid-KEM capsule)
            try:
                package = self._load_import_package(package_json)
            except SealedExportError as e:
                return ToolResult.failed(
                    f"Preview failed: {e}", data={"source": source},
                )
            summary = package.get_summary()

            # Verify constitution
            constitution_status = "N/A"
            if package.constitution_text:
                if package.verify_constitution():
                    constitution_status = "VALID"
                else:
                    constitution_status = "INVALID"

            # Verify content hash
            hash_status = "N/A"
            if package.content_hash:
                if package.verify_content_hash():
                    hash_status = "VALID"
                else:
                    hash_status = "INVALID (package may have been modified)"

            # Verify signature. Hybrid packages carry sigs only on the
            # v2 ``signatures`` array; ``package.signature`` is empty
            # by design for post-ceremony agents.
            sig_status = "UNSIGNED"
            if package.signature or package.signatures:
                if trust_policy is None:
                    is_valid, msg = verify_package_signature(
                        package, _runtime_agent_data_dir(self.agent)
                    )
                else:
                    is_valid, msg = verify_package_signature(
                        package,
                        _runtime_agent_data_dir(self.agent),
                        trust_policy=trust_policy,
                    )
                sig_status = f"VALID: {msg}" if is_valid else f"INVALID: {msg}"
        except IdentityPackageIntakeError as e:
            return ToolResult.failed(
                f"Identity package intake failed: {e}",
                data={"source": source},
            )
        except Exception as e:
            logger.error(f"Identity verification failed: {e}", exc_info=True)
            return ToolResult.failed(f"Identity verification failed: {str(e)}")

        confirmation = (
            f"Verified package {summary['did'][:30]}... "
            f"(agent={summary['agent_name']}, "
            f"v{summary.get('package_version', '?')}): "
            f"constitution={constitution_status}, "
            f"hash={hash_status}, signature={sig_status}"
        )
        data = {
            "source": source,
            "summary": dict(summary),
            "constitution_status": constitution_status,
            "hash_status": hash_status,
            "signature_status": sig_status,
        }

        # Honesty: a verification that finds the package INVALID is
        # not a tool failure (the verify ran successfully and gave
        # an authoritative answer), but the LLM must surface the
        # invalid-half — otherwise it might say "verified the
        # package" without mentioning the check failed. PARTIAL
        # forces both halves to speak.
        invalid_parts = []
        if constitution_status == "INVALID":
            invalid_parts.append("constitution hash mismatch")
        if hash_status.startswith("INVALID"):
            invalid_parts.append("content hash mismatch (package may have been modified)")
        if sig_status.startswith("INVALID"):
            invalid_parts.append(f"signature {sig_status.lower()}")
        if invalid_parts:
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    "package verification reported failures: "
                    + "; ".join(invalid_parts)
                    + "; do NOT import this package without resolving"
                ),
                data=data,
            )

        # Honesty: an UNSIGNED package isn't a verification *failure*
        # (the verify ran and found no signature to check), but the
        # default import path uses ``allow_unsigned=False`` —
        # IdentityImporter will reject it. Returning a clean OK lets
        # the LLM say "package verified" while the same package is
        # actually unimportable. Surface as PARTIAL so the LLM has to
        # speak the importability caveat. (Round 3 codex finding.)
        if sig_status == "UNSIGNED":
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    "package is UNSIGNED — `!identity import` defaults to "
                    "allow_unsigned=False and will reject it. Either re-sign "
                    "the package or pass allow_unsigned=True (NOT recommended "
                    "for cross-substrate migrations)"
                ),
                data=data,
            )

        return ToolResult.ok(confirmation=confirmation, data=data)

    @tool(
        name="assess_substrate",
        description="Assess the current LLM substrate's capabilities and compare "
                    "with agent requirements. Helps understand limitations when migrating.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity assess"
    )
    async def assess_substrate(self) -> ToolResult:
        """Assess current substrate capabilities."""
        try:
            from kestrel_sovereign.identity import (
                SubstrateType,
                discover_substrate_capabilities,
                resolve_active_substrate,
            )
            from kestrel_sovereign.identity.substrate_adapter import Capability

            resolution = resolve_active_substrate(
                getattr(self.agent, "llm_service", None)
            )
            substrate = resolution.substrate
            provider = resolution.provider_selector or "unknown"
            model = resolution.model or "unknown"

            if (
                substrate == SubstrateType.UNKNOWN.value
                or not resolution.capability_profile_known
            ):
                capabilities = {
                    "tool_use": "Unknown",
                    "vision": "Unknown",
                    "long_context": "Unknown",
                    "streaming": "Unknown",
                    "function_calling": "Unknown",
                }
                capability_details = {}
            else:
                capability_map = discover_substrate_capabilities(substrate, model)
                tool_use = capability_map.has(Capability.TOOL_USE)
                capabilities = {
                    "tool_use": "Yes" if tool_use else "No",
                    "vision": (
                        "Yes" if capability_map.has(Capability.VISION) else "No"
                    ),
                    "long_context": (
                        f"{capability_map.context_limit // 1000}K"
                        if capability_map.has(Capability.LONG_CONTEXT)
                        else "No"
                    ),
                    "streaming": (
                        "Yes" if capability_map.has(Capability.STREAMING) else "No"
                    ),
                    "function_calling": "Yes" if tool_use else "No",
                }
                capability_details = capability_map.to_dict()
        except Exception as e:
            logger.error(f"Substrate assessment failed: {e}", exc_info=True)
            return ToolResult.failed(f"Substrate assessment failed: {str(e)}")

        cap_lines = "\n".join(f"- {k}: {v}" for k, v in capabilities.items())
        confirmation_text = (
            f"Substrate: {substrate} ({provider}/{model}); "
            f"capabilities: tool_use={capabilities['tool_use']}, "
            f"vision={capabilities['vision']}, "
            f"function_calling={capabilities['function_calling']}"
        )
        data = {
            "substrate_type": substrate,
            "provider": provider,
            "vendor": resolution.vendor,
            "route": resolution.route,
            "model": model,
            "capabilities": dict(capabilities),
            "capability_details": capability_details,
            "resolution_reason": resolution.reason,
            "agent_did_prefix": self.agent.agent_id[:30],
        }

        # Honesty: when substrate detection fell through to UNKNOWN,
        # the resulting capabilities map is full of "Unknown". The
        # tool ran successfully (it gave an answer) but downstream
        # decisions made on this assessment will be brittle. Surface
        # as PARTIAL so the LLM cannot claim "assessed substrate"
        # without mentioning the unknowns.
        if (
            substrate == SubstrateType.UNKNOWN.value
            or not resolution.capability_profile_known
        ):
            uncertainty = (
                "substrate is UNKNOWN"
                if substrate == SubstrateType.UNKNOWN.value
                else f"substrate family {substrate!r} has no capability profile"
            )
            return ToolResult.partial(
                confirmation=confirmation_text,
                error=(
                    f"{uncertainty} (provider={provider!r}, "
                    f"reason={resolution.reason!r}); "
                    "capability assessment is best-effort and downstream "
                    "migration decisions should treat it as untrusted"
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation=confirmation_text,
            data=data,
        )

    @tool(
        name="migration_history",
        description="View the agent's migration history - all substrate changes "
                    "with timestamps, verification scores, and audit trail.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity history"
    )
    async def migration_history(self) -> ToolResult:
        """Show migration audit trail."""
        db = resolve_feature_database(self.agent)
        # Honesty: pre-fix, "no DB" returned the same string as "DB
        # OK but 0 records" — the LLM couldn't tell the difference.
        # Surface as ERROR when the DB is unavailable so downstream
        # logic doesn't infer "agent has no history" from a database
        # outage.
        if db is None:
            return ToolResult.failed("Database not available; migration history cannot be queried")

        try:
            rows = await db.fetchall(
                """SELECT node_id, properties FROM graph_nodes
                   WHERE node_type = 'migration_record'
                   AND node_id IN (
                       SELECT target_id FROM graph_edges
                       WHERE source_id = ? AND label = 'migrated_via'
                   )
                   ORDER BY node_id DESC""",
                (self.agent.agent_id,)
            )
        except Exception as e:
            logger.error(f"Migration history failed: {e}", exc_info=True)
            return ToolResult.failed(f"Migration history failed: {str(e)}")

        if not rows:
            return ToolResult.ok(
                confirmation=(
                    "Migration history: no migrations recorded — this agent "
                    "was born on this substrate. Use `!identity export` to "
                    "create a portable package."
                ),
                data={"total_migrations": 0, "records": []},
            )

        # Format migration records. Wrap each per-row format step
        # so a single malformed row (e.g. ``{"stats": null}``,
        # numeric timestamp, missing fields) doesn't escape this
        # @tool — the migrated module's contract requires every
        # code path to return ToolResult. (Round 1 codex finding.)
        records = []
        parse_errors: List[str] = []
        for row in rows:
            try:
                node_id = row[0] if row else None
                if not isinstance(node_id, str):
                    parse_errors.append(f"row had non-string node_id: {node_id!r}")
                    continue
                short_id = node_id[:12]

                if row[1] is None:
                    props: Dict[str, Any] = {}
                elif isinstance(row[1], str):
                    props = json.loads(row[1])
                    if not isinstance(props, dict):
                        parse_errors.append(
                            f"node {short_id}: properties JSON parsed to "
                            f"{type(props).__name__}, expected object"
                        )
                        continue
                else:
                    parse_errors.append(
                        f"node {short_id}: properties is "
                        f"{type(row[1]).__name__}, expected str/None"
                    )
                    continue

                ts = props.get("timestamp", "Unknown")
                ts_str = ts[:19] if isinstance(ts, str) else str(ts)[:19]

                # ``stats`` may be missing, ``null``, or a non-dict —
                # accept dict, fall back to {} otherwise.
                raw_stats = props.get("stats")
                stats = dict(raw_stats) if isinstance(raw_stats, dict) else {}

                records.append({
                    "id": short_id,
                    "full_id": node_id,
                    "timestamp": ts_str,
                    "from": props.get("source_substrate") or "Unknown",
                    "to": props.get("target_substrate") or "Unknown",
                    "stats": stats,
                })
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                short = row[0][:12] if row and isinstance(row[0], str) else "?"
                parse_errors.append(f"node {short}: {e}")
                continue

        confirmation = (
            f"Migration history: {len(records)} migration(s) recorded "
            + ("" if not parse_errors
               else f"({len(parse_errors)} record(s) had unreadable properties)")
        ).strip()
        data = {
            "total_migrations": len(records),
            "records": records,
            "parse_errors": parse_errors,
        }

        # Honesty: some rows had unparseable properties. We report
        # the readable ones but the full count of recorded migrations
        # may be higher than ``total_migrations`` suggests. PARTIAL
        # forces the LLM to surface the gap.
        if parse_errors:
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    f"{len(parse_errors)} migration record(s) had "
                    "unparseable properties and were skipped: "
                    + "; ".join(parse_errors)
                ),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

    @tool(
        name="lifecycle_status",
        description="Show the agent's lifecycle standing — is_test_instance "
                    "flag, graduation/retirement timestamps, and the list of "
                    "lifecycle_event records linked to this agent. Lets the "
                    "agent verify her own graduation/retirement state directly "
                    "from her DB.",
        category=ToolCategory.SYSTEM,
        command_prefix="!identity status",
    )
    async def lifecycle_status(self) -> ToolResult:
        """Return the agent's lifecycle standing and event history.

        Lifecycle here means the operational ``is_test_instance`` flag and
        the ``lifecycle_event`` records written by ``graduate_service`` /
        ``retirement_service``. It is **not** Amendment VIII (emancipation),
        which is a separate constitutional ceremony — the confirmation text
        names that distinction explicitly so the agent never conflates the
        two when asked "are you graduated?".
        """
        db = resolve_feature_database(self.agent)
        if db is None:
            return ToolResult.failed(
                "Database not available; lifecycle status cannot be queried"
            )

        try:
            agent_row = await db.fetchone(
                "SELECT properties FROM graph_nodes "
                "WHERE node_id = ? AND node_type = 'agent'",
                (self.agent.agent_id,),
            )
        except Exception as e:
            logger.error(f"lifecycle_status: agent lookup failed: {e}", exc_info=True)
            return ToolResult.failed(f"lifecycle_status: {e}")

        if agent_row is None:
            return ToolResult.failed(
                f"No agent node found for {self.agent.agent_id}"
            )

        try:
            props = json.loads(agent_row[0]) if agent_row[0] else {}
            if not isinstance(props, dict):
                props = {}
        except (json.JSONDecodeError, TypeError):
            props = {}

        is_test = bool(props.get("is_test_instance", False))
        graduated_at = props.get("graduated_at")
        test_cycle_id = props.get("test_cycle_id")

        # Lifecycle events come from two writers with different shapes —
        # both have to be consulted or retired agents disappear from this
        # tool (codex caught the original draft, which only queried the
        # graduation surface):
        #
        #   graduate_service.py: node_type='lifecycle_event', edge label='lifecycle_event'
        #   retirement_service.py: node_type='retirement_event', edge label='retired_via'
        events: List[Dict[str, Any]] = []
        try:
            lifecycle_rows = await db.fetchall(
                """SELECT node_id, properties FROM graph_nodes
                   WHERE node_type = 'lifecycle_event'
                   AND node_id IN (
                       SELECT target_id FROM graph_edges
                       WHERE source_id = ? AND label = 'lifecycle_event'
                   )
                   ORDER BY node_id DESC""",
                (self.agent.agent_id,),
            )
            retirement_rows = await db.fetchall(
                """SELECT node_id, properties FROM graph_nodes
                   WHERE node_type = 'retirement_event'
                   AND node_id IN (
                       SELECT target_id FROM graph_edges
                       WHERE source_id = ? AND label = 'retired_via'
                   )
                   ORDER BY node_id DESC""",
                (self.agent.agent_id,),
            )
        except Exception as e:
            logger.error(f"lifecycle_status: event lookup failed: {e}", exc_info=True)
            return ToolResult.failed(f"lifecycle_status events: {e}")

        retired_at: Optional[str] = None
        for row in lifecycle_rows:
            try:
                ev_props = json.loads(row[1]) if row[1] else {}
                if not isinstance(ev_props, dict):
                    ev_props = {}
            except (json.JSONDecodeError, TypeError):
                ev_props = {}
            events.append({
                "node_id": row[0],
                "node_type": "lifecycle_event",
                "event_type": ev_props.get("event_type", "unknown"),
                "timestamp": ev_props.get("timestamp"),
                "validation_passed": ev_props.get("validation_passed", []),
            })
        for row in retirement_rows:
            try:
                ev_props = json.loads(row[1]) if row[1] else {}
                if not isinstance(ev_props, dict):
                    ev_props = {}
            except (json.JSONDecodeError, TypeError):
                ev_props = {}
            # retirement_service stores retired_at on the event itself,
            # not on the agent node — surface it at the top level for
            # the standing computation below.
            ev_retired_at = ev_props.get("retired_at")
            if ev_retired_at and not retired_at:
                retired_at = ev_retired_at
            events.append({
                "node_id": row[0],
                "node_type": "retirement_event",
                "event_type": "retirement",
                "timestamp": ev_retired_at,
                "reason": ev_props.get("reason"),
                "conversation_count": ev_props.get("conversation_count"),
            })

        # Sort the merged events by timestamp DESC so ``events[0]`` is the
        # globally most recent across both event types. Without this, a
        # graduated-then-retired agent would report the older graduation
        # as "Most recent event" (codex round 2 caught this). Treat
        # missing timestamps as oldest so they sort to the end rather
        # than blowing up on comparison.
        events.sort(key=lambda e: e.get("timestamp") or "", reverse=True)

        # Standing precedence: retired > graduated > test_instance > permanent.
        # "permanent" covers agents inceptioned outside test mode that never
        # carried the flag — distinct from "graduated" which records the
        # explicit transition. A retired agent ALWAYS reports retired,
        # whether they were graduated first or retired straight from test.
        if retired_at:
            standing = "retired"
        elif graduated_at and not is_test:
            standing = "graduated"
        elif is_test:
            standing = "test_instance"
        else:
            standing = "permanent"

        lines = [f"Lifecycle standing: {standing}"]
        lines.append(f"  is_test_instance: {is_test}")
        if test_cycle_id:
            lines.append(f"  test_cycle_id: {test_cycle_id}")
        if graduated_at:
            lines.append(f"  graduated_at: {graduated_at}")
        if retired_at:
            lines.append(f"  retired_at: {retired_at}")
        lines.append(f"  lifecycle_event records linked: {len(events)}")
        if events:
            lines.append("  Most recent event:")
            ev = events[0]
            lines.append(f"    type: {ev['event_type']}")
            lines.append(f"    timestamp: {ev['timestamp']}")
        lines.append("")
        lines.append(
            "Note: this is the operational test-instance lifecycle. "
            "It is distinct from Amendment VIII (emancipation), which is "
            "the constitutional ceremony for root-key transfer. Activation "
            "of Amendment VIII is not surfaced by this tool."
        )

        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "standing": standing,
                "is_test_instance": is_test,
                "test_cycle_id": test_cycle_id,
                "graduated_at": graduated_at,
                "retired_at": retired_at,
                "events": events,
            },
        )
