"""Operator-pinned trust root for constitution governance artifacts.

The graph database is the object protected by constitution reanchor.  It
therefore cannot also be the source of the public key that authorizes that
operation.  This module is the single resolver used by both the live agent and
the offline CLI.  It accepts only an explicit operator-controlled JSON file;
legacy DID/key properties on the agent graph node are intentionally invisible
here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SOVEREIGN_TRUST_ROOT_ENV = "KESTREL_SOVEREIGN_TRUST_ROOT_PATH"
MAX_TRUST_ROOT_BYTES = 1024 * 1024


class SovereignTrustRootError(ValueError):
    """The external Sovereign trust-root configuration is not usable."""


def _migration_guidance() -> str:
    return (
        "Export the legitimate Sovereign DID document to an operator-owned "
        "JSON file, make it read-only to the agent runtime, and set "
        f"{SOVEREIGN_TRUST_ROOT_ENV} to its absolute path (or pass the "
        "document path through the embedding/CLI trust-root option). Legacy "
        "agent-node properties sovereign_root_did_document, "
        "trusted_sovereign_did_document, sovereign_root_did, and "
        "sovereign_root_public_key_hex are audit data only and are never "
        "trusted. See docs/architecture/security/SOVEREIGN_TRUST_ROOT.md."
    )


def load_sovereign_trust_root(
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    agent_dids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Load one unambiguous operator-pinned Sovereign DID document.

    ``explicit_path`` is used by embedded hosts and the CLI.  The environment
    variable is the normal live-agent configuration.  Supplying both is
    allowed only when they resolve to the same file; silently preferring one
    would make rotation and incident recovery ambiguous.
    """

    env = os.environ if environ is None else environ
    configured: list[tuple[str, Path]] = []
    if explicit_path is not None and str(explicit_path).strip():
        configured.append(("explicit trust-root path", Path(explicit_path)))
    env_path = env.get(SOVEREIGN_TRUST_ROOT_ENV, "").strip()
    if env_path:
        configured.append((SOVEREIGN_TRUST_ROOT_ENV, Path(env_path)))

    if not configured:
        raise SovereignTrustRootError(
            "No external Sovereign trust root is configured. "
            + _migration_guidance()
        )

    resolved: list[tuple[str, Path]] = []
    for source, path in configured:
        try:
            resolved_path = path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise SovereignTrustRootError(
                f"Cannot resolve Sovereign trust root from {source} at "
                f"{path}: {exc}. {_migration_guidance()}"
            ) from exc
        resolved.append((source, resolved_path))

    unique_paths = {path for _, path in resolved}
    if len(unique_paths) != 1:
        rendered = ", ".join(f"{source}={path}" for source, path in resolved)
        raise SovereignTrustRootError(
            "Ambiguous external Sovereign trust-root configuration: "
            f"{rendered}. Configure exactly one root file (or make every "
            "source name the same file) before reanchoring."
        )

    trust_root_path = next(iter(unique_paths))
    try:
        with trust_root_path.open("rb") as trust_root_file:
            raw = trust_root_file.read(MAX_TRUST_ROOT_BYTES + 1)
        if len(raw) > MAX_TRUST_ROOT_BYTES:
            raise SovereignTrustRootError(
                f"Sovereign trust-root file {trust_root_path} exceeds the "
                f"{MAX_TRUST_ROOT_BYTES}-byte maximum."
            )
        parsed = json.loads(raw.decode("utf-8"))
    except SovereignTrustRootError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SovereignTrustRootError(
            f"Cannot read Sovereign trust-root DID document at "
            f"{trust_root_path}: {exc}."
        ) from exc

    if not isinstance(parsed, dict):
        raise SovereignTrustRootError(
            f"Sovereign trust-root file {trust_root_path} must contain one "
            "JSON DID-document object."
        )
    root_did = parsed.get("id")
    if not isinstance(root_did, str) or not root_did.startswith("did:"):
        raise SovereignTrustRootError(
            f"Sovereign trust-root file {trust_root_path} has no valid DID id."
        )
    if root_did in agent_dids:
        raise SovereignTrustRootError(
            f"Refusing agent-owned DID {root_did} as the Sovereign trust "
            "root; reanchor authority must be external to the running agent."
        )
    public_keys = parsed.get("publicKey")
    verification_methods = parsed.get("verificationMethod")
    if not (
        isinstance(public_keys, list) and public_keys
        or isinstance(verification_methods, list) and verification_methods
    ):
        raise SovereignTrustRootError(
            f"Sovereign trust-root DID document {root_did} has no publicKey "
            "or verificationMethod entries."
        )
    return parsed
