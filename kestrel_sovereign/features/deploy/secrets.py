"""GCP Secret Manager sync — port of ``scripts/cloudrun/setup_secrets.sh``.

This module is the Python port of the bash setup script for sub-PR 1.2 of
epic #1050 (host-side bash → Python). The original script hardcoded three
secret names; this implementation derives the (env-var → secret-name)
mapping directly from ``deploy_config.toml``'s ``[profiles.*.secrets]``
sections, so adding a new secret to a profile is the only place the
operator needs to touch.

Public surface
--------------

* :func:`derive_secret_mapping` — extract the ``{env_var: secret_name}``
  dict from a parsed deploy config (single profile or all, deduped).
* :func:`load_env_file` — thin wrapper around ``dotenv.dotenv_values``.
* :func:`sync_secret` — create-or-update a single secret in GCP Secret
  Manager.
* :func:`sync_all_secrets` — orchestrator used by the CLI; iterates the
  derived mapping, reads values from .env, and reports a per-secret
  result.

Independence from :class:`DeployManager`
----------------------------------------

The agent-tool surface (``DeployFeature.deploy_agent``) intentionally
does NOT expose secrets sync — agents shouldn't be syncing developer
secrets — so this module is standalone. ``DeployManager`` is not
involved; the only thing we need from the deploy package is the
``deploy_config.toml`` schema, which we receive as a plain dict.

Secret Manager API enablement
-----------------------------

The bash script ran ``gcloud services enable secretmanager.googleapis.com``
as a best-effort step. We deliberately do NOT replicate that here — the
whole point of the bash-to-Python port is to eliminate the ``gcloud`` CLI
dependency so Kestrel works on Windows without WSL. The Python SDK will
surface a clear ``PermissionDenied`` / ``FailedPrecondition`` if the API
isn't enabled, and the operator enables it once via the GCP console (or
``gcloud services enable secretmanager.googleapis.com`` on a *nix box) at
project setup time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

# Recognized actions on a SecretSyncResult. The dry-run variants exist so
# callers can render a preview ("would create...") without ambiguity vs a
# real create that already happened.
ACTION_CREATED = "created"
ACTION_UPDATED = "updated"
ACTION_SKIPPED = "skipped"
ACTION_ERROR = "error"
ACTION_DRY_RUN_CREATE = "dry-run-create"
ACTION_DRY_RUN_UPDATE = "dry-run-update"


@dataclass
class SecretSyncResult:
    """Outcome of one (env-var → secret-name) mapping entry.

    ``action`` is one of:

    * ``"created"`` — secret didn't exist; created with one initial version.
    * ``"updated"`` — secret existed; new version added.
    * ``"skipped"`` — env var not present in .env (warn, don't fail).
    * ``"error"`` — GCP API call raised. Detail carries the message.
    * ``"dry-run-create"`` / ``"dry-run-update"`` — would-be action under
      ``--dry-run``; nothing was actually mutated.
    """

    secret_name: str
    env_var: str
    action: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Config-driven mapping
# ---------------------------------------------------------------------------

# A Secret Manager secret reference in ``deploy_config.toml`` looks like
# ``"<secret-name>:<version>"`` — e.g. ``"kestrel-openai-key:latest"``.
# Anything that doesn't match (notably the ``${VAR}`` placeholders used in
# the azure-style profile section) is skipped. We don't try to match the
# secret-name character class strictly because GCP allows
# ``[a-zA-Z][a-zA-Z0-9_-]*`` and we'd rather let GCP itself reject malformed
# names with a clear error than reinvent the rules.
_SECRET_REF_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9_.-]+$")


def _is_secret_manager_ref(value: str) -> bool:
    """Return True if ``value`` looks like ``<secret-name>:<version>``.

    ``${...}`` substitution placeholders (azure-style) and other unexpected
    shapes return False — they're not GCP Secret Manager refs and must be
    skipped to avoid garbage names being created.
    """
    if not isinstance(value, str):
        return False
    if value.startswith("${") and value.endswith("}"):
        return False
    return bool(_SECRET_REF_PATTERN.match(value))


def _strip_version(secret_ref: str) -> str:
    """``"kestrel-openai-key:latest"`` → ``"kestrel-openai-key"``."""
    return secret_ref.split(":", 1)[0]


def derive_secret_mapping(
    deploy_config: Dict[str, Any],
    profile: Optional[str] = None,
) -> Dict[str, str]:
    """Return ``{env_var: secret_name}`` derived from a parsed deploy config.

    Args:
        deploy_config: Parsed ``deploy_config.toml`` as a dict (the result
            of ``toml.load``). Only the ``[profiles.*]`` subtree is
            consulted.
        profile: If given, only that profile's ``[profiles.<name>.secrets]``
            section is scanned. If None, all profiles are merged.

    Returns:
        Mapping from environment-variable name to GCP Secret Manager
        secret name (without the ``:version`` suffix). Values like
        ``${AZURE_OPENAI_KEY}`` are skipped — they're azure-style
        substitutions, not Secret Manager refs.

    Raises:
        ValueError: If two profiles map the same env var to different
            secret names. That's a config inconsistency the operator
            should know about, not something we should silently pick a
            winner for.
        KeyError: If ``profile`` is given but doesn't exist in the config.
    """
    profiles = deploy_config.get("profiles", {}) or {}

    if profile is not None:
        if profile not in profiles:
            raise KeyError(
                f"profile '{profile}' not found in deploy_config "
                f"(available: {sorted(profiles.keys())})"
            )
        profile_iter = [(profile, profiles[profile])]
    else:
        profile_iter = list(profiles.items())

    # We track first-seen ``(env_var → (secret_name, profile))`` so we can
    # emit a helpful conflict message naming both offending profiles.
    mapping: Dict[str, str] = {}
    origins: Dict[str, str] = {}

    for prof_name, prof_data in profile_iter:
        secrets_section = (prof_data or {}).get("secrets", {}) or {}
        for env_var, raw_value in secrets_section.items():
            if not _is_secret_manager_ref(raw_value):
                logger.debug(
                    "skipping non-Secret-Manager ref in profile=%s: %s=%r",
                    prof_name, env_var, raw_value,
                )
                continue

            secret_name = _strip_version(raw_value)
            existing = mapping.get(env_var)
            if existing is not None and existing != secret_name:
                raise ValueError(
                    f"conflicting secret mapping for env var '{env_var}': "
                    f"profile '{origins[env_var]}' uses '{existing}', "
                    f"profile '{prof_name}' uses '{secret_name}'"
                )
            mapping[env_var] = secret_name
            origins[env_var] = prof_name

    return mapping


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def load_env_file(path: Path) -> Dict[str, str]:
    """Read a ``.env`` file via :func:`dotenv.dotenv_values`.

    Returns a flat ``{var: value}`` dict. ``dotenv_values`` handles
    comments, quoting, and multi-line values; we use it so the bash port
    matches the bash ``grep | cut | tr`` behaviour for any well-formed
    .env that worked under the script.

    No env-var expansion is performed — we want to store the raw value
    in Secret Manager, not the expanded form. Entries with ``None`` values
    (lines like ``FOO=`` with no RHS) are dropped.

    Raises:
        FileNotFoundError: If ``path`` doesn't exist. Operators care about
            the difference between "file missing" and "file empty"; bash's
            ``[ ! -f .env ] && exit 1`` had the same intent.
    """
    if not path.exists():
        raise FileNotFoundError(f".env file not found: {path}")

    # Imported lazily so that pure config-loading code (e.g. tests of
    # derive_secret_mapping) doesn't require dotenv to be importable.
    from dotenv import dotenv_values

    # ``interpolate=False`` is critical — secrets are values, not templates.
    # The bash script we're porting (`scripts/cloudrun/setup_secrets.sh`)
    # used ``cut -d'=' -f2-`` which is a literal byte slice, so a secret
    # like ``KESTREL_DATA_KEY=${SALT}`` would have been uploaded verbatim.
    # ``dotenv_values`` defaults to ``interpolate=True``, which would
    # silently expand ``${SALT}`` against earlier entries / ``os.environ``
    # — corrupting any password/key that happens to contain ``${...}``.
    raw = dotenv_values(str(path), interpolate=False)
    # dotenv_values returns Dict[str, Optional[str]]; drop None values.
    return {k: v for k, v in raw.items() if v is not None}


# ---------------------------------------------------------------------------
# Single-secret sync
# ---------------------------------------------------------------------------

def _project_path(client: Any, project_id: str) -> str:
    """``"projects/<id>"`` — used as the parent for ``create_secret``."""
    return client.common_project_path(project_id)


def _secret_path(client: Any, project_id: str, secret_name: str) -> str:
    """``"projects/<id>/secrets/<name>"`` — used as the name for
    ``get_secret`` and the parent for ``add_secret_version``."""
    return client.secret_path(project_id, secret_name)


def sync_secret(
    client: Any,
    project_id: str,
    secret_name: str,
    value: str,
    *,
    dry_run: bool = False,
) -> SecretSyncResult:
    """Create-or-update a single secret in GCP Secret Manager.

    Behaviour mirrors ``setup_secrets.sh``: if ``get_secret`` returns the
    secret exists, we add a new version; otherwise we ``create_secret``
    (replication policy ``automatic``) and then add the initial version.

    Args:
        client: A constructed
            :class:`google.cloud.secretmanager.SecretManagerServiceClient`.
            Passed in (rather than constructed inside) so tests can mock
            without monkeypatching the import.
        project_id: GCP project ID (no ``"projects/"`` prefix).
        secret_name: Name of the secret in Secret Manager.
        value: Secret payload, stored as UTF-8 bytes.
        dry_run: If True, no API mutation calls are made; a probe
            ``get_secret`` is still issued to determine
            create-vs-update so the dry-run preview reflects reality.

    Returns:
        :class:`SecretSyncResult` describing what happened.
    """
    # Lazy import — see module docstring. We want operators who never run
    # ``kestrel deploy secrets sync`` to not pay the import cost just for
    # having ``deploy/secrets.py`` on disk.
    from google.api_core import exceptions as gcp_exc
    from google.cloud import secretmanager

    parent = _project_path(client, project_id)
    secret_full_path = _secret_path(client, project_id, secret_name)

    # Probe — does the secret exist? get_secret raises NotFound if not.
    secret_exists = False
    try:
        client.get_secret(request={"name": secret_full_path})
        secret_exists = True
    except gcp_exc.NotFound:
        secret_exists = False
    except Exception as e:  # noqa: BLE001 — surface as result, not crash
        return SecretSyncResult(
            secret_name=secret_name,
            env_var="",  # caller fills in
            action=ACTION_ERROR,
            detail=f"get_secret failed: {e}",
        )

    if dry_run:
        return SecretSyncResult(
            secret_name=secret_name,
            env_var="",
            action=ACTION_DRY_RUN_UPDATE if secret_exists else ACTION_DRY_RUN_CREATE,
            detail=("would add new version" if secret_exists else "would create with automatic replication"),
        )

    try:
        if not secret_exists:
            # Create the secret container with automatic replication —
            # matches the bash script's ``--replication-policy="automatic"``.
            client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_name,
                    "secret": {
                        "replication": {"automatic": {}},
                    },
                }
            )

        # Add the version — this is the same call for both create and
        # update; Secret Manager versions are independent of the secret
        # container.
        client.add_secret_version(
            request={
                "parent": secret_full_path,
                "payload": {"data": value.encode("utf-8")},
            }
        )
    except Exception as e:  # noqa: BLE001 — surface as result, not crash
        return SecretSyncResult(
            secret_name=secret_name,
            env_var="",
            action=ACTION_ERROR,
            detail=f"{type(e).__name__}: {e}",
        )

    return SecretSyncResult(
        secret_name=secret_name,
        env_var="",
        action=ACTION_CREATED if not secret_exists else ACTION_UPDATED,
        detail="",
    )


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def sync_all_secrets(
    deploy_config: Dict[str, Any],
    env_path: Path,
    project_id: str,
    *,
    profile: Optional[str] = None,
    dry_run: bool = False,
) -> List[SecretSyncResult]:
    """Sync every secret in ``deploy_config`` to GCP Secret Manager.

    The Secret Manager API must be enabled on ``project_id`` before this
    call. We deliberately do NOT call ``gcloud services enable
    secretmanager.googleapis.com`` (that would defeat the
    Windows-friendly point of the bash-to-Python port). Enable it once
    via the GCP console or, on a *nix box,
    ``gcloud services enable secretmanager.googleapis.com``.

    Args:
        deploy_config: Parsed ``deploy_config.toml`` dict.
        env_path: Path to the ``.env`` file with secret values.
        project_id: GCP project ID.
        profile: If given, only that profile's secrets are synced.
        dry_run: If True, no Secret Manager mutations are made and no
            ``SecretManagerServiceClient`` is constructed for the
            mutation path. ``get_secret`` calls are still issued via the
            client so the preview reflects which secrets exist.

    Returns:
        One :class:`SecretSyncResult` per secret-mapping entry. Missing
        env-var values produce a ``"skipped"`` result; no exception is
        raised so a partial sync (e.g. only OPENAI_API_KEY set in .env)
        succeeds for the present keys.
    """
    mapping = derive_secret_mapping(deploy_config, profile=profile)

    if not mapping:
        logger.info(
            "no Secret Manager refs found in deploy_config "
            "(profile=%s) — nothing to sync",
            profile,
        )
        return []

    env_values = load_env_file(env_path)

    # SDK client is constructed lazily inside the loop, only on the first
    # non-skipped value. This keeps ``--dry-run --env-file /dev/null``
    # totally offline (every entry skips, client never constructed) and
    # makes "no values present in .env" cases a no-op rather than a
    # GCP-creds-required failure.
    client: Any = None

    results: List[SecretSyncResult] = []
    for env_var, secret_name in mapping.items():
        value = env_values.get(env_var)

        if value is None or value == "":
            results.append(SecretSyncResult(
                secret_name=secret_name,
                env_var=env_var,
                action=ACTION_SKIPPED,
                detail=f"env var {env_var} not set in {env_path}",
            ))
            continue

        if client is None:
            # Lazy import + lazy construct — see module docstring.
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()

        result = sync_secret(
            client,
            project_id,
            secret_name,
            value,
            dry_run=dry_run,
        )
        # sync_secret doesn't know the env var name; stamp it here so the
        # CLI can render "created kestrel-openai-key <- OPENAI_API_KEY".
        result.env_var = env_var
        results.append(result)

    return results


__all__ = [
    "SecretSyncResult",
    "ACTION_CREATED",
    "ACTION_UPDATED",
    "ACTION_SKIPPED",
    "ACTION_ERROR",
    "ACTION_DRY_RUN_CREATE",
    "ACTION_DRY_RUN_UPDATE",
    "derive_secret_mapping",
    "load_env_file",
    "sync_secret",
    "sync_all_secrets",
]
