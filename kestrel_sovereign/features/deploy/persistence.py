"""Fail-closed persistence policy for Cloud Run deployments.

Cloud Run's writable container filesystem is disposable.  A profile therefore
has to choose one of two honest contracts:

``ephemeral_demo``
    One instance at most, development/test only, and identity is explicitly
    demo-scoped.  Cold starts may mint a new temporary identity.

``durable_sovereign``
    A pinned encrypted identity bundle is restored from Secret Manager and all
    coherent state lives in PostgreSQL.  Multiple instances may then share the
    same identity and transactional backend without relying on local files.

The validation lives outside the provider so both the manager and direct
provider callers enforce exactly the same invariant.
"""

from __future__ import annotations

from .models import DeployManagerError, DeploymentProfile, DeployProviderType


PERSISTENCE_ENV = "KESTREL_DEPLOYMENT_PERSISTENCE"
EPHEMERAL_DEMO = "ephemeral_demo"
DURABLE_SOVEREIGN = "durable_sovereign"

_DURABLE_SECRET_KEYS = (
    "KESTREL_DATABASE_URL",
    "KESTREL_HOLD_EVIDENCE_DATABASE_URL",
    "KESTREL_DATA_KEY",
    "KESTREL_IDENTITY_BUNDLE",
)


def _normalized(value: object) -> str:
    return str(value or "").strip().lower()


def _enabled(value: object) -> bool:
    return _normalized(value) in {"1", "true", "yes", "on"}


def _pinned_secret_version(secret_ref: object) -> bool:
    """Return whether a Secret Manager ref names an immutable version.

    Secret versions are numeric.  ``latest`` can resolve differently for two
    instances of the same revision and is therefore not an identity-custody
    boundary.
    """
    if not isinstance(secret_ref, str) or ":" not in secret_ref:
        return False
    name, version = secret_ref.rsplit(":", 1)
    return bool(name) and version.isdigit() and int(version) > 0


def validate_cloudrun_persistence(profile: DeploymentProfile) -> None:
    """Reject a Cloud Run profile that can lose or equivocate identity."""
    if profile.provider != DeployProviderType.CLOUD_RUN:
        return

    mode = _normalized(profile.persistence_mode)
    runtime_mode = _normalized(profile.env_vars.get(PERSISTENCE_ENV))
    if mode not in {EPHEMERAL_DEMO, DURABLE_SOVEREIGN}:
        raise DeployManagerError(
            "Cloud Run profile must explicitly set persistence_mode to "
            f"'{EPHEMERAL_DEMO}' or '{DURABLE_SOVEREIGN}'; local container "
            "identity/state is disposable"
        )
    if runtime_mode != mode:
        raise DeployManagerError(
            f"Cloud Run profile persistence_mode={mode!r} must set "
            f"env_vars.{PERSISTENCE_ENV}={mode!r} so the container enforces "
            "the same policy at startup"
        )

    environment = _normalized(profile.env_vars.get("KESTREL_ENV"))
    backend = _normalized(profile.env_vars.get("KESTREL_DB_BACKEND"))

    if mode == EPHEMERAL_DEMO:
        if profile.max_instances != 1:
            raise DeployManagerError(
                "ephemeral_demo Cloud Run profiles require max_instances=1; "
                "otherwise separate instances can mint different keys for "
                "the same configured service"
            )
        if environment in {"prod", "production"}:
            raise DeployManagerError(
                "ephemeral_demo identity cannot be advertised as production; "
                "use durable_sovereign custody or a development/test profile"
            )
        return

    # Durable multi-agent custody needs one independently bound bundle and
    # database identity per agent.  The current single bundle contract cannot
    # represent that without ambiguity, so refuse rather than mint locally.
    if profile.is_multi_agent or _enabled(
        profile.env_vars.get("KESTREL_MULTI_AGENT")
    ) or _enabled(profile.env_vars.get("KESTREL_HOST_AUTOSTART")):
        raise DeployManagerError(
            "durable_sovereign Cloud Run currently supports single-agent "
            "profiles only; multi-agent profiles need per-agent custody and "
            "database bindings and are refused"
        )
    if backend != "postgres":
        raise DeployManagerError(
            "durable_sovereign Cloud Run requires KESTREL_DB_BACKEND=postgres; "
            "SQLite on the disposable container filesystem is not durable or "
            "multi-instance coherent"
        )

    expected_did = str(profile.env_vars.get("KESTREL_EXPECTED_DID") or "").strip()
    if not expected_did.startswith("did:") or "${" in expected_did:
        raise DeployManagerError(
            "durable_sovereign Cloud Run requires a resolved "
            "env_vars.KESTREL_EXPECTED_DID so restored keys and PostgreSQL "
            "state are bound to one declared identity"
        )

    missing = [key for key in _DURABLE_SECRET_KEYS if key not in profile.secrets]
    if missing:
        raise DeployManagerError(
            "durable_sovereign Cloud Run is missing Secret Manager custody "
            f"references: {', '.join(missing)}"
        )
    unpinned = [
        key
        for key in _DURABLE_SECRET_KEYS
        if not _pinned_secret_version(profile.secrets.get(key))
    ]
    if unpinned:
        raise DeployManagerError(
            "durable_sovereign custody secrets must use immutable numeric "
            "Secret Manager versions (never :latest): "
            + ", ".join(unpinned)
        )
    if (
        profile.secrets["KESTREL_DATABASE_URL"]
        == profile.secrets["KESTREL_HOLD_EVIDENCE_DATABASE_URL"]
    ):
        raise DeployManagerError(
            "durable_sovereign Hold custody requires independent primary and "
            "evidence database secret references"
        )


__all__ = [
    "DURABLE_SOVEREIGN",
    "EPHEMERAL_DEMO",
    "PERSISTENCE_ENV",
    "validate_cloudrun_persistence",
]
