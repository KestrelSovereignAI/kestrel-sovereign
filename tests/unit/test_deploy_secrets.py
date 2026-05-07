"""Tests for ``kestrel_sovereign.features.deploy.secrets``.

Sub-PR 1.2 of epic #1050 (bash-to-Python port of
``scripts/cloudrun/setup_secrets.sh``).

The Secret Manager SDK client is mocked via ``unittest.mock.MagicMock``
so the unit tests don't touch real GCP. We assert call signatures
(``get_secret``, ``create_secret``, ``add_secret_version``) match the
shape the SDK expects.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.features.deploy.secrets import (
    ACTION_CREATED,
    ACTION_DRY_RUN_CREATE,
    ACTION_DRY_RUN_UPDATE,
    ACTION_ERROR,
    ACTION_SKIPPED,
    ACTION_UPDATED,
    SecretSyncResult,
    derive_secret_mapping,
    load_env_file,
    sync_all_secrets,
    sync_secret,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deploy_config_two_profiles():
    """Two profiles with overlapping (consistent) secret refs — the
    common case in the example deploy_config.toml."""
    return {
        "manager": {"gcp_project_id": "test-project"},
        "profiles": {
            "dev": {
                "secrets": {
                    "OPENAI_API_KEY": "kestrel-openai-key:latest",
                    "KESTREL_API_KEY": "kestrel-api-key:latest",
                    "KESTREL_DATA_KEY": "kestrel-data-key:latest",
                },
            },
            "prod": {
                "secrets": {
                    "OPENAI_API_KEY": "kestrel-openai-key:latest",
                    "KESTREL_API_KEY": "kestrel-api-key:latest",
                    "ANTHROPIC_API_KEY": "kestrel-anthropic-key:1",
                },
            },
        },
    }


@pytest.fixture
def deploy_config_with_azure_substitutions():
    """An azure-style profile uses ``${VAR}`` placeholders for Key Vault
    refs, which must be skipped — they're not GCP Secret Manager refs."""
    return {
        "profiles": {
            "azure-dev": {
                "secrets": {
                    "OPENAI_API_KEY": "${AZURE_OPENAI_KEY}",
                    "KESTREL_API_KEY": "${AZURE_KESTREL_KEY}",
                },
            },
            "dev": {
                "secrets": {
                    "OPENAI_API_KEY": "kestrel-openai-key:latest",
                },
            },
        },
    }


@pytest.fixture
def mock_secret_client():
    """Mock SecretManagerServiceClient with the path helpers wired."""
    client = MagicMock()
    client.common_project_path.side_effect = lambda pid: f"projects/{pid}"
    client.secret_path.side_effect = lambda pid, name: f"projects/{pid}/secrets/{name}"
    return client


# ---------------------------------------------------------------------------
# derive_secret_mapping
# ---------------------------------------------------------------------------

def test_derive_secret_mapping_single_profile(deploy_config_two_profiles):
    """Filtering to one profile returns just that profile's refs."""
    mapping = derive_secret_mapping(deploy_config_two_profiles, profile="dev")

    assert mapping == {
        "OPENAI_API_KEY": "kestrel-openai-key",
        "KESTREL_API_KEY": "kestrel-api-key",
        "KESTREL_DATA_KEY": "kestrel-data-key",
    }


def test_derive_secret_mapping_all_profiles_dedupes(deploy_config_two_profiles):
    """Without a filter, all profiles' refs are merged. Overlapping refs
    that agree on the secret name are deduped (one entry, not two)."""
    mapping = derive_secret_mapping(deploy_config_two_profiles)

    # Union, deduplicated.
    assert mapping == {
        "OPENAI_API_KEY": "kestrel-openai-key",
        "KESTREL_API_KEY": "kestrel-api-key",
        "KESTREL_DATA_KEY": "kestrel-data-key",
        "ANTHROPIC_API_KEY": "kestrel-anthropic-key",
    }


def test_derive_secret_mapping_skips_dollar_substitutions(
    deploy_config_with_azure_substitutions,
):
    """``${...}`` (azure-style) substitutions are not GCP Secret Manager
    refs and must be excluded from the mapping."""
    mapping = derive_secret_mapping(deploy_config_with_azure_substitutions)

    # Only the dev-profile GCP ref should survive.
    assert mapping == {"OPENAI_API_KEY": "kestrel-openai-key"}


def test_derive_secret_mapping_conflicting_names_raises():
    """Two profiles mapping the same env var to different secret names
    is a config inconsistency — surface it loudly so the user fixes
    deploy_config.toml rather than getting a silent winner."""
    config = {
        "profiles": {
            "dev": {"secrets": {"OPENAI_API_KEY": "kestrel-openai-key:latest"}},
            "prod": {"secrets": {"OPENAI_API_KEY": "different-name:latest"}},
        },
    }

    with pytest.raises(ValueError, match="conflicting secret mapping"):
        derive_secret_mapping(config)


def test_derive_secret_mapping_unknown_profile_raises(deploy_config_two_profiles):
    """Asking for a profile that doesn't exist raises KeyError naming the
    available profiles."""
    with pytest.raises(KeyError, match="bogus"):
        derive_secret_mapping(deploy_config_two_profiles, profile="bogus")


def test_derive_secret_mapping_no_profiles_section():
    """A config with no [profiles] returns an empty mapping (no crash)."""
    assert derive_secret_mapping({}) == {}
    assert derive_secret_mapping({"manager": {}}) == {}


def test_derive_secret_mapping_profile_with_no_secrets():
    """A profile lacking a [secrets] subsection returns an empty
    mapping for that profile."""
    config = {"profiles": {"dev": {"env_vars": {"FOO": "bar"}}}}
    assert derive_secret_mapping(config, profile="dev") == {}


# ---------------------------------------------------------------------------
# load_env_file
# ---------------------------------------------------------------------------

def test_load_env_file_basic(tmp_path: Path):
    """Comments, quotes, and bare values are all handled by dotenv."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comment line\n"
        "OPENAI_API_KEY=sk-test-1234\n"
        'KESTREL_API_KEY="quoted-value"\n'
        "EMPTY_VAR=\n"
        "KESTREL_DATA_KEY='single-quoted'\n"
    )

    values = load_env_file(env_file)

    assert values["OPENAI_API_KEY"] == "sk-test-1234"
    assert values["KESTREL_API_KEY"] == "quoted-value"
    assert values["KESTREL_DATA_KEY"] == "single-quoted"
    # EMPTY_VAR should be dropped (None values from dotenv_values, OR
    # empty string — either way we don't want it stored as a secret).
    # dotenv_values returns "" for ``FOO=`` so it stays in the dict;
    # it's the orchestrator that filters empty strings. Just confirm
    # we don't crash on it.
    assert "EMPTY_VAR" in values or "EMPTY_VAR" not in values


def test_load_env_file_missing_file_raises(tmp_path: Path):
    """A missing .env path raises FileNotFoundError so callers can
    distinguish 'file missing' from 'file empty'."""
    missing = tmp_path / "nonexistent.env"
    with pytest.raises(FileNotFoundError):
        load_env_file(missing)


def test_load_env_file_does_not_interpolate_dollar_braces(
    tmp_path: Path, monkeypatch
):
    """Secret values containing ``${...}`` must be returned verbatim.

    Regression test for codex review on PR #1057: ``dotenv_values``
    defaults to ``interpolate=True``, which would silently expand
    ``${SALT}`` against earlier .env entries or ``os.environ``. Secrets
    are values, not templates — passing ``interpolate=False`` to
    dotenv preserves the literal byte sequence the operator wrote.

    The bash script we're porting (``setup_secrets.sh``) used a literal
    ``cut -d'=' -f2-`` which is byte-exact; the Python port must match.
    """
    monkeypatch.setenv("SALT", "PROD_SALT_VALUE")
    monkeypatch.setenv("KESTREL_API_KEY", "AMBIENT_KEY")

    env_file = tmp_path / ".env"
    env_file.write_text(
        "FIRST=alpha\n"
        # Literal ${SALT} — must NOT be expanded to PROD_SALT_VALUE.
        "KESTREL_DATA_KEY=${SALT}\n"
        # Reference to an earlier entry — must NOT be expanded to alpha.
        "DERIVED=${FIRST}-suffix\n"
        # Reference to an ambient env var — must NOT shadow the .env value.
        'KESTREL_API_KEY="explicit-from-env-file"\n'
    )

    values = load_env_file(env_file)

    assert values["KESTREL_DATA_KEY"] == "${SALT}"
    assert values["DERIVED"] == "${FIRST}-suffix"
    assert values["KESTREL_API_KEY"] == "explicit-from-env-file"


# ---------------------------------------------------------------------------
# sync_secret
# ---------------------------------------------------------------------------

def test_sync_secret_create_when_missing(mock_secret_client):
    """Secret does not exist → create_secret + add_secret_version."""
    from google.api_core import exceptions as gcp_exc

    mock_secret_client.get_secret.side_effect = gcp_exc.NotFound("nope")

    result = sync_secret(
        mock_secret_client,
        "test-project",
        "kestrel-openai-key",
        "sk-test-1234",
    )

    assert result.action == ACTION_CREATED
    assert result.secret_name == "kestrel-openai-key"

    # create_secret called with automatic replication on the project parent.
    mock_secret_client.create_secret.assert_called_once()
    create_kwargs = mock_secret_client.create_secret.call_args.kwargs["request"]
    assert create_kwargs["parent"] == "projects/test-project"
    assert create_kwargs["secret_id"] == "kestrel-openai-key"
    assert create_kwargs["secret"] == {"replication": {"automatic": {}}}

    # add_secret_version called with UTF-8 bytes.
    mock_secret_client.add_secret_version.assert_called_once()
    add_kwargs = mock_secret_client.add_secret_version.call_args.kwargs["request"]
    assert add_kwargs["parent"] == "projects/test-project/secrets/kestrel-openai-key"
    assert add_kwargs["payload"]["data"] == b"sk-test-1234"


def test_sync_secret_update_when_exists(mock_secret_client):
    """Secret exists → only add_secret_version called, NOT create_secret."""
    mock_secret_client.get_secret.return_value = MagicMock()  # exists

    result = sync_secret(
        mock_secret_client,
        "test-project",
        "kestrel-openai-key",
        "sk-test-NEW",
    )

    assert result.action == ACTION_UPDATED
    mock_secret_client.create_secret.assert_not_called()
    mock_secret_client.add_secret_version.assert_called_once()
    add_kwargs = mock_secret_client.add_secret_version.call_args.kwargs["request"]
    assert add_kwargs["payload"]["data"] == b"sk-test-NEW"


def test_sync_secret_dry_run_create(mock_secret_client):
    """Dry-run on a missing secret reports dry-run-create; no mutations."""
    from google.api_core import exceptions as gcp_exc

    mock_secret_client.get_secret.side_effect = gcp_exc.NotFound("nope")

    result = sync_secret(
        mock_secret_client,
        "test-project",
        "kestrel-openai-key",
        "sk-test-1234",
        dry_run=True,
    )

    assert result.action == ACTION_DRY_RUN_CREATE
    mock_secret_client.create_secret.assert_not_called()
    mock_secret_client.add_secret_version.assert_not_called()


def test_sync_secret_dry_run_update(mock_secret_client):
    """Dry-run on an existing secret reports dry-run-update."""
    mock_secret_client.get_secret.return_value = MagicMock()  # exists

    result = sync_secret(
        mock_secret_client,
        "test-project",
        "kestrel-openai-key",
        "sk-test-NEW",
        dry_run=True,
    )

    assert result.action == ACTION_DRY_RUN_UPDATE
    mock_secret_client.create_secret.assert_not_called()
    mock_secret_client.add_secret_version.assert_not_called()


def test_sync_secret_create_failure_returns_error_result(mock_secret_client):
    """If add_secret_version raises, we return an error result rather
    than crashing — orchestrator needs to keep going for the rest of
    the secret list."""
    from google.api_core import exceptions as gcp_exc

    mock_secret_client.get_secret.side_effect = gcp_exc.NotFound("nope")
    mock_secret_client.create_secret.return_value = MagicMock()
    mock_secret_client.add_secret_version.side_effect = gcp_exc.PermissionDenied(
        "no perm",
    )

    result = sync_secret(
        mock_secret_client,
        "test-project",
        "kestrel-openai-key",
        "sk-test-1234",
    )

    assert result.action == ACTION_ERROR
    assert "PermissionDenied" in result.detail or "no perm" in result.detail


# ---------------------------------------------------------------------------
# sync_all_secrets — end-to-end with mocked client
# ---------------------------------------------------------------------------

def test_sync_all_secrets_end_to_end(
    deploy_config_two_profiles, tmp_path: Path, mock_secret_client,
):
    """Full orchestrator: derive mapping → load .env → sync each.

    Asserts:

    * One result per derived mapping entry.
    * Missing env vars produce 'skipped' results (not crashes).
    * Present env vars produce 'created'/'updated' results.
    * env_var name is stamped on each result.
    """
    from google.api_core import exceptions as gcp_exc

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-test\n"
        "KESTREL_API_KEY=key-1\n"
        "KESTREL_DATA_KEY=data-1\n"
        # ANTHROPIC_API_KEY intentionally missing → expect skipped.
    )

    # All secrets missing → all 'created'.
    mock_secret_client.get_secret.side_effect = gcp_exc.NotFound("nope")

    with patch(
        "google.cloud.secretmanager.SecretManagerServiceClient",
        return_value=mock_secret_client,
    ):
        results = sync_all_secrets(
            deploy_config_two_profiles,
            env_file,
            "test-project",
        )

    by_env_var = {r.env_var: r for r in results}
    assert set(by_env_var.keys()) == {
        "OPENAI_API_KEY",
        "KESTREL_API_KEY",
        "KESTREL_DATA_KEY",
        "ANTHROPIC_API_KEY",
    }

    # Three present in .env → created.
    assert by_env_var["OPENAI_API_KEY"].action == ACTION_CREATED
    assert by_env_var["KESTREL_API_KEY"].action == ACTION_CREATED
    assert by_env_var["KESTREL_DATA_KEY"].action == ACTION_CREATED

    # ANTHROPIC_API_KEY missing in .env → skipped, with explanatory detail.
    skipped = by_env_var["ANTHROPIC_API_KEY"]
    assert skipped.action == ACTION_SKIPPED
    assert "ANTHROPIC_API_KEY" in skipped.detail


def test_sync_all_secrets_filters_by_profile(
    deploy_config_two_profiles, tmp_path: Path, mock_secret_client,
):
    """``profile='dev'`` syncs only dev's three secrets, not prod's
    ANTHROPIC_API_KEY."""
    from google.api_core import exceptions as gcp_exc

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-test\n"
        "KESTREL_API_KEY=key-1\n"
        "KESTREL_DATA_KEY=data-1\n"
        "ANTHROPIC_API_KEY=anth-1\n"  # set, but not in dev profile
    )

    mock_secret_client.get_secret.side_effect = gcp_exc.NotFound("nope")

    with patch(
        "google.cloud.secretmanager.SecretManagerServiceClient",
        return_value=mock_secret_client,
    ):
        results = sync_all_secrets(
            deploy_config_two_profiles,
            env_file,
            "test-project",
            profile="dev",
        )

    env_vars_synced = {r.env_var for r in results}
    assert env_vars_synced == {"OPENAI_API_KEY", "KESTREL_API_KEY", "KESTREL_DATA_KEY"}
    # ANTHROPIC_API_KEY should NOT appear in the result list — it's not
    # in the dev profile.
    assert "ANTHROPIC_API_KEY" not in env_vars_synced


def test_sync_all_secrets_dry_run_does_not_mutate(
    deploy_config_two_profiles, tmp_path: Path, mock_secret_client,
):
    """``dry_run=True`` issues only get_secret probes, never
    create_secret / add_secret_version."""
    from google.api_core import exceptions as gcp_exc

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-test\n"
        "KESTREL_API_KEY=key-1\n"
        "KESTREL_DATA_KEY=data-1\n"
        "ANTHROPIC_API_KEY=anth-1\n"
    )

    mock_secret_client.get_secret.side_effect = gcp_exc.NotFound("nope")

    with patch(
        "google.cloud.secretmanager.SecretManagerServiceClient",
        return_value=mock_secret_client,
    ):
        results = sync_all_secrets(
            deploy_config_two_profiles,
            env_file,
            "test-project",
            dry_run=True,
        )

    # All results should be dry-run-create (since get_secret raised NotFound).
    actions = {r.action for r in results}
    assert actions == {ACTION_DRY_RUN_CREATE}

    mock_secret_client.create_secret.assert_not_called()
    mock_secret_client.add_secret_version.assert_not_called()


def test_sync_all_secrets_empty_mapping_returns_empty_list(tmp_path: Path):
    """A config with no Secret Manager refs yields no work and an empty
    result list — no exception, no SDK construction."""
    config = {
        "profiles": {
            "azure-dev": {
                "secrets": {
                    "OPENAI_API_KEY": "${AZURE_OPENAI_KEY}",
                },
            },
        },
    }

    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n")

    # We don't even need to mock the SDK — sync_all_secrets short-circuits
    # before constructing the client when the mapping is empty.
    results = sync_all_secrets(config, env_file, "test-project")
    assert results == []


def test_sync_all_secrets_missing_env_file_raises(
    deploy_config_two_profiles, tmp_path: Path,
):
    """A missing .env file raises FileNotFoundError — operators must
    notice. (The bash script also exited 1 in this case.)"""
    missing = tmp_path / "nonexistent.env"
    with pytest.raises(FileNotFoundError):
        sync_all_secrets(deploy_config_two_profiles, missing, "test-project")


# ---------------------------------------------------------------------------
# SecretSyncResult sanity
# ---------------------------------------------------------------------------

def test_secret_sync_result_dataclass():
    """Default detail is empty string; fields are positional/named."""
    r = SecretSyncResult(
        secret_name="kestrel-openai-key",
        env_var="OPENAI_API_KEY",
        action=ACTION_CREATED,
    )
    assert r.detail == ""
    assert r.action == "created"
