"""Drift guard for the live ``deploy_config.toml``.

Epic #1050 sub-PR 1.4 (``epic/tier-1.4-cutover``) reconciled the live
``deploy_config.toml`` with the legacy ``scripts/cloudrun/*.sh`` flag
sets — adding the missing OAuth/GitHub-App secrets to ``[profiles.dev]``,
the ``KESTREL_MULTI_AGENT``/``KESTREL_REQUIRE_OAUTH`` env vars to
``[profiles.prod]``, the OAuth secrets to ``[profiles.multi-agent-dev]``,
and resolving the ``kestrel-multi_agent`` (underscore) image-name
divergence by switching ``[profiles.dev]`` to
``deployment_mode = "multi_agent"``.

This module asserts each canonical profile has the secret keys + env
keys it needs to be a Cloud Run deploy at parity with the bash legacy.
We do **not** pin specific Secret Manager versions or values — those
evolve. Key presence is the contract.

The goal is to catch future drift early: if someone adds a new agent
feature that needs a Cloud Run secret and forgets to wire it through
to ``[profiles.dev.secrets]``, the deploy would silently start running
without that secret. A failing assertion here blocks the merge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel_sovereign.features.deploy.manager import DeployManager
from kestrel_sovereign.features.deploy.models import DeployProviderType


# Repo root: tests/unit/<this_file> -> ../../ -> repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def live_config():
    """Parse the actual checked-in ``deploy_config.toml``.

    We don't construct ``DeployManager()`` from process-cwd — the test
    runner's cwd may differ from the repo root depending on how pytest
    was launched. Loading the file by absolute path keeps the fixture
    deterministic regardless of invocation.
    """
    import toml

    config_path = REPO_ROOT / "deploy_config.toml"
    assert config_path.exists(), (
        f"deploy_config.toml not found at {config_path}; this test loads "
        "the live file as a drift guard against the bash legacy."
    )
    return toml.load(config_path)


def test_live_config_loads_into_manager(live_config):
    """Every profile in the checked-in TOML must parse into a profile.

    Loading via :class:`DeployManager` exercises the full config →
    profile pipeline (including ``_expand_env_vars`` and the
    ``deployment_mode`` → dockerfile mapping) and would surface any
    missing required field as a parse error / dropped profile.
    """
    manager = DeployManager(config=live_config)

    expected_profiles = {
        "dev",
        "prod",
        "multi-agent-dev",
        "multi-agent-prod",
        "azure-multi-agent-dev",
        "azure-dev",
    }
    actual_profiles = set(manager.profiles.keys())

    missing = expected_profiles - actual_profiles
    assert not missing, (
        f"Profiles missing from manager.profiles after loading "
        f"deploy_config.toml: {sorted(missing)}. Did the TOML lose a "
        f"required field (service_name / region) for one of them?"
    )


def test_dev_profile_has_bash_parity_secrets(live_config):
    """``[profiles.dev.secrets]`` must cover what deploy_dev.sh did.

    The legacy bash script set 11 secret refs. Sub-PR 1.4 added the
    OAuth + GitHub-App secrets that the original TOML was missing;
    this assertion locks them in.
    """
    manager = DeployManager(config=live_config)
    dev = manager.get_profile("dev")

    expected = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "KESTREL_API_KEY",
        "KESTREL_DATA_KEY",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "KESTREL_SESSION_SECRET",
        "GITHUB_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY",
        "GITHUB_APP_WEBHOOK_SECRET",
    }
    actual = set(dev.secrets.keys())

    missing = expected - actual
    assert not missing, (
        f"[profiles.dev.secrets] missing keys vs legacy "
        f"scripts/cloudrun/deploy_dev.sh: {sorted(missing)}."
    )


def test_dev_profile_has_bash_parity_env_vars(live_config):
    """``[profiles.dev.env_vars]`` must include the runtime knobs the
    legacy ``deploy_dev.sh`` set on the Cloud Run service."""
    manager = DeployManager(config=live_config)
    dev = manager.get_profile("dev")

    expected = {
        "KESTREL_ENV",
        "KESTREL_DB_BACKEND",
        "KESTREL_DB_PATH",
        "KESTREL_AGENTS",
        "KESTREL_REQUIRE_OAUTH",
        "KESTREL_ALLOWED_EMAILS",
    }
    actual = set(dev.env_vars.keys())

    missing = expected - actual
    assert not missing, (
        f"[profiles.dev.env_vars] missing keys vs legacy "
        f"scripts/cloudrun/deploy_dev.sh: {sorted(missing)}."
    )


def test_dev_profile_uses_multi_agent_image(live_config):
    """The dev service runs the multi-agent host (Kestrel + kestrel-demo).

    With ``deployment_mode = "multi_agent"``,
    :meth:`DeployManagerCore.build_image_reference` resolves to
    ``gcr.io/<project>/kestrel-multi_agent:<tag>`` — matching what
    ``kestrel deploy build`` and ``.github/workflows/deploy.yml`` push.
    """
    manager = DeployManager(config=live_config)
    dev = manager.get_profile("dev")

    assert dev.is_multi_agent, (
        "deploy_config.toml [profiles.dev] must set "
        "deployment_mode = \"multi_agent\" so kestrel deploy dev pushes "
        "the kestrel-multi_agent image (matches build.sh / deploy.yml)."
    )

    ref = manager.build_image_reference("dev", "v1.2.3")
    assert ref.endswith("/kestrel-multi_agent:v1.2.3"), (
        f"Expected dev image ref to end with /kestrel-multi_agent:v1.2.3, "
        f"got {ref!r}. build_image_reference may have lost the multi_agent "
        f"suffix logic."
    )


def test_prod_profile_has_bash_parity_secrets(live_config):
    """``[profiles.prod.secrets]`` must cover the OAuth secrets the
    legacy ``deploy_prod.sh`` set."""
    manager = DeployManager(config=live_config)
    prod = manager.get_profile("prod")

    expected = {
        "OPENAI_API_KEY",
        "KESTREL_API_KEY",
        "KESTREL_DATA_KEY",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "KESTREL_SESSION_SECRET",
    }
    actual = set(prod.secrets.keys())

    missing = expected - actual
    assert not missing, (
        f"[profiles.prod.secrets] missing keys vs legacy "
        f"scripts/cloudrun/deploy_prod.sh: {sorted(missing)}."
    )


def test_prod_profile_has_bash_parity_env_vars(live_config):
    """``[profiles.prod.env_vars]`` must include ``KESTREL_MULTI_AGENT``,
    ``KESTREL_REQUIRE_OAUTH``, and ``KESTREL_ALLOWED_EMAILS`` —
    enabling the multi-agent host inside the single-agent image (the
    legacy ``deploy_prod.sh`` shape)."""
    manager = DeployManager(config=live_config)
    prod = manager.get_profile("prod")

    expected = {
        "KESTREL_ENV",
        "KESTREL_DB_BACKEND",
        "KESTREL_DB_PATH",
        "KESTREL_MULTI_AGENT",
        "KESTREL_REQUIRE_OAUTH",
        "KESTREL_ALLOWED_EMAILS",
    }
    actual = set(prod.env_vars.keys())

    missing = expected - actual
    assert not missing, (
        f"[profiles.prod.env_vars] missing keys vs legacy "
        f"scripts/cloudrun/deploy_prod.sh: {sorted(missing)}."
    )


def test_prod_profile_uses_single_agent_image(live_config):
    """Prod uses the single-agent ``kestrel`` image (matches deploy_prod.sh
    IMAGE_NAME=kestrel) but runs the host via ``KESTREL_MULTI_AGENT=true``
    in env."""
    manager = DeployManager(config=live_config)
    prod = manager.get_profile("prod")

    assert not prod.is_multi_agent, (
        "deploy_config.toml [profiles.prod] must NOT set "
        "deployment_mode = \"multi_agent\" — prod uses the single-agent "
        "kestrel image with KESTREL_MULTI_AGENT=true in env."
    )

    ref = manager.build_image_reference("prod", "v1.2.3")
    assert ref.endswith("/kestrel:v1.2.3"), (
        f"Expected prod image ref to end with /kestrel:v1.2.3, got {ref!r}."
    )


def test_multi_agent_dev_profile_has_bash_parity_secrets(live_config):
    """``[profiles.multi-agent-dev.secrets]`` must cover the OAuth
    secrets the legacy ``deploy_multi_agent_dev.sh`` set."""
    manager = DeployManager(config=live_config)
    profile = manager.get_profile("multi-agent-dev")

    expected = {
        "OPENAI_API_KEY",
        "KESTREL_API_KEY",
        "KESTREL_DATA_KEY",
        "LIGHTHOUSE_API_KEY",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "KESTREL_SESSION_SECRET",
    }
    actual = set(profile.secrets.keys())

    missing = expected - actual
    assert not missing, (
        f"[profiles.multi-agent-dev.secrets] missing keys vs legacy "
        f"scripts/cloudrun/deploy_multi_agent_dev.sh: {sorted(missing)}."
    )


def test_multi_agent_dev_profile_has_allowed_emails(live_config):
    """The multi-agent-dev profile gates access by email like dev/prod."""
    manager = DeployManager(config=live_config)
    profile = manager.get_profile("multi-agent-dev")

    assert "KESTREL_ALLOWED_EMAILS" in profile.env_vars, (
        "[profiles.multi-agent-dev.env_vars] must include "
        "KESTREL_ALLOWED_EMAILS to gate Cloud Run access by Google account."
    )


def test_all_cloudrun_profiles_target_same_provider(live_config):
    """Every Cloud Run profile must declare provider=cloudrun (not
    accidentally land as Azure / unknown)."""
    manager = DeployManager(config=live_config)

    for name in ("dev", "prod", "multi-agent-dev", "multi-agent-prod"):
        profile = manager.get_profile(name)
        assert profile.provider == DeployProviderType.CLOUD_RUN, (
            f"[profiles.{name}] must use provider = \"cloudrun\"; got "
            f"{profile.provider}."
        )


def test_kestrel_allowed_emails_expansion(live_config, monkeypatch):
    """``${KESTREL_ALLOWED_EMAILS}`` must round-trip through
    :meth:`DeployManagerCore._expand_env_vars` — i.e. the operator's
    shell-exported value lands on the deployed service.

    The bash legacy used ``${KESTREL_ALLOWED_EMAILS:?...}``; the TOML
    placeholder is the parameter-expansion-friendly equivalent.
    """
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", "alice@example.com,bob@example.com")

    manager = DeployManager(config=live_config)
    dev = manager.get_profile("dev")

    assert dev.env_vars["KESTREL_ALLOWED_EMAILS"] == "alice@example.com,bob@example.com", (
        f"Expected ${{KESTREL_ALLOWED_EMAILS}} to expand to the env-var "
        f"value; got {dev.env_vars['KESTREL_ALLOWED_EMAILS']!r}. The "
        f"DeployManagerCore._expand_env_vars contract may have regressed."
    )
