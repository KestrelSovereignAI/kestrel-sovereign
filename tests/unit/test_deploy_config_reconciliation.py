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
    """Prod keeps OAuth secrets and adds pinned durable custody."""
    manager = DeployManager(config=live_config)
    prod = manager.get_profile("prod")

    expected = {
        "OPENAI_API_KEY",
        "KESTREL_API_KEY",
        "KESTREL_DATABASE_URL",
        "KESTREL_HOLD_EVIDENCE_DATABASE_URL",
        "KESTREL_DATA_KEY",
        "KESTREL_IDENTITY_BUNDLE",
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
    """Prod declares the durable single-agent runtime contract."""
    manager = DeployManager(config=live_config)
    prod = manager.get_profile("prod")

    expected = {
        "KESTREL_ENV",
        "KESTREL_DB_BACKEND",
        "KESTREL_DB_PATH",
        "KESTREL_DEPLOYMENT_PERSISTENCE",
        "KESTREL_EXPECTED_DID",
        "KESTREL_REQUIRE_OAUTH",
        "KESTREL_ALLOWED_EMAILS",
    }
    actual = set(prod.env_vars.keys())

    missing = expected - actual
    assert not missing, (
        f"[profiles.prod.env_vars] missing keys vs legacy "
        f"scripts/cloudrun/deploy_prod.sh: {sorted(missing)}."
    )
    assert "KESTREL_MULTI_AGENT" not in actual
    assert prod.env_vars["KESTREL_DB_BACKEND"] == "postgres"


def test_prod_profile_uses_single_agent_image(live_config):
    """Durable prod uses the single-agent ``kestrel`` image."""
    manager = DeployManager(config=live_config)
    prod = manager.get_profile("prod")

    assert not prod.is_multi_agent, (
        "deploy_config.toml [profiles.prod] must NOT set "
        "deployment_mode = \"multi_agent\" — durable multi-agent custody "
        "is not implemented."
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


def test_cloudrun_profiles_declare_honest_persistence(live_config):
    """Live Cloud Run profiles cannot imply durable local container state."""
    from kestrel_sovereign.features.deploy.persistence import (
        validate_cloudrun_persistence,
    )
    from kestrel_sovereign.features.deploy.models import DeployManagerError

    manager = DeployManager(config=live_config)
    for name in ("dev", "multi-agent-dev"):
        profile = manager.get_profile(name)
        assert profile.is_ephemeral_demo
        assert profile.max_instances == 1
        validate_cloudrun_persistence(profile)

    prod = manager.get_profile("prod")
    assert prod.is_durable_sovereign
    assert prod.env_vars["KESTREL_DB_BACKEND"] == "postgres"
    assert not prod.is_multi_agent
    for key in (
        "KESTREL_DATABASE_URL",
        "KESTREL_HOLD_EVIDENCE_DATABASE_URL",
        "KESTREL_DATA_KEY",
        "KESTREL_IDENTITY_BUNDLE",
    ):
        name, version = prod.secrets[key].rsplit(":", 1)
        assert name
        assert version.isdigit(), f"{key} must use an immutable numeric version"

    # The checked-in value is an operator placeholder until deploy time.
    assert prod.env_vars["KESTREL_EXPECTED_DID"] == "${KESTREL_PROD_EXPECTED_DID}"

    multi_prod = manager.get_profile("multi-agent-prod")
    with pytest.raises(DeployManagerError, match="single-agent profiles only"):
        validate_cloudrun_persistence(multi_prod)


def test_production_ceremony_provisions_only_the_cluster_probe_privilege():
    """Least-privilege runtime roles can read the Hold cluster identity."""

    deployment_guide = (REPO_ROOT / "docs/deployment/README.md").read_text()

    assert (
        "GRANT EXECUTE ON FUNCTION pg_catalog.pg_control_system()"
        in deployment_guide
    )
    assert "both PostgreSQL runtime roles" in deployment_guide
    assert "GRANT pg_monitor" not in deployment_guide


def test_prod_instance_cap_matches_provisioned_database(live_config):
    """``prod``'s scaling numbers must match the substrate behind them.

    ``durable_sovereign`` *permits* horizontal scale, but permission is not
    capacity.  Each serving instance opens up to ``max_pool_size`` (10) pooled
    plus ``_advisory_max_pool_size`` (4) runtime PostgreSQL connections and one
    serialized Hold connection on the primary database. The provisioned Cloud
    SQL instance is a ``db-f1-micro`` with a ~25 connection ceiling — so a
    second instance exhausts it.

    The floor is the same argument read the other way: scaling to zero is safe
    only because custody is durable.  A cold start restores the pinned bundle
    from Secret Manager and reads coherent state from PostgreSQL, so it costs a
    slower first request and nothing else.  Under ``ephemeral_demo`` the same
    value would silently discard the identity, which is why the two are
    asserted together rather than as independent numbers.

    This is deliberately a config assertion rather than a runtime check: the
    connection ceiling is a property of the provisioned database, which the
    profile cannot introspect.  Raising the cap is a paired change with the
    database tier, and this test is what makes the pairing fail loudly rather
    than silently at scale-up.
    """
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    manager = DeployManager(config=live_config)
    prod = manager.get_profile("prod")

    assert prod.max_instances == 1, (
        "prod max_instances was raised without raising the Cloud SQL tier; "
        "see the comment above [profiles.prod] in deploy_config.toml"
    )
    assert prod.min_instances == 0 and prod.is_durable_sovereign, (
        "prod scales to zero, which is only safe under durable custody; if "
        "the persistence mode changes, the floor must be revisited with it"
    )

    # Pin the per-instance connection cost the cap is derived from, so a change
    # to pool sizing surfaces here rather than as exhaustion in production.
    backend = PostgresBackend(dsn="postgresql://u:p@127.0.0.1:5432/db")
    hold_primary_pool_size = 1
    per_instance = (
        backend._max_pool_size
        + backend._advisory_max_pool_size
        + hold_primary_pool_size
    )
    assert per_instance <= 25
    assert per_instance * 2 > 25, (
        "two instances no longer exhaust a db-f1-micro; re-derive the cap "
        "instead of assuming this rationale still holds"
    )


@pytest.mark.asyncio
async def test_prod_deploy_accepts_resolved_durable_custody(
    live_config, monkeypatch
):
    """A resolved DID plus pinned custody reaches the provider offline."""
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", "ops@example.com")
    monkeypatch.setenv(
        "KESTREL_PROD_EXPECTED_DID",
        "did:web:agents.kestrelsovereign.com:kestrel",
    )
    manager = DeployManager(config=live_config)
    stub = _install_offline_stubs(manager, monkeypatch)

    result = await manager.deploy_profile("prod", tag="v1.2.3")

    assert result["success"] is True
    assert len(stub.deploy_calls) == 1
    deployed = stub.deploy_calls[0]["profile"]
    assert deployed.env_vars["KESTREL_EXPECTED_DID"].startswith("did:web:")


@pytest.mark.asyncio
async def test_manager_rejects_unsafe_local_cloudrun_before_provider(monkeypatch):
    """Manager-level deployment fails before a disposable profile is sent."""
    config = {
        "manager": {"gcp_project_id": "test-project"},
        "profiles": {
            "unsafe": {
                "provider": "cloudrun",
                "service_name": "unsafe-prod",
                "region": "us-central1",
                "min_instances": 1,
                "max_instances": 10,
                "env_vars": {
                    "KESTREL_ENV": "production",
                    "KESTREL_DB_BACKEND": "sqlite",
                },
            }
        },
    }
    manager = DeployManager(config=config)
    stub = _install_offline_stubs(manager, monkeypatch)

    result = await manager.deploy_profile("unsafe", tag="immutable")

    assert result["success"] is False
    assert "explicitly set persistence_mode" in result["error"]
    assert stub.deploy_calls == []


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


def test_build_image_reference_uses_profile_scoped_project(live_config):
    """Codex review on the final epic→main PR: when a profile sets its
    own ``gcp_project_id``, ``build_image_reference`` must use THAT
    project, not the manager-level value. Otherwise a config with a
    profile-scoped override silently builds the wrong image ref.

    This synthetic config has manager=alpha, dev=alpha, prod=beta —
    the prod image ref must point at beta.
    """
    config = {
        "manager": {"gcp_project_id": "alpha-project", "image_name": "kestrel"},
        "profiles": {
            "dev": {
                "provider": "cloudrun",
                "service_name": "kestrel-dev",
                "region": "us-central1",
                "gcp_project_id": "alpha-project",
            },
            "prod": {
                "provider": "cloudrun",
                "service_name": "kestrel-prod",
                "region": "us-central1",
                "gcp_project_id": "beta-project",
            },
        },
    }
    manager = DeployManager(config=config)

    assert (
        manager.build_image_reference("dev", "v1")
        == "gcr.io/alpha-project/kestrel:v1"
    )
    assert (
        manager.build_image_reference("prod", "v1")
        == "gcr.io/beta-project/kestrel:v1"
    ), "prod profile's gcp_project_id must override the manager-level value"


def test_get_provider_caches_per_profile_project(live_config):
    """Two profiles targeting different GCP projects each get their
    own CloudRunProvider — not a single one bound to whichever profile
    constructed first. Codex review caught the cache-shared-state bug
    on the final epic→main PR.
    """
    config = {
        "manager": {"gcp_project_id": "alpha-project"},
        "profiles": {
            "dev": {
                "provider": "cloudrun",
                "service_name": "kestrel-dev",
                "region": "us-central1",
                "gcp_project_id": "alpha-project",
            },
            "prod": {
                "provider": "cloudrun",
                "service_name": "kestrel-prod",
                "region": "us-central1",
                "gcp_project_id": "beta-project",
            },
        },
    }
    manager = DeployManager(config=config)

    # Patch CloudRunProvider so we don't try to import google-cloud-run.
    constructed: list = []

    class _StubProvider:
        def __init__(self, project_id):
            self.project_id = project_id
            constructed.append(project_id)

    from kestrel_sovereign.features.deploy import core as core_mod

    real_cls = core_mod.CloudRunProvider
    core_mod.CloudRunProvider = _StubProvider
    try:
        p_alpha = manager._get_provider(
            DeployProviderType.CLOUD_RUN, gcp_project_id="alpha-project"
        )
        p_beta = manager._get_provider(
            DeployProviderType.CLOUD_RUN, gcp_project_id="beta-project"
        )
    finally:
        core_mod.CloudRunProvider = real_cls

    # Two separate provider instances, one per project.
    assert p_alpha is not p_beta
    assert p_alpha.project_id == "alpha-project"
    assert p_beta.project_id == "beta-project"
    # Cache hit: re-asking for alpha returns the same instance.
    p_alpha_again = manager._get_provider(
        DeployProviderType.CLOUD_RUN, gcp_project_id="alpha-project"
    )
    # The cache holds the STUBBED instance from above, so we re-stub
    # to assert hit behavior.
    core_mod.CloudRunProvider = _StubProvider
    try:
        p_alpha_again2 = manager._get_provider(
            DeployProviderType.CLOUD_RUN, gcp_project_id="alpha-project"
        )
    finally:
        core_mod.CloudRunProvider = real_cls
    assert p_alpha_again is p_alpha_again2


@pytest.mark.asyncio
async def test_deploy_profile_rejects_empty_string_placeholder(
    live_config, monkeypatch
):
    """Codex review on the final epic→main PR: when ``${VAR}`` resolves
    to an empty string (env var SET but blank — common with unset
    GitHub secrets that become ``""`` in step env), the previous
    validator missed it because the literal ``${`` wasn't in the
    expanded value. Result: ``kestrel deploy dev`` would deploy with
    OAuth enabled but no allowed emails, locking everyone out.

    Bash ``${VAR:?...}`` errored on EITHER unset OR empty; the new
    validator does too.
    """
    from kestrel_sovereign.features.deploy.models import DeployManagerError

    # Set the env var to empty string — the substitution produces "".
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", "")
    manager = DeployManager(config=live_config)

    result = await manager.deploy_profile("dev")

    assert result["success"] is False
    msg = result["error"]
    assert "empty" in msg or "unresolved" in msg
    assert "KESTREL_ALLOWED_EMAILS" in msg


@pytest.mark.asyncio
async def test_deploy_profile_rejects_latest_tag(live_config, monkeypatch):
    """#1441: ``kestrel deploy <profile>`` defaulted ``tag="latest"``,
    and Cloud Run Admin v2 ``update_service`` compares templates as
    strings — so deploying ``:latest`` after a fresh build silently
    no-ops because the existing service template already references
    ``:latest``. Every CI deploy since the bash retirement (epic
    #1050 sub-PR 1.4) was a phantom rollout. Manager-level guard
    refuses moving-alias tags; CI workflow now passes the build's
    resolved per-invocation tag (``v0.15.1`` or ``dev-<7sha>``)
    explicitly via ``--tag``.
    """
    # Satisfy the placeholder validator first so we reach the tag guard.
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", "ops@example.com")
    manager = DeployManager(config=live_config)

    result = await manager.deploy_profile("dev", tag="latest")

    assert result["success"] is False
    err = result["error"]
    assert "Refusing to deploy 'dev'" in err
    assert "'latest'" in err
    assert "#1441" in err
    assert "--tag" in err


@pytest.mark.asyncio
async def test_deploy_profile_rejects_empty_tag(live_config, monkeypatch):
    """An empty tag would resolve to ``gcr.io/.../image:`` — invalid as
    a Docker reference and another silent failure mode. Treated the
    same as ``:latest`` by the moving-alias guard."""
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", "ops@example.com")
    manager = DeployManager(config=live_config)

    result = await manager.deploy_profile("dev", tag="")

    assert result["success"] is False
    assert "Refusing to deploy 'dev'" in result["error"]


class _StubProvider:
    """Records the image/profile passed to ``deploy`` without contacting
    any cloud. Used to assert the moving-alias guard's pass/fail
    behavior without ever reaching real provider code, even when GCP
    or Azure credentials happen to be present in the test env."""

    def __init__(self):
        self.deploy_calls: list[dict] = []

    async def deploy(self, *, image, service_name, profile):
        self.deploy_calls.append(
            {"image": image, "service_name": service_name, "profile": profile}
        )
        return {"service_url": "https://stub", "revision": "stub-rev-1"}


def _install_offline_stubs(manager, monkeypatch) -> _StubProvider:
    """Replace the provider AND the post-deploy health check so the
    test cannot reach any real cloud or HTTP endpoint."""
    stub = _StubProvider()
    monkeypatch.setattr(manager, "_get_provider", lambda *a, **kw: stub)

    async def _fake_health(_url, *_args, **_kw):
        return True

    monkeypatch.setattr(manager, "_verify_health", _fake_health)
    return stub


@pytest.mark.asyncio
async def test_deploy_profile_accepts_concrete_version_tag(
    live_config, monkeypatch
):
    """Concrete tags (``v0.15.1``, ``dev-abc1234``) pass the guard and
    reach the provider with the expected image string. Provider AND
    health check are stubbed — this test must never contact a real
    cloud or HTTP endpoint, even when credentials are available."""
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", "ops@example.com")
    manager = DeployManager(config=live_config)
    stub = _install_offline_stubs(manager, monkeypatch)

    result = await manager.deploy_profile("dev", tag="v0.15.1")

    assert result["success"] is True
    assert len(stub.deploy_calls) == 1
    assert stub.deploy_calls[0]["image"].endswith(":v0.15.1"), (
        f"unexpected image ref: {stub.deploy_calls[0]['image']}"
    )


@pytest.mark.asyncio
async def test_deploy_profile_latest_tag_allowed_on_azure(
    live_config, monkeypatch
):
    """Codex review caught: the moving-alias guard is Cloud Run-specific
    (Admin v2 ``update_service`` compares image strings; #1441). Azure
    Container Apps deploys are not known to share this bug, so the
    guard must NOT block ``kestrel deploy azure-dev`` with the CLI
    default tag — that would be a brand-new regression for an
    unrelated provider. Provider + health check stubbed to keep this
    fully offline."""
    for var in (
        "OPENAI_API_KEY", "KESTREL_API_KEY", "KESTREL_DATA_KEY",
        "LIGHTHOUSE_API_KEY",
    ):
        monkeypatch.setenv(var, "test-placeholder")
    manager = DeployManager(config=live_config)
    stub = _install_offline_stubs(manager, monkeypatch)

    result = await manager.deploy_profile("azure-dev", tag="latest")

    # Guard must not fire for non-Cloud-Run providers.
    assert result["success"] is True
    assert len(stub.deploy_calls) == 1
    assert stub.deploy_calls[0]["image"].endswith(":latest")


@pytest.mark.asyncio
async def test_deploy_profile_rejects_unresolved_placeholders(
    live_config, monkeypatch
):
    """Codex review on PR #1064: when ``${KESTREL_ALLOWED_EMAILS}`` is
    unset in the runtime env, ``_expand_env_vars`` leaves the literal
    placeholder in the value. The bash scripts errored on missing env
    via ``${VAR:?...}``; ``deploy_profile`` must mirror that — refuse
    to deploy with broken config rather than push the literal
    ``${...}`` to Cloud Run.
    """
    from kestrel_sovereign.features.deploy.models import DeployManagerError

    monkeypatch.delenv("KESTREL_ALLOWED_EMAILS", raising=False)
    manager = DeployManager(config=live_config)

    # deploy_profile catches DeployManagerError and returns it in the
    # ``error`` field of the result dict (matches the shape both the
    # agent !deploy tool and the kestrel deploy CLI render).
    result = await manager.deploy_profile("dev")

    assert result["success"] is False
    assert "unresolved" in result["error"]
    assert "KESTREL_ALLOWED_EMAILS" in result["error"]
    assert "kestrel deploy dev" in result["error"]
