"""Agent-bound identity export placement regressions for #2604."""

from __future__ import annotations

from pathlib import Path

from kestrel_sovereign.identity.protected_export import (
    configured_identity_export_roots,
    effective_identity_export_roots,
    identity_export_directory,
)
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.setup.env_file import write_env


def test_runtime_agent_data_directory_is_the_default(tmp_path):
    agent_root = tmp_path / "agent_data" / "claw"

    resolved = identity_export_directory(agent_data_dir=agent_root, env={})

    assert resolved == agent_root.resolve()
    assert resolved.is_absolute()


def test_per_agent_override_wins_over_process_environment(tmp_path):
    agent_root = tmp_path / "agent_data" / "claw"
    override = agent_root / "continuity"

    resolved = identity_export_directory(
        agent_data_dir=agent_root,
        per_agent_override=override,
        env={"KESTREL_DATA_DIR": str(tmp_path / "shared")},
    )

    assert resolved == override.resolve()


def test_intentional_environment_override_wins_without_agent_config(tmp_path):
    override = tmp_path / "operator-exports"

    resolved = identity_export_directory(
        agent_data_dir=tmp_path / "agent_data" / "claw",
        env={"KESTREL_DATA_DIR": str(override)},
    )

    assert resolved == override.resolve()


def test_process_managed_binding_wins_over_legacy_environment_override(tmp_path):
    child_root = tmp_path / "agent_data" / "claw"

    resolved = identity_export_directory(
        env={
            "KESTREL_IDENTITY_EXPORT_DIR": str(child_root),
            "KESTREL_DATA_DIR": str(tmp_path / "shared-parent-root"),
        },
    )

    assert resolved == child_root.resolve()


def test_direct_single_agent_falls_back_to_kestrel_db_path(tmp_path):
    agent_root = tmp_path / "single-agent"

    resolved = identity_export_directory(
        env={"KESTREL_DB_PATH": str(agent_root)},
    )

    assert resolved == agent_root.resolve()


def test_legacy_default_is_absolute_and_cwd_bound(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = identity_export_directory(env={})

    assert resolved == (tmp_path / "agent_data").resolve()
    assert resolved.is_absolute()


def test_doctor_roots_include_each_agent_and_configured_override(tmp_path):
    config = MultiAgentConfig(
        agents={
            "claw": LocalAgentConfig(
                data_dir="agent_data/claw",
                identity_export_dir="continuity",
                port=8801,
            ),
            "emma": LocalAgentConfig(
                data_dir="agent_data/emma",
                port=8802,
            ),
        }
    )
    config.save(tmp_path / "multi_agent.toml")

    roots = configured_identity_export_roots(tmp_path, env={})

    assert (tmp_path / "agent_data" / "claw").resolve() in roots
    assert (tmp_path / "agent_data" / "claw" / "continuity").resolve() in roots
    assert (tmp_path / "agent_data" / "emma").resolve() in roots


def test_effective_roots_use_dotenv_when_process_path_is_unset(tmp_path):
    write_env(tmp_path / ".env", {"KESTREL_DATA_DIR": "dotenv-exports"})

    roots = effective_identity_export_roots(tmp_path, process_env={})

    assert (tmp_path / "dotenv-exports").resolve() in roots


def test_effective_roots_use_process_environment_without_dotenv(tmp_path):
    roots = effective_identity_export_roots(
        tmp_path,
        process_env={"KESTREL_DATA_DIR": "live-exports"},
    )

    assert (tmp_path / "live-exports").resolve() in roots


def test_effective_roots_process_environment_overrides_dotenv(tmp_path):
    write_env(tmp_path / ".env", {"KESTREL_DATA_DIR": "dotenv-exports"})

    roots = effective_identity_export_roots(
        tmp_path,
        process_env={"KESTREL_DATA_DIR": "live-exports"},
    )

    assert (tmp_path / "live-exports").resolve() in roots
    assert (tmp_path / "dotenv-exports").resolve() not in roots


def test_effective_roots_empty_process_value_suppresses_dotenv(tmp_path):
    write_env(tmp_path / ".env", {"KESTREL_DATA_DIR": "dotenv-exports"})

    roots = effective_identity_export_roots(
        tmp_path,
        process_env={"KESTREL_DATA_DIR": ""},
    )

    assert roots == ((tmp_path / "agent_data").resolve(),)


def test_effective_roots_support_legacy_agent_data_dir(tmp_path):
    roots = effective_identity_export_roots(
        tmp_path,
        process_env={"AGENT_DATA_DIR": "legacy-exports"},
    )

    assert (tmp_path / "legacy-exports").resolve() in roots


def test_effective_roots_unset_case_is_the_project_default(tmp_path):
    roots = effective_identity_export_roots(tmp_path, process_env={})

    assert roots == ((tmp_path / "agent_data").resolve(),)


def test_effective_roots_resolve_relative_values_against_absolute_project(tmp_path):
    nested_project = tmp_path / "nested" / "project"
    nested_project.mkdir(parents=True)

    roots = effective_identity_export_roots(
        nested_project,
        process_env={"KESTREL_IDENTITY_EXPORT_DIR": "relative/exports"},
    )

    assert (nested_project / "relative" / "exports").resolve() in roots
