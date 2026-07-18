"""CLI coverage for metadata-only legacy identity export hardening."""

from __future__ import annotations

import stat
from types import SimpleNamespace

from kestrel_sovereign import cli
from kestrel_sovereign import doctor
from kestrel_sovereign.identity import protected_export
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.setup.env_file import write_env


def test_identity_harden_exports_command_is_registered():
    args = cli.build_parser().parse_args(["identity", "harden-exports"])
    assert args.command == "identity"
    assert args.identity_command == "harden-exports"


def test_identity_harden_exports_changes_only_metadata(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    monkeypatch.delenv("KESTREL_DATA_DIR", raising=False)
    monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    export_root = tmp_path / "agent_data"
    export_root.mkdir(mode=0o755)
    package = export_root / "identity_legacy.json"
    secret = "continuity-package-secret"
    package.write_text(secret, encoding="utf-8")
    package.chmod(0o644)

    exit_code = cli.cmd_identity(SimpleNamespace(identity_command="harden-exports"))
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "hardened=1" in output
    assert "contents were not read" in output
    assert secret not in output
    assert package.read_text(encoding="utf-8") == secret
    assert stat.S_IMODE(export_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(package.stat().st_mode) == 0o600


def test_identity_harden_exports_refuses_link_without_touching_target(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    monkeypatch.delenv("KESTREL_DATA_DIR", raising=False)
    monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    export_root = tmp_path / "agent_data"
    export_root.mkdir(mode=0o700)
    outside = tmp_path / "outside.json"
    outside.write_text("outside-secret", encoding="utf-8")
    (export_root / "identity_link.json").symlink_to(outside)

    exit_code = cli.cmd_identity(SimpleNamespace(identity_command="harden-exports"))
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "refused=1" in output
    assert "outside-secret" not in output
    assert outside.read_text(encoding="utf-8") == "outside-secret"


def test_identity_harden_exports_includes_per_agent_configured_root(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    monkeypatch.delenv("KESTREL_DATA_DIR", raising=False)
    monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    agent_root = tmp_path / "agent_data" / "claw"
    export_root = agent_root / "continuity"
    export_root.mkdir(parents=True, mode=0o755)
    package = export_root / "identity_agent_bound.json"
    package.write_text("continuity-secret", encoding="utf-8")
    package.chmod(0o644)
    MultiAgentConfig(
        agents={
            "claw": LocalAgentConfig(
                data_dir="agent_data/claw",
                identity_export_dir="continuity",
                port=8801,
            )
        }
    ).save(tmp_path / "multi_agent.toml")

    exit_code = cli.cmd_identity(SimpleNamespace(identity_command="harden-exports"))
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "hardened=1" in output
    assert stat.S_IMODE(export_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(package.stat().st_mode) == 0o600
    assert package.read_text(encoding="utf-8") == "continuity-secret"


def test_identity_harden_exports_does_not_chmod_agent_root_without_exports(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    monkeypatch.delenv("KESTREL_DATA_DIR", raising=False)
    monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    agent_root = tmp_path / "agent_data" / "claw"
    agent_root.mkdir(parents=True, mode=0o755)
    agent_root.chmod(0o755)
    (agent_root / "kestrel_prime.db").touch()
    MultiAgentConfig(
        agents={
            "claw": LocalAgentConfig(
                data_dir="agent_data/claw",
                port=8801,
            )
        }
    ).save(tmp_path / "multi_agent.toml")

    exit_code = cli.cmd_identity(SimpleNamespace(identity_command="harden-exports"))
    capsys.readouterr()

    assert exit_code == 0
    assert stat.S_IMODE(agent_root.stat().st_mode) == 0o755


def test_doctor_and_hardener_use_identical_effective_roots(
    tmp_path,
    monkeypatch,
    capsys,
):
    secret = "must-not-appear-in-diagnostics"
    write_env(
        tmp_path / ".env",
        {
            "KESTREL_DATA_DIR": "dotenv-exports",
            "KESTREL_DATA_KEY": secret,
        },
    )
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    monkeypatch.delenv("KESTREL_IDENTITY_EXPORT_DIR", raising=False)
    monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    monkeypatch.setenv("KESTREL_DATA_DIR", "live-exports")
    captured = {}

    def capture_doctor(roots):
        captured["doctor"] = tuple(roots)
        return []

    def capture_hardener(roots):
        captured["hardener"] = tuple(roots)
        return protected_export.LegacyIdentityExportHardeningResult()

    monkeypatch.setattr(doctor, "audit_legacy_identity_exports", capture_doctor)
    monkeypatch.setattr(
        protected_export,
        "harden_legacy_identity_exports",
        capture_hardener,
    )

    doctor._check_legacy_identity_exports(tmp_path, doctor.DoctorReport())
    exit_code = cli.cmd_identity(SimpleNamespace(identity_command="harden-exports"))
    output = capsys.readouterr().out

    assert exit_code == 0
    assert captured["doctor"] == captured["hardener"]
    assert (tmp_path / "live-exports").resolve() in captured["doctor"]
    assert (tmp_path / "dotenv-exports").resolve() not in captured["doctor"]
    assert secret not in output
