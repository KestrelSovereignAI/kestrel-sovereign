"""CLI coverage for metadata-only legacy identity export hardening."""

from __future__ import annotations

import stat
from types import SimpleNamespace

from kestrel_sovereign import cli


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
