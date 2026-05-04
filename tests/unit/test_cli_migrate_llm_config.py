"""Unit tests for the ``kestrel migrate-llm-config`` CLI surface (#939)."""

from __future__ import annotations

import toml

from kestrel_sovereign.cli import build_parser, cmd_migrate_llm_config


def _parse(argv):
    return build_parser().parse_args(argv)


def test_parser_defaults():
    args = _parse(["migrate-llm-config"])
    assert args.force is False
    assert args.project_dir is None
    assert args.command == "migrate-llm-config"


def test_parser_accepts_force_and_project_dir(tmp_path):
    args = _parse([
        "migrate-llm-config", "--force", "--project-dir", str(tmp_path),
    ])
    assert args.force is True
    assert args.project_dir == str(tmp_path)


def test_no_source_exits_zero_with_helpful_message(tmp_path, capsys):
    args = _parse(["migrate-llm-config", "--project-dir", str(tmp_path)])
    rc = cmd_migrate_llm_config(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Nothing to migrate" in out
    assert "[llm]" in out


def test_diverged_exits_one_with_diff(tmp_path, capsys):
    (tmp_path / "kestrel.toml").write_text(toml.dumps({
        "llm": {"route_priority": ["openai:api"]},
    }))
    (tmp_path / "llm_config.toml").write_text(toml.dumps({
        "route_priority": ["ollama:local"],
    }))
    args = _parse(["migrate-llm-config", "--project-dir", str(tmp_path)])
    rc = cmd_migrate_llm_config(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "differs from" in err
    assert "openai:api" in err
    assert "ollama:local" in err
    assert "--force" in err


def test_clean_migrate_exits_zero_and_reports_paths(tmp_path, capsys):
    (tmp_path / "llm_config.toml").write_text(toml.dumps({
        "route_priority": ["ollama:local"],
    }))
    args = _parse(["migrate-llm-config", "--project-dir", str(tmp_path)])
    rc = cmd_migrate_llm_config(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Migrated llm_config.toml" in out
    assert "llm_config.toml.bak" in out
    assert (tmp_path / "kestrel.toml").exists()
    assert (tmp_path / "llm_config.toml.bak").exists()
    assert not (tmp_path / "llm_config.toml").exists()


def test_already_clean_exits_zero(tmp_path, capsys):
    payload = {"route_priority": ["ollama:local"]}
    (tmp_path / "kestrel.toml").write_text(toml.dumps({"llm": payload}))
    (tmp_path / "llm_config.toml").write_text(toml.dumps(payload))
    args = _parse(["migrate-llm-config", "--project-dir", str(tmp_path)])
    rc = cmd_migrate_llm_config(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "already matches" in out
    assert (tmp_path / "llm_config.toml.bak").exists()


def test_force_overrides_divergence(tmp_path, capsys):
    (tmp_path / "kestrel.toml").write_text(toml.dumps({
        "llm": {"route_priority": ["openai:api"]},
    }))
    (tmp_path / "llm_config.toml").write_text(toml.dumps({
        "route_priority": ["ollama:local"],
    }))
    args = _parse([
        "migrate-llm-config", "--force", "--project-dir", str(tmp_path),
    ])
    rc = cmd_migrate_llm_config(args)
    assert rc == 0
    parsed = toml.loads((tmp_path / "kestrel.toml").read_text())
    assert parsed["llm"]["route_priority"] == ["ollama:local"]
