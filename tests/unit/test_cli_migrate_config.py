from __future__ import annotations

import toml

from kestrel_sovereign.cli import build_parser, cmd_migrate_config


def _parse(argv):
    return build_parser().parse_args(argv)


def test_parser_defaults():
    args = _parse(["migrate-config"])
    assert args.project_dir is None
    assert args.command == "migrate-config"


def test_parser_accepts_project_dir(tmp_path):
    args = _parse(["migrate-config", "--project-dir", str(tmp_path)])
    assert args.project_dir == str(tmp_path)


def test_clean_migrate_exits_zero_and_is_idempotent(tmp_path, capsys):
    (tmp_path / "kestrel.toml").write_text(toml.dumps({
        "agent": {"name": "Existing"},
    }))
    (tmp_path / "model_mandate.toml").write_text(toml.dumps({
        "defaults": {"preferred": "", "cheap_model": "auto"},
        "mandates": {},
    }))
    (tmp_path / "model_catalog.toml").write_text(toml.dumps({
        "context_limits_override": {"gpt-5": 128000},
    }))
    args = _parse(["migrate-config", "--project-dir", str(tmp_path)])

    first = cmd_migrate_config(args)
    second = cmd_migrate_config(args)
    parsed = toml.loads((tmp_path / "kestrel.toml").read_text())

    assert first == 0
    assert second == 0
    out = capsys.readouterr().out
    assert "Migrated legacy model config" in out
    assert "already has the unified model config sections" in out
    assert parsed["agent"]["name"] == "Existing"
    assert parsed["llm"]["mandate"]["defaults"]["cheap_model"] == "auto"
    assert parsed["llm"]["catalog"]["context_limits_override"]["gpt-5"] == 128000


def test_parse_error_exits_one_and_preserves_files(tmp_path, capsys):
    original_kestrel = "[agent]\nname = 'Existing'\n"
    original_mandate = "[broken\nnot = valid\n"
    (tmp_path / "kestrel.toml").write_text(original_kestrel)
    (tmp_path / "model_mandate.toml").write_text(original_mandate)
    (tmp_path / "model_catalog.toml").write_text(toml.dumps({
        "context_limits_override": {"gpt-5": 128000},
    }))
    args = _parse(["migrate-config", "--project-dir", str(tmp_path)])

    rc = cmd_migrate_config(args)

    assert rc == 1
    err = capsys.readouterr().err
    assert "not valid TOML" in err
    assert "No files were changed" in err
    assert (tmp_path / "kestrel.toml").read_text() == original_kestrel
    assert (tmp_path / "model_mandate.toml").read_text() == original_mandate
