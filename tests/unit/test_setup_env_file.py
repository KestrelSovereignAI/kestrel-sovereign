"""Unit tests for kestrel_sovereign.setup.env_file."""

from __future__ import annotations

import time

import pytest

from kestrel_sovereign.setup.env_file import (
    EnvWriteResult,
    read_env,
    write_env,
)


def test_read_env_missing_file_returns_empty(tmp_path):
    assert read_env(tmp_path / "nope.env") == {}


def test_read_env_parses_key_value_pairs(tmp_path):
    p = tmp_path / ".env"
    p.write_text("FOO=bar\nBAZ=qux\n")
    assert read_env(p) == {"FOO": "bar", "BAZ": "qux"}


def test_read_env_strips_quotes(tmp_path):
    p = tmp_path / ".env"
    p.write_text('SINGLE=\'a b\'\nDOUBLE="c d"\nNAKED=plain\n')
    assert read_env(p) == {"SINGLE": "a b", "DOUBLE": "c d", "NAKED": "plain"}


def test_read_env_skips_comments_and_blanks(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# header\n\nKEY=value\n# trailing\n")
    assert read_env(p) == {"KEY": "value"}


def test_write_env_creates_file_when_absent(tmp_path):
    p = tmp_path / ".env"
    result = write_env(p, {"NEW": "value"})
    assert p.exists()
    assert read_env(p) == {"NEW": "value"}
    assert result.added == ("NEW",)
    assert result.backup_path is None  # No prior file to back up


def test_write_env_preserves_unrelated_keys(tmp_path):
    p = tmp_path / ".env"
    p.write_text("KEEP=me\nALSO=here\n")
    write_env(p, {"NEW": "v"})
    parsed = read_env(p)
    assert parsed == {"KEEP": "me", "ALSO": "here", "NEW": "v"}


def test_write_env_preserves_comments_and_blank_lines(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# header comment\n\nFOO=1\n\n# inline\nBAR=2\n")
    write_env(p, {"FOO": "10"})
    text = p.read_text()
    assert "# header comment" in text
    assert "# inline" in text
    assert "FOO=10" in text
    assert "BAR=2" in text


def test_write_env_backs_up_before_changing(tmp_path):
    p = tmp_path / ".env"
    p.write_text("FOO=old\n")
    result = write_env(p, {"FOO": "new"})
    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert result.backup_path.read_text() == "FOO=old\n"
    assert read_env(p) == {"FOO": "new"}


def test_write_env_no_op_when_value_unchanged(tmp_path):
    p = tmp_path / ".env"
    p.write_text("FOO=same\n")
    result = write_env(p, {"FOO": "same"})
    assert result.backup_path is None
    assert result.added == ()
    assert result.updated == ()


def test_write_env_no_op_when_no_changes(tmp_path):
    p = tmp_path / ".env"
    p.write_text("FOO=value\n")
    result = write_env(p, {})
    assert result == EnvWriteResult(path=p, backup_path=None, added=(), updated=())


def test_write_env_refuses_empty_value_by_default(tmp_path):
    p = tmp_path / ".env"
    p.write_text("FOO=existing\n")
    write_env(p, {"FOO": "", "BAR": ""})
    # Neither change applies — empty values are filtered out
    assert read_env(p) == {"FOO": "existing"}


def test_write_env_allow_empty_writes_blank(tmp_path):
    p = tmp_path / ".env"
    write_env(p, {"FLAG": ""}, allow_empty=True)
    text = p.read_text()
    assert "FLAG=" in text


def test_write_env_quotes_values_with_spaces(tmp_path):
    p = tmp_path / ".env"
    write_env(p, {"PHRASE": "hello world"})
    text = p.read_text()
    assert 'PHRASE="hello world"' in text


def test_write_env_does_not_quote_safe_values(tmp_path):
    p = tmp_path / ".env"
    write_env(
        p,
        {
            "PATH_LIKE": "/usr/local/bin",
            "URL_LIKE": "http://localhost:8888",
            "DOTTED": "foo.bar.baz",
        },
    )
    text = p.read_text()
    assert "PATH_LIKE=/usr/local/bin" in text
    assert "URL_LIKE=http://localhost:8888" in text
    assert "DOTTED=foo.bar.baz" in text


def test_write_env_idempotent_across_runs(tmp_path):
    p = tmp_path / ".env"
    write_env(p, {"FOO": "v"})
    first_text = p.read_text()
    backup_count_before = len(list(tmp_path.glob(".env.backup-*")))

    # Run again with the same content — must not produce another backup
    result = write_env(p, {"FOO": "v"})
    second_text = p.read_text()
    backup_count_after = len(list(tmp_path.glob(".env.backup-*")))

    assert first_text == second_text
    assert result.backup_path is None
    assert backup_count_before == backup_count_after


def test_write_env_backup_filename_has_timestamp(tmp_path):
    p = tmp_path / ".env"
    p.write_text("FOO=v1\n")
    result = write_env(p, {"FOO": "v2"})
    assert result.backup_path is not None
    assert ".env.backup-" in result.backup_path.name
    # The format is YYYYMMDD-HHMMSS
    suffix = result.backup_path.name.split(".env.backup-", 1)[1]
    assert len(suffix) == 15
    assert suffix[8] == "-"


def test_write_env_secret_with_special_chars_roundtrips(tmp_path):
    """Real-world: API keys with = and / characters must survive write/read."""
    p = tmp_path / ".env"
    secret = "sk-ant-api03-aBc=DeF/gHi+jKl_-MnO"
    write_env(p, {"ANTHROPIC_API_KEY": secret})
    assert read_env(p)["ANTHROPIC_API_KEY"] == secret


def test_write_env_distinguishes_added_from_updated(tmp_path):
    p = tmp_path / ".env"
    p.write_text("OLD=keep\nCHANGED=v1\n")
    result = write_env(p, {"CHANGED": "v2", "BRAND_NEW": "v"})
    assert set(result.added) == {"BRAND_NEW"}
    assert set(result.updated) == {"CHANGED"}


def test_write_env_keeps_non_keyvalue_lines_untouched(tmp_path):
    """Lines that don't match KEY=VAL syntax must be preserved verbatim."""
    p = tmp_path / ".env"
    p.write_text("export FOO=value\n# comment\nFOO=bar\n")
    write_env(p, {"FOO": "baz"})
    text = p.read_text()
    # The 'export FOO=...' line is non-matching to our regex (has 'export ' prefix)
    # and should be preserved as-is. The plain 'FOO=' line gets updated.
    assert "export FOO=value" in text
    assert "FOO=baz" in text
