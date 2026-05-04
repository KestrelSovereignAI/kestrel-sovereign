"""Unit tests for kestrel_sovereign.setup.migrate_llm_config (#939)."""

from __future__ import annotations

import toml

from kestrel_sovereign.setup.migrate_llm_config import migrate_llm_config
from kestrel_sovereign.setup.toml_file import read_toml


def _write_legacy(project_dir, payload):
    (project_dir / "llm_config.toml").write_text(toml.dumps(payload))


def test_no_source_is_a_clean_noop(tmp_path):
    """If there's no llm_config.toml, exit without touching anything."""
    result = migrate_llm_config(tmp_path)
    assert result.action == "no_source"
    assert not (tmp_path / "kestrel.toml").exists()
    assert not (tmp_path / "llm_config.toml.bak").exists()


def test_clean_migration_into_fresh_kestrel_toml(tmp_path):
    """First run: no kestrel.toml. Source content lands under [llm];
    source renamed to .bak; no backup needed (nothing to back up)."""
    payload = {
        "route_priority": ["ollama:local", "openai:api"],
        "vendors": {"ollama": {"is_cloud": False}},
    }
    _write_legacy(tmp_path, payload)

    result = migrate_llm_config(tmp_path)

    assert result.action == "migrated"
    assert result.bak_path == tmp_path / "llm_config.toml.bak"
    assert result.backup_path is None  # no prior kestrel.toml to back up
    assert not (tmp_path / "llm_config.toml").exists()
    assert (tmp_path / "llm_config.toml.bak").exists()

    parsed = read_toml(tmp_path / "kestrel.toml")
    assert parsed["llm"] == payload


def test_migration_preserves_unrelated_kestrel_toml_sections(tmp_path):
    """Existing [agent], [council], etc. must survive the migration."""
    (tmp_path / "kestrel.toml").write_text(toml.dumps({
        "agent": {"name": "Kestrel"},
        "council": {"min_members": 3},
    }))
    _write_legacy(tmp_path, {"route_priority": ["ollama:local"]})

    result = migrate_llm_config(tmp_path)
    assert result.action == "migrated"
    assert result.backup_path is not None  # prior kestrel.toml backed up
    assert result.backup_path.exists()

    parsed = read_toml(tmp_path / "kestrel.toml")
    assert parsed["agent"] == {"name": "Kestrel"}
    assert parsed["council"] == {"min_members": 3}
    assert parsed["llm"] == {"route_priority": ["ollama:local"]}


def test_already_clean_when_llm_section_matches_source(tmp_path):
    """Idempotence: if [llm] already mirrors llm_config.toml byte-for-byte,
    we still rename the source to .bak but don't touch kestrel.toml."""
    payload = {"route_priority": ["ollama:local"]}
    (tmp_path / "kestrel.toml").write_text(toml.dumps({"llm": payload, "agent": {"name": "X"}}))
    _write_legacy(tmp_path, payload)

    pre_kestrel = (tmp_path / "kestrel.toml").read_bytes()

    result = migrate_llm_config(tmp_path)
    assert result.action == "already_clean"
    assert result.bak_path == tmp_path / "llm_config.toml.bak"
    assert result.backup_path is None  # no kestrel.toml change → no backup
    assert (tmp_path / "kestrel.toml").read_bytes() == pre_kestrel
    assert not (tmp_path / "llm_config.toml").exists()


def test_diverged_without_force_refuses_and_returns_diff(tmp_path):
    """Different content + no --force → don't write, surface the diff."""
    (tmp_path / "kestrel.toml").write_text(toml.dumps({
        "llm": {"route_priority": ["openai:api"]},
    }))
    _write_legacy(tmp_path, {"route_priority": ["ollama:local"]})

    pre_kestrel = (tmp_path / "kestrel.toml").read_bytes()
    pre_source = (tmp_path / "llm_config.toml").read_bytes()

    result = migrate_llm_config(tmp_path)
    assert result.action == "diverged"
    assert result.diff is not None
    assert "openai:api" in result.diff
    assert "ollama:local" in result.diff
    # Both files left untouched.
    assert (tmp_path / "kestrel.toml").read_bytes() == pre_kestrel
    assert (tmp_path / "llm_config.toml").read_bytes() == pre_source
    assert not (tmp_path / "llm_config.toml.bak").exists()


def test_diverged_with_force_lets_source_win(tmp_path):
    """--force: llm_config.toml content overrides existing [llm] via deep-merge."""
    (tmp_path / "kestrel.toml").write_text(toml.dumps({
        "llm": {"route_priority": ["openai:api"], "vendors": {"openai": {"is_cloud": True}}},
        "agent": {"name": "X"},
    }))
    _write_legacy(tmp_path, {"route_priority": ["ollama:local"]})

    result = migrate_llm_config(tmp_path, force=True)
    assert result.action == "migrated"
    assert result.backup_path is not None and result.backup_path.exists()

    parsed = read_toml(tmp_path / "kestrel.toml")
    # Source wins on the changed key.
    assert parsed["llm"]["route_priority"] == ["ollama:local"]
    # Deep-merge keeps untouched nested tables — bare keys in source can't
    # erase whole sub-tables it never mentioned.
    assert parsed["llm"]["vendors"] == {"openai": {"is_cloud": True}}
    assert parsed["agent"] == {"name": "X"}


def test_full_idempotence_round_trip(tmp_path):
    """End-to-end: migrate once → second run is a no_source no-op,
    kestrel.toml is byte-stable after the first migrate."""
    _write_legacy(tmp_path, {"route_priority": ["ollama:local"]})

    first = migrate_llm_config(tmp_path)
    assert first.action == "migrated"
    after_first = (tmp_path / "kestrel.toml").read_bytes()

    second = migrate_llm_config(tmp_path)
    assert second.action == "no_source"
    assert (tmp_path / "kestrel.toml").read_bytes() == after_first


def test_existing_bak_does_not_get_clobbered(tmp_path):
    """If a prior run already produced llm_config.toml.bak, a subsequent
    --force migration must not silently overwrite it."""
    (tmp_path / "llm_config.toml.bak").write_text("# from a prior migration\n")
    _write_legacy(tmp_path, {"route_priority": ["ollama:local"]})

    result = migrate_llm_config(tmp_path)
    assert result.action == "migrated"
    # The new bak file must not be the literal .bak path (which already
    # held an older backup).
    assert result.bak_path != tmp_path / "llm_config.toml.bak"
    assert result.bak_path.name.startswith("llm_config.toml.bak.")
    # Original .bak still intact.
    assert (tmp_path / "llm_config.toml.bak").read_text() == "# from a prior migration\n"


def test_malformed_source_returns_parse_error_and_preserves_source(tmp_path):
    """Malformed source TOML must NOT be silently treated as an empty
    dict. ``read_toml`` is deliberately tolerant for runtime config (so a
    bad file doesn't break boot), but the migration tool needs strict
    parsing — otherwise a corrupted llm_config.toml + missing [llm] in
    kestrel.toml both look like {}, and the source gets renamed to .bak
    with a misleading 'success' message, silently destroying the user's
    only LLM config."""
    source = tmp_path / "llm_config.toml"
    original = "[broken\nnot = valid\n"
    source.write_text(original)

    result = migrate_llm_config(tmp_path)

    assert result.action == "parse_error"
    assert result.error  # carries the parser message
    # Source file MUST be preserved verbatim for the user to fix.
    assert source.exists()
    assert source.read_text() == original
    assert not (tmp_path / "llm_config.toml.bak").exists()
    # kestrel.toml must not be created.
    assert not (tmp_path / "kestrel.toml").exists()


def test_malformed_source_does_not_touch_existing_kestrel_toml(tmp_path):
    """If the user already has a populated kestrel.toml [llm] and the
    legacy source is broken, we must NOT touch either file."""
    pre_kestrel = toml.dumps({
        "llm": {"route_priority": ["openai:api"]},
        "agent": {"name": "X"},
    })
    (tmp_path / "kestrel.toml").write_text(pre_kestrel)
    source = tmp_path / "llm_config.toml"
    original = "[broken\nnot = valid\n"
    source.write_text(original)

    result = migrate_llm_config(tmp_path, force=True)  # even with --force

    assert result.action == "parse_error"
    # Both files untouched.
    assert source.read_text() == original
    assert (tmp_path / "kestrel.toml").read_text() == pre_kestrel
    assert not (tmp_path / "llm_config.toml.bak").exists()
