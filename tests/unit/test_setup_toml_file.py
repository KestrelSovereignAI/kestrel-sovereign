"""Unit tests for kestrel_sovereign.setup.toml_file."""

from __future__ import annotations

import toml

from kestrel_sovereign.setup.toml_file import read_toml, write_toml


def test_read_toml_missing_returns_empty(tmp_path):
    assert read_toml(tmp_path / "nope.toml") == {}


def test_read_toml_corrupted_returns_empty(tmp_path):
    p = tmp_path / "broken.toml"
    p.write_text("[broken\nthis = is not valid\n")
    # We tolerate corruption rather than raising; the wizard then
    # treats the file as empty and overwrites (after backup).
    assert read_toml(p) == {}


def test_write_toml_creates_file_when_absent(tmp_path):
    p = tmp_path / "kestrel.toml"
    result = write_toml(p, {"agent": {"name": "Kestrel"}})
    assert p.exists()
    assert result.changed
    assert result.backup_path is None
    assert read_toml(p) == {"agent": {"name": "Kestrel"}}


def test_write_toml_deep_merges_nested_tables(tmp_path):
    p = tmp_path / "kestrel.toml"
    p.write_text(toml.dumps({
        "llm": {
            "route_priority": ["openai:api"],
            "vendors": {
                "openai": {"is_cloud": True},
                "ollama": {"is_cloud": False},
            },
        }
    }))
    write_toml(p, {"llm": {"route_priority": ["ollama:local", "openai:api"]}})
    parsed = read_toml(p)
    assert parsed["llm"]["route_priority"] == ["ollama:local", "openai:api"]
    # Untouched nested tables must survive.
    assert parsed["llm"]["vendors"]["openai"] == {"is_cloud": True}
    assert parsed["llm"]["vendors"]["ollama"] == {"is_cloud": False}


def test_write_toml_no_op_when_identical(tmp_path):
    p = tmp_path / "kestrel.toml"
    p.write_text(toml.dumps({"llm": {"route_priority": ["ollama:local"]}}))
    result = write_toml(p, {"llm": {"route_priority": ["ollama:local"]}})
    assert not result.changed
    assert result.backup_path is None


def test_write_toml_backs_up_before_change(tmp_path):
    p = tmp_path / "kestrel.toml"
    original = toml.dumps({"llm": {"route_priority": ["openai:api"]}})
    p.write_text(original)
    result = write_toml(p, {"llm": {"route_priority": ["ollama:local"]}})
    assert result.changed
    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert result.backup_path.read_text() == original


def test_write_toml_shallow_merge_replaces_top_keys(tmp_path):
    p = tmp_path / "kestrel.toml"
    p.write_text(toml.dumps({
        "llm": {"route_priority": ["openai:api"], "vendors": {"openai": {"is_cloud": True}}},
        "agent": {"name": "X"},
    }))
    write_toml(p, {"llm": {"route_priority": ["ollama:local"]}}, deep_merge=False)
    parsed = read_toml(p)
    # With deep_merge=False, the entire 'llm' table is replaced
    assert parsed["llm"] == {"route_priority": ["ollama:local"]}
    assert "vendors" not in parsed["llm"]
    # Sibling tables are preserved.
    assert parsed["agent"] == {"name": "X"}


def test_write_toml_idempotent_across_runs(tmp_path):
    p = tmp_path / "kestrel.toml"
    payload = {"llm": {"route_priority": ["ollama:local"]}}
    write_toml(p, payload)
    first_text = p.read_text()
    write_toml(p, payload)
    second_text = p.read_text()
    assert first_text == second_text


def test_write_toml_preserves_unrelated_tables(tmp_path):
    """A wizard write to [llm] must not touch [council], [voice], etc."""
    p = tmp_path / "kestrel.toml"
    p.write_text(toml.dumps({
        "llm": {"route_priority": ["openai:api"]},
        "council": {"min_members": 3, "max_rounds": 5},
        "voice": {"tts_provider_priority": ["piper", "openai"]},
    }))
    write_toml(p, {"llm": {"route_priority": ["ollama:local"]}})
    parsed = read_toml(p)
    assert parsed["council"] == {"min_members": 3, "max_rounds": 5}
    assert parsed["voice"] == {"tts_provider_priority": ["piper", "openai"]}
