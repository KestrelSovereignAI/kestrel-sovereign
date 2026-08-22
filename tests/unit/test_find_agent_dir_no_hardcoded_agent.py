"""``find_agent_dir`` must not privilege one deployment's agent name (#2769).

The "common locations" list used to lead with a specific personal agent's
directory, so any host without ``KESTREL_DB_PATH`` silently resolved to that one
agent even when other agents were present. Named agents belong to the generic
``agent_data`` scan, which privileges nobody.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.agent_config import find_agent_dir


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)


def _make_agent(root, name):
    agent_dir = root / "agent_data" / name
    agent_dir.mkdir(parents=True)
    (agent_dir / "kestrel_prime.db").write_text("")
    return agent_dir


def test_no_single_agent_name_is_hardcoded(monkeypatch, tmp_path):
    """With two agents present and no override, resolution must be by the
    generic sorted scan — not by a baked-in name."""
    monkeypatch.chdir(tmp_path)
    _make_agent(tmp_path, "claw")
    expected = _make_agent(tmp_path, "aardvark")

    assert find_agent_dir() == expected


def test_resolution_is_deterministic(monkeypatch, tmp_path):
    """Directory iteration order is filesystem-dependent; the scan sorts so the
    same tree resolves identically on every call and platform."""
    monkeypatch.chdir(tmp_path)
    for name in ("zebra", "claw", "banana"):
        _make_agent(tmp_path, name)

    assert {find_agent_dir() for _ in range(5)} == {tmp_path / "agent_data" / "banana"}


def test_explicit_env_override_still_wins(monkeypatch, tmp_path):
    """Removing the hardcoded name must not weaken the supported override."""
    monkeypatch.chdir(tmp_path)
    _make_agent(tmp_path, "aardvark")
    chosen = _make_agent(tmp_path, "claw")
    monkeypatch.setenv("KESTREL_DB_PATH", str(chosen))

    assert find_agent_dir() == chosen


def test_explicit_hint_still_wins(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_agent(tmp_path, "aardvark")
    chosen = _make_agent(tmp_path, "claw")

    assert find_agent_dir(str(chosen)) == chosen


def test_returns_none_when_no_agent_exists(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent_data").mkdir()

    assert find_agent_dir() is None
