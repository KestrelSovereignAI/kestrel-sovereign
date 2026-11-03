"""The "Start the native host" runbook settings reach the host (#3878).

The runbook's first draft set ``KESTREL_BASE_URL`` and
``KESTREL_A2A_PUSH_ENABLED`` as bare shell assignments, which a child process
never inherits. It now puts them in the project ``.env``. These checks run the
documented lines through the environment ``kestrel start`` hands the host.
"""

from __future__ import annotations

import re
from pathlib import Path

from kestrel_sovereign.a2a.native_publication import (
    NativeHostingConfiguration,
    unpublished_cohost_warning,
)
from kestrel_sovereign.multi_agent.config import HostConfig
from kestrel_sovereign.paths import spawned_agent_env

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "docs/architecture/a2a/OFFLINE_CUTOVER.md"
RELEASE_NOTE = REPO / "docs/releases/v0.54.1.md"
SETTINGS = ("KESTREL_BASE_URL", "KESTREL_A2A_PUSH_ENABLED")


def _section(text: str, heading: str) -> str:
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : None if end == -1 else end]


def _code_blocks(section: str) -> list[tuple[str, list[str]]]:
    return [
        (language, body.strip().splitlines())
        for language, body in re.findall(r"```(\w*)\n(.*?)```", section, re.S)
    ]


def _documented_env_lines() -> list[str]:
    section = _section(RUNBOOK.read_text(encoding="utf-8"), "Start the native host")
    [lines] = [body for language, body in _code_blocks(section) if language == "ini"]
    return lines


def test_the_runbook_names_both_settings_in_the_project_env():
    lines = _documented_env_lines()
    assert [line.split("=", 1)[0] for line in lines] == list(SETTINGS)
    section = _section(RUNBOOK.read_text(encoding="utf-8"), "Start the native host")
    assert "project `.env`" in section
    # No shell snippet in the section sets either variable without export.
    for language, body in _code_blocks(section):
        if language in {"sh", "bash", "shell", "console"}:
            for line in body:
                if any(f"{name}=" in line for name in SETTINGS):
                    assert line.lstrip().startswith("export "), line


def test_the_documented_lines_reach_the_host_and_publish(tmp_path, monkeypatch):
    """Written to ``.env``, they are in the host's environment and publish.

    The runbook also says the file outranks a stale exported value, which is
    the precedence ``kestrel start`` applies.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("\n".join(_documented_env_lines()) + "\n")
    monkeypatch.setenv("KESTREL_BASE_URL", "http://stale.invalid:1")
    monkeypatch.delenv("KESTREL_A2A_PUSH_ENABLED", raising=False)
    monkeypatch.delenv("KESTREL_A2A_DISCOVERY_AGENT", raising=False)

    host_env = spawned_agent_env(project)

    documented = dict(line.split("=", 1) for line in _documented_env_lines())
    assert {name: host_env[name] for name in SETTINGS} == documented
    hosting = NativeHostingConfiguration.from_environment(host_env)
    assert hosting is not None
    assert hosting.public_base_url == documented["KESTREL_BASE_URL"]
    assert hosting.push is None
    assert unpublished_cohost_warning(host_env, agents=2, host_port=8888) is None


def test_the_runbook_gives_the_advice_the_startup_warning_gives():
    """For the default host port, each documented line is the warning's own."""
    warning = unpublished_cohost_warning({}, agents=2, host_port=HostConfig().port)
    assert warning is not None
    for line in _documented_env_lines():
        assert line in warning


def test_the_release_note_matches_the_runbook():
    note = RELEASE_NOTE.read_text(encoding="utf-8")
    assert "`KESTREL_BASE_URL=http://127.0.0.1:<port>`" in note
    assert "`KESTREL_A2A_PUSH_ENABLED=false`" in note
    assert "project `.env`" in note
    assert "OFFLINE_CUTOVER.md#start-the-native-host" in note
