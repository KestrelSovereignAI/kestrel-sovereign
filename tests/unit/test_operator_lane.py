"""Host lifecycle verbs require the operator lane (#3233).

``kestrel create|start|terminate|restart|update`` ran on local process
access alone, so an agent with the host-shell capability could create,
kill, restart or re-image agents, the host and the fleet under ordinary
tool consent. The lane: the environment the verb was invoked with must
carry the host's stable sovereign key, compared against the project's
``.env``; an agent's sanitized subprocess environment never does.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from kestrel_sovereign import cli
from kestrel_sovereign.security.operator_lane import (
    LIFECYCLE_VERBS,
    operator_lane_refusal,
)

KEY = "stable-sovereign-key-3233"


@pytest.fixture
def project(tmp_path):
    (tmp_path / ".env").write_text(f'KESTREL_API_KEY="{KEY}"\n')
    return tmp_path


# ---------------------------------------------------------------------------
# The lane
# ---------------------------------------------------------------------------


def test_no_credential_is_refused(project):
    refusal = operator_lane_refusal("terminate", project, {"PATH": "/usr/bin", "HOME": "/h"})
    assert refusal is not None
    assert "no sovereign credential in the invoking environment" in refusal
    assert "agent's shell never carries it" in refusal


def test_the_projects_key_is_admitted_and_quoting_is_normalized(project):
    assert operator_lane_refusal("terminate", project, {"KESTREL_API_KEY": KEY}) is None
    assert operator_lane_refusal("terminate", project, {"KESTREL_API_KEY": f"'{KEY}'"}) is None


def test_a_different_key_is_refused(project):
    refusal = operator_lane_refusal("update", project, {"KESTREL_API_KEY": "guessed"})
    assert refusal is not None
    assert "not the host's sovereign key" in refusal


def test_a_project_without_a_stable_key_cannot_open_the_lane(tmp_path):
    refusal = operator_lane_refusal("create", tmp_path, {"KESTREL_API_KEY": KEY})
    assert refusal is not None
    assert "kestrel setup keys" in refusal


def test_the_reference_is_the_env_file_not_the_process_environment(project, monkeypatch):
    """The CLI's own ``os.environ`` is not the reference: a handler that
    rehydrated ``.env`` would otherwise satisfy the lane for anyone."""
    monkeypatch.setenv("KESTREL_API_KEY", "whatever-the-process-holds")
    assert operator_lane_refusal("start", project, {}) is not None
    assert operator_lane_refusal("start", project, {"KESTREL_API_KEY": KEY}) is None


# ---------------------------------------------------------------------------
# Dispatch: one gate before any handler
# ---------------------------------------------------------------------------

VERB_ARGV = {
    "create": ["create", "Nobody"],
    "start": ["start"],
    "terminate": ["terminate"],
    "restart": ["restart"],
    "update": ["update", "--dry-run"],
}


@pytest.mark.parametrize("verb", sorted(LIFECYCLE_VERBS))
def test_every_lifecycle_verb_is_refused_before_its_handler_runs(verb, project, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["kestrel", *VERB_ARGV[verb]])
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    handler = f"cmd_{verb}"
    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", ""), "HOME": "/h"}, clear=True):
        with patch.object(cli, handler) as mocked:
            rc = cli.main()
    assert rc == 1
    mocked.assert_not_called()
    assert "refused" in capsys.readouterr().err


@pytest.mark.parametrize("verb", sorted(LIFECYCLE_VERBS))
def test_every_lifecycle_verb_runs_for_the_operator(verb, project, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["kestrel", *VERB_ARGV[verb]])
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", ""), "KESTREL_API_KEY": KEY}, clear=True):
        with patch.object(cli, f"cmd_{verb}", return_value=0) as mocked:
            rc = cli.main()
    assert rc == 0
    mocked.assert_called_once()


def test_non_lifecycle_verbs_need_no_lane(project, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["kestrel", "status"])
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True):
        with patch.object(cli, "cmd_status", return_value=0) as mocked:
            assert cli.main() == 0
    mocked.assert_called_once()


def test_the_lane_is_judged_on_the_environment_at_entry(project, monkeypatch):
    """Something between entry and dispatch that puts the key into
    ``os.environ`` (a ``.env`` load) must not open the lane."""
    monkeypatch.setattr(sys, "argv", ["kestrel", "terminate"])
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    seen = {}

    def leak_then_build():
        os.environ["KESTREL_API_KEY"] = KEY
        return cli.build_parser.__wrapped__() if hasattr(cli.build_parser, "__wrapped__") else _real_build()

    _real_build = cli.build_parser
    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True):
        with patch.object(cli, "build_parser", side_effect=leak_then_build):
            with patch.object(cli, "cmd_terminate") as mocked:
                rc = cli.main()
    assert rc == 1
    mocked.assert_not_called()
