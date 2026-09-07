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
from kestrel_sovereign.security import operator_lane
from kestrel_sovereign.security.operator_lane import (
    LIFECYCLE_VERBS,
    held_port_admits,
    operator_lane_refusal,
    ports_the_verb_touches,
)


@pytest.fixture(autouse=True)
def nothing_listens(monkeypatch):
    """By default no port is held, so the file comparison is what is under test.
    The live-host tests below override this per case."""
    monkeypatch.setattr(operator_lane, "held_port_admits", lambda port, presented, *, bind="0.0.0.0": None)

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


@pytest.mark.parametrize(
    "presented",
    [
        pytest.param("guessed", id="unrelated"),
        pytest.param(KEY[:-1], id="proper-prefix"),
        pytest.param(KEY + "x", id="key-plus-one"),
        pytest.param(KEY[1:], id="proper-suffix"),
        pytest.param(KEY.upper(), id="case-changed"),
    ],
)
def test_a_different_key_is_refused(project, presented):
    """Whole-key equality by fingerprint: a prefix, a suffix, or one extra
    byte is as wrong as an unrelated string (byte-at-a-time recovery is
    exactly what the fingerprint + compare_digest design exists to stop)."""
    refusal = operator_lane_refusal("update", project, {"KESTREL_API_KEY": presented})
    assert refusal is not None
    assert "not the host's sovereign key" in refusal


def test_a_project_without_a_stable_key_cannot_open_the_lane(tmp_path):
    refusal = operator_lane_refusal("create", tmp_path, {"KESTREL_API_KEY": KEY})
    assert refusal is not None
    assert "kestrel setup keys" in refusal
    # A live-process verb with nothing running and no file: also refused,
    # with the reference named (an EnvironmentFile host puts the key there too).
    refusal = operator_lane_refusal("terminate", tmp_path, {"KESTREL_API_KEY": KEY})
    assert refusal is not None
    assert "nothing is running to verify" in refusal


def test_the_reference_is_the_env_file_not_the_process_environment(project, monkeypatch):
    """The CLI's own ``os.environ`` is not the reference: a handler that
    rehydrated ``.env`` would otherwise satisfy the lane for anyone."""
    monkeypatch.setenv("KESTREL_API_KEY", "whatever-the-process-holds")
    assert operator_lane_refusal("start", project, {}) is not None
    assert operator_lane_refusal("start", project, {"KESTREL_API_KEY": KEY}) is None


# ---------------------------------------------------------------------------
# Dispatch: one gate before any handler
# ---------------------------------------------------------------------------

# A literal list, not `sorted(LIFECYCLE_VERBS)`: a test that derives its
# cases from the set under test shrinks with the set, so dropping a verb
# from the lane would drop its case too.
THE_FIVE_VERBS = ["create", "start", "terminate", "restart", "update"]


def test_the_lane_covers_exactly_the_five_verbs():
    assert set(LIFECYCLE_VERBS) == set(THE_FIVE_VERBS)


VERB_ARGV = {
    "create": ["create", "Nobody"],
    "start": ["start"],
    "terminate": ["terminate"],
    "restart": ["restart"],
    "update": ["update", "--dry-run"],
}


@pytest.mark.parametrize("verb", THE_FIVE_VERBS)
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


@pytest.mark.parametrize("verb", THE_FIVE_VERBS)
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


# ---------------------------------------------------------------------------
# The reference is the host acted on, not a file the caller wrote
# ---------------------------------------------------------------------------


def _held(verdicts):
    """A fake `held_port_admits`: port → None (free) / True / False."""
    calls = []

    def fake(port, presented, *, bind="0.0.0.0"):
        calls.append((port, presented))
        return verdicts.get(port)

    fake.calls = calls
    return fake


@pytest.mark.parametrize("verb", ["terminate", "restart", "update"])
def test_a_held_port_that_rejects_the_key_refuses_the_verb(project, monkeypatch, verb):
    """A caller-written project can point multi_agent.toml at any port;
    the process holding it decides, not the file."""
    (project / "multi_agent.toml").write_text("[host]\nport = 8912\n")
    fake = _held({8912: False})
    monkeypatch.setattr(operator_lane, "held_port_admits", fake)
    refusal = operator_lane_refusal(verb, project, {"KESTREL_API_KEY": KEY})
    assert refusal is not None
    assert "holds :8912" in refusal and "not touching it" in refusal
    assert fake.calls == [(8912, KEY)]


def test_a_held_port_that_admits_the_key_opens_the_lane(project, monkeypatch):
    (project / "multi_agent.toml").write_text("[host]\nport = 8912\n")
    monkeypatch.setattr(operator_lane, "held_port_admits", _held({8912: True}))
    assert operator_lane_refusal("terminate", project, {"KESTREL_API_KEY": KEY}) is None


def test_an_environmentfile_host_is_verified_by_its_live_process(tmp_path, monkeypatch):
    """No .env at all, but the live host accepts the key: admitted."""
    (tmp_path / "multi_agent.toml").write_text("[host]\nport = 8912\n")
    monkeypatch.setattr(operator_lane, "held_port_admits", _held({8912: True}))
    assert operator_lane_refusal("restart", tmp_path, {"KESTREL_API_KEY": KEY}) is None


def test_every_port_the_verb_touches_is_checked(project, monkeypatch):
    (project / "multi_agent.toml").write_text(
        "[host]\nport = 8912\n\n"
        "[agents.Emma]\ndata_dir = \"agent_data/emma\"\nport = 8801\n\n"
        "[agents.Nellie]\ndata_dir = \"agent_data/nellie\"\nport = 8802\n"
    )
    assert ports_the_verb_touches(project, None) == ("0.0.0.0", [8801, 8802, 8912])
    assert ports_the_verb_touches(project, "Nellie") == ("0.0.0.0", [8802, 8912])
    # The host is fine but a named agent's port is held by something that
    # rejects the key: `terminate Nellie` refuses.
    fake = _held({8912: True, 8802: False})
    monkeypatch.setattr(operator_lane, "held_port_admits", fake)
    refusal = operator_lane_refusal("terminate", project, {"KESTREL_API_KEY": KEY}, agent_name="Nellie")
    assert refusal is not None and "holds :8802" in refusal
    # Terminate-all touches every agent port too.
    fake = _held({8912: True, 8801: False})
    monkeypatch.setattr(operator_lane, "held_port_admits", fake)
    refusal = operator_lane_refusal("terminate", project, {"KESTREL_API_KEY": KEY})
    assert refusal is not None and "holds :8801" in refusal


def test_create_and_start_do_not_probe_live_ports(project, monkeypatch):
    (project / "multi_agent.toml").write_text("[host]\nport = 8912\n")
    fake = _held({8912: False})
    monkeypatch.setattr(operator_lane, "held_port_admits", fake)
    assert operator_lane_refusal("create", project, {"KESTREL_API_KEY": KEY}) is None
    assert operator_lane_refusal("start", project, {"KESTREL_API_KEY": KEY}) is None
    assert fake.calls == []


def test_held_port_admits_against_real_sockets(monkeypatch):
    """The real probe: a free port is None; a listener that is not a Kestrel
    server (the review's victim process) is False, and is left untouched."""
    import socket

    monkeypatch.undo()  # use the real held_port_admits
    from kestrel_sovereign.security.operator_lane import held_port_admits as real

    victim = socket.socket()
    victim.bind(("127.0.0.1", 0))
    victim.listen(1)
    port = victim.getsockname()[1]
    try:
        assert real(port, KEY, bind="127.0.0.1") is False
        victim.close()
        assert real(port, KEY, bind="127.0.0.1") is None
    finally:
        try:
            victim.close()
        except OSError:
            pass


def test_the_reviews_victim_scenario_end_to_end(tmp_path, monkeypatch, capsys):
    """A caller-written project whose multi_agent.toml names a port held by
    an unrelated process, with a caller-chosen key in .env and exported:
    `kestrel terminate --force` refuses and the process survives."""
    import socket

    monkeypatch.undo()
    victim = socket.socket()
    victim.bind(("127.0.0.1", 0))
    victim.listen(1)
    port = victim.getsockname()[1]
    try:
        (tmp_path / ".env").write_text("KESTREL_API_KEY=attacker-chosen\n")
        (tmp_path / "multi_agent.toml").write_text(f"[host]\nport = {port}\nbind = \"127.0.0.1\"\n")
        monkeypatch.setattr(sys, "argv", ["kestrel", "terminate", "--force"])
        monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
        with patch.dict(os.environ, {"PATH": os.environ.get("PATH", ""), "KESTREL_API_KEY": "attacker-chosen"}, clear=True):
            with patch.object(cli, "cmd_terminate") as terminate:
                rc = cli.main()
        assert rc == 1
        terminate.assert_not_called()
        assert f"holds :{port}" in capsys.readouterr().err
        # Still bound and listening: nothing signalled it. (A connect probe
        # would only measure the backlog, which the lane's own refused HTTP
        # probe still occupies.)
        assert victim.getsockname()[1] == port
        assert victim.fileno() != -1
    finally:
        victim.close()
