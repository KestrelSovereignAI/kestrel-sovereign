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
    OperatorLaneRefused,
    operator_lane_refusal,
    vouch_pid,
    vouch_response,
)


@pytest.fixture(autouse=True)
def lane_disarmed(monkeypatch):
    """Every test starts with the vouch disarmed; the ones about the signal
    arm it themselves. `main()` arms it for admitted lifecycle verbs."""
    monkeypatch.setattr(operator_lane, "_presented_key", None)

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
    # A live-process verb with no file is admitted at dispatch: whatever it
    # signals still has to vouch (an EnvironmentFile host is verified there).
    assert operator_lane_refusal("terminate", tmp_path, {"KESTREL_API_KEY": KEY}) is None


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
# The signal: the process about to die must vouch
# ---------------------------------------------------------------------------

import json
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


class _VouchingHost(HTTPServer):
    """A loopback HTTP server that answers the vouch route with a given key."""

    def __init__(self, key, *, wrong=False):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(inner):
                url = urlparse(inner.path)
                if url.path != "/api/auth/vouch":
                    inner.send_response(404); inner.end_headers(); return
                nonce = parse_qs(url.query).get("nonce", [""])[0]
                answer = vouch_response(key, nonce)
                if wrong:
                    answer = "0" * len(answer)
                body = json.dumps({"vouch": answer}).encode()
                inner.send_response(200)
                inner.send_header("Content-Type", "application/json")
                inner.send_header("Content-Length", str(len(body)))
                inner.end_headers()
                inner.wfile.write(body)

        super().__init__(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.shutdown()
        self.server_close()


@pytest.fixture
def vouching_host():
    host = _VouchingHost(KEY)
    yield host
    host.stop()


def test_vouch_answers_and_nonces_are_bound_to_the_key():
    nonce = "ab" * 16
    assert vouch_response(KEY, nonce) == vouch_response(f'"{KEY}"', nonce)
    assert vouch_response(KEY, nonce) != vouch_response(KEY + "x", nonce)
    assert vouch_response(KEY, nonce) != vouch_response(KEY, "cd" * 16)


def test_this_process_vouches_when_one_of_its_listeners_answers(vouching_host):
    """The test process owns the vouching listener, so its own PID vouches."""
    assert vouch_pid(os.getpid(), KEY) is True
    assert vouch_pid(os.getpid(), "some-other-key") is False


def test_a_wrong_answer_does_not_vouch():
    host = _VouchingHost(KEY, wrong=True)
    try:
        assert vouch_pid(os.getpid(), KEY) is False
    finally:
        host.stop()


def _throwaway_listener(bind="127.0.0.1"):
    """A child process that listens and never speaks HTTP: the review's victim.

    A child, not this process, so a fail-open would kill something harmless.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"import socket,sys,time; s=socket.socket(); s.bind(('{bind}',0)); s.listen(5); "
         "print(s.getsockname()[1], flush=True); time.sleep(120)"],
        stdout=subprocess.PIPE, text=True,
    )
    port = int(proc.stdout.readline().strip())
    return proc, port


def test_a_listener_that_is_not_a_kestrel_host_does_not_vouch():
    proc, port = _throwaway_listener()
    try:
        assert vouch_pid(proc.pid, KEY) is False
    finally:
        proc.kill(); proc.wait()


def test_a_pid_with_no_listeners_does_not_vouch():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        assert vouch_pid(proc.pid, KEY) is False
    finally:
        proc.kill(); proc.wait()


def test_kill_process_refuses_an_unvouched_pid_only_while_the_lane_is_armed():
    from kestrel_sovereign.multi_agent.process_manager import ProcessManager

    proc, port = _throwaway_listener()
    try:
        operator_lane.activate_operator_lane(KEY)
        with pytest.raises(OperatorLaneRefused, match=f"PID {proc.pid}"):
            ProcessManager.kill_process(proc.pid, force=True)
        assert proc.poll() is None  # alive: nothing was signalled
        # The host managing its own agents is not an invoker: disarmed, the
        # chokepoint behaves as before.
        operator_lane._presented_key = None
        assert ProcessManager.kill_process(proc.pid, force=True) is True
        proc.wait(timeout=10)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()


def _attacker_project(tmp_path, port, *, bind="::1", pid_file_pid=None):
    """A caller-written project: its own key, a toml naming a port it does not
    own (with a bind chosen to defeat a port probe), optionally a pid file."""
    (tmp_path / ".env").write_text("KESTREL_API_KEY=attacker-chosen\n")
    (tmp_path / "multi_agent.toml").write_text(f'[host]\nport = {port}\nbind = "{bind}"\n')
    if pid_file_pid is not None:
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / ".host.pid").write_text(str(pid_file_pid))
    return tmp_path


@pytest.mark.parametrize("argv", [["terminate"], ["terminate", "--force"], ["restart"]])
def test_the_reviews_victim_scenarios_end_to_end_through_the_real_terminate(tmp_path, monkeypatch, capsys, argv):
    """Round 2's two live exploits, against the REAL cmd_terminate: a victim on
    0.0.0.0 with the config saying bind = "::1" (the port probe missed it), and
    a caller-written logs/.host.pid naming the victim (no port at all). Both
    reach kill_process, where the victim cannot vouch, and the verb refuses
    with the victim alive. The victim is a throwaway child."""
    victim, port = _throwaway_listener("0.0.0.0")
    try:
        project = _attacker_project(tmp_path, port, bind="::1", pid_file_pid=victim.pid)
        monkeypatch.setattr(sys, "argv", ["kestrel", *argv])
        monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
        with patch.dict(os.environ, {"PATH": os.environ.get("PATH", ""), "KESTREL_API_KEY": "attacker-chosen"}, clear=True):
            rc = cli.main()
        assert rc == 1
        err = capsys.readouterr().err
        assert "did not vouch" in err and f"PID {victim.pid}" in err
        assert victim.poll() is None, "the victim was signalled"
    finally:
        victim.kill(); victim.wait()


def test_a_vouching_host_is_signalled_and_a_non_vouching_neighbour_is_not(tmp_path, monkeypatch, capsys, vouching_host):
    """Round 2's laundering case: two processes on one port number. The one
    that answers the challenge is this test process (its vouching listener);
    the other is a throwaway child. Whatever named the port, only the
    vouched PID may be signalled — and this process must not be, either,
    because the verb refuses as a whole once any target fails to vouch."""
    from kestrel_sovereign.multi_agent.process_manager import ProcessManager

    victim, port = _throwaway_listener("0.0.0.0")
    try:
        operator_lane.activate_operator_lane(KEY)
        # Both PIDs are "on" some port; the chokepoint judges each PID alone.
        assert vouch_pid(os.getpid(), KEY) is True
        with pytest.raises(OperatorLaneRefused):
            ProcessManager.kill_process(victim.pid, force=True)
        assert victim.poll() is None
    finally:
        victim.kill(); victim.wait()


def test_main_disarms_the_lane_when_the_verb_returns(project, monkeypatch):
    """Arming is scoped to one verb's execution: an in-process caller of
    main() (a test suite, an embedding tool) must not leave every later
    kill_process gated. The gate caught this as a leak into another file."""
    monkeypatch.setattr(sys, "argv", ["kestrel", "terminate"])
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    seen = {}
    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", ""), "KESTREL_API_KEY": KEY}, clear=True):
        with patch.object(cli, "cmd_terminate", side_effect=lambda args: seen.setdefault("armed", operator_lane.operator_lane_is_active()) and 0):
            cli.main()
    assert seen["armed"] is True
    assert operator_lane.operator_lane_is_active() is False
