"""Whether a live process serves an agent, however it was launched (#3522).

``kestrel start`` writes a PID file for what it launches; a server started
directly or by a container's Uvicorn entrypoint writes none. The process
serving an agent therefore records itself in the agent's data directory, and
the guards that must not change an agent beneath a live process read that
record as well as the launcher's. Liveness that cannot be established counts
as running.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign import cli
from kestrel_sovereign.multi_agent import liveness
from kestrel_sovereign.multi_agent.liveness import (
    SERVING_RECORD_DIRNAME,
    ServingRecord,
    _serving_record_holder,
    agent_holder,
    serving_holder,
)
from kestrel_sovereign.multi_agent.process_manager import _MAX_PID, ProcessManager


@pytest.fixture
def data_dir(tmp_path):
    path = tmp_path / "agent_data" / "emma"
    path.mkdir(parents=True)
    return path


def _rewrite(record: ServingRecord, **fields) -> None:
    payload = json.loads(record.path.read_text())
    payload.update(fields)
    record.path.write_text(json.dumps(payload))


def test_no_record_means_nothing_serves_it(tmp_path, data_dir):
    assert serving_holder(data_dir) is None
    assert agent_holder(tmp_path, "Emma", data_dir) is None


def test_a_live_serving_record_names_the_process_and_how_to_stop_it(
    tmp_path, data_dir
):
    record = ServingRecord.acquire(data_dir)

    holder = agent_holder(tmp_path, "Emma", data_dir)

    assert f"PID {os.getpid()} serves it" in holder.evidence
    assert str(record.path) in holder.evidence
    expected = (
        f"taskkill /PID {os.getpid()}"
        if sys.platform == "win32"
        else f"kill {os.getpid()}"
    )
    assert holder.command == expected
    assert record.path.parent == data_dir / SERVING_RECORD_DIRNAME

    record.release()
    assert agent_holder(tmp_path, "Emma", data_dir) is None


def test_the_launcher_records_come_first(tmp_path, data_dir):
    """A managed host needs `kestrel terminate`, not a raw kill (#2920)."""
    ServingRecord.acquire(data_dir)
    host_pid = tmp_path / "logs" / ".host.pid"
    ProcessManager.write_pid(host_pid, os.getpid(), port=8888)

    holder = agent_holder(tmp_path, "Emma", data_dir)

    assert holder.command == "kestrel terminate"
    assert f"host PID {os.getpid()}" in holder.evidence


def test_each_instance_owns_its_record(data_dir):
    """Two holders of one directory never remove each other's record."""
    first = ServingRecord.acquire(data_dir)
    second = ServingRecord.acquire(data_dir)
    assert first.path != second.path

    first.release()

    assert second.path.exists()
    assert serving_holder(data_dir) is not None
    second.release()
    assert serving_holder(data_dir) is None


def test_a_record_of_an_exited_process_is_stale_and_swept(data_dir):
    stale = ServingRecord.acquire(data_dir)
    # Same PID, another start instant: the process it named has exited and
    # its number now belongs to someone else.
    _rewrite(stale, started_at=1.0)
    assert serving_holder(data_dir) is None

    live = ServingRecord.acquire(data_dir)

    assert not stale.path.exists(), "acquiring sweeps records proven stale"
    assert live.path.exists()


@pytest.mark.parametrize(
    "fields, shown",
    [
        ({"host": "a-container"}, "on host 'a-container'"),
        ({"pid_namespace": "pid:[4026532999]"}, "in PID namespace"),
    ],
)
def test_a_record_written_elsewhere_counts_as_running(data_dir, fields, shown):
    """Its PID means nothing here, so it cannot be proven stopped.

    Probed locally, the recorded PID and start time read as stale.
    """
    record = ServingRecord.acquire(data_dir)
    _rewrite(record, started_at=1.0, **fields)

    holder = serving_holder(data_dir)

    assert holder is not None
    assert holder.command is None
    assert shown in holder.evidence
    assert str(record.path) in holder.remedy
    # ...and acquiring here does not sweep what it cannot judge.
    ServingRecord.acquire(data_dir)
    assert record.path.exists()


def _without(key):
    """A record's content without ``key``."""

    def content(payload):
        del payload[key]
        return json.dumps(payload)

    return content


def _with(**fields):
    """A record's content with ``fields`` replaced."""

    def content(payload):
        payload.update(fields)
        return json.dumps(payload)

    return content


def _with_literal(key, literal):
    """A record's content with ``key`` holding the JSON text ``literal``.

    For values ``json.dumps`` would not write as given: ``NaN``, the
    infinities, ``1e400`` (which parses as an infinity) and integers too
    large for a float.
    """

    def content(payload):
        payload[key] = "@@literal@@"
        return json.dumps(payload).replace('"@@literal@@"', literal)

    return content


def _as_text(text):
    """Content that is not a record at all."""
    return lambda payload: text


_OVERFLOWING_INTEGER = "1" + "0" * 400

#: Records that do not prove themselves stale: each identifying field missing,
#: holding a value that is not one, or naming somewhere else, and records that
#: are not records. Planted over a record that WOULD be provably stale, so the
#: field alone decides that it is kept.
_UNPROVEN = [
    pytest.param(_without("pid"), id="pid-missing"),
    pytest.param(_with(pid=None), id="pid-null"),
    pytest.param(_with(pid=str(os.getpid())), id="pid-string"),
    pytest.param(_with(pid=True), id="pid-bool"),
    pytest.param(_with(pid=float(os.getpid())), id="pid-float"),
    pytest.param(_with(pid=0), id="pid-zero"),
    pytest.param(_with(pid=-4), id="pid-negative"),
    pytest.param(_with(pid=_MAX_PID + 1), id="pid-beyond-platform-range"),
    pytest.param(_with_literal("pid", "1e400"), id="pid-overflowing-number"),
    pytest.param(_without("started_at"), id="started-at-missing"),
    pytest.param(_with(started_at=None), id="started-at-null"),
    pytest.param(_with(started_at="1.0"), id="started-at-string"),
    pytest.param(_with(started_at=True), id="started-at-bool"),
    pytest.param(_with(started_at=0), id="started-at-zero"),
    pytest.param(_with(started_at=-1.0), id="started-at-negative"),
    pytest.param(_with_literal("started_at", "NaN"), id="started-at-nan"),
    pytest.param(_with_literal("started_at", "Infinity"), id="started-at-infinity"),
    pytest.param(
        _with_literal("started_at", "-Infinity"), id="started-at-negative-infinity"
    ),
    pytest.param(
        _with_literal("started_at", "1e400"), id="started-at-overflowing-number"
    ),
    pytest.param(
        _with_literal("started_at", _OVERFLOWING_INTEGER),
        id="started-at-overflowing-integer",
    ),
    pytest.param(_without("host"), id="host-missing"),
    pytest.param(_with(host=None), id="host-null"),
    pytest.param(_with(host=42), id="host-non-string"),
    pytest.param(_with(host=""), id="host-empty"),
    pytest.param(_with(host="a-container"), id="host-foreign"),
    pytest.param(_without("pid_namespace"), id="pid-namespace-missing"),
    pytest.param(_with(pid_namespace=7), id="pid-namespace-non-string"),
    pytest.param(_with(pid_namespace=""), id="pid-namespace-empty"),
    pytest.param(
        _with(pid_namespace="pid:[4026532999]"), id="pid-namespace-foreign"
    ),
    pytest.param(_as_text(str(os.getpid())), id="bare-pid"),
    pytest.param(_as_text("[1, 2]"), id="not-an-object"),
    pytest.param(_as_text("{half a record"), id="not-json"),
    pytest.param(
        _as_text('{"pid": ' + "1" * 5000 + "}"), id="integer-too-long-to-parse"
    ),
    pytest.param(_as_text("[" * 100_000 + "]" * 100_000), id="nested-too-deep"),
]

#: What a probe of the recorded PID finds, each the proof that a local,
#: well-formed record is stale: nothing runs as that PID, or what does started
#: at another instant than the planted ``started_at`` of 1.0.
_STALE_PROBES = [
    pytest.param(lambda pid: (False, None), id="pid-absent"),
    pytest.param(lambda pid: (True, 1000.0), id="pid-reused"),
]


def _plant(record: ServingRecord, content) -> None:
    """Rewrite a record so that, probed here, the PID it names reads as stale."""
    payload = json.loads(record.path.read_text())
    payload["started_at"] = 1.0
    record.path.write_text(content(payload))


def _boot_sweep(data_dir) -> None:
    """What every booting agent does first: acquire its own record."""
    ServingRecord.acquire(data_dir).release()


@pytest.mark.parametrize("probe", _STALE_PROBES)
@pytest.mark.parametrize("content", _UNPROVEN)
def test_a_record_not_proven_stale_is_kept_and_held(
    tmp_path, data_dir, content, probe, monkeypatch
):
    """Deleted, or read as stopped, only on positive proof that it is stale.

    A record missing a field, holding a value that is not one, or written
    somewhere else may name a live process, so its PID proves nothing about
    this process table (#3522).
    """
    record = ServingRecord.acquire(data_dir)
    _plant(record, content)
    monkeypatch.setattr(ProcessManager, "_probe_process", staticmethod(probe))

    _boot_sweep(data_dir)

    assert record.path.exists()
    holder = _serving_record_holder(record.path)
    assert holder is not None
    assert not holder.verified
    assert holder.command is None
    assert str(record.path) in holder.remedy
    # ...so the guards keep refusing an offline write or an update.
    assert agent_holder(tmp_path, "Emma", data_dir) is not None
    cfg = SimpleNamespace(data_dir=Path("agent_data/emma"))
    assert cli._agent_holder(tmp_path, "Emma", cfg) is not None


@pytest.mark.parametrize("probe", _STALE_PROBES)
def test_a_record_proven_stale_is_removed(tmp_path, data_dir, probe, monkeypatch):
    """Local, every field valid, and its PID absent or reused: deleted."""
    record = ServingRecord.acquire(data_dir)
    _plant(record, _with())
    monkeypatch.setattr(ProcessManager, "_probe_process", staticmethod(probe))

    assert _serving_record_holder(record.path) is None
    assert agent_holder(tmp_path, "Emma", data_dir) is None

    _boot_sweep(data_dir)

    assert not record.path.exists()


def test_the_boot_sweep_deletes_only_a_local_record_naming_a_stale_pid(data_dir):
    """Probed for real: of all the planted records, one is proven stale."""
    kept = []
    for param in _UNPROVEN:
        record = ServingRecord.acquire(data_dir)
        _plant(record, param.values[0])
        kept.append(record.path)
    stale = ServingRecord.acquire(data_dir)
    _plant(stale, _with())

    live = ServingRecord.acquire(data_dir)

    assert not stale.path.exists()
    assert live.path.exists()
    assert all(path.exists() for path in kept)


def test_a_record_naming_no_host_is_not_local_even_on_a_nameless_host(
    data_dir, monkeypatch
):
    """An empty host names no machine, so matching it proves nothing."""
    monkeypatch.setattr(liveness, "_here", lambda: ("", None))
    record = ServingRecord.acquire(data_dir)
    _plant(record, _with(host="", pid_namespace=None))

    _boot_sweep(data_dir)

    assert record.path.exists()
    assert not _serving_record_holder(record.path).verified


def test_the_guard_and_the_sweep_share_one_predicate(data_dir, monkeypatch):
    """They cannot disagree about which records are stale."""
    record = ServingRecord.acquire(data_dir)
    judged = []
    verdict = liveness.AgentHolder(evidence="judged", remedy="", verified=False)

    def judge(path, payload):
        judged.append(path)
        return verdict

    monkeypatch.setattr(liveness, "_record_holder", judge)

    assert _serving_record_holder(record.path) is verdict
    liveness._remove_stale_records(record.path.parent)
    assert record.path.exists()

    verdict = None
    assert _serving_record_holder(record.path) is None
    liveness._remove_stale_records(record.path.parent)
    assert not record.path.exists()
    assert judged == [record.path] * 4


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX symbolic links")
def test_a_record_linking_to_nothing_is_held_not_released(tmp_path, data_dir):
    """Only an entry that is gone was released; a dangling link is unreadable."""
    directory = data_dir / SERVING_RECORD_DIRNAME
    directory.mkdir()
    link = directory / "4242-dangling.json"
    link.symlink_to(directory / "missing-target.json")

    _boot_sweep(data_dir)

    assert link.is_symlink()
    holder = serving_holder(data_dir)
    assert holder is not None
    assert not holder.verified
    assert "cannot be read" in holder.evidence
    assert agent_holder(tmp_path, "Emma", data_dir) is not None


def test_the_guard_and_the_sweep_list_the_same_records(data_dir):
    """A staging-style name is neither read by the guard nor swept."""
    record = ServingRecord.acquire(data_dir)
    hidden = record.path.parent / f".{record.path.name}"
    hidden.write_text(record.path.read_text())
    _plant(record, _with())
    hidden.write_text(record.path.read_text())

    assert liveness._record_paths(record.path.parent) == [record.path]
    _boot_sweep(data_dir)

    assert not record.path.exists()
    assert hidden.exists()
    assert serving_holder(data_dir) is None


def test_an_unreadable_record_counts_as_running(data_dir):
    record = ServingRecord.acquire(data_dir)
    record.path.write_text("{half a record")

    holder = serving_holder(data_dir)

    assert holder is not None
    assert "cannot be read" in holder.evidence
    ServingRecord.acquire(data_dir)
    assert record.path.exists()


def test_a_process_whose_identity_cannot_be_checked_counts_as_running(
    data_dir, monkeypatch
):
    ServingRecord.acquire(data_dir)
    monkeypatch.setattr(
        ProcessManager, "_probe_process", staticmethod(lambda pid: (True, None))
    )

    holder = serving_holder(data_dir)

    assert holder is not None
    assert holder.command is None
    assert "identity cannot be established" in holder.evidence


def test_a_verified_record_is_preferred_over_an_unprovable_one(data_dir):
    unprovable = ServingRecord.acquire(data_dir)
    _rewrite(unprovable, host="a-container")
    ServingRecord.acquire(data_dir)

    holder = serving_holder(data_dir)

    assert holder.command is not None


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions that bind a non-root user",
)
def test_records_that_cannot_be_listed_count_as_running(data_dir):
    directory = data_dir / SERVING_RECORD_DIRNAME
    directory.mkdir()
    directory.chmod(0)
    try:
        holder = serving_holder(data_dir)
    finally:
        directory.chmod(0o700)

    assert holder is not None
    assert "cannot be listed" in holder.evidence


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\xff\xfe not a pid",
        b'{"pid": 1e400}',
        b'{"pid": true}',
        b'{"pid": -4}',
        b"0",
        b'{"pid": %(pid)d, "started_at": NaN}',
        b'{"pid": %(pid)d, "started_at": Infinity}',
        b'{"pid": %(pid)d, "started_at": 1e400}',
        b'{"pid": %(pid)d, "started_at": "1.0"}',
    ],
    ids=[
        "empty",
        "not-utf8",
        "overflowing-pid",
        "bool-pid",
        "negative-pid",
        "zero-bare-pid",
        "nan-started-at",
        "infinite-started-at",
        "overflowing-started-at",
        "string-started-at",
    ],
)
def test_an_unreadable_launcher_record_counts_as_running(
    tmp_path, data_dir, content
):
    """The launch it records may still be running; that cannot be told.

    A start time that is not one names this live process, so compared with
    its real start it used to read as another process: stale (#3522).
    """
    host_pid = tmp_path / "logs" / ".host.pid"
    host_pid.parent.mkdir()
    host_pid.write_bytes(content.replace(b"%(pid)d", str(os.getpid()).encode()))

    holder = agent_holder(tmp_path, "Emma", data_dir)

    assert holder is not None
    assert not holder.verified
    # `kestrel terminate` cannot signal a PID it cannot read.
    assert holder.command is None
    assert "`kestrel terminate`" in holder.remedy
    assert str(host_pid) in holder.remedy


def test_an_unreadable_agent_record_does_not_hide_a_running_host(
    tmp_path, data_dir
):
    """The holder a working command stops is the one reported."""
    ProcessManager.agent_pid_file(data_dir).write_text("")
    ProcessManager.write_pid(tmp_path / "logs" / ".host.pid", os.getpid())

    holder = agent_holder(tmp_path, "Emma", data_dir)

    assert holder.verified
    assert holder.command == "kestrel terminate"


def test_a_verified_serving_record_wins_over_an_unreadable_launcher_record(
    tmp_path, data_dir
):
    ProcessManager.agent_pid_file(data_dir).write_text("")
    ServingRecord.acquire(data_dir)

    holder = agent_holder(tmp_path, "Emma", data_dir)

    assert holder.verified
    assert holder.command is not None


def test_a_record_without_a_pid_or_host_counts_as_running(data_dir):
    record = ServingRecord.acquire(data_dir)
    record.path.write_text(json.dumps({"started_at": 1.0}))

    holder = serving_holder(data_dir)

    assert holder is not None
    assert not holder.verified
    assert "does not record a valid PID, start time, host and PID namespace" in (
        holder.evidence
    )


def test_a_record_is_published_whole(data_dir):
    """No staging file is left to read as an unreadable record."""
    record = ServingRecord.acquire(data_dir)

    assert sorted(p.name for p in record.path.parent.iterdir()) == [
        record.path.name
    ]
    payload = json.loads(record.path.read_text())
    assert payload["pid"] == os.getpid()
    assert payload["data_dir"] == str(data_dir)


def test_the_reanchor_guard_refuses_a_directly_launched_agent(
    tmp_path, data_dir, capsys
):
    """The same reading guards an offline database write."""
    cfg = SimpleNamespace(data_dir=Path("agent_data/emma"))
    ServingRecord.acquire(data_dir)

    holder = cli._agent_holder(tmp_path, "Emma", cfg)
    cli._report_agent_running("Emma", holder)

    err = capsys.readouterr().err
    assert "agent 'Emma' appears to be running" in err
    assert f"PID {os.getpid()} serves it" in err
    assert "to avoid DB corruption" in err
