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
from kestrel_sovereign.multi_agent.liveness import (
    SERVING_RECORD_DIRNAME,
    ServingRecord,
    agent_holder,
    serving_holder,
)
from kestrel_sovereign.multi_agent.process_manager import ProcessManager


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
    "content", [b"", b"\xff\xfe not a pid", b'{"pid": 1e400}'],
    ids=["empty", "not-utf8", "overflowing-pid"],
)
def test_an_unreadable_launcher_record_counts_as_running(
    tmp_path, data_dir, content
):
    """The launch it records may still be running; that cannot be told."""
    host_pid = tmp_path / "logs" / ".host.pid"
    host_pid.parent.mkdir()
    host_pid.write_bytes(content)

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
    assert "names no PID or host" in holder.evidence


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
