"""Computer-use runtime files belong to the agent, and captures age out (#3279).

``audit_log_path`` and ``capture_dir`` were bare relative paths resolved
against the host process's cwd. The shipped per-agent config says the audit
path "is relative to Emma's storage dir"; in fact the live host wrote every
agent's rows -- four DIDs, 1,688 rows -- into one file in the source checkout,
and per-agent settings collided instead of separating. #3243's uncapped
captures went through the same code and nothing pruned them.

These tests drive real files: the defect was about where bytes landed, so a
test that only inspects a computed path would not show it is fixed.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.computer_use import capture
from kestrel_sovereign.features.computer_use.feature import (
    _DEFAULT_CAPTURE_RETENTION_DAYS,
    ComputerUseFeature,
    _config_int,
)
from kestrel_sovereign.privacy import PrivacyConfig

DAY = 24 * 60 * 60


class FakeApprovalQueue:
    async def request_approval(self, feature_name, tool_name, tool_args, timeout):
        return True, "once"


class FakeSecurityFeature:
    def __init__(self):
        self.approval_queue = FakeApprovalQueue()


class FakeAgent:
    """Models what the feature reads off a real ``KestrelAgent``.

    ``storage_path`` is the database FILE, as every production launcher
    passes it (``agent_data/<Agent>/kestrel_prime.db``).
    """

    def __init__(self, *, did: str = "did:test:agent", storage_path: Any = None):
        self.did = did
        if storage_path is not None:
            self.storage_path = storage_path
        self.privacy_config = PrivacyConfig(computer_access=True)
        self.granted_capabilities = frozenset(
            {
                "filesystem_read",
                "shell_execution_sandboxed",
                "shell_execution_host",
            }
        )
        self.features = {"security": FakeSecurityFeature()}

    def get_feature(self, name):
        return self.features.get(name)


def _agent_dir(root: Path, name: str) -> Path:
    d = root / "agent_data" / name
    d.mkdir(parents=True)
    return d


def _enabled(**over: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "backend": "local",
        "allowed_paths": [],
        "deny_paths": [],
        "auto_approved_binaries": ["echo"],
        "denied_binaries": ["rm"],
    }
    cfg.update(over)
    return cfg


async def _feature(agent: FakeAgent, cfg: dict[str, Any]) -> ComputerUseFeature:
    f = ComputerUseFeature(agent)
    f._cfg = cfg
    await f.initialize()
    return f


def _artifact_set(d: Path, *, age_days: float, run_id: str | None = None) -> list[Path]:
    """One capture's three files, all last written ``age_days`` ago."""
    rid = run_id or uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    paths = [d / f"{rid}.stdout", d / f"{rid}.stderr", d / f"{rid}.json"]
    for p in paths:
        p.write_text("x")
    _age(*paths, days=age_days)
    return paths


def _age(*paths: Path, days: float) -> None:
    t = time.time() - days * DAY
    for p in paths:
        os.utime(p, (t, t), follow_symlinks=False)


@pytest.fixture()
def elsewhere(tmp_path: Path, monkeypatch) -> Path:
    """A host cwd that is NOT any agent's storage dir, like the checkout."""
    cwd = tmp_path / "checkout"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


# --- resolution ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_relative_audit_log_path_lands_in_the_agents_storage_dir(
    tmp_path: Path, elsewhere: Path
):
    emma = _agent_dir(tmp_path, "Emma")
    agent = FakeAgent(did="did:test:emma", storage_path=str(emma / "kestrel_prime.db"))
    f = await _feature(
        agent, {"enabled": False, "audit_log_path": ".kestrel/computer_use_audit.jsonl"}
    )

    # A disabled-call refusal is audited, so a real row has to land somewhere.
    env = await f.fs_read(path=str(tmp_path / "anything.txt"))
    assert env.error.startswith("readiness:")

    audit = emma / ".kestrel" / "computer_use_audit.jsonl"
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [r["agent_did"] for r in rows] == ["did:test:emma"]
    assert not (elsewhere / ".kestrel").exists(), "nothing may land in the host cwd"


@pytest.mark.asyncio
async def test_a_relative_capture_dir_lands_in_the_agents_storage_dir(
    tmp_path: Path, elsewhere: Path
):
    emma = _agent_dir(tmp_path, "Emma")
    agent = FakeAgent(storage_path=str(emma / "kestrel_prime.db"))
    f = await _feature(
        agent,
        _enabled(
            audit_log_path=str(tmp_path / "audit.jsonl"),
            capture_dir=".kestrel/computer_use_captures",
        ),
    )

    env = await f.shell(command="echo captured", capture_output=True, timeout=10)

    assert env.status is ToolResultStatus.OK, env.error
    manifest = Path(env.data["manifest_path"])
    assert manifest.parent == emma / ".kestrel" / "computer_use_captures"
    assert Path(env.data["stdout_path"]).read_text() == "captured\n"
    assert not (elsewhere / ".kestrel").exists(), "nothing may land in the host cwd"


@pytest.mark.asyncio
async def test_the_defaults_resolve_against_the_storage_dir(tmp_path: Path, elsewhere: Path):
    emma = _agent_dir(tmp_path, "Emma")
    f = await _feature(FakeAgent(storage_path=str(emma / "kestrel_prime.db")), _enabled())

    assert f._audit.path == emma / ".kestrel" / "computer_use_audit.jsonl"
    assert f._capture_dir == emma / ".kestrel" / "computer_use_captures"


@pytest.mark.asyncio
async def test_a_relative_storage_path_is_pinned_absolute_at_initialize(
    tmp_path: Path, monkeypatch
):
    """``KESTREL_DB_PATH`` may be relative. The paths are pinned when the
    feature starts, so a later change of the process cwd cannot move them."""
    monkeypatch.chdir(tmp_path)
    _agent_dir(tmp_path, "Emma")
    f = await _feature(
        FakeAgent(storage_path="agent_data/Emma/kestrel_prime.db"), _enabled()
    )

    assert f._audit.path == tmp_path / "agent_data" / "Emma" / ".kestrel" / "computer_use_audit.jsonl"
    assert f._capture_dir.is_absolute()


@pytest.mark.asyncio
async def test_absolute_paths_are_used_unchanged(tmp_path: Path, elsewhere: Path):
    emma = _agent_dir(tmp_path, "Emma")
    audit = tmp_path / "ops" / "audit.jsonl"
    captures = tmp_path / "ops" / "captures"
    f = await _feature(
        FakeAgent(storage_path=str(emma / "kestrel_prime.db")),
        _enabled(audit_log_path=str(audit), capture_dir=str(captures)),
    )

    assert f._audit.path == audit
    assert f._capture_dir == captures
    env = await f.shell(command="echo hi", capture_output=True, timeout=10)
    assert Path(env.data["manifest_path"]).parent == captures
    assert audit.is_file()


@pytest.mark.asyncio
async def test_home_relative_paths_are_used_unchanged(
    tmp_path: Path, elsewhere: Path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    emma = _agent_dir(tmp_path, "Emma")
    f = await _feature(
        FakeAgent(storage_path=str(emma / "kestrel_prime.db")),
        _enabled(
            audit_log_path="~/kestrel/audit.jsonl",
            capture_dir="~/kestrel/captures",
        ),
    )

    assert f._audit.path == home / "kestrel" / "audit.jsonl"
    assert f._capture_dir == home / "kestrel" / "captures"


@pytest.mark.asyncio
async def test_two_agents_with_the_default_config_get_separate_audit_files(
    tmp_path: Path, elsewhere: Path
):
    """The measured defect: four agents, one interleaved file in the
    checkout. Each agent reads its own per-agent kestrel.toml, which sets
    neither path, exactly as the live host does."""
    features = []
    for name in ("Emma", "Kite"):
        d = _agent_dir(tmp_path, name)
        (d / "kestrel.toml").write_text("[features.computer_use]\nenabled = false\n")
        f = ComputerUseFeature(
            FakeAgent(did=f"did:test:{name.lower()}", storage_path=str(d / "kestrel_prime.db"))
        )
        await f.initialize()  # through the loader, not a pre-populated _cfg
        await f.fs_read(path=str(tmp_path / "anything.txt"))
        features.append((name, d, f))

    for name, d, f in features:
        audit = d / ".kestrel" / "computer_use_audit.jsonl"
        assert f._audit.path == audit
        dids = {json.loads(line)["agent_did"] for line in audit.read_text().splitlines()}
        assert dids == {f"did:test:{name.lower()}"}, f"{name}'s audit holds {dids}"
    assert not (elsewhere / ".kestrel").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_path", [None, "", MagicMock()])
async def test_a_relative_path_with_no_storage_dir_is_refused_not_put_in_the_cwd(
    tmp_path: Path, elsewhere: Path, caplog, storage_path
):
    """No fallback: "somewhere shared" is the defect. A MagicMock agent
    answers every attribute and is even a PathLike, so it must not count."""
    agent = FakeAgent()
    if storage_path is not None:
        agent.storage_path = storage_path
    with caplog.at_level(logging.ERROR):
        f = await _feature(agent, _enabled())

    assert f._backend is None
    assert f._audit is None
    env = await f.fs_read(path=str(tmp_path / "anything.txt"))
    assert env.error.startswith("readiness:")
    assert not (elsewhere / ".kestrel").exists()
    assert any(
        r.levelno == logging.ERROR and "audit_log_path" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_a_relative_capture_dir_with_no_storage_dir_is_refused_too(
    tmp_path: Path, elsewhere: Path
):
    """The audit log alone being placeable is not enough: the capture
    directory must not fall back to the cwd either. The audit log it could
    place still records the refusals."""
    audit = tmp_path / "audit.jsonl"
    f = await _feature(FakeAgent(), _enabled(audit_log_path=str(audit)))

    assert f._backend is None
    assert f._capture_dir is None
    assert not (elsewhere / ".kestrel").exists()
    env = await f.fs_read(path=str(tmp_path / "anything.txt"))
    assert env.error.startswith("readiness:")
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [r["allowed_by"] for r in rows] == [["denied:readiness:backend"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [7, "", None, "captures\x00elsewhere"])
async def test_an_unusable_path_value_is_refused(tmp_path: Path, elsewhere: Path, value):
    emma = _agent_dir(tmp_path, "Emma")
    f = await _feature(
        FakeAgent(storage_path=str(emma / "kestrel_prime.db")),
        _enabled(capture_dir=value),
    )

    assert f._backend is None


# --- a refused re-initialisation fails closed (#3476) ---------------------------
#
# A refusal used to return before touching anything, so the backend built by an
# earlier, good initialize stayed live: readiness passed and calls kept running
# while the log said no computer-use call would run.


async def _working_feature(
    tmp_path: Path, agent: FakeAgent | None = None, **over: Any
) -> tuple[ComputerUseFeature, Path]:
    """A feature whose first initialize succeeded and whose tools run."""
    emma = _agent_dir(tmp_path, "Emma")
    work = tmp_path / "work"
    work.mkdir()
    (work / "note.txt").write_text("hello")
    if agent is None:
        agent = FakeAgent(storage_path=str(emma / "kestrel_prime.db"))
    else:
        agent.storage_path = str(emma / "kestrel_prime.db")
    f = await _feature(agent, _enabled(allowed_paths=[str(work)], **over))
    envs = await _calls(f, work)
    assert [e.status for e in envs] == [ToolResultStatus.OK] * 3, [e.error for e in envs]
    return f, work


async def _calls(f: ComputerUseFeature, work: Path) -> list:
    return [
        await f.shell(command="echo STILL_RUNS", timeout=10),
        await f.fs_read(path=str(work / "note.txt")),
        await f.fs_list(path=str(work)),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["capture_dir", "audit_log_path"])
async def test_a_refused_reinitialise_leaves_nothing_running(
    tmp_path: Path, elsewhere: Path, key: str
):
    f, work = await _working_feature(tmp_path)

    f._cfg[key] = ""
    await f.initialize()

    assert f._backend is None
    assert f._capture_dir is None
    for env in await _calls(f, work):
        assert env.status is ToolResultStatus.ERROR
        assert env.error.startswith("readiness:backend not initialized: ")
        # The caller is told why, not only that it is unready.
        assert f"features.computer_use.{key}" in env.error


@pytest.mark.asyncio
async def test_a_refused_reinitialise_audits_its_refusals_with_the_reason(
    tmp_path: Path, elsewhere: Path
):
    """The audit log could still be placed, so it records the refusals."""
    f, work = await _working_feature(tmp_path)

    f._cfg["capture_dir"] = ""
    await f.initialize()
    await _calls(f, work)

    rows = [json.loads(line) for line in f._audit.path.read_text().splitlines()][-3:]
    assert [r["allowed_by"] for r in rows] == [["denied:readiness:backend"]] * 3
    assert all("features.computer_use.capture_dir" in r["error"] for r in rows)


@pytest.mark.asyncio
async def test_a_refused_reinitialise_writes_nothing_to_the_audit_log_it_no_longer_names(
    tmp_path: Path, elsewhere: Path
):
    """A refused first initialize with this config has no audit log, and a
    refused re-initialise must leave the same state, not the old file."""
    f, work = await _working_feature(tmp_path)
    old_audit = f._audit.path
    before = old_audit.read_bytes()

    f._cfg["audit_log_path"] = ""
    await f.initialize()
    await _calls(f, work)

    assert f._audit is None
    assert old_audit.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["capture_dir", "audit_log_path"])
async def test_a_good_initialise_after_a_refused_one_restores_service(
    tmp_path: Path, elsewhere: Path, key: str
):
    f, work = await _working_feature(tmp_path)
    f._cfg[key] = ""
    await f.initialize()

    del f._cfg[key]
    await f.initialize()

    envs = await _calls(f, work)
    assert [e.status for e in envs] == [ToolResultStatus.OK] * 3, [e.error for e in envs]
    env = await f.shell(command="echo back", capture_output=True, timeout=10)
    assert env.status is ToolResultStatus.OK, env.error
    assert Path(env.data["stdout_path"]).read_text() == "back\n"


@pytest.mark.asyncio
async def test_a_backend_refusal_on_reinitialise_is_reported_to_the_caller(
    tmp_path: Path, elsewhere: Path
):
    agent = FakeAgent()
    f, work = await _working_feature(tmp_path, agent)

    agent.granted_capabilities = frozenset({"filesystem_read"})
    await f.initialize()

    assert f._backend is None
    for env in await _calls(f, work):
        assert env.status is ToolResultStatus.ERROR
        assert "backend refused init" in env.error


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown_first", [False, True], ids=["reinit", "shutdown-reinit"])
async def test_the_previous_backend_is_shut_down_exactly_once(
    tmp_path: Path, elsewhere: Path, shutdown_first: bool
):
    f, _ = await _working_feature(tmp_path)
    old = f._backend
    shut: list[Any] = []

    async def _shutdown() -> None:
        shut.append(old)

    old.shutdown = _shutdown

    if shutdown_first:
        await f.shutdown()
    f._cfg["capture_dir"] = ""
    await f.initialize()
    await f.shutdown()

    assert shut == [old]


@pytest.mark.asyncio
async def test_no_call_runs_after_shutdown(tmp_path: Path, elsewhere: Path):
    f, work = await _working_feature(tmp_path)

    await f.shutdown()

    for env in await _calls(f, work):
        assert env.error.startswith("readiness:backend not initialized")


class _ReinitialisingApprovalQueue:
    """Re-initialises the feature while a call waits for its approval."""

    def __init__(self, change):
        self.change = change
        self.feature: ComputerUseFeature | None = None

    async def request_approval(self, feature_name, tool_name, tool_args, timeout):
        self.change(self.feature._cfg)
        await self.feature.initialize()
        return True, "once"


def _refuse(cfg: dict[str, Any]) -> None:
    cfg["capture_dir"] = ""


def _rebuild(cfg: dict[str, Any]) -> None:
    """Same configuration: a new backend these gates never evaluated."""


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [_refuse, _rebuild], ids=["refused", "rebuilt"])
async def test_a_call_authorised_across_a_reinitialise_does_not_run(
    tmp_path: Path, elsewhere: Path, change
):
    emma = _agent_dir(tmp_path, "Emma")
    agent = FakeAgent(storage_path=str(emma / "kestrel_prime.db"))
    queue = _ReinitialisingApprovalQueue(change)
    agent.features["security"].approval_queue = queue
    audit = tmp_path / "audit.jsonl"
    f = await _feature(agent, _enabled(audit_log_path=str(audit)))
    queue.feature = f
    target = tmp_path / "ran"

    # ``touch`` is not auto-approved, so the call waits in the queue.
    env = await f.shell(command=f"touch {target}", timeout=10)

    assert env.status is ToolResultStatus.ERROR
    assert env.error.startswith("readiness:feature was re-initialised")
    assert not target.exists(), "the command ran"
    last = json.loads(audit.read_text().splitlines()[-1])
    assert last["allowed_by"][-1] == "denied:readiness:reinitialized"
    if change is _refuse:
        assert "features.computer_use.capture_dir" in env.error


# --- the legacy shared file -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_legacy_cwd_audit_log_is_named_once_and_left_untouched(
    tmp_path: Path, elsewhere: Path, caplog
):
    legacy = elsewhere / ".kestrel" / "computer_use_audit.jsonl"
    legacy.parent.mkdir()
    legacy.write_text('{"agent_did":"did:a"}\n{"agent_did":"did:b"}\n')
    before = legacy.read_bytes()
    emma = _agent_dir(tmp_path, "Emma")

    with caplog.at_level(logging.INFO):
        f = await _feature(FakeAgent(storage_path=str(emma / "kestrel_prime.db")), _enabled())

    notes = [r for r in caplog.records if "legacy" in r.getMessage()]
    assert len(notes) == 1
    assert notes[0].levelno == logging.INFO
    assert str(legacy) in notes[0].getMessage()
    assert str(f._audit.path) in notes[0].getMessage()
    assert legacy.read_bytes() == before, "the shared file must not be rewritten"


@pytest.mark.asyncio
async def test_no_legacy_note_when_there_is_no_legacy_file(
    tmp_path: Path, elsewhere: Path, caplog
):
    emma = _agent_dir(tmp_path, "Emma")
    with caplog.at_level(logging.INFO):
        await _feature(FakeAgent(storage_path=str(emma / "kestrel_prime.db")), _enabled())

    assert not [r for r in caplog.records if "legacy" in r.getMessage()]


@pytest.mark.asyncio
async def test_no_legacy_note_when_the_cwd_is_the_storage_dir(
    tmp_path: Path, monkeypatch, caplog
):
    """Then the cwd-relative file IS this agent's file, not a legacy one."""
    emma = _agent_dir(tmp_path, "Emma")
    audit = emma / ".kestrel" / "computer_use_audit.jsonl"
    audit.parent.mkdir()
    audit.write_text("")
    monkeypatch.chdir(emma)

    with caplog.at_level(logging.INFO):
        await _feature(FakeAgent(storage_path=str(emma / "kestrel_prime.db")), _enabled())

    assert not [r for r in caplog.records if "legacy" in r.getMessage()]


# --- capture retention: the pruning primitive -----------------------------------


def test_prune_removes_an_old_set_and_keeps_a_fresh_one(tmp_path: Path):
    d = tmp_path / "captures"
    old = _artifact_set(d, age_days=20)
    fresh = _artifact_set(d, age_days=1)

    removed = capture.prune(d, cutoff=time.time() - 14 * DAY)

    assert removed == 1
    assert not any(p.exists() for p in old)
    assert all(p.exists() for p in fresh)


def test_prune_never_touches_anything_outside_the_capture_dir(tmp_path: Path):
    d = tmp_path / "captures"
    d.mkdir()
    rid = uuid.uuid4().hex
    # A run-shaped set in the PARENT: same names, same age, wrong directory.
    beside = _artifact_set(tmp_path, age_days=30, run_id=rid)
    # A run-shaped set in a SUBDIRECTORY: pruning does not recurse.
    nested = _artifact_set(d / "nested", age_days=30)
    # Links inside the directory pointing out of it.
    target_manifest = tmp_path / "outside-manifest.json"
    target_stream = tmp_path / "outside-stream.txt"
    target_manifest.write_text("{}")
    target_stream.write_text("keep me")
    linked = uuid.uuid4().hex
    (d / f"{linked}.json").symlink_to(target_manifest)
    _age(d / f"{linked}.json", target_manifest, days=30)
    streams_linked = uuid.uuid4().hex
    (d / f"{streams_linked}.stdout").symlink_to(target_stream)
    (d / f"{streams_linked}.json").write_text("{}")
    _age(d / f"{streams_linked}.json", d / f"{streams_linked}.stdout", target_stream, days=30)

    capture.prune(d, cutoff=time.time() - 14 * DAY)

    assert all(p.exists() for p in beside)
    assert all(p.exists() for p in nested)
    # A linked manifest is skipped rather than followed.
    assert (d / f"{linked}.json").is_symlink()
    assert target_manifest.read_text() == "{}"
    # A linked stream goes as a link; what it points at stays.
    assert not (d / f"{streams_linked}.stdout").is_symlink()
    assert target_stream.read_text() == "keep me"


def test_prune_only_considers_runtime_named_artifacts(tmp_path: Path):
    """An operator's own old file in the directory is never a candidate."""
    d = tmp_path / "captures"
    d.mkdir()
    keep = [d / "notes.json", d / "review.stdout", d / f"{uuid.uuid4().hex.upper()}.json"]
    for p in keep:
        p.write_text("mine")
    _age(*keep, days=365)

    assert capture.prune(d, cutoff=time.time() - 14 * DAY) == 0
    assert all(p.exists() for p in keep)


def test_prune_keeps_a_run_still_in_flight(tmp_path: Path):
    """Streams with no manifest yet are a run that has not finished."""
    d = tmp_path / "captures"
    rid = uuid.uuid4().hex
    streams = _artifact_set(d, age_days=30, run_id=rid)[:2]
    (d / f"{rid}.json").unlink()

    assert capture.prune(d, cutoff=time.time() - 14 * DAY) == 0
    assert all(p.exists() for p in streams)


def test_prune_never_deletes_a_protected_file(tmp_path: Path):
    """The audit log is protected whatever it is called and wherever it is.
    The rest of its set still goes."""
    d = tmp_path / "captures"
    stdout, stderr, manifest = _artifact_set(d, age_days=30)

    removed = capture.prune(d, cutoff=time.time() - 14 * DAY, protected=[manifest])

    assert manifest.read_text() == "x"
    assert not stdout.exists() and not stderr.exists()
    assert removed == 0, "a set that kept a member was not pruned"


@pytest.mark.parametrize("dangling", [False, True])
def test_a_protected_link_inside_the_capture_dir_keeps_its_own_entry(
    tmp_path: Path, dangling: bool
):
    """``audit_log_path`` names the entry the log is written through. If that
    entry is a link sitting under a run-shaped name, removing it would cut
    the next write off from every earlier row."""
    d = tmp_path / "captures"
    rid = uuid.uuid4().hex
    _, _, manifest = _artifact_set(d, age_days=30, run_id=rid)
    target = tmp_path / "real-audit.jsonl"
    if not dangling:
        target.write_text("rows")
    entry = d / f"{rid}.stdout"
    entry.unlink()
    entry.symlink_to(target)

    capture.prune(d, cutoff=time.time() - 14 * DAY, protected=[entry])

    assert entry.is_symlink()
    assert not manifest.exists()


def test_prune_protects_by_identity_not_by_spelling(tmp_path: Path):
    """On a case-insensitive filesystem (the macOS and Windows default) one
    file has several spellings. A string comparison would miss the one the
    pruner builds from the lowercase manifest name."""
    d = tmp_path / "captures"
    d.mkdir()
    rid = uuid.uuid4().hex
    audit = d / f"{rid.upper()}.stdout"
    audit.write_text("rows")
    if not (d / f"{rid}.stdout").exists():
        pytest.skip("case-sensitive filesystem: one spelling per file")
    manifest = d / f"{rid}.json"
    manifest.write_text("{}")
    _age(audit, manifest, days=30)

    capture.prune(d, cutoff=time.time() - 14 * DAY, protected=[audit])

    assert audit.read_text() == "rows"
    assert not manifest.exists()


def test_a_protected_path_that_is_a_link_protects_its_target(tmp_path: Path):
    """``audit_log_path`` may name a link. The file it writes through is the
    one that must survive, wherever that file sits."""
    d = tmp_path / "captures"
    rid = uuid.uuid4().hex
    stdout, _, manifest = _artifact_set(d, age_days=30, run_id=rid)
    link = tmp_path / "audit.jsonl"
    link.symlink_to(stdout)

    capture.prune(d, cutoff=time.time() - 14 * DAY, protected=[link])

    assert stdout.read_text() == "x"
    assert not manifest.exists()


def test_a_link_to_a_protected_file_goes_and_the_file_stays(tmp_path: Path):
    """What is unlinked is the directory entry. Removing a link cannot harm
    the file it points at, so the link does not pin its set forever."""
    d = tmp_path / "captures"
    rid = uuid.uuid4().hex
    _, _, manifest = _artifact_set(d, age_days=30, run_id=rid)
    audit = tmp_path / "audit.jsonl"
    audit.write_text("rows")
    (d / f"{rid}.stdout").unlink()
    (d / f"{rid}.stdout").symlink_to(audit)

    assert capture.prune(d, cutoff=time.time() - 14 * DAY, protected=[audit]) == 1

    assert not (d / f"{rid}.stdout").is_symlink()
    assert not manifest.exists()
    assert audit.read_text() == "rows"


def test_prune_of_a_missing_dir_is_a_no_op(tmp_path: Path):
    assert capture.prune(tmp_path / "never-created", cutoff=time.time()) == 0


def test_prune_keeps_the_manifest_when_a_stream_cannot_be_removed(tmp_path: Path):
    """Manifest last: an interrupted set stays recognisable to the next prune."""
    d = tmp_path / "captures"
    rid = uuid.uuid4().hex
    old = _artifact_set(d, age_days=30, run_id=rid)
    (d / f"{rid}.stderr").unlink()
    (d / f"{rid}.stderr").mkdir()  # unlink() of a directory fails

    assert capture.prune(d, cutoff=time.time() - 14 * DAY) == 0
    assert old[-1].exists()


# --- capture retention: the feature's policy ------------------------------------


@pytest.mark.asyncio
async def test_initialize_prunes_old_capture_sets(tmp_path: Path, elsewhere: Path):
    emma = _agent_dir(tmp_path, "Emma")
    captures = emma / ".kestrel" / "computer_use_captures"
    old = _artifact_set(captures, age_days=_DEFAULT_CAPTURE_RETENTION_DAYS + 1)
    fresh = _artifact_set(captures, age_days=_DEFAULT_CAPTURE_RETENTION_DAYS - 1)

    await _feature(FakeAgent(storage_path=str(emma / "kestrel_prime.db")), _enabled())

    assert not any(p.exists() for p in old)
    assert all(p.exists() for p in fresh)


@pytest.mark.asyncio
async def test_every_initialize_prunes_not_only_the_first(tmp_path: Path, elsewhere: Path):
    emma = _agent_dir(tmp_path, "Emma")
    f = await _feature(FakeAgent(storage_path=str(emma / "kestrel_prime.db")), _enabled())
    old = _artifact_set(f._capture_dir, age_days=30)

    await f.initialize()

    assert not any(p.exists() for p in old)


@pytest.mark.asyncio
async def test_a_disabled_feature_still_ages_out_its_captures(
    tmp_path: Path, elsewhere: Path
):
    emma = _agent_dir(tmp_path, "Emma")
    old = _artifact_set(emma / ".kestrel" / "computer_use_captures", age_days=30)

    await _feature(FakeAgent(storage_path=str(emma / "kestrel_prime.db")), {"enabled": False})

    assert not any(p.exists() for p in old)


@pytest.mark.asyncio
async def test_retention_is_configurable(tmp_path: Path, elsewhere: Path):
    emma = _agent_dir(tmp_path, "Emma")
    captures = emma / ".kestrel" / "computer_use_captures"
    three_days = _artifact_set(captures, age_days=3)
    one_day = _artifact_set(captures, age_days=1)

    await _feature(
        FakeAgent(storage_path=str(emma / "kestrel_prime.db")),
        _enabled(capture_retention_days=2),
    )

    assert not any(p.exists() for p in three_days)
    assert all(p.exists() for p in one_day)


@pytest.mark.asyncio
async def test_zero_retention_disables_pruning(tmp_path: Path, elsewhere: Path):
    emma = _agent_dir(tmp_path, "Emma")
    old = _artifact_set(emma / ".kestrel" / "computer_use_captures", age_days=400)

    await _feature(
        FakeAgent(storage_path=str(emma / "kestrel_prime.db")),
        _enabled(capture_retention_days=0),
    )

    assert all(p.exists() for p in old)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audit_name",
    [
        "audit.jsonl",
        # Named like a manifest, so only the audit log's protection saves it.
        f"{'a' * 32}.json",
    ],
)
async def test_an_audit_log_inside_the_capture_dir_survives(
    tmp_path: Path, elsewhere: Path, audit_name: str
):
    emma = _agent_dir(tmp_path, "Emma")
    f = await _feature(
        FakeAgent(storage_path=str(emma / "kestrel_prime.db")),
        _enabled(audit_log_path=f"captures/{audit_name}", capture_dir="captures"),
    )
    await f.shell(command="echo hi", timeout=10)
    _age(f._audit.path, days=400)
    _artifact_set(f._capture_dir, age_days=400)

    await f._prune_captures(force=True)

    assert f._audit.path.read_text().strip(), "the audit log must never be pruned"


@pytest.mark.asyncio
async def test_captures_after_initialize_prune_at_most_once_a_day(
    tmp_path: Path, elsewhere: Path
):
    emma = _agent_dir(tmp_path, "Emma")
    f = await _feature(FakeAgent(storage_path=str(emma / "kestrel_prime.db")), _enabled())
    stale = _artifact_set(f._capture_dir, age_days=30)

    # Within a day of the prune at initialize: a capture does not prune again.
    env = await f.shell(command="echo one", capture_output=True, timeout=10)
    assert env.status is ToolResultStatus.OK, env.error
    assert all(p.exists() for p in stale)

    # A day later, the next capture prunes -- and keeps the run it just made.
    f._last_capture_prune -= DAY + 1
    env = await f.shell(command="echo two", capture_output=True, timeout=10)
    assert env.status is ToolResultStatus.OK, env.error
    assert not any(p.exists() for p in stale)
    assert Path(env.data["manifest_path"]).exists()


@pytest.mark.asyncio
async def test_an_uncaptured_run_does_not_prune(tmp_path: Path, elsewhere: Path):
    """Only a capture adds to the directory, so only a capture checks it."""
    emma = _agent_dir(tmp_path, "Emma")
    f = await _feature(FakeAgent(storage_path=str(emma / "kestrel_prime.db")), _enabled())
    stale = _artifact_set(f._capture_dir, age_days=30)
    f._last_capture_prune -= DAY + 1

    await f.shell(command="echo hi", timeout=10)

    assert all(p.exists() for p in stale)


@pytest.mark.asyncio
async def test_a_retention_longer_than_the_epoch_prunes_nothing_and_does_not_crash(
    tmp_path: Path, elsewhere: Path
):
    """``days * 86400`` subtracted from a float overflows for a large enough
    int, and a config typo must not take initialize down."""
    emma = _agent_dir(tmp_path, "Emma")
    old = _artifact_set(emma / ".kestrel" / "computer_use_captures", age_days=400)

    f = await _feature(
        FakeAgent(storage_path=str(emma / "kestrel_prime.db")),
        _enabled(capture_retention_days=10**400),
    )

    assert f._capture_retention_days == 10**400
    assert all(p.exists() for p in old)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "storage_path",
    [
        "~nosuchuser3279/agent_data/Emma/kestrel_prime.db",
        "agent_data/Em\x00ma/kestrel_prime.db",
    ],
)
async def test_an_unreadable_storage_dir_is_refused_not_raised(
    tmp_path: Path, elsewhere: Path, storage_path: str
):
    """A home directory that cannot be determined, or a path no filesystem
    call accepts, must leave the feature unready, not abort the agent's
    boot."""
    f = await _feature(FakeAgent(storage_path=storage_path), _enabled())

    assert f._backend is None
    assert not (elsewhere / ".kestrel").exists()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (float("inf"), _DEFAULT_CAPTURE_RETENTION_DAYS),
        (float("nan"), _DEFAULT_CAPTURE_RETENTION_DAYS),
        (None, _DEFAULT_CAPTURE_RETENTION_DAYS),
        (0, 0),
        (30, 30),
        ("7", 7),
        (-1, _DEFAULT_CAPTURE_RETENTION_DAYS),
        ("forever", _DEFAULT_CAPTURE_RETENTION_DAYS),
    ],
)
def test_capture_retention_days_is_read_or_refused(configured, expected):
    assert (
        _config_int(
            configured,
            default=_DEFAULT_CAPTURE_RETENTION_DAYS,
            name="x",
            minimum=0,
        )
        == expected
    )
