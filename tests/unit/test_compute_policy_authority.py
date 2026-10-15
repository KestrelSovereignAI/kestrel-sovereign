"""``ComputePolicy`` is enforced where it is reported (#2483).

Every setting the compute status tools report must change what the runtime
does, and the runtime must read it from the one policy instance status
reports. Before #2483 several settings were decorative: network approval,
trash retention, Docker allowance and the deletable prefixes changed nothing,
and the default timeout and ``trash_enabled`` reported values nothing used.

The table below gives each enforced setting two values and the behavior each
produces; a setting the runtime ignored produces the same behavior for both.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Tuple

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.compute import feature as feature_module
from kestrel_sovereign.features.compute.destructive_policy import (
    DEFAULT_DELETABLE_PREFIXES as EXPORTED_DEFAULT_DELETABLE_PREFIXES,
)
from kestrel_sovereign.features.compute.executors import (
    DockerExecutor,
    LocalExecutor,
    UvExecutor,
)
from kestrel_sovereign.features.compute.feature import ComputeFeature
from kestrel_sovereign.features.compute.models import (
    ALLOW_DOCKER_ENV,
    ALLOW_LOCAL_COMPUTE_ENV,
    AUTO_APPROVE_RISK_ENV,
    DEFAULT_DELETABLE_PREFIXES,
    DEFAULT_TIMEOUT_ENV,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_ENV,
    NETWORK_FINDING_CATEGORIES,
    ComputePolicy,
    ComputeScript,
    ExecutionRecord,
    ScriptState,
)
from kestrel_sovereign.features.compute.script_analyzer import ScriptAnalyzer
from kestrel_sovereign.features.compute.script_signer import ScriptSigner
from kestrel_sovereign.features.compute.script_store import ScriptStore
from kestrel_sovereign.features.compute.trash_manager import (
    TrashManager,
    TrashRetentionError,
)

BENIGN_PYTHON = 'print("hello")\n'
# One medium ``network`` finding: risk 10, far below an auto-approve
# threshold of 101.
NETWORK_PYTHON = "import socket\nsocket.socket()\n"

_COMPUTE_ENV = (
    AUTO_APPROVE_RISK_ENV,
    MAX_TIMEOUT_ENV,
    ALLOW_DOCKER_ENV,
    ALLOW_LOCAL_COMPUTE_ENV,
    DEFAULT_TIMEOUT_ENV,
)


@pytest.fixture(autouse=True)
def _clean_compute_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No ambient compute setting may decide a case, and trash stays in tmp."""
    for name in _COMPUTE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("KESTREL_DEMO_SERVER", raising=False)
    monkeypatch.setenv("KESTREL_TRASH_DIR", str(tmp_path / "trash"))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _RecordingExecutor:
    """Executor double that records what it was asked to run."""

    def __init__(self) -> None:
        self.ran: List[ComputeScript] = []

    @property
    def is_available(self) -> bool:
        return True

    def supports_language(self, language: str) -> bool:
        return True

    async def execute(self, script: ComputeScript) -> ExecutionRecord:
        self.ran.append(script)
        return ExecutionRecord(
            id=f"exec-{len(self.ran)}",
            script_id=script.id,
            completed_at=datetime.now(),
            exit_code=0,
        )


class _ApprovingQueue:
    """Approval queue double: records every request and approves it."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []

    async def request_approval(self, **kwargs: Any) -> Tuple[bool, str]:
        self.requests.append(kwargs)
        return True, "once"


@dataclass
class _Harness:
    feature: ComputeFeature
    store: ScriptStore
    signer: ScriptSigner
    executor: _RecordingExecutor
    queue: _ApprovingQueue

    async def signed_script(self, content: str) -> ComputeScript:
        script = ComputeScript(
            id=f"script-{len(content)}-{abs(hash(content))}",
            name="probe",
            language="python",
            content=content,
            purpose="policy authority probe",
        )
        await self.store.save(script)
        await self.signer.sign_and_update(script)
        script.state = ScriptState.SIGNED
        await self.store.update(script)
        return script


async def _run_harness(tmp_path: Path, policy: ComputePolicy) -> _Harness:
    """A feature wired to ``policy``, a recording executor and an approving
    queue, without the full initialize()."""
    from cryptography.hazmat.primitives.asymmetric import ec

    db_path = str(tmp_path / f"scripts-{id(policy)}.db")
    store = ScriptStore(db_path)
    await store.initialize()
    signer = ScriptSigner("did:ethr:0xpolicy", db_path)
    signer._private_key = ec.generate_private_key(ec.SECP256K1())
    signer._public_key = signer._private_key.public_key()

    async def _keys_loaded() -> bool:
        return True

    signer._load_keys = _keys_loaded

    queue = _ApprovingQueue()
    agent = SimpleNamespace(
        features={"SecurityFeature": SimpleNamespace(approval_queue=queue)}
    )
    feature = ComputeFeature(agent)
    feature.script_store = store
    feature.signer = signer
    feature.analyzer = ScriptAnalyzer()
    feature.policy = policy
    executor = _RecordingExecutor()
    feature.executors = {"uv": executor}
    feature._initialized = True
    feature._init_lock = asyncio.Lock()
    return _Harness(feature, store, signer, executor, queue)


async def _initialized_feature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: ComputePolicy | None = None,
) -> ComputeFeature:
    """Run the real initialize(); Docker reported present so only the policy
    decides whether its executor is built. An executor counts as available
    exactly when one was built, so no host probe runs."""
    monkeypatch.setattr(ComputeFeature, "_docker_available", lambda self: True)

    async def _built(executor: Any) -> bool:
        return executor is not None

    monkeypatch.setattr(ComputeFeature, "_executor_is_available", staticmethod(_built))
    if policy is not None:
        monkeypatch.setattr(
            ComputePolicy,
            "from_env",
            classmethod(lambda cls, environ=None: policy),
        )
    agent = SimpleNamespace(storage_path=str(tmp_path / "agent" / "agent.db"), did=None)
    (tmp_path / "agent").mkdir(exist_ok=True)
    feature = ComputeFeature(agent)
    await feature.initialize()
    return feature


def _age_trash_entry(trash_dir: Path, days: int) -> Path:
    stamp = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d_%H%M%S")
    entry = trash_dir / f"{stamp}_000000_probe"
    entry.mkdir(parents=True)
    (entry / "old.txt").write_text("old")
    return entry


# ---------------------------------------------------------------------------
# Probes: each runs the real consumer under one policy and names the outcome
# ---------------------------------------------------------------------------


async def _probe_approval(tmp_path: Path, policy: ComputePolicy, content: str) -> str:
    harness = await _run_harness(tmp_path, policy)
    script = await harness.signed_script(content)
    result = await harness.feature.run_script(script.id, executor="uv")
    assert result.status is ToolResultStatus.OK, result.error
    assert len(harness.executor.ran) == 1
    return "asked" if harness.queue.requests else "auto-approved"


async def probe_benign_approval(tmp_path, monkeypatch, policy):
    return await _probe_approval(tmp_path, policy, BENIGN_PYTHON)


async def probe_network_approval(tmp_path, monkeypatch, policy):
    return await _probe_approval(tmp_path, policy, NETWORK_PYTHON)


async def probe_ten_minute_timeout(tmp_path, monkeypatch, policy):
    harness = await _run_harness(tmp_path, policy)
    script = await harness.signed_script(BENIGN_PYTHON)
    result = await harness.feature.run_script(script.id, executor="uv", timeout=600)
    if result.status is ToolResultStatus.ERROR:
        assert "exceeds maximum" in result.error
        assert harness.executor.ran == []
        return "refused"
    return "ran"


async def probe_docker_staging_floor(tmp_path, monkeypatch, policy):
    feature = await _initialized_feature(tmp_path, monkeypatch, policy)
    return feature.executors["docker"]._legacy_staging_age_seconds


async def probe_empty_two_day_old_trash(tmp_path, monkeypatch, policy):
    trash_dir = tmp_path / "trash"
    entry = _age_trash_entry(trash_dir, days=2)
    feature = ComputeFeature(SimpleNamespace())
    feature.policy = policy
    feature.trash_manager = TrashManager(trash_dir, policy=policy)
    result = await feature.empty_trash(older_than_days=1, dry_run=False)
    if result.status is ToolResultStatus.ERROR:
        assert entry.is_dir(), "a refused empty deleted nothing"
        return "refused"
    assert not entry.exists()
    return "deleted"


async def probe_docker_executor_built(tmp_path, monkeypatch, policy):
    feature = await _initialized_feature(tmp_path, monkeypatch, policy)
    return feature.executors.get("docker") is not None


async def probe_local_executor_built(tmp_path, monkeypatch, policy):
    feature = await _initialized_feature(tmp_path, monkeypatch, policy)
    return "local" in feature.executors


async def probe_rm_in_compute_temp_dir(tmp_path, monkeypatch, policy):
    """Run a real ``rm`` through the local executor built with ``policy``,
    against a file under a ``/tmp/kestrel_compute_*`` directory."""
    trash_dir = tmp_path / "trash"
    workspace = Path(tempfile.mkdtemp(prefix="kestrel_compute_policy_", dir="/tmp"))
    try:
        target = workspace / "scratch.txt"
        target.write_text("scratch")
        executor = LocalExecutor(require_env_flag=False, policy=policy)
        script = ComputeScript(
            id="prefix-probe",
            name="prefix-probe",
            language="bash",
            content=f'rm -f "{target}"',
            purpose="observe where deletion goes",
        )
        record = await executor.execute(script)
        assert record.exit_code == 0, record.stderr
        assert not target.exists()
        trashed = list(trash_dir.glob("*/scratch.txt"))
        return "trashed" if trashed else "deleted"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


Probe = Callable[[Path, pytest.MonkeyPatch, ComputePolicy], Awaitable[Any]]


@dataclass(frozen=True)
class _Case:
    setting: str
    probe: Probe
    first: Any
    first_outcome: Any
    second: Any
    second_outcome: Any
    base: Tuple[Tuple[str, Any], ...] = ()


CASES = [
    _Case(
        "auto_approve_below_risk",
        probe_benign_approval,
        0, "asked",
        101, "auto-approved",
    ),
    _Case(
        "require_approval_for_network",
        probe_network_approval,
        True, "asked",
        False, "auto-approved",
        base=(("auto_approve_below_risk", 101),),
    ),
    _Case(
        "max_timeout_seconds",
        probe_ten_minute_timeout,
        300, "refused",
        3600, "ran",
        base=(("auto_approve_below_risk", 101),),
    ),
    _Case(
        "max_timeout_seconds",
        probe_docker_staging_floor,
        300, 300 + DockerExecutor.LEGACY_STAGING_GRACE_SECONDS,
        7200, 7200 + DockerExecutor.LEGACY_STAGING_GRACE_SECONDS,
    ),
    _Case(
        "trash_retention_days",
        probe_empty_two_day_old_trash,
        30, "refused",
        1, "deleted",
    ),
    _Case(
        "allow_docker",
        probe_docker_executor_built,
        True, True,
        False, False,
    ),
    _Case(
        "allow_local",
        probe_local_executor_built,
        False, False,
        True, True,
    ),
    _Case(
        "deletable_prefixes",
        probe_rm_in_compute_temp_dir,
        DEFAULT_DELETABLE_PREFIXES, "deleted",
        (), "trashed",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[f"{case.setting}-{case.probe.__name__}" for case in CASES],
)
async def test_each_setting_changes_what_its_consumer_does(
    case: _Case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = []
    for index, value in enumerate((case.first, case.second)):
        case_dir = tmp_path / f"value-{index}"
        case_dir.mkdir()
        monkeypatch.setenv("KESTREL_TRASH_DIR", str(case_dir / "trash"))
        policy = ComputePolicy(**dict(case.base), **{case.setting: value})
        outcomes.append(await case.probe(case_dir, monkeypatch, policy))

    assert outcomes == [case.first_outcome, case.second_outcome]
    assert outcomes[0] != outcomes[1], f"{case.setting} changed nothing"


def test_every_reported_setting_is_enforced_or_deprecated() -> None:
    """The table covers every live setting; anything else status reports is
    a deprecated, fixed value. A new field must join one or the other."""
    live = {field.name for field in dataclasses.fields(ComputePolicy)}
    assert {case.setting for case in CASES} == live
    assert set(ComputePolicy().to_dict()) == live | {"default_timeout_seconds", "trash_enabled"}


# ---------------------------------------------------------------------------
# One instance, threaded to every consumer, and reported as enforced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_threads_one_policy_instance_to_every_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every value differs from its default, so a consumer that built its own
    # default policy instead of taking this one cannot pass by coincidence.
    policy = ComputePolicy(
        auto_approve_below_risk=7,
        require_approval_for_network=False,
        max_timeout_seconds=120,
        trash_retention_days=3,
        allow_local=True,
        deletable_prefixes=("/tmp/kestrel_scratch_",),
    )
    feature = await _initialized_feature(tmp_path, monkeypatch, policy)

    assert feature.policy is policy
    assert set(feature.executors) == {"uv", "docker", "local"}
    consumers = [feature.trash_manager, *feature.executors.values()]
    for consumer in consumers:
        assert consumer.compute_policy is policy, type(consumer).__name__
        assert consumer._policy.deletable_prefixes == list(policy.deletable_prefixes)
    assert not hasattr(feature, "_destructive_policy")

    assert feature.executors["docker"]._legacy_staging_age_seconds == (
        120 + DockerExecutor.LEGACY_STAGING_GRACE_SECONDS
    )
    with pytest.raises(TrashRetentionError):
        feature.trash_manager.empty(older_than_days=2, dry_run=True)

    status = await feature.get_compute_policy()
    capabilities = await feature.get_compute_capabilities()
    assert status.data["policy"] == policy.to_dict()
    assert capabilities.data["policy"] == policy.to_dict()


@pytest.mark.asyncio
async def test_status_reports_the_prefixes_the_executors_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status used to report two prefixes while executors used four."""
    feature = await _initialized_feature(tmp_path, monkeypatch)

    reported = (await feature.get_compute_policy()).data["policy"]["deletable_prefixes"]

    assert reported == list(DEFAULT_DELETABLE_PREFIXES)
    for executor in feature.executors.values():
        assert executor._policy.deletable_prefixes == reported


@pytest.mark.asyncio
async def test_status_reports_the_default_timeout_run_script_uses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """KESTREL_COMPUTE_DEFAULT_TIMEOUT used to show up in status while
    run_script kept its own 300s default."""
    monkeypatch.setenv(DEFAULT_TIMEOUT_ENV, "60")
    with caplog.at_level(logging.WARNING):
        feature = await _initialized_feature(tmp_path, monkeypatch)

    status = await feature.get_compute_policy()

    run_script_default = (
        inspect.signature(ComputeFeature.run_script).parameters["timeout"].default
    )
    assert status.data["policy"]["default_timeout_seconds"] == run_script_default
    assert run_script_default == DEFAULT_TIMEOUT_SECONDS == 300
    assert DEFAULT_TIMEOUT_ENV in caplog.text
    assert "no effect" in caplog.text


@pytest.mark.asyncio
async def test_disallowed_docker_is_unavailable_to_run_script_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KESTREL_COMPUTE_ALLOW_DOCKER=false used to change only a status label."""
    monkeypatch.setenv(ALLOW_DOCKER_ENV, "false")
    feature = await _initialized_feature(tmp_path, monkeypatch)

    capabilities = await feature.get_compute_capabilities()
    status = await feature.get_compute_policy()

    assert feature.executors["docker"] is None
    assert capabilities.data["executors"] == {"uv": True, "docker": False}
    assert status.data["policy"]["allow_docker"] is False
    assert "disabled" in status.data["explanation"]["executors"]["docker"]


@pytest.mark.parametrize(
    ("value", "enabled"),
    [("true", True), ("TRUE", False), ("1", False), ("false", False)],
)
def test_local_permission_matches_the_local_executor_guard(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    enabled: bool,
) -> None:
    """Status used to call ``TRUE`` enabled while the executor refused it."""
    monkeypatch.setenv(ALLOW_LOCAL_COMPUTE_ENV, value)

    assert ComputePolicy.from_env().allow_local is enabled
    assert LocalExecutor().is_available is enabled


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, ComputePolicy()),
        ({AUTO_APPROVE_RISK_ENV: "15"}, ComputePolicy(auto_approve_below_risk=15)),
        ({MAX_TIMEOUT_ENV: "60"}, ComputePolicy(max_timeout_seconds=60)),
        ({ALLOW_DOCKER_ENV: "false"}, ComputePolicy(allow_docker=False)),
        ({ALLOW_DOCKER_ENV: "TRUE"}, ComputePolicy(allow_docker=True)),
        ({ALLOW_LOCAL_COMPUTE_ENV: "true"}, ComputePolicy(allow_local=True)),
    ],
)
def test_from_env_reads_each_variable_into_its_setting(
    environ: Dict[str, str],
    expected: ComputePolicy,
) -> None:
    assert ComputePolicy.from_env(environ) == expected


# ---------------------------------------------------------------------------
# Deprecated settings: warn, change nothing, report the effective value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setting", "value", "effective"),
    [
        ("default_timeout_seconds", 60, DEFAULT_TIMEOUT_SECONDS),
        ("trash_enabled", False, True),
        ("trash_enabled", True, True),
    ],
)
def test_a_deprecated_setting_warns_and_reports_the_effective_value(
    caplog: pytest.LogCaptureFixture,
    setting: str,
    value: Any,
    effective: Any,
) -> None:
    with caplog.at_level(logging.WARNING):
        policy = ComputePolicy(**{setting: value})

    assert f"ComputePolicy.{setting} is deprecated and has no effect" in caplog.text
    assert getattr(policy, setting) == effective
    assert policy.to_dict()[setting] == effective
    assert policy == ComputePolicy()


def test_unset_deprecated_settings_do_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        policy = ComputePolicy()
        dataclasses.replace(policy, max_timeout_seconds=60)
        ComputePolicy.from_env({})

    assert caplog.records == []


@pytest.mark.asyncio
async def test_a_deprecated_default_timeout_does_not_change_the_run_timeout(
    tmp_path: Path,
) -> None:
    harness = await _run_harness(
        tmp_path,
        ComputePolicy(auto_approve_below_risk=101, default_timeout_seconds=60),
    )
    script = await harness.signed_script(BENIGN_PYTHON)

    result = await harness.feature.run_script(script.id, executor="uv")

    assert result.status is ToolResultStatus.OK
    assert harness.executor.ran[0].timeout_seconds == DEFAULT_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_disabling_trash_does_not_make_rm_delete(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``trash_enabled=False`` is refused as a no-op: honoring it would turn
    every rm into a real deletion."""
    trash_dir = tmp_path / "trash"
    target = tmp_path / "precious.txt"
    target.write_text("irreplaceable")
    with caplog.at_level(logging.WARNING):
        policy = ComputePolicy(trash_enabled=False)
    executor = LocalExecutor(require_env_flag=False, policy=policy)
    script = ComputeScript(
        id="trash-disabled",
        name="trash-disabled",
        language="bash",
        content=f'rm "{target}"',
        purpose="prove trash cannot be turned off",
    )

    record = await executor.execute(script)

    assert record.exit_code == 0, record.stderr
    assert next(trash_dir.glob("*/precious.txt")).read_text() == "irreplaceable"
    assert "trash_enabled is deprecated" in caplog.text


# ---------------------------------------------------------------------------
# Fail-closed defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_default_policy_asks_before_running_even_a_benign_script(
    tmp_path: Path,
) -> None:
    harness = await _run_harness(tmp_path, ComputePolicy())
    script = await harness.signed_script(BENIGN_PYTHON)

    await harness.feature.run_script(script.id, executor="uv")

    assert len(harness.queue.requests) == 1


def test_default_policy_values_are_the_previous_effective_ones() -> None:
    policy = ComputePolicy()

    assert policy.auto_approve_below_risk == 0
    assert policy.require_approval_for_network is True
    assert policy.max_timeout_seconds == 3600
    assert policy.trash_retention_days == 30
    assert policy.allow_local is False
    # The four prefixes executors honored before #2483, whatever status said.
    assert list(policy.deletable_prefixes) == [
        "/tmp/kestrel_compute_",
        "/tmp/kestrel_scratch_",
        "/private/tmp/kestrel_compute_",
        "/private/tmp/kestrel_scratch_",
    ]
    assert EXPORTED_DEFAULT_DELETABLE_PREFIXES == policy.deletable_prefixes
    assert ComputePolicy.from_env({}) == policy


@pytest.mark.parametrize(
    "prefixes",
    [
        ["/home/"],
        ["/"],
        ["/tmp/"],
        [*DEFAULT_DELETABLE_PREFIXES, "/data/scratch_"],
    ],
)
def test_deletable_prefixes_cannot_widen_real_deletion(prefixes: List[str]) -> None:
    with pytest.raises(ValueError, match="may only narrow"):
        ComputePolicy(deletable_prefixes=prefixes)


def test_deletable_prefixes_rejects_a_bare_string() -> None:
    with pytest.raises(TypeError):
        ComputePolicy(deletable_prefixes="/tmp/kestrel_compute_")


def test_negative_retention_is_refused() -> None:
    with pytest.raises(ValueError, match="trash_retention_days"):
        ComputePolicy(trash_retention_days=-1)


@pytest.mark.parametrize("value", ["0", "-5"])
def test_a_maximum_timeout_below_one_second_is_refused(value: str) -> None:
    """A zero maximum used to reach the Docker staging sweep as "use the
    default" by accident of ``0 or 3600``; read as policy it would shrink the
    floor to the grace period and sweep live staging directories early."""
    with pytest.raises(ValueError, match="max_timeout_seconds"):
        ComputePolicy.from_env({MAX_TIMEOUT_ENV: value})


@pytest.mark.asyncio
async def test_status_says_when_a_demo_server_skips_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The approval settings do not apply on a demo server, and status says
    so instead of promising that network scripts always ask."""
    harness = await _run_harness(tmp_path, ComputePolicy())
    quiet = await harness.feature.get_compute_policy()
    monkeypatch.setenv("KESTREL_DEMO_SERVER", "1")
    script = await harness.signed_script(NETWORK_PYTHON)

    loud = await harness.feature.get_compute_policy()
    result = await harness.feature.run_script(script.id, executor="uv")

    assert "demo_server" not in quiet.data["explanation"]
    assert "never asks" in loud.data["explanation"]["demo_server"]
    assert result.status is ToolResultStatus.OK
    assert harness.queue.requests == []


def test_the_policy_cannot_change_after_it_is_threaded() -> None:
    policy = ComputePolicy()

    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.allow_docker = False  # type: ignore[misc]


def test_retention_bounds_every_empty_including_a_dry_run(tmp_path: Path) -> None:
    trash_dir = tmp_path / "trash"
    young = _age_trash_entry(trash_dir, days=5)
    manager = TrashManager(trash_dir, policy=ComputePolicy(trash_retention_days=10))

    with pytest.raises(TrashRetentionError):
        manager.empty(older_than_days=0, dry_run=True)
    with pytest.raises(TrashRetentionError):
        manager.empty(older_than_days=-1, dry_run=False)

    assert manager.empty(dry_run=False) == 0
    assert young.is_dir()


@pytest.mark.asyncio
async def test_empty_trash_reports_the_retention_refusal(tmp_path: Path) -> None:
    feature = ComputeFeature(SimpleNamespace())
    feature.trash_manager = TrashManager(tmp_path / "trash")

    result = await feature.empty_trash(older_than_days=7, dry_run=True)

    assert result.status is ToolResultStatus.ERROR
    assert result.data == {
        "dry_run": True,
        "older_than_days": 7,
        "retention_days": 30,
        "applied": False,
    }


def test_the_tool_defaults_are_the_policy_defaults() -> None:
    """The tool signatures keep their literal defaults, because the published
    schemas and the feature inventory are read from them; this holds those
    literals to the policy's numbers."""
    run_script = inspect.signature(ComputeFeature.run_script).parameters
    empty_trash = inspect.signature(ComputeFeature.empty_trash).parameters

    assert run_script["timeout"].default == ComputePolicy().default_timeout_seconds
    assert empty_trash["older_than_days"].default == ComputePolicy().trash_retention_days


def test_network_categories_are_ones_the_analyzer_emits() -> None:
    """Renaming an analyzer category must not silently disable the gate."""
    analyzer = ScriptAnalyzer()
    emitted = set()
    for language, content in [
        ("python", NETWORK_PYTHON),
        ("python", "import requests\nrequests.get('https://example.com')\n"),
        ("bash", "wget https://example.com/file\n"),
        ("bash", "nc -v 10.0.0.1 4444\n"),
    ]:
        script = ComputeScript(
            id="category-probe",
            name="category-probe",
            language=language,
            content=content,
            purpose="category probe",
        )
        categories = {finding.category for finding in analyzer.analyze(script).findings}
        assert categories & NETWORK_FINDING_CATEGORIES, (language, content, categories)
        emitted |= categories
    assert NETWORK_FINDING_CATEGORIES <= emitted


def test_executors_built_without_a_policy_use_the_default_one() -> None:
    for executor in (UvExecutor(), DockerExecutor(), LocalExecutor()):
        assert executor.compute_policy == ComputePolicy()
        assert executor._policy.deletable_prefixes == list(DEFAULT_DELETABLE_PREFIXES)
    assert DockerExecutor()._legacy_staging_age_seconds == (
        ComputePolicy().max_timeout_seconds + DockerExecutor.LEGACY_STAGING_GRACE_SECONDS
    )


def test_feature_module_reads_no_compute_environment_directly() -> None:
    """The feature builds its policy through ComputePolicy.from_env only."""
    source = inspect.getsource(feature_module)
    for name in _COMPUTE_ENV:
        assert f'"{name}"' not in source, name
