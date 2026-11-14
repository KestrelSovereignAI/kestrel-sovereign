"""Original Host ADMIT checks before any two-domain worker resource exists.

The resource-construction sentinel is a test double. These source tests use
the real feature authority, Runtime, SDK models and encrypted SQLite; they
do not demonstrate container, TLS, Unix-socket or browser conformance.
"""

import asyncio
import copy
from contextlib import asynccontextmanager
from dataclasses import replace
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceUnsupported

from kestrel_sovereign.auth import CallerContext, caller_context_scope
from kestrel_sovereign.features.visual_computer import feature as feature_module
from kestrel_sovereign.features.visual_computer import VisualComputerFeature
from kestrel_sovereign.visual_computer import docker
from kestrel_sovereign.visual_computer.local_worker import (
    LocalTwoDomainDriver,
    LocalTwoDomainProvider,
)
from kestrel_sovereign.visual_computer.runtime import VisualAction
from kestrel_sovereign.visual_computer.runtime import VisualComputerRuntime
from kestrel_sovereign.visual_computer.repository import CustodyRecord
from kestrel_sovereign.visual_computer.worker_profile import DRIVER, PROFILE, PROVIDER
from tests.integration.test_artifact_service import PolicyHolder
from tests.shared.installed_worker_requirement import worker_module


class ResourceBoundaryReached(Exception):
    """The admitted source program reached the forbidden native boundary."""


def original_admit_check():
    """Observe the actual awaiting Runtime frame, without wrapping its check."""
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if frame.f_code is VisualComputerRuntime.admit.__code__:
                return frame.f_locals["check"]
            frame = frame.f_back
    finally:
        del frame
    raise AssertionError("resource boundary did not originate in Runtime.admit")


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_provider", [False, True])
async def test_nonlocal_shared_driver_key_does_not_select_local_setup(
    harness, tmp_path, configured_provider,
):
    """Real adapter construction only: no daemon or browser operation."""
    socket = str(tmp_path / "absent-daemon.sock")
    driver = docker.DockerComputerDriver(docker_socket=socket)
    providers = {}
    if configured_provider:
        providers[docker.PROVIDER] = docker.DockerEnvironmentProvider(
            docker_socket=socket, image="browser@sha256:" + "a" * 64,
            credential_key=b"k" * 32,
        )
    runtime = VisualComputerRuntime(
        harness.repository, harness.runtime.authority,
        providers=providers, drivers={DRIVER: driver},
    )
    assert runtime.providers == providers
    assert runtime.drivers[DRIVER] is driver
    assert driver.egress_guard is None


@pytest.mark.asyncio
@pytest.mark.parametrize("configuration", [
    "provider-only", "driver-only", "nonlocal-driver", "complete",
])
async def test_selected_local_adapters_require_and_bind_complete_original_pair(
    harness, tmp_path, configuration,
):
    socket = str(tmp_path / "absent-daemon.sock")
    provider = None if configuration == "driver-only" else LocalTwoDomainProvider(
        docker_socket=socket, browser_image="browser@sha256:" + "a" * 64,
        broker_image="broker@sha256:" + "b" * 64, credential_key=b"k" * 32,
    )
    driver = None if configuration == "provider-only" else (
        docker.DockerComputerDriver(docker_socket=socket)
        if configuration == "nonlocal-driver" else LocalTwoDomainDriver(docker_socket=socket)
    )
    providers = {} if provider is None else {PROVIDER: provider}
    drivers = {} if driver is None else {DRIVER: driver}
    if configuration != "complete":
        with pytest.raises(ResourceUnsupported, match="complete_explicit_local_two_domain_profile_required"):
            VisualComputerRuntime(
                harness.repository, harness.runtime.authority,
                providers=providers, drivers=drivers,
            )
        if provider is not None:
            assert provider._setup_host is None
        if type(driver) is LocalTwoDomainDriver:
            assert driver._setup_host is None and driver.egress_guard is None
        return
    runtime = VisualComputerRuntime(
        harness.repository, harness.runtime.authority, providers=providers, drivers=drivers,
    )
    host = provider._setup_host
    assert host is not None and driver._setup_host is host
    assert host._runtime is runtime and host._provider is provider and host._driver is driver
    assert driver.egress_guard == runtime.egress.operation


@pytest_asyncio.fixture
async def local_setup(harness, monkeypatch, tmp_path):
    worker_module("kestrel_cloud_runpod.remote_worker_tls",
                  required_callables=("PinnedWorkerTarget", "PinnedWorkerTLS"))
    key = "local-two-domain-source-test-only-key"
    monkeypatch.setenv("KESTREL_API_KEY", key)
    socket = str(tmp_path / "absent-local-daemon.sock")
    browser_image = "source-browser@sha256:" + "a" * 64
    provider = LocalTwoDomainProvider(
        docker_socket=socket,
        browser_image=browser_image,
        broker_image="source-broker@sha256:" + "b" * 64,
        credential_key=b"k" * 32,
    )
    driver = LocalTwoDomainDriver(docker_socket=socket)
    config = {
        "enabled": True,
        "granted_actions": [action.value for action in VisualAction],
        "providers": {PROVIDER: {}},
        "drivers": {DRIVER: {}},
    }
    monkeypatch.setattr(
        feature_module, "load_section", lambda _section: {"visual_computer": config}
    )
    # Factory discovery is controlled here; the registered adapters and their
    # private setup host remain the actual canonical product objects.
    monkeypatch.setattr(
        feature_module, "construct_adapters",
        lambda **_settings: ({PROVIDER: provider}, {DRIVER: driver}),
    )
    agent = PolicyHolder()
    agent.did = "did:test:local-two-domain"
    agent._raw_storage = SimpleNamespace(db=harness.db)
    agent.features = {}
    agent._track_background_task = (
        lambda coroutine, *, name: asyncio.create_task(coroutine, name=name)
    )
    feature = VisualComputerFeature(agent)
    await feature.initialize()
    agent.features[feature.name] = feature
    await feature.on_enable()
    caller = CallerContext.sovereign(credential=key)

    @asynccontextmanager
    async def bound():
        with caller_context_scope(caller):
            async with feature._bind_context(caller) as context:
                yield context

    def request(context, **changes):
        return harness.request(
            context, provider_id=PROVIDER, profile_id=PROFILE,
            image=browser_image, **changes,
        )

    try:
        yield SimpleNamespace(
            feature=feature, runtime=feature.runtime, provider=provider,
            driver=driver, host=provider._setup_host, agent=agent,
            bound=bound, request=request, key=key,
        )
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_deserialized_context_cannot_reach_local_setup(local_setup, monkeypatch):
    setup = local_setup
    reached = []

    async def forbidden(original):
        reached.append(original)
        raise ResourceBoundaryReached

    monkeypatch.setattr(setup.host, "_create_original_resources", forbidden)
    async with setup.bound() as context:
        with pytest.raises(OperatorAuthorizationError):
            await setup.runtime.admit(replace(context), setup.request(context))
    assert reached == []
    assert setup.runtime.egress._calls == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["outside", "status", "forwarded", "noop"])
async def test_current_host_context_and_quoted_facts_do_not_issue_setup(
    local_setup, source,
):
    setup = local_setup
    runtime = setup.runtime
    async with setup.bound() as context:
        # Construct real same-owner quoted custody through the canonical SQL
        # API and genuine Host ADMIT check. The row is factual input, not an
        # invitation to issue setup outside Runtime.admit's original frame.
        async with runtime.operation(context, VisualAction.ADMIT) as check:
            request = runtime._request(context, setup.request(context))
            record = await runtime.repository.admit(
                runtime.scope(context), CustodyRecord(
                    request=request, session_id=uuid4().hex,
                    last_activity_at=runtime._now(),
                ), check,
            )
            quote = await runtime._call(context, check, setup.provider.quote, request)
            runtime._quote(record, quote)
            record.quote, record.phase = quote, "quoted"
            await runtime.repository.save(runtime.scope(context), record, check)
        stored = await runtime.repository.get(runtime.scope(context), request_id=request.request_id)
        assert stored.phase == "quoted" and stored.session is None
        if source == "outside":
            with pytest.raises(OperatorAuthorizationError):
                setup.host._issue_local_setup((context, record, check))
            with pytest.raises(OperatorAuthorizationError):
                await setup.provider.acquire(request, quote)
        else:
            action = VisualAction.STATUS if source == "status" else VisualAction.ADMIT
            async with runtime.operation(context, action) as current:
                def forwarded():
                    return current()

                supplied = current if source == "status" else (
                    forwarded if source == "forwarded" else lambda: None
                )
                with pytest.raises(OperatorAuthorizationError):
                    setup.host._issue_local_setup(supplied)
        retained = await runtime.repository.get(runtime.scope(context), request_id=request.request_id)
        assert retained.phase == "quoted" and retained.lease is None
        assert retained.connection is None and retained.session is None
    assert runtime.egress._calls == {}


@pytest.mark.asyncio
async def test_only_original_one_shot_runtime_admission_can_issue_setup(
    local_setup, monkeypatch,
):
    setup = local_setup
    issue = setup.host._issue_local_setup
    admissions = []
    sentinel = ResourceBoundaryReached()

    def observe_original(admission):
        admissions.append(admission)
        assert admission._check is original_admit_check()
        assert not admission._consumed
        with pytest.raises(OperatorAuthorizationError):
            issue(copy.copy(admission))
        with pytest.raises(OperatorAuthorizationError):
            type(setup.host)._issue_local_setup(copy.copy(setup.host), admission)
        assert not admission._consumed
        original = issue(admission)
        assert admission._consumed
        assert original._admission is admission
        with pytest.raises(OperatorAuthorizationError):
            issue(admission)
        return original

    async def boundary(original):
        assert original._admission is admissions[0]
        raise sentinel

    monkeypatch.setattr(setup.host, "_issue_local_setup", observe_original)
    monkeypatch.setattr(setup.host, "_create_original_resources", boundary)
    async with setup.bound() as context:
        with pytest.raises(ResourceBoundaryReached) as caught:
            await setup.runtime.admit(context, setup.request(context))
        assert caught.value is sentinel
        assert len(admissions) == 1 and not admissions[0]._active
        # A new genuine operation does not renew the retired representation.
        async with setup.runtime.operation(context, VisualAction.ADMIT):
            with pytest.raises(OperatorAuthorizationError):
                issue(admissions[0])
    assert setup.runtime.egress._calls == {}


@pytest.mark.asyncio
async def test_local_setup_retains_same_original_admit_check_and_owner(
    local_setup, monkeypatch,
):
    setup = local_setup
    captured = []
    sentinel = ResourceBoundaryReached()

    async def boundary(original):
        captured.append(original)
        assert original._check is original_admit_check()
        assert original._context is context
        assert original._owner_task is asyncio.current_task()
        assert original._loop is asyncio.get_running_loop()
        assert original._binding is original
        assert setup.runtime.egress._calls[original._key] is original
        assert original._key[:4] == (
            context.tenant_id, context.principal_id, context.acting_agent_id,
            original._record.session_id,
        )
        assert original._key[-1] == "local-worker-setup"
        assert original._record.session is None
        original._require_origin()
        original._check_original()
        with pytest.raises(OperatorAuthorizationError):
            copy.copy(original)._require_origin()

        async def foreign_task():
            with pytest.raises(OperatorAuthorizationError):
                original._require_origin()

        child = asyncio.create_task(foreign_task())
        await child  # Retain and join this actual, harmless Task.
        assert child.done() and not child.cancelled()
        raise sentinel

    monkeypatch.setattr(setup.host, "_create_original_resources", boundary)
    async with setup.bound() as context:
        with pytest.raises(ResourceBoundaryReached) as caught:
            await setup.runtime.admit(context, setup.request(context))
        assert caught.value is sentinel
    assert len(captured) == 1
    original = captured[0]
    assert original._finished.done()
    assert not original._active
    assert setup.runtime.egress._calls == {}
    with pytest.raises(OperatorAuthorizationError):
        original._check_original()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [
    "binding", "task", "loop", "context", "record", "map", "provider", "driver",
])
async def test_local_setup_identity_changes_refuse_before_resource_work(
    local_setup, monkeypatch, changed,
):
    setup = local_setup
    sentinel = ResourceBoundaryReached()
    verified = []

    async def boundary(original):
        attribute = {"binding": "_binding", "task": "_owner_task", "loop": "_loop",
                     "context": "_context", "record": "_record"}.get(changed)
        if changed == "map":
            saved = setup.runtime.egress._calls[original._key]
            setup.runtime.egress._calls[original._key] = copy.copy(original)
        elif changed in {"provider", "driver"}:
            mapping, key = ((setup.runtime.providers, PROVIDER) if changed == "provider"
                            else (setup.runtime.drivers, DRIVER))
            saved = mapping[key]
            mapping[key] = copy.copy(saved)
        else:
            saved = getattr(original, attribute)
            changed_value = (replace(saved) if changed == "context" else
                             saved.model_copy() if changed == "record" else object())
            setattr(original, attribute, changed_value)
        try:
            with pytest.raises(OperatorAuthorizationError):
                original._require_origin()
            verified.append(changed)
        finally:
            if changed == "map":
                setup.runtime.egress._calls[original._key] = saved
            elif changed in {"provider", "driver"}:
                mapping[key] = saved
            else:
                setattr(original, attribute, saved)
        original._require_origin()
        raise sentinel

    monkeypatch.setattr(setup.host, "_create_original_resources", boundary)
    async with setup.bound() as context:
        with pytest.raises(ResourceBoundaryReached) as caught:
            await setup.runtime.admit(context, setup.request(context))
        assert caught.value is sentinel
    assert verified == [changed]
    assert setup.runtime.egress._calls == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawal", ["grant", "generation", "privacy", "key", "expiry"])
async def test_local_setup_withdrawal_after_issuance_precedes_first_resource(
    local_setup, monkeypatch, withdrawal,
):
    from kestrel_sovereign.privacy import PrivacyMode

    setup = local_setup
    issued, resource_entries = [], []
    issue = setup.host._issue_local_setup

    def withdraw_after_original_issue(admission):
        original = issue(admission)
        issued.append(original)
        if withdrawal == "grant":
            setup.feature._cfg["granted_actions"].remove(VisualAction.ADMIT.value)
        elif withdrawal == "generation":
            setup.feature._generation += 1
        elif withdrawal == "privacy":
            setup.agent.set_mode(PrivacyMode.ISOLATED)
        elif withdrawal == "key":
            monkeypatch.setenv("KESTREL_API_KEY", "withdrawn-source-test-key")
        else:
            monkeypatch.setattr(
                setup.runtime.authority, "clock", lambda: admission._context.expires_at,
            )
        return original

    async def forbidden(original):
        resource_entries.append(original)
        raise ResourceBoundaryReached

    # The wrapper forwards the real issuer and its SAME arguments/result. It
    # changes genuine current host state only after the original owner exists.
    monkeypatch.setattr(setup.host, "_issue_local_setup", withdraw_after_original_issue)
    monkeypatch.setattr(setup.host, "_create_original_resources", forbidden)
    async with setup.bound() as context:
        with pytest.raises(OperatorAuthorizationError):
            await setup.runtime.admit(context, setup.request(context))
    assert len(issued) == 1
    assert resource_entries == []
    assert not issued[0]._active
    assert setup.runtime.egress._calls == {}
    with pytest.raises(OperatorAuthorizationError):
        issued[0]._check_original()
