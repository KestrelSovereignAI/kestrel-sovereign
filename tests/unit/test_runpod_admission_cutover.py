"""Fresh-work refusal is distinct from retained Runpod lifecycle access.

No provider/model/network is simulated as successful here. Sentinels prove
that denied fresh requests never reach the manager, lifecycle or HTTP door.
Existing shared lifecycle tests retain the historical release/race coverage.
"""

import asyncio

import pytest

from kestrel_sovereign import cli_runpod
from kestrel_sovereign.features.training.adapters.runpod_adapter import RunPodTrainingAdapter
from kestrel_sovereign.features.training.factory import TrainingProviderFactory
from kestrel_sovereign.features.training.protocol import (
    DownloadError,
    GenerationError,
    ProviderNotAvailableError,
    TrainingProviderError,
    TrainingStatusError,
    TrainingSubmissionError,
)
from kestrel_sovereign.features.training.types import GenerationConfig


def _unexpected(*_args, **_kwargs):
    raise AssertionError("retired fresh admission reached an effectful dependency")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["training", "generation", "generation_with_session"])
async def test_retired_adapter_refuses_before_any_manager_or_task(monkeypatch, operation):
    adapter = RunPodTrainingAdapter()
    monkeypatch.setattr(adapter, "_get_manager", _unexpected)
    before = asyncio.all_tasks()
    if operation == "training":
        with pytest.raises(TrainingSubmissionError, match="catalog"):
            await adapter.start_training("owned-companion", b"private avatar")
    else:
        with pytest.raises(GenerationError, match="catalog"):
            await adapter.generate_image(
                GenerationConfig(prompt="private original intent", lora_path="untrusted"),
                session=object() if operation.endswith("with_session") else None,
            )
    assert adapter._active_jobs == {}
    assert asyncio.all_tasks() == before


def test_runpod_capabilities_are_recovery_only():
    capabilities = TrainingProviderFactory.get_capabilities("runpod")
    assert capabilities.training is False
    assert capabilities.generation is False
    assert capabilities.supports_lora_download is True


@pytest.mark.parametrize("selection", ["get_default_provider", "get_generation_provider", "get_uncensored_provider"])
def test_default_selection_does_not_try_retired_runpod(monkeypatch, selection):
    monkeypatch.delenv("GENERATION_PROVIDER", raising=False)
    monkeypatch.setattr(TrainingProviderFactory, "_effective_priority", lambda: ["runpod"])
    monkeypatch.setattr(TrainingProviderFactory, "get_provider", _unexpected)
    assert getattr(TrainingProviderFactory, selection)() is None


@pytest.mark.parametrize("selection", ["get_default_provider", "get_generation_provider", "get_uncensored_provider"])
def test_forced_retired_runpod_cannot_fall_through_to_another_model(monkeypatch, selection):
    monkeypatch.setenv("GENERATION_PROVIDER", "runpod")
    monkeypatch.setattr(TrainingProviderFactory, "_create_provider", _unexpected)
    with pytest.raises(ProviderNotAvailableError, match="runpod"):
        getattr(TrainingProviderFactory, selection)()


def test_retired_deploy_cli_fails_before_configuration_or_manager(monkeypatch, capsys):
    from types import SimpleNamespace

    monkeypatch.setattr(cli_runpod, "_check_api_key", _unexpected)
    monkeypatch.setattr(cli_runpod, "_load_manager", _unexpected)
    assert cli_runpod.cmd_runpod(SimpleNamespace(
        runpod_command="deploy", target="lora-trainer", profile="training", test=False,
    )) == 1
    assert "catalog" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("purpose", ["training", "generation"])
def test_cached_historical_provider_does_not_authorize_new_work(monkeypatch, purpose):
    retained = RunPodTrainingAdapter()
    monkeypatch.setattr(TrainingProviderFactory, "_instances", {"runpod": retained})
    assert TrainingProviderFactory.get_provider("runpod") is retained
    with pytest.raises(ProviderNotAvailableError, match="catalog"):
        TrainingProviderFactory.get_provider("runpod", purpose=purpose)
    assert TrainingProviderFactory._instances["runpod"] is retained


def test_available_listing_separates_fresh_and_historical_providers(monkeypatch):
    monkeypatch.setattr(TrainingProviderFactory, "_effective_priority", lambda: ["runpod"])
    monkeypatch.setattr(TrainingProviderFactory, "_check_provider_available", lambda _name: True)
    assert TrainingProviderFactory.list_available_providers() == []
    assert TrainingProviderFactory.list_available_providers(include_recovery=True) == ["runpod"]


@pytest.mark.parametrize("purpose", ["", "recovery", "unsupported", True])
def test_invalid_fresh_purpose_fails_before_provider_creation(monkeypatch, purpose):
    monkeypatch.setattr(TrainingProviderFactory, "_create_provider", _unexpected)
    with pytest.raises(ValueError, match="purpose"):
        TrainingProviderFactory.get_provider("runpod", purpose=purpose)


def test_unknown_fresh_provider_fails_before_creation(monkeypatch):
    monkeypatch.setattr(TrainingProviderFactory, "_entry_points_loaded", True)
    monkeypatch.setattr(TrainingProviderFactory, "_ep_provider_capabilities", {})
    monkeypatch.setattr(TrainingProviderFactory, "_create_provider", _unexpected)
    with pytest.raises(ProviderNotAvailableError, match="unknown-provider"):
        TrainingProviderFactory.get_provider("unknown-provider", purpose="training")


@pytest.mark.parametrize("selection", ["get_default_provider", "get_generation_provider"])
def test_forced_unavailable_provider_never_selects_another_backend(monkeypatch, selection):
    class Unavailable:
        def is_available(self):
            return False

    created = []

    def create(name):
        created.append(name)
        return Unavailable()

    monkeypatch.setenv("GENERATION_PROVIDER", "vertex_ai")
    monkeypatch.setattr(TrainingProviderFactory, "_instances", {})
    monkeypatch.setattr(TrainingProviderFactory, "_entry_points_loaded", True)
    monkeypatch.setattr(TrainingProviderFactory, "_create_provider", create)
    with pytest.raises(ProviderNotAvailableError, match="vertex_ai.*unavailable"):
        getattr(TrainingProviderFactory, selection)()
    assert created == ["vertex_ai"]


def test_forced_filtered_generation_cannot_bypass_requested_capability(monkeypatch):
    monkeypatch.setenv("GENERATION_PROVIDER", "replicate")
    monkeypatch.setattr(TrainingProviderFactory, "_create_provider", _unexpected)
    with pytest.raises(ProviderNotAvailableError, match="uncensored generation"):
        TrainingProviderFactory.get_generation_provider(uncensored=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["denied", "unknown"])
@pytest.mark.parametrize("operation", [
    "training", "generation", "factory_training", "factory_generation",
    "factory_uncensored", "factory_explicit",
])
async def test_original_native_denial_precedes_fresh_selection_or_retirement(
    monkeypatch, outcome, operation
):
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError, ProcessRuntimeExecutionFence,
        bind_execution_custody, execution_commit_outcome, execution_denial_scopes,
    )

    monkeypatch.delenv("GENERATION_PROVIDER", raising=False)
    monkeypatch.setattr(TrainingProviderFactory, "_effective_priority", lambda: ["runpod"])
    adapter = RunPodTrainingAdapter()
    monkeypatch.setattr(adapter, "_get_manager", _unexpected)
    # Actual original standalone runtime custody, deliberately denied. This
    # grants no positive tenant/catalog/physical-resource permission.
    with bind_execution_custody(ProcessRuntimeExecutionFence("owned-cutover-control")) as scope:
        if outcome == "unknown":
            scope.preserve_commit_uncertainty("unknown")
        else:
            scope.revoke("original cutover invocation retired")
        with pytest.raises(ExecutionAuthorityError) as observed:
            if operation == "training":
                await adapter.start_training("owned-companion", b"private avatar")
            elif operation == "generation":
                await adapter.generate_image(GenerationConfig(prompt="private", lora_path="untrusted"))
            elif operation == "factory_explicit":
                TrainingProviderFactory.get_provider("runpod", purpose="training")
            else:
                selector = {
                    "factory_training": "get_default_provider",
                    "factory_generation": "get_generation_provider",
                    "factory_uncensored": "get_uncensored_provider",
                }[operation]
                getattr(TrainingProviderFactory, selector)()
        if outcome == "unknown":
            assert execution_commit_outcome(observed.value) == "unknown"
        else:
            assert scope in execution_denial_scopes(observed.value)


async def _original_control(outcome):
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError, ProcessRuntimeExecutionFence,
        bind_execution_custody, is_execution_control_error,
        retain_native_execution_stream_close,
    )

    if outcome == "retirement":
        import httpx

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b""

            async def aclose(self):
                raise OSError("original native response close failed")

        response = httpx.Response(200, stream=Stream())
        retain_native_execution_stream_close(response)
        with pytest.raises(BaseExceptionGroup) as observed:
            await response.aclose()
        original = observed.value
    else:
        with bind_execution_custody(ProcessRuntimeExecutionFence("owned-provider-control")) as scope:
            if outcome == "unknown":
                scope.preserve_commit_uncertainty("unknown")
            else:
                scope.revoke("original provider scope retired")
            with pytest.raises(ExecutionAuthorityError) as observed:
                scope.require_work()
        original = observed.value
        if outcome == "group":
            original = ExceptionGroup("original outcomes", [ValueError("decode"), original])
    assert is_execution_control_error(original)
    failure = RuntimeError("declared provider boundary failed")
    failure.__cause__ = original
    return failure


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["denied", "unknown", "group", "retirement"])
@pytest.mark.parametrize("boundary", ["discover", "construct", "available", "builtin_construct", "builtin_import"])
async def test_factory_does_not_normalize_original_control_into_unavailability(
    monkeypatch, outcome, boundary
):
    failure = await _original_control(outcome)
    if boundary == "builtin_import":
        wrapped = ImportError("declared builtin import failed")
        wrapped.__cause__ = failure
        failure = wrapped

    class Provider:
        def __init__(self):
            if boundary in {"construct", "builtin_construct", "builtin_import"}:
                raise failure

        def is_available(self):
            raise failure

    monkeypatch.setattr(TrainingProviderFactory, "_instances", {})
    monkeypatch.setattr(TrainingProviderFactory, "_entry_points_loaded", boundary != "discover")
    monkeypatch.setattr(TrainingProviderFactory, "_ep_provider_classes", {"owned": Provider})
    if boundary == "discover":
        def discover():
            raise failure
        monkeypatch.setattr(TrainingProviderFactory, "_discover_entry_point_providers", discover)
    if boundary in {"builtin_construct", "builtin_import"}:
        from kestrel_sovereign.features.training.adapters import runpod_adapter
        monkeypatch.setattr(runpod_adapter, "RunPodTrainingAdapter", Provider)
    with pytest.raises(type(failure)) as observed:
        if boundary == "discover":
            TrainingProviderFactory._ensure_entry_points_loaded()
        elif boundary == "construct":
            TrainingProviderFactory._create_provider("owned")
        elif boundary in {"builtin_construct", "builtin_import"}:
            TrainingProviderFactory._create_provider("runpod")
        else:
            TrainingProviderFactory.get_provider("owned")
    assert observed.value is failure
    assert not TrainingProviderFactory._instances


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["denied", "unknown", "group", "retirement"])
async def test_all_provider_closes_join_without_discarding_later_original_control(
    monkeypatch, outcome
):
    from kestrel_sovereign.execution_custody import _explicit_control_chain, is_execution_control_error

    failure = await _original_control(outcome)
    ordinary = ValueError("ordinary earlier close failure")
    joined = []

    class Provider:
        def __init__(self, name, error=None):
            self.name, self.error = name, error

        async def close(self):
            await asyncio.sleep(0)
            joined.append(self.name)
            if self.error is not None:
                raise self.error

    retained = {
        "ordinary": Provider("ordinary", ordinary),
        "control": Provider("control", failure),
        "completed": Provider("completed"),
    }
    monkeypatch.setattr(TrainingProviderFactory, "_instances", retained.copy())
    with pytest.raises(BaseException) as observed:
        await TrainingProviderFactory.close_providers()
    assert set(joined) == set(retained)
    assert is_execution_control_error(observed.value)
    chain = tuple(_explicit_control_chain(observed.value))
    assert failure in chain and ordinary in chain
    assert TrainingProviderFactory._instances == {
        "ordinary": retained["ordinary"], "control": retained["control"],
    }


@pytest.mark.asyncio
async def test_restarted_adapter_does_not_claim_unknown_original_resources_cleaned(monkeypatch):
    """No synthetic adoption: unknown local ID keeps external custody unresolved."""
    restarted = RunPodTrainingAdapter()
    monkeypatch.setattr(restarted, "_get_manager", _unexpected)
    with pytest.raises(TrainingStatusError, match="Unknown job"):
        await restarted.get_status("retained-original-job")
    with pytest.raises(DownloadError, match="Unknown job"):
        await restarted.download_weights("retained-original-job")
    assert await restarted.cancel("retained-original-job") is False
    with pytest.raises(TrainingProviderError, match="cleanup was not confirmed"):
        await restarted.cleanup("retained-original-job")
    assert restarted._manager is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["denied", "unknown", "group", "retirement"])
async def test_unknown_cleanup_does_not_wrap_original_native_status_control(monkeypatch, outcome):
    original = await _original_control(outcome)
    failure = TrainingStatusError("original status could not be read")
    failure.__cause__ = original

    def status(*_args, **_kwargs):
        raise failure

    adapter = RunPodTrainingAdapter()
    monkeypatch.setattr(adapter._lifecycle, "local_status", status)
    with pytest.raises(TrainingStatusError) as observed:
        await adapter.cleanup("retained-original-job")
    assert observed.value is failure
    assert observed.value.__cause__ is original
