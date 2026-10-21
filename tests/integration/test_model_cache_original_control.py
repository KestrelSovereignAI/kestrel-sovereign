"""Native refresh control remains observable after shared-cache callbacks.

Real local PostgreSQL COMMIT precedes a transparent acknowledgement fault.
No provider network or inference is involved. The public discovery consumer
uses an empty configured route inventory and the actual shared cache.
"""

import asyncio
import contextvars
import sqlite3

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    bind_execution_custody,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.llm.model_cache import SharedModelCache, VendorDiscoveryOutcome
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review30_pg import lose_ack
from tests.unit.test_execution_custody_review30 import service_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel"])
@pytest.mark.parametrize("entry", ["waiter", "late-waiter", "fresh", "stale", "early-snapshot",
                                 "forced", "tool", "publish", "restore", "amend"])
async def test_background_catalog_original_commit_control_survives_callback_and_readers(
    native_pg, tmp_path, monkeypatch, carrier, entry,
):
    backend, observer, schema = native_pg
    entered, release, finished, observing = (asyncio.Event() for _ in range(4))
    produced = asyncio.Event()
    lose_ack(monkeypatch, cancelled=False, entered=entered, release=release)
    cache = SharedModelCache()
    models = [ModelInfo(id="native-catalog", provider="native", display_name="Native catalog",
                        category=ModelCategory.CHAT)]
    if entry == "fresh":
        cache.set(models)
    else:
        cache.set_stale(models)
    initial_snapshot = cache.snapshot()
    monkeypatch.setattr("kestrel_sovereign.llm.model_discovery.get_shared_model_cache", lambda: cache)
    service = service_owner()
    service.config = {}
    originals, original_graphs = [], []
    selected_path = tmp_path / "actual-discovery-entry.db"
    actual_select = service._select_discovery_routes

    def observe_actual_selection():
        with sqlite3.connect(selected_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS actual_entry (id INTEGER)")
            connection.execute("INSERT INTO actual_entry VALUES (1)")
        return actual_select()

    monkeypatch.setattr(service, "_select_discovery_routes", observe_actual_selection)

    async def refresh():
        try:
            await backend.execute("INSERT INTO effects VALUES (181, 'original catalog refresh')")
        except BaseException as error:
            originals.append(error)
            original_graphs.append(tuple(_explicit_control_chain(error)))
            produced.set()
            raise

    async def consume():
        # A distinct agent reads this process-wide cache under its OWN live
        # custody, not the refresh writer's now-UNKNOWN transaction scope.
        with bind_execution_custody(GenerationFence()):
            observing.set()
            if entry in {"waiter", "late-waiter"}:
                return await cache.wait_for_refresh()
            if entry == "early-snapshot":
                return cache.snapshot()
            if entry == "publish":
                return cache.set(models)
            if entry == "restore":
                return cache.set_stale(models)
            if entry == "amend":
                return cache.record_failures(initial_snapshot.catalog_version,
                    {"native": VendorDiscoveryOutcome.failed("ordinary discovery outage")})
            if entry == "forced":
                return await service.discover_all_models(use_cache=False)
            if entry == "tool":
                from kestrel_sovereign.features.model.feature import ModelAgent
                feature = ModelAgent(agent=None)
                feature.llm_service = service
                return await feature.list_models(use_cache=False)
            return await service.discover_all_models(stale_while_revalidate=entry == "stale")

    source = reader = None
    with bind_execution_custody(GenerationFence()):
        try:
            assert cache.refresh_in_background(refresh)
            source = cache._refresh_task
            source.add_done_callback(lambda _: finished.set())
            await asyncio.wait_for(entered.wait(), 5)
            assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
            if entry == "waiter":
                reader = create_owned_task(consume(), context=contextvars.Context(), retain_originals=True)
                await asyncio.wait_for(observing.wait(), 5)
                await asyncio.sleep(0)
            if carrier == "cancel":
                source.cancel("actual original catalog COMMIT observer withdrawal")
            # Caller withdrawal is not permission to abandon the already
            # accepted native COMMIT owner; release its physical epilogue.
            release.set()
            if entry == "early-snapshot":
                # The original producer sets this Event immediately before
                # raising. Its reader wakes before done callbacks are queued;
                # the actual Task is done but callback bookkeeping is pending.
                async with asyncio.timeout(5):
                    await produced.wait()
                assert source.done() and cache._refresh_task is source
                early_error = None
                try:
                    cache.snapshot()
                except BaseException as error:
                    early_error = error
                assert early_error is not None, "completed original refresh escaped cache read before callback"
                assert originals[0] in tuple(_explicit_control_chain(early_error))
            await await_owned_task(source)
            await asyncio.wait_for(finished.wait(), 5)
            assert len(originals) == 1 and execution_commit_outcome(originals[0]) == "unknown"
            for attempt in range(3):
                if reader is None:
                    reader = create_owned_task(consume(), context=contextvars.Context(), retain_originals=True)
                outcome = await await_owned_task(reader)
                reader = None
                assert not selected_path.exists(), "forced discovery entered new native work before observing original shared control"
                assert outcome.error is not None, "shared refresh control became an ordinary cache miss or catalog"
                assert originals[0] in tuple(_explicit_control_chain(outcome.error)), (
                    "shared refresh discarded its actual original native commit receipt"
                )
                assert is_execution_control_error(outcome.error)
                assert execution_commit_outcome(outcome.error) == "unknown"
                assert tuple(_explicit_control_chain(originals[0])) == original_graphs[0]
                if attempt == 1:
                    cache.clear()  # Catalog invalidation cannot settle native uncertainty.
            async def try_independent_refresh():
                with bind_execution_custody(GenerationFence()):
                    return cache.refresh_in_background(refresh)

            refusal_reader = create_owned_task(try_independent_refresh(),
                                               context=contextvars.Context(), retain_originals=True)
            refused = (await await_owned_task(refusal_reader)).error
            assert refused is not None, "shared refresh restarted after unresolved original commit control"
            assert originals[0] in tuple(_explicit_control_chain(refused)), (
                "shared refresh restarted after unresolved original commit control"
            )
            assert tuple(_explicit_control_chain(originals[0])) == original_graphs[0]
            assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
        finally:
            release.set()
            if reader is not None:
                await await_owned_task(reader)
            # Baseline failure can schedule a replacement on the stale path.
            # Retire only that exact fixture-owned Task, never a foreign process.
            current = cache._refresh_task
            if current is not None and current is not source:
                if not current.done():
                    current.cancel()
                await await_owned_task(current)
            if source is not None:
                if not source.done():
                    source.cancel()
                await await_owned_task(source)

@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel"])
@pytest.mark.parametrize("consumer", ["discovery", "tool"])
async def test_cached_catalog_rechecks_original_refresh_after_reconciliation(
    native_pg, monkeypatch, carrier, consumer,
):
    backend, observer, schema = native_pg
    committed, ack_release, reconciling, reader_release = (asyncio.Event() for _ in range(4))
    cache = SharedModelCache()
    cache.set_stale([])
    service = service_owner()
    service.config = {}
    monkeypatch.setattr("kestrel_sovereign.llm.model_discovery.get_shared_model_cache", lambda: cache)
    originals, graphs = [], []
    armed = False
    actual_reconcile = service.reconcile_embedding_capabilities

    async def reconcile_then_wait(*, use_cache=True):
        await actual_reconcile(use_cache=use_cache)
        if armed:
            reconciling.set()
            await reader_release.wait()

    monkeypatch.setattr(service, "reconcile_embedding_capabilities", reconcile_then_wait)

    async def refresh():
        with bind_execution_custody(GenerationFence()):
            try:
                await backend.execute("INSERT INTO effects VALUES (183, 'pending original refresh')")
            except BaseException as error:
                originals.append(error)
                graphs.append(tuple(_explicit_control_chain(error)))
                raise

    async def consume():
        with bind_execution_custody(GenerationFence()):
            if consumer == "tool":
                from kestrel_sovereign.features.model.feature import ModelAgent
                feature = ModelAgent(agent=None)
                feature.llm_service = service
                return await feature.list_models(use_cache=True)
            return await service.discover_all_models(use_cache=True)

    source = reader = None
    try:
        with monkeypatch.context() as ack:
            lose_ack(ack, cancelled=False, entered=committed, release=ack_release)
            assert cache.refresh_in_background(refresh)
            source = cache._refresh_task
            await asyncio.wait_for(committed.wait(), 5)
            # Actual independent forced discovery publishes its empty configured
            # inventory while the SAME original refresh remains pending.
            with bind_execution_custody(GenerationFence()):
                assert await service.discover_all_models(use_cache=False) == []
            assert cache.fresh_snapshot() is not None and not source.done()
            armed = True
            reader = create_owned_task(consume(), context=contextvars.Context(), retain_originals=True)
            await asyncio.wait_for(reconciling.wait(), 5)
            if carrier == "cancel":
                source.cancel("actual original catalog COMMIT observer withdrawal")
            ack_release.set()
            await await_owned_task(source)
            assert len(originals) == 1 and execution_commit_outcome(originals[0]) == "unknown"
            reader_release.set()
            outcome = await await_owned_task(reader)
        assert outcome.error is not None, "cache-hit completion disclosed healthy catalog after original shared control"
        assert is_execution_control_error(outcome.error)
        assert execution_commit_outcome(outcome.error) == "unknown"
        assert originals[0] in tuple(_explicit_control_chain(outcome.error))
        assert tuple(_explicit_control_chain(originals[0])) == graphs[0]
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=183') == 1
    finally:
        ack_release.set()
        reader_release.set()
        if reader is not None:
            await await_owned_task(reader)
        if source is not None:
            await await_owned_task(source)


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_debt", [False, True])
async def test_stopped_refresh_is_local_only_without_native_retirement_debt(
    tmp_path, monkeypatch, cleanup_debt,
):
    from kestrel_sovereign.execution_custody import (
        execution_cleanup_debts, execution_denial_scopes,
        execution_terminal_error, require_execution_work,
    )
    from kestrel_sovereign.llm import codex_adapter

    monkeypatch.setattr(codex_adapter.tempfile, "tempdir", str(tmp_path))
    cache = SharedModelCache()
    original_models = [ModelInfo(id="prior-shared-catalog", provider="native",
                                 display_name="Prior", category=ModelCategory.CHAT)]
    cache.set_stale(original_models)
    entered, release, observing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    paths = []

    async def refresh():
        if cleanup_debt:
            _, owned = codex_adapter._materialize_app_server_data_images(
                [{"type": "image", "url": "data:image/png;base64,AAAA"}],
                retirement_owner=asyncio.current_task(),
            )
            paths.extend(owned)
            paths[0].unlink()
            paths[0].mkdir()  # Actual owned-leaf replacement faults native unlink.
        entered.set()
        await release.wait()
        try:
            require_execution_work()
        except BaseException as original:
            failures = codex_adapter._cleanup_app_server_data_images(
                paths, retirement_owner=asyncio.current_task(),
            )
            raise execution_terminal_error(original, *failures, retain_originals=True,
                                           isolate_control_originals=True)

    async def observe():
        with bind_execution_custody(GenerationFence()):
            observing.set()
            return await cache.wait_for_refresh()

    source = waiter = replacement = None
    try:
        with bind_execution_custody(GenerationFence()) as owner_scope:
            assert cache.refresh_in_background(refresh)
            source = cache._refresh_task
            await asyncio.wait_for(entered.wait(), 5)
            waiter = create_owned_task(observe(), context=contextvars.Context(), retain_originals=True)
            await asyncio.wait_for(observing.wait(), 5)
            await asyncio.sleep(0)
        # The real owner context has retired, while the independent observer
        # still joins that SAME original refresh rather than replacing it.
        release.set()
        original = (await await_owned_task(source)).error
        observed = (await await_owned_task(waiter)).error
        assert is_execution_control_error(original)
        if cleanup_debt:
            assert execution_denial_scopes(original) is None
        else:
            assert owner_scope in execution_denial_scopes(original)
        assert original in tuple(_explicit_control_chain(observed))
        assert execution_commit_outcome(original) is None
        with bind_execution_custody(GenerationFence()):
            other_error = None
            try:
                models = cache.get_any()
            except BaseException as error:
                other_error = error
            if cleanup_debt:
                assert other_error is not None, "unsettled native refresh retirement became a healthy shared cache"
                assert original in tuple(_explicit_control_chain(other_error))
                assert any(resource is paths[0] and task is source
                           for resource, task in execution_cleanup_debts(other_error))
            else:
                assert other_error is None, "a stopped refresh owner became a shared-cache Hold on another agent"
                assert models == original_models

                async def independent_refresh():
                    cache.set([ModelInfo(id="independent-catalog", provider="native",
                                         display_name="Independent", category=ModelCategory.CHAT)])

                assert cache.refresh_in_background(independent_refresh)
                replacement = cache._refresh_task
                assert (await await_owned_task(replacement)).error is None
                assert cache.get()[0].id == "independent-catalog"
    finally:
        release.set()
        for task in (source, waiter, replacement):
            if task is not None:
                await await_owned_task(task)
        for path in paths:
            path.rmdir()


@pytest.mark.asyncio
async def test_background_refresh_cannot_enter_factory_after_original_scope_retirement(tmp_path):
    from kestrel_sovereign.execution_custody import require_execution_work

    cache = SharedModelCache()
    leaf = tmp_path / "not-admitted"
    async def factory():
        leaf.write_bytes(b"actual forbidden native entry")
        require_execution_work()

    source = None
    try:
        with bind_execution_custody(GenerationFence()):
            assert cache.refresh_in_background(factory)
            source = cache._refresh_task
        outcome = await await_owned_task(source)
        assert outcome.error is not None and is_execution_control_error(outcome.error)
        assert not leaf.exists(), "shared refresh entered its factory after original runtime admission retired"
    finally:
        if source is not None:
            await await_owned_task(source)
        leaf.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_ordinary_native_refresh_failure_remains_availability_not_shared_control():
    import aiosqlite

    closed = await aiosqlite.connect(":memory:")
    await closed.close()
    cache = SharedModelCache()
    models = [ModelInfo(id="prior-ordinary-catalog", provider="native",
                        display_name="Prior", category=ModelCategory.CHAT)]
    cache.set_stale(models)
    failures = []

    async def refresh():
        try:
            await closed.execute("SELECT 1")
        except ValueError as error:
            failures.append(error)
            raise

    assert cache.refresh_in_background(refresh)
    source = cache._refresh_task
    outcome = await await_owned_task(source)
    assert failures and outcome.error is failures[0] and not is_execution_control_error(outcome.error)
    cache_error = None
    try:
        current = cache.get_any()
    except BaseException as error:
        cache_error = error
    assert cache_error is None, "ordinary native availability became shared execution control"
    assert current == models and await cache.wait_for_refresh() is False
