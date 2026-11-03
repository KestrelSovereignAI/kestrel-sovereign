"""Runtime admission must not buy a native Pod for a known unsupported worker.

Only canonical current host/SQLite and owned TLS peers are used. No remote
worker, provider registration, or physical billing conclusion is implied.
"""
import json
from datetime import UTC, datetime
import os
from pathlib import Path

import pytest
from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionStorage
from kestrel_sdk.resources import ResourceUnsupported

from tests.integration.test_visual_custodied_runtime import NativeFixtureProvider, runtime_for
from tests.integration.test_visual_remote_runtime_review import NoRemoteWorker
from tests.integration.test_visual_runpod_acquisition import SCOPE, original_host
from tests.integration.test_visual_runpod_acquisition import custody as custody, vendor as vendor, peer as peer
from tests.integration.test_visual_runpod_quote import quoted_operation
from tests.integration.test_visual_runpod_quote import quote_peer as quote_peer


class NativeQuotedProvider(NativeFixtureProvider):
    native_environment_host_api_version = 1
    native_environment_lifecycle_api_version = 1

    def __init__(self, host):
        super().__init__(host, mode='genuine')
        self.environment_host = host.host

    async def capabilities(self):
        return self.host.caps

    async def quote(self, request):
        async with self.environment_host.quote_operation(request) as call:
            return await call.run_original()

    async def inspect(self, scope, lease):
        async with self.environment_host.inspect_operation(scope, lease) as call:
            return await call.run_original()

    async def acquire_with_custody(self, request, quote, custody):
        from kestrel_cloud_runpod.cpu_environment_codec import deserialize_cpu_environment_plan
        row = await self.host.store.get(SCOPE, request_id=request.request_id, guard=self.host.check)
        projection = row.remote_environment
        assert projection is not None
        self.host.plan = deserialize_cpu_environment_plan(projection.plan_bytes,
            expected_storage_sha256=projection.plan_storage_sha256)
        self.host.peer.reply = dict(id='owned-native-pod', name='kestrel-cpu-'+request.digest()[:32],
            status='RUNNING', image=self.host.plan.profile.image, dataCenterId='US-TX-3', disk=20,
            cloud='SECURE', ports=[], env={}, mounts={}, args='', registry=None, locked=False,
            template=None, globalNetworking={'enabled': False}, startedAt=datetime.now(UTC).isoformat(),
            cpu=dict(id='cpu5c', vcpuCount=4, memory=10), cost=0.16)
        return await super().acquire_with_custody(request, quote, custody)


@pytest.mark.asyncio
@pytest.mark.parametrize('quote_kind', ['unquoted', 'quoted'])
async def test_runtime_refuses_native_create_before_known_worker_refusal(
    custody, vendor, peer, quote_peer, monkeypatch, quote_kind, tmp_path
):
    selected = peer if quote_kind == 'unquoted' else quote_peer
    scope = (original_host(custody, vendor, peer, monkeypatch) if quote_kind == 'unquoted'
        else quoted_operation(custody, vendor, quote_peer, monkeypatch, activate=False))
    caught = None
    async with scope as host:
        host.peer = selected
        provider = (NativeFixtureProvider(host, mode='genuine') if quote_kind == 'unquoted'
            else NativeQuotedProvider(host))
        if quote_kind == 'quoted':
            runtime = host.runtime
            runtime.providers[host.request.provider_id] = provider
            host.host._provider = None
            host.host._bind_provider(provider)
        else:
            runtime = runtime_for(host, provider, NoRemoteWorker())
        try:
            await runtime.admit(host.context, host.request)
        except BaseException as error:
            caught = error
        row = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        storage = (EnvironmentAcquisitionStorage.deserialize_private(row.acquisition.storage)
            if row.acquisition is not None else None)
        checkpoint = None if storage is None else storage.checkpoint
        facts = dict(quote_kind=quote_kind, error_type=None if caught is None else type(caught).__name__,
            error=None if caught is None else str(caught), paths=[r[0] for r in selected.requests],
            native_create_count=sum(r[0] == '/v2/pods' for r in selected.requests),
            phase=row.phase, private_quote=row.remote_environment is not None,
            checkpoint=None if checkpoint is None else checkpoint.state,
            saved_preparation=storage is not None and storage.preparation is not None,
            saved_native_binding=row.acquisition is not None and row.acquisition.binding is not None,
            revision=row.revision,
            lease=None if row.lease is None else row.lease.provider_resource_id,
            legacy_calls=provider.legacy_calls, connects=provider.connects)
    facts['native_closed'] = host.native.is_closed
    facts['original_close_count'] = len(host.closes)
    destination = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', str(tmp_path)))
    (destination/('admit-'+quote_kind+'.json')).write_text(json.dumps(facts, indent=2)+'\n')
    assert facts['native_create_count'] == 0, 'known unsupported worker admission sent a native paid-create POST'
    assert isinstance(caught, ResourceUnsupported) and 'worker' in str(caught).lower()
    assert provider.connects == provider.legacy_calls == 0
    assert row.lease is None and (checkpoint is None or checkpoint.state == 'unprepared')
    assert storage is None or storage.preparation is None
    assert row.acquisition is None or row.acquisition.binding is None
    assert facts['native_closed'] and facts['original_close_count'] == 1
    assert not host.store._original_results and not host.store.db.backend._original_result_holds


@pytest.mark.asyncio
@pytest.mark.parametrize('door', ['copied-facade', 'copied-attempt', 'unwrapped-root',
    'detached-consumer', 'detached-registration', 'coherent-detachment'])
async def test_runtime_consumer_cannot_be_copied_or_detached_before_native_attach(
    custody, vendor, peer, monkeypatch, door
):
    import copy
    from dataclasses import replace
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.visual_computer.storage_acquisition import _CoreAcquisitionAttempt

    observed = {"expected_inner_EA": False, "post_inner_assertions": False}

    class RestrictedProvider(NativeFixtureProvider):
        async def acquire_with_custody(self, request, quote, custody):
            self.modern_calls += 1
            if door == 'copied-facade':
                custody = copy.copy(custody)
            async with custody.operation(request, quote) as attempt:
                root = attempt._result
                assert root._runtime_custody is custody
                assert root._acceptance.runtime_custody is root._registration.runtime_custody is custody
                before = await self.host.store.db.fetchone(
                    'SELECT payload, revision FROM visual_computer_requests')
                original = root._runtime_custody, root._registration, attempt._runtime_custody
                try:
                    if door == 'copied-attempt':
                        selected = copy.copy(attempt)
                    elif door == 'unwrapped-root':
                        selected = _CoreAcquisitionAttempt(root)
                    else:
                        selected = attempt
                    if door in {'detached-consumer', 'coherent-detachment'}:
                        root._runtime_custody = None
                    if door in {'detached-registration', 'coherent-detachment'}:
                        root._registration = replace(root._registration, runtime_custody=None)
                    if door == 'coherent-detachment':
                        attempt._runtime_custody = None
                    with pytest.raises(ExecutionAuthorityError):
                        await self._run_original_attempt(selected, request, quote)
                    observed["expected_inner_EA"] = True
                finally:
                    root._runtime_custody, attempt._runtime_custody = original[0], original[2]
                    # Preserve every genuine native writer/join receipt if an
                    # unsafe inverse actually advances the row; restore only
                    # the adversarially changed restriction field.
                    root._registration = replace(root._registration, runtime_custody=original[0])
                after = await self.host.store.db.fetchone(
                    'SELECT payload, revision FROM visual_computer_requests')
                assert bytes(after[0]) == bytes(before[0]) and after[1] == before[1]
                assert root._native is None and attempt.checkpoint.state == 'unprepared'
                observed["post_inner_assertions"] = True
                raise ResourceUnsupported('native worker admission remains refused')

    async with original_host(custody, vendor, peer, monkeypatch) as host:
        provider = RestrictedProvider(host, mode='genuine')
        runtime = runtime_for(host, provider, NoRemoteWorker())
        if door == 'copied-facade':
            with pytest.raises(ExecutionAuthorityError):
                await runtime.admit(host.context, host.request)
        else:
            with pytest.raises(ResourceUnsupported, match='worker'):
                await runtime.admit(host.context, host.request)
        current = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        assert current.lease is None and not peer.requests and not peer.snis
        assert current.acquisition.binding is None
        saved = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage)
        assert saved.checkpoint.state == 'unprepared' and saved.preparation is None
    if door != 'copied-facade':
        assert observed == {"expected_inner_EA": True, "post_inner_assertions": True}
    assert host.native.is_closed and len(host.closes) == 1
    assert not host.store._original_results and not host.store.db.backend._original_result_holds


@pytest.mark.asyncio
async def test_explicit_repository_source_conformance_keeps_original_once_post(
    custody, vendor, peer, monkeypatch
):
    from tests.integration.test_visual_runpod_acquisition import original_acquisition
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        root = original.attempt._result
        assert root._runtime_custody is root._acceptance.runtime_custody is root._registration.runtime_custody is None
        assert await original.attempt.begin_dispatch() is True
        phase = await original.handle.run_original()
        checkpoint = await original.handle.persist_minimal()
        original.producer.finish_optional_cost(phase)
        await original.attempt.record_accepted(checkpoint.lease.model_copy(update={'state': 'pending'}))
    assert len(peer.requests) == 1 and peer.requests[0][0] == '/v2/pods'
    assert original.native.is_closed and len(original.closes) == 1
    assert not original.store._original_results and not original.store.db.backend._original_result_holds


@pytest.mark.asyncio
@pytest.mark.parametrize('door', ['before', 'closed', 'after'])
@pytest.mark.parametrize('forwarding', ['original-guard', 'wrapped-guard', 'same-consumer'])
async def test_runtime_invocation_cannot_reissue_its_original_acquisition(
    custody, vendor, peer, monkeypatch, tmp_path, door, forwarding
):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError

    observed = {}

    class ReissuingProvider(NativeFixtureProvider):
        async def acquire_with_custody(self, request, quote, custody):
            self.modern_calls += 1
            root = custody._attempt._result
            assert root._repository._original_results[root._acceptance] is root
            observed['allocated_before_exposure'] = root.checkpoint.state == 'unprepared'

            async def reissue():
                current = await custody._repository.get(SCOPE, request_id=request.request_id,
                    guard=custody._guard)
                original = await self.host.store.db.fetchone(
                    'SELECT payload, revision FROM visual_computer_requests')
                selected_guard = custody._guard
                if forwarding == 'wrapped-guard':
                    def selected_guard():
                        custody._guard()
                kwargs = {'runtime_custody': custody} if forwarding == 'same-consumer' else {}
                observed['physical_closed_at_reissue'] = root._phase == 'closed'
                observed['restriction_retained'] = self.host.store._original_results[root._acceptance] is root
                try:
                    async with custody._repository.original_acquisition_result(SCOPE, current,
                            request, quote, guard=selected_guard, **kwargs) as direct:
                        # A deleted issuance guard must reach the actual once-POST
                        # producer, rather than merely change an error string.
                        await self._run_original_attempt(direct, request, quote)
                except ExecutionAuthorityError:
                    observed['issuance_refused'] = True
                changed = await self.host.store.db.fetchone(
                    'SELECT payload, revision FROM visual_computer_requests')
                observed['row_unchanged'] = bytes(original[0]) == bytes(changed[0]) and original[1] == changed[1]
                raise ResourceUnsupported('native worker admission remains refused')

            if door == 'before':
                await reissue()
            elif door == 'closed':
                async with custody.operation(request, quote):
                    await root.aclose()
                    assert root not in root._backend._original_result_holds and root._closed.done()
                    await reissue()
            else:
                async with custody.operation(request, quote) as attempt:
                    assert attempt is custody._attempt
                await reissue()

    async with original_host(custody, vendor, peer, monkeypatch) as host:
        provider = ReissuingProvider(host, mode='genuine')
        runtime = runtime_for(host, provider, NoRemoteWorker())
        with pytest.raises(ResourceUnsupported, match='worker'):
            await runtime.admit(host.context, host.request)
        row = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        saved = EnvironmentAcquisitionStorage.deserialize_private(row.acquisition.storage)
        observed.update(native_create_count=len(peer.requests), checkpoint=saved.checkpoint.state,
            saved_preparation=saved.preparation is not None, binding=row.acquisition.binding is not None,
            lease=row.lease is not None)
    observed.update(native_closed=host.native.is_closed, original_close_count=len(host.closes))
    Path(os.environ.get('CURRENT_ROW_PROOF_DIR', str(tmp_path))).joinpath(
        'invocation-'+door+'-'+forwarding+'.json').write_text(json.dumps(observed, indent=2)+'\n')
    assert observed['native_create_count'] == 0, 'runtime provider invocation shed restriction and sent POST'
    assert observed['issuance_refused'] and observed['restriction_retained'] and observed['row_unchanged']
    assert observed['allocated_before_exposure'] and observed['physical_closed_at_reissue'] == (door == 'closed')
    assert observed['checkpoint'] == 'unprepared' and not observed['saved_preparation']
    assert not observed['binding'] and not observed['lease']
    assert observed['native_closed'] and observed['original_close_count'] == 1
    assert not host.store._original_results and not host.store.db.backend._original_result_holds


@pytest.mark.asyncio
@pytest.mark.parametrize('door', ['detached-marker', 'detached-owner', 'detached-close'])
async def test_detached_runtime_registration_cannot_be_hidden_from_fresh_issuance(
    custody, vendor, peer, monkeypatch, tmp_path, door
):
    from dataclasses import replace
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    observed = {}
    class DetachingProvider(NativeFixtureProvider):
        async def acquire_with_custody(self, request, quote, custody):
            async with custody.operation(request, quote) as attempt:
                root = attempt._result
                original = root._owner_task, root._runtime_custody, attempt._runtime_custody
                try:
                    if door == 'detached-owner':
                        root._owner_task = None
                    else:
                        root._runtime_custody = None
                        root._registration = replace(root._registration, runtime_custody=None)
                        attempt._runtime_custody = None
                    if door == 'detached-close':
                        await root.aclose()
                    observed['restriction_entry_present'] = root._acceptance in self.host.store._original_results
                    current = await custody._repository.get(SCOPE, request_id=request.request_id, guard=custody._guard)
                    try:
                        async with custody._repository.original_acquisition_result(SCOPE, current,
                                request, quote, guard=custody._guard) as direct:
                            await self._run_original_attempt(direct, request, quote)
                    except ExecutionAuthorityError:
                        observed['issuance_refused'] = True
                finally:
                    root._owner_task, root._runtime_custody, attempt._runtime_custody = original
                    root._registration = replace(root._registration, runtime_custody=original[1])
                raise ResourceUnsupported('native worker admission remains refused')
    caught = None
    async with original_host(custody, vendor, peer, monkeypatch) as host:
        provider=DetachingProvider(host, mode='genuine')
        runtime=runtime_for(host, provider, NoRemoteWorker())
        try:
            await runtime.admit(host.context, host.request)
        except BaseException as error:
            caught=error
        row=await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        storage=EnvironmentAcquisitionStorage.deserialize_private(row.acquisition.storage)
        observed.update(native_create_count=len(peer.requests),phase=row.phase,checkpoint=storage.checkpoint.state,
            lease=None if row.lease is None else row.lease.provider_resource_id,
            caught=None if caught is None else type(caught).__name__,
            original_results=len(host.store._original_results),holds=len(host.store.db.backend._original_result_holds))
    observed.update(native_closed=host.native.is_closed,closes=len(host.closes))
    Path(os.environ.get('CURRENT_ROW_PROOF_DIR',str(tmp_path))).joinpath('detachment-'+door+'.json').write_text(json.dumps(observed,indent=2)+'\n')
    assert observed['native_create_count']==0, 'mutable root fields hid runtime restriction and sent POST'
    assert observed.get('issuance_refused') and observed['restriction_entry_present']
    assert isinstance(caught, ResourceUnsupported)
    assert observed['checkpoint']=='unprepared' and observed['lease'] is None
    assert observed['original_results']==observed['holds']==0
    assert observed['native_closed'] and observed['closes']==1


@pytest.mark.asyncio
async def test_runtime_invocation_cannot_reissue_from_original_joined_child(
    custody, vendor, peer, monkeypatch, tmp_path
):
    from kestrel_sovereign._async_ownership import OwnedTaskAdmission, OwnedTaskOutcome, create_owned_task, await_owned_task
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    observed={}
    class ChildProvider(NativeFixtureProvider):
        async def acquire_with_custody(self, request, quote, custody):
            async def direct():
                current=await custody._repository.get(SCOPE,request_id=request.request_id,guard=custody._guard)
                async with custody._repository.original_acquisition_result(SCOPE,current,request,quote,guard=custody._guard) as attempt:
                    await self._run_original_attempt(attempt,request,quote)
            admission=OwnedTaskAdmission()
            outcome=None
            try:
                create_owned_task(direct(),admission=admission,retain_originals=True,name='original-provider-reissue-child')
            finally:
                if admission.task is not None:
                    outcome=await await_owned_task(admission.task)
            observed['original_joined_outcome']=type(outcome) is OwnedTaskOutcome
            observed['issuance_refused']=isinstance(outcome.error,ExecutionAuthorityError)
            observed['child_error']=None if outcome.error is None else type(outcome.error).__name__
            observed['original_owner_cancellation']=outcome.cancellation is not None
            raise ResourceUnsupported('native worker admission remains refused')
    async with original_host(custody,vendor,peer,monkeypatch) as host:
        provider=ChildProvider(host,mode='genuine');runtime=runtime_for(host,provider,NoRemoteWorker())
        with pytest.raises(ResourceUnsupported,match='worker'):
            await runtime.admit(host.context,host.request)
        row=await host.store.get(SCOPE,request_id=host.request.request_id,guard=host.check)
        storage=EnvironmentAcquisitionStorage.deserialize_private(row.acquisition.storage)
        observed.update(native_create_count=len(peer.requests),checkpoint=storage.checkpoint.state,
            lease=None if row.lease is None else row.lease.provider_resource_id,
            original_results=len(host.store._original_results),holds=len(host.store.db.backend._original_result_holds))
    observed.update(native_closed=host.native.is_closed,closes=len(host.closes))
    Path(os.environ.get('CURRENT_ROW_PROOF_DIR',str(tmp_path))).joinpath('invocation-child.json').write_text(json.dumps(observed,indent=2)+'\n')
    assert observed['native_create_count']==0,'foreign Task bypassed original runtime invocation restriction and sent POST'
    assert observed['original_joined_outcome'] and observed['issuance_refused'] and not observed['original_owner_cancellation']
    assert observed['checkpoint']=='unprepared' and observed['lease'] is None
    assert observed['original_results']==observed['holds']==0 and observed['native_closed'] and observed['closes']==1


@pytest.mark.asyncio
@pytest.mark.parametrize('child_scope', ['descendant', 'after-return', 'independent'])
async def test_runtime_invocation_keeps_restriction_across_joined_child_and_other_quoted_row(
    custody, vendor, quote_peer, monkeypatch, tmp_path, child_scope
):
    from kestrel_sovereign._async_ownership import (
        OwnedTaskAdmission, OwnedTaskOutcome, create_owned_task, await_owned_task,
    )
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.visual_computer.repository import CustodyRecord
    from kestrel_cloud_runpod.cpu_environment_codec import deserialize_cpu_environment_plan

    import asyncio
    observed = {'child_scope': child_scope}
    admission = OwnedTaskAdmission()
    allow_child = asyncio.Event()
    joined = []

    async def join_child():
        if not joined and admission.task is not None:
            joined.append(await await_owned_task(admission.task))
        outcome = joined[0]
        observed['original_child_joined'] = type(outcome) is OwnedTaskOutcome
        observed['child_error'] = None if outcome.error is None else type(outcome.error).__name__
        observed['original_cancellation'] = outcome.cancellation is not None
        observed['issuance_refused'] = isinstance(outcome.error, ExecutionAuthorityError)

    class ChildProvider(NativeQuotedProvider):
        async def acquire_with_custody(self, request, quote, custody):
            self.modern_calls += 1
            parent_task = custody._task
            def forwarding_guard():
                custody._guard()
            async def other_request():
                await allow_child.wait()
                observed['different_task'] = asyncio.current_task() is not parent_task
                observed['different_guard'] = forwarding_guard is not custody._guard
                current = await self.host.store.get(SCOPE, request_id=other.request.request_id,
                    guard=forwarding_guard)
                observed['different_row'] = current.request.request_id != request.request_id
                before = await self.host.store.db.fetchone(
                    'SELECT payload, revision FROM visual_computer_requests WHERE request_id = ?',
                    (other.request.request_id,))
                projection = current.remote_environment
                self.host.plan = deserialize_cpu_environment_plan(projection.plan_bytes,
                    expected_storage_sha256=projection.plan_storage_sha256)
                self.host.peer.reply = dict(id='owned-native-other-pod',
                    name='kestrel-cpu-'+other.request.digest()[:32], status='RUNNING',
                    image=self.host.plan.profile.image, dataCenterId='US-TX-3', disk=20,
                    cloud='SECURE', ports=[], env={}, mounts={}, args='', registry=None,
                    locked=False, template=None, globalNetworking={'enabled': False},
                    startedAt=datetime.now(UTC).isoformat(),
                    cpu=dict(id='cpu5c', vcpuCount=4, memory=10), cost=0.16)
                try:
                    async with self.host.store.original_acquisition_result(SCOPE, current,
                            other.request, other.quote, guard=forwarding_guard) as attempt:
                        await self._run_original_attempt(attempt, other.request, other.quote)
                finally:
                    after = await self.host.store.db.fetchone(
                        'SELECT payload, revision FROM visual_computer_requests WHERE request_id = ?',
                        (other.request.request_id,))
                    observed['other_row_unchanged'] = bytes(before[0]) == bytes(after[0]) and before[1] == after[1]
            self.other_request = other_request
            if child_scope != 'independent':
                create_owned_task(other_request(), admission=admission, retain_originals=True,
                    name='original-provider-other-request-child')
            if child_scope != 'after-return':
                allow_child.set()
                await join_child()
            raise ResourceUnsupported('native worker admission remains refused')

    async with quoted_operation(custody, vendor, quote_peer, monkeypatch, activate=False) as host:
        host.peer = quote_peer
        provider = ChildProvider(host)
        host.runtime.providers[host.request.provider_id] = provider
        host.host._provider = None
        host.host._bind_provider(provider)
        other_request = host.request.model_copy(update={'request_id': 'other-original-request'})
        other = await host.store.admit(SCOPE,
            CustodyRecord(request=other_request, session_id='other-original-session'), host.check)
        async with host.host._activate_quote(host.context, other) as call:
            other_quote = await call.run_original()
            other = await host.store._accepted_runpod_quote(SCOPE, call, other_quote)
        observed['other_real_private_quote'] = other.remote_environment is not None
        observed['other_quoted_revision'] = other.revision
        if child_scope == 'independent':
            async def independent_child():
                await allow_child.wait()
                await provider.other_request()
            # Created before Runtime activation: independent original task,
            # with the same authentic live host grant, not a copied context.
            create_owned_task(independent_child(), admission=admission, retain_originals=True,
                name='independently-admitted-original-acquisition')
        try:
            with pytest.raises(ResourceUnsupported, match='worker'):
                await host.runtime.admit(host.context, host.request)
        finally:
            if child_scope == 'after-return':
                observed['parent_roots_retired_before_child'] = not host.store._original_results
            allow_child.set()
            if admission.task is not None:
                await join_child()
        primary = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        secondary = await host.store.get(SCOPE, request_id=other.request.request_id, guard=host.check)
        observed.update(native_create_count=sum(r[0] == '/v2/pods' for r in quote_peer.requests),
            catalog_count=sum(r[0] != '/v2/pods' for r in quote_peer.requests),
            primary_state=primary.phase, primary_lease=primary.lease is not None,
            secondary_state=secondary.phase, secondary_lease=secondary.lease is not None,
            secondary_binding=secondary.acquisition is not None and secondary.acquisition.binding is not None,
            remaining_roots=len(host.store._original_results),
            remaining_holds=len(host.store.db.backend._original_result_holds))
    observed.update(native_closed=host.native.is_closed, original_closes=len(host.closes))
    Path(os.environ.get('CURRENT_ROW_PROOF_DIR', str(tmp_path))).joinpath(
        'invocation-other-row-child-'+child_scope+'.json').write_text(json.dumps(observed, indent=2)+'\n')
    restricted = child_scope != 'independent'
    assert observed['native_create_count'] == (0 if restricted else 1), 'runtime acquisition scope differs'
    assert observed['original_child_joined'] and observed['issuance_refused'] is restricted
    assert observed['different_task'] and observed['different_guard'] and observed['different_row']
    assert observed['other_real_private_quote'] and observed['other_row_unchanged'] is restricted
    assert not observed['original_cancellation'] and not observed['primary_lease']
    assert observed['secondary_lease'] is (not restricted) and observed['secondary_binding'] is (not restricted)
    if child_scope == 'after-return':
        assert observed['parent_roots_retired_before_child']
    assert observed['remaining_roots'] == observed['remaining_holds'] == 0
    assert observed['native_closed'] and observed['original_closes'] == 1
