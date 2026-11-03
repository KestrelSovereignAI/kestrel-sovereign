"""Genuine optional canonical catalog TLS -> exact same-row CURRENT quote ACK.

The owned peer is not Runpod and declared fixture caps do not prove a remote
worker. No provider is registered, no cloud/paid create or connect is performed.
"""

import copy
import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace

import pytest
from kestrel_sdk.computer_use import EnvironmentCapabilities
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceOutcomeUnknown
from pydantic import SecretBytes, SecretStr

from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.visual_computer.repository import (
    CustodyRecord,
    VisualCustodyRepository,
)
from kestrel_sovereign.visual_computer.runpod_environment import RunpodEnvironmentHost
from kestrel_sovereign.visual_computer.runtime import (
    VisualComputerRuntime,
    VisualPolicy,
)
from tests.integration.test_visual_custody_backends import SCOPE
from tests.integration.test_visual_custody_backends import (
    custody as custody,  # noqa: PLC0414 - pytest fixture re-export.
)
from tests.integration.test_visual_runpod_acquisition import original_host
from tests.integration.test_visual_runpod_acquisition import (
    vendor as vendor,  # noqa: PLC0414 - pytest fixture re-export.
)
from tests.utils.tls import make_tls_contexts, tls_http_server

_REFUSALS = (
    ExecutionAuthorityError,
    OperatorAuthorizationError,
    ResourceOutcomeUnknown,
    BaseExceptionGroup,
    OSError,
)

CATALOG = {
    "cpus": [
        {
            "id": "cpu5c",
            "name": "Compute",
            "group": "Gen 5",
            "vcpu": {"min": 2, "max": 32},
            "ramGbPerVcpu": 2.5,
            "price": {"securePerVcpu": 0.04, "serverlessPerVcpu": 0.03},
            "availability": "HIGH",
            "dataCenters": [
                {"id": "US-TX-3", "name": "Texas", "availability": "MEDIUM"}
            ],
        }
    ]
}


@pytest.fixture
def quote_peer(tmp_path):
    context, _, certificate = make_tls_contexts(tmp_path)
    state = SimpleNamespace(
        requests=[],
        snis=[],
        reply=CATALOG,
        raw=None,
        encoding=[],
        before_reply=None,
        resets=[],
        chunked=False,
    )
    context.set_servername_callback(lambda sock, name, ctx: state.snis.append(name))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError) as error:
                state.resets.append(type(error).__name__)

        def do_GET(self):
            state.requests.append(
                (self.path, dict(self.headers), self.connection.version())
            )
            if state.before_reply is not None:
                state.before_reply()
            raw = json.dumps(state.reply).encode() if state.raw is None else state.raw
            self.send_response(getattr(state, "get_status", 200))
            self.send_header("Content-Type", "application/json")
            for value in state.encoding:
                self.send_header("Content-Encoding", value)
            if state.chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            if state.chunked:
                for offset in range(0, len(raw), 8192):
                    chunk = raw[offset : offset + 8192]
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            else:
                self.wfile.write(raw)

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            state.requests.append((self.path, dict(self.headers), self.connection.version()))
            raw = json.dumps(state.reply).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_DELETE(self):
            state.requests.append((self.path, dict(self.headers), self.connection.version()))
            if state.before_reply is not None:
                state.before_reply()
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    with tls_http_server(context, Handler) as server:
        state.url = f"https://127.0.0.1:{server.server_port}/v2"
        state.ca = certificate.read_bytes()
        yield state


@asynccontextmanager
async def quoted_operation(
    custody, vendor, quote_peer, monkeypatch, *, operation_seconds=120, activate=True
):
    async with original_host(
        custody, vendor, quote_peer, monkeypatch, operation_seconds=operation_seconds
    ) as original:
        store = original.store
        # Explicit test-only provider correlation/caps; not remote registration.
        provider = object()

        def permitted_destination(location):
            assert location == "remote"

        runtime = VisualComputerRuntime(
            store,
            original.authority,
            providers={original.request.provider_id: provider},
            drivers={},
            check_destination=permitted_destination,
            policy=VisualPolicy(
                permitted_locations=frozenset({"remote"}),
                max_lifetime_seconds=3600,
                max_hourly_cost_usd=original.request.max_hourly_cost_usd,
                max_total_cost_usd=original.request.max_total_cost_usd,
            ),
        )
        caps = EnvironmentCapabilities(
            provider_id=original.request.provider_id,
            location="remote",
            surfaces=frozenset({"browser"}),
            operations=frozenset({"screenshot", "click"}),
            cpu_only=True,
            profile_persistence=False,
        )
        host = RunpodEnvironmentHost(
            runtime,
            profile=original.plan.profile,
            noncompute=original.plan.noncompute,
            capabilities=caps,
            account="owned-native-account",
            api_key=SecretStr("owned-native-api-key"),
            base_url=quote_peer.url,
            ca_data=SecretBytes(quote_peer.ca),
            credential_selector="owned-fixture",
        )
        host._bind_provider(provider)
        value = await store.admit(
            SCOPE,
            CustodyRecord(request=original.request, session_id="native-quote-session"),
            original.check,
        )
        # original_host configures a create reply; this operation serves catalog only.
        quote_peer.reply = CATALOG
        fields = dict(
            vars(original),
            runtime=runtime,
            host=host,
            record=value,
            caps=caps,
            provider=provider,
        )
        if activate:
            async with host._activate_quote(original.context, value) as call:
                yield SimpleNamespace(**fields, call=call)
        else:
            yield SimpleNamespace(**fields)


@pytest.mark.asyncio
async def test_native_catalog_quote_atomic_private_plan_ack_and_reopen(
    custody, vendor, quote_peer, monkeypatch
):
    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        with pytest.raises(ExecutionAuthorityError):
            h.call.check_parse_current()
        async with h.host.quote_operation(h.request) as original:
            quote = await original.run_original()
        assert h.call._client_closed and h.call._worker_exited
        assert h.call._ack is not None
        assert len(quote_peer.requests) == 1
        assert (
            quote_peer.requests[0][0]
            == "/v2/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=4"
        )
        assert quote_peer.requests[0][1]["Accept-Encoding"] == "identity"
        assert quote_peer.requests[0][2] == "TLSv1.3"
        value = await h.store._accepted_runpod_quote(SCOPE, h.call, quote)
        assert value.phase == "quoted" and value.quote == quote and value.revision == 1
        projection = value.remote_environment
        assert projection is not None and projection.capabilities == h.caps
        codec = __import__(
            "kestrel_cloud_runpod.cpu_environment_codec",
            fromlist=["deserialize_cpu_environment_plan"],
        )
        plan = codec.deserialize_cpu_environment_plan(
            projection.plan_bytes,
            expected_storage_sha256=projection.plan_storage_sha256,
        )
        codec.require_cpu_environment_quote(
            plan, quote, capabilities=h.caps, at=plan.planned_at
        )
        assert plan.request == h.request
        raw = bytes(
            (await h.store.db.fetchone("SELECT payload FROM visual_computer_requests"))[
                0
            ]
        )
        assert b"owned-native-account" not in raw and b"cpu5c" not in raw
        assert "remote_environment" not in value.model_dump()
        assert "owned-native-account" not in repr(value)
        assert projection.model_dump() == {"schema_version": 1}
        with pytest.raises(ExecutionAuthorityError):
            await original.run_original()
        with pytest.raises(ExecutionAuthorityError):
            original.check_parse_current()
        db = await custody[1]()
        try:
            restored = await VisualCustodyRepository(db).get(
                SCOPE, request_id=h.request.request_id, guard=h.check
            )
            assert (
                restored.remote_environment.private_value()
                == projection.private_value()
            )
            assert restored.quote == quote and len(quote_peer.requests) == 1
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_current_quote_owner_never_admitted_through_late_result_entrypoints(
    custody, vendor, quote_peer, monkeypatch
):
    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        async with h.store._current_runpod_quote(h.call) as current:
            from kestrel_sovereign.storage.db.original_result_custody import (
                _original_result_profile,
            )

            with pytest.raises(ExecutionAuthorityError):
                _original_result_profile(current)
            entered = []
            native = h.store.db.backend._write_native_row

            async def observe_engine(owner):
                entered.append(owner)
                return await native(owner)

            monkeypatch.setattr(h.store.db.backend, "_write_native_row", observe_engine)
            for method in (
                "_append_original_signal_outcome",
                "_settle_original_action_result",
                "_settle_original_acquisition_result",
                "_write_original_result",
            ):
                with pytest.raises(ExecutionAuthorityError):
                    await getattr(h.store.db.backend, method)(current)
            assert not entered, "CURRENT OWNER ENTERED THROUGH A LATE-RESULT METHOD"
            with pytest.raises(ExecutionAuthorityError):
                copy.copy(current).check_current()
        assert not quote_peer.requests and not quote_peer.snis
        assert (
            await h.store.get(SCOPE, request_id=h.request.request_id, guard=h.check)
        ).phase == "admitted"


@pytest.mark.asyncio
async def test_quote_original_deadline_at_native_commit_denies_durable_write(
    custody, vendor, quote_peer, monkeypatch
):
    store, _ = custody
    reached = []

    async def wait_expiry(context):
        import asyncio

        await asyncio.sleep(
            max(0, (context.expires_at - datetime.now(UTC)).total_seconds()) + 0.02
        )

    with pytest.raises(_REFUSALS):
        async with quoted_operation(
            custody, vendor, quote_peer, monkeypatch, operation_seconds=2
        ) as h:
            if store.db.backend_type == "postgres":
                import kestrel_sovereign.storage.db.postgres as module

                prove = module._prove_committable

                async def retire_after_actual_probe(connection):
                    await prove(connection)
                    if h.call._state == "quoted":
                        reached.append("postgres-post-committability-probe")
                        await wait_expiry(h.context)

                monkeypatch.setattr(
                    module, "_prove_committable", retire_after_actual_probe
                )
            else:
                conn = store.db.backend._connection
                execute = conn._execute

                async def expire_at_queued_commit(fn, *args, **kwargs):
                    if (
                        fn == conn._conn.commit
                        and h.call._state == "quoted"
                        and conn._conn._current_row_owner is h.call._mutation
                    ):

                        def original_native_commit():
                            reached.append("sqlite-queued-native-commit")
                            time.sleep(
                                max(
                                    0,
                                    (
                                        h.context.expires_at - datetime.now(UTC)
                                    ).total_seconds(),
                                )
                                + 0.02
                            )
                            return fn(*args, **kwargs)

                        return await execute(original_native_commit)
                    return await execute(fn, *args, **kwargs)

                monkeypatch.setattr(conn, "_execute", expire_at_queued_commit)
            await h.call.run_original()
    assert reached and len(quote_peer.requests) == 1
    current = await store.get(SCOPE, request_id="original")
    assert (
        current.phase == "admitted"
        and current.quote is None
        and current.remote_environment is None
    ), "CURRENT QUOTE COMMITTED AFTER ITS ORIGINAL PERMISSION DEADLINE"
    assert not store._original_results and not store.db.backend._original_result_holds


@pytest.mark.asyncio
async def test_quote_bind_refusal_retires_the_original_pristine_native_client(
    custody, vendor, quote_peer, monkeypatch
):
    from kestrel_sovereign.visual_computer.runpod_environment import (
        _OriginalRunpodQuote,
    )

    captured = []
    original_bind = _OriginalRunpodQuote._bind_original_vendor_quote
    with pytest.raises(_REFUSALS):
        async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:

            def retire_at_original_bind(call, control):
                assert call is h.call
                captured.append(control.transport._client)
                h.current["live"] = False
                return original_bind(call, control)

            monkeypatch.setattr(
                _OriginalRunpodQuote,
                "_bind_original_vendor_quote",
                retire_at_original_bind,
            )
            await h.call.run_original()
    assert len(captured) == 1 and captured[0].is_closed
    assert h.call._client_closed and not h.call._worker_entered
    assert not quote_peer.requests and not quote_peer.snis
    store = custody[0]
    assert not store._original_results and not store.db.backend._original_result_holds
    assert (await store.get(SCOPE, request_id="original")).phase == "admitted"


@pytest.mark.asyncio
async def test_quote_plan_restore_uses_original_bytes_without_catalog_or_repricing(
    custody, vendor, quote_peer, monkeypatch
):
    async with quoted_operation(
        custody, vendor, quote_peer, monkeypatch, activate=False
    ) as h:
        async with h.host._activate_quote(h.context, h.record) as call:
            quote = await call.run_original()
            original = await h.store._accepted_runpod_quote(SCOPE, call, quote)
        db = await custody[1]()
        try:
            store = VisualCustodyRepository(db)
            value = await store.get(
                SCOPE, request_id=h.request.request_id, guard=h.check
            )
            runtime = VisualComputerRuntime(
                store,
                h.authority,
                providers={h.request.provider_id: h.provider},
                drivers={},
                check_destination=h.runtime.check_destination,
                policy=h.runtime.policy,
            )
            host = RunpodEnvironmentHost(
                runtime,
                profile=h.plan.profile,
                noncompute=h.plan.noncompute,
                capabilities=h.caps,
                account="owned-native-account",
                api_key=SecretStr("owned-native-api-key"),
                base_url=quote_peer.url,
                ca_data=SecretBytes(quote_peer.ca),
                credential_selector="owned-fixture",
            )
            host._bind_provider(h.provider)
            async with host._activate_quote(h.context, value) as restored:
                plan = restored.restore_original()
                assert (
                    plan.request == h.request
                    and plan.digest() == value.remote_environment.plan_digest
                )
                assert not restored._worker_entered and restored._thread is None
                with pytest.raises(ExecutionAuthorityError):
                    restored.restore_original()
            assert (
                value.remote_environment.private_value()
                == original.remote_environment.private_value()
            )
            assert len(quote_peer.requests) == 1
            host._account = "foreign-account"
            with pytest.raises(ExecutionAuthorityError):
                async with host._activate_quote(h.context, value) as restored:
                    restored.restore_original()
            assert len(quote_peer.requests) == 1
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_quote_prepare_once_ack_preserves_original_plan_and_correlation(
    custody, vendor, quote_peer, monkeypatch
):
    import base64
    import hashlib

    from kestrel_sdk.computer_use import EnvironmentAcquisitionPreparation

    from kestrel_sovereign.visual_computer.runpod_acquisition import (
        prepare_runpod_acquisition_producer,
    )

    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        quote = await h.call.run_original()
        value = await h.store._accepted_runpod_quote(SCOPE, h.call, quote)
        projection = value.remote_environment
        codec = h.host._modules["cpu_environment_codec"]
        plan = codec.deserialize_cpu_environment_plan(
            projection.plan_bytes,
            expected_storage_sha256=projection.plan_storage_sha256,
        )
        dto = plan.pod_request(
            h.request, plan.profile, plan.noncompute, at=plan.planned_at
        )
        prepared = h.control.prepare_cpu_pod(dto, plan=plan)
        body = prepared.exact_body()
        identity = b"PRIVATE-ORIGINAL-QUOTE-WORKER-TLS"
        preparation = EnvironmentAcquisitionPreparation(
            create_payload=SecretBytes(body), worker_identity=SecretBytes(identity)
        )
        async with h.store.original_acquisition_result(
            SCOPE, value, h.request, quote, guard=h.check
        ) as attempt:
            await prepare_runpod_acquisition_producer(
                attempt,
                h.control,
                dto,
                prepared=prepared,
                plan=plan,
                preparation=preparation,
            )
            assert attempt.checkpoint.state == "prepared"
            assert attempt._result._last_ack.entry.kind == "prepare"
            assert (
                attempt._result._revision == 3
            )  # quote, allocate, one prepare+correlation ACK
            private = json.loads(attempt._result._plain)
            assert (
                base64.b64decode(private["acquisition"]["storage"]).count(
                    base64.b64encode(body)
                )
                == 1
            )
            assert base64.b64encode(body).decode() not in json.dumps(
                private["remote_environment"]
            )
        durable = await h.store.get(
            SCOPE, request_id=h.request.request_id, guard=h.check
        )
        correlation = durable.remote_environment.prepared
        assert correlation is not None
        assert (
            correlation.plan_sha256 == prepared.plan_sha256.get_secret_value()
            and correlation.binding == prepared.binding.get_secret_value()
        )
        assert correlation.create_sha256 == hashlib.sha256(body).hexdigest()
        assert (
            correlation.worker_identity_sha256 == hashlib.sha256(identity).hexdigest()
        )
        expected = projection.private_value()
        expected["prepared"] = durable.remote_environment.private_value()["prepared"]
        assert durable.remote_environment.private_value() == expected
        db = await custody[1]()
        try:
            restored = await VisualCustodyRepository(db).get(
                SCOPE, request_id=h.request.request_id, guard=h.check
            )
            assert restored.remote_environment.private_value() == expected
            from kestrel_sdk.computer_use import EnvironmentAcquisitionStorage

            assert (
                EnvironmentAcquisitionStorage.deserialize_private(
                    restored.acquisition.storage
                ).preparation
                == preparation
            )
        finally:
            await db.close()
        assert len(quote_peer.requests) == 1  # never a create POST


class _BoundedQuoteProvider:
    """Test-only complete SDK negotiation, deliberately stops before create."""

    native_environment_host_api_version = 1

    def __init__(self, host):
        self.environment_host = host
        self.acquisition_entries = []
        self.acquisition_attempts = 0
        self.uncustodied_creates = 0

    async def capabilities(self):
        return self.environment_host._capabilities

    async def quote(self, request):
        async with self.environment_host.quote_operation(request) as call:
            return await call.run_original()

    async def acquire_with_custody(self, request, quote, custody):
        self.acquisition_attempts += 1
        from kestrel_sdk.resources import ResourceUnsupported

        row = await custody._repository.get(
            request.scope, request_id=request.request_id, guard=custody._guard
        )
        assert (
            row.quote == quote
            and row.remote_environment is not None
            and row.phase == "quoted"
        )
        self.acquisition_entries.append(row)
        raise ResourceUnsupported("C1q fixture stops before paid create")

    async def acquire(self, *args):
        self.uncustodied_creates += 1
        raise AssertionError("uncustodied paid create forbidden")

    async def inspect(self, *args):
        raise AssertionError("C1q has no remote inspect")

    async def connect(self, *args):
        raise AssertionError("C1q has no remote connect")

    async def release(self, *args):
        raise AssertionError("C1q has no remote release")

    async def renew(self, *args):
        raise AssertionError("C1q has fixed expiry")


@pytest.mark.asyncio
async def test_runtime_uses_quote_native_ack_and_never_generic_presave(
    custody, vendor, quote_peer, monkeypatch
):
    from kestrel_sdk.resources import ResourceUnsupported

    async with quoted_operation(
        custody, vendor, quote_peer, monkeypatch, activate=False
    ) as h:
        provider = _BoundedQuoteProvider(h.host)
        h.host._provider = None
        h.runtime.providers[h.request.provider_id] = provider
        h.host._bind_provider(provider)
        with pytest.raises(ResourceUnsupported, match="stops before paid create"):
            await h.runtime.admit(h.context, h.request)
        assert (
            len(provider.acquisition_entries) == 1 and provider.uncustodied_creates == 0
        )
        value = await h.store.get(SCOPE, request_id=h.request.request_id, guard=h.check)
        assert (
            value.phase == "quoted"
            and value.quote is not None
            and value.remote_environment is not None
        )
        # The original allocation ACK now precedes provider exposure, including
        # providers that never enter the SDK operation. Stale generic retirement
        # cannot overwrite that original, still-unprepared checkpoint.
        from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionStorage
        saved = EnvironmentAcquisitionStorage.deserialize_private(value.acquisition.storage)
        assert saved.checkpoint.state == 'unprepared' and saved.preparation is None
        assert value.acquisition.binding is None and value.lease is None
        assert value.revision == provider.acquisition_entries[0].revision
        assert not h.store._original_results and not h.store.db.backend._original_result_holds
        assert len(quote_peer.requests) == 1


@pytest.mark.asyncio
async def test_recovered_native_quote_refuses_missing_custody_no_fallback(
    custody, vendor, quote_peer, monkeypatch
):
    from kestrel_sdk.resources import ResourceUnsupported

    async with quoted_operation(
        custody, vendor, quote_peer, monkeypatch, activate=False
    ) as h:
        async with h.host._activate_quote(h.context, h.record) as call:
            quote = await call.run_original()
            await h.store._accepted_runpod_quote(SCOPE, call, quote)
        provider = _BoundedQuoteProvider(h.host)
        provider.native_environment_host_api_version = None
        provider.environment_host = None
        h.runtime.providers[h.request.provider_id] = provider
        with pytest.raises(ResourceUnsupported, match="fallback forbidden"):
            await h.runtime.admit(h.context, h.request)
        assert provider.uncustodied_creates == 0 and not provider.acquisition_entries
        assert len(quote_peer.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door", ["copied", "foreign-task", "host-retired", "turn-retired", "config-changed"]
)
async def test_quote_current_admission_negatives_before_native_io(
    custody, vendor, quote_peer, monkeypatch, door
):
    import asyncio

    with pytest.raises(_REFUSALS):
        async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
            if door == "copied":
                await copy.copy(h.call).run_original()
            elif door == "foreign-task":
                await asyncio.create_task(h.call.run_original())
            elif door == "host-retired":
                h.current["live"] = False
                await h.call.run_original()
            elif door == "turn-retired":
                h.turn.revoke("original turn retired without Task.cancel")
                await h.call.run_original()
            elif door == "config-changed":
                h.host._capabilities = h.caps.model_copy(
                    update={"operations": frozenset({"screenshot"})}
                )
                await h.call.run_original()
            else:
                pytest.fail("unknown door")
    assert not quote_peer.requests and not quote_peer.snis
    assert (
        not custody[0]._original_results
        and not custody[0].db.backend._original_result_holds
    )
    value = await custody[0].get(SCOPE, request_id="original")
    assert value.phase == "admitted" and value.remote_environment is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire", ["gzip", "duplicate-identity", "oversize", "chunked", "exact", "identity"]
)
async def test_quote_raw_bounds_before_canonical_decoder(
    custody, vendor, quote_peer, monkeypatch, wire
):
    import threading

    import httpx

    calls = []
    code = httpx.Response.json.__code__
    decoder = httpx.Response._get_content_decoder.__code__
    decoders = []

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is code:
            calls.append(frame.f_code)
        if event == "call" and frame.f_code is decoder:
            decoders.append(frame.f_code)

    previous = threading.getprofile()
    threading.setprofile(observe)
    try:
        async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
            raw = json.dumps(CATALOG).encode()
            quote_peer.raw = raw + b" " * (
                1024 * 1024 - len(raw) + (wire in {"oversize", "chunked"})
            )
            if wire == "gzip":
                quote_peer.encoding = ["gzip"]
            elif wire == "duplicate-identity":
                quote_peer.encoding = ["identity", "identity"]
            elif wire == "identity":
                quote_peer.encoding = ["identity"]
            elif wire == "chunked":
                quote_peer.chunked = True
            if wire in {"exact", "identity"}:
                await h.call.run_original()
                assert calls and decoders and h.call._raw_bytes == 1024 * 1024
            else:
                with pytest.raises(_REFUSALS):
                    await h.call.run_original()
                assert not calls, "CANONICAL JSON ENTERED BEFORE RAW RESPONSE BOUNDS"
                if wire != "chunked":
                    assert not decoders, (
                        "CONTENT DECODER ALLOCATED BEFORE RAW HEADER BOUNDS"
                    )
                assert h.call._raw_bytes <= 1024 * 1024
            assert (
                len(quote_peer.requests) == 1
                and h.call._worker_exited
                and h.call._client_closed
            )
            assert not h.call._thread.sockets and not h.call._thread.requests
            current = await h.store.get(
                SCOPE, request_id=h.request.request_id, guard=h.check
            )
            assert current.phase == (
                "quoted" if wire in {"exact", "identity"} else "admitted"
            )
    finally:
        threading.setprofile(previous)


@pytest.mark.asyncio
async def test_quote_stale_row_cas_refuses_after_actual_catalog(
    custody, vendor, quote_peer, monkeypatch
):
    import asyncio
    import threading

    ready = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    other = await custody[1]()
    changed = []
    try:
        async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:

            def await_independent_revision():
                loop.call_soon_threadsafe(ready.set)
                if not release.wait(5):
                    raise TimeoutError("independent original-row revision not joined")

            quote_peer.before_reply = await_independent_revision

            async def independent_writer():
                await asyncio.wait_for(ready.wait(), 5)
                try:
                    await other.execute(
                        "UPDATE visual_computer_requests SET revision=revision+1"
                    )
                    changed.append(True)
                finally:
                    release.set()

            writer = asyncio.create_task(independent_writer())
            try:
                with pytest.raises(_REFUSALS):
                    await h.call.run_original()
                await writer
            finally:
                release.set()
                await asyncio.gather(writer, return_exceptions=True)
            assert changed and len(quote_peer.requests) == 1
            row = await h.store.db.fetchone(
                "SELECT payload, revision FROM visual_computer_requests"
            )
            assert row[1] == 1
            value = h.store._open(
                h.store._owner(SCOPE), h.request.request_id, row[0], CustodyRecord
            )
            assert value.phase == "admitted" and value.quote is None
    finally:
        release.set()
        await other.close()


@pytest.mark.asyncio
async def test_quote_stop_drains_original_native_call_without_state_rewrite(
    custody, vendor, quote_peer, monkeypatch
):
    import asyncio
    import threading

    ready = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    drained = []
    with pytest.raises(_REFUSALS):
        async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:

            def hold_response():
                loop.call_soon_threadsafe(ready.set)
                if not release.wait(5):
                    raise TimeoutError("quote stop not joined")

            quote_peer.before_reply = hold_response

            async def stop_original():
                await asyncio.wait_for(ready.wait(), 5)
                h.call.revoke()
                release.set()
                await h.runtime.egress.revoke_session(SCOPE, h.record.session_id)
                drained.append(True)

            stop = asyncio.create_task(stop_original())
            try:
                await h.call.run_original()
            finally:
                release.set()
                await asyncio.gather(stop)
    assert drained and len(quote_peer.requests) == 1
    assert h.call._worker_exited and h.call._client_closed
    assert not h.call._thread.sockets and not h.call._thread.requests
    assert (
        not h.runtime.egress._calls
        and not h.store._original_results
        and not h.store.db.backend._original_result_holds
    )
    row = await h.store.get(SCOPE, request_id="original")
    assert row.phase == "admitted" and row.remote_environment is None


@pytest.mark.asyncio
async def test_quote_lost_original_ack_preserves_committed_facts_without_replay(
    custody, vendor, quote_peer, monkeypatch
):
    from kestrel_sovereign.execution_custody import execution_commit_outcome

    async with quoted_operation(
        custody, vendor, quote_peer, monkeypatch, activate=False
    ) as h:
        backend = h.store.db.backend
        commit = backend._commit_current_runpod_quote

        async def lose_original_ack(owner):
            await commit(owner)
            raise OSError("controlled original quote ACK publication loss")

        with monkeypatch.context() as patch:
            patch.setattr(backend, "_commit_current_runpod_quote", lose_original_ack)
            async with h.host._activate_quote(h.context, h.record) as call:
                with pytest.raises(_REFUSALS) as lost:
                    await call.run_original()
                assert execution_commit_outcome(lost.value) == "committed"
                assert call._ack is None
                with pytest.raises(ExecutionAuthorityError):
                    await call.run_original()
        row = await h.store.get(SCOPE, request_id=h.request.request_id, guard=h.check)
        assert row.phase == "quoted" and row.remote_environment is not None
        async with h.host._activate_quote(h.context, row) as restored:
            assert restored.restore_original().request == h.request
        assert len(quote_peer.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["turn", "host"])
async def test_quote_seventeen_actual_tls_bytes_stop_without_commit(
    custody, vendor, quote_peer, monkeypatch, retirement
):
    import ssl

    from kestrel_sovereign.execution_custody import is_execution_control_error

    native_send = ssl.SSLSocket.send
    written = []
    sockets = []
    with pytest.raises(_REFUSALS) as denied:
        async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:

            def partial(sock, data, flags=0):
                if bytes(data[:4]) == b"GET " and not sockets:
                    sockets.append(sock)
                    count = native_send(sock, data[:17], flags)
                    written.append(bytes(data[:count]))
                    if retirement == "turn":
                        h.turn.revoke("original Stop after17 TLS application bytes")
                    else:
                        h.current["live"] = False
                    return count
                count = native_send(sock, data, flags)
                if sockets and sock is sockets[0]:
                    written.append(bytes(data[:count]))
                return count

            monkeypatch.setattr(ssl.SSLSocket, "send", partial)
            await h.call.run_original()
    if retirement == "turn":
        assert is_execution_control_error(denied.value)
    assert len(written) == 1 and len(written[0]) == 17
    assert h.call._thread.sent and h.call._worker_exited and h.call._client_closed
    assert not h.call._thread.sockets and not h.call._thread.requests
    assert not quote_peer.requests
    row = await custody[0].get(SCOPE, request_id="original")
    assert row.phase == "admitted" and row.remote_environment is None


@pytest.mark.asyncio
async def test_runtime_quote_host_retirement_does_not_rewrite_current_only_row(
    custody, vendor, quote_peer, monkeypatch
):
    with pytest.raises(_REFUSALS):
        async with quoted_operation(
            custody, vendor, quote_peer, monkeypatch, activate=False
        ) as h:
            provider = _BoundedQuoteProvider(h.host)
            h.host._provider = None
            h.runtime.providers[h.request.provider_id] = provider
            h.host._bind_provider(provider)
            quote_peer.before_reply = lambda: h.current.update(live=False)
            await h.runtime.admit(h.context, h.request)
    assert (
        len(quote_peer.requests) == 1
        and not provider.acquisition_entries
        and provider.uncustodied_creates == 0
    )
    value = await custody[0].get(SCOPE, request_id="original")
    assert (
        value.phase == "admitted"
        and value.revision == 0
        and value.remote_environment is None
    ), "CURRENT-ONLY QUOTE FAILURE RETIRED THE ROW AFTER ORIGINAL HOST WITHDRAWAL"


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["account", "credential"])
async def test_quoted_original_native_binding_required_before_prepare_ack(
    custody, vendor, quote_peer, monkeypatch, changed
):
    from kestrel_sdk.computer_use import EnvironmentAcquisitionPreparation
    from kestrel_sdk.resources import ResourceUnsupported

    from kestrel_sovereign.visual_computer.runpod_acquisition import (
        prepare_runpod_acquisition_producer,
    )

    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        quote = await h.call.run_original()
        value = await h.store._accepted_runpod_quote(SCOPE, h.call, quote)
        envelope = value.remote_environment
        plan = h.host._modules[
            "cpu_environment_codec"
        ].deserialize_cpu_environment_plan(
            envelope.plan_bytes, expected_storage_sha256=envelope.plan_storage_sha256
        )
        dto = plan.pod_request(
            h.request, plan.profile, plan.noncompute, at=plan.planned_at
        )
        foreign = vendor.clients.RunpodControlPlaneClient(
            api_key="foreign-key"
            if changed == "credential"
            else "owned-native-api-key",
            base_url=quote_peer.url,
            prepared_cpu_account="foreign-account"
            if changed == "account"
            else "owned-native-account",
            prepared_cpu_ca_data=quote_peer.ca,
            prepared_cpu_acquisition=True,
            connect_timeout=2,
            read_timeout=4,
        )
        try:
            prepared = foreign.prepare_cpu_pod(dto, plan=plan)
            preparation = EnvironmentAcquisitionPreparation(
                create_payload=SecretBytes(prepared.exact_body()),
                worker_identity=SecretBytes(b"PRIVATE-ORIGINAL-WORKER"),
            )
            async with h.store.original_acquisition_result(
                SCOPE, value, h.request, quote, guard=h.check
            ) as attempt:
                with pytest.raises(ResourceUnsupported, match="binding differs"):
                    await prepare_runpod_acquisition_producer(
                        attempt,
                        foreign,
                        dto,
                        prepared=prepared,
                        plan=plan,
                        preparation=preparation,
                    )
                assert attempt.checkpoint.state == "unprepared"
            assert foreign.transport._client.is_closed is False
        finally:
            foreign.close()
        current = await h.store.get(
            SCOPE, request_id=h.request.request_id, guard=h.check
        )
        assert current.remote_environment.prepared is None
        assert len(quote_peer.requests) == 1


@pytest.mark.asyncio
async def test_native_quote_profile_never_retrofits_old_plain_sdk_quote(
    custody, vendor, quote_peer, monkeypatch
):
    from kestrel_sdk.resources import ResourceUnsupported

    from tests.integration.test_visual_runpod_acquisition import original_quote

    async with quoted_operation(
        custody, vendor, quote_peer, monkeypatch, activate=False
    ) as h:
        h.record.quote = original_quote(h.plan)
        h.record.phase = "quoted"
        await h.store.save(SCOPE, h.record, h.check)
        provider = _BoundedQuoteProvider(h.host)
        h.host._provider = None
        h.runtime.providers[h.request.provider_id] = provider
        h.host._bind_provider(provider)
        with pytest.raises(ResourceUnsupported, match="private.*quote.*absent"):
            await h.runtime.admit(h.context, h.request)
        assert provider.acquisition_attempts == 0 and provider.uncustodied_creates == 0
        assert not quote_peer.requests and not quote_peer.snis
