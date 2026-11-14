"""Original local two-domain setup and a pinned, finite browser transport.

Only Runtime.admit issues setup, inside its existing ADMIT operation. A setup
record is not a session or a transferable grant. Successful setup leaves its
original physical closer in the driver's existing connection table; it leaves
no continuing setup authority. No remote provider, Human, relay or files API is
negotiated by this first profile.
"""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import timedelta
from decimal import Decimal
import hashlib
import hmac
import json
import os
from pathlib import Path
import ssl
import tempfile
from collections.abc import Mapping
from uuid import uuid4

from pydantic import SecretBytes, SecretStr, ValidationError
from kestrel_sdk.computer_use import (
    ActionReceipt, ComputerSession, DriverConnection, EnvironmentCapabilities,
    EnvironmentLease, EnvironmentQuote, ScreenFrame, ScreenObservation,
)
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import (
    ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceUnsupported,
)
from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, await_owned_operation, await_owned_task, create_owned_task, raise_owned_outcome,
)
from kestrel_sovereign.execution_custody import execution_terminal_error
from .docker import DockerBoundary, DockerCommandError, _identity, _json, _now, _record
from .dispatch import native_dispatch, native_trace
from .worker_profile import (
    CONTRACT, DRIVER, LABEL, MAX_REQUEST, OPERATIONS, PRIVATE,
    PROFILE, PROVIDER, TOKEN, WORKER_RETIRE_WAIT_SECONDS, LocalWorkerProfile, _names, container_arguments,
    verify_container,
)

_ISSUER = object()


class _OriginalLocalWorkerAdmission:
    """Lexical representation of the SAME original Runtime ADMIT, no DTO grant."""
    def __init__(self, runtime, context, record, check, *, _issuer):
        from .runtime import VisualComputerRuntime, _LOCAL_ADMIT_ISSUER
        if type(self) is not _OriginalLocalWorkerAdmission or type(runtime) is not VisualComputerRuntime or _issuer is not _LOCAL_ADMIT_ISSUER:
            raise OperatorAuthorizationError("original Runtime ADMIT issuer required")
        self._runtime, self._context, self._record, self._check = runtime, context, record, check
        self._owner_task, self._loop = asyncio.current_task(), asyncio.get_running_loop()
        self._original = self
        self._binding = runtime.authority._operations.get(self._owner_task)
        self._active, self._consumed = True, False
        self._facts = (runtime, context, record, check, self._owner_task, self._loop, self._binding)
        self.require_original()

    def require_original(self):
        if (type(self) is not _OriginalLocalWorkerAdmission or self._original is not self
                or not self._active or asyncio.current_task() is not self._owner_task
                or asyncio.get_running_loop() is not self._loop
                or any(a is not b for a, b in zip(self._facts,
                    (self._runtime, self._context, self._record, self._check, self._owner_task, self._loop, self._binding), strict=True))
                or self._binding is None or self._binding.context is not self._context
                or not self._binding.active or self._runtime.authority._operations.get(self._owner_task) is not self._binding):
            raise OperatorAuthorizationError("original Runtime ADMIT lifetime differs")
        self._check()


def _program_arguments(role, flag):
    if (role, flag) not in {("browser", "--initialize"), ("broker", "--initialize")}:
        raise ResourceUnsupported("fixed_worker_bootstrap_program_required")
    return ["python", "-B", "-m", "kestrel_sovereign.visual_computer.worker_entry", role, flag]


def _tls_material(expiry):
    """One original CA/client and two role keys; private temporary loader files."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    now = _now()
    ca_key = Ed25519PrivateKey.generate()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "original-local-worker")])
    def certificate(key, common, *, server=False, ca=False):
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common)])
        builder = (x509.CertificateBuilder().subject_name(subject).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now-timedelta(seconds=5)).not_valid_after(expiry)
            .add_extension(x509.BasicConstraints(ca=ca, path_length=0 if ca else None), critical=True))
        if not ca:
            builder = builder.add_extension(x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH if server else ExtendedKeyUsageOID.CLIENT_AUTH]), critical=True)
            if server:
                builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName(common)]), critical=False)
        return builder.sign(ca_key, algorithm=None)
    def private(key):
        return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption()).decode()
    ca = certificate(ca_key, "original-local-worker", ca=True)
    ca_pem = ca.public_bytes(serialization.Encoding.PEM).decode()
    client_key = Ed25519PrivateKey.generate()
    client = certificate(client_key, "original-host-client")
    roles, contexts = {}, {}
    for role in ("normal", "priority"):
        key = Ed25519PrivateKey.generate()
        server_name = role+".worker.kestrel.invalid"
        cert = certificate(key, server_name, server=True)
        roles[role] = {"certificate": cert.public_bytes(serialization.Encoding.PEM).decode(),
            "key": private(key), "sha256": hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()}
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_3
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(cadata=ca_pem)
        with tempfile.TemporaryDirectory(prefix="original-worker-client-") as directory:
            paths = []
            for label, data in (("certificate", client.public_bytes(serialization.Encoding.PEM)),
                                ("key", private(client_key).encode())):
                path = Path(directory)/label
                with open(path, "xb", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
                    stream.write(data)
                paths.append(str(path))
            context.load_cert_chain(*paths)
        contexts[role] = context
    return ca_pem, hashlib.sha256(client.public_bytes(serialization.Encoding.DER)).hexdigest(), roles, contexts


def _transport_types():
    # Host-only optional dependency. Nothing imports this package in either
    # image; absence cannot negotiate a weaker transport or a ready session.
    try:
        from kestrel_cloud_runpod.remote_worker_tls import PinnedWorkerTarget, PinnedWorkerTLS
    except ImportError:
        raise ResourceUnsupported("canonical_pinned_worker_transport_unavailable") from None
    return PinnedWorkerTarget, PinnedWorkerTLS


def _original_loopback_endpoints(record, cid, native_instance):
    """Descriptive original daemon facts only; no IO or session authority."""
    failure = "original_loopback_role_endpoint_required"
    if (type(record) is not dict or type(cid) is not str or TOKEN.fullmatch(cid) is None
            or record.get("Id") != cid or type(native_instance) is not tuple
            or len(native_instance) != 2 or type(native_instance[0]) is not int
            or native_instance[0] <= 0 or type(native_instance[1]) is not str or not native_instance[1]):
        raise ResourceUnsupported(failure)
    state, settings = record.get("State"), record.get("NetworkSettings")
    if (type(state) is not dict or state.get("Running") is not True
            or type(state.get("Pid")) is not int or state["Pid"] <= 0
            or (state["Pid"], state.get("StartedAt")) != native_instance
            or type(settings) is not dict):
        raise ResourceUnsupported(failure)
    ports = settings.get("Ports")
    if type(ports) is not dict or set(ports) != {"41401/tcp", "41402/tcp"}:
        raise ResourceUnsupported(failure)
    result = []
    for port in ("41401/tcp", "41402/tcp"):
        bindings = ports[port]
        if (type(bindings) is not list or len(bindings) != 1 or type(bindings[0]) is not dict
                or set(bindings[0]) != {"HostIp", "HostPort"} or bindings[0]["HostIp"] != "127.0.0.1"):
            raise ResourceUnsupported(failure)
        value = bindings[0]["HostPort"]
        if (type(value) is not str or not 1 <= len(value) <= 5 or value[0] == "0"
                or not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 65535):
            raise ResourceUnsupported(failure)
        result.append(int(value))
    if result[0] == result[1]:
        raise ResourceUnsupported(failure)
    return tuple(result)


def _verify_broker_network(network, native_id, identity, members):
    """Closed v2 facts measured on the supported daemon, not an egress firewall."""
    if (type(network) is not dict or type(native_id) is not str or TOKEN.fullmatch(native_id) is None
            or network.get("Id") != native_id or network.get("Name") != _names(identity)["network"]
            or network.get("Labels") != {LABEL: identity} or network.get("Driver") != "bridge"
            or network.get("Scope") != "local" or network.get("EnableIPv4") is not True
            or any(network.get(key) is not False
                   for key in ("Internal", "EnableIPv6", "Attachable", "Ingress", "ConfigOnly"))
            or network.get("ConfigFrom") != {"Network": ""}
            or network.get("Options") != {"com.docker.network.enable_ipv4": "true"}
            or type(network.get("Containers")) is not dict or set(network["Containers"]) != set(members)):
        raise ResourceUnsupported("original_broker_network_changed")


class _OriginalLocalWorkerSetup:
    def __init__(self, host, admission, *, _issuer):
        if type(self) is not _OriginalLocalWorkerSetup or type(host) is not LocalWorkerSetupHost or _issuer is not _ISSUER:
            raise OperatorAuthorizationError("original local worker issuer required")
        self._admission = admission
        self._host, self._context, self._record, self._check = host, admission._context, admission._record, admission._check
        self._owner_task, self._loop = asyncio.current_task(), asyncio.get_running_loop()
        self._binding = self
        self._active, self._used = True, False
        self._finished = self._loop.create_future()
        self._original_finished = self._finished
        self._resources = None
        self._handed_off = False
        self._facts = (host, self._context, self._record, self._check, self._owner_task, self._loop, admission)
        self._request, self._quote = self._record.request, self._record.quote
        scope = self._record.request.scope
        self._key = (scope.tenant_id, scope.principal_id, scope.agent_id, self._record.session_id,
                     None, None, None, self._record.request.request_id, "local-worker-setup")
        self._original_key = self._key

    def _require_cleanup_original(self):
        if (type(self) is not _OriginalLocalWorkerSetup or self._binding is not self
                or self._finished is not self._original_finished or self._key != self._original_key
                or any(a is not b for a, b in zip(self._facts,
                    (self._host, self._context, self._record, self._check, self._owner_task, self._loop, self._admission), strict=True))):
            raise OperatorAuthorizationError("original local setup cleanup backing differs")

    def _check_original(self):
        self._require_cleanup_original()
        runtime = self._host._runtime
        if (type(self) is not _OriginalLocalWorkerSetup or self._binding is not self
                or not self._active or self._loop is not asyncio.get_running_loop()
                or self._finished is not self._original_finished
                or not self._admission._active
                or self._record.request is not self._request or self._record.quote is not self._quote
                or runtime.authority._operations.get(self._owner_task) is not self._admission._binding
                or any(a is not b for a, b in zip(self._facts,
                    (self._host, self._context, self._record, self._check, self._owner_task, self._loop, self._admission), strict=True))
                or runtime.egress._calls.get(self._key) is not self):
            raise OperatorAuthorizationError("original local worker setup retired")
        self._check()
        self._host._require_configuration()
        runtime._request(self._context, self._record.request)

    def _require_origin(self):
        if asyncio.current_task() is not self._owner_task:
            raise OperatorAuthorizationError("original local setup Task differs")
        self._check_original()

    def _require_retirement(self, record, *, failed):
        self._require_cleanup_original()
        self._host._require_configuration()
        if (self._loop is not asyncio.get_running_loop() or self._active or not self._used
                or record.request != self._request or record.quote != self._quote
                or record.session_id != self._record.session_id
                or record.remote_environment is not None or record.acquisition is not None):
            raise ResourceConflict("original local setup retirement custody differs")
        if failed:
            self._host._runtime.egress._require_environment_call(self)
            if self._handed_off:
                raise ResourceConflict("handed-off setup is not failed setup custody")
        # Additional same-object checks AFTER actual original awaits. Neither
        # completion nor a closed flag substitutes for those original joins.
        if self._finished.result() is not None:
            raise ResourceOutcomeUnknown("original setup completion differs")
        resources = self._resources
        if resources is None:
            if record.phase not in {"admitted", "quoted", "retired"} or record.lease is not None:
                raise ResourceOutcomeUnknown("original setup physical closer absent")
            return
        admission, task = resources._close_admission, resources._close_admission.task
        if (resources.original is not self or resources._binding is not resources
                or admission is not resources._original_close_admission or task is None
                or getattr(task, "_kestrel_owned_admission", None) is not admission
                or resources._closed is not True or resources._closing is not True
                or task.result() is not None):
            raise ResourceOutcomeUnknown("original setup physical retirement unconfirmed")
        if record.lease is not None and (
                record.lease.lease_id != "local-worker-" + resources.identity
                or record.lease.provider_resource_id != resources.containers.get("browser")):
            raise ResourceConflict("original retired local lease differs")
        if record.connection is not None and record.connection != resources._connection:
            raise ResourceConflict("original retired local connection differs")

    async def _join_retirement(self, record, *, failed):
        self._require_cleanup_original()
        outcome = await await_owned_task(self._finished)
        raise_owned_outcome(outcome, operation="original setup completion", retain_originals=True)
        if self._resources is not None:
            resources = self._resources
            admission = resources._close_admission
            task = admission.task
            if (resources.original is not self or resources._binding is not resources
                    or admission is not resources._original_close_admission or task is None
                    or getattr(task, "_kestrel_owned_admission", None) is not admission):
                raise ResourceOutcomeUnknown("original setup physical closer absent")
            outcome = await await_owned_task(task)
            raise_owned_outcome(outcome, operation="original setup physical retirement", retain_originals=True)
        self._require_retirement(record, failed=failed)

    def revoke(self):
        # Close write admission. The external closer waits the original setup
        # Future; it never cancels native cleanup a second time.
        self._active = False

    async def run_original(self):
        self._require_origin()
        if self._used:
            raise OperatorAuthorizationError("original local setup is one-shot")
        self._used = True
        failure = None
        try:
            runtime = self._host._runtime
            remaining = min(runtime.policy.operation_seconds, (self._context.expires_at-runtime._now()).total_seconds())
            async with asyncio.timeout(remaining):
                self._check_original()
                await self._host._create_original_resources(self)
                self._check_original()
                session = await runtime._acquire_open(self._context, self._record, self._check_original)
                self._check_original()
                self._handed_off = True
                return session
        except BaseException as error:
            failure = error
            raise
        finally:
            cleanup_failure = None
            try:
                if not self._handed_off and self._resources is not None:
                    await self._resources.close()
            except BaseException as cleanup:
                cleanup_failure = cleanup
                try:
                    if (isinstance(cleanup, asyncio.CancelledError)
                            and self._resources._binding is self._resources
                            and self._resources._close_admission is self._resources._original_close_admission
                            and self._resources._close_admission.task is not None
                            and getattr(self._resources._close_admission.task, "_kestrel_owned_admission", None)
                                is self._resources._close_admission):
                        # Canonical close has physically joined this SAME original
                        # retirement Task before propagating its caller withdrawal.
                        # The setup Future describes retirement, not that caller.
                        from kestrel_sovereign._async_ownership import owned_task_error
                        cleanup_failure = owned_task_error(self._resources._close_admission.task)
                except BaseException as observation:
                    cleanup_failure = execution_terminal_error(cleanup, observation, retain_originals=True)
                    raise execution_terminal_error(failure, cleanup_failure, retain_originals=True)
                raise execution_terminal_error(failure, cleanup, retain_originals=True)
            finally:
                # Future means this fixed setup program AND its original
                # native cleanup have finished; not that the browser exited.
                if not self._finished.done():
                    if cleanup_failure is None:
                        self._finished.set_result(None)
                    else:
                        self._finished.set_exception(cleanup_failure)
                        self._finished.exception()  # Retrieve; external joins still raise the same failure.


class LocalWorkerSetupHost:
    """Trusted factory binding, inert outside the original Runtime ADMIT."""
    def __init__(self, runtime, provider, driver):
        from .runtime import VisualComputerRuntime
        if (type(runtime) is not VisualComputerRuntime or type(provider) is not LocalTwoDomainProvider
                or type(driver) is not LocalTwoDomainDriver or provider.boundary.socket != driver.boundary.socket):
            raise ResourceUnsupported("exact_local_two_domain_adapters_required")
        self._runtime, self._provider, self._driver = runtime, provider, driver
        self._original = (runtime, provider, driver, provider.profile)
        self._configuration = (provider.profile, provider.profile.digest(), provider.boundary.socket,
                               driver.boundary.socket, provider._key.get_secret_value())

    def _require_configuration(self):
        provider, driver, runtime = self._provider, self._driver, self._runtime
        actual = (provider.profile, provider.profile.digest(), provider.boundary.socket,
                  driver.boundary.socket, provider._key.get_secret_value())
        if (any(a is not b for a, b in zip(self._original, (runtime, provider, driver, provider.profile), strict=True))
                or actual != self._configuration or runtime.providers.get(PROVIDER) is not provider
                or runtime.drivers.get(DRIVER) is not driver or provider._setup_host is not self
                or driver._setup_host is not self):
            raise OperatorAuthorizationError("original local worker configuration changed")

    def _issue_local_setup(self, admission):
        self._require_configuration()
        if type(admission) is not _OriginalLocalWorkerAdmission:
            raise OperatorAuthorizationError("exact original Runtime ADMIT required")
        admission.require_original()
        context, record = admission._context, admission._record
        if (admission._runtime is not self._runtime or admission._consumed
                or record.request.scope != self._runtime.scope(context) or record.session is not None
                or record.request.provider_id != PROVIDER or record.request.profile_id != PROFILE):
            raise OperatorAuthorizationError("original pre-session Host ADMIT required")
        admission._consumed = True
        return _OriginalLocalWorkerSetup(self, admission, _issuer=_ISSUER)

    def _current_setup(self):
        # The existing finite egress map is the sole setup registry. A private
        # adapter cannot manufacture another ComputerSession to borrow it.
        calls = [call for call in self._runtime.egress._calls.values()
                 if type(call) is _OriginalLocalWorkerSetup and call._host is self
                 and call._owner_task is asyncio.current_task()
                 and call._active and call._admission._active
                 and call._loop is asyncio.get_running_loop()
                 and self._runtime.authority._operations.get(call._owner_task) is call._admission._binding]
        if len(calls) != 1:
            raise OperatorAuthorizationError("one original local setup required")
        calls[0]._require_origin()
        return calls[0]

    def _retained_setup(self, record):
        calls = [call for call in self._runtime.egress._calls.values()
                 if type(call) is _OriginalLocalWorkerSetup and call._host is self
                 and call._record.session_id == record.session_id
                 and call._request.request_id == record.request.request_id
                 and call._request.scope == record.request.scope]
        if len(calls) > 1:
            raise ResourceConflict("original local setup cleanup custody is ambiguous")
        return calls[0] if calls else None

    def _discard_retired_setup(self, original, record):
        original._require_retirement(record, failed=True)
        if record.phase not in {"retired", "released"}:
            raise ResourceConflict("original setup row has no durable retirement")
        resources = original._resources
        retained = self._driver._connections.get(record.session_id)
        if retained is not None and retained is not resources:
            raise ResourceConflict("original local connection cleanup custody differs")
        if retained is resources and resources is not None:
            self._driver._connections.pop(record.session_id)
        self._runtime.egress._calls.pop(original._key)

    async def _discard_released_connection(self, record):
        resources = self._driver._connections.get(record.session_id)
        if resources is None:
            return
        if record.phase != "released" or record.lease is None or record.lease.state != "released":
            raise ResourceConflict("original local connection has no durable release")
        await resources.original._join_retirement(record, failed=False)
        if self._driver._connections.get(record.session_id) is not resources:
            raise ResourceConflict("original local connection cleanup custody changed")
        self._driver._connections.pop(record.session_id)

    async def _create_original_resources(self, original):
        original._require_origin()
        resources = _LocalWorkerResources(original)
        # Retain the original physical closer BEFORE the first fallible native
        # constructor/publication. An exception cannot hide an accepted child.
        original._resources = resources
        await resources.create()


def bind_local_worker_hosts(runtime):
    provider, driver = runtime.providers.get(PROVIDER), runtime.drivers.get(DRIVER)
    # The driver key is shared with nonlocal adapters. Only a selected local
    # adapter requires this original local provider/driver pair.
    if provider is None and not isinstance(driver, LocalTwoDomainDriver):
        return
    if type(provider) is not LocalTwoDomainProvider or type(driver) is not LocalTwoDomainDriver:
        raise ResourceUnsupported("complete_explicit_local_two_domain_profile_required")
    if provider._setup_host is not None or driver._setup_host is not None:
        raise ResourceConflict("local_worker_adapters_already_bound")
    host = LocalWorkerSetupHost(runtime, provider, driver)
    provider._setup_host = driver._setup_host = host
    if driver.egress_guard is None:
        driver.egress_guard = runtime.egress.operation


class _LocalWorkerResources:
    """Original exact resource handles; closing grants no work or restart."""
    def __init__(self, original):
        self.original, self.boundary = original, original._host._provider.boundary
        self.profile = original._host._provider.profile
        self.identity = _identity(original._record.request.scope, original._record.request.request_id)
        self.names = _names(self.identity)
        self.containers, self.images, self.incarnations = {}, {}, {}
        self.volume = self.network = None
        self._closed = False
        self._closing = False
        self._tls = None
        self._connection = None
        self._native_instances = {}
        self._container_waits = {}
        self._volume_identity = None
        self._network_identity = None
        self._pending_creates = set()
        self._broker_endpoints = None
        self._mount_sources = {}
        self._close_admission = OwnedTaskAdmission()
        self._original_close_admission, self._binding = self._close_admission, self

    async def _command(self, args, *, data=None):
        self.original._check_original()
        return await self.boundary.invoke(args, data=data, current=self.original._check_original)

    async def _inspect(self, role, *, current=None):
        cid = self.containers[role]
        raw = (await self._command(["container", "inspect", cid]) if current is None else
               await self.boundary.invoke(["container", "inspect", cid], current=current))
        value = json.loads(raw)[0]
        verify_container(value, self.images[role], self.profile, role, self.identity)
        if value["Id"] != cid or not value["State"]["Running"] or value["State"]["Pid"] <= 0:
            raise ResourceOutcomeUnknown("original_worker_not_running")
        if role in self._mount_sources and value["Mounts"][0]["Source"] != self._mount_sources[role]:
            raise ResourceConflict("original_worker_mount_source_changed")
        return value

    def _require_broker_attachment(self, record):
        settings = record.get("NetworkSettings")
        attachments = settings.get("Networks") if type(settings) is dict else None
        if (type(attachments) is not dict or set(attachments) != {self.names["network"]}
                or type(attachments[self.names["network"]]) is not dict
                or attachments[self.names["network"]].get("NetworkID") != self._network_identity):
            raise ResourceUnsupported("original_broker_attachment_changed")

    def _require_broker_endpoints(self, record):
        endpoints = _original_loopback_endpoints(record, self.containers["broker"], self._native_instances["broker"])
        if endpoints != self._broker_endpoints:
            raise ResourceUnsupported("original_broker_loopback_map_changed")
        return endpoints

    async def verify_live(self, current):
        current()
        if self._closed:
            raise ResourceNotFound("original_local_worker_closed")
        for role in ("browser", "broker"):
            record = await self._inspect(role, current=current)
            if (record["State"]["Pid"], record["State"]["StartedAt"]) != self._native_instances[role]:
                raise ResourceNotFound("original_worker_incarnation_changed")
        raw = await self.boundary.invoke(["network", "inspect", self.network], current=current)
        network = json.loads(raw)[0]
        _verify_broker_network(network, self._network_identity, self.identity, {self.containers["broker"]})
        broker = await self._inspect("broker", current=current)
        self._require_broker_attachment(broker)
        self._require_broker_endpoints(broker)
        volume = json.loads(await self.boundary.invoke(["volume", "inspect", self.volume], current=current))[0]
        if self._volume_facts(volume) != self._volume_identity:
            raise ResourceConflict("original_ipc_volume_replaced")
        current()

    @staticmethod
    def _volume_facts(volume):
        return {name: volume.get(name) for name in ("Name", "Driver", "Mountpoint", "CreatedAt", "Options", "Labels", "Scope")}

    async def _require_absent(self, kind, name):
        try:
            await self._command([kind, "inspect", name])
        except ResourceNotFound:
            return
        raise ResourceConflict("preexisting_local_worker_resource_requires_reconciliation")

    async def _create_named(self, kind, name, arguments):
        # Retain the original create before its first possible native effect.
        # CLI retirement and a subsequent absent-name inspection cannot unsend
        # this request or prove that the daemon will never finish creating it.
        key = (kind, name)
        self._pending_creates.add(key)
        output = await self._command(arguments)
        identity = output.decode().strip()
        if (identity != name if kind == "volume" else TOKEN.fullmatch(identity) is None):
            raise ResourceOutcomeUnknown("original_" + kind + "_create_ack_unknown")
        self._pending_creates.remove(key)
        return identity

    async def create(self):
        info = json.loads(await self._command(["info", "--format", "{{json .}}"]))
        if (info.get("ServerVersion") != "29.8.2" or info.get("OSType") != "linux"
                or info.get("Architecture") != "aarch64" or info.get("DefaultRuntime") != "runc"):
            raise ResourceUnsupported("unsupported_docker_platform_requires_native_conformance")
        for role in ("browser", "broker"):
            image = json.loads(await self._command(["image", "inspect", self.profile.image(role)]))[0]
            if (image.get("Id") != self.profile.image(role) or image.get("Os") != "linux"
                    or image.get("Architecture") != "arm64"
                    or image.get("Config", {}).get("Labels", {}).get(LABEL + ".contract") != CONTRACT):
                raise ResourceUnsupported("exact_complete_sdk046_worker_image_required")
            self.images[role] = image
        # Retain a preissued exact name before the native create. A missing
        # result remains UNKNOWN; no second create or automatic retry follows.
        await self._require_absent("volume", self.names["volume"])
        # Read-only refusals still own no resource. Persist acquiring under the
        # SAME original check immediately before the first possible create;
        # no failed/unknown create can become never-acquired retirement.
        self.original._check_original()
        runtime = self.original._host._runtime
        self.original._record.phase = "acquiring"
        await runtime.repository.save(runtime.scope(self.original._context),
            self.original._record, self.original._check_original)
        self.original._check_original()
        self.volume = self.names["volume"]
        await self._create_named("volume", self.volume, ["volume", "create", "--driver", "local", "--opt", "type=tmpfs",
            "--opt", "device=tmpfs", "--opt", "o=size=1048576,uid=10002,gid=10003,mode=0710",
            "--label", LABEL + "=" + self.identity, self.volume])
        volume = json.loads(await self._command(["volume", "inspect", self.volume]))[0]
        if (volume.get("Name") != self.volume or volume.get("Driver") != "local"
                or volume.get("Labels", {}).get(LABEL) != self.identity
                or volume.get("Options") != {"type": "tmpfs", "device": "tmpfs",
                    "o": "size=1048576,uid=10002,gid=10003,mode=0710"}):
            raise ResourceUnsupported("original_worker_ipc_volume_differs")
        self._volume_identity = self._volume_facts(volume)
        await self._require_absent("network", self.names["network"])
        self.network = self.names["network"]  # Bounded unknown-create discovery only.
        native_id = await self._create_named("network", self.network, ["network", "create", "--driver", "bridge", "--ipv6=false",
            "--label", LABEL + "=" + self.identity, self.network])
        self.network = self._network_identity = native_id
        network = json.loads(await self._command(["network", "inspect", native_id]))[0]
        if network.get("Id") != native_id or network.get("Name") != self.names["network"]:
            raise ResourceOutcomeUnknown("original_network_identity_unknown")
        _verify_broker_network(network, native_id, self.identity, set())
        for role in ("browser", "broker"):
            # A known name is a cleanup target only after exact label/image
            # inspection. Never delete an arbitrary same-named object.
            await self._require_absent("container", self.names[role])
            self.containers[role] = self.names[role]
            cid = await self._create_named("container", self.names[role],
                container_arguments(self.profile, role, self.identity))
            self.containers[role] = cid
            record = json.loads(await self._command(["container", "inspect", cid]))[0]
            verify_container(record, self.images[role], self.profile, role, self.identity)
            if record["Mounts"][0]["Source"] != volume["Mountpoint"]:
                raise ResourceUnsupported("original_ipc_mount_source_differs")
            self._mount_sources[role] = record["Mounts"][0]["Source"]
            await self._command(["start", cid])
            started = await self._inspect(role)
            self._native_instances[role] = (started["State"]["Pid"], started["State"]["StartedAt"])
            if role == "broker":
                self._require_broker_attachment(started)
                # Retain the same original instance before fallible map
                # validation. No application credentials exist at this point.
                self._broker_endpoints = _original_loopback_endpoints(
                    started, self.containers["broker"], self._native_instances["broker"])
            raw = await self._command(["exec", cid, "python", "-B", "-c",
                "from pathlib import Path;import time;p=Path('" + (PRIVATE if role == "browser" else "/tmp/kestrel-broker") + "/incarnation');"
                "\nfor _ in range(100):\n if p.is_file():break\n time.sleep(.02)\nelse:raise SystemExit(1)"
                "\nprint(p.read_text())"])
            incarnation = raw.decode().strip()
            if TOKEN.fullmatch(incarnation) is None:
                raise ResourceOutcomeUnknown("worker_incarnation_unknown")
            self.incarnations[role] = incarnation

    async def bootstrap(self, request, credential):
        # Runtime created this genuine SessionRequest after acquiring/saving
        # its lease. This is not an admission-time fabricated session.
        original = self.original
        original._require_origin()
        if self._broker_endpoints is None:
            raise ResourceUnsupported("original_loopback_role_endpoint_required")
        payload = request.model_dump(mode="json")
        payload["required_capabilities"] = sorted(request.required_capabilities - {"takeover"})
        browser = {"schema_version": 1, "secret": credential, "request": payload,
            "document_html": "<!doctype html><title>Private workspace</title><body contenteditable=true style='min-height:500px;font:24px sans-serif'></body>",
            "incarnation": self.incarnations["browser"], "network_profile": None,
            "profile_sha256": self.profile.digest()}
        await self._command(["exec", "-i", self.containers["browser"], *_program_arguments("browser", "--initialize")],
                            data=_json(browser)+b"\n")
        admission_raw = await self._command(["exec", self.containers["browser"], "python", "-B", "-c",
            "from pathlib import Path;import time;p=Path('"+PRIVATE+"/admission.json');"
            "\nfor _ in range(200):\n if p.is_file():break\n time.sleep(.05)\nelse:raise SystemExit(1)"
            "\nprint(p.read_text())"])
        try:
            admission = json.loads(admission_raw)
            if (admission["uid"] != 10002 or admission["gid"] != 10003
                    or type(admission["groups"]) is not list or not set(admission["groups"]) <= {10003}
                    or admission["capabilities"] != {key: 0 for key in ("CapEff", "CapBnd", "CapPrm", "CapInh", "CapAmb")}
                    or admission["no_new_privileges"] is not True or admission["rtprio_limit"] != [0, 0]
                    or admission["worker_incarnation"] != self.incarnations["browser"]
                    or any(value not in admission["sandbox"] for value in ("PID namespaces\tYes", "Network namespaces\tYes", "Seccomp-BPF sandbox\tYes"))):
                raise ValueError
        except (ValueError, TypeError, KeyError):
            raise ResourceUnsupported("original_browser_sandbox_not_proven") from None
        ca, client, roles, contexts = _tls_material(request.expires_at)
        broker = {"profile": PROFILE, "identity": self.identity, "profile_sha256": self.profile.digest(),
            "browser_incarnation": self.incarnations["browser"], "broker_incarnation": self.incarnations["broker"],
            "expires_at": request.expires_at.isoformat(), "client_ca": ca, "client_sha256": client, "roles": roles}
        await self._command(["exec", "-i", self.containers["broker"], *_program_arguments("broker", "--initialize")],
                            data=_json(broker)+b"\n")
        record = await self._inspect("broker")
        self._require_broker_attachment(record)
        endpoints = self._require_broker_endpoints(record)
        target_type, transport_type = _transport_types()
        transports = {}
        for role, port in zip(("normal", "priority"), endpoints, strict=True):
            target = target_type(address="127.0.0.1", port=port,
                server_name=role+".worker.kestrel.invalid", certificate_sha256=roles[role]["sha256"],
                expires_at=request.expires_at)
            transports[role] = transport_type(target, contexts[role])
        # Readiness is bounded, content-free and descriptive. It never supplies
        # authority or replaces the subsequent real controller status.
        await self._command(["exec", self.containers["broker"], "python", "-B", "-c",
            "from pathlib import Path;import time;p=Path('/tmp/kestrel-broker/ready');"
            "\nfor _ in range(200):\n if p.is_file() and p.read_text()=='ready':break\n time.sleep(.05)\nelse:raise SystemExit(1)"])
        self._tls = transports

    async def open_original(self, request):
        original = self.original
        await self.verify_live(original._check_original)
        value = await original._host._driver._rpc(request.session_id, "status",
            {"scope": request.scope.model_dump(mode="json"), "session_id": request.session_id},
            priority=True, current=original._check_original)
        session = _record(ComputerSession, value)
        if (session.scope != request.scope or session.session_id != request.session_id
                or session.lease_id != request.lease_id or session.driver_id != DRIVER
                or session.expires_at != request.expires_at or session.state != "ready"
                or session.control_epoch != 0 or session.controller != "agent"):
            raise ResourceConflict("original_worker_session_status_differs")
        return session

    async def close(self):
        if (type(self) is not _LocalWorkerResources or self._binding is not self
                or self._close_admission is not self._original_close_admission
                or self.original._resources is not self):
            raise ResourceConflict("original_local_worker_cleanup_owner_differs")
        # Descriptive transport admission, not authority. It only removes
        # access, synchronously before Task publication/any cleanup await.
        self._closing = True
        admission_failure = None
        if self._close_admission.task is None:
            try:
                create_owned_task(self._retire(), admission=self._close_admission, retirement_resource=self,
                                  retain_originals=True, name="original-two-domain-retirement")
            except BaseException as error:
                admission_failure = error
        if self._close_admission.task is None:
            raise admission_failure or ResourceOutcomeUnknown("original_cleanup_admission_unknown")
        outcome = await await_owned_task(self._close_admission.task)
        if admission_failure is not None:
            raise execution_terminal_error(admission_failure, outcome.error, outcome.cancellation, retain_originals=True)
        return raise_owned_outcome(outcome, operation="original two-domain retirement", retain_originals=True)

    async def _retire(self):
        if self._closed:
            return
        errors = []
        for role, target in reversed(tuple(self.containers.items())):
            create_key = ("container", target)
            known_original = role in self._native_instances or TOKEN.fullmatch(target) is not None
            signal_error = None
            try:
                record = await self.boundary.inspect(target)
                if (record.get("Config", {}).get("Labels", {}).get(LABEL) != self.identity
                        or record.get("Image") != self.profile.image(role)
                        or (known_original and record.get("Id") != target)):
                    raise ResourceConflict("original_cleanup_container_identity_differs")
                cid = record["Id"]
                if create_key in self._pending_creates:
                    if TOKEN.fullmatch(cid) is None:
                        raise ResourceOutcomeUnknown("original_created_container_identity_unknown")
                    # Exact owned inspection can discover a committed create.
                    # Retain that immutable ID before any fallible cleanup.
                    self.containers[role] = target = cid
                    known_original = True
                    self._pending_creates.remove(create_key)
                state = record.get("State", {})
                if state.get("Running"):
                    # The exact fixed programs interpret SIGTERM as lifecycle
                    # admission withdrawal, drain their original handles and
                    # exit zero only after their own native joins. This is not
                    # another worker command/permission or an action retry.
                    try:
                        await self.boundary.invoke(["kill", "--signal=SIGTERM", cid])
                    except DockerCommandError as error:
                        signal_error = error
                        # Only the measured daemon refusal for this exact CID
                        # admits reinspection. It is neither a wait ACK nor a
                        # reason to retransmit a signal or force removal.
                        diagnostic = (b"Error response from daemon: cannot kill container: " + cid.encode()
                            + b": container " + cid.encode() + b" is not running\n")
                        if (type(error) is not DockerCommandError
                                or error.args != ("docker_operation_outcome_unknown",)
                                or error.diagnostics.get_secret_value() != diagnostic):
                            raise
                        started_at = state.get("StartedAt")
                        native = self._native_instances.get(role)
                        def require_original_stopped(observed):
                            stopped = observed.get("State", {})
                            if (TOKEN.fullmatch(cid) is None
                                    or state.get("Running") is not True
                                    or type(state.get("Pid")) is not int or state["Pid"] <= 0
                                    or not isinstance(started_at, str) or not started_at
                                    or started_at == "0001-01-01T00:00:00Z"
                                    or (native is not None and native != (state["Pid"], started_at))
                                    or observed.get("Id") != cid or observed.get("Image") != self.profile.image(role)
                                    or observed.get("Config", {}).get("Labels", {}).get(LABEL) != self.identity
                                    or stopped.get("StartedAt") != started_at or stopped.get("Status") != "exited"
                                    or any(stopped.get(key) is not False for key in
                                        ("Running", "Paused", "Restarting", "OOMKilled", "Dead"))
                                    or type(stopped.get("Pid")) is not int or stopped["Pid"] != 0
                                    or type(stopped.get("ExitCode")) is not int or stopped["ExitCode"] != 0
                                    or stopped.get("Error") != ""):
                                raise ResourceOutcomeUnknown("original_stopped_signal_instance_not_confirmed")
                        require_original_stopped(await self.boundary.inspect(cid))
                never_started = (state.get("Status") == "created" and state.get("Pid") == 0
                                 and role not in self._native_instances)
                if not never_started:
                    raw = await self.boundary.invoke(["wait", cid], timeout=WORKER_RETIRE_WAIT_SECONDS)
                    if raw.strip() != b"0":
                        raise ResourceOutcomeUnknown("original_worker_native_retirement_unknown")
                    self._container_waits[cid] = 0  # Actual native wait ACK before the next fallible observation.
                finished = await self.boundary.inspect(cid)
                if signal_error is not None:
                    require_original_stopped(finished)
                if (finished["State"].get("Running") is not False
                        or finished["State"].get("ExitCode") != 0 or finished["State"].get("Pid") != 0):
                    raise ResourceOutcomeUnknown("original_worker_native_wait_not_confirmed")
                await self.boundary.invoke(["container", "rm", cid])
                try:
                    await self.boundary.inspect(cid)
                except ResourceNotFound:
                    continue
                raise ResourceOutcomeUnknown("original_container_retirement_unknown")
            except ResourceNotFound as missing:
                if signal_error is not None:
                    errors.append(execution_terminal_error(signal_error, missing, retain_originals=True))
                elif known_original and self._container_waits.get(target) != 0:
                    unknown = ResourceOutcomeUnknown("original_worker_disappeared_without_native_wait")
                    unknown.__cause__ = missing
                    errors.append(unknown)
                elif create_key in self._pending_creates:
                    unknown = ResourceOutcomeUnknown("original_container_create_outcome_unknown")
                    unknown.__cause__ = missing
                    errors.append(unknown)
            except BaseException as error:
                errors.append(execution_terminal_error(signal_error, error, retain_originals=True)
                    if signal_error is not None and error is not signal_error else error)
        if errors:
            # Keep dependent original IPC/network resources while any native
            # process/container retirement is unresolved. No forced green.
            raise execution_terminal_error(None, *errors, retain_originals=True)
        for kind, target in (("network", self.network), ("volume", self.volume)):
            if target is None:
                continue
            try:
                create_key = (kind, target)
                raw = await self.boundary.invoke([kind, "inspect", target])
                record = json.loads(raw)[0]
                if record.get("Labels", {}).get(LABEL) != self.identity:
                    raise ResourceConflict("original_cleanup_resource_identity_differs")
                if create_key in self._pending_creates:
                    if record.get("Name") != target:
                        raise ResourceConflict("original_created_resource_name_differs")
                    if kind == "network":
                        native_id = record.get("Id")
                        if not isinstance(native_id, str) or TOKEN.fullmatch(native_id) is None:
                            raise ResourceOutcomeUnknown("original_created_network_identity_unknown")
                        self.network = self._network_identity = target = native_id
                    else:
                        self._volume_identity = self._volume_facts(record)
                    self._pending_creates.remove(create_key)
                if ((kind == "network" and self._network_identity is not None and record.get("Id") != self._network_identity)
                        or (kind == "volume" and self._volume_identity is not None and self._volume_facts(record) != self._volume_identity)):
                    raise ResourceConflict("original_cleanup_instance_identity_differs")
                await self.boundary.invoke([kind, "rm", target])
                try:
                    await self.boundary.invoke([kind, "inspect", target])
                except ResourceNotFound:
                    continue
                raise ResourceOutcomeUnknown("original_resource_retirement_unknown")
            except ResourceNotFound as missing:
                if create_key in self._pending_creates:
                    unknown = ResourceOutcomeUnknown("original_" + kind + "_create_outcome_unknown")
                    unknown.__cause__ = missing
                    errors.append(unknown)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise execution_terminal_error(None, *errors, retain_originals=True)
        self._closed = True


class LocalTwoDomainProvider:
    provider_id = PROVIDER

    def __init__(self, *, docker_socket, browser_image, broker_image, credential_key):
        _transport_types()
        self.profile = LocalWorkerProfile(browser_image, broker_image)
        if type(credential_key) is SecretBytes:
            credential_key = credential_key.get_secret_value()
        if type(credential_key) is not bytes or len(credential_key) != 32:
            raise ResourceUnsupported("private_32_byte_worker_key_required")
        self._key = SecretBytes(credential_key)
        self.boundary, self._setup_host = DockerBoundary(docker_socket), None

    async def capabilities(self):
        return EnvironmentCapabilities(provider_id=PROVIDER, location="local", surfaces=frozenset({"browser"}), operations=OPERATIONS)

    def _request(self, request):
        if (request.provider_id != PROVIDER or request.profile_id != PROFILE
                or request.image != self.profile.browser_image or request.location != "local"
                or request.region is not None or request.surface != "browser"
                or not request.required_capabilities <= OPERATIONS
                or not 0 < (request.expires_at-_now()).total_seconds() <= 3600):
            raise ResourceUnsupported("unsupported_two_domain_worker_request")

    async def quote(self, request):
        self._request(request)
        return EnvironmentQuote(quote_id="local-worker-"+request.digest(), request_id=request.request_id,
            request_sha256=request.digest(), scope=request.scope, provider_id=PROVIDER, location="local",
            expires_at=request.expires_at, hourly_cost_usd=Decimal(0), estimated_total_cost_usd=Decimal(0),
            capabilities=await self.capabilities())

    async def acquire(self, request, quote):
        self._request(request)
        original = self._setup_host._current_setup()
        if request is not original._record.request or quote is not original._record.quote:
            raise ResourceConflict("original_local_request_quote_required")
        resource = original._resources
        if resource is None:
            raise ResourceOutcomeUnknown("original_local_resources_absent")
        return EnvironmentLease(lease_id="local-worker-"+resource.identity, request_id=request.request_id,
            quote_id=quote.quote_id, scope=request.scope, provider_id=PROVIDER, location="local",
            state="pending", expires_at=request.expires_at, provider_resource_id=resource.containers["browser"])

    async def connect(self, scope, lease):
        original = self._setup_host._current_setup()
        resources = original._resources
        if lease.scope != scope or lease.lease_id != "local-worker-"+resources.identity:
            raise ResourceConflict("original_local_lease_differs")
        endpoint = {"profile": PROFILE, "identity": resources.identity,
                    "profile_sha256": self.profile.digest(), "incarnations": resources.incarnations,
                    "containers": resources.containers, "scope": scope.model_dump(mode="json"),
                    "expires_at": lease.expires_at.isoformat()}
        secret = hmac.new(self._key.get_secret_value(), b"kestrel-local-worker-session-v1\0"+_json(endpoint), hashlib.sha256).hexdigest()
        connection = DriverConnection(lease_id=lease.lease_id, driver_id=DRIVER,
            endpoint=SecretStr(_json(endpoint).decode()), credential=SecretStr(secret), expires_at=lease.expires_at)
        resources._connection = connection
        return connection

    async def inspect(self, scope, lease):
        resources = self._setup_host._driver._resources_for_lease(scope, lease)
        original = self._setup_host._current_setup()
        await resources.verify_live(original._check_original)
        return lease.model_copy(update={"state": "ready"})

    async def release(self, scope, lease):
        resources = self._setup_host._driver._resources_for_lease(scope, lease)
        await resources.close()
        return await self._original_release_receipt(scope, resources.original._record, lease)

    async def _original_release_receipt(self, scope, record, lease):
        """Observe an already issued original closer; never start retirement.

        Runtime supplies its freshly checked releasing row after a failed
        observation. The ordinary provider path supplies the original setup
        record. Both retain the same lease/container/connection identities and
        join the same original setup Future and admitted retirement Task.
        """
        resources = self._setup_host._driver._resources_for_lease(scope, lease)
        admission = resources._close_admission
        task = admission.task
        if (record.request.scope != scope or record.lease is None
                or record.lease.lease_id != lease.lease_id
                or record.lease.provider_resource_id != lease.provider_resource_id
                or admission is not resources._original_close_admission or task is None
                or getattr(task, "_kestrel_owned_admission", None) is not admission):
            raise ResourceOutcomeUnknown("original_local_worker_retirement_not_issued")
        await resources.original._join_retirement(record, failed=False)
        if resources._close_admission is not admission or admission.task is not task:
            raise ResourceOutcomeUnknown("original_local_worker_retirement_changed")
        return lease.model_copy(update={"state": "released", "released_at": _now(), "actual_cost_usd": Decimal(0)})


class LocalTwoDomainDriver:
    driver_id = DRIVER

    def __init__(self, *, docker_socket, egress_guard=None):
        _transport_types()
        if egress_guard is not None and not callable(egress_guard):
            raise ResourceUnsupported("original_egress_borrow_required")
        self.boundary, self.egress_guard = DockerBoundary(docker_socket), egress_guard
        self._setup_host = None
        self._connections = {}

    def _resources_for_lease(self, scope, lease):
        matches = [value for value in self._connections.values()
            if value._connection.lease_id == lease.lease_id
            and value.original._record.request.scope == scope
            and value.containers["browser"] == lease.provider_resource_id]
        if len(matches) != 1:
            raise ResourceOutcomeUnknown("original_local_worker_closer_unavailable")
        return matches[0]

    async def open(self, request, connection):
        original = self._setup_host._current_setup()
        resources = original._resources
        if (connection is not resources._connection or request.session_id != original._record.session_id
                or request.scope != original._record.request.scope or request.lease_id != connection.lease_id
                or connection.driver_id != DRIVER or request.expires_at > connection.expires_at
                or request.surface != "browser" or not request.required_capabilities <= OPERATIONS):
            raise ResourceConflict("original_local_session_connection_differs")
        if request.session_id in self._connections:
            raise ResourceConflict("local_worker_session_already_open")
        self._connections[request.session_id] = resources
        await resources.bootstrap(request, connection.credential.get_secret_value())
        return await resources.open_original(request)

    @asynccontextmanager
    async def _admission(self, current, resources):
        def original_current():
            current()
            if resources._closing or resources._closed:
                raise ResourceOutcomeUnknown("original_worker_application_admission_closed")
        original_current()
        yield original_current
        original_current()

    async def _rpc(self, session_id, operation, arguments, *, priority=False, current=None):
        resources = self._connections.get(session_id)
        if resources is None or resources._closing or resources._closed or resources._tls is None:
            raise ResourceNotFound("original_local_connection_unavailable")
        if current is None:
            raise OperatorAuthorizationError("original_worker_call_admission_required")
        current()
        await resources.verify_live(current)
        identifier = uuid4().hex
        raw = _json({"schema_version": 1, "secret": resources._connection.credential.get_secret_value(),
            "id": identifier, "operation": operation, "arguments": arguments})+b"\n"
        if len(raw) > MAX_REQUEST:
            raise ResourceUnsupported("worker_request_limit")
        trace = native_trace()
        if operation == "act" and trace is not None:
            # Conservative before transport entry: a lost send never becomes
            # a fabricated native no-dispatch proof.
            trace.input_started = trace.rpc_started = True
        response = await resources._tls["priority" if priority else "normal"].exchange(
            raw, admission=self._admission(current, resources), timeout_seconds=20)
        current()
        try:
            value = json.loads(response)
            if type(value) is not dict or value.get("id") != identifier or value.get("schema_version") != 1:
                raise ValueError
        except (ValueError, TypeError, UnicodeError):
            raise ResourceOutcomeUnknown("invalid_private_worker_response") from None
        if "error" in value:
            reason = value["error"]
            if type(reason) is not str:
                raise ResourceOutcomeUnknown("invalid_private_worker_response") from None
            if reason in {"stale_control", "stale_observation", "action_conflict", "control_conflict", "control_pending"}:
                raise ResourceConflict(reason)
            if reason in {"unsupported_request", "unsupported_action", "unsupported_control"}:
                raise ResourceUnsupported(reason)
            raise ResourceOutcomeUnknown("original_worker_operation_unknown")
        if "result" not in value:
            raise ResourceOutcomeUnknown("original_worker_result_absent")
        return value["result"]

    async def _session_rpc(self, session, operation, arguments, *, priority=False):
        runtime = self._setup_host._runtime
        # Runtime._session_call already activated the exact session-purpose
        # call. No new activation or authority is reconstructed here.
        calls = [(key, call) for key, call in runtime.egress._calls.items()
                 if key[:7] == runtime.egress._key(session, "unused")[:7]
                 and key[-1] == "session" and asyncio.current_task() in call.jobs]
        if len(calls) != 1:
            raise OperatorAuthorizationError("original_session_borrow_required")
        # Reuse the already registered exact native borrower, rather than
        # manufacturing another Task/activation for its SDK method.
        current = calls[0][1].check
        return await self._rpc(session.session_id, operation, arguments, priority=priority, current=current)

    async def observe(self, session):
        value = await self._session_rpc(session, "observe", {"session": session.model_dump(mode="json")})
        try:
            return ScreenObservation(frame=_record(ScreenFrame, value["frame"]),
                image=SecretBytes(base64.b64decode(value["image_base64"], validate=True)))
        except (KeyError, ValueError, TypeError, ValidationError):
            raise ResourceOutcomeUnknown("invalid_private_worker_observation") from None

    async def status(self, scope, session_id):
        resources = self._connections.get(session_id)
        if resources is None or resources.original._record.request.scope != scope:
            raise ResourceNotFound("original_worker_session_unavailable")
        matches = [call for key, call in self._setup_host._runtime.egress._calls.items()
                   if key[:4] == (scope.tenant_id, scope.principal_id, scope.agent_id, session_id)
                   and key[-1] == "session" and asyncio.current_task() in call.jobs]
        if len(matches) != 1:
            raise OperatorAuthorizationError("original_session_status_borrow_required")
        session = matches[0].session
        return _record(ComputerSession, await self._session_rpc(session, "status",
            {"scope": scope.model_dump(mode="json"), "session_id": session_id}, priority=True))

    async def act(self, session, request):
        if self.egress_guard is None:
            raise OperatorAuthorizationError("original_action_egress_required")
        with native_dispatch(session, request, DRIVER):
            async def invoke():
                async with self.egress_guard(session, request.action_id) as current:
                    return _record(ActionReceipt, await self._rpc(session.session_id, "act",
                        {"session": session.model_dump(mode="json"), "request": request.model_dump(mode="json")}, current=current))
            result = await await_owned_operation(invoke(), name="local-worker-original-action", cancel_on_caller_cancellation=True)
            from .dispatch import raise_driver_borrow_outcome
            return raise_driver_borrow_outcome(result, operation="local worker original action", retain_originals=True)

    async def action_status(self, session, action_id):
        return _record(ActionReceipt, await self._session_rpc(session, "action_status",
            {"session": session.model_dump(mode="json"), "action_id": action_id}, priority=True))

    async def control(self, session, request):
        return _record(ComputerSession, await self._session_rpc(session, "control",
            {"session": session.model_dump(mode="json"), "request": request.model_dump(mode="json")}, priority=True))


def environment_provider_factory(*, settings: Mapping[str, object]):
    if set(settings) != {"docker_socket", "browser_image", "broker_image", "credential_key"}:
        raise ResourceUnsupported("explicit_fixed_two_domain_provider_settings_required")
    return LocalTwoDomainProvider(**settings)


def computer_driver_factory(*, settings: Mapping[str, object]):
    if set(settings) not in ({"docker_socket"}, {"docker_socket", "egress_guard"}):
        raise ResourceUnsupported("explicit_fixed_two_domain_driver_settings_required")
    return LocalTwoDomainDriver(**settings)
