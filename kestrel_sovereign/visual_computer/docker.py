"""Concrete SDK adapters for one reviewed, local, offline Docker browser.

The host owns authorization and encrypted custody. Docker labels reconcile the
same admitted request; they are never caller authority. Every exchange targets
an immutable container ID and the exact running process incarnation.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from pydantic import SecretBytes, SecretStr, ValidationError
from kestrel_sdk.computer_use import (
    ActionReceipt, ActionRequest, ComputerSession, ControlRequest, DriverConnection,
    EnvironmentCapabilities, EnvironmentLease, EnvironmentQuote, EnvironmentRequest,
    ScreenObservation, ScreenFrame, SessionRequest,
)
from kestrel_sdk.resources import (
    ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown,
    ResourceUnsupported,
)
from .docker_policy import container_arguments, verify_container
from .docker_egress import DockerEgressPolicy, HTTPGateway, EgressDenied, EgressUnknown

PROVIDER = "local-docker-browser"
PROFILE = "offline-v1"
OPERATIONS = frozenset({"screenshot", "click", "type", "key", "scroll", "wait", "takeover", "files"})
# Takeover transfers custody and fences automation; human input transport is a
# separate capability. It never exposes CDP or unrestricted browser execution.
LABEL = "io.kestrel.visual-computer"
WORKER_CONTRACT = "sdk-0.43-http-v1"
MAX_OUTPUT = 24 * 1024 * 1024
ID = re.compile(r"^[a-f0-9]{64}$")
READ_INCARNATION = (
    "from pathlib import Path; import time; p=Path('/tmp/kestrel-browser/incarnation'); "
    "\nfor _ in range(100):\n if p.is_file(): break\n time.sleep(.02)\nelse: raise SystemExit(1)"
    "\nprint(p.read_text())"
)
IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[a-f0-9]{64}$")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _quote_hash(quote):
    payload = quote.model_dump(mode="json")
    payload["capabilities"]["operations"] = sorted(quote.capabilities.operations)
    payload["capabilities"]["surfaces"] = sorted(quote.capabilities.surfaces)
    return _hash(payload)


def _now():
    return datetime.now(UTC)


def _record(model, value):
    # JSON validation is intentional for strict SDK frozenset/datetime records.
    try:
        return model.model_validate_json(_json(value))
    except (ValidationError, ValueError, TypeError):
        raise ResourceOutcomeUnknown("invalid_private_worker_response") from None


class DockerCommandError(ResourceOutcomeUnknown):
    """Private diagnostics require explicit access, never exception/repr logging."""

    def __init__(self, reason, *, diagnostics=b""):
        self.diagnostics = SecretBytes(diagnostics)
        super().__init__(reason)


class DockerBoundary:
    """Local Unix daemon only; no inherited Docker context, proxy or credentials."""

    def __init__(self, socket):
        if not isinstance(socket, str) or not Path(socket).is_absolute():
            raise ResourceUnsupported("explicit_local_docker_socket_required")
        self.socket = socket
        executable = shutil.which("docker")
        if not executable:
            raise ResourceUnsupported("docker_cli_unavailable")
        self.executable = executable

    async def invoke(self, arguments, *, data=None, timeout=25, current=None):
        # A new empty Docker config disables user contexts and proxy injection.
        # HOME remains the real home, but no operator configuration is consumed.
        environment = {"HOME": os.environ.get("HOME", ""), "PATH": os.defpath,
                       "PYTHON_DOTENV_DISABLED": "1"}
        with tempfile.TemporaryDirectory(prefix="kestrel-private-docker-") as config:
            launch = asyncio.create_task(asyncio.create_subprocess_exec(
                self.executable, "--config", config, "--host", f"unix://{self.socket}",
                *arguments, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=environment, cwd=Path(__file__).parent,
            ))
            try:
                process = await asyncio.shield(launch)
            except BaseException:
                # Cancellation can race OS process creation. Retain the launch
                # until its exact process is known, then reap without stdin.
                process = await launch
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()
                raise
            async def read(stream):
                chunks, size = [], 0
                while chunk := await stream.read(min(65536, MAX_OUTPUT + 1 - size)):
                    size += len(chunk)
                    if size > MAX_OUTPUT:
                        raise DockerCommandError("docker_response_limit")
                    chunks.append(chunk)
                return b"".join(chunks)
            dispatched = False
            async def write():
                nonlocal dispatched
                try:
                    if current is not None:
                        current()
                    if data:
                        dispatched = True
                        process.stdin.write(data)
                        await process.stdin.drain()
                finally:
                    process.stdin.close()
            async def retire():
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()
            streams = [asyncio.create_task(write()), asyncio.create_task(read(process.stdout)),
                       asyncio.create_task(read(process.stderr))]
            try:
                async with asyncio.timeout(timeout):
                    _, output, error = await asyncio.gather(*streams)
                    await process.wait()
                    if current is not None:
                        current()
            except (TimeoutError, asyncio.CancelledError, OSError, DockerCommandError):
                await retire()
                # Killing the CLI cannot unsend an accepted Docker/browser call.
                raise ResourceOutcomeUnknown("docker_transport_outcome_unknown") from None
            except BaseException:
                # Authority can revoke during process launch or output delivery.
                # Join this owned process even when its check rejects the call.
                await retire()
                if dispatched:
                    raise ResourceOutcomeUnknown("driver_effect_outcome_unknown") from None
                raise
            finally:
                for task in streams:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*streams, return_exceptions=True)
            if process.returncode:
                # Only this exact daemon response establishes current absence.
                target = arguments[-1]
                absent = re.compile(rb"(?:Error: |Error response from daemon: )No such (?:object|image|container): "
                                    + re.escape(target.encode()) + rb"\s*")
                if arguments[:2] in (["container", "inspect"], ["image", "inspect"]) and absent.fullmatch(error):
                    raise ResourceNotFound("docker_resource_absent")
                raise DockerCommandError("docker_operation_outcome_unknown", diagnostics=output + error)
            return output

    async def inspect(self, target):
        try:
            result = json.loads(await self.invoke(["container", "inspect", target]))
            if not isinstance(result, list) or len(result) != 1:
                raise ValueError
            return result[0]
        except (ValueError, TypeError, IndexError):
            raise ResourceOutcomeUnknown("invalid_docker_inspection") from None

    async def platform(self):
        try:
            info = json.loads(await self.invoke(["info", "--format", "{{json .}}"] ))
        except (ValueError, TypeError):
            raise ResourceUnsupported("unverified_docker_platform") from None
        if (info.get("ServerVersion") != "29.8.2" or info.get("OSType") != "linux"
                or info.get("Architecture") != "aarch64" or info.get("DefaultRuntime") != "runc"):
            raise ResourceUnsupported("unsupported_docker_platform_requires_real_conformance")

    async def rpc(self, cid, value, *, mode=None, current=None):
        command = ["exec", "-i", cid, "python", "/opt/kestrel-browser/rpc_client.py"]
        if mode:
            command.append(mode)
        try:
            return json.loads(await self.invoke(command, data=_json(value) + b"\n", current=current))
        except (ValueError, TypeError):
            raise ResourceOutcomeUnknown("invalid_private_worker_response") from None


def _identity(scope, request_id):
    return _hash({"scope": scope.model_dump(mode="json"), "request_id": request_id})


def _scope(lease, scope):
    if lease.scope != scope or lease.provider_id != PROVIDER or lease.location != "local":
        raise ResourceNotFound("environment_unavailable")
    identity = _identity(scope, lease.request_id)
    if lease.lease_id != "docker-" + identity:
        raise ResourceNotFound("environment_unavailable")
    return identity


def _admit(record, image, identity):
    verify_container(record, image["Id"], identity, label=LABEL)
    config = record["Config"]
    if (image.get("Config", {}).get("Labels", {}).get(LABEL + ".worker-contract") != WORKER_CONTRACT
            or config.get("Env") != image.get("Config", {}).get("Env")
            or config.get("Cmd") != ["python", "/opt/kestrel-browser/worker.py"]
            or config.get("Entrypoint") is not None or config.get("OpenStdin") is not False
            or config.get("Tty") is not False or record.get("RestartCount") != 0
            or record.get("NetworkSettings", {}).get("Ports") not in ({}, None)):
        raise ResourceUnsupported("unsupported_browser_container_configuration")


class DockerEnvironmentProvider:
    """Idempotent acquisition with the canonical host as sole durable owner."""

    def __init__(self, *, docker_socket, image, credential_key, profile_id=None, network_policy=None):
        self.policy = DockerEgressPolicy.from_settings(network_policy)
        self.profile_id = self.policy.profile_id if self.policy is not None else PROFILE
        self.operations = OPERATIONS | ({"navigate"} if self.policy is not None else set())
        if not isinstance(image, str) or not IMAGE.fullmatch(image) or profile_id not in {None, self.profile_id}:
            raise ResourceUnsupported("unsupported_docker_image_or_profile")
        if isinstance(credential_key, SecretBytes):
            credential_key = credential_key.get_secret_value()
        if not isinstance(credential_key, bytes) or len(credential_key) != 32:
            raise ResourceUnsupported("private_32_byte_credential_key_required")
        self.boundary, self.image = DockerBoundary(docker_socket), image
        # This local-only profile pins Docker's content-addressed image ID
        # (configuration digest), not a remote registry manifest/index digest.
        self.image_id = "sha256:" + image.rsplit(":", 1)[1]
        self._key = SecretBytes(credential_key)

    async def capabilities(self):
        return EnvironmentCapabilities(provider_id=PROVIDER, location="local", surfaces=frozenset({"browser"}),
                                       operations=frozenset(self.operations))

    def _request(self, request):
        if (request.provider_id != PROVIDER or request.profile_id != self.profile_id or request.image != self.image
                or request.location != "local" or request.region is not None or request.surface != "browser"
                or not request.required_capabilities <= self.operations
                or not 0 < (request.expires_at - _now()).total_seconds() <= 3600):
            raise ResourceUnsupported("unsupported_offline_browser_request")

    async def quote(self, request):
        self._request(request)
        return EnvironmentQuote(quote_id="docker-" + request.digest(), request_id=request.request_id,
                                request_sha256=request.digest(), scope=request.scope, provider_id=PROVIDER,
                                location="local", expires_at=request.expires_at,
                                hourly_cost_usd=Decimal(0), estimated_total_cost_usd=Decimal(0),
                                capabilities=await self.capabilities())

    async def acquire(self, request, quote):
        self._request(request)
        if quote != await self.quote(request):
            raise ResourceConflict("environment_quote_mismatch")
        await self.boundary.platform()
        identity = _identity(request.scope, request.request_id)
        name = "kestrel-visual-" + identity
        image = json.loads(await self.boundary.invoke(["image", "inspect", self.image_id]))[0]
        if (image.get("Id") != self.image_id or image.get("Os") != "linux" or image.get("Architecture") != "arm64"
                or image.get("Config", {}).get("Labels", {}).get(LABEL + ".worker-contract") != WORKER_CONTRACT):
            raise ResourceUnsupported("unsupported_browser_image_platform")
        try:
            record = await self.boundary.inspect(name)
        except ResourceNotFound:
            arguments = container_arguments(image["Id"], identity, worker=True,
                                            label=LABEL, name_prefix="kestrel-visual-")
            # Bind the exact accepted constraints without plaintext owner fields.
            arguments[1:1] = ["--label", f"{LABEL}.request={request.digest()}",
                              "--label", f"{LABEL}.quote={_quote_hash(quote)}",
                              "--label", f"{LABEL}.expiry={request.expires_at.isoformat()}"]
            await self.boundary.invoke(arguments)
            record = await self.boundary.inspect(name)
        labels = record["Config"].get("Labels") or {}
        if (labels.get(LABEL + ".request") != request.digest()
                or labels.get(LABEL + ".quote") != _quote_hash(quote)):
            raise ResourceConflict("environment_request_conflict")
        _admit(record, image, identity)
        state = record["State"]
        if state["Status"] == "created":
            await self.boundary.invoke(["start", record["Id"]])
            record = await self.boundary.inspect(record["Id"])
        # No restart: a retired worker never regains accepted-action state.
        live = record["State"]["Running"] and record["State"].get("Pid", 0) > 0
        return EnvironmentLease(lease_id="docker-" + identity, request_id=request.request_id,
                                quote_id=quote.quote_id, scope=request.scope, provider_id=PROVIDER, location="local",
                                state="pending" if live else "failed", expires_at=request.expires_at,
                                provider_resource_id=record["Id"])

    async def _owned(self, scope, lease):
        identity = _scope(lease, scope)
        if lease.provider_resource_id is None or not ID.fullmatch(lease.provider_resource_id):
            raise ResourceOutcomeUnknown("exact_container_identity_required")
        record = await self.boundary.inspect(lease.provider_resource_id)
        if (record.get("Id") != lease.provider_resource_id
                or record["Config"].get("Labels", {}).get(LABEL) != identity):
            raise ResourceConflict("environment_identity_mismatch")
        return record, identity

    async def inspect(self, scope, lease):
        _scope(lease, scope)
        try:
            record, identity = await self._owned(scope, lease)
        except ResourceNotFound:
            # The exact immutable resource is physically absent at this boundary.
            return lease.model_copy(update={"state": "released", "released_at": _now(), "actual_cost_usd": Decimal(0)})
        if lease.state == "released":
            raise ResourceConflict("released_environment_still_exists")
        live = record["State"]["Running"] and record["State"].get("Pid", 0) > 0
        state = "pending" if live and _now() < lease.expires_at else "failed"
        if live:
            connection = await self.connect(scope, lease)
            driver = DockerComputerDriver(docker_socket=self.boundary.socket, network_policy=self.policy)
            try:
                admission = await driver._admission(connection)
                if admission:
                    state = "ready"
            except ResourceOutcomeUnknown:
                pass  # Pending readiness is explicit; it never grants input.
        return lease.model_copy(update={"state": state})

    async def connect(self, scope, lease):
        record, identity = await self._owned(scope, lease)
        if _now() >= lease.expires_at or not record["State"]["Running"]:
            raise ResourceNotFound("live_environment_unavailable")
        image = json.loads(await self.boundary.invoke(["image", "inspect", self.image_id]))[0]
        _admit(record, image, identity)
        incarnation = (await self.boundary.invoke(["exec", record["Id"], "python", "-c", READ_INCARNATION])).decode().strip()
        if not ID.fullmatch(incarnation):
            raise ResourceOutcomeUnknown("worker_incarnation_unknown")
        endpoint = {"profile_id": self.profile_id, "incarnation": incarnation, "container_id": record["Id"], "image_id": image["Id"], "identity": identity,
                    "scope": scope.model_dump(mode="json"), "lease_id": lease.lease_id,
                    "expires_at": lease.expires_at.isoformat(), "pid": record["State"]["Pid"],
                    "started_at": record["State"]["StartedAt"]}
        secret = hmac.new(self._key.get_secret_value(), b"kestrel-worker-v1\0" + _json(endpoint), hashlib.sha256).hexdigest()
        return DriverConnection(lease_id=lease.lease_id, driver_id=PROVIDER, endpoint=SecretStr(_json(endpoint).decode()),
                                credential=SecretStr(secret), expires_at=lease.expires_at)

    async def release(self, scope, lease):
        _scope(lease, scope)
        try:
            record, _ = await self._owned(scope, lease)
        except ResourceNotFound:
            return lease.model_copy(update={"state": "released", "released_at": _now(), "actual_cost_usd": Decimal(0)})
        await self.boundary.invoke(["container", "rm", "--force", record["Id"]])
        try:
            await self.boundary.inspect(record["Id"])
        except ResourceNotFound:
            return lease.model_copy(update={"state": "released", "released_at": _now(), "actual_cost_usd": Decimal(0)})
        raise ResourceOutcomeUnknown("physical_release_unconfirmed")


class DockerComputerDriver:
    driver_id = PROVIDER

    def __init__(self, *, docker_socket, egress_guard=None, network_policy=None):
        if egress_guard is not None and not callable(egress_guard):
            raise ResourceUnsupported("invalid_private_egress_guard")
        self.boundary = DockerBoundary(docker_socket)
        self.egress_guard = egress_guard
        self.policy = DockerEgressPolicy.from_settings(network_policy)
        self.profile_id = self.policy.profile_id if self.policy is not None else PROFILE
        self.operations = OPERATIONS | ({"navigate"} if self.policy is not None else set())
        self._pumps = {}
        self._borrowers = {}
        self._connections = {}

    def workspace_transfer(self):
        return DockerWorkspaceTransfer(self)

    async def _bound(self, connection, *, cleanup=False):
        try:
            endpoint = json.loads(connection.endpoint.get_secret_value())
            if (set(endpoint) != {"container_id", "image_id", "identity", "scope", "lease_id", "expires_at", "pid", "started_at", "incarnation", "profile_id"}
                    or not ID.fullmatch(endpoint["container_id"]) or not ID.fullmatch(endpoint["incarnation"]) or not ID.fullmatch(endpoint["identity"])
                    or endpoint["profile_id"] != self.profile_id or connection.driver_id != PROVIDER or endpoint["lease_id"] != connection.lease_id
                    or datetime.fromisoformat(endpoint["expires_at"]) != connection.expires_at
                    or (_now() >= connection.expires_at and not cleanup)):
                raise ValueError
            record = await self.boundary.inspect(endpoint["container_id"])
            image = json.loads(await self.boundary.invoke(["image", "inspect", endpoint["image_id"]]))[0]
            if image.get("Id") != endpoint["image_id"]:
                raise ValueError
            _admit(record, image, endpoint["identity"])
            incarnation = (await self.boundary.invoke(["exec", endpoint["container_id"], "python", "-c", READ_INCARNATION])).decode().strip()
            state = record["State"]
            if (incarnation != endpoint["incarnation"] or not state["Running"] or state["Pid"] != endpoint["pid"]
                    or state["StartedAt"] != endpoint["started_at"]):
                raise ValueError
            return endpoint
        except (KeyError, ValueError, TypeError, RuntimeError):
            raise ResourceNotFound("worker_incarnation_unavailable") from None

    async def _admission(self, connection):
        endpoint = await self._bound(connection)
        raw = await self.boundary.invoke(["exec", endpoint["container_id"], "python", "-c",
            "from pathlib import Path; import sys; p=Path('/tmp/kestrel-browser/admission.json'); sys.stdout.buffer.write(p.read_bytes())"])
        try:
            result = json.loads(raw)
            if (result["uid"] == 0 or result["capabilities"] != {key: 0 for key in ("CapEff", "CapBnd", "CapPrm", "CapInh", "CapAmb")}
                    or result["rtprio_limit"] != [0, 0] or 0 in result["groups"] or result["no_new_privileges"] is not True
                    or result["worker_incarnation"] != endpoint["incarnation"] or not result["browser_version"]
                    or any(value not in result["sandbox"] for value in ("PID namespaces\tYes", "Network namespaces\tYes", "Seccomp-BPF sandbox\tYes"))):
                raise ValueError
            return result
        except (KeyError, ValueError, TypeError):
            raise ResourceUnsupported("browser_sandbox_admission_failed") from None

    async def open(self, request, connection):
        # All public identity/bounds checks happen before any credential is sent.
        if (connection.driver_id != self.driver_id or connection.lease_id != request.lease_id
                or request.expires_at > connection.expires_at or _now() >= request.expires_at
                or request.surface != "browser" or not request.required_capabilities <= self.operations
                or (self.policy is not None and self.egress_guard is None)):
            raise ResourceUnsupported("invalid_browser_connection_or_capabilities")
        endpoint = await self._bound(connection)
        if endpoint["scope"] != request.scope.model_dump(mode="json"):
            raise ResourceNotFound("session_unavailable")
        previous = self._connections.get(request.session_id)
        if previous is not None and previous != connection:
            raise ResourceConflict("session_connection_conflict")
        # Bootstrap is one-shot. A lost result is unknown and must reconcile this
        # exact incarnation; it never spawns a replacement or changes the secret.
        await self.boundary.invoke(["exec", endpoint["container_id"], "python", "-c",
            "import pathlib,time; p=pathlib.Path('/tmp/kestrel-browser'); "
            "\nfor _ in range(100):\n if p.is_dir(): break\n time.sleep(.05)\nelse: raise SystemExit(1)"])
        marker = await self.boundary.invoke(["exec", endpoint["container_id"], "python", "-c",
            "from pathlib import Path; print(int(Path('/tmp/kestrel-browser/initialized').exists()))"])
        self._connections[request.session_id] = connection
        if marker.strip() == b"0":
            payload = request.model_dump(mode="json")
            payload["required_capabilities"] = sorted(request.required_capabilities - {"takeover"})
            # The initial synthetic empty document has no inherited website,
            # cookie profile, file access or operator content.
            await self.boundary.rpc(endpoint["container_id"], {"schema_version": 1,
                "secret": connection.credential.get_secret_value(), "request": payload, "incarnation": endpoint["incarnation"],
                "network_profile": self.policy.profile_id if self.policy is not None else None,
                "document_html": "<!doctype html><title>Private workspace</title><body contenteditable=true style='min-height:500px;font:24px sans-serif'></body>"}, mode="initialize")
        elif marker.strip() != b"1":
            raise ResourceOutcomeUnknown("worker_bootstrap_state_unknown")
        await self.boundary.invoke(["exec", endpoint["container_id"], "python", "-c",
            "import pathlib,time; p=pathlib.Path('/tmp/kestrel-browser/worker.sock'); "
            "\nfor _ in range(200):\n if p.exists(): break\n time.sleep(.05)\nelse: raise SystemExit(1)"])
        await self._admission(connection)
        session = await self.status(request.scope, request.session_id)
        if (session.lease_id != request.lease_id or session.expires_at != request.expires_at
                or session.scope != request.scope):
            raise ResourceConflict("session_request_conflict")
        return session

    async def _rpc(self, session_id, operation, arguments, *, priority=False, cleanup=False, current=None):
        connection = self._connections.get(session_id)
        if connection is None:
            raise ResourceNotFound("session_connection_unavailable")
        endpoint = await self._bound(connection, cleanup=cleanup)
        if current is not None:
            current()
        identifier = uuid4().hex
        response = await self.boundary.rpc(endpoint["container_id"], {"schema_version": 1,
            "secret": connection.credential.get_secret_value(), "id": identifier,
            "operation": operation, "arguments": arguments}, mode="priority" if priority else None, current=current)
        if current is not None:
            current()
        if response.get("id") != identifier:
            raise ResourceOutcomeUnknown("worker_response_identity_unknown")
        if "error" in response:
            reason = response["error"]
            if reason in {"stale_control", "stale_observation", "action_conflict", "control_conflict", "control_pending", "workspace_conflict"}:
                raise ResourceConflict(reason)
            if reason in {"unsupported_request", "unsupported_action", "unsupported_control", "workspace_unsupported"}:
                raise ResourceUnsupported(reason)
            if reason in {"control_outcome_unknown", "transport_outcome_unknown", "workspace_outcome_unknown"}:
                raise ResourceOutcomeUnknown(reason)
            raise ResourceNotFound("session_operation_unavailable")
        if "result" not in response:
            raise ResourceOutcomeUnknown("worker_response_unknown")
        return response["result"]

    async def status(self, scope, session_id):
        return _record(ComputerSession, await self._rpc(session_id, "status", {
            "scope": scope.model_dump(mode="json"), "session_id": session_id}, priority=True))

    async def observe(self, session):
        result = await self._rpc(session.session_id, "observe", {"session": session.model_dump(mode="json")})
        try:
            data = base64.b64decode(result["image_base64"], validate=True)
            return ScreenObservation(frame=_record(ScreenFrame, result["frame"]), image=SecretBytes(data))
        except (ValueError, KeyError, TypeError, ValidationError):
            raise ResourceOutcomeUnknown("invalid_worker_observation") from None

    async def _pump(self, session, action_id):
        gateway = HTTPGateway(self.policy)
        while True:
            jobs = await self._rpc(session.session_id, "network_poll", {
                "session": session.model_dump(mode="json"), "action_id": action_id})
            if not isinstance(jobs, list) or len(jobs) > 16:
                raise ResourceOutcomeUnknown("invalid_network_job_batch")
            for job in jobs:
                if (job.get("action_id") != action_id or job.get("session_id") != session.session_id
                        or job.get("control_epoch") != session.control_epoch):
                    raise ResourceOutcomeUnknown("network_job_identity_mismatch")
                # The callback binds the original RPC's authority. It runs on
                # this owned task and remains held through reply and cleanup.
                async with self.egress_guard(session, action_id) as current:
                    current()
                    try:
                        result = await gateway.fetch(job, current)
                        reply = {"outcome": "completed", "status": result["status"], "headers": result["headers"],
                                 "body_base64": base64.b64encode(result["body"]).decode()}
                    except EgressDenied:
                        reply = {"outcome": "denied"}
                    except EgressUnknown:
                        reply = {"outcome": "unknown"}
                    current()
                    await self._rpc(session.session_id, "network_reply", {
                        "session": session.model_dump(mode="json"), "action_id": action_id,
                        "job_id": job["job_id"], "reply": reply}, current=current)
                    current()
            await asyncio.sleep(.03)

    async def _stop_pump(self, session_id):
        entry = self._pumps.pop(session_id, None)
        if entry is not None:
            entry[1].cancel()
            await asyncio.gather(entry[1], return_exceptions=True)
        return entry

    async def act(self, session, request):
        if self.egress_guard is None:
            return await self._perform_act(session, request)
        async def borrow():
            async with self.egress_guard(session, request.action_id) as current:
                current()
                return await self._perform_act(session, request, current=current)
        if session.session_id in self._borrowers:
            raise ResourceConflict("action_borrower_already_active")
        borrower = asyncio.create_task(borrow())
        self._borrowers[session.session_id] = borrower
        try:
            return await borrower
        finally:
            self._borrowers.pop(session.session_id, None)
            if not borrower.done():
                borrower.cancel()
            await asyncio.gather(borrower, return_exceptions=True)

    async def _perform_act(self, session, request, *, current=None):
        if session.session_id in self._pumps:
            raise ResourceConflict("network_action_already_active")
        if self.policy is not None and request.action.kind != "wait":
            if self.egress_guard is None:
                raise ResourceUnsupported("active_host_egress_guard_required")
            self._pumps[session.session_id] = (request.action_id, asyncio.create_task(self._pump(session, request.action_id)))
        try:
            return _record(ActionReceipt, await self._rpc(session.session_id, "act", {
                "session": session.model_dump(mode="json"), "request": request.model_dump(mode="json")}, current=current))
        finally:
            entry = await self._stop_pump(session.session_id)
            if entry is not None:
                try:
                    await self._rpc(session.session_id, "network_cancel", {
                        "session": session.model_dump(mode="json"), "action_id": entry[0]}, priority=True)
                except (ResourceNotFound, ResourceConflict, ResourceOutcomeUnknown):
                    # The original action result remains completed/unknown as
                    # recorded. Cleanup uncertainty cannot authorize another input.
                    pass

    async def action_status(self, session, action_id):
        return _record(ActionReceipt, await self._rpc(session.session_id, "action_status", {
            "session": session.model_dump(mode="json"), "action_id": action_id}, priority=True))

    async def control(self, session, request):
        borrower = self._borrowers.get(session.session_id)
        if borrower is not None and borrower is not asyncio.current_task():
            borrower.cancel()
            await asyncio.gather(borrower, return_exceptions=True)
        entry = await self._stop_pump(session.session_id)
        if entry is not None:
            await self._rpc(session.session_id, "network_cancel", {
                "session": session.model_dump(mode="json"), "action_id": entry[0]}, priority=True)
        return _record(ComputerSession, await self._rpc(session.session_id, "control", {
            "session": session.model_dump(mode="json"), "request": request.model_dump(mode="json")}, priority=True))


class DockerWorkspaceTransfer:
    """Negotiated bounded extension; the host ArtifactService remains the broker."""

    def __init__(self, driver):
        self.driver = driver

    async def _call(self, session, operation, arguments):
        return await self.driver._rpc(session.session_id, "workspace", {
            "session": session.model_dump(mode="json"), "operation": operation, "arguments": arguments},
            cleanup=operation in {"release_export", "abort_import"})

    async def prepare_export(self, session, request):
        from kestrel_sdk.workspaces import WorkspaceSnapshot
        return _record(WorkspaceSnapshot, await self._call(session, "prepare_export", {"request": request.model_dump(mode="json")}))

    async def read_export(self, session, snapshot_id, path):
        status = await self.status(session, snapshot_id)
        if status is None or status.snapshot is None:
            raise ResourceNotFound("workspace_snapshot_unavailable")
        file = next((file for file in status.snapshot.manifest.files if file.path == path), None)
        if file is None:
            raise ResourceNotFound("workspace_file_not_selected")
        result = await self._call(session, "read_export", {"snapshot_id": snapshot_id, "path": path})
        try:
            content = base64.b64decode(result["content_base64"], validate=True)
            if len(content) != file.size_bytes or hashlib.sha256(content).hexdigest() != file.sha256:
                raise ValueError
            return SecretBytes(content)
        except (KeyError, ValueError, TypeError):
            raise ResourceOutcomeUnknown("workspace_snapshot_integrity_failure") from None

    async def release_export(self, session, transfer_id):
        from kestrel_sdk.workspaces import WorkspaceTransferStatus
        return _record(WorkspaceTransferStatus, await self._call(session, "release_export", {"transfer_id": transfer_id}))

    async def prepare_import(self, session, request):
        from kestrel_sdk.workspaces import WorkspaceTransferStatus
        return _record(WorkspaceTransferStatus, await self._call(session, "prepare_import", {"request": request.model_dump(mode="json")}))

    async def write_import_file(self, session, transfer_id, file, content):
        from kestrel_sdk.workspaces import WorkspaceTransferStatus
        value = content.get_secret_value()
        if len(value) > 128 * 1024 or len(value) != file.size_bytes or hashlib.sha256(value).hexdigest() != file.sha256:
            raise ResourceUnsupported("workspace_file_exceeds_transport_or_integrity_bounds")
        return _record(WorkspaceTransferStatus, await self._call(session, "write_import_file", {
            "transfer_id": transfer_id, "file": file.model_dump(mode="json"),
            "content_base64": base64.b64encode(value).decode()}))

    async def commit_import(self, session, transfer_id):
        from kestrel_sdk.workspaces import WorkspaceMaterialization
        return _record(WorkspaceMaterialization, await self._call(session, "commit_import", {"transfer_id": transfer_id}))

    async def abort_import(self, session, transfer_id):
        from kestrel_sdk.workspaces import WorkspaceTransferStatus
        return _record(WorkspaceTransferStatus, await self._call(session, "abort_import", {"transfer_id": transfer_id}))

    async def status(self, session, transfer_id):
        from kestrel_sdk.workspaces import WorkspaceTransferStatus
        result = await self._call(session, "status", {"transfer_id": transfer_id})
        return _record(WorkspaceTransferStatus, result) if result is not None else None


def environment_provider_factory(*, settings: Mapping[str, object]):
    if set(settings) - {"docker_socket", "image", "credential_key", "profile_id", "network_policy"}:
        raise ResourceUnsupported("unknown_private_docker_settings")
    try:
        return DockerEnvironmentProvider(**settings)
    except TypeError:
        raise ResourceUnsupported("invalid_private_docker_settings") from None


def computer_driver_factory(*, settings: Mapping[str, object]):
    if "docker_socket" not in settings or set(settings) - {"docker_socket", "egress_guard", "network_policy"}:
        raise ResourceUnsupported("invalid_private_docker_settings")
    return DockerComputerDriver(**settings)
