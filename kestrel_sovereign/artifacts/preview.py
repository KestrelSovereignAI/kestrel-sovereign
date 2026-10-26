"""Bounded, local-only derivative jobs; never a source store or authority.

Each host app owns one admission pool across every agent. Slots are retained
when native creation/cleanup is unknown, so uncertainty cannot expand the
aggregate decoder budget. Only an immutable, locally installed image is used.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
import re
import struct
from uuid import uuid4
import zlib

from kestrel_sovereign._async_process import start_async_process, _terminate_and_await
from kestrel_sovereign._async_ownership import create_owned_task, await_owned_task, raise_owned_outcome
from kestrel_sdk.resources import ResourceError
from kestrel_sdk.operator import OperatorAuthorizationError

MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_PREVIEW_EDGE = 1024
MAX_PREVIEW_PIXELS = MAX_PREVIEW_EDGE ** 2
MAX_RAW_BYTES = MAX_PREVIEW_PIXELS * 3 + 12
_MEMORY = 512 * 1024 * 1024
_CPUS = 500_000_000
_PIDS = 16
_IMAGE = re.compile(r'(?:sha256:[a-f0-9]{64}|[a-zA-Z0-9./:_-]+@sha256:[a-f0-9]{64})\Z')


class ArtifactPreviewUnavailable(ResourceError):
    """No admitted local renderer, resource slot or verified cleanup exists."""


class ArtifactPreviewRefused(ResourceError):
    """Source or derivative exceeds the approved preview contract."""


@dataclass(frozen=True)
class ArtifactPreview:
    png: bytes
    width: int
    height: int

    def payload(self, ref):
        import base64
        return {'source': ref.model_dump(mode='json'), 'media_type': 'image/png',
                'width': self.width, 'height': self.height,
                'sha256': hashlib.sha256(self.png).hexdigest(),
                'content_base64': base64.b64encode(self.png).decode('ascii')}


def encode_preview(raw: bytes) -> ArtifactPreview:
    """Encode only a fixed RGB layout; the host never imports an image decoder."""
    if len(raw) < 12 or len(raw) > MAX_RAW_BYTES or raw[:4] != b'KAP1':
        raise ArtifactPreviewRefused('invalid renderer output')
    width, height = struct.unpack('!II', raw[4:12])
    if (not 1 <= width <= MAX_PREVIEW_EDGE or not 1 <= height <= MAX_PREVIEW_EDGE
            or width * height > MAX_PREVIEW_PIXELS or len(raw) != 12 + width * height * 3):
        raise ArtifactPreviewRefused('invalid renderer dimensions')
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    rows = b''.join(b'\0' + raw[12 + offset:12 + offset + width * 3]
                    for offset in range(0, width * height * 3, width * 3))
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))
    return ArtifactPreview(png, width, height)


@dataclass
class _Job:
    name: str
    endpoint: str | None = None
    identity: str | None = None
    image_id: str | None = None
    finished: bool = False
    creation_started: bool = False
    create_acknowledged: bool = False
    cleanup_confirmed: bool = False


class ArtifactPreviewPool:
    """Two process-tree cgroups per app, no unbounded waiter queue."""

    capacity = 2

    def __init__(self):
        self._jobs: dict[str, _Job] = {}

    @property
    def active_count(self):
        return len(self._jobs)

    @property
    def cleanup_pending(self):
        return tuple(job.name for job in self._jobs.values())

    def reserve(self):
        if len(self._jobs) >= self.capacity:
            raise ArtifactPreviewUnavailable('preview capacity is unavailable')
        job = _Job('kestrel-artifact-preview-' + uuid4().hex)
        self._jobs[job.name] = job
        return job

    def complete(self, job):
        if job.creation_started and not job.cleanup_confirmed:
            raise ArtifactPreviewUnavailable('preview cleanup remains unknown')
        self._jobs.pop(job.name, None)


def host_preview_pool(app) -> ArtifactPreviewPool:
    """Host attach seam; app state, rather than a process-global singleton."""
    pool = getattr(app.state, 'artifact_preview_pool', None)
    if pool is None:
        pool = ArtifactPreviewPool()
        app.state.artifact_preview_pool = pool
    if not isinstance(pool, ArtifactPreviewPool):
        raise ArtifactPreviewUnavailable('host preview pool is unavailable')
    return pool


async def _bounded_read(stream, maximum):
    result = bytearray()
    while data := await stream.read(min(65536, maximum + 1 - len(result))):
        result.extend(data)
        if len(result) > maximum:
            raise ArtifactPreviewRefused('renderer output exceeds its bound')
    return bytes(result)


class DockerArtifactRenderer:
    """One-shot sandbox for Pillow/Poppler; no host or remote decode fallback."""

    def __init__(self, image: str | None):
        self.image = image

    @staticmethod
    def _env():
        # Never forward deployment/provider credentials to a Docker CLI child.
        return {**{name: os.environ[name] for name in ('PATH', 'HOME', 'DOCKER_CONFIG') if name in os.environ},
                'LC_ALL': 'C'}

    async def _command(self, arguments, *, stdin=None, check=lambda: None, output_limit=65536, timeout=4):
        process = await start_async_process(['docker', *arguments], env=self._env(),
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        native_wait = create_owned_task(process.wait(), name='preview-cli-exit')
        async def write():
            if stdin is None:
                return
            try:
                for offset in range(0, len(stdin), 65536):
                    check(); process.stdin.write(stdin[offset:offset + 65536]); await process.stdin.drain(); check()
            finally:
                process.stdin.close()
        async def wait_native():
            # TaskGroup cancellation must not cancel CPython's physical process
            # waiter. Python 3.14 can otherwise publish into its cancelled Future.
            return await asyncio.shield(native_wait)
        async def collect():
            async with asyncio.TaskGroup() as group:
                group.create_task(write())
                output = group.create_task(_bounded_read(process.stdout, output_limit))
                group.create_task(_bounded_read(process.stderr, 8192))
                group.create_task(wait_native())
            return process.returncode, output.result()
        completion = create_owned_task(collect(), name='preview-cli-io')
        cancellation = None
        try:
            done, _ = await asyncio.wait({completion}, timeout=timeout)
        except asyncio.CancelledError as error:
            cancellation = error; done = set()
        # Reuse the host's native private-process-group closer. The completion
        # and physical waiter remain owned until that tree has actually exited.
        cleanup = create_owned_task(_terminate_and_await(process, completion,
            terminate_grace=.1, reap_timeout=2), name='preview-cli-cleanup')
        outcome = await await_owned_task(cleanup, cancellation)
        result = raise_owned_outcome(outcome, operation='preview CLI cleanup')
        if not done:
            raise TimeoutError('preview CLI deadline exceeded')
        return result

    async def _endpoint(self):
        explicit = os.environ.get('DOCKER_HOST')
        if explicit is not None:
            endpoint = explicit
        else:
            args = ['context', 'inspect']
            if os.environ.get('DOCKER_CONTEXT'):
                args.append(os.environ['DOCKER_CONTEXT'])
            status, value = await self._command([*args, '--format', '{{json .Endpoints.docker.Host}}'])
            if status != 0:
                raise ArtifactPreviewUnavailable('local Docker endpoint is unavailable')
            endpoint = json.loads(value)
        if not isinstance(endpoint, str) or not endpoint.startswith('unix:///') or '\n' in endpoint:
            raise ArtifactPreviewUnavailable('preview requires a local Unix Docker endpoint')
        return endpoint

    async def _verify_host(self, prefix):
        status, value = await self._command([*prefix, 'info', '--format', '{{json .}}'])
        if status != 0:
            raise ArtifactPreviewUnavailable('local Docker is unavailable')
        info = json.loads(value)
        if (info.get('OSType') != 'linux' or info.get('CgroupVersion') not in {'1', '2'}
                or not all(info.get(key) is True for key in ('MemoryLimit', 'SwapLimit', 'CpuCfsQuota', 'PidsLimit'))):
            raise ArtifactPreviewUnavailable('aggregate kernel limits are unavailable')

    @staticmethod
    def _verify_container(value):
        host, config = value['HostConfig'], value['Config']
        if (host.get('Memory') != _MEMORY or host.get('MemorySwap') != _MEMORY
                or host.get('NanoCpus') != _CPUS or host.get('PidsLimit') != _PIDS
                or host.get('NetworkMode') != 'none' or host.get('ReadonlyRootfs') is not True
                or host.get('IpcMode') != 'none' or host.get('PidMode') not in {'', 'private'}
                or host.get('CgroupnsMode') != 'private' or host.get('Runtime') != 'runc'
                or host.get('Privileged') is not False or host.get('CapDrop') != ['ALL']
                or host.get('SecurityOpt') != ['no-new-privileges']
                or host.get('Binds') or host.get('Devices') or host.get('DeviceRequests')
                or config.get('OpenStdin') is not True or config.get('User') != '65532:65532' or config.get('Volumes')
                or host.get('LogConfig', {}).get('Type') != 'none'
                or host.get('Tmpfs') != {'/tmp': 'rw,nosuid,nodev,noexec,size=64m,mode=1777'}
                or any(mount.get('Type') != 'tmpfs' or mount.get('Destination') != '/tmp' for mount in value.get('Mounts', []))):
            raise ArtifactPreviewUnavailable('renderer sandbox limits do not match')

    async def render(self, pool: ArtifactPreviewPool, content: bytes, media_type: str, check, *, job=None):
        if (not self.image or not _IMAGE.fullmatch(self.image)):
            raise ArtifactPreviewUnavailable('configure an immutable local renderer image')
        if not 0 < len(content) <= MAX_SOURCE_BYTES or media_type not in {'application/pdf', 'image/png', 'image/jpeg', 'image/gif', 'image/webp'}:
            raise ArtifactPreviewRefused('source does not have an approved preview')
        if job is None:
            job = pool.reserve()
        elif pool._jobs.get(job.name) is not job or job.finished:
            raise ArtifactPreviewUnavailable('preview admission does not match')
        failure = None
        try:
            async with asyncio.timeout(12):
                check(); job.endpoint = await self._endpoint(); prefix = ['--host', job.endpoint]
                await self._verify_host(prefix); check()
                status, image = await self._command([*prefix, 'image', 'inspect', self.image, '--format', '{{.Id}}'])
                if status != 0 or not re.fullmatch(rb'sha256:[a-f0-9]{64}\s*', image):
                    raise ArtifactPreviewUnavailable('pinned renderer image is not installed')
                job.image_id = image.decode().strip()
                job.creation_started = True
                status, identity = await self._command([*prefix, 'create', '--interactive', '--pull=never', '--name', job.name,
                    '--label', 'io.kestrel.artifact-preview.job=' + job.name,
                    '--network=none', '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                    '--memory=512m', '--memory-swap=512m', '--cpus=0.5', '--pids-limit=16',
                    '--user=65532:65532', '--ipc=none', '--cgroupns=private', '--runtime=runc', '--log-driver=none',
                    '--ulimit=nofile=32:32', '--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777',
                    job.image_id], output_limit=1024)
                if status != 0 or not re.fullmatch(rb'[a-f0-9]{64}\s*', identity):
                    raise ArtifactPreviewUnavailable('renderer creation outcome requires cleanup')
                job.create_acknowledged = True; job.identity = identity.decode().strip()
                status, detail = await self._command([*prefix, 'inspect', job.identity])
                if status != 0:
                    raise ArtifactPreviewUnavailable('renderer custody is unavailable')
                descriptor = json.loads(detail)[0]
                if descriptor.get('Id') != job.identity or descriptor.get('Name') != '/' + job.name:
                    raise ArtifactPreviewUnavailable('renderer identity does not match')
                self._verify_container(descriptor); check()
                request = json.dumps({'media_type': media_type, 'size': len(content)}, separators=(',', ':')).encode() + b'\n' + content
                status, raw = await self._command([*prefix, 'start', '--attach', '--interactive', job.identity],
                    stdin=request, check=check, output_limit=MAX_RAW_BYTES, timeout=9)
                check()
                if status != 0:
                    raise ArtifactPreviewRefused('renderer refused this source')
                result = encode_preview(raw)
        except ExceptionGroup as error:
            denied = error.subgroup(OperatorAuthorizationError)
            if denied is not None:
                # Preserve current-authority refusal across the I/O task group.
                failure = denied.exceptions[0]
                while isinstance(failure, BaseExceptionGroup):
                    failure = failure.exceptions[0]
            else:
                failure = ArtifactPreviewUnavailable('preview exceeds its bound')
        except (OperatorAuthorizationError, ArtifactPreviewUnavailable, ArtifactPreviewRefused) as error:
            failure = error
        except (OSError, ValueError, KeyError, IndexError, TypeError, TimeoutError) as error:
            failure = ArtifactPreviewUnavailable('preview is unavailable or exceeds its bound')
            failure.__cause__ = error
        finally:
            job.finished = True
            await self._cleanup(pool, job)
        if job.creation_started and not job.cleanup_confirmed:
            raise ArtifactPreviewUnavailable('preview cleanup remains unknown')
        if failure is not None:
            raise failure
        return result

    async def _cleanup(self, pool, job):
        # Repeated caller cancellation cannot abandon an accepted native rm.
        cleanup = create_owned_task(self._cleanup_native(pool, job), name='preview-container-cleanup')
        outcome = await await_owned_task(cleanup)
        raise_owned_outcome(outcome, operation='preview container cleanup')

    async def _cleanup_native(self, pool, job):
        if job.creation_started and not job.create_acknowledged:
            # A lost create ACK never authorizes blind capacity release. Exact
            # unique name+job label+immutable image can recover the native ID.
            try:
                status, detail = await self._command(['--host', job.endpoint, 'inspect', job.name])
                if status == 0:
                    value = json.loads(detail)[0]
                    if (value.get('Name') == '/' + job.name
                            and value.get('Config', {}).get('Image') == job.image_id
                            and value.get('Config', {}).get('Labels', {}).get('io.kestrel.artifact-preview.job') == job.name
                            and re.fullmatch(r'[a-f0-9]{64}', value.get('Id', ''))):
                        job.identity = value['Id']; job.create_acknowledged = True
            except (OSError, ValueError, TimeoutError, ExceptionGroup):
                pass
        if job.create_acknowledged:
            try:
                prefix = ['--host', job.endpoint]
                status, _ = await self._command([*prefix, 'rm', '--force', job.identity], timeout=4)
                if status != 0:
                    # A prior rm ACK can be lost. The acknowledged original ID
                    # cannot be created again, so a daemon list proves absence.
                    status, present = await self._command([*prefix, 'container', 'ls', '--all',
                        '--filter', 'id=' + job.identity, '--format', '{{.ID}}'], timeout=4)
                    job.cleanup_confirmed = status == 0 and not present.strip()
                else:
                    job.cleanup_confirmed = True
            except (OSError, TimeoutError, ExceptionGroup):
                pass  # Keep the slot until cleanup can actually be reconciled.
        if not job.creation_started or job.cleanup_confirmed:
            pool.complete(job)

    async def reconcile_cleanup(self, pool):
        for job in tuple(pool._jobs.values()):
            if job.finished:
                await self._cleanup(pool, job)
        return pool.cleanup_pending
