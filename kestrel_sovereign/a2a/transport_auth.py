"""Dedicated authentication lane for automatic local A2A transport.

The sovereign API key authorizes an operator. Automatic peer routing must
never receive that credential: doing so lets a peer turn transport admission
into operator authority at legacy HTTP views. This module owns the separate
transport credential and the deliberately small set of routes it may reach.
Principal authority for task reads and mutations is still established by the
signed A2A envelope inside those routes.
"""

from __future__ import annotations

import os
import re
import secrets
import stat
import tempfile
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from kestrel_sovereign.auth import normalize_api_key


A2A_TRANSPORT_KEY_ENV = "KESTREL_A2A_TRANSPORT_KEY"
A2A_TRANSPORT_KEY_HEADER = "X-Kestrel-A2A-Key"
A2A_TRANSPORT_KEY_FILE = ".kestrel-a2a-transport.key"
A2A_TRANSPORT_ONLY_ENV = "KESTREL_A2A_TRANSPORT_ONLY"
_MAX_TRANSPORT_KEY_BYTES = 4096
_SOVEREIGN_API_KEY_ENV = "KESTREL_API_KEY"

_ROUTED_A2A_PATH = re.compile(
    r"^(?:/api/agents/[^/]+)?/api/agent/"
    r"(?:invoke|tasks/send|tasks/.+/(?:read|cancel|subscribe))$"
)


class A2ATransportKeyError(RuntimeError):
    """The fleet transport credential could not be loaded safely."""


def _validate_transport_key(value: str, *, source: str) -> str:
    selected = value.strip()
    if not selected:
        raise A2ATransportKeyError(f"{source} is empty")
    if any(
        ord(character) < 0x20 or ord(character) == 0x7F
        for character in selected
    ):
        raise A2ATransportKeyError(f"{source} contains control characters")
    if any(ord(character) > 0x7E for character in selected):
        raise A2ATransportKeyError(
            f"{source} must contain only header-safe ASCII characters"
        )
    return selected


def _configured_sovereign_api_keys(
    environment: MutableMapping[str, str] | None,
) -> tuple[str, ...]:
    """Return every operator credential relevant to a child launch.

    The project environment may override the host's exported key for the child,
    but the host keeps serving with its own key. A transport credential must
    therefore be distinct from both authority domains.
    """

    configured_values = [os.environ.get(_SOVEREIGN_API_KEY_ENV)]
    if environment is not None:
        configured_values.append(environment.get(_SOVEREIGN_API_KEY_ENV))

    normalized: list[str] = []
    for configured in configured_values:
        if not isinstance(configured, str) or not configured:
            continue
        candidate = normalize_api_key(configured)
        if candidate is not None and candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def _read_transport_key(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise A2ATransportKeyError(
            "A2A transport key cannot be opened safely"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise A2ATransportKeyError("A2A transport key is not a regular file")
        if os.name == "posix" and (
            opened.st_uid != os.geteuid() or opened.st_mode & 0o077
        ):
            raise A2ATransportKeyError(
                "A2A transport key must be owned by the current user with mode 0600"
            )
        chunks: list[bytes] = []
        remaining = _MAX_TRANSPORT_KEY_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        material = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(material) > _MAX_TRANSPORT_KEY_BYTES:
        raise A2ATransportKeyError("A2A transport key is too large")
    try:
        decoded = material.decode("utf-8")
    except UnicodeDecodeError as error:
        raise A2ATransportKeyError("A2A transport key is not UTF-8") from error
    return _validate_transport_key(decoded, source="A2A transport key file")


def _load_or_create_transport_key(project_root: Path) -> str:
    """Load one project key, or publish a new owner-only key atomically."""

    path = project_root / A2A_TRANSPORT_KEY_FILE
    try:
        return _read_transport_key(path)
    except A2ATransportKeyError as read_error:
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise A2ATransportKeyError(
                "A2A transport key path cannot be inspected"
            ) from error
        else:
            raise read_error

    generated = secrets.token_urlsafe(32)
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{A2A_TRANSPORT_KEY_FILE}.",
            dir=project_root,
        )
    except OSError as error:
        raise A2ATransportKeyError(
            "A2A transport key cannot be staged in the Kestrel project"
        ) from error
    temporary = Path(temporary_name)
    try:
        material = generated.encode("utf-8") + b"\n"
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(material)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            return _read_transport_key(path)
        except OSError as error:
            raise A2ATransportKeyError(
                "A2A transport key cannot be published atomically"
            ) from error
        return generated
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def ensure_a2a_transport_key(
    environment: MutableMapping[str, str] | None = None,
    *,
    project_root: Path | None = None,
) -> str:
    """Return one durable fleet transport key, creating it when absent.

    ``ProcessManager`` passes its already-resolved child environment so an
    explicit project ``.env`` value retains the launcher's established
    precedence. When neither source supplies a value, an owner-only project
    key file keeps independently launched host and child processes on the same
    credential. The selected value is installed in both mappings without
    deriving it from the sovereign key.
    """

    exported = os.environ.get(A2A_TRANSPORT_KEY_ENV)
    child_value = (
        environment.get(A2A_TRANSPORT_KEY_ENV)
        if environment is not None
        else None
    )
    if isinstance(child_value, str) and child_value.strip():
        selected = _validate_transport_key(
            child_value,
            source=A2A_TRANSPORT_KEY_ENV,
        )
    elif isinstance(exported, str) and exported.strip():
        selected = _validate_transport_key(
            exported,
            source=A2A_TRANSPORT_KEY_ENV,
        )
    else:
        if project_root is None:
            from kestrel_sovereign.paths import project_dir

            project_root = project_dir()
        selected = _load_or_create_transport_key(Path(project_root).resolve())

    selected_bytes = selected.encode("utf-8")
    aliases_sovereign_key = any(
        secrets.compare_digest(selected_bytes, sovereign_key.encode("utf-8"))
        for sovereign_key in _configured_sovereign_api_keys(environment)
    )
    if aliases_sovereign_key:
        raise A2ATransportKeyError(
            "A2A transport key must be distinct from the sovereign API key"
        )

    if environment is not None:
        environment[A2A_TRANSPORT_KEY_ENV] = selected
    # Replace blanks as well as absent values. ``setdefault`` preserves an
    # empty ``.env.example`` binding and generated a different key on every
    # call, splitting peer routers from the auth middleware.
    os.environ[A2A_TRANSPORT_KEY_ENV] = selected
    return selected


def is_a2a_transport_path(method: str, path: str) -> bool:
    """Whether the peer credential may enter this HTTP route.

    Roster discovery is transport metadata. Every task read/mutation route
    admitted here performs its own signed-principal verification. Legacy GET
    task list/detail/subscription views are intentionally absent: those are
    operator/UI surfaces and remain on the sovereign lane.
    """

    normalized_method = method.upper()
    if normalized_method == "GET" and path == "/api/agents":
        return True
    return (
        normalized_method == "POST"
        and _ROUTED_A2A_PATH.fullmatch(path) is not None
    )


def is_a2a_transport_only_process(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Whether this process is a host-managed peer with no sovereign lane."""

    environ = os.environ if environment is None else environment
    return environ.get(A2A_TRANSPORT_ONLY_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


__all__ = [
    "A2A_TRANSPORT_KEY_ENV",
    "A2A_TRANSPORT_KEY_FILE",
    "A2A_TRANSPORT_KEY_HEADER",
    "A2A_TRANSPORT_ONLY_ENV",
    "A2ATransportKeyError",
    "ensure_a2a_transport_key",
    "is_a2a_transport_path",
    "is_a2a_transport_only_process",
]
