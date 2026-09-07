"""The operator lane for host lifecycle verbs (#3233).

``kestrel create|start|terminate|restart|update`` provision, control, kill
and re-image agents, the host and the fleet. They ran on local process
access alone: whoever could exec the ``kestrel`` binary was the operator.
An agent with the Amendment IX host-shell capability can exec it, and its
shell consent — ASK, AUTO, a scoped approval — is operational consent, not
authority over a peer, the host, or the fleet (the two-axis doctrine, the
#3143 audit). Restart re-enters terminate; update ends in restart.

The lane is a presented credential, verified by the process acted on:

1. At dispatch, the environment the verb was **invoked with** must carry
   the host's stable sovereign key, ``KESTREL_API_KEY``, and it must match
   the project's ``.env`` when that file exists. An agent's shell subprocess
   never has it — ``sanitized_subprocess_env`` is an allowlist that drops
   every Kestrel variable (#3148) — while the host process (which loads
   ``.env`` at boot), the restart coordinator's detached ``kestrel restart``
   and its update profile inherit it. An operator exports it for the shell
   session that runs the verb; the CLI does not read it from ``.env`` on
   the invoker's behalf, because that is how an agent's re-entry became the
   operator.
2. At the signal, ``ProcessManager.kill_process`` — the one chokepoint every
   lifecycle kill passes through — refuses to signal a PID that has not
   **vouched**: the CLI connects to one of the PID's own listening sockets,
   confirms through the kernel's connection table that the *accepted* end
   of that very connection belongs to the PID, and only then asks
   ``GET /api/auth/vouch?nonce=…``; only a holder of the presented key can
   answer with the right HMAC. The answer is bound to the connection, not
   to an address: a second process squatting the same port on a more
   specific bind answers for itself and is credited to itself, never to
   the PID under judgement. The key is never transmitted and a fresh nonce
   defeats replay. A project directory, its ``.env``, its
   ``multi_agent.toml``, the pid files, the port number and the bind are
   all things the invoker can write; the process that accepted the
   connection is not. A process identity ``(pid, start time)`` vouches
   once: a host mid-graceful-shutdown has already closed its listeners
   when the SIGKILL escalation arrives, and must not be un-vouched by
   that. A PID that never vouches — not a Kestrel host, a host under
   another key, a hung host — is left alone and reported; the verb
   continues with its other targets and refuses as a whole. A hung host
   is stopped by PID, by hand.

What the lane does not do, on purpose: it does not infer the invoker from
a marker, a TTY, or process ancestry (all changeable by an evading shell),
and it is not a secret-custody boundary — same-user file access is the
filesystem and shell policy's job. Export the key per session, not in a
login profile an agent's ``bash -l`` would source.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple

from kestrel_sovereign.security.sovereign_key import (
    normalize_sovereign_api_key,
    sovereign_key_fingerprint,
)

logger = logging.getLogger(__name__)

LIFECYCLE_VERBS = frozenset({"create", "start", "terminate", "restart", "update"})
# The verbs that signal, restart, or re-image processes already running.
LIVE_PROCESS_VERBS = frozenset({"terminate", "restart", "update"})

VOUCH_PATH = "/api/auth/vouch"
_VOUCH_DOMAIN = b"kestrel/operator-lane/vouch/v1\x00"
NONCE_HEX_LENGTH = 32

_HOW = (
    "The operator lane is the invoking environment carrying the host's stable "
    "sovereign KESTREL_API_KEY: export it for this shell session (not in a "
    "login profile) and retry. An agent's shell never carries it, by design."
)


class OperatorLaneRefused(RuntimeError):
    """A lifecycle verb tried to signal a process that did not vouch."""


# --------------------------------------------------------------------------
# The challenge, shared by the host route and the CLI's verifier
# --------------------------------------------------------------------------


def vouch_response(key: str, nonce_hex: str) -> str:
    """The answer a host holding ``key`` gives to ``nonce_hex``."""
    secret = normalize_sovereign_api_key(key).encode("utf-8")
    return hmac.new(secret, _VOUCH_DOMAIN + bytes.fromhex(nonce_hex), hashlib.sha256).hexdigest()


def is_valid_nonce(nonce_hex: str) -> bool:
    if not isinstance(nonce_hex, str) or len(nonce_hex) != NONCE_HEX_LENGTH:
        return False
    try:
        bytes.fromhex(nonce_hex)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------
# Dispatch: the presented credential
# --------------------------------------------------------------------------


def configured_sovereign_key(project_dir: Path) -> str:
    """The project's stable sovereign key from ``.env``, normalized; "" if none."""
    env_file = project_dir / ".env"
    if not env_file.exists():
        return ""
    from dotenv import dotenv_values

    return normalize_sovereign_api_key(dotenv_values(env_file).get("KESTREL_API_KEY") or "")


def operator_lane_refusal(
    verb: str, project_dir: Path, invoking_env: Mapping[str, str]
) -> Optional[str]:
    """Why ``kestrel <verb>`` may not run from this environment, or None.

    ``invoking_env`` must be the process environment as it was at CLI entry,
    captured before any code path can load ``.env`` into ``os.environ``.
    Admission here only opens the lane; each process the verb signals is
    still vouched at the signal.
    """
    presented = normalize_sovereign_api_key(invoking_env.get("KESTREL_API_KEY") or "")
    if not presented:
        return (
            f"kestrel {verb} refused: no sovereign credential in the invoking "
            f"environment. {_HOW}"
        )
    configured = configured_sovereign_key(project_dir)
    if configured and not hmac.compare_digest(
        sovereign_key_fingerprint(presented), sovereign_key_fingerprint(configured)
    ):
        return (
            f"kestrel {verb} refused: the presented KESTREL_API_KEY is not the "
            f"host's sovereign key. {_HOW}"
        )
    if not configured and verb not in LIVE_PROCESS_VERBS:
        # Nothing will be signalled (so nothing can vouch) and there is no
        # file to compare with: the lane cannot be established. A host whose
        # key lives elsewhere (an EnvironmentFile, a container secret) puts
        # the same key in <project>/.env as well.
        return (
            f"kestrel {verb} refused: the project at {project_dir} has no stable "
            "KESTREL_API_KEY in .env to verify the presented credential against. "
            "The lane's reference is that file: a host whose key lives elsewhere "
            "puts the same key there too; a fresh project runs `kestrel setup keys` first."
        )
    return None


# --------------------------------------------------------------------------
# The signal: the vouch
# --------------------------------------------------------------------------

_presented_key: Optional[str] = None
# Identities — (pid, start time) — that vouched while the lane was armed.
_vouched: set = set()


def activate_operator_lane(presented: str) -> None:
    """Arm the vouch for this process: every ``kill_process`` from now on
    must be answered by the target. Called by the CLI once dispatch admits a
    lifecycle verb; never by the host itself, whose own process management
    is not an invoker's re-entry."""
    global _presented_key
    _presented_key = normalize_sovereign_api_key(presented)
    _vouched.clear()


def deactivate_operator_lane() -> None:
    """Disarm. The CLI does this when the verb returns, so arming is scoped to
    one verb's execution — an in-process caller of ``main()`` (a test, an
    embedding tool) must not leave every later ``kill_process`` gated."""
    global _presented_key
    _presented_key = None
    _vouched.clear()


def operator_lane_is_active() -> bool:
    return bool(_presented_key)


def _listeners(pid: int) -> List[Tuple[str, int, int]]:
    """``(dial address, port, family)`` for every socket ``pid`` listens on.

    A wildcard bind is dialled through loopback of its family; a specific
    address — loopback or not — is dialled as reported. The answer is bound
    to the key, not to an interface, so a host bound to ``10.0.0.5`` can
    vouch too.
    """
    import socket

    try:
        import psutil
    except ImportError:
        return []
    endpoints: List[Tuple[str, int, int]] = []
    try:
        for conn in psutil.Process(pid).net_connections(kind="inet"):
            if conn.status != psutil.CONN_LISTEN or not conn.laddr:
                continue
            ip, port = conn.laddr.ip, conn.laddr.port
            if ip == "0.0.0.0":
                endpoints.append(("127.0.0.1", port, socket.AF_INET))
            elif ip == "::":
                endpoints.append(("::1", port, socket.AF_INET6))
            else:
                endpoints.append((ip, port, conn.family))
    except (psutil.Error, OSError):
        return []
    return sorted(set(endpoints))


def _connection_belongs_to(pid: int, local: Tuple[str, int]) -> bool:
    """Whether ``pid`` owns the accepted end of the connection whose client
    side is ``local`` — the kernel's connection table, not an address."""
    try:
        import psutil
    except ImportError:
        return False
    local_ip, local_port = local[0], local[1]
    try:
        for conn in psutil.Process(pid).net_connections(kind="inet"):
            if (
                conn.status == psutil.CONN_ESTABLISHED
                and conn.raddr
                and conn.raddr.port == local_port
                and conn.raddr.ip == local_ip
            ):
                return True
    except (psutil.Error, OSError):
        return False
    return False


def _challenge_over_own_connection(pid: int, address: str, port: int, family: int, key: str) -> bool:
    """One challenge: connect, prove the acceptor is ``pid``, then ask."""
    import http.client
    import json
    import socket

    nonce = secrets.token_hex(NONCE_HEX_LENGTH // 2)
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        sock.connect((address, port))
        local = sock.getsockname()
        # The kernel completes the handshake before the process calls
        # accept(); until then the connection is not yet a descriptor of the
        # PID and the table does not attribute it. Give the acceptor a moment
        # — a live server accepts within milliseconds — before concluding
        # that the accepted end belongs to someone else.
        deadline = time.monotonic() + 1.5
        while not _connection_belongs_to(pid, (local[0], local[1])):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        conn = http.client.HTTPConnection(address, port, timeout=2.0)
        conn.sock = sock
        conn.request("GET", f"{VOUCH_PATH}?nonce={nonce}", headers={"Host": "localhost"})
        response = conn.getresponse()
        if response.status != 200:
            return False
        try:
            answer = json.loads(response.read().decode("utf-8")).get("vouch")
        except (ValueError, AttributeError):
            return False
        return isinstance(answer, str) and hmac.compare_digest(answer, vouch_response(key, nonce))
    except (OSError, http.client.HTTPException):
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def vouch_pid(pid: int, presented: str) -> bool:
    """Whether ``pid`` answers the nonce challenge with ``presented``'s HMAC
    over a connection the kernel attributes to ``pid``.

    Asked of each listening socket the process owns; one correct answer
    over a connection it accepted vouches. Nothing about the port number,
    the bind address, or a pid file enters the decision — only the
    process's own answer on its own connection.
    """
    expected_key = normalize_sovereign_api_key(presented)
    for address, port, family in _listeners(pid):
        if _challenge_over_own_connection(pid, address, port, family, expected_key):
            return True
    return False


def require_vouched_pid(pid: int, started_at: Optional[float] = None) -> None:
    """Refuse to signal ``pid`` unless its identity vouched. No-op when the
    lane is inactive.

    The identity is ``(pid, start time)``; it vouches once per armed lane.
    A graceful shutdown closes the listeners before the process exits, so
    the SIGKILL escalation that follows a SIGTERM could never vouch afresh —
    and a process that vouched a moment ago is the same process until its
    start time says otherwise.
    """
    if not operator_lane_is_active():
        return
    if started_at is None:
        from kestrel_sovereign.multi_agent.process_manager import ProcessManager

        started_at = ProcessManager.process_start_time(pid)
    identity = (pid, round(started_at, 3) if started_at is not None else None)
    if identity in _vouched:
        return
    if vouch_pid(pid, _presented_key or ""):
        _vouched.add(identity)
        return
    raise OperatorLaneRefused(
        f"refusing to signal PID {pid}: it did not vouch for the presented "
        "KESTREL_API_KEY on any connection it accepted (not a Kestrel host, "
        "a host under another key, or a hung host). A hung Kestrel host is "
        "stopped by PID, by hand."
    )
