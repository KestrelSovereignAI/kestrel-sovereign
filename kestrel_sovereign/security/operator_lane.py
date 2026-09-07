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
   **vouched**: the CLI opens one of the PID's own listening loopback
   sockets and asks ``GET /api/auth/vouch?nonce=…``; only a Kestrel host
   holding the same stable key can answer with the right HMAC. The key is
   never transmitted, a fresh nonce defeats replay, and the answer is bound
   to the exact process about to die. A project directory, its ``.env``,
   its ``multi_agent.toml``, the pid files and the port number are all
   files and numbers the invoker can write; the process listening on the
   socket is not. A PID that does not vouch — a process that is not a
   Kestrel host, a host under another key, a hung host — is left alone,
   and the verb refuses. A hung host is stopped by PID, by hand.

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


def activate_operator_lane(presented: str) -> None:
    """Arm the vouch for this process: every ``kill_process`` from now on
    must be answered by the target. Called by the CLI once dispatch admits a
    lifecycle verb; never by the host itself, whose own process management
    is not an invoker's re-entry."""
    global _presented_key
    _presented_key = normalize_sovereign_api_key(presented)


def operator_lane_is_active() -> bool:
    return bool(_presented_key)


def _loopback_listeners(pid: int) -> List[Tuple[str, int]]:
    """``(host, port)`` loopback endpoints ``pid`` is listening on.

    A wildcard bind (``0.0.0.0`` / ``::``) is reached through loopback of the
    same family; a socket bound to a non-loopback interface only is not
    probed — it cannot be this host's operator-facing listener.
    """
    try:
        import psutil
    except ImportError:
        return []
    endpoints: List[Tuple[str, int]] = []
    try:
        for conn in psutil.Process(pid).net_connections(kind="inet"):
            if conn.status != psutil.CONN_LISTEN or not conn.laddr:
                continue
            ip, port = conn.laddr.ip, conn.laddr.port
            if ip in ("0.0.0.0", "127.0.0.1"):
                endpoints.append(("127.0.0.1", port))
            elif ip in ("::", "::1"):
                endpoints.append(("[::1]", port))
    except (psutil.Error, OSError):
        return []
    return sorted(set(endpoints))


def vouch_pid(pid: int, presented: str) -> bool:
    """Whether ``pid`` answers the nonce challenge with ``presented``'s HMAC.

    Asked of each listening loopback socket the process owns; one correct
    answer vouches. Nothing about the port number, the bind address, or a
    pid file enters the decision — only the process's own answer.
    """
    import httpx

    expected_key = normalize_sovereign_api_key(presented)
    for host, port in _loopback_listeners(pid):
        nonce = secrets.token_hex(NONCE_HEX_LENGTH // 2)
        try:
            response = httpx.get(
                f"http://{host}:{port}{VOUCH_PATH}",
                params={"nonce": nonce},
                timeout=2.0,
            )
        except httpx.HTTPError:
            continue
        if response.status_code != 200:
            continue
        try:
            answer = response.json().get("vouch")
        except ValueError:
            continue
        if isinstance(answer, str) and hmac.compare_digest(
            answer, vouch_response(expected_key, nonce)
        ):
            return True
    return False


def require_vouched_pid(pid: int) -> None:
    """Refuse to signal ``pid`` unless it vouched. No-op when the lane is inactive."""
    if not operator_lane_is_active():
        return
    if vouch_pid(pid, _presented_key or ""):
        return
    raise OperatorLaneRefused(
        f"refusing to signal PID {pid}: it did not vouch for the presented "
        "KESTREL_API_KEY on any of its loopback listeners (not a Kestrel host, "
        "a host under another key, or a hung host). Nothing was signalled. "
        "A hung Kestrel host is stopped by PID, by hand."
    )
