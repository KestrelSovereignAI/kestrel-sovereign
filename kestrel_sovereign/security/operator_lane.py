"""The operator lane for host lifecycle verbs (#3233).

``kestrel create|start|terminate|restart|update`` provision, control, kill
and re-image agents, the host and the fleet. They ran on local process
access alone: whoever could exec the ``kestrel`` binary was the operator.
An agent with the Amendment IX host-shell capability can exec it, and its
shell consent — ASK, AUTO, a scoped approval — is operational consent, not
authority over a peer, the host, or the fleet (the two-axis doctrine, the
#3143 audit). Restart re-enters terminate; update ends in restart.

The lane is a presented credential, verified by the thing acted on:

1. The environment the verb was **invoked with** must carry the host's
   stable sovereign key, ``KESTREL_API_KEY``. An agent's shell subprocess
   never has it — ``sanitized_subprocess_env`` is an allowlist that drops
   every Kestrel variable (#3148) — while the host process (which loads the
   project ``.env`` at boot), the restart coordinator's detached
   ``kestrel restart``, and its update profile inherit it. An operator
   exports it for the shell session that runs the verb. The CLI does not
   read it from ``.env`` on the invoker's behalf; that is how an agent's
   re-entry became the operator.
2. It must match the project's configured key in ``.env`` (by fingerprint).
3. For the verbs that reach running processes — terminate, restart, update
   — every port the verb would touch that is **held** must authenticate the
   presented key as its sovereign credential, or the verb refuses. This is
   the part a caller cannot choose: a project directory, its ``.env`` and
   its ``multi_agent.toml`` are files the invoker can write, but the host
   listening on the port is not. Without it, a caller could point a fresh
   directory's ``multi_agent.toml`` at the real host's port and have
   ``terminate --force`` reap "an orphan" that is the live fleet. A held
   port whose holder does not speak Kestrel, or does not accept the key, is
   left alone — a hung host is stopped by hand, by PID.

What the lane does not do, on purpose: it does not infer the invoker from
a marker, a TTY, or process ancestry (all changeable by an evading shell),
and it is not a secret-custody boundary — same-user file access is the
filesystem and shell policy's job. Export the key per session, not in a
login profile an agent's ``bash -l`` would source.
"""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple

from kestrel_sovereign.security.sovereign_key import (
    normalize_sovereign_api_key,
    sovereign_key_fingerprint,
)

LIFECYCLE_VERBS = frozenset({"create", "start", "terminate", "restart", "update"})
# The verbs that signal, restart, or re-image processes already running.
LIVE_PROCESS_VERBS = frozenset({"terminate", "restart", "update"})

_HOW = (
    "The operator lane is the invoking environment carrying the host's stable "
    "sovereign KESTREL_API_KEY: export it for this shell session (not in a "
    "login profile) and retry. An agent's shell never carries it, by design."
)


def configured_sovereign_key(project_dir: Path) -> str:
    """The project's stable sovereign key from ``.env``, normalized; "" if none."""
    env_file = project_dir / ".env"
    if not env_file.exists():
        return ""
    from dotenv import dotenv_values

    return normalize_sovereign_api_key(dotenv_values(env_file).get("KESTREL_API_KEY") or "")


def ports_the_verb_touches(project_dir: Path, agent_name: Optional[str]) -> Tuple[str, List[int]]:
    """``(bind, ports)`` a lifecycle verb in ``project_dir`` may signal or restart.

    The host port always; with an agent name that agent's port, otherwise
    every local agent's. Read from the project's ``multi_agent.toml``, with
    the same defaults the verbs themselves use when it is absent.
    """
    from kestrel_sovereign.multi_agent.config import (
        MULTI_AGENT_CONFIG_FILENAME,
        MultiAgentConfig,
    )

    config = MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)
    ports = [config.host.port]
    local = config.get_local_agents()
    if agent_name:
        if agent_name in local:
            ports.append(local[agent_name].port)
    else:
        ports.extend(cfg.port for cfg in local.values())
    return config.host.bind, sorted(set(ports))


def held_port_admits(port: int, presented: str, *, bind: str = "0.0.0.0") -> Optional[bool]:
    """Whether the process holding ``port`` accepts ``presented`` as sovereign.

    ``None`` when nothing holds the port. ``True`` only when a Kestrel server
    answers an authenticated host route with 200 for the key; a listener
    that is not a Kestrel server, or that rejects the key, is ``False``.
    """
    import httpx

    from kestrel_sovereign.multi_agent.process_manager import ProcessManager

    if not ProcessManager.is_port_in_use(port, bind):
        return None
    try:
        response = httpx.get(
            f"http://localhost:{port}/api/agents",
            headers={"X-API-Key": presented},
            timeout=2.0,
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _refuse_held_ports(verb: str, project_dir: Path, presented: str, agent_name: Optional[str]) -> Optional[str]:
    bind, ports = ports_the_verb_touches(project_dir, agent_name)
    for port in ports:
        admitted = held_port_admits(port, presented, bind=bind)
        if admitted is False:
            return (
                f"kestrel {verb} refused: a process holds :{port} and did not accept "
                "the presented KESTREL_API_KEY as this host's sovereign credential; "
                "not touching it. If it is a hung Kestrel host, stop it by PID. "
                f"{_HOW}"
            )
    return None


def operator_lane_refusal(
    verb: str,
    project_dir: Path,
    invoking_env: Mapping[str, str],
    *,
    agent_name: Optional[str] = None,
) -> Optional[str]:
    """Why ``kestrel <verb>`` may not run from this environment, or None.

    ``invoking_env`` must be the process environment as it was at CLI entry,
    captured before any code path can load ``.env`` into ``os.environ``.
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
    if verb in LIVE_PROCESS_VERBS:
        refusal = _refuse_held_ports(verb, project_dir, presented, agent_name)
        if refusal is not None:
            return refusal
        if not configured:
            # Nothing live vouched for the key and the project has no file
            # to compare against: the lane cannot be established. A host
            # whose key lives elsewhere (an EnvironmentFile, a container
            # secret) places the same key in <project>/.env as well.
            bind, ports = ports_the_verb_touches(project_dir, agent_name)
            if all(held_port_admits(p, presented, bind=bind) is None for p in ports):
                return (
                    f"kestrel {verb} refused: nothing is running to verify the "
                    f"presented credential against, and the project at {project_dir} "
                    "has no KESTREL_API_KEY in .env to compare it with. The lane's "
                    "reference is that file: put the host's stable key there as well."
                )
        return None
    if not configured:
        return (
            f"kestrel {verb} refused: the project at {project_dir} has no stable "
            "KESTREL_API_KEY in .env to verify the presented credential against. "
            "The lane's reference is that file: a host whose key lives elsewhere "
            "(an EnvironmentFile, a container secret) puts the same key there too; "
            "a fresh project runs `kestrel setup keys` first."
        )
    return None
