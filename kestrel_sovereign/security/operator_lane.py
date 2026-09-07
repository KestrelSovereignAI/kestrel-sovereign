"""The operator lane for host lifecycle verbs (#3233).

``kestrel create|start|terminate|restart|update`` provision, control, kill
and re-image agents, the host and the fleet. They ran on local process
access alone: whoever could exec the ``kestrel`` binary was the operator.
An agent with the Amendment IX host-shell capability can exec it, and its
shell consent — ASK, AUTO, a scoped approval — is operational consent, not
authority over a peer, the host, or the fleet (the two-axis doctrine, the
#3143 audit). Restart re-enters terminate; update ends in restart.

The lane is explicit: the environment the verb was **invoked with** must
carry the host's stable sovereign key, ``KESTREL_API_KEY``, matching the
one the project is configured with. That is the same credential every
sovereign HTTP route and the in-process restart authority already require;
it is presented, never inferred. An agent's shell subprocess never has it —
``sanitized_subprocess_env`` is an allowlist that drops every Kestrel
variable (#3148) — while the host process (which loads the project ``.env``
at boot), the restart coordinator's detached ``kestrel restart``, and its
update profile's ``kestrel feature sync`` inherit it. An operator exports it
for the shell session that runs the verb.

What the lane does not do, on purpose:

- It does not read the key from ``.env`` on the invoker's behalf. The verbs
  used to; that is exactly how an agent's re-entry became the operator.
  ``.env`` is only the reference the presented key is compared against.
- It does not infer the invoker from a marker, a TTY, or process ancestry.
  All three are local facts an evading shell can change; a credential the
  subprocess was never given cannot be un-stripped.
- It is not a secret-custody boundary. Same-user file access (an agent
  reading ``.env``, or a login shell that sources a profile the operator
  put the key in) is the filesystem and shell policy's job. Export the key
  per session, not in a login profile.
"""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Mapping, Optional

from kestrel_sovereign.security.sovereign_key import (
    normalize_sovereign_api_key,
    sovereign_key_fingerprint,
)

LIFECYCLE_VERBS = frozenset({"create", "start", "terminate", "restart", "update"})

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


def operator_lane_refusal(
    verb: str, project_dir: Path, invoking_env: Mapping[str, str]
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
    if not configured:
        return (
            f"kestrel {verb} refused: the project at {project_dir} has no stable "
            "KESTREL_API_KEY in .env to verify the presented credential against. "
            "Run `kestrel setup keys` first."
        )
    if hmac.compare_digest(
        sovereign_key_fingerprint(presented), sovereign_key_fingerprint(configured)
    ):
        return None
    return (
        f"kestrel {verb} refused: the presented KESTREL_API_KEY is not the "
        f"host's sovereign key. {_HOW}"
    )
