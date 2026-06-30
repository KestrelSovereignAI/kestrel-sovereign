"""Reference example: a feature CLI group registered via ``kestrel_sovereign.cli``.

This is a *worked example* for feature authors, not a wired-in core command. A
real feature package ships its own equivalent of this module and points an entry
point at :func:`add_example_subparser`::

    [project.entry-points."kestrel_sovereign.cli"]
    example = "my_package.cli:add_example_subparser"

It then gains a working ``kestrel example status`` subcommand with **zero** edits
to ``cli.py`` — :func:`kestrel_sovereign.cli_extensions.register_cli_extensions`
discovers and registers it.

Out-of-process / thin-client pattern
------------------------------------

The CLI runs host-side, out-of-process from the live agent, so this command
**cannot** touch in-process feature state. Instead it is a thin client over the
feature's own router endpoint — here ``GET /api/example/status`` — exactly like
``cli_features`` queries ``/api/features``. The router is the real primitive;
the CLI is a convenience wrapper over its HTTP surface.
"""

from __future__ import annotations

import argparse
import sys


def add_example_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``example`` command group (the ``add_<name>_subparser``
    convention discovered by the CLI extension system)."""
    parser = subparsers.add_parser(
        "example",
        help="Reference feature CLI group (kestrel_sovereign.cli extension example)",
    )
    sub = parser.add_subparsers(dest="example_command")

    status_p = sub.add_parser(
        "status", help="Show the feature's status via its router endpoint"
    )
    status_p.add_argument(
        "--agent", default=None,
        help="Agent name to route to (defaults to the host's first agent)",
    )

    # Register the dispatch handler the same way cli_serve / cli_embeddings do —
    # the CLI dispatcher drains ``args._handler`` for extension commands.
    parser.set_defaults(_handler=run)


def run(args: argparse.Namespace) -> int:
    """Dispatch ``kestrel example <subcommand>``."""
    cmd = getattr(args, "example_command", None)
    if cmd == "status":
        return _cmd_status(args)
    print("usage: kestrel example {status}", file=sys.stderr)
    return 1


def _cmd_status(args: argparse.Namespace) -> int:
    """Thin client over the feature router: GET /api/example/status."""
    from kestrel_sovereign import cli

    endpoint = _resolve_agent_endpoint(cli, getattr(args, "agent", None))
    if endpoint is None:
        print("No running agent found to query.", file=sys.stderr)
        return 1
    base_url, api_key = endpoint

    import httpx

    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        resp = httpx.get(f"{base_url}/api/example/status", headers=headers, timeout=5.0)
    except httpx.RequestError as exc:
        print(f"Could not reach the agent at {base_url}: {exc}", file=sys.stderr)
        return 1
    if resp.status_code != 200:
        print(f"Feature router returned HTTP {resp.status_code}", file=sys.stderr)
        return 1
    print(resp.text)
    return 0


def _resolve_agent_endpoint(cli, agent_name):
    """Resolve (base_url, api_key) for a running agent via host config.

    Mirrors how ``cli_features`` discovers the live agent HTTP API — host config
    + the public auth key — never in-process state.
    """
    from kestrel_sovereign.multi_agent.config import MULTI_AGENT_CONFIG_FILENAME

    project_dir = cli._get_project_dir()
    multi_agent = cli.MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)
    local_agents = multi_agent.get_local_agents()

    if agent_name is None:
        agent_name = next(iter(local_agents), None)
    cfg = local_agents.get(agent_name) if agent_name else None
    if cfg is None:
        return None
    return cli._detect_running_agent_server(agent_name, cfg, multi_agent)
