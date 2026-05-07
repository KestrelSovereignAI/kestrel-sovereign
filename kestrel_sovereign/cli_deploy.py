"""
``kestrel deploy`` CLI commands — sub-PR 1.1 of epic #1050 (bash-to-Python port).

Exposes the existing :class:`DeployManager` Python implementation as a
first-class CLI subcommand so Kestrel can be deployed to Cloud Run from
Windows hosts that don't have bash. The bash scripts under
``scripts/cloudrun/`` continue to work for now and are removed in
sub-PR 1.4 once ``deploy_config.toml`` reconciliation lands.

This module is a thin shell around
:class:`kestrel_sovereign.features.deploy.manager.DeployManager`. The
manager itself owns provider dispatch, session state, and health
verification; the CLI just bridges argparse → ``asyncio.run`` →
human-readable output.

Subcommands
-----------

``kestrel deploy <profile>``
    Deploy a profile. Optional ``--tag`` overrides the image tag
    (default ``latest``).

``kestrel deploy status``
    Show every active deployment session in this manager instance.

``kestrel deploy teardown <profile>``
    Delete the deployed service for a profile.

``kestrel deploy logs <profile>``
    Print the most recent log lines (default 100, override with
    ``--lines``).

``kestrel deploy list``
    List every deployment across every provider configured in profiles.

``kestrel deploy health <profile>``
    Health-check the deployed service for a profile.

Each subcommand returns process exit code 0 if the operation reports
``success=True``, 1 otherwise. Initialization failures (e.g. no
``GCP_PROJECT_ID``) print a friendly error and return 1; we do not
propagate the traceback past the CLI boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, Dict, Optional

from kestrel_sovereign.features.deploy.manager import DeployManager
from kestrel_sovereign.features.deploy.models import DeployManagerError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argparse subcommand wiring
# ---------------------------------------------------------------------------

def add_deploy_subcommands(subparsers: "argparse._SubParsersAction") -> None:
    """Register ``kestrel deploy {<profile>,status,teardown,logs,list,health}``.

    Called from :func:`kestrel_sovereign.cli.build_parser`. The dispatch
    is positional-first (``kestrel deploy dev``), with named subcommands
    (``status``, ``list``, etc.) handled by checking the first positional
    argument inside :func:`cmd_deploy`. Argparse's nested-subparsers
    flow would force ``kestrel deploy deploy dev`` which is awkward —
    we keep the ergonomic shape and dispatch by hand.
    """
    deploy_p = subparsers.add_parser(
        "deploy",
        help="Deploy and manage Kestrel on Cloud Run / Azure Container Apps",
    )
    # ``target`` is either a profile name (e.g. ``dev``) or a literal
    # subcommand (``status``, ``list``, ``teardown``, ``logs``, ``health``).
    # We disambiguate at handler time. nargs='?' lets ``kestrel deploy``
    # alone print help.
    deploy_p.add_argument(
        "target",
        nargs="?",
        default=None,
        help=(
            "Profile name (deploys it) OR one of: status, teardown, logs, "
            "list, health"
        ),
    )
    deploy_p.add_argument(
        "profile",
        nargs="?",
        default=None,
        help=(
            "Profile name when ``target`` is a subcommand "
            "(e.g. ``kestrel deploy teardown dev``). Ignored when "
            "``target`` is itself a profile name."
        ),
    )
    deploy_p.add_argument(
        "--tag",
        type=str,
        default="latest",
        help="Image tag to deploy (default: latest)",
    )
    deploy_p.add_argument(
        "--lines",
        type=int,
        default=100,
        help="Number of log lines for `kestrel deploy logs <profile>` (default: 100)",
    )
    deploy_p.add_argument(
        "--json",
        action="store_true",
        help="Print the raw result dict as JSON instead of a pretty summary",
    )


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

# Subcommand keywords. Anything else in the ``target`` slot is treated as
# a profile name and triggers a deploy.
_SUBCOMMANDS = {"status", "teardown", "logs", "list", "health"}


def _print_kv(result: Dict[str, Any], skip: Optional[set] = None) -> None:
    """Pretty-print a result dict as ``key: value`` lines.

    Nested dicts (like ``session``) get expanded one level. Skip keys
    listed in ``skip`` (used by ``logs`` to avoid dumping the log blob
    twice).
    """
    skip = skip or set()
    for key, value in result.items():
        if key in skip:
            continue
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            print(f"{key}: ({len(value)} item(s))")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")


def _print_list(result: Dict[str, Any]) -> None:
    """Pretty-print ``list_all_deployments`` output as a simple table."""
    deployments = result.get("deployments", [])
    if not deployments:
        print("No deployments found.")
        return

    # Provider list_deployments outputs are heterogeneous: CloudRunProvider
    # emits {name, status, url, created}; AzureContainerProvider emits a
    # similar shape with `name`. Older callers (and tests) used `service`,
    # so accept either. Fall back across the same field family.
    headers = ["service", "provider", "status", "url"]
    print("  ".join(h.upper().ljust(20) for h in headers))
    print("  ".join("-" * 20 for _ in headers))
    for dep in deployments:
        row_values = [
            dep.get("name") or dep.get("service") or "-",
            dep.get("provider", "-"),
            dep.get("status", "-"),
            dep.get("url", "-"),
        ]
        row = [str(v)[:20].ljust(20) for v in row_values]
        print("  ".join(row))


# ---------------------------------------------------------------------------
# Manager bootstrap
# ---------------------------------------------------------------------------

def _build_manager() -> Optional[DeployManager]:
    """Construct a :class:`DeployManager`, returning None on init failure.

    Init failures (missing config, missing ``GCP_PROJECT_ID``, etc.)
    print a friendly message and return None so the caller can return
    exit code 1 without a traceback.
    """
    try:
        return DeployManager()
    except DeployManagerError as e:
        print(f"error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        # Anything else (FileNotFoundError on deploy_config.toml,
        # malformed TOML, etc.) — surface as a friendly error too.
        # We don't want to swallow programming bugs forever, but the
        # CLI boundary is the place where traceback noise hurts most.
        print(f"error: failed to initialize deploy manager: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _resolve_profile_name(args, *, subcommand: str) -> Optional[str]:
    """For ``kestrel deploy <subcommand> <profile>``, the profile lands
    in ``args.profile`` (because ``target`` ate the subcommand). Return
    it or print a usage error and yield None.
    """
    profile = args.profile
    if not profile:
        print(
            f"error: ``kestrel deploy {subcommand}`` requires a profile name "
            f"(e.g. ``kestrel deploy {subcommand} dev``)",
            file=sys.stderr,
        )
        return None
    return profile


def _cmd_deploy_profile(args, profile_name: str) -> int:
    """Dispatch a profile deploy."""
    manager = _build_manager()
    if manager is None:
        return 1

    try:
        result = asyncio.run(manager.deploy_profile(profile_name, args.tag))
    except DeployManagerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_kv(result)

    return 0 if result.get("success") else 1


def _cmd_deploy_status(args) -> int:
    """``kestrel deploy status`` — show what is actually deployed.

    The agent path tracks deployments via in-memory ``_sessions`` for the
    lifetime of the agent process. CLI invocations are short-lived and
    sessionless, so we query the providers directly via
    ``list_all_deployments`` — that is the persistent source of truth.
    """
    manager = _build_manager()
    if manager is None:
        return 1

    result = asyncio.run(manager.list_all_deployments())

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("success") else 1

    if not result.get("success"):
        print(f"error: {result.get('error', 'list_all_deployments failed')}")
        return 1

    deployments = result.get("deployments", [])
    if not deployments:
        print("No active deployments.")
        return 0

    print(f"active_deployments: {len(deployments)}")
    _print_list(result)
    return 0


def _cmd_deploy_teardown(args) -> int:
    profile_name = _resolve_profile_name(args, subcommand="teardown")
    if profile_name is None:
        return 1

    manager = _build_manager()
    if manager is None:
        return 1

    result = asyncio.run(manager.teardown_profile(profile_name))

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_kv(result)

    return 0 if result.get("success") else 1


def _cmd_deploy_logs(args) -> int:
    profile_name = _resolve_profile_name(args, subcommand="logs")
    if profile_name is None:
        return 1

    manager = _build_manager()
    if manager is None:
        return 1

    result = asyncio.run(manager.get_profile_logs(profile_name, lines=args.lines))

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("success") else 1

    if not result.get("success"):
        _print_kv(result)
        return 1

    # Print the actual log payload last so it's easy to redirect.
    _print_kv(result, skip={"logs"})
    print()
    print(result.get("logs", ""))
    return 0


def _cmd_deploy_list(args) -> int:
    manager = _build_manager()
    if manager is None:
        return 1

    result = asyncio.run(manager.list_all_deployments())

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif not result.get("success"):
        _print_kv(result)
    else:
        _print_list(result)

    return 0 if result.get("success") else 1


def _cmd_deploy_health(args) -> int:
    profile_name = _resolve_profile_name(args, subcommand="health")
    if profile_name is None:
        return 1

    manager = _build_manager()
    if manager is None:
        return 1

    result = asyncio.run(manager.health_check_profile(profile_name))

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_kv(result)

    return 0 if result.get("success") else 1


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def cmd_deploy(args) -> int:
    """Top-level dispatcher for ``kestrel deploy ...``.

    Disambiguates positional ``target``: if it matches a known subcommand
    keyword (``status``, ``teardown``, ``logs``, ``list``, ``health``)
    we route to that handler; otherwise we treat it as a profile name
    to deploy.
    """
    target = args.target

    if target is None:
        print(
            "Usage: kestrel deploy <profile> [--tag TAG]\n"
            "       kestrel deploy status\n"
            "       kestrel deploy teardown <profile>\n"
            "       kestrel deploy logs <profile> [--lines N]\n"
            "       kestrel deploy list\n"
            "       kestrel deploy health <profile>",
            file=sys.stderr,
        )
        return 1

    if target == "status":
        return _cmd_deploy_status(args)
    if target == "teardown":
        return _cmd_deploy_teardown(args)
    if target == "logs":
        return _cmd_deploy_logs(args)
    if target == "list":
        return _cmd_deploy_list(args)
    if target == "health":
        return _cmd_deploy_health(args)

    # Anything else: treat as a profile name to deploy.
    return _cmd_deploy_profile(args, target)


__all__ = [
    "add_deploy_subcommands",
    "cmd_deploy",
]
