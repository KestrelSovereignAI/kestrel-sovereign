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

``kestrel deploy secrets sync``
    Push values from ``.env`` into GCP Secret Manager for every
    ``[profiles.*.secrets]`` entry in ``deploy_config.toml``. Replaces
    ``scripts/cloudrun/setup_secrets.sh`` for Windows-friendly operators.
    See ``kestrel deploy secrets sync --help``.

``kestrel deploy build``
    Build (and push) the cloudrun docker images. Replaces
    ``scripts/cloudrun/build.sh`` (multi-arch buildx) and
    ``scripts/cloudrun/build_multi_agent.sh`` (single-arch fallback).
    See ``kestrel deploy build --help``.

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
import os
import sys
from pathlib import Path
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
            "list, health, secrets, build"
        ),
    )
    deploy_p.add_argument(
        "profile",
        nargs="?",
        default=None,
        help=(
            "Profile name when ``target`` is a subcommand "
            "(e.g. ``kestrel deploy teardown dev``); subverb when "
            "``target`` is ``secrets`` (currently only ``sync``). "
            "Ignored when ``target`` is itself a profile name or "
            "``build``."
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

    # ---- ``kestrel deploy secrets sync`` flags --------------------------
    # We share one parser for all subcommands (rather than nested
    # subparsers) to keep ``kestrel deploy <profile>`` ergonomic. The
    # secrets-only flags below are inert when ``target`` isn't
    # ``secrets``. ``--profile`` would collide with the positional
    # ``profile``, so its dest is ``secrets_profile``.
    deploy_p.add_argument(
        "--profile",
        dest="secrets_profile",
        type=str,
        default=None,
        help=(
            "[secrets sync] Limit secrets sync to one profile's "
            "[profiles.<name>.secrets] section. Default: all profiles, "
            "deduped."
        ),
    )
    deploy_p.add_argument(
        "--env-file",
        dest="env_file",
        type=str,
        default=None,
        help=(
            "[secrets sync] Path to the .env file with secret values "
            "(default: ./.env, relative to the directory you ran kestrel from)."
        ),
    )
    deploy_p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=(
            "[secrets sync] Print what would happen without mutating "
            "Secret Manager. With every secret skipped (e.g. an empty "
            ".env), no GCP client is constructed at all."
        ),
    )

    # ---- ``kestrel deploy build`` flags ---------------------------------
    # These flags share the same parser (rather than nested subparsers)
    # for the same ergonomic reason as the secrets flags above. They
    # are inert when ``target`` isn't ``build``.
    deploy_p.add_argument(
        "--target",
        dest="build_target",
        type=str,
        default=None,
        help=(
            "[build] Build only one image by name "
            "(e.g. ``kestrel`` or ``kestrel-multi_agent``). Default: "
            "build all DEFAULT_TARGETS."
        ),
    )
    deploy_p.add_argument(
        "--no-push",
        dest="no_push",
        action="store_true",
        help=(
            "[build] Skip the docker push step — produces local-only "
            "images. Default: push to gcr.io after building."
        ),
    )
    deploy_p.add_argument(
        "--no-multi-arch",
        dest="no_multi_arch",
        action="store_true",
        help=(
            "[build] Use plain ``docker build`` instead of "
            "``docker buildx build`` (single-arch local). Mirrors the "
            "legacy ``build_multi_agent.sh`` flow for operators who "
            "can't run buildx."
        ),
    )
    deploy_p.add_argument(
        "--platforms",
        dest="build_platforms",
        type=str,
        default=None,
        help=(
            "[build] Comma-separated buildx platforms (e.g. "
            "``linux/amd64,linux/arm64``). Default: "
            "``linux/amd64,linux/arm64`` matching scripts/cloudrun/build.sh."
        ),
    )


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

# Subcommand keywords. Anything else in the ``target`` slot is treated as
# a profile name and triggers a deploy.
_SUBCOMMANDS = {"status", "teardown", "logs", "list", "health", "secrets", "build"}


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
# ``kestrel deploy secrets sync`` — port of scripts/cloudrun/setup_secrets.sh
# ---------------------------------------------------------------------------

# Pretty rendering for one SecretSyncResult line.
_SECRETS_ACTION_PADDING = len("dry-run-update")


def _render_secret_result(result) -> str:
    """``"created   kestrel-openai-key <- OPENAI_API_KEY"``."""
    action = result.action.ljust(_SECRETS_ACTION_PADDING)
    line = f"{action} {result.secret_name} <- {result.env_var or '(unknown)'}"
    if result.detail:
        line += f"  ({result.detail})"
    return line


def _load_deploy_config_for_secrets() -> Optional[Dict[str, Any]]:
    """Load deploy_config.toml for the secrets path.

    Returns None and prints a friendly error if the file is missing or
    malformed. We intentionally don't reuse ``DeployManager`` here —
    secrets sync is independent of provider/session machinery, and
    constructing a DeployManager would force the operator to have a
    valid GCP_PROJECT_ID set even if they only want a dry-run preview.
    """
    import toml

    # Resolve from the operator's CWD — same semantics as
    # ``kestrel_sovereign.config.load_config``, which is what the agent
    # ``!deploy`` tool already uses. An installed/global ``kestrel`` CLI
    # invoked from the operator's project directory works; running from
    # an unrelated CWD errors clearly. Codex round 3 flagged the
    # in-repo-subdirectory case; round 9 flagged that resolving to
    # ``_get_project_dir()`` broke the installed-CLI case. CWD matches
    # the existing convention and works for both via the same rule.
    config_path = Path("deploy_config.toml")
    if not config_path.exists():
        print(
            f"error: deploy_config.toml not found at {config_path.resolve()}. "
            f"Run from the project directory containing deploy_config.toml, "
            f"or copy deploy_config.toml.example and configure it.",
            file=sys.stderr,
        )
        return None

    try:
        with config_path.open("r", encoding="utf-8") as f:
            return toml.load(f)
    except Exception as e:
        print(f"error: failed to parse {config_path}: {e}", file=sys.stderr)
        return None


_PLACEHOLDER_PROJECT = "your-gcp-project-id"


def _is_real_project_id(value: Optional[str]) -> bool:
    """Filter out unset / example placeholder values."""
    return bool(value) and value != _PLACEHOLDER_PROJECT


def _resolve_project_id(
    config: Optional[Dict[str, Any]] = None,
    profile: Optional[str] = None,
) -> Optional[str]:
    """Find the GCP project ID for the secrets sync.

    Order of precedence:

    1. ``GCP_PROJECT_ID`` env var (matches the bash script).
    2. ``[profiles.<profile>].gcp_project_id`` from ``deploy_config.toml``
       — only consulted when ``--profile`` was given. Cloud Run profiles
       can override the manager value (DeployManagerCore._load_profiles).
       Codex review on PR #1057 caught that we ignored this earlier.
    3. ``[manager].gcp_project_id``.

    With ``profile=None`` (the default all-profiles scan), if any Cloud
    Run profile sets a ``gcp_project_id`` that disagrees with the
    manager's, refuse to pick a winner — print an error pointing the
    operator at ``--profile``.

    Returns None on error (the caller should propagate exit 1).
    """
    env_value = os.getenv("GCP_PROJECT_ID")
    if env_value:
        return env_value

    if config is None:
        config = _load_deploy_config_for_secrets()
        if config is None:
            return None

    manager_section = config.get("manager", {}) or {}
    manager_value = manager_section.get("gcp_project_id")

    profiles = config.get("profiles", {}) or {}

    if profile is not None:
        prof_data = profiles.get(profile, {}) or {}
        prof_value = prof_data.get("gcp_project_id")
        if _is_real_project_id(prof_value):
            return prof_value
        if _is_real_project_id(manager_value):
            return manager_value
        print(
            f"error: GCP project ID not set for profile '{profile}'. "
            f"Either export GCP_PROJECT_ID, set [manager].gcp_project_id, "
            f"or set [profiles.{profile}].gcp_project_id in "
            f"deploy_config.toml.",
            file=sys.stderr,
        )
        return None

    # Default scan: only Cloud Run profiles that *actually contribute* a
    # Secret Manager ref need a project ID. A profile with no [secrets]
    # section, or only ``${...}`` placeholders, won't drive any sync work
    # and shouldn't be able to fail the preflight (codex review on PR
    # #1057 v7 → v8). Inline the eligibility check so this stays in sync
    # with derive_secret_mapping's filter.
    from kestrel_sovereign.features.deploy.secrets import _is_secret_manager_ref

    def _has_syncable_secret(prof_data: Optional[Dict[str, Any]]) -> bool:
        secrets = (prof_data or {}).get("secrets", {}) or {}
        return any(_is_secret_manager_ref(v) for v in secrets.values())

    cloudrun_profiles = [
        (name, data) for name, data in profiles.items()
        if (data or {}).get("provider", "cloudrun").lower() in {"cloudrun", "cloud_run"}
        and _has_syncable_secret(data)
    ]

    effective: Dict[str, Optional[str]] = {}
    for name, data in cloudrun_profiles:
        prof_value = (data or {}).get("gcp_project_id")
        if _is_real_project_id(prof_value):
            effective[name] = prof_value
        elif _is_real_project_id(manager_value):
            effective[name] = manager_value
        else:
            effective[name] = None

    if cloudrun_profiles:
        unset = sorted(n for n, v in effective.items() if v is None)
        if unset:
            print(
                "error: Cloud Run profile(s) have no GCP project ID "
                f"({', '.join(unset)}). Either set [manager].gcp_project_id, "
                f"set [profiles.<name>.gcp_project_id] for those profiles, "
                f"export GCP_PROJECT_ID, or use `--profile <name>` to sync a "
                f"single configured profile.",
                file=sys.stderr,
            )
            return None

        distinct = sorted({v for v in effective.values() if v is not None})
        if len(distinct) > 1:
            print(
                "error: Cloud Run profiles target different GCP projects "
                f"({', '.join(distinct)}). Use `--profile <name>` to sync "
                "one profile's secrets, or align gcp_project_id across "
                "profiles.",
                file=sys.stderr,
            )
            return None

        return distinct[0]

    if _is_real_project_id(manager_value):
        return manager_value

    print(
        "error: GCP project ID not set. Either export GCP_PROJECT_ID or "
        "set [manager].gcp_project_id in deploy_config.toml.",
        file=sys.stderr,
    )
    return None


def _cmd_deploy_secrets(args) -> int:
    """``kestrel deploy secrets <subverb>`` dispatcher.

    Currently only ``sync`` is implemented — future verbs (``list``,
    ``rotate``) would land here. The subverb is in ``args.profile``
    because the existing CLI shape uses ``profile`` as the second
    positional slot regardless of what the first slot means.
    """
    subverb = args.profile  # second positional = secrets subverb

    if subverb is None:
        print(
            "Usage: kestrel deploy secrets sync [--profile NAME] "
            "[--env-file PATH] [--dry-run] [--json]",
            file=sys.stderr,
        )
        return 1

    if subverb != "sync":
        print(
            f"error: unknown secrets subverb '{subverb}'. "
            f"Available: sync.",
            file=sys.stderr,
        )
        return 1

    return _cmd_deploy_secrets_sync(args)


def _cmd_deploy_secrets_sync(args) -> int:
    """Run the actual sync — port of scripts/cloudrun/setup_secrets.sh.

    Always prints a per-secret result line and a summary footer so the
    operator can see exactly what happened. Exit code is 0 unless any
    result has ``action == "error"``.
    """
    # Lazy import keeps test_cli_deploy.py's existing tests independent
    # of the secrets module surface.
    from kestrel_sovereign.features.deploy.secrets import (
        ACTION_CREATED,
        ACTION_DRY_RUN_CREATE,
        ACTION_DRY_RUN_UPDATE,
        ACTION_ERROR,
        ACTION_SKIPPED,
        ACTION_UPDATED,
        sync_all_secrets,
    )

    config = _load_deploy_config_for_secrets()
    if config is None:
        return 1

    # ``--profile`` (dest=secrets_profile) is the secrets-namespace flag —
    # the second positional ``profile`` is the subverb name (``sync``).
    profile = args.secrets_profile

    project_id = _resolve_project_id(config=config, profile=profile)
    if project_id is None:
        return 1

    # Default ``.env`` resolves to CWD (same convention as
    # ``deploy_config.toml`` above — see _load_deploy_config_for_secrets
    # for the rationale). An explicit ``--env-file`` value is used as-
    # given (caller-CWD relative).
    env_path = Path(args.env_file) if args.env_file else Path(".env")

    try:
        results = sync_all_secrets(
            config,
            env_path,
            project_id,
            profile=profile,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        # derive_secret_mapping raises KeyError for unknown profile name.
        print(f"error: {e.args[0] if e.args else e}", file=sys.stderr)
        return 1
    except ValueError as e:
        # Conflicting secret name across profiles.
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        # SDK construction failure (e.g. no ADC creds) shows up here. We
        # surface the message rather than a traceback because operators
        # care about "fix your auth" not "look at line 412".
        print(f"error: secrets sync failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        # Render dataclasses as plain dicts for stable JSON.
        payload = {
            "success": all(r.action != ACTION_ERROR for r in results),
            "project_id": project_id,
            "profile": profile,
            "dry_run": args.dry_run,
            "results": [
                {
                    "secret_name": r.secret_name,
                    "env_var": r.env_var,
                    "action": r.action,
                    "detail": r.detail,
                }
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0 if payload["success"] else 1

    # Pretty output.
    if not results:
        print(
            "no Secret Manager refs found in deploy_config "
            f"(profile={profile or 'all'}); nothing to sync."
        )
        return 0

    for r in results:
        print(_render_secret_result(r))

    counts = {
        ACTION_CREATED: 0,
        ACTION_UPDATED: 0,
        ACTION_SKIPPED: 0,
        ACTION_ERROR: 0,
        ACTION_DRY_RUN_CREATE: 0,
        ACTION_DRY_RUN_UPDATE: 0,
    }
    for r in results:
        counts[r.action] = counts.get(r.action, 0) + 1

    summary_parts = []
    if counts[ACTION_CREATED]:
        summary_parts.append(f"{counts[ACTION_CREATED]} created")
    if counts[ACTION_UPDATED]:
        summary_parts.append(f"{counts[ACTION_UPDATED]} updated")
    if counts[ACTION_DRY_RUN_CREATE]:
        summary_parts.append(f"{counts[ACTION_DRY_RUN_CREATE]} would-create")
    if counts[ACTION_DRY_RUN_UPDATE]:
        summary_parts.append(f"{counts[ACTION_DRY_RUN_UPDATE]} would-update")
    summary_parts.append(f"{counts[ACTION_SKIPPED]} skipped")
    summary_parts.append(f"{counts[ACTION_ERROR]} errors")
    print(", ".join(summary_parts))

    return 0 if counts[ACTION_ERROR] == 0 else 1


# ---------------------------------------------------------------------------
# ``kestrel deploy build`` — port of scripts/cloudrun/build.sh
# ---------------------------------------------------------------------------

def _cmd_deploy_build(args) -> int:
    """Run the cloudrun image build — port of ``scripts/cloudrun/build.sh``.

    Builds (and, by default, pushes) the canonical
    :data:`DEFAULT_TARGETS`. ``--target NAME`` narrows to one image;
    ``--no-push`` keeps the build local; ``--no-multi-arch`` falls back
    to plain ``docker build`` (mirroring ``build_multi_agent.sh``);
    ``--platforms`` overrides the buildx platform list.

    Project-ID resolution reuses ``_resolve_project_id`` (CWD
    ``deploy_config.toml`` → ``[manager].gcp_project_id``); the build
    isn't profile-scoped, so we call with ``profile=None``. A missing
    or placeholder project ID prints a friendly error and returns 1.

    GitHub token resolution: env first, then ``gh auth token``. Missing
    token is a warning, not an error — Dockerfiles without private
    repo deps build fine without it.
    """
    # Lazy import — keeps existing test_cli_deploy.py tests independent
    # of the build module surface (same idiom as secrets sync above).
    from kestrel_sovereign.features.deploy.build import (
        DEFAULT_TARGETS,
        BuildError,
        build_all,
        resolve_github_token,
    )

    # Project ID: env wins outright (matches build.sh exactly — the
    # bash script uses ``${GCP_PROJECT_ID:?...}``, never reads any
    # config). Fall back to deploy_config.toml so operators don't have
    # to export the var if it's already in their config. The build is
    # project-wide so we always pass profile=None.
    env_project = os.getenv("GCP_PROJECT_ID")
    if env_project:
        project_id: Optional[str] = env_project
    else:
        # No env var — try deploy_config.toml. If config is missing and
        # the operator hasn't exported the var, that's a hard error
        # (matches the bash script's ``${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}``
        # but with the friendly Python config-driven fallback).
        config = _load_deploy_config_for_secrets()
        if config is None:
            # _load_deploy_config_for_secrets already printed the error.
            return 1
        project_id = _resolve_project_id(config=config, profile=None)
        if project_id is None:
            return 1

    # Filter targets by --target name if given.
    targets = list(DEFAULT_TARGETS)
    if args.build_target:
        matching = [t for t in targets if t.image_name == args.build_target]
        if not matching:
            available = ", ".join(t.image_name for t in DEFAULT_TARGETS)
            print(
                f"error: unknown build target '{args.build_target}'. "
                f"Available: {available}.",
                file=sys.stderr,
            )
            return 1
        targets = matching

    push = not args.no_push
    multi_arch = not args.no_multi_arch
    # The bash script accepted the tag as a positional arg
    # (``./scripts/cloudrun/build.sh v1.2.3``). The CLI's existing
    # second positional ``profile`` slot — used as a profile name for
    # ``kestrel deploy <profile>`` and as a subverb for
    # ``kestrel deploy secrets sync`` — also receives that value when
    # ``target == "build"``. Honor it as the tag so users typing the
    # same shape they used in bash don't silently push ``:latest``.
    # Codex review on PR #1060 caught the silent fallback. ``--tag``
    # still wins if both are present (explicit beats positional).
    if args.tag and args.tag != "latest":
        tag = args.tag
    elif getattr(args, "profile", None):
        tag = args.profile
    else:
        tag = args.tag  # default "latest"

    # Platforms: comma-split if provided. Defaults differ between push
    # and no-push: ``--push`` defaults to multi-arch (matches build.sh);
    # ``--no-push`` defaults to single-arch (linux/amd64) because
    # buildx ``--load`` doesn't support multi-arch and silently failing
    # the user's "local-only" workflow is a worse default than picking
    # a sensible single platform. Codex review on PR #1060 caught the
    # original "default rejects --no-push by default" footgun.
    if args.build_platforms:
        platforms = tuple(
            p.strip() for p in args.build_platforms.split(",") if p.strip()
        )
        if not platforms:
            print(
                "error: --platforms must be a non-empty comma-separated list "
                "(e.g. linux/amd64,linux/arm64).",
                file=sys.stderr,
            )
            return 1
    elif push or not multi_arch:
        platforms = ("linux/amd64", "linux/arm64")
    else:
        # --no-push (with multi-arch buildx) → must be single-platform.
        platforms = ("linux/amd64",)

    github_token = resolve_github_token()
    if github_token is None:
        # Match the bash script wording verbatim — operators who recognize
        # this from setup_secrets.sh / build.sh shouldn't have to wonder
        # if it means something different here.
        print(
            "WARNING: No GITHUB_TOKEN found. Build may fail if private "
            "repos are dependencies.",
            file=sys.stderr,
        )

    # Pre-build banner so the operator sees what's about to happen
    # before docker buildx prints its own (long) progress lines.
    if not args.json:
        for tgt in targets:
            ref_base = f"gcr.io/{project_id}/{tgt.image_name}"
            tags = f"{{{tag},latest}}" if tag != "latest" else "{latest}"
            print(
                f"building {tgt.image_name}: {tgt.dockerfile} -> {ref_base}:{tags}"
            )

    try:
        results = build_all(
            project_id=project_id,
            tag=tag,
            targets=targets,
            platforms=platforms,
            push=push,
            multi_arch=multi_arch,
            github_token=github_token,
        )
    except BuildError as e:
        # ``build_all`` attaches partial successes; render them so the
        # operator sees what got built before the failure.
        if args.json:
            payload = {
                "success": False,
                "project_id": project_id,
                "tag": tag,
                "error": {
                    "image_ref": e.image_ref,
                    "command": e.command,
                    "stderr": e.stderr,
                },
                "results": [
                    {
                        "image_name": r.target.image_name,
                        "image_refs": r.image_refs,
                        "pushed": r.pushed,
                        "duration_seconds": r.duration_seconds,
                    }
                    for r in e.partial_results
                ],
            }
            print(json.dumps(payload, indent=2))
        else:
            for r in e.partial_results:
                duration = (
                    f"{r.duration_seconds:.1f}s"
                    if r.duration_seconds is not None else "?s"
                )
                state = "pushed" if r.pushed else "local only"
                print(f"built {r.target.image_name} in {duration} ({state})")
            print(
                f"error: docker build failed for {e.image_ref}",
                file=sys.stderr,
            )
            if e.stderr:
                print(e.stderr, file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — surface as friendly error
        # Anything else — docker not installed, etc. The
        # ``_default_runner`` raises FileNotFoundError when docker
        # isn't on PATH; we don't want a traceback at the CLI boundary.
        if args.json:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        else:
            print(f"error: build failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "success": True,
            "project_id": project_id,
            "tag": tag,
            "results": [
                {
                    "image_name": r.target.image_name,
                    "image_refs": r.image_refs,
                    "pushed": r.pushed,
                    "duration_seconds": r.duration_seconds,
                }
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    # Pretty per-target line + summary footer.
    errors = 0  # by construction zero here, but symmetric with secrets
    for r in results:
        duration = (
            f"{r.duration_seconds:.1f}s" if r.duration_seconds is not None else "?s"
        )
        state = "pushed" if r.pushed else "local only"
        print(f"built {r.target.image_name} in {duration} ({state})")
    print(f"{len(results)} built, {errors} errors")
    return 0


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def cmd_deploy(args) -> int:
    """Top-level dispatcher for ``kestrel deploy ...``.

    Disambiguates positional ``target``: if it matches a known subcommand
    keyword (``status``, ``teardown``, ``logs``, ``list``, ``health``,
    ``secrets``) we route to that handler; otherwise we treat it as a
    profile name to deploy.
    """
    target = args.target

    if target is None:
        print(
            "Usage: kestrel deploy <profile> [--tag TAG]\n"
            "       kestrel deploy status\n"
            "       kestrel deploy teardown <profile>\n"
            "       kestrel deploy logs <profile> [--lines N]\n"
            "       kestrel deploy list\n"
            "       kestrel deploy health <profile>\n"
            "       kestrel deploy secrets sync [--profile NAME] [--env-file PATH] [--dry-run]\n"
            "       kestrel deploy build [--tag TAG] [--target NAME] [--no-push] [--no-multi-arch] [--platforms LIST]",
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
    if target == "secrets":
        return _cmd_deploy_secrets(args)
    if target == "build":
        return _cmd_deploy_build(args)

    # Anything else: treat as a profile name to deploy.
    return _cmd_deploy_profile(args, target)


__all__ = [
    "add_deploy_subcommands",
    "cmd_deploy",
]
