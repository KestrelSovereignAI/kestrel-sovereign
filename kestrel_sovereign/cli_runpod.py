"""``kestrel runpod {deploy,status,stop,kill}`` CLI command - sub-PR 4
of epic #1050 (bash-to-Python port of
``scripts/runpod/deploy_lora_trainer.sh``).

The bash predecessor was a 211-line wrapper around the existing
:class:`kestrel_cloud_runpod.manager.RunPodManager`. We import that
manager directly (no subprocess shell-out) and drive it via
``asyncio.run``. The bash-side argument parsing translates 1:1 to
argparse.

Subverbs
--------

- ``kestrel runpod deploy lora-trainer`` - retained only to explain retired
  fresh admission. Use the authorized catalog workflow; no provider/config
  lookup, model mapping, profile fallback or health probe is performed.
- ``kestrel runpod status [--profile training]`` - print active /
  resumable pod state.
- ``kestrel runpod stop [--profile training]`` - stop the session
  (pod stays around for resume).
- ``kestrel runpod kill [--profile training]`` - terminate the pod
  completely (no resume).

Historical resource access
--------------------------

- ``RUNPOD_API_KEY`` must be set; we error with a clear message if
  not, matching the bash predecessor.
- ``runpod_config.toml`` must exist at the project root with the
  named profile present.

Implementation notes
--------------------

The :class:`RunPodManager` lives in the optional
``kestrel-cloud-runpod`` extracted package (``#462``). We import it
lazily so operators who never use RunPod don't pay the import cost
(or take an unconditional hard-fail when the package isn't
installed). The error message points to the install path.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_PROFILE = "training"
_VALID_TARGETS = ("lora-trainer",)
_RUNPOD_CONFIG_FILE = "runpod_config.toml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Repo root - where ``runpod_config.toml`` is expected to live."""
    return Path(__file__).resolve().parent.parent


def _check_api_key() -> bool:
    """Return True if ``RUNPOD_API_KEY`` is set, else False (caller
    prints the error)."""
    return bool(os.environ.get("RUNPOD_API_KEY"))


def _print_missing_api_key() -> None:
    print(
        "error: RUNPOD_API_KEY environment variable is required.\n"
        "\n"
        "Set your RunPod API key:\n"
        "  export RUNPOD_API_KEY=\"your-runpod-key\"\n",
        file=sys.stderr,
    )


def _load_manager():
    """Lazily import and instantiate :class:`RunPodManager`.

    Returns the manager instance. Raises a friendly :class:`SystemExit`
    if the package isn't installed - the rest of the CLI continues to
    work for operators who don't need RunPod.
    """
    try:
        from kestrel_cloud_runpod.manager import RunPodManager
    except ImportError as e:
        print(
            "error: RunPod manager not available - install "
            "kestrel-cloud-runpod to enable `kestrel runpod`:\n"
            f"  uv pip install kestrel-cloud-runpod\n"
            f"  ({e})",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # The bash predecessor ran ``cd "$PROJECT_ROOT"`` before invoking
    # the manager so ``runpod_config.toml`` resolved correctly
    # regardless of where the operator typed the command. Mirror that:
    # construct the manager with CWD pinned to the project root so
    # operators with the installed ``kestrel`` CLI on PATH (running
    # from any directory) get the same lookup behavior. Codex review
    # on PR #1074 caught the regression. We restore the original CWD
    # immediately after construction; runtime calls never see the chdir.
    project_root = _project_root()
    if not (project_root / _RUNPOD_CONFIG_FILE).is_file():
        print(
            f"error: {_RUNPOD_CONFIG_FILE} not found at "
            f"{project_root / _RUNPOD_CONFIG_FILE}.\n"
            f"  Copy {_RUNPOD_CONFIG_FILE}.example and configure your "
            f"profiles.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    original_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        return RunPodManager()
    finally:
        os.chdir(original_cwd)


def _validate_profile(manager, profile_name: str) -> bool:
    """Confirm the profile exists in ``runpod_config.toml``. Prints a
    clear error listing the available profiles on miss."""
    profiles = getattr(manager, "profiles", None) or {}
    if profile_name in profiles:
        return True
    print(
        f"error: No '{profile_name}' profile found in {_RUNPOD_CONFIG_FILE}",
        file=sys.stderr,
    )
    if profiles:
        print(
            f"  Available profiles: {sorted(profiles.keys())}",
            file=sys.stderr,
        )
    else:
        cfg = _project_root() / _RUNPOD_CONFIG_FILE
        print(
            f"  No profiles loaded - check {cfg} exists and has "
            f"[profiles.<name>] sections.",
            file=sys.stderr,
        )
    return False


def _print_profile(profile) -> None:
    """Mirror the bash predecessor's profile-banner output."""
    print("Training Profile:")
    print(f"   GPU: {getattr(profile, 'gpu_type_id', '<unset>')}")
    print(f"   Image: {getattr(profile, 'image_name', '<unset>')}")
    print(
        f"   Template: "
        f"{getattr(profile, 'template_id', None) or 'None (public image)'}"
    )
    print(
        f"   Network Volume: "
        f"{getattr(profile, 'network_volume_id', None) or 'None (ephemeral)'}"
    )
    print(f"   Port: {getattr(profile, 'inference_port', '<unset>')}")
    print()


# ---------------------------------------------------------------------------
# Async actions - one per bash branch
# ---------------------------------------------------------------------------

async def _action_status(manager, profile_name: str) -> int:
    print("Checking pod status...")
    status = await manager.get_status()
    if status.get("active"):
        print("Pod is ACTIVE")
        print(f"   Pod ID: {status.get('pod_id')}")
        print(f"   Status: {status.get('status')}")
        print(f"   Base URL: {status.get('base_url')}")
        print(f"   Expires: {status.get('expires_at')}")
    else:
        print("No active pod")
        try:
            stopped = await manager.find_stopped_pod(
                "lora_training", profile_name,
            )
        except TypeError:
            # Older signature in legacy package builds.
            stopped = await manager.find_stopped_pod()
        if stopped:
            pod_id = stopped.get("id") if isinstance(stopped, dict) else None
            print(f"Found stopped pod {pod_id} - can resume")
    return 0


async def _action_stop(manager) -> int:
    print("Stopping pod (keeping for resume)...")
    result = await manager.stop_session()
    print(f"   Result: {result}")
    return 0


async def _action_kill(manager) -> int:
    print("Terminating pod completely...")
    status = await manager.get_status()
    pod_id = status.get("pod_id")
    if pod_id:
        await asyncio.to_thread(manager.provider.terminate_pod, pod_id)
        print("   Terminated")
        return 0
    print("   No active pod to terminate")
    return 0


# ---------------------------------------------------------------------------
# Subverb dispatchers
# ---------------------------------------------------------------------------

def _cmd_deploy(args) -> int:
    """Reject retired legacy deployment before configuration or provider effects."""
    target: Optional[str] = getattr(args, "target", None)
    if target not in _VALID_TARGETS:
        print(
            f"error: unknown deploy target {target!r}\n"
            f"  Valid targets: {list(_VALID_TARGETS)}",
            file=sys.stderr,
        )
        return 1

    from kestrel_sovereign.features.training.adapters.runpod_adapter import (
        reject_legacy_runpod_new_work,
    )
    from kestrel_sovereign.features.training.protocol import TrainingSubmissionError

    try:
        reject_legacy_runpod_new_work(TrainingSubmissionError)
    except TrainingSubmissionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _cmd_status(args) -> int:
    """``kestrel runpod status [--profile NAME]``."""
    if not _check_api_key():
        _print_missing_api_key()
        return 1
    profile_name: str = getattr(args, "profile", None) or _DEFAULT_PROFILE
    manager = _load_manager()
    if not _validate_profile(manager, profile_name):
        return 1
    _print_profile(manager.profiles[profile_name])
    return asyncio.run(_action_status(manager, profile_name))


def _cmd_stop(args) -> int:
    """``kestrel runpod stop [--profile NAME]``."""
    if not _check_api_key():
        _print_missing_api_key()
        return 1
    profile_name: str = getattr(args, "profile", None) or _DEFAULT_PROFILE
    manager = _load_manager()
    if not _validate_profile(manager, profile_name):
        return 1
    return asyncio.run(_action_stop(manager))


def _cmd_kill(args) -> int:
    """``kestrel runpod kill [--profile NAME]``."""
    if not _check_api_key():
        _print_missing_api_key()
        return 1
    profile_name: str = getattr(args, "profile", None) or _DEFAULT_PROFILE
    manager = _load_manager()
    if not _validate_profile(manager, profile_name):
        return 1
    return asyncio.run(_action_kill(manager))


# ---------------------------------------------------------------------------
# Argparse subcommand wiring
# ---------------------------------------------------------------------------

def add_runpod_subcommand(
    subparsers: "argparse._SubParsersAction",
) -> None:
    """Register ``kestrel runpod {deploy,status,stop,kill}`` under the
    parent subparsers."""
    runpod_p = subparsers.add_parser(
        "runpod",
        help="RunPod GPU lifecycle (LoRA training pods) - port of "
             "scripts/runpod/deploy_lora_trainer.sh (epic #1050 tier 4).",
    )
    runpod_sub = runpod_p.add_subparsers(dest="runpod_command")

    deploy_p = runpod_sub.add_parser(
        "deploy",
        help="Retired legacy deployment; explains authorized catalog migration",
    )
    deploy_p.add_argument(
        "target",
        choices=_VALID_TARGETS,
        help="Deployment target (currently: lora-trainer)",
    )
    deploy_p.add_argument(
        "--profile",
        type=str,
        default=_DEFAULT_PROFILE,
        help=f"runpod_config.toml profile (default: {_DEFAULT_PROFILE})",
    )
    deploy_p.add_argument(
        "--test",
        action="store_true",
        help="Legacy compatibility flag; deployment remains retired",
    )

    status_p = runpod_sub.add_parser(
        "status",
        help="Show RunPod pod status (active or resumable)",
    )
    status_p.add_argument(
        "--profile",
        type=str,
        default=_DEFAULT_PROFILE,
        help=f"runpod_config.toml profile (default: {_DEFAULT_PROFILE})",
    )

    stop_p = runpod_sub.add_parser(
        "stop",
        help="Stop the pod (keep for resume)",
    )
    stop_p.add_argument(
        "--profile",
        type=str,
        default=_DEFAULT_PROFILE,
        help=f"runpod_config.toml profile (default: {_DEFAULT_PROFILE})",
    )

    kill_p = runpod_sub.add_parser(
        "kill",
        help="Terminate the pod completely (no resume)",
    )
    kill_p.add_argument(
        "--profile",
        type=str,
        default=_DEFAULT_PROFILE,
        help=f"runpod_config.toml profile (default: {_DEFAULT_PROFILE})",
    )


# ---------------------------------------------------------------------------
# Top-level handler
# ---------------------------------------------------------------------------

def cmd_runpod(args) -> int:
    """Dispatch ``kestrel runpod ...``.

    Exit codes:
        0 - success
        1 - missing RUNPOD_API_KEY, missing/invalid profile,
            kestrel-cloud-runpod not installed, or RunPod-side
            failure (e.g. GPU unavailable).
    """
    sub = getattr(args, "runpod_command", None)
    if sub == "deploy":
        return _cmd_deploy(args)
    if sub == "status":
        return _cmd_status(args)
    if sub == "stop":
        return _cmd_stop(args)
    if sub == "kill":
        return _cmd_kill(args)

    print(
        "Usage:\n"
        "  kestrel runpod deploy lora-trainer [--profile NAME] [--test]\n"
        "  kestrel runpod status              [--profile NAME]\n"
        "  kestrel runpod stop                [--profile NAME]\n"
        "  kestrel runpod kill                [--profile NAME]",
        file=sys.stderr,
    )
    return 1


__all__ = [
    "add_runpod_subcommand",
    "cmd_runpod",
]
