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

- ``kestrel runpod deploy lora-trainer [--profile training|training-4090]
  [--test]`` - start a LoRA-training pod via
  :meth:`RunPodManager.start_session`. ``--test`` runs a post-deploy
  ``/health`` + ``/openapi.json`` probe (matches the bash flag).
- ``kestrel runpod status [--profile training]`` - print active /
  resumable pod state.
- ``kestrel runpod stop [--profile training]`` - stop the session
  (pod stays around for resume).
- ``kestrel runpod kill [--profile training]`` - terminate the pod
  completely (no resume).

Validation
----------

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
from typing import Any, Dict, Optional


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
        print(f"Pod is ACTIVE")
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


async def _action_deploy(manager, profile_name: str, *,
                         test_after: bool) -> int:
    print("Starting training pod...")
    print("   (Using template_id for GCR private registry auth)")
    print()
    try:
        result = await manager.start_session(
            task_profile=profile_name,
            model_name="FLUX.1-dev",
            ttl_seconds=3600,
            metadata={
                "name": "kestrel-lora-trainer",
                "purpose": "lora_training",
            },
        )
    except Exception as e:  # noqa: BLE001 - bash predecessor is broad here
        msg = str(e)
        print(f"Deployment failed: {msg}", file=sys.stderr)
        if "no longer any instances available" in msg.lower():
            print("\nGPU Availability Issue:", file=sys.stderr)
            print(
                f"   Profile '{profile_name}' GPU unavailable in "
                "US-KS-2 datacenter.",
                file=sys.stderr,
            )
            print(
                "   (Network volume requires pods in same datacenter)",
                file=sys.stderr,
            )
            print("\n   Options:", file=sys.stderr)
            print("   1. Wait and retry later", file=sys.stderr)
            other = (
                "training-4090" if profile_name == "training" else "training"
            )
            print(
                f"   2. Try the other profile: "
                f"kestrel runpod deploy lora-trainer --profile {other}",
                file=sys.stderr,
            )
            print(
                "   3. Check RunPod dashboard for availability",
                file=sys.stderr,
            )
        return 1

    print(f"Pod created successfully!")
    print(f"   Pod ID: {result.get('pod_id')}")
    print(f"   Status: {result.get('status')}")
    print(f"   Base URL: {result.get('base_url')}")
    print(f"   TTL: {result.get('ttl_seconds')}s")
    print()

    if test_after:
        base_url = result.get("base_url")
        if base_url:
            print("Testing endpoints...")
            try:
                import httpx
            except ImportError:
                print(
                    "   (httpx not installed - skipping post-deploy probes)"
                )
                return 0
            try:
                health = httpx.get(f"{base_url}/health", timeout=60)
                print(f"   /health: {health.status_code}")
                print(f"   {health.text[:200]}")
            except Exception as e:  # noqa: BLE001
                print(f"   /health: {e}")
            try:
                openapi = httpx.get(f"{base_url}/openapi.json", timeout=30)
                if openapi.status_code == 200:
                    spec = openapi.json()
                    paths = list(spec.get("paths", {}).keys())
                    print(f"   Available endpoints: {paths}")
            except Exception as e:  # noqa: BLE001
                print(f"   /openapi.json: {e}")
    return 0


# ---------------------------------------------------------------------------
# Subverb dispatchers
# ---------------------------------------------------------------------------

def _cmd_deploy(args) -> int:
    """``kestrel runpod deploy lora-trainer [--profile NAME] [--test]``."""
    target: Optional[str] = getattr(args, "target", None)
    if target not in _VALID_TARGETS:
        print(
            f"error: unknown deploy target {target!r}\n"
            f"  Valid targets: {list(_VALID_TARGETS)}",
            file=sys.stderr,
        )
        return 1

    if not _check_api_key():
        _print_missing_api_key()
        return 1

    profile_name: str = getattr(args, "profile", None) or _DEFAULT_PROFILE
    test_after: bool = getattr(args, "test", False)

    print("Kestrel LoRA Trainer - RunPod Deployment")
    print()
    print(f"Profile: {profile_name}")

    manager = _load_manager()
    if not _validate_profile(manager, profile_name):
        return 1
    profile = manager.profiles[profile_name]
    _print_profile(profile)

    return asyncio.run(_action_deploy(
        manager, profile_name, test_after=test_after,
    ))


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
        help="Start a RunPod pod for LoRA training",
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
        help="Run /health + /openapi.json probes after deploy",
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
