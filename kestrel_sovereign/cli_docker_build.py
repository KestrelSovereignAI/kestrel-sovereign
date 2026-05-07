"""``kestrel docker build <preset>`` CLI command - sub-PR 4 of epic
#1050 (bash-to-Python port of ``scripts/docker/build_*.sh``).

Builds one of four specialized images via Cloud Build
(``gcloud builds submit``) and pushes to GCR. Replaces these four
shell predecessors:

- ``scripts/docker/build_gpu_cloud.sh``
- ``scripts/docker/build_lora_trainer.sh``
- ``scripts/docker/build_ollama_server.sh``
- ``scripts/docker/build_simpletuner.sh``

Hangs under the existing ``kestrel docker`` parent next to
``kestrel docker remote``. The naming split:

- ``kestrel docker build <preset>`` - GCR / Cloud-Build flow
  for specialty images (this module).
- ``kestrel docker remote {build,run}`` - local workstation flow for
  the lightweight remote-LLM container (``cli_docker_remote.py``).

Preserved config drift: ``simpletuner`` and ``lora-trainer`` both
write to the same ``kestrel-lora`` image name. That's how the bash
predecessors were configured - the operator picks one or the other.
We preserve it; flagged in the table below.

Presets
-------

================ ===================== ==================================
Preset            GCR image             Dockerfile
================ ===================== ==================================
gpu-cloud         kestrel-gpu           docker/Dockerfile.gpu
lora-trainer      kestrel-lora          docker/Dockerfile.lora-trainer
ollama-server     kestrel-ollama        docker/Dockerfile.ollama-server
simpletuner       kestrel-lora *(!)*    docker/Dockerfile.simpletuner
================ ===================== ==================================

``simpletuner`` overwrites ``lora-trainer``'s GCR image - this is the
existing config-drift carry-over from the bash predecessors.

Configuration
-------------

- ``GCP_PROJECT_ID`` - required env var. The bash predecessors had
  ``PROJECT_ID="YOUR_PROJECT_ID"`` literal placeholders (operators
  were meant to fill them in). The Python port replaces that
  rotting-bit with proper env-var resolution and errors clearly if
  unset.
- ``--tag`` - image tag (default: ``latest``).
- ``--no-cache`` - pass ``--no-cache`` through to ``docker build``
  inside Cloud Build (matches the predecessors that hardcoded it for
  the LoRA trainer's enormous PyTorch wheels).
- ``--list`` - print the preset table and exit. Lighter than a
  separate ``presets list`` subverb.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from kestrel_sovereign._subprocess_helpers import run_streaming


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildPreset:
    """One Cloud-Build target: GCR image name + Dockerfile + timeout."""

    name: str
    image: str
    dockerfile: str
    description: str
    # Per-step timeout in seconds. The bash predecessors set this per
    # script - LoRA was 3600s, ollama 1200s, GPU 1800s, simpletuner had
    # a separate cloudbuild yaml. We store the same per-preset value.
    timeout_seconds: int = 1800
    # Per-build total Cloud Build timeout (top-level ``timeout`` in the
    # yaml). Defaults to ``timeout_seconds + a buffer``; presets that
    # need more (e.g. simpletuner's CUDA image) override.
    total_timeout_seconds: Optional[int] = None
    # Cloud Build runner machine type — defaults to gcloud's default.
    # Heavy CUDA builds need ``E2_HIGHCPU_32`` to fit in their step
    # timeout.
    machine_type: Optional[str] = None
    # Disk size for the build VM in GB — defaults to gcloud default
    # (~100GB). Large images need 200+.
    disk_size_gb: Optional[int] = None
    # ``simpletuner`` had its own ``docker/cloudbuild-simpletuner.yaml``
    # file (already vendored); we point at it instead of synthesizing
    # one. None => synthesize a yaml inline.
    cloudbuild_yaml: Optional[str] = None


_PRESETS: Dict[str, BuildPreset] = {
    "gpu-cloud": BuildPreset(
        name="gpu-cloud",
        image="kestrel-gpu",
        dockerfile="docker/Dockerfile.gpu",
        description=(
            "GPU-enabled Kestrel runtime (CUDA 11.8 + Python 3.11 + "
            "Ollama + Kestrel agent framework). 15-30 min build."
        ),
        timeout_seconds=1800,
    ),
    "lora-trainer": BuildPreset(
        name="lora-trainer",
        image="kestrel-lora",
        dockerfile="docker/Dockerfile.lora-trainer",
        description=(
            "LoRA training image (CUDA devel multi-stage, ~12-14GB, "
            "PyTorch + diffusers). 20-40 min build."
        ),
        timeout_seconds=3600,
    ),
    "ollama-server": BuildPreset(
        name="ollama-server",
        image="kestrel-ollama",
        dockerfile="docker/Dockerfile.ollama-server",
        description=(
            "Ollama server image (official base + persistent model "
            "storage + health check). ~20 min build."
        ),
        timeout_seconds=1200,
    ),
    "simpletuner": BuildPreset(
        name="simpletuner",
        # NOTE: shares the kestrel-lora image name with lora-trainer.
        # Pre-existing config drift from the bash predecessors;
        # preserved here intentionally.
        image="kestrel-lora",
        dockerfile="docker/Dockerfile.simpletuner",
        description=(
            "SimpleTuner FLUX.2 training+inference image. NOTE: "
            "overwrites the lora-trainer image (kestrel-lora) - pick "
            "one preset for that GCR slot."
        ),
        # Resource settings carried over from the vendored
        # docker/cloudbuild-simpletuner.yaml — codex review v3 on PR
        # #1074 caught that synthesizing without these would let the
        # large CUDA image hit the default Cloud Build disk/timeout.
        timeout_seconds=3600,             # per-step (was ``timeout: 3600s``)
        total_timeout_seconds=7200,       # top-level (was ``timeout: 7200s``)
        machine_type="E2_HIGHCPU_32",
        disk_size_gb=200,
        # We deliberately do NOT use docker/cloudbuild-simpletuner.yaml
        # here — that vendored config has a hardcoded
        # ``gcr.io/YOUR_PROJECT_ID/kestrel-lora`` placeholder that
        # would override the operator's resolved GCP_PROJECT_ID and
        # ``--tag``. Codex review on PR #1074 caught the silent
        # ignore. The synthesized yaml from
        # ``_synthesize_cloudbuild_yaml`` substitutes both correctly.
        # The vendored yaml stays in-tree for direct
        # ``gcloud builds submit --config=...`` use as a reference.
        cloudbuild_yaml=None,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Repo root - the Cloud Build context."""
    return Path(__file__).resolve().parent.parent


def _require_gcp_project() -> Optional[str]:
    """Return ``GCP_PROJECT_ID`` or None. Caller prints the error."""
    val = os.environ.get("GCP_PROJECT_ID")
    if not val:
        return None
    return val


def _print_missing_project_error() -> None:
    print(
        "error: GCP_PROJECT_ID is not set.\n"
        "\n"
        "The bash predecessors had `PROJECT_ID=\"YOUR_PROJECT_ID\"` as a\n"
        "placeholder; the Python port resolves the real project from\n"
        "the environment. Set it and retry:\n"
        "  export GCP_PROJECT_ID=\"my-gcp-project\"\n",
        file=sys.stderr,
    )


def _print_preset_table() -> None:
    print("Available `kestrel docker build <preset>` presets:\n")
    width = max(len(p.name) for p in _PRESETS.values()) + 2
    img_width = max(len(p.image) for p in _PRESETS.values()) + 2
    df_width = max(len(p.dockerfile) for p in _PRESETS.values()) + 2
    print(
        f"  {'preset'.ljust(width)}{'image'.ljust(img_width)}"
        f"{'dockerfile'.ljust(df_width)}description"
    )
    print(
        f"  {'-'*(width-2)}  {'-'*(img_width-2)}  "
        f"{'-'*(df_width-2)}  {'-'*40}"
    )
    for preset in _PRESETS.values():
        print(
            f"  {preset.name.ljust(width)}{preset.image.ljust(img_width)}"
            f"{preset.dockerfile.ljust(df_width)}{preset.description}"
        )


def _synthesize_cloudbuild_yaml(
    preset: BuildPreset, project: str, tag: str, *, no_cache: bool,
) -> str:
    """Render a single-step Cloud Build yaml for ``preset``.

    Always tags both ``:<tag>`` and ``:latest`` (deduped when
    ``tag == "latest"``) — matches the bash predecessors and the
    Tier 1.3 ``kestrel deploy build`` convention so ``:latest``-pinning
    deploys keep working after a versioned build.
    """
    image_ref = f"gcr.io/{project}/{preset.image}:{tag}"
    latest_ref = f"gcr.io/{project}/{preset.image}:latest"
    args: List[str] = ["build"]
    if no_cache:
        args.append("--no-cache")
    args.extend(["-f", preset.dockerfile, "-t", image_ref])
    if tag != "latest":
        args.extend(["-t", latest_ref])
    args.append(".")
    args_yaml = ", ".join(f"'{a}'" for a in args)

    images_block = f"  - '{image_ref}'\n"
    if tag != "latest":
        images_block += f"  - '{latest_ref}'\n"

    # Top-level timeout: default to step timeout + 30min buffer for
    # the implicit push step Cloud Build adds. Presets that override
    # (simpletuner) get the explicit value.
    total_timeout = (
        preset.total_timeout_seconds
        if preset.total_timeout_seconds is not None
        else preset.timeout_seconds + 1800
    )

    options_lines: List[str] = []
    if preset.machine_type:
        options_lines.append(f"  machineType: '{preset.machine_type}'\n")
    if preset.disk_size_gb is not None:
        options_lines.append(f"  diskSizeGb: {preset.disk_size_gb}\n")
    options_block = ""
    if options_lines:
        options_block = "options:\n" + "".join(options_lines)

    return (
        f"steps:\n"
        f"  - name: 'gcr.io/cloud-builders/docker'\n"
        f"    args: [{args_yaml}]\n"
        f"    timeout: {preset.timeout_seconds}s\n"
        f"images:\n"
        f"{images_block}"
        f"{options_block}"
        f"timeout: {total_timeout}s\n"
    )


# ---------------------------------------------------------------------------
# Build subverb handler
# ---------------------------------------------------------------------------

def _cmd_docker_build(args) -> int:
    """``kestrel docker build <preset> [--tag latest] [--no-cache]``.

    Streams Cloud Build's progress through the parent terminal.
    """
    if getattr(args, "list", False):
        _print_preset_table()
        return 0

    preset_name: Optional[str] = getattr(args, "preset", None)
    if not preset_name:
        print(
            "error: missing preset.\n"
            "  Run `kestrel docker build --list` to see options.",
            file=sys.stderr,
        )
        return 1

    preset = _PRESETS.get(preset_name)
    if preset is None:
        print(
            f"error: unknown preset {preset_name!r}.\n"
            f"  Available: {sorted(_PRESETS.keys())}",
            file=sys.stderr,
        )
        return 1

    project = _require_gcp_project()
    if project is None:
        _print_missing_project_error()
        return 1

    tag: str = getattr(args, "tag", None) or "latest"
    no_cache: bool = getattr(args, "no_cache", False)
    image_ref = f"gcr.io/{project}/{preset.image}:{tag}"

    print(f"Building Cloud Build preset: {preset_name}")
    print(f"  Project:    {project}")
    print(f"  Image:      {image_ref}")
    print(f"  Dockerfile: {preset.dockerfile}")
    print(f"  Tag:        {tag}")
    print(f"  No-cache:   {no_cache}")
    print()

    repo = _repo_root()

    # Verify the Dockerfile exists in the repo (matches the
    # build_simpletuner.sh predecessor's preflight).
    dockerfile_path = repo / preset.dockerfile
    if not dockerfile_path.is_file():
        print(
            f"error: {preset.dockerfile} not found at {dockerfile_path}.\n"
            "  Run `kestrel docker build` from the kestrel project root, "
            "or fix the preset.",
            file=sys.stderr,
        )
        return 1

    # Resolve the cloudbuild yaml: vendored file (simpletuner) or
    # synthesized in a temp dir.
    if preset.cloudbuild_yaml:
        config_path = repo / preset.cloudbuild_yaml
        if not config_path.is_file():
            print(
                f"error: {preset.cloudbuild_yaml} not found at "
                f"{config_path}.",
                file=sys.stderr,
            )
            return 1
        with tempfile.TemporaryDirectory() as _:
            return _submit_cloud_build(
                repo, project, str(config_path),
            )
    else:
        yaml_text = _synthesize_cloudbuild_yaml(
            preset, project, tag, no_cache=no_cache,
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", prefix=f"cloudbuild-{preset.name}-",
            delete=False,
        ) as f:
            f.write(yaml_text)
            config_path = Path(f.name)
        try:
            return _submit_cloud_build(
                repo, project, str(config_path),
            )
        finally:
            try:
                config_path.unlink()
            except OSError:
                pass


def _submit_cloud_build(
    repo: Path, project: str, config_path: str,
) -> int:
    """Run ``gcloud builds submit --config=<yaml> --project=<id> .``.

    Streams output. Returns the gcloud exit code.
    """
    print(f"Starting Cloud Build (config: {config_path}) ...")
    rc = run_streaming(
        [
            "gcloud", "builds", "submit",
            f"--config={config_path}",
            f"--project={project}",
            ".",
        ],
        cwd=repo,
    )
    if rc != 0:
        print(
            f"error: gcloud builds submit returned {rc}",
            file=sys.stderr,
        )
        return rc
    print()
    print("Done.")
    return 0


# ---------------------------------------------------------------------------
# Argparse subcommand wiring
# ---------------------------------------------------------------------------

def add_docker_build_subcommand(
    subparsers: "argparse._SubParsersAction",
) -> None:
    """Register ``kestrel docker build <preset>`` under the shared
    ``kestrel docker`` parent (which also owns ``remote``).

    Uses the shared
    :func:`kestrel_sovereign.cli_docker_remote.get_or_create_docker_subparsers`
    helper so both this module and :mod:`cli_docker_remote` can
    contribute subverbs without fighting over which one creates the
    parser. Either order of registration works.
    """
    from kestrel_sovereign.cli_docker_remote import (
        get_or_create_docker_subparsers,
    )
    docker_subparsers = get_or_create_docker_subparsers(subparsers)
    build_p = docker_subparsers.add_parser(
        "build",
        help="Build a specialty image via Cloud Build "
             "(GPU/LoRA/Ollama/SimpleTuner)",
    )
    build_p.add_argument(
        "preset",
        nargs="?",
        choices=sorted(_PRESETS.keys()),
        default=None,
        help="Preset to build (omit with --list to print the table)",
    )
    build_p.add_argument(
        "--tag",
        type=str,
        default="latest",
        help="Image tag (default: latest)",
    )
    build_p.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        help="Pass --no-cache to the inner docker build (slower, "
             "guaranteed-clean rebuild)",
    )
    build_p.add_argument(
        "--list",
        action="store_true",
        help="Print the preset table and exit",
    )


# ---------------------------------------------------------------------------
# Top-level handler
# ---------------------------------------------------------------------------

def cmd_docker_build(args) -> int:
    """Dispatch ``kestrel docker build ...``.

    Exit codes:
        0 - success
        1 - missing GCP_PROJECT_ID, unknown preset, missing
            Dockerfile, or non-zero gcloud exit.
    """
    return _cmd_docker_build(args)


__all__ = [
    "add_docker_build_subcommand",
    "cmd_docker_build",
    "BuildPreset",
]
