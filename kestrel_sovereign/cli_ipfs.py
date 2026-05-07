"""``kestrel ipfs {build,deploy,pin}`` CLI command — sub-PR 4 of
epic #1050 (bash-to-Python port of ``scripts/ipfs/{build,deploy,
pin_agents}.sh``).

Three subverbs that drive a self-hosted Kubo IPFS node backed by GCS
block storage:

- ``kestrel ipfs build [--tag TAG]`` — build + push the custom Kubo
  image (``docker/ipfs/Dockerfile.gcs``) to
  ``gcr.io/$GCP_PROJECT_ID/kestrel-ipfs-gcs:<tag>`` (and ``:latest``).
- ``kestrel ipfs deploy {create|update|delete|status|ssh}
  [--zone us-central1-a]`` — manage the GCE VM that runs the Kubo
  image. Creates an ``e2-small`` VM tagged ``kestrel-ipfs`` with
  swarm/gateway/API firewall rules, GCS-backed
  (``gs://kestrel-ipfs``) so the VM is stateless.
- ``kestrel ipfs pin [--manifest <path>]`` — take a consistent SQLite
  snapshot of every agent DB under ``agent_data/*/kestrel_prime.db``
  and pin the snapshot via the running IPFS API. Default API:
  ``http://localhost:5001`` (override with ``--api-url``).

Implementation notes
--------------------

We shell out to ``gcloud`` (not ``google-cloud-compute``) — the deploy
surface is tiny enough that a compute-SDK dependency is unjustified.
This matches the bash predecessor's approach. Every subprocess call
streams its output via :func:`run_streaming` so the operator sees
``gcloud compute instances create-with-container`` progress live
instead of a 30-second silent block.

Cross-platform: ``gcloud`` and ``docker`` work on Windows
PowerShell; the SQLite-backup path uses pure-stdlib :mod:`sqlite3`
which is platform-agnostic. ``urllib.request`` replaces the bash
predecessor's ``curl`` for the IPFS HTTP API.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from kestrel_sovereign._subprocess_helpers import run_streaming


# ---------------------------------------------------------------------------
# Constants — match the bash predecessors verbatim so operators can read
# the bash and the Python interchangeably.
# ---------------------------------------------------------------------------

_IMAGE_NAME = "kestrel-ipfs-gcs"
_INSTANCE_NAME = "kestrel-ipfs"
_DEFAULT_ZONE = "us-central1-a"
_MACHINE_TYPE = "e2-small"
_GCS_BUCKET = "kestrel-ipfs"
_NETWORK_TAG = "kestrel-ipfs"
_DOCKERFILE_REL = Path("docker") / "ipfs" / "Dockerfile.gcs"
_DOCKER_BUILD_CONTEXT_REL = Path("docker") / "ipfs"
_DEFAULT_IPFS_API = "http://localhost:5001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Repo root — used as the docker build context cwd."""
    return Path(__file__).resolve().parent.parent


def _require_gcp_project() -> Optional[str]:
    """Return ``GCP_PROJECT_ID`` or None. Caller prints the error.

    The bash predecessors used ``${GCP_PROJECT_ID:?...}`` — a hard error
    on the first reference. We surface it as a clean CLI error instead
    of a bash-style ``unbound variable`` trace.
    """
    val = os.environ.get("GCP_PROJECT_ID")
    if not val:
        return None
    return val


def _print_missing_project_error() -> None:
    print(
        "error: GCP_PROJECT_ID is not set.\n"
        "\n"
        "Set the project to deploy the Kestrel IPFS node into:\n"
        "  export GCP_PROJECT_ID=\"my-gcp-project\"\n",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# build subverb
# ---------------------------------------------------------------------------

def _cmd_build(args) -> int:
    """``kestrel ipfs build [--tag TAG]``.

    Builds and pushes ``gcr.io/$GCP_PROJECT_ID/kestrel-ipfs-gcs:<tag>``
    plus ``:latest``. Single-platform (``linux/amd64`` — the GCE
    target).
    """
    project = _require_gcp_project()
    if project is None:
        _print_missing_project_error()
        return 1

    tag: str = getattr(args, "tag", None) or "latest"

    image_versioned = f"gcr.io/{project}/{_IMAGE_NAME}:{tag}"
    image_latest = f"gcr.io/{project}/{_IMAGE_NAME}:latest"

    print("Building Kestrel IPFS (Kubo + GCS) image...")
    print(f"  Project:  {project}")
    print(f"  Tag:      {tag}")
    print(f"  Platform: linux/amd64 (GCE target)")
    print()

    repo = _repo_root()

    rc = run_streaming(
        [
            "docker", "build",
            "--platform", "linux/amd64",
            "-f", str(_DOCKERFILE_REL),
            "-t", image_versioned,
            "-t", image_latest,
            str(_DOCKER_BUILD_CONTEXT_REL),
        ],
        cwd=repo,
    )
    if rc != 0:
        return rc

    print()
    print("Pushing to GCR...")
    rc = run_streaming(["docker", "push", image_versioned])
    if rc != 0:
        return rc
    rc = run_streaming(["docker", "push", image_latest])
    if rc != 0:
        return rc

    print()
    print(f"Done! Image: {image_versioned}")
    return 0


# ---------------------------------------------------------------------------
# deploy subverb
# ---------------------------------------------------------------------------

def _resolve_service_account(project: str) -> str:
    """Return the service-account email to attach to the VM.

    Honors ``IPFS_SERVICE_ACCOUNT`` env var (matches the bash
    predecessor). Else queries ``gcloud iam service-accounts list``
    and falls back to the default compute SA. We do *not* fail hard
    on lookup failure — gcloud will fail with its own actionable
    error during the create call if the SA is wrong.
    """
    explicit = os.environ.get("IPFS_SERVICE_ACCOUNT")
    if explicit:
        return explicit

    # Best-effort: query the default compute SA. We use a Popen with
    # piped stdout because run_streaming inherits the parent's stdout
    # and we want to swallow the (often empty) output.
    try:
        res = subprocess.run(
            [
                "gcloud", "iam", "service-accounts", "list",
                f"--project={project}",
                "--filter=email:compute@developer.gserviceaccount.com",
                "--format=value(email)",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (res.stdout or "").strip()
        if out:
            return out.splitlines()[0].strip()
    except (FileNotFoundError, OSError):
        pass
    # Last-resort literal: the bash predecessor would have left this
    # empty, causing gcloud create to error out — match that.
    return ""


def _print_deploy_info(project: str, zone: str) -> None:
    print("Kestrel IPFS Node (GCE)")
    print(f"  Project:  {project}")
    print(f"  Instance: {_INSTANCE_NAME}")
    print(f"  Zone:     {zone}")
    print(f"  Machine:  {_MACHINE_TYPE}")
    print(f"  Bucket:   gs://{_GCS_BUCKET}")
    print()


def _ensure_firewall_rules(project: str) -> int:
    """Create the three firewall rules if they don't exist.

    Idempotent — describe first, create only on miss. Same idiom as
    the bash predecessor.
    """
    print("Setting up firewall rules...")

    rules = [
        {
            "name": f"{_NETWORK_TAG}-swarm",
            "rules": "tcp:4001,udp:4001",
            "source_ranges": "0.0.0.0/0",
            "description": "IPFS swarm (libp2p peering)",
        },
        {
            "name": f"{_NETWORK_TAG}-gateway",
            "rules": "tcp:8080",
            "source_ranges": "0.0.0.0/0",
            "description": "IPFS gateway (HTTP)",
        },
        {
            "name": f"{_NETWORK_TAG}-api",
            "rules": "tcp:5001",
            # Internal CIDRs + IAP range — same as the bash predecessor.
            "source_ranges": "10.128.0.0/9,35.235.240.0/20",
            "description": "IPFS API (internal + IAP only)",
        },
    ]

    for r in rules:
        # Check exists.
        describe = subprocess.run(
            [
                "gcloud", "compute", "firewall-rules", "describe", r["name"],
                f"--project={project}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if describe.returncode == 0:
            continue
        rc = run_streaming(
            [
                "gcloud", "compute", "firewall-rules", "create", r["name"],
                f"--project={project}",
                "--direction=INGRESS",
                "--action=ALLOW",
                f"--rules={r['rules']}",
                f"--source-ranges={r['source_ranges']}",
                f"--target-tags={_NETWORK_TAG}",
                f"--description={r['description']}",
                "--quiet",
            ],
        )
        if rc != 0:
            return rc

    print("  Firewall rules ready.")
    return 0


def _deploy_create(project: str, zone: str) -> int:
    _print_deploy_info(project, zone)
    rc = _ensure_firewall_rules(project)
    if rc != 0:
        return rc

    sa_email = _resolve_service_account(project)
    image = f"gcr.io/{project}/{_IMAGE_NAME}:latest"

    print("Creating instance...")
    cmd = [
        "gcloud", "compute", "instances", "create-with-container",
        _INSTANCE_NAME,
        f"--project={project}",
        f"--zone={zone}",
        f"--machine-type={_MACHINE_TYPE}",
        "--boot-disk-size=10GB",
        "--boot-disk-type=pd-standard",
        f"--tags={_NETWORK_TAG}",
        "--scopes=storage-full",
        f"--container-image={image}",
        f"--container-env=KUBO_GCS_BUCKET={_GCS_BUCKET}",
        "--container-mount-host-path=host-path=/var/ipfs,mount-path=/home/ipfs/.ipfs",
        "--metadata=google-logging-enabled=true",
        "--quiet",
    ]
    if sa_email:
        cmd.insert(8, f"--service-account={sa_email}")

    rc = run_streaming(cmd)
    if rc != 0:
        return rc

    # Read the IP back so the operator gets the same hand-off text the
    # bash predecessor printed.
    ip_res = subprocess.run(
        [
            "gcloud", "compute", "instances", "describe", _INSTANCE_NAME,
            f"--project={project}",
            f"--zone={zone}",
            "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    external_ip = (ip_res.stdout or "").strip()

    print()
    print("Instance created!")
    print()
    if external_ip:
        print(f"  External IP: {external_ip}")
        print(f"  Swarm:       /ip4/{external_ip}/tcp/4001")
        print(f"  Gateway:     http://{external_ip}:8080")
        print(f"  API:         http://{external_ip}:5001 (internal only)")
    return 0


def _deploy_update(project: str, zone: str) -> int:
    _print_deploy_info(project, zone)
    image = f"gcr.io/{project}/{_IMAGE_NAME}:latest"

    print("Updating container image...")
    rc = run_streaming(
        [
            "gcloud", "compute", "instances", "update-container",
            _INSTANCE_NAME,
            f"--project={project}",
            f"--zone={zone}",
            f"--container-image={image}",
            f"--container-env=KUBO_GCS_BUCKET={_GCS_BUCKET}",
            "--quiet",
        ],
    )
    if rc != 0:
        return rc

    print("Updated! Instance will pull new image on next restart.")
    print(
        f"  Restart now: gcloud compute instances reset {_INSTANCE_NAME} "
        f"--zone={zone} --project={project}"
    )
    return 0


def _deploy_delete(project: str, zone: str, *, assume_yes: bool = False) -> int:
    _print_deploy_info(project, zone)
    print("WARNING: This will delete the IPFS VM instance.")
    print(f"  GCS blocks in gs://{_GCS_BUCKET} are NOT affected.")
    print()
    if not assume_yes:
        try:
            reply = input(f"Delete {_INSTANCE_NAME}? [y/N] ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            print("Cancelled.")
            return 0

    rc = run_streaming(
        [
            "gcloud", "compute", "instances", "delete", _INSTANCE_NAME,
            f"--project={project}",
            f"--zone={zone}",
            "--quiet",
        ],
    )
    if rc != 0:
        return rc
    print("Deleted.")
    return 0


def _deploy_status(project: str, zone: str) -> int:
    _print_deploy_info(project, zone)

    describe = subprocess.run(
        [
            "gcloud", "compute", "instances", "describe", _INSTANCE_NAME,
            f"--project={project}",
            f"--zone={zone}",
            "--format=value(status)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if describe.returncode != 0:
        print(f"  Instance not found. Create with: kestrel ipfs deploy create")
        return 0

    status = (describe.stdout or "").strip()
    ip_res = subprocess.run(
        [
            "gcloud", "compute", "instances", "describe", _INSTANCE_NAME,
            f"--project={project}",
            f"--zone={zone}",
            "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    external_ip = (ip_res.stdout or "").strip()
    print(f"  Status: {status}")
    print(f"  IP:     {external_ip}")
    return 0


def _deploy_ssh(project: str, zone: str) -> int:
    return run_streaming(
        [
            "gcloud", "compute", "ssh", _INSTANCE_NAME,
            f"--project={project}",
            f"--zone={zone}",
        ],
    )


def _cmd_deploy(args) -> int:
    """``kestrel ipfs deploy {create|update|delete|status|ssh}``."""
    project = _require_gcp_project()
    if project is None:
        _print_missing_project_error()
        return 1

    zone: str = getattr(args, "zone", None) or _DEFAULT_ZONE
    # Default action ``status`` matches the bash predecessor's
    # ``ACTION="${1:-status}"`` — running ``kestrel ipfs deploy`` with
    # no subverb is the operator's "is the node up?" health-check
    # path, not a usage error. Codex review on PR #1074 caught the
    # regression.
    action: str = getattr(args, "action", None) or "status"
    assume_yes: bool = getattr(args, "yes", False)

    if action == "create":
        return _deploy_create(project, zone)
    if action == "update":
        return _deploy_update(project, zone)
    if action == "delete":
        return _deploy_delete(project, zone, assume_yes=assume_yes)
    if action == "status":
        return _deploy_status(project, zone)
    if action == "ssh":
        return _deploy_ssh(project, zone)

    print(
        "Usage: kestrel ipfs deploy {create|update|delete|status|ssh} "
        "[--zone us-central1-a]\n"
        "\n"
        "  create  - Create IPFS VM instance\n"
        "  update  - Update container image\n"
        "  delete  - Delete VM (GCS blocks preserved)\n"
        "  status  - Check instance status\n"
        "  ssh     - SSH into the VM",
        file=sys.stderr,
    )
    return 1


# ---------------------------------------------------------------------------
# pin subverb
# ---------------------------------------------------------------------------

def _ipfs_api_get(api_url: str, path: str, *, timeout: float = 5.0) -> dict:
    """GET ``<api_url>/api/v0/<path>`` and parse JSON. Raises on HTTP error."""
    full = f"{api_url.rstrip('/')}/api/v0/{path.lstrip('/')}"
    req = urllib.request.Request(full, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _ipfs_add_file(api_url: str, file_path: Path, *, name: str,
                   timeout: float = 60.0) -> str:
    """POST to ``<api_url>/api/v0/add?pin=true`` with multipart body.

    Returns the CID. Uses :mod:`urllib` and :mod:`email.encoders` to
    avoid pulling in ``requests`` as a dependency.
    """
    import uuid
    boundary = f"----kestrel-ipfs-{uuid.uuid4().hex}"
    full = (
        f"{api_url.rstrip('/')}/api/v0/add?pin=true&quieter=true"
    )
    file_bytes = file_path.read_bytes()

    body = []
    body.append(f"--{boundary}\r\n".encode("utf-8"))
    body.append(
        (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{name}"\r\n'
        ).encode("utf-8")
    )
    body.append(b"Content-Type: application/octet-stream\r\n\r\n")
    body.append(file_bytes)
    body.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    payload = b"".join(body)

    req = urllib.request.Request(
        full,
        data=payload,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(payload)),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
    # /api/v0/add returns one JSON object per line; with quieter=true
    # only the final aggregate line is returned but we parse defensively.
    last = text.strip().splitlines()[-1]
    parsed = json.loads(last)
    return parsed["Hash"]


def _snapshot_sqlite(src: Path, dst: Path) -> None:
    """Take a consistent backup of ``src`` into ``dst`` via the SQLite
    backup API. Same path the bash predecessor used (it shelled out to
    Python for this anyway)."""
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dst))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()


def _format_size(n: int) -> str:
    s: float = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if s < 1024:
            return f"{s:.1f} {unit}"
        s /= 1024
    return f"{s:.1f} TB"


def _agent_data_dir() -> Path:
    """Default agent-data dir — honors ``KESTREL_AGENT_DATA``."""
    from_env = os.environ.get("KESTREL_AGENT_DATA")
    if from_env:
        return Path(from_env).expanduser().resolve()
    return _repo_root() / "agent_data"


def _cmd_pin(args) -> int:
    """``kestrel ipfs pin [--api-url URL] [--manifest PATH]``.

    Default behaviour — iterate ``agent_data/*/kestrel_prime.db``,
    snapshot each DB into a tempfile, POST to the IPFS API's
    ``/api/v0/add?pin=true``. Matches the bash predecessor.
    """
    api_url: str = getattr(args, "api_url", None) or _DEFAULT_IPFS_API
    manifest_arg: Optional[str] = getattr(args, "manifest", None)

    if manifest_arg:
        agent_dir_root = Path(manifest_arg).expanduser().resolve()
    else:
        agent_dir_root = _agent_data_dir()

    print("Pinning agent snapshots to IPFS")
    print(f"  API:  {api_url}")
    print(f"  Data: {agent_dir_root}")
    print()

    # Reachability check.
    try:
        info = _ipfs_api_get(api_url, "id", timeout=5.0)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(
            f"ERROR: Cannot reach IPFS API at {api_url}: {e}\n"
            "  Local:  ipfs daemon must be running\n"
            "  Remote: check firewall and instance status",
            file=sys.stderr,
        )
        return 1
    print(f"  Peer:  {info.get('ID', '<unknown>')}")
    print()

    if not agent_dir_root.exists():
        print(
            f"  Data directory does not exist: {agent_dir_root}",
            file=sys.stderr,
        )
        return 1

    found_any = False
    for entry in sorted(agent_dir_root.iterdir()):
        if not entry.is_dir():
            continue
        agent_name = entry.name
        db_path = entry / "kestrel_prime.db"
        if not db_path.is_file():
            print(f"  SKIP {agent_name} - no database at {db_path}")
            continue
        found_any = True
        with tempfile.NamedTemporaryFile(
            prefix=f"ipfs-{agent_name}-", suffix=".db", delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _snapshot_sqlite(db_path, tmp_path)
            size = tmp_path.stat().st_size
            try:
                cid = _ipfs_add_file(
                    api_url,
                    tmp_path,
                    name=f"{agent_name}/kestrel_prime.db",
                    timeout=120.0,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                    json.JSONDecodeError, KeyError) as e:
                print(f"  {agent_name}: FAILED - {e}")
                continue
            print(f"  {agent_name}: {cid} ({_format_size(size)})")
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    if not found_any:
        print(
            "  No agent databases found (looking for "
            "<agent_data>/<name>/kestrel_prime.db).",
            file=sys.stderr,
        )

    print()
    print(
        f"Done. View pins: curl -s {api_url}/api/v0/pin/ls | "
        f"python -m json.tool"
    )
    return 0


# ---------------------------------------------------------------------------
# Argparse subcommand wiring
# ---------------------------------------------------------------------------

def add_ipfs_subcommand(
    subparsers: "argparse._SubParsersAction",
) -> None:
    """Register ``kestrel ipfs {build,deploy,pin}`` under the parent
    subparsers.

    Each subverb mirrors the bash predecessor's positional layout,
    with named flags for everything that was a positional in bash.
    """
    ipfs_p = subparsers.add_parser(
        "ipfs",
        help="Self-hosted IPFS node (Kubo + GCS) lifecycle - port of "
             "scripts/ipfs/{build,deploy,pin_agents}.sh (epic #1050 tier 4).",
    )
    ipfs_sub = ipfs_p.add_subparsers(dest="ipfs_command")

    build_p = ipfs_sub.add_parser(
        "build",
        help="Build and push the Kubo+GCS image to GCR",
    )
    build_p.add_argument(
        "--tag",
        type=str,
        default="latest",
        help="Image tag (default: latest)",
    )

    deploy_p = ipfs_sub.add_parser(
        "deploy",
        help="Manage the GCE VM running the IPFS node",
    )
    deploy_p.add_argument(
        "action",
        nargs="?",
        choices=("create", "update", "delete", "status", "ssh"),
        default=None,
        help="Lifecycle action",
    )
    deploy_p.add_argument(
        "--zone",
        type=str,
        default=_DEFAULT_ZONE,
        help=f"GCE zone (default: {_DEFAULT_ZONE})",
    )
    deploy_p.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the interactive y/N prompt for `delete`",
    )

    pin_p = ipfs_sub.add_parser(
        "pin",
        help="Pin every agent's DB snapshot to the IPFS node",
    )
    pin_p.add_argument(
        "--api-url",
        dest="api_url",
        type=str,
        default=_DEFAULT_IPFS_API,
        help=f"IPFS API URL (default: {_DEFAULT_IPFS_API})",
    )
    pin_p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Override agent-data directory (default: $KESTREL_AGENT_DATA "
             "or <repo>/agent_data)",
    )


# ---------------------------------------------------------------------------
# Top-level handler
# ---------------------------------------------------------------------------

def cmd_ipfs(args) -> int:
    """Dispatch ``kestrel ipfs ...``.

    Exit codes:
        0 - success
        1 - missing GCP_PROJECT_ID, gcloud failure, or IPFS API
            unreachable on `pin`.
    """
    sub = getattr(args, "ipfs_command", None)
    if sub == "build":
        return _cmd_build(args)
    if sub == "deploy":
        return _cmd_deploy(args)
    if sub == "pin":
        return _cmd_pin(args)

    print(
        "Usage:\n"
        "  kestrel ipfs build  [--tag TAG]\n"
        "  kestrel ipfs deploy {create|update|delete|status|ssh} "
        "[--zone us-central1-a]\n"
        "  kestrel ipfs pin    [--api-url URL] [--manifest PATH]",
        file=sys.stderr,
    )
    return 1


__all__ = [
    "add_ipfs_subcommand",
    "cmd_ipfs",
]
