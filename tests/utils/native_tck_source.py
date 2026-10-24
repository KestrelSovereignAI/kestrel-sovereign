"""Materialize the pinned, reviewed TCK in a test-owned temporary checkout."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", *arguments],
        cwd=root, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(f"TCK source preparation failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _verify_digest(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"TCK custody digest mismatch: {path.name}")


@contextmanager
def prepared_tck_source(
    parent: Path, *, provenance_dir: Path | None = None,
) -> Iterator[Path]:
    """Apply the reviewed patch to its exact public base, then remove our checkout.

    The enclosing pytest fixture owns this lifetime. An operator checkout or
    environment is never used as the conformance runner.
    """
    provenance = provenance_dir or Path(__file__).resolve().parents[1] / "interop/a2a_tck"
    manifest = tomllib.loads((provenance / "SOURCE.toml").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported TCK source manifest version")
    for field in ("upstream_commit", "patched_tree", "spec_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest[field]):
            raise ValueError(f"TCK source requires an immutable {field}")
    patch = provenance / "UPSTREAM.patch"
    _verify_digest(patch, manifest["patch_sha256"])
    _verify_digest(provenance / "LICENSE", manifest["license_sha256"])
    with tempfile.TemporaryDirectory(prefix="native-a2a-tck-", dir=parent) as work:
        root = Path(work)
        _git(root, "init", "--quiet")
        _git(root, "fetch", "--quiet", "--no-tags", "--depth=1",
             manifest["upstream_repository"], manifest["upstream_commit"])
        _git(root, "checkout", "--quiet", "--detach", manifest["upstream_commit"])
        _git(root, "apply", "--index", "--binary", str(patch.resolve()))
        if _git(root, "write-tree") != manifest["patched_tree"]:
            raise ValueError("TCK patched tree does not match the reviewed source")
        _verify_digest(root / "LICENSE", manifest["license_sha256"])
        version = json.loads((root / "specification/version.json").read_text())
        if (version["branch"], version["commitHash"]) != (
            "v" + manifest["protocol_version"], manifest["spec_commit"],
        ):
            raise ValueError("TCK specification differs from the reviewed protocol revision")
        yield root
