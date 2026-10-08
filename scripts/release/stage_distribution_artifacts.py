"""Stage only the wheel and source distribution, before release keys are loaded.

uv creates a .gitignore in its output directory. Signing that directory directly
signs an auxiliary file that neither publication path downloads. A fresh staging
directory makes the signed inventory identical to the package upload inventory.
Historical manifests and packages are deliberately not modified by this tool.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
from pathlib import Path

from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version


def _regular_file(fd: int) -> os.stat_result:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Distribution inputs must be regular, single-link files")
    return info


def stage_distributions(source: Path, destination: Path, release_tag: str) -> list[str]:
    """Copy one matching Core wheel and sdist to a new, exclusive directory.

    Refuse unexpected inputs instead of silently widening the signed inventory.
    The only ignored entry is uv's regular .gitignore. No input is deleted.
    """
    if not re.fullmatch(r"v[0-9]+(?:\.[0-9]+){2}(?:[-+][A-Za-z0-9.]+)?", release_tag):
        raise ValueError("Release tag must be vX.Y.Z")
    version = Version(release_tag[1:])
    source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened: list[tuple[str, int, os.stat_result]] = []
    try:
        kinds = []
        for name in sorted(os.listdir(source_fd)):
            fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_fd
            )
            try:
                info = _regular_file(fd)
                if name == ".gitignore":
                    continue
                if name.endswith(".whl"):
                    project, found_version, _, _ = parse_wheel_filename(name)
                    kind = "wheel"
                elif name.endswith(".tar.gz"):
                    project, found_version = parse_sdist_filename(name)
                    kind = "sdist"
                else:
                    raise ValueError(f"Unexpected build output: {name}")
                if (
                    canonicalize_name(project) != "kestrel-sovereign"
                    or found_version != version
                ):
                    raise ValueError(f"Distribution does not match release tag: {name}")
                kinds.append(kind)
                opened.append((name, fd, info))
                fd = -1
            finally:
                if fd != -1:
                    os.close(fd)
        if sorted(kinds) != ["sdist", "wheel"]:
            raise ValueError(
                "Expected exactly one matching wheel and one source distribution"
            )

        # Never reuse a directory containing stale artifacts or an old manifest.
        destination.mkdir(mode=0o700)
        destination_fd = os.open(
            destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            for name, fd, before in opened:
                out_fd = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=destination_fd,
                )
                with os.fdopen(out_fd, "wb") as output:
                    while chunk := os.read(fd, 1024 * 1024):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                after = _regular_file(fd)
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError(f"Distribution changed during staging: {name}")
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        return [name for name, _, _ in opened]
    finally:
        for _, fd, _ in opened:
            os.close(fd)
        os.close(source_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()
    try:
        names = stage_distributions(args.source, args.destination, args.release_tag)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Cannot stage release distributions: {exc}\n")
    print("Staged release distributions: " + ", ".join(names))


if __name__ == "__main__":
    main()
