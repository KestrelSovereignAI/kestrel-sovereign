"""Synchronize public-repository mirrors of contributor-facing documents.

The root CONTRIBUTING.md is canonical.  The copy in public_repo_files is used
when preparing the public repository, so this small deterministic script and
its unit test keep the two from becoming independently edited documents.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONTRIBUTING = PROJECT_ROOT / "CONTRIBUTING.md"
PUBLIC_CONTRIBUTING = (
    PROJECT_ROOT / "scripts" / "public_repo_files" / "CONTRIBUTING.md"
)
CANONICAL_NOTICE = """<!-- canonical-contributing-notice:start -->
> **Canonical contributor guide:** this root file is the source of truth. The
> public-repository copy at `scripts/public_repo_files/CONTRIBUTING.md` is
> synchronized from it. After editing this guide, run
> `uv run python scripts/sync_public_repo_files.py --check` before opening a
> pull request.
<!-- canonical-contributing-notice:end -->"""
PUBLIC_NOTICE = """<!-- canonical-contributing-notice:start -->
> **Public contributor-guide mirror:** this file is generated from the
> canonical `CONTRIBUTING.md` in the Kestrel Sovereign source repository. Do
> not edit this copy independently; submit contributor-guide changes to the
> canonical source.
<!-- canonical-contributing-notice:end -->"""


def render_public_contributing() -> str:
    """Render the public mirror with source-appropriate provenance."""
    canonical = CANONICAL_CONTRIBUTING.read_text()
    if canonical.count(CANONICAL_NOTICE) != 1:
        raise ValueError("Canonical contributor-guide notice must occur exactly once")
    return canonical.replace(CANONICAL_NOTICE, PUBLIC_NOTICE)


def contributing_copy_is_current() -> bool:
    """Return whether the public CONTRIBUTING mirror matches its source."""
    return PUBLIC_CONTRIBUTING.read_text() == render_public_contributing()


def sync_contributing_copy() -> None:
    """Regenerate the public CONTRIBUTING mirror from the canonical guide."""
    PUBLIC_CONTRIBUTING.write_text(render_public_contributing())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without modifying the public mirror",
    )
    args = parser.parse_args()

    if args.check:
        if contributing_copy_is_current():
            print("Public CONTRIBUTING mirror is current.")
            return 0
        print(
            "Public CONTRIBUTING mirror is stale; run "
            "uv run python scripts/sync_public_repo_files.py"
        )
        return 1

    sync_contributing_copy()
    print("Synchronized public CONTRIBUTING mirror.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
