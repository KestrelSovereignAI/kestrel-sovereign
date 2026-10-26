#!/usr/bin/env python3
"""Retired ownerless Runpod helper; see Core issue #3828."""

from pathlib import Path
import sys


def main() -> None:
    """Refuse before reading credentials, dotenv or any provider client."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kestrel_sovereign.legacy_runpod import LegacyRunpodPathError

    raise SystemExit(str(LegacyRunpodPathError(Path(__file__).name)))


if __name__ == "__main__":
    main()
