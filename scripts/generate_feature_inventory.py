#!/usr/bin/env python3
"""Generate the canonical Kestrel feature inventory from code discovery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kestrel_sovereign.feature_inventory import (
    CANONICAL_INVENTORY,
    build_inventory,
    render_inventory_json,
    render_inventory_markdown,
    write_canonical_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover Kestrel features, tools, endpoints, and commands."
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format for stdout or --output (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write generated output to this path instead of stdout.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"Regenerate {CANONICAL_INVENTORY.name} in place.",
    )
    args = parser.parse_args()

    if args.write:
        write_canonical_inventory()
        print(f"Wrote {CANONICAL_INVENTORY}")
        return 0

    inventory = build_inventory()
    output = (
        render_inventory_json(inventory)
        if args.format == "json"
        else render_inventory_markdown(inventory)
    )
    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
