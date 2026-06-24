#!/usr/bin/env python3
"""Project the OKF docs corpus into a published documentation site.

This is a *projection*, not a migration: ``docs/`` stays the source of truth and
this command emits a renderable site (Mintlify today, other emitters later) from
OKF frontmatter. Publication is default-deny:

1. ``privacy: public`` only (never ``internal`` / ``private`` /
   ``review-before-public``).
2. ``type`` must be in :data:`PUBLISHABLE_TYPES` (curated, not "every public
   doc"). The bulk migration marked ~238 files ``public``; most of those are
   issue bodies, audit ledgers, and review records that do not belong on a
   public site.
3. ``status`` must not be in :data:`STALE_STATUSES` unless ``--include-stale``
   is passed, in which case the doc renders with a staleness banner.

Run ``--check`` in CI to fail if the published tree drifts from the corpus.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# Reuse the OKF parser so the site can never disagree with the validator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from docs_okf import (  # noqa: E402
    DEFAULT_DOCS_ROOT,
    PROJECT_ROOT,
    display_path,
    first_h1,
    markdown_files,
    read_okf_document,
)

DEFAULT_SITE_ROOT = PROJECT_ROOT / "site"

# Curated allow-list of document types that belong on a public docs site.
# Everything else (Issue Body, Audit Ledger, Review Record, Audit Report,
# Review Lane, Test Report, Migration Plan, ...) is internal by nature.
PUBLISHABLE_TYPES = {
    "Documentation Index",
    "User Guide",
    "Guide",
    "Runbook",
    "Use Case",
    "Principle Document",
    "Vision Document",
    "Design Note",
    "Architecture Spec",
    "Generated Reference",
}

# Statuses that should not be published without an explicit opt-in + banner.
STALE_STATUSES = {"needs-revalidation", "snapshot", "historical", "aspirational"}

STATUS_BANNER = {
    "needs-revalidation": (
        "warning",
        "This page has not been re-verified against the current codebase and may "
        "be out of date.",
    ),
    "snapshot": (
        "info",
        "This page is a point-in-time snapshot and is not maintained as living "
        "documentation.",
    ),
    "historical": (
        "info",
        "This page is preserved for historical context and does not describe the "
        "current system.",
    ),
    "aspirational": (
        "warning",
        "This page describes intended/future behavior that may not be implemented "
        "yet.",
    ),
}

# Order of nav groups in the rendered site. Types not listed fall to the end.
GROUP_ORDER = [
    "Documentation Index",
    "Vision Document",
    "Principle Document",
    "User Guide",
    "Guide",
    "Use Case",
    "Runbook",
    "Architecture Spec",
    "Design Note",
    "Generated Reference",
]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def select_pages(
    docs_root: Path,
    *,
    include_stale: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Return ``(pages, skipped)`` where each page carries its OKF metadata.

    ``skipped`` is a list of ``(path, reason)`` so the build is never a silent
    truncation — callers log it.
    """
    pages: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    for path in markdown_files(docs_root):
        rel = display_path(path)
        doc, error = read_okf_document(path)
        if error:
            skipped.append((rel, f"frontmatter error: {error}"))
            continue
        if doc is None:
            skipped.append((rel, "no OKF frontmatter"))
            continue

        fm = doc.frontmatter
        privacy = str(fm.get("privacy", "")).strip().lower()
        doc_type = str(fm.get("type", "")).strip()
        status = str(fm.get("status", "")).strip().lower()

        if privacy != "public":
            skipped.append((rel, f"privacy={privacy or 'unset'}"))
            continue
        if doc_type not in PUBLISHABLE_TYPES:
            skipped.append((rel, f"type not publishable: {doc_type or 'unset'}"))
            continue
        if status in STALE_STATUSES and not include_stale:
            skipped.append((rel, f"status={status} (use --include-stale)"))
            continue

        pages.append(
            {
                "src": path,
                "rel": rel,
                "type": doc_type,
                "status": status,
                "title": fm.get("title") or first_h1(doc.body) or path.stem,
                "description": fm.get("description", ""),
                "body": doc.body,
                "frontmatter": fm,
            }
        )

    return pages, skipped


# --------------------------------------------------------------------------- #
# Mintlify emitter
# --------------------------------------------------------------------------- #
def _site_slug(rel: str) -> str:
    """``docs/user-documentation/FOO.md`` -> ``user-documentation/FOO``."""
    p = Path(rel)
    if p.parts and p.parts[0] == "docs":
        p = Path(*p.parts[1:])
    return p.with_suffix("").as_posix()


def _yaml_quote(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_mintlify_page(page: dict[str, Any]) -> str:
    """Rewrite an OKF doc into a Mintlify MDX page (frontmatter + banner)."""
    lines = ["---", f"title: {_yaml_quote(page['title'])}"]
    if page["description"]:
        lines.append(f"description: {_yaml_quote(page['description'])}")
    lines.append("---")
    lines.append("")

    banner = STATUS_BANNER.get(page["status"])
    if banner:
        kind, text = banner
        tag = {"warning": "Warning", "info": "Info"}.get(kind, "Note")
        lines.append(f"<{tag}>{text}</{tag}>")
        lines.append("")

    body = page["body"].lstrip("\n")
    # Drop a leading duplicate H1 — Mintlify renders the title from frontmatter.
    body_lines = body.splitlines()
    if body_lines and body_lines[0].strip().startswith("# "):
        body_lines = body_lines[1:]
        while body_lines and not body_lines[0].strip():
            body_lines = body_lines[1:]
    lines.append("\n".join(body_lines).rstrip())
    lines.append("")
    return "\n".join(lines)


def build_navigation(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[str]] = {}
    for page in pages:
        by_group.setdefault(page["type"], []).append(_site_slug(page["rel"]))

    def group_key(name: str) -> tuple[int, str]:
        return (GROUP_ORDER.index(name) if name in GROUP_ORDER else len(GROUP_ORDER), name)

    nav = []
    for group in sorted(by_group, key=group_key):
        nav.append({"group": group, "pages": sorted(by_group[group])})
    return nav


def render_mint_json(pages: list[dict[str, Any]]) -> str:
    config = {
        "$schema": "https://mintlify.com/schema.json",
        "name": "Kestrel Sovereign",
        "colors": {"primary": "#0F766E", "light": "#14B8A6", "dark": "#0F766E"},
        "navigation": build_navigation(pages),
        "footerSocials": {"github": "https://github.com/KestrelSovereignAI"},
    }
    return json.dumps(config, indent=2) + "\n"


# --------------------------------------------------------------------------- #
# Build / check
# --------------------------------------------------------------------------- #
EMITTERS = {"mintlify": (render_mint_json, render_mintlify_page, "mdx")}


def build_site(
    *,
    docs_root: Path,
    site_root: Path,
    emitter: str,
    include_stale: bool,
    check: bool,
) -> int:
    render_manifest, render_page, ext = EMITTERS[emitter]
    pages, skipped = select_pages(docs_root, include_stale=include_stale)

    if not pages:
        print("ERROR: no publishable pages selected", file=sys.stderr)
        return 1

    # Stage into a temp tree, then either diff (check) or swap in (build).
    staging = site_root.parent / (site_root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    (staging / ("mint.json" if emitter == "mintlify" else "manifest.json")).write_text(
        render_manifest(pages), encoding="utf-8"
    )
    for page in pages:
        out = staging / (_site_slug(page["rel"]) + "." + ext)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_page(page), encoding="utf-8")

    print(f"Selected {len(pages)} pages for the {emitter} site; skipped {len(skipped)}.")
    for rel, reason in skipped[:15]:
        print(f"  skip {rel}: {reason}")
    if len(skipped) > 15:
        print(f"  ... and {len(skipped) - 15} more skipped (run with -v to see all)")

    if check:
        drift = _diff_trees(site_root, staging)
        shutil.rmtree(staging)
        if drift:
            for line in drift:
                print(f"ERROR: {line}", file=sys.stderr)
            print(
                f"Published site is stale ({len(drift)} differences). "
                "Run: uv run python scripts/build_docs_site.py build",
                file=sys.stderr,
            )
            return 1
        print(f"Published site at {display_path(site_root)} is current.")
        return 0

    if site_root.exists():
        shutil.rmtree(site_root)
    staging.rename(site_root)
    print(f"Wrote site to {display_path(site_root)}")
    return 0


def _diff_trees(current: Path, staged: Path) -> list[str]:
    drift: list[str] = []
    if not current.exists():
        return [f"{display_path(current)} does not exist"]

    staged_files = {p.relative_to(staged) for p in staged.rglob("*") if p.is_file()}
    current_files = {p.relative_to(current) for p in current.rglob("*") if p.is_file()}

    for rel in sorted(staged_files - current_files):
        drift.append(f"missing from published site: {rel.as_posix()}")
    for rel in sorted(current_files - staged_files):
        drift.append(f"stale file in published site: {rel.as_posix()}")
    for rel in sorted(staged_files & current_files):
        if not filecmp.cmp(staged / rel, current / rel, shallow=False):
            drift.append(f"content drift: {rel.as_posix()}")
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"), help="build or verify the site")
    parser.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS_ROOT)
    parser.add_argument("--site-root", type=Path, default=DEFAULT_SITE_ROOT)
    parser.add_argument("--emitter", choices=tuple(EMITTERS), default="mintlify")
    parser.add_argument(
        "--include-stale",
        action="store_true",
        help="publish needs-revalidation/snapshot docs with a staleness banner",
    )
    args = parser.parse_args()

    return build_site(
        docs_root=args.docs_root.resolve(),
        site_root=args.site_root.resolve(),
        emitter=args.emitter,
        include_stale=args.include_stale,
        check=args.command == "check",
    )


if __name__ == "__main__":
    raise SystemExit(main())
