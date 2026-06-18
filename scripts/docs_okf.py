#!/usr/bin/env python3
"""Validate and inventory Kestrel's Open Knowledge Format documents.

Phase 0 is intentionally opt-in: by default, validation checks markdown files
that already have YAML frontmatter. Use ``--all`` to audit the whole corpus and
report files that still need OKF metadata.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_ROOT = PROJECT_ROOT / "docs"
RESERVED_NAMES = {"index.md", "log.md"}
GENERATED_FEATURE_DOCS = {
    "developer": DEFAULT_DOCS_ROOT / "generated" / "FEATURES_developer.md",
    "user": DEFAULT_DOCS_ROOT / "generated" / "FEATURES_user.md",
    "investor": DEFAULT_DOCS_ROOT / "generated" / "FEATURES_investor.md",
}
DEFAULT_GENERATED_INDEX_ROOTS = [
    DEFAULT_DOCS_ROOT / "audit",
    DEFAULT_DOCS_ROOT / "generated",
    DEFAULT_DOCS_ROOT / "architecture",
]


@dataclass(frozen=True)
class OkfDocument:
    path: Path
    frontmatter: dict[str, Any]
    body: str

    @property
    def rel_path(self) -> str:
        return display_path(self.path)


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def markdown_files(root: Path = DEFAULT_DOCS_ROOT) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if path.is_file() and ".pytest_cache" not in path.parts
    )


def split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str, str | None]:
    """Return ``(frontmatter, body, error)`` for a markdown document."""
    if not text.startswith("---\n"):
        return None, text, None

    end = text.find("\n---\n", 4)
    if end == -1:
        return None, text, "frontmatter opening marker has no closing marker"

    raw = text[4:end]
    body = text[end + len("\n---\n") :]
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        return None, body, f"frontmatter is not valid YAML: {exc}"

    if not isinstance(parsed, dict):
        return None, body, "frontmatter must be a YAML mapping"
    return parsed, body, None


def read_okf_document(path: Path) -> tuple[OkfDocument | None, str | None]:
    text = path.read_text(encoding="utf-8")
    frontmatter, body, error = split_frontmatter(text)
    if error:
        return None, error
    if frontmatter is None:
        return None, None
    return OkfDocument(path=path, frontmatter=frontmatter, body=body), None


def first_h1(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def validate_document(doc: OkfDocument) -> list[str]:
    errors: list[str] = []
    fm = doc.frontmatter

    doc_type = fm.get("type")
    if not isinstance(doc_type, str) or not doc_type.strip():
        errors.append("missing required non-empty frontmatter field: type")

    for field in ("title", "description"):
        value = fm.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"frontmatter field must be a non-empty string when present: {field}")

    tags = fm.get("tags")
    if tags is not None and not isinstance(tags, list):
        errors.append("frontmatter field must be a list when present: tags")

    for field in ("canonical", "generated"):
        value = fm.get(field)
        if value is not None and not isinstance(value, bool):
            errors.append(f"frontmatter field must be a boolean when present: {field}")

    if not first_h1(doc.body) and doc.path.name not in RESERVED_NAMES:
        errors.append("missing H1 heading in document body")

    return errors


def validate_files(paths: list[Path], *, include_all: bool = False) -> int:
    checked = 0
    skipped = 0
    failures: list[str] = []

    for path in paths:
        doc, error = read_okf_document(path)
        rel = display_path(path)
        if error:
            failures.append(f"{rel}: {error}")
            continue
        if doc is None:
            if include_all and path.name not in RESERVED_NAMES:
                failures.append(f"{rel}: missing OKF frontmatter")
            else:
                skipped += 1
            continue

        checked += 1
        for doc_error in validate_document(doc):
            failures.append(f"{rel}: {doc_error}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(f"Checked {checked} OKF docs; skipped {skipped}; failures {len(failures)}.", file=sys.stderr)
        return 1

    print(f"OKF validation passed: checked {checked}; skipped {skipped}.")
    return 0


def build_inventory(paths: list[Path], *, include_all: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        doc, error = read_okf_document(path)
        if error:
            rows.append(
                {
                    "path": display_path(path),
                    "okf": False,
                    "error": error,
                }
            )
            continue
        if doc is None:
            if include_all:
                text = path.read_text(encoding="utf-8", errors="ignore")
                rows.append(
                    {
                        "path": display_path(path),
                        "okf": False,
                        "title": first_h1(text),
                    }
                )
            continue

        fm = doc.frontmatter
        rows.append(
            {
                "path": doc.rel_path,
                "okf": True,
                "type": fm.get("type", ""),
                "title": fm.get("title") or first_h1(doc.body),
                "description": fm.get("description", ""),
                "status": fm.get("status", ""),
                "owner": fm.get("owner", ""),
                "canonical": fm.get("canonical", False),
                "generated": fm.get("generated", False),
                "tags": fm.get("tags", []),
            }
        )
    return rows


def print_inventory(rows: list[dict[str, Any]], *, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    print("| Path | OKF | Type | Status | Title |")
    print("|---|---:|---|---|---|")
    for row in rows:
        print(
            "| {path} | {okf} | {type} | {status} | {title} |".format(
                path=row.get("path", ""),
                okf="yes" if row.get("okf") else "no",
                type=row.get("type", ""),
                status=row.get("status", ""),
                title=str(row.get("title", "")).replace("|", "\\|"),
            )
        )


def markdown_table_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def metadata_scalar(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")


def relative_link(from_dir: Path, target: str) -> str:
    target_path = PROJECT_ROOT / target
    try:
        return Path(target_path).relative_to(from_dir).as_posix()
    except ValueError:
        return target


def render_index(root: Path) -> str:
    root = root.resolve()
    rows = build_inventory(markdown_files(root), include_all=True)
    okf_rows = [row for row in rows if row.get("okf")]
    pending_rows = [
        row for row in rows
        if not row.get("okf") and Path(str(row.get("path", ""))).name not in RESERVED_NAMES
    ]
    rel_root = display_path(root)

    lines = [
        f"# OKF Index - {rel_root}",
        "",
        "Generated by `uv run python scripts/docs_okf.py index`.",
        "",
        f"- OKF concepts: {len(okf_rows)}",
        f"- Pending markdown files: {len(pending_rows)}",
        "",
        "## OKF Concepts",
        "",
    ]
    if okf_rows:
        lines.extend(["| Path | Type | Status | Title |", "|---|---|---|---|"])
        for row in sorted(okf_rows, key=lambda item: str(item.get("path", ""))):
            path = str(row.get("path", ""))
            link = relative_link(root, path)
            lines.append(
                f"| [{markdown_table_escape(path)}]({link}) | "
                f"{markdown_table_escape(row.get('type'))} | "
                f"{markdown_table_escape(row.get('status'))} | "
                f"{markdown_table_escape(row.get('title'))} |"
            )
    else:
        lines.append("No OKF concepts have been opted in for this subtree yet.")

    lines.extend(["", "## Pending Markdown", ""])
    if pending_rows:
        lines.extend(["| Path | H1 |", "|---|---|"])
        for row in sorted(pending_rows, key=lambda item: str(item.get("path", ""))):
            path = str(row.get("path", ""))
            link = relative_link(root, path)
            lines.append(
                f"| [{markdown_table_escape(path)}]({link}) | "
                f"{markdown_table_escape(row.get('title'))} |"
            )
    else:
        lines.append("No pending markdown files in this subtree.")

    lines.append("")
    return "\n".join(lines)


def render_log(root: Path) -> str:
    root = root.resolve()
    rows = [
        row for row in build_inventory(markdown_files(root), include_all=False)
        if row.get("okf")
    ]
    rel_root = display_path(root)
    lines = [
        f"# OKF Log - {rel_root}",
        "",
        "Generated by `uv run python scripts/docs_okf.py log` from OKF frontmatter.",
        "",
    ]
    if not rows:
        lines.extend(["No OKF concepts in this subtree yet.", ""])
        return "\n".join(lines)

    lines.extend(["| Timestamp | Status | Type | Title | Path |", "|---|---|---|---|---|"])
    detailed_rows = []
    for path in markdown_files(root):
        doc, error = read_okf_document(path)
        if doc is None or error:
            continue
        fm = doc.frontmatter
        detailed_rows.append(
            {
                "timestamp": fm.get("timestamp", ""),
                "status": fm.get("status", ""),
                "type": fm.get("type", ""),
                "title": fm.get("title") or first_h1(doc.body),
                "path": doc.rel_path,
            }
        )
    for row in sorted(detailed_rows, key=lambda item: (metadata_scalar(item["timestamp"]), str(item["path"]))):
        link = relative_link(root, str(row["path"]))
        lines.append(
            f"| {markdown_table_escape(metadata_scalar(row['timestamp']))} | "
            f"{markdown_table_escape(row['status'])} | "
            f"{markdown_table_escape(row['type'])} | "
            f"{markdown_table_escape(row['title'])} | "
            f"[{markdown_table_escape(row['path'])}]({link}) |"
        )
    lines.append("")
    return "\n".join(lines)


def write_or_check_generated(
    roots: list[Path],
    *,
    filename: str,
    renderer: Any,
    check: bool,
) -> int:
    failures: list[str] = []
    for root in roots:
        root = root.resolve()
        output = root / filename
        expected = renderer(root)
        if check:
            if not output.exists():
                failures.append(f"{display_path(output)} does not exist")
                continue
            actual = output.read_text(encoding="utf-8")
            if actual != expected:
                diff = "\n".join(
                    difflib.unified_diff(
                        actual.splitlines(),
                        expected.splitlines(),
                        fromfile=display_path(output),
                        tofile="generated",
                        lineterm="",
                    )
                )
                failures.append(diff)
            continue
        output.write_text(expected, encoding="utf-8")
        print(f"Wrote {display_path(output)}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    if check:
        print(f"Generated {filename} files are current.")
    return 0


def check_generated_feature_docs() -> int:
    failures: list[str] = []
    for audience, path in GENERATED_FEATURE_DOCS.items():
        doc, error = read_okf_document(path)
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if error:
            failures.append(f"{rel}: {error}")
            continue
        if doc is None:
            failures.append(f"{rel}: missing OKF frontmatter")
            continue

        fm = doc.frontmatter
        expected = {
            "type": "Generated Reference",
            "generated": True,
            "source": "/KESTREL_FEATURES.md",
            "audience": audience,
            "generator": "scripts/generate_feature_docs.py",
        }
        for key, value in expected.items():
            if fm.get(key) != value:
                failures.append(f"{rel}: expected {key}={value!r}, got {fm.get(key)!r}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print("Generated feature docs have OKF metadata.")
    return 0


def resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.paths:
        resolved: list[Path] = []
        for raw_path in args.paths:
            path = Path(raw_path).resolve()
            if path.is_dir():
                resolved.extend(markdown_files(path))
            else:
                resolved.append(path)
        return sorted(resolved)
    return markdown_files(args.root.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate OKF frontmatter")
    validate.add_argument("paths", nargs="*", help="Specific markdown files to validate")
    validate.add_argument("--root", type=Path, default=DEFAULT_DOCS_ROOT, help="Docs root to scan")
    validate.add_argument("--all", action="store_true", help="Require frontmatter on every markdown file")

    inventory = subparsers.add_parser("inventory", help="Emit an OKF inventory")
    inventory.add_argument("paths", nargs="*", help="Specific markdown files to inventory")
    inventory.add_argument("--root", type=Path, default=DEFAULT_DOCS_ROOT, help="Docs root to scan")
    inventory.add_argument("--all", action="store_true", help="Include non-OKF markdown files")
    inventory.add_argument("--format", choices=("markdown", "json"), default="markdown")

    generated = subparsers.add_parser("check-generated", help="Check generated docs metadata")
    generated.set_defaults(check_generated=True)

    index = subparsers.add_parser("index", help="Generate OKF index.md files")
    index.add_argument("roots", nargs="*", type=Path, help="Directory roots to index")
    index.add_argument("--check", action="store_true", help="Fail if generated index.md files are stale")

    log = subparsers.add_parser("log", help="Generate OKF log.md files")
    log.add_argument("roots", nargs="*", type=Path, help="Directory roots to log")
    log.add_argument("--check", action="store_true", help="Fail if generated log.md files are stale")

    args = parser.parse_args()
    if args.command == "validate":
        return validate_files(resolve_input_paths(args), include_all=args.all)
    if args.command == "inventory":
        print_inventory(
            build_inventory(resolve_input_paths(args), include_all=args.all),
            fmt=args.format,
        )
        return 0
    if args.command == "check-generated":
        return check_generated_feature_docs()
    if args.command == "index":
        roots = args.roots or DEFAULT_GENERATED_INDEX_ROOTS
        return write_or_check_generated(roots, filename="index.md", renderer=render_index, check=args.check)
    if args.command == "log":
        roots = args.roots or DEFAULT_GENERATED_INDEX_ROOTS
        return write_or_check_generated(roots, filename="log.md", renderer=render_log, check=args.check)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
