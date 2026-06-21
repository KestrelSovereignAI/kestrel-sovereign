#!/usr/bin/env python3
"""Generate docs/audit/REPO_MAP.md — a file-tree + per-file purpose index.

Always-on context for the kestrel-agent GitHub App so it can answer
questions about files outside its anchored seed set without hallucinating.
See issue #791 for the rationale.

Usage:
    python scripts/generate_repo_map.py
    python scripts/generate_repo_map.py --output docs/audit/REPO_MAP.md
    python scripts/generate_repo_map.py --check  # exit non-zero if regeneration changes the file

Determinism: output is sorted by path; same input always produces same bytes.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "audit" / "REPO_MAP.md"

# Files we deliberately exclude from the map. These add no signal and cost tokens.
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    ".venv",
    "venv",
    ".claude",
    "demo-output",
    "playwright-report",
}

# Top-level docs we list under their own section, not inside docs/.
ROOT_DOCS_PRIORITY = [
    "README.md",
    "QUICKSTART.md",
    "KESTREL_FEATURES.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",
]

# Top-level directory order in the output, for reading flow.
DIR_ORDER = [
    "kestrel_sovereign",
    "kestrel_sdk",
    "endpoints",
    "server.py",
    "host.py",
    "main.py",
    "packages",
    "features",
    "scripts",
    "docker",
    "docs",
    "tests",
    "demos",
    "prompts",
    "sdk",
    "control-panel",
    ".github",
    ".devcontainer",
]


@dataclass
class FileEntry:
    path: str
    summary: str = ""
    symbols: list[str] = field(default_factory=list)


def tracked_files() -> list[str]:
    """Return all git-tracked files, sorted, relative to repo root."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(line for line in out.splitlines() if line.strip())


def is_excluded(path: str) -> bool:
    parts = Path(path).parts
    return any(p in EXCLUDE_DIRS for p in parts)


def first_sentence(text: str, max_chars: int = 200) -> str:
    """Pull a one-liner from a docstring or markdown blob."""
    text = text.strip()
    if not text:
        return ""
    # Take up to the first blank line, then up to the first sentence.
    paragraph = text.split("\n\n", 1)[0]
    paragraph = " ".join(paragraph.split())
    for terminator in (". ", "! ", "? "):
        if terminator in paragraph[:max_chars]:
            return paragraph.split(terminator, 1)[0].rstrip(".!?") + "."
    if len(paragraph) > max_chars:
        return paragraph[: max_chars - 1].rstrip() + "…"
    return paragraph


def summarize_python(path: Path) -> FileEntry:
    """Extract module docstring + public top-level symbol signatures."""
    entry = FileEntry(path=str(path.relative_to(REPO_ROOT)))
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return entry

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Don't fail the whole map on one broken file
        entry.summary = "(unparseable Python source)"
        return entry

    docstring = ast.get_docstring(tree) or ""
    entry.summary = first_sentence(docstring)

    for node in tree.body:
        name = getattr(node, "name", None)
        if not name or name.startswith("_"):
            continue
        if isinstance(node, ast.ClassDef):
            entry.symbols.append(f"class {name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = []
            for arg in node.args.args[:4]:
                args.append(arg.arg)
            if len(node.args.args) > 4:
                args.append("…")
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            entry.symbols.append(f"{prefix} {name}({', '.join(args)})")
    return entry


def summarize_markdown(path: Path) -> FileEntry:
    entry = FileEntry(path=str(path.relative_to(REPO_ROOT)))
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return entry

    lines = text.splitlines()
    body_start = 0
    title = ""
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            title = stripped.lstrip("# ").strip()
            body_start = idx + 1
            break
        body_start = idx
        break

    body = "\n".join(lines[body_start:]).strip()
    # Strip blockquote markers and front-matter for cleaner summaries
    body = body.replace("\n> ", " ")
    summary = first_sentence(body)
    if title and summary:
        entry.summary = f"{title} — {summary}"
    elif title:
        entry.summary = title
    else:
        entry.summary = summary
    return entry


def summarize_other(path: Path) -> FileEntry:
    entry = FileEntry(path=str(path.relative_to(REPO_ROOT)))
    if path.suffix in {".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"}:
        entry.summary = "(configuration)"
    elif path.suffix in {".sh", ".zsh", ".bash"}:
        try:
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines()[:10]:
                stripped = line.strip()
                if stripped.startswith("#") and not stripped.startswith("#!"):
                    entry.summary = first_sentence(stripped.lstrip("# "))
                    break
        except (OSError, UnicodeDecodeError):
            pass
    elif path.suffix in {".html", ".css", ".js", ".mjs", ".ts", ".tsx", ".jsx"}:
        entry.summary = f"({path.suffix.lstrip('.')} asset)"
    return entry


def summarize_path(rel_path: str) -> FileEntry:
    abs_path = REPO_ROOT / rel_path
    if not abs_path.is_file():
        return FileEntry(path=rel_path)
    suffix = abs_path.suffix.lower()
    if suffix == ".py":
        return summarize_python(abs_path)
    if suffix in {".md", ".markdown"}:
        return summarize_markdown(abs_path)
    return summarize_other(abs_path)


def group_by_dir(entries: list[FileEntry]) -> dict[str, list[FileEntry]]:
    groups: dict[str, list[FileEntry]] = defaultdict(list)
    for entry in entries:
        parts = Path(entry.path).parts
        if len(parts) == 1:
            groups["."].append(entry)
        else:
            top = parts[0]
            groups[top].append(entry)
    return groups


def fmt_entry(entry: FileEntry) -> str:
    summary = entry.summary or "—"
    line = f"- **{entry.path}** — {summary}"
    if entry.symbols:
        # Cap at 8 symbols to keep size predictable
        sig_list = entry.symbols[:8]
        if len(entry.symbols) > 8:
            sig_list.append("…")
        line += "\n  - " + "; ".join(f"`{s}`" for s in sig_list)
    return line


def fmt_section(title: str, entries: list[FileEntry], description: str = "") -> str:
    lines = [f"## {title}"]
    if description:
        lines.append("")
        lines.append(description)
    lines.append("")
    for entry in sorted(entries, key=lambda e: e.path):
        lines.append(fmt_entry(entry))
    return "\n".join(lines)


def build_map(files: list[str]) -> str:
    relevant: list[FileEntry] = []
    for rel in files:
        if is_excluded(rel):
            continue
        relevant.append(summarize_path(rel))

    groups = group_by_dir(relevant)

    py_count = sum(1 for e in relevant if e.path.endswith(".py"))
    md_count = sum(1 for e in relevant if e.path.endswith(".md"))
    other_count = len(relevant) - py_count - md_count

    today = dt.date.today().isoformat()
    timestamp = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    header = [
        "---",
        "type: Audit Ledger",
        "status: generated",
        "generated: true",
        f"timestamp: {timestamp}",
        "title: Kestrel Sovereign — Repo Map",
        "---",
        "",
        "# Kestrel Sovereign — Repo Map",
        "",
        "Auto-generated file-tree + per-file purpose index. Always-loaded context for the kestrel-agent",
        "GitHub App (issue #791). Do **not** edit by hand — regenerate via `python scripts/generate_repo_map.py`.",
        "",
        f"**Generated:** {today}",
        f"**Scope:** {len(relevant)} tracked files ({py_count} `.py`, {md_count} `.md`, {other_count} other). "
        "Excludes `__pycache__`, `node_modules`, `.venv`, `.claude`, build artifacts.",
        "",
        "**Format per file:** `path — one-line purpose` plus the public top-level Python symbols on the next line",
        "(classes and functions; private `_name` skipped).",
        "",
        "---",
        "",
    ]

    sections: list[str] = []

    # Top-level docs first, in priority order, then everything else at root
    root_entries = groups.pop(".", [])
    if root_entries:
        priority_lookup = {name: i for i, name in enumerate(ROOT_DOCS_PRIORITY)}
        priority = [e for e in root_entries if Path(e.path).name in priority_lookup]
        rest = [e for e in root_entries if Path(e.path).name not in priority_lookup]
        priority.sort(key=lambda e: priority_lookup[Path(e.path).name])
        rest.sort(key=lambda e: e.path)
        sections.append(
            fmt_section(
                "Top-level files",
                priority + rest,
                "Repo entry points and standard project files.",
            )
        )

    # Then directories in DIR_ORDER, then any leftover dirs alphabetically
    seen: set[str] = set()
    for top in DIR_ORDER:
        if top in groups:
            sections.append(fmt_section(f"`{top}/`", groups[top]))
            seen.add(top)

    for top in sorted(groups):
        if top in seen:
            continue
        sections.append(fmt_section(f"`{top}/`", groups[top]))

    return "\n".join(header) + "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path (default: docs/audit/REPO_MAP.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the generated content differs from the file on disk",
    )
    args = parser.parse_args()

    files = tracked_files()
    output = build_map(files)

    out_path = Path(args.output)
    if args.check:
        existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        if existing != output:
            sys.stderr.write(
                f"REPO_MAP.md is out of date. Re-run: python scripts/generate_repo_map.py\n"
            )
            return 1
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    size_kb = len(output.encode("utf-8")) / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB, ~{int(size_kb * 256)} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
