#!/usr/bin/env python3
"""Verify OKF docs against repo evidence and render policy.

This is intentionally stricter than OKF validation. OKF answers "is this a
machine-readable concept?" This verifier answers "can we trust and route it?"
by checking local links, repo/code references, recent merged PRs, and public
render eligibility.
"""

from __future__ import annotations

import argparse
import difflib
import functools
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Load the sibling ``docs_okf`` module by its own file path rather than via
# ``from scripts import docs_okf``. The bare package name ``scripts`` is not
# unique across the venv — a sibling editable install (e.g.
# ``kestrel-feature-workflows``) can ship its own top-level ``scripts`` package
# and shadow this repo's, making the package-name import resolve to the wrong
# module. Path-based loading is collision-proof and independent of sys.path
# ordering.
import importlib.util as _importlib_util

_DOCS_OKF_PATH = Path(__file__).resolve().parent / "docs_okf.py"
_docs_okf_spec = _importlib_util.spec_from_file_location(
    "kestrel_scripts_docs_okf", _DOCS_OKF_PATH
)
docs_okf = _importlib_util.module_from_spec(_docs_okf_spec)
# Register before executing so machinery that resolves a class's module by name
# (e.g. ``dataclasses`` inspecting ``sys.modules[cls.__module__]``) finds it.
sys.modules[_docs_okf_spec.name] = docs_okf
_docs_okf_spec.loader.exec_module(docs_okf)


PROJECT_ROOT = docs_okf.PROJECT_ROOT
DOCS_ROOT = docs_okf.DEFAULT_DOCS_ROOT
DEFAULT_SINCE = "2026-06-01"
DEFAULT_REPORT = DOCS_ROOT / "generated" / "DOC_VERIFICATION.md"
DEFAULT_MANIFEST = DOCS_ROOT / "generated" / "RENDER_MANIFEST.json"
DEFAULT_IGNORED_PRS = {1821}
GENERATED_TIMESTAMP = "2026-06-24T00:00:00Z"

INTERNAL_TYPES = {
    "Audit Ledger",
    "Audit Report",
    "Demo Script",
    "Issue Body",
    "Review Lane",
    "Review Record",
}
ARCHIVE_STATUSES = {"historical", "snapshot"}
NON_PUBLIC_PRIVACY = {"internal", "private", "review-before-public"}
PATH_EXTENSIONS = {
    ".cjs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
PATHISH_RE = re.compile(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.@/-]+\.[A-Za-z0-9]+)(?![A-Za-z0-9_.-])")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
FRONTMATTER_PATH_RE = re.compile(r"^(?:resource|source|generator):\s+(.+)$", re.MULTILINE)
PR_NUMBER_RE = re.compile(r"#(\d+)|\(#(\d+)\)")


@dataclass(frozen=True)
class RecentPr:
    sha: str
    title: str
    merged_at: str
    number: int | None
    files: tuple[str, ...]


@dataclass(frozen=True)
class DocVerification:
    path: str
    title: str
    doc_type: str
    status: str
    privacy: str
    render: str
    missing_links: tuple[str, ...]
    missing_code_refs: tuple[str, ...]
    existing_code_refs: tuple[str, ...]
    recent_prs: tuple[dict[str, Any], ...]
    findings: tuple[str, ...]


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True)


@functools.cache
def tracked_paths() -> frozenset[str]:
    return frozenset(run_git(["ls-files"]).splitlines())


@functools.cache
def tracked_dirs() -> frozenset[str]:
    dirs: set[str] = set()
    for path in tracked_paths():
        parts = Path(path).parts
        for index in range(1, len(parts)):
            dirs.add(Path(*parts[:index]).as_posix())
    return frozenset(dirs)


def repo_path_exists(candidate: str) -> bool:
    return candidate in tracked_paths() or candidate in tracked_dirs()


def parse_pr_number(title: str) -> int | None:
    matches = PR_NUMBER_RE.findall(title)
    if not matches:
        return None
    first = matches[-1]
    value = first[0] or first[1]
    return int(value) if value else None


def recent_prs(since: str, *, ignored_prs: set[int]) -> list[RecentPr]:
    raw = run_git([
        "log",
        "--first-parent",
        f"--since={since}",
        "--pretty=%H%x1f%s%x1f%cI",
    ])
    prs: list[RecentPr] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        sha, title, merged_at = line.split("\x1f", 2)
        files = tuple(
            item
            for item in run_git([
                "show",
                "--first-parent",
                "--name-only",
                "--pretty=format:",
                sha,
            ]).splitlines()
            if item.strip()
        )
        pr_number = parse_pr_number(title)
        if pr_number is None or pr_number in ignored_prs:
            continue
        prs.append(
            RecentPr(
                sha=sha,
                title=title,
                merged_at=merged_at,
                number=pr_number,
                files=files,
            )
        )
    return prs


def strip_link_target(target: str) -> str:
    target = target.strip()
    if " " in target and not target.startswith("<"):
        target = target.split(" ", 1)[0]
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return target.split("#", 1)[0]


def resolve_doc_link(doc_path: Path, target: str) -> Path | None:
    target = strip_link_target(target)
    if not target or target.startswith(("#", "http://", "https://", "mailto:", "tel:")):
        return None
    if target.startswith("/"):
        return (PROJECT_ROOT / target.lstrip("/")).resolve()
    relative = (doc_path.parent / target).resolve()
    if relative.exists():
        return relative
    root_relative = (PROJECT_ROOT / target).resolve()
    if root_relative.exists():
        return root_relative
    return relative


def missing_markdown_links(doc: docs_okf.OkfDocument) -> tuple[str, ...]:
    missing: list[str] = []
    for raw_target in MARKDOWN_LINK_RE.findall(doc.body):
        resolved = resolve_doc_link(doc.path, raw_target)
        if resolved is None:
            continue
        try:
            rel_target = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            missing.append(raw_target)
            continue
        if not repo_path_exists(rel_target):
            missing.append(raw_target)
    return tuple(sorted(set(missing)))


def candidate_paths(doc: docs_okf.OkfDocument) -> set[str]:
    candidates: set[str] = set()
    text = "\n".join([doc.body, "\n".join(FRONTMATTER_PATH_RE.findall(doc.path.read_text(encoding="utf-8")))])
    for match in PATHISH_RE.findall(text):
        cleaned = match.strip("`'\".,);:")
        if cleaned.startswith(("http://", "https://")):
            continue
        if Path(cleaned).suffix not in PATH_EXTENSIONS:
            continue
        if cleaned.startswith("/"):
            cleaned = cleaned.lstrip("/")
        candidates.add(cleaned)
    for raw_target in MARKDOWN_LINK_RE.findall(doc.body):
        target = strip_link_target(raw_target)
        if not target or target.startswith(("http://", "https://", "mailto:", "tel:", "#")):
            continue
        resolved = resolve_doc_link(doc.path, target)
        if resolved is None:
            continue
        try:
            candidates.add(resolved.relative_to(PROJECT_ROOT).as_posix())
        except ValueError:
            pass
    return candidates


def classify_code_refs(doc: docs_okf.OkfDocument) -> tuple[tuple[str, ...], tuple[str, ...]]:
    existing: set[str] = set()
    missing: set[str] = set()
    for candidate in candidate_paths(doc):
        if candidate.startswith("docs/"):
            continue
        docs_relative_path = "docs/" + candidate
        if repo_path_exists(candidate):
            existing.add(candidate)
        elif repo_path_exists(docs_relative_path):
            continue
        elif candidate.startswith((
            ".github/",
            "demos/",
            "docker/",
            "kestrel_sovereign/",
            "scripts/",
            "tests/",
        )):
            missing.add(candidate)
    return tuple(sorted(existing)), tuple(sorted(missing))


def report_link_for(path: str) -> str:
    target = "../" + path.removeprefix("docs/")
    if any(char.isspace() for char in target):
        return f"<{target}>"
    return target


def render_channel(doc: docs_okf.OkfDocument) -> str:
    fm = doc.frontmatter
    privacy = str(fm.get("privacy") or "").strip()
    status = str(fm.get("status") or "").strip()
    doc_type = str(fm.get("type") or "").strip()
    if privacy == "private":
        return "exclude"
    if privacy in NON_PUBLIC_PRIVACY:
        return "internal"
    if status in ARCHIVE_STATUSES:
        return "archive"
    if doc_type in INTERNAL_TYPES:
        return "internal"
    if status == "needs-revalidation":
        return "internal"
    return "public"


def is_specific_prefix(path: str) -> bool:
    parts = Path(path).parts
    return len(parts) >= 2 and path not in {
        ".github/workflows",
        "demos",
        "docker",
        "kestrel_sovereign",
        "scripts",
        "tests",
    }


def relevant_prs(doc: docs_okf.OkfDocument, existing_refs: tuple[str, ...], prs: list[RecentPr]) -> tuple[dict[str, Any], ...]:
    doc_rel = doc.rel_path
    ref_files = set(existing_refs)
    ref_dirs = {
        ref
        for ref in existing_refs
        if ref in tracked_dirs() and is_specific_prefix(ref)
    }
    ref_prefixes = {
        ref.rsplit("/", 1)[0]
        for ref in existing_refs
        if "/" in ref and is_specific_prefix(ref.rsplit("/", 1)[0])
    }
    related: list[dict[str, Any]] = []
    for pr in prs:
        reason = ""
        files = set(pr.files)
        if doc_rel in files:
            reason = "changed_doc"
        elif ref_files and files.intersection(ref_files):
            reason = "touched_referenced_file"
        elif ref_dirs and any(any(path == ref or path.startswith(ref + "/") for ref in ref_dirs) for path in files):
            reason = "touched_referenced_dir"
        elif ref_prefixes and any(any(path == prefix or path.startswith(prefix + "/") for prefix in ref_prefixes) for path in files):
            reason = "touched_referenced_code"
        if not reason:
            continue
        related.append(
            {
                "number": pr.number,
                "title": pr.title,
                "merged_at": pr.merged_at,
                "sha": pr.sha[:12],
                "reason": reason,
            }
        )
    return tuple(related[:8])


def verify_docs(
    *,
    since: str,
    ignored_prs: set[int] | None = None,
    with_activity: bool = False,
) -> list[DocVerification]:
    """Verify every OKF doc.

    The committed ledger/manifest are a pure function of doc *content* (render
    routing, local links, code references), so by default this skips the
    HEAD-relative recent-PR query entirely — that keeps the generated artifacts
    deterministic across unrelated merges and needs no git history. Pass
    ``with_activity=True`` for the live review-queue view (``audit --activity``).
    """
    prs = recent_prs(since, ignored_prs=ignored_prs or set()) if with_activity else []
    verifications: list[DocVerification] = []
    for path in docs_okf.markdown_files(DOCS_ROOT):
        if path.name in docs_okf.RESERVED_NAMES:
            continue
        doc, error = docs_okf.read_okf_document(path)
        if error or doc is None:
            continue
        # Generated docs (REPO_MAP.md, FEATURES_*.md, …) are machine output with
        # a "do not edit by hand" body. Their link/code-ref sets are a volatile
        # function of repo size — REPO_MAP alone embeds ~1400 code refs that grow
        # every time the tree grows — and they are regenerated out-of-band (e.g.
        # the nightly repo-map job). Embedding those sets in the committed
        # ledger/manifest restales the artifacts on every regen and reds out
        # ``test_docs_verify`` on main (#1971). Verifying their links is also
        # meaningless: you cannot fix a generated file by editing it. So we keep
        # the doc in render routing but emit empty, stable link/ref sets.
        if bool(doc.frontmatter.get("generated")):
            existing_refs, missing_refs, missing_links = (), (), ()
        else:
            existing_refs, missing_refs = classify_code_refs(doc)
            missing_links = missing_markdown_links(doc)
        related_prs = relevant_prs(doc, existing_refs, prs)
        render = render_channel(doc)
        status = str(doc.frontmatter.get("status") or "")
        findings: list[str] = []
        if missing_links:
            findings.append("missing_local_links")
        if missing_refs:
            findings.append("missing_code_refs")
        if render == "public" and status == "needs-revalidation":
            findings.append("public_doc_needs_revalidation")
        if related_prs and status in {"active", "implemented", "generated"}:
            findings.append("recent_prs_should_be_reviewed")
        verifications.append(
            DocVerification(
                path=doc.rel_path,
                title=str(doc.frontmatter.get("title") or docs_okf.first_h1(doc.body)),
                doc_type=str(doc.frontmatter.get("type") or ""),
                status=status,
                privacy=str(doc.frontmatter.get("privacy") or ""),
                render=render,
                missing_links=missing_links,
                missing_code_refs=missing_refs,
                existing_code_refs=existing_refs,
                recent_prs=related_prs,
                findings=tuple(findings),
            )
        )
    return sorted(verifications, key=lambda item: item.path)


# Findings derived from HEAD-relative git activity rather than doc content.
# These are excluded from the committed artifacts (they would restale on every
# unrelated merge) and surfaced live via ``audit --activity`` instead.
ACTIVITY_FINDINGS = frozenset({"recent_prs_should_be_reviewed"})


def content_findings(item: DocVerification) -> list[str]:
    return [f for f in item.findings if f not in ACTIVITY_FINDINGS]


def verification_to_dict(item: DocVerification) -> dict[str, Any]:
    # Content-derived only: the routing manifest must not embed recent-PR data,
    # which is HEAD-relative and would make the committed file perpetually stale.
    return {
        "path": item.path,
        "title": item.title,
        "type": item.doc_type,
        "status": item.status,
        "privacy": item.privacy,
        "render": item.render,
        "missing_links": list(item.missing_links),
        "missing_code_refs": list(item.missing_code_refs),
        "existing_code_refs": list(item.existing_code_refs),
        "findings": content_findings(item),
    }


def render_report(items: list[DocVerification]) -> str:
    """Render the committed verification ledger from doc *content* only.

    Deterministic across unrelated merges: it embeds no HEAD-relative data, so
    ``audit --check`` only trips when the docs themselves change. The recent-PR
    review queue lives in the live ``audit --activity`` view, not here.
    """
    by_render: dict[str, int] = {}
    for item in items:
        by_render[item.render] = by_render.get(item.render, 0) + 1
    missing_link_count = sum(1 for item in items if item.missing_links)
    missing_code_count = sum(1 for item in items if item.missing_code_refs)
    lines = [
        "---",
        "type: Generated Reference",
        "title: Documentation Verification Ledger",
        "description: Generated verification report for OKF docs covering renderer routing, local links, and code references.",
        "resource: /docs/generated/DOC_VERIFICATION.md",
        "tags: [docs, okf, verification, render-manifest]",
        "status: generated",
        "owner: documentation",
        "canonical: false",
        "generated: true",
        "source: /scripts/docs_verify.py",
        f"timestamp: {GENERATED_TIMESTAMP}",
        "privacy: internal",
        "---",
        "",
        "# Documentation Verification Ledger",
        "",
        "Generated by `uv run python scripts/docs_verify.py audit`. Recent-PR "
        "review relevance is a live signal; run `audit --activity` for it.",
        "",
        f"- OKF docs checked: {len(items)}.",
        f"- Render routing: {', '.join(f'{key}={by_render[key]}' for key in sorted(by_render))}.",
        f"- Docs with missing local links: {missing_link_count}.",
        f"- Docs with missing repo/code references: {missing_code_count}.",
        "",
        "## Findings",
        "",
    ]
    finding_rows = [item for item in items if content_findings(item)]
    if finding_rows:
        lines.extend(["| Path | Render | Status | Findings |", "|---|---|---|---|"])
        for item in finding_rows:
            lines.append(
                f"| [{docs_okf.markdown_table_escape(item.path)}]({report_link_for(item.path)}) | "
                f"{docs_okf.markdown_table_escape(item.render)} | "
                f"{docs_okf.markdown_table_escape(item.status)} | "
                f"{docs_okf.markdown_table_escape(', '.join(content_findings(item)))} |"
            )
    else:
        lines.append("No verification findings.")
    lines.extend(["", "## All Documents", ""])
    lines.extend(["| Path | Render | Status | Type | Code Refs |", "|---|---|---|---|---:|"])
    for item in items:
        lines.append(
            f"| [{docs_okf.markdown_table_escape(item.path)}]({report_link_for(item.path)}) | "
            f"{docs_okf.markdown_table_escape(item.render)} | "
            f"{docs_okf.markdown_table_escape(item.status)} | "
            f"{docs_okf.markdown_table_escape(item.doc_type)} | "
            f"{len(item.existing_code_refs)} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_activity(items: list[DocVerification], *, since: str) -> str:
    """Live, non-committed view: docs with relevant recent-PR activity.

    Computed on demand from current git history; intentionally NOT written to a
    committed file because it is HEAD-relative and would never stay current.
    """
    active = [item for item in items if item.recent_prs]
    lines = [
        f"# Documentation Activity (recent PRs since {since})",
        "",
        f"- Docs with relevant recent PRs: {len(active)}.",
        "",
    ]
    if not active:
        lines.append("No docs have relevant recent-PR activity in this window.")
        return "\n".join(lines) + "\n"
    lines.extend(["| Path | Render | Status | Findings | Recent PRs |", "|---|---|---|---|---|"])
    for item in active:
        prs = ", ".join(
            f"#{pr['number']}" if pr.get("number") else pr["sha"]
            for pr in item.recent_prs[:5]
        )
        lines.append(
            f"| {docs_okf.markdown_table_escape(item.path)} | "
            f"{docs_okf.markdown_table_escape(item.render)} | "
            f"{docs_okf.markdown_table_escape(item.status)} | "
            f"{docs_okf.markdown_table_escape(', '.join(item.findings))} | "
            f"{docs_okf.markdown_table_escape(prs)} |"
        )
    return "\n".join(lines) + "\n"


def render_manifest(items: list[DocVerification]) -> str:
    payload = {
        "generated_by": "uv run python scripts/docs_verify.py manifest",
        "docs_root": "docs",
        "routes": [verification_to_dict(item) for item in items],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_or_check(path: Path, expected: str, *, check: bool) -> int:
    if check:
        if not path.exists():
            print(f"ERROR: {docs_okf.display_path(path)} does not exist", file=sys.stderr)
            return 1
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            diff = "\n".join(
                difflib.unified_diff(
                    actual.splitlines(),
                    expected.splitlines(),
                    fromfile=docs_okf.display_path(path),
                    tofile="generated",
                    lineterm="",
                )
            )
            print(f"ERROR: {diff}", file=sys.stderr)
            return 1
        print(f"{docs_okf.display_path(path)} is current.")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")
    print(f"Wrote {docs_okf.display_path(path)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Generate the documentation verification ledger")
    audit.add_argument("--since", default=DEFAULT_SINCE, help="Git --since value for recent PR relevance")
    audit.add_argument(
        "--ignore-pr",
        action="append",
        type=int,
        default=sorted(DEFAULT_IGNORED_PRS),
        help="PR number to ignore as metadata-only or otherwise non-content-changing",
    )
    audit.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    audit.add_argument("--check", action="store_true", help="Fail if the generated report is stale")
    audit.add_argument("--format", choices=("markdown", "json"), default="markdown")
    audit.add_argument(
        "--activity",
        action="store_true",
        help="Print the live recent-PR review queue to stdout (never committed)",
    )

    manifest = subparsers.add_parser("manifest", help="Generate renderer routing manifest")
    manifest.add_argument("--since", default=DEFAULT_SINCE, help="Git --since value for recent PR relevance")
    manifest.add_argument(
        "--ignore-pr",
        action="append",
        type=int,
        default=sorted(DEFAULT_IGNORED_PRS),
        help="PR number to ignore as metadata-only or otherwise non-content-changing",
    )
    manifest.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    manifest.add_argument("--check", action="store_true", help="Fail if the generated manifest is stale")

    args = parser.parse_args()
    ignored_prs = set(args.ignore_pr or [])

    if args.command == "audit" and args.activity:
        # Live, non-committed view — needs git history for the recent-PR window.
        items = verify_docs(since=args.since, ignored_prs=ignored_prs, with_activity=True)
        print(render_activity(items, since=args.since), end="")
        return 0

    # Committed artifacts: content-only, no git history needed, deterministic.
    items = verify_docs(since=args.since, ignored_prs=ignored_prs)
    if args.command == "audit":
        if args.format == "json":
            rendered = json.dumps([verification_to_dict(item) for item in items], indent=2, sort_keys=True) + "\n"
        else:
            rendered = render_report(items)
        return write_or_check(args.output, rendered, check=args.check)
    if args.command == "manifest":
        return write_or_check(args.output, render_manifest(items), check=args.check)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
