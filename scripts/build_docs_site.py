#!/usr/bin/env python3
"""Project the OKF docs corpus into a published documentation site.

This is a *projection*, not a migration: ``docs/`` stays the source of truth and
this command emits a renderable static site (Starlight; Mintlify for preview)
from OKF frontmatter.

Publication routing is owned by ``scripts/docs_verify.render_channel``, which
buckets every doc into ``public`` / ``internal`` / ``archive`` / ``exclude``
from its frontmatter. This builder publishes the ``public`` channel by default
(``--channels`` to opt archive docs in, rendered with a snapshot banner). It
does not re-derive its own type/privacy/status gate — there is one routing
policy, shared with the verification ledger.

Run ``--check`` in CI to fail if the published tree drifts from the corpus.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# Reuse the OKF parser and the docs_verify render router so the site can never
# disagree with the validator or with the shipped publication-routing policy.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from docs_okf import (  # noqa: E402
    DEFAULT_DOCS_ROOT,
    PROJECT_ROOT,
    display_path,
    first_h1,
    markdown_files,
    read_okf_document,
)
from docs_verify import render_channel  # noqa: E402

DEFAULT_SITE_ROOT = PROJECT_ROOT / "site"

# Publication routing is owned by scripts/docs_verify.py's render_channel():
# every doc routes to public / internal / archive / exclude from its OKF
# frontmatter. The site builder consumes that single classifier rather than
# re-deriving a parallel type/privacy/status gate. ``public`` is the default
# channel; ``archive`` can be opted in (rendered with a snapshot banner).
PUBLISHABLE_CHANNELS = {"public"}

# Status-driven banners for docs pulled in from non-default channels (archive).
STATUS_BANNER = {
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
    "needs-revalidation": (
        "warning",
        "This page has not been re-verified against the current codebase and may "
        "be out of date.",
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
    channels: set[str] = PUBLISHABLE_CHANNELS,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Return ``(pages, skipped)`` for docs whose render channel is requested.

    Routing is delegated to ``docs_verify.render_channel`` — the same classifier
    the verification ledger uses — so the site and the ledger can never disagree
    about what is public. ``skipped`` carries the channel each excluded doc
    routed to, so the build is never a silent truncation.
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

        channel = render_channel(doc)
        if channel not in channels:
            skipped.append((rel, f"render={channel}"))
            continue

        fm = doc.frontmatter
        pages.append(
            {
                "src": path,
                "rel": rel,
                "type": str(fm.get("type", "")).strip(),
                "status": str(fm.get("status", "")).strip().lower(),
                "channel": channel,
                "title": fm.get("title") or first_h1(doc.body) or path.stem,
                "description": fm.get("description", ""),
                "body": doc.body,
                "frontmatter": fm,
            }
        )

    return pages, skipped


# --------------------------------------------------------------------------- #
# Emitters
# --------------------------------------------------------------------------- #
# An emitter turns the selected pages into a full site tree (returned as a
# ``{relative_path: file_contents}`` mapping). The build/check pipeline is
# emitter-agnostic, so adding a renderer never touches the publish gate.
#
# GitHub Pages serves static HTML, so the production emitter is Starlight
# (Astro) — its output compiles to a static `dist/`. Mintlify is kept only as a
# preview/prototyping format (it is a hosted SaaS and cannot deploy to Pages).
# --------------------------------------------------------------------------- #
SITE_TITLE = "Kestrel Sovereign"
GITHUB_ORG_URL = "https://github.com/KestrelSovereignAI"
GITHUB_REPO_URL = "https://github.com/KestrelSovereignAI/kestrel-sovereign"
DEFAULT_BRANCH = "main"
# Project-pages default. For a custom domain (CNAME) override to "" via --base.
DEFAULT_PAGES_BASE = "/kestrel-sovereign"


def _slugify_segment(segment: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", segment.lower()).strip("-")
    return slug or "page"


def _site_slug(rel: str) -> str:
    """``docs/user-documentation/FOO_BAR.md`` -> ``user-documentation/foo-bar``.

    Emit Starlight's canonical (lowercase, hyphenated) slug form so the content
    id, sidebar entry, and inter-page links all agree — Starlight slugifies
    paths on its own, and a raw ``README`` would not resolve.
    """
    p = Path(rel)
    if p.parts and p.parts[0] == "docs":
        p = Path(*p.parts[1:])
    return "/".join(_slugify_segment(seg) for seg in p.with_suffix("").parts)


def _yaml_quote(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _strip_leading_h1(body: str) -> str:
    """Themes render the title from frontmatter; drop a duplicate leading H1."""
    lines = body.lstrip("\n").splitlines()
    if lines and lines[0].strip().startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).rstrip()


_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")


def rewrite_local_images(
    body: str, src_dir: Path, url_prefix: str
) -> tuple[str, dict[str, Path], list[str]]:
    """Resolve relative image embeds against the source doc.

    Local images live outside the docs tree (e.g. demo screenshots under
    ``demos/``), so their relative paths do not survive the projection. Copy
    each one that exists into the site under a content-hashed name and rewrite
    the link to ``url_prefix/<name>``; drop the rest with a note (never a silent
    deletion). Returns ``(new_body, {asset_name: source_path}, dropped)``.
    """
    assets: dict[str, Path] = {}
    dropped: list[str] = []

    def repl(match: re.Match[str]) -> str:
        pre, url, post = match.groups()
        raw = url.strip()
        if raw.startswith(("http://", "https://", "/")):
            return match.group(0)
        target = raw.split()[0].split("#")[0]
        asset_src = (src_dir / target).resolve()
        if asset_src.is_file():
            digest = hashlib.sha1(str(asset_src).encode()).hexdigest()[:8]
            safe = re.sub(r"[^A-Za-z0-9._-]", "-", asset_src.name)
            name = f"{digest}-{safe}"
            assets[name] = asset_src
            return f"{pre}{url_prefix}/{name}{post}"
        dropped.append(target)
        return "*(image unavailable in published docs)*"

    return _IMAGE_RE.sub(repl, body), assets, dropped


_LINK_RE = re.compile(r"(?<!!)(\[[^\]]+\]\()([^)]+)(\))")


def rewrite_local_links(
    body: str,
    src_dir: Path,
    *,
    base: str,
    published: dict[str, str],
    repo_root: Path,
) -> str:
    """Repoint relative Markdown links so the published site has no dead links.

    A projected page still carries the corpus's relative links. Three cases:

    * target is a *published* doc -> rewrite to its site slug (``base/slug/``);
    * target is any other real repo file/dir (an internal doc, or source like
      ``server.py``) -> rewrite to its GitHub URL so the link still resolves;
    * external / anchor-only / unresolvable -> left untouched.
    """

    def repl(match: re.Match[str]) -> str:
        pre, url, post = match.groups()
        raw = url.strip()
        if not raw or raw[0] in "#/" or raw.startswith(
            ("http://", "https://", "mailto:", "tel:", "//")
        ):
            return match.group(0)

        # Split an optional ` "title"` suffix and a `#anchor`.
        head, _, title = raw.partition(" ")
        title = f" {title}" if title else ""
        target, _, frag = head.partition("#")
        anchor = f"#{frag}" if frag else ""
        if not target:
            return match.group(0)

        abs_path = (src_dir / target).resolve()
        try:
            rel = abs_path.relative_to(repo_root).as_posix()
        except ValueError:
            return match.group(0)

        if rel in published:
            return f"{pre}{base}/{published[rel]}/{anchor}{title}{post}"
        if abs_path.exists():
            kind = "tree" if abs_path.is_dir() else "blob"
            return f"{pre}{GITHUB_REPO_URL}/{kind}/{DEFAULT_BRANCH}/{rel}{anchor}{title}{post}"
        return match.group(0)

    return _LINK_RE.sub(repl, body)


def _published_map(pages: list[dict[str, Any]]) -> dict[str, str]:
    """Map each published doc's repo-relative path to its site slug."""
    return {page["rel"]: _site_slug(page["rel"]) for page in pages}


def _grouped(pages: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        by_group.setdefault(page["type"], []).append(page)

    def group_key(name: str) -> tuple[int, str]:
        return (GROUP_ORDER.index(name) if name in GROUP_ORDER else len(GROUP_ORDER), name)

    return [
        (group, sorted(by_group[group], key=lambda p: _site_slug(p["rel"])))
        for group in sorted(by_group, key=group_key)
    ]


class StarlightEmitter:
    """Emit an Astro + Starlight project that builds to static HTML for Pages."""

    name = "starlight"

    def __init__(self, base: str = DEFAULT_PAGES_BASE) -> None:
        self.base = base.rstrip("/")

    def render_page(self, page: dict[str, Any], body: str | None = None) -> str:
        lines = ["---", f"title: {_yaml_quote(page['title'])}"]
        if page["description"]:
            lines += [f"description: {_yaml_quote(page['description'])}"]
        lines += ["---", ""]

        banner = STATUS_BANNER.get(page["status"])
        if banner:
            kind, text = banner
            directive = {"warning": "caution", "info": "note"}.get(kind, "note")
            lines += [f":::{directive}", text, ":::", ""]

        lines += [_strip_leading_h1(page["body"] if body is None else body), ""]
        return "\n".join(lines)

    def tree(self, pages: list[dict[str, Any]]) -> dict[str, str | Path]:
        files: dict[str, str | Path] = {}
        published = _published_map(pages)
        for page in pages:
            src_dir = Path(page["src"]).parent
            body, assets, dropped = rewrite_local_images(
                page["body"], src_dir, f"{self.base}/_assets"
            )
            body = rewrite_local_links(
                body, src_dir, base=self.base, published=published, repo_root=PROJECT_ROOT
            )
            files[f"src/content/docs/{_site_slug(page['rel'])}.md"] = self.render_page(page, body)
            for name, src in assets.items():
                files[f"public/_assets/{name}"] = src
            for target in dropped:
                print(f"  WARN {page['rel']}: dropped missing image {target}", file=sys.stderr)
        files["src/content/docs/index.md"] = self._landing(pages)
        files["astro.config.mjs"] = self._astro_config(pages)
        files["src/content.config.ts"] = _STARLIGHT_CONTENT_CONFIG
        files["package.json"] = _STARLIGHT_PACKAGE_JSON
        files["tsconfig.json"] = _STARLIGHT_TSCONFIG
        files[".gitignore"] = "node_modules/\ndist/\n.astro/\n"
        return files

    def _landing(self, pages: list[dict[str, Any]]) -> str:
        lines = [
            "---",
            f"title: {_yaml_quote(SITE_TITLE)}",
            'description: "Sovereign AI agents you own and run yourself."',
            "---",
            "",
            "Generated from the OKF documentation corpus. Browse by section:",
            "",
        ]
        for group, group_pages in _grouped(pages):
            lines.append(f"## {group}")
            lines.append("")
            for page in group_pages:
                slug = _site_slug(page["rel"])
                lines.append(f"- [{page['title']}]({self.base}/{slug}/)")
            lines.append("")
        return "\n".join(lines)

    def _astro_config(self, pages: list[dict[str, Any]]) -> str:
        sidebar = [
            {
                "label": group,
                "items": [
                    {"label": page["title"], "slug": _site_slug(page["rel"])}
                    for page in group_pages
                ],
            }
            for group, group_pages in _grouped(pages)
        ]
        sidebar_json = json.dumps(sidebar, indent=6)
        return _STARLIGHT_ASTRO_CONFIG.format(
            base=self.base or "/",
            title=json.dumps(SITE_TITLE),
            github=json.dumps(GITHUB_ORG_URL),
            sidebar=sidebar_json,
        )


class MintlifyEmitter:
    """Preview-only emitter. Mintlify is hosted SaaS — not for Pages deploys."""

    name = "mintlify"

    def render_page(self, page: dict[str, Any], body: str | None = None) -> str:
        lines = ["---", f"title: {_yaml_quote(page['title'])}"]
        if page["description"]:
            lines += [f"description: {_yaml_quote(page['description'])}"]
        lines += ["---", ""]
        banner = STATUS_BANNER.get(page["status"])
        if banner:
            kind, text = banner
            tag = {"warning": "Warning", "info": "Info"}.get(kind, "Note")
            lines += [f"<{tag}>{text}</{tag}>", ""]
        lines += [_strip_leading_h1(page["body"] if body is None else body), ""]
        return "\n".join(lines)

    def tree(self, pages: list[dict[str, Any]]) -> dict[str, str | Path]:
        files: dict[str, str | Path] = {}
        published = _published_map(pages)
        for page in pages:
            src_dir = Path(page["src"]).parent
            body, assets, _ = rewrite_local_images(page["body"], src_dir, "/_assets")
            body = rewrite_local_links(
                body, src_dir, base="", published=published, repo_root=PROJECT_ROOT
            )
            files[f"{_site_slug(page['rel'])}.mdx"] = self.render_page(page, body)
            for name, src in assets.items():
                files[f"_assets/{name}"] = src
        nav = [
            {"group": group, "pages": [_site_slug(p["rel"]) for p in group_pages]}
            for group, group_pages in _grouped(pages)
        ]
        files["mint.json"] = json.dumps(
            {
                "$schema": "https://mintlify.com/schema.json",
                "name": SITE_TITLE,
                "colors": {"primary": "#0F766E", "light": "#14B8A6", "dark": "#0F766E"},
                "navigation": nav,
                "footerSocials": {"github": GITHUB_ORG_URL},
            },
            indent=2,
        ) + "\n"
        return files


_STARLIGHT_PACKAGE_JSON = """{
  "name": "kestrel-sovereign-docs",
  "type": "module",
  "version": "0.0.0",
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview"
  },
  "dependencies": {
    "@astrojs/starlight": "^0.30.0",
    "astro": "^5.1.0",
    "sharp": "^0.33.5"
  }
}
"""

_STARLIGHT_TSCONFIG = """{
  "extends": "astro/tsconfigs/strict"
}
"""

_STARLIGHT_CONTENT_CONFIG = """import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
};
"""

_STARLIGHT_ASTRO_CONFIG = """// @ts-check
import {{ defineConfig }} from 'astro/config';
import starlight from '@astrojs/starlight';

// Generated by scripts/build_docs_site.py — do not edit by hand.
export default defineConfig({{
  site: 'https://kestrelsovereignai.github.io',
  base: '{base}',
  integrations: [
    starlight({{
      title: {title},
      social: {{ github: {github} }},
      sidebar: {sidebar},
    }}),
  ],
}});
"""


EMITTERS: dict[str, Any] = {
    "starlight": StarlightEmitter,
    "mintlify": MintlifyEmitter,
}


def build_site(
    *,
    docs_root: Path,
    site_root: Path,
    emitter: str,
    base: str,
    channels: set[str],
    check: bool,
) -> int:
    emitter_cls = EMITTERS[emitter]
    renderer = emitter_cls(base) if emitter == "starlight" else emitter_cls()
    pages, skipped = select_pages(docs_root, channels=channels)

    if not pages:
        print("ERROR: no publishable pages selected", file=sys.stderr)
        return 1

    # Stage into a temp tree, then either diff (check) or swap in (build).
    staging = site_root.parent / (site_root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for rel_path, contents in renderer.tree(pages).items():
        out = staging / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, Path):
            shutil.copyfile(contents, out)  # binary asset (e.g. screenshot)
        else:
            out.write_text(contents, encoding="utf-8")

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
    parser.add_argument("--emitter", choices=tuple(EMITTERS), default="starlight")
    parser.add_argument(
        "--base",
        default=DEFAULT_PAGES_BASE,
        help='GitHub Pages base path (starlight). Use "" for a custom domain.',
    )
    parser.add_argument(
        "--channels",
        default="public",
        help="comma-separated docs_verify render channels to publish "
        "(public, archive, internal). Default: public.",
    )
    args = parser.parse_args()

    channels = {c.strip() for c in args.channels.split(",") if c.strip()}

    return build_site(
        docs_root=args.docs_root.resolve(),
        site_root=args.site_root.resolve(),
        emitter=args.emitter,
        base=args.base,
        channels=channels,
        check=args.command == "check",
    )


if __name__ == "__main__":
    raise SystemExit(main())
