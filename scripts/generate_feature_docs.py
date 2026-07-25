#!/usr/bin/env python3
"""Generate audience-specific feature docs from the canonical KESTREL_FEATURES.md.

Reads the canonical source of truth and uses an LLM to transform it for different audiences.
No annotations needed in the canonical file — audience profiles are system prompts here.

Usage:
    uv run python scripts/generate_feature_docs.py --audience investor
    uv run python scripts/generate_feature_docs.py --all
    uv run python scripts/generate_feature_docs.py --audience user --model openai/gpt-5.1
    uv run python scripts/generate_feature_docs.py --audience investor --dry-run
    uv run python scripts/generate_feature_docs.py --sync-protected-contracts
"""

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from kestrel_sovereign.llm.model_catalog import get_catalog_service
from kestrel_sovereign.llm.model_metadata import ModelInfo
from kestrel_sovereign.llm.model_selection import resolve_provider_default

# ---------------------------------------------------------------------------
# Audience profiles
# ---------------------------------------------------------------------------

AUDIENCES: dict[str, dict] = {
    "developer": {
        "title": "Developer & AI Agent Reference",
        "system": (
            "You are a technical writer producing a developer-oriented feature reference. "
            "Your audience is software engineers and AI agents who will integrate with or "
            "extend this framework.\n\n"
            "Rules:\n"
            "- Keep ALL code links, file paths, class names, and function signatures\n"
            "- Keep tables — they're great for quick scanning\n"
            "- Preserve the hierarchical structure\n"
            "- Add brief 'quick start' hints where useful (e.g. which endpoint to call)\n"
            "- Use precise technical language — your readers know Python, FastAPI, async\n"
            "- Keep it concise — this is a reference, not a tutorial\n"
            "- Output clean GitHub-flavored markdown\n"
            "- Preserve limitations and diagnostic-versus-runtime distinctions; "
            "never upgrade an estimate or conditional path into a guarantee\n"
            "- Do NOT invent features or details not present in the source"
        ),
    },
    "user": {
        "title": "What Can Kestrel Do For You?",
        "system": (
            "You are a product writer creating a friendly, non-technical feature overview. "
            "Your audience is end users who want to understand what Kestrel can do for them — "
            "not how it works internally.\n\n"
            "Rules:\n"
            "- REMOVE all code links, file paths, class names, function signatures\n"
            "- REMOVE internal architecture details (mixins, adapters, database schemas)\n"
            "- REMOVE deployment/Docker/testing sections — users don't care\n"
            "- REWRITE technical concepts into benefits:\n"
            "  - 'secp256k1 keypairs' → 'your agent has its own secure identity'\n"
            "  - '5 privacy levels' → 'you control exactly how private your conversations are'\n"
            "  - 'multi-LLM' → 'works with ChatGPT, Claude, Gemini, and local models'\n"
            "- If the source names privacy presets, keep those exact preset names and do not invent replacements\n"
            "- Keep the hierarchical structure but simplify headings\n"
            "- Use 'you/your' language — speak directly to the user\n"
            "- Preserve limitations in plain language; do not describe context "
            "diagnostics as exact or conditional context behavior as universal\n"
            "- Include the privacy levels table but rewrite for non-technical readers\n"
            "- Aim for ~300 lines, scannable, with short paragraphs\n"
            "- Output clean GitHub-flavored markdown\n"
            "- Do NOT invent features not present in the source"
        ),
    },
    "investor": {
        "title": "Platform Capabilities & Technical Moat",
        "system": (
            "You are a strategic analyst writing a feature overview for investors and "
            "business stakeholders evaluating Kestrel Sovereign as a platform.\n\n"
            "Rules:\n"
            "- REMOVE all code links, file paths, class names, function signatures\n"
            "- REMOVE implementation details — focus on WHAT and WHY, not HOW\n"
            "- REFRAME features as competitive advantages and market differentiators:\n"
            "  - Constitutional AI → 'governance framework with no industry equivalent'\n"
            "  - Multi-LLM → 'vendor-independent, no platform lock-in'\n"
            "  - DIDs → 'portable identity — users own their AI, can move between providers'\n"
            "  - Privacy system → 'enterprise-grade privacy controls from day one'\n"
            "  - Feature plugin system → 'extensible platform architecture'\n"
            "- Add a brief executive summary at the top (3-4 sentences)\n"
            "- Use quantitative metrics only when they are explicitly present in the "
            "source document; do not invent or hardcode counts\n"
            "- Do NOT derive route counts from route lists; say 'documented route families' "
            "unless the source explicitly states a route count\n"
            "- Do NOT translate 'current audited snapshot' into 'independently audited' "
            "or imply external audit unless the source explicitly says so\n"
            "- If you mention privacy presets, use the exact preset names from the source: "
            "`ephemeral`, `isolated`, `anonymous`, `normal`, `public`\n"
            "- If the source names route classes or HTTP methods, preserve those exactly rather than renaming them\n"
            "- Preserve limitations and conditional status. Do not promise that "
            "context remains coherent regardless of model or that context "
            "diagnostics exactly reproduce the production prompt\n"
            "- Organize around: Platform Architecture, AI Capabilities, Data Sovereignty, "
            "Security & Privacy, Deployment Flexibility, Extensibility\n"
            "- Keep it to ~250-350 lines, professional tone\n"
            "- Output clean GitHub-flavored markdown\n"
            "- Do NOT invent features or metrics not present in the source"
        ),
    },
}

# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = PROJECT_ROOT / "KESTREL_FEATURES.md"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "generated"
_DISCOVERY_REFRESH_CACHE: dict[str, list[ModelInfo]] = {}
BOUNDARY_START = "<!-- BEGIN PROTECTED PACKAGE BOUNDARY CONTRACT -->"
BOUNDARY_END = "<!-- END PROTECTED PACKAGE BOUNDARY CONTRACT -->"
CONTEXT_START = "<!-- BEGIN PROTECTED CONTEXT HONESTY CONTRACT -->"
CONTEXT_END = "<!-- END PROTECTED CONTEXT HONESTY CONTRACT -->"
NON_BUNDLED_ALIASES_PATTERN = re.compile(
    r"<!-- NON_BUNDLED_SURFACE_ALIASES:\s*(.*?)\s*-->",
    re.DOTALL,
)
OWNERSHIP_PROMOTION_TERMS = (
    "built-in",
    "built in",
    "bundled",
    "core feature",
    "core capability",
    "core module",
    "native voice",
    "native wallet",
    "native integration",
    "ships with",
    "part of the base install",
    "included in the base install",
    "no separate install",
)
CONTEXT_OVERCLAIM_PATTERNS = (
    (
        re.compile(
            r"\b(?:context|conversation) (?:remains?|stays?) coherent "
            r"regardless\b",
            re.IGNORECASE,
        ),
        "promises context coherence regardless of runtime constraints",
    ),
    (
        re.compile(
            r"\bcontext (?:status|diagnostics?) "
            r"(?:is an exact|are exact|provides? an exact|exactly reproduces?)\b",
            re.IGNORECASE,
        ),
        "promotes a context diagnostic projection to an exact trace",
    ),
    (
        re.compile(
            r"\bautomatic (?:durable )?salvage "
            r"(?:is|runs|applies|protects) (?:the )?"
            r"(?:default|all routes|every prune)\b",
            re.IGNORECASE,
        ),
        "promotes conditional automatic salvage to universal behavior",
    ),
)
_INLINE_LINK_RE = re.compile(
    r"(?P<prefix>!?\[[^\]\n]*\]\()(?P<target>[^)\s]+)(?P<suffix>\))"
)
GENERATED_LINK_ALIASES = {
    # Older LLM output invented this former-looking module name. The current
    # tool-registry owner is the useful maintained target.
    "kestrel_sovereign/kestrel_agent_tools.py": (
        "kestrel_sovereign/agent/tool_registry.py"
    ),
}


def build_okf_frontmatter(
    audience: str,
    *,
    provider: str,
    model_name: str,
    generated_at: datetime,
) -> str:
    """Build deterministic OKF metadata for a generated audience doc."""
    metadata = {
        "type": "Generated Reference",
        "title": AUDIENCES[audience]["title"],
        "description": f"Audience-specific {audience} view generated from the canonical Kestrel feature inventory.",
        "resource": f"/docs/generated/FEATURES_{audience}.md",
        "tags": ["features", "generated-docs", audience],
        "timestamp": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "generated",
        "generated": True,
        "canonical": False,
        "source": "/KESTREL_FEATURES.md",
        "audience": audience,
        "generator": "scripts/generate_feature_docs.py",
        "model": f"{provider}/{model_name}",
        "regenerate": f"uv run python scripts/generate_feature_docs.py --audience {audience}",
    }
    dumped = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{dumped}\n---\n\n"


def parse_okf_frontmatter(path: Path) -> dict | None:
    """Return OKF frontmatter for a markdown file, or None when absent/invalid."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    parsed = yaml.safe_load(text[4:end]) or {}
    return parsed if isinstance(parsed, dict) else None


def _extract_protected_contract(
    source: str,
    start_marker: str,
    end_marker: str,
    label: str,
) -> str:
    """Return one canonical protected contract block verbatim."""
    start = source.find(start_marker)
    end = source.find(end_marker)
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"Canonical feature inventory is missing the protected {label} contract"
        )
    return source[start : end + len(end_marker)]


def extract_boundary_contract(source: str) -> str:
    """Return the canonical protected package-boundary block verbatim."""
    return _extract_protected_contract(
        source,
        BOUNDARY_START,
        BOUNDARY_END,
        "package boundary",
    )


def extract_context_contract(source: str) -> str:
    """Return the canonical protected context-honesty block verbatim."""
    return _extract_protected_contract(
        source,
        CONTEXT_START,
        CONTEXT_END,
        "context honesty",
    )


def extract_non_bundled_surface_aliases(
    boundary_contract: str,
) -> dict[str, tuple[str, ...]]:
    """Read protected aliases and every non-bundled registry identifier.

    The canonical contract carries audience-friendly aliases (for example,
    ``"GitHub integration"``).  Registry keys are authoritative for the
    complete catalog, though, so include each external row's stable key as an
    additional alias.  Otherwise a transformation can evade the guard merely
    by shortening an extracted surface's name (for example, ``"GitHub"``).
    """
    match = NON_BUNDLED_ALIASES_PATTERN.search(boundary_contract)
    if match is None:
        raise ValueError(
            "Protected package boundary contract is missing its "
            "NON_BUNDLED_SURFACE_ALIASES declaration"
        )

    aliases: dict[str, tuple[str, ...]] = {}
    for group in match.group(1).split(";"):
        values = tuple(
            value.strip().casefold()
            for value in group.split("|")
            if value.strip()
        )
        if values:
            aliases[values[0]] = values
    if not aliases:
        raise ValueError(
            "Protected package boundary contract declares no non-bundled aliases"
        )

    from kestrel_sovereign.feature_registry import PackageBoundary, load_registry

    non_bundled_boundaries = {
        PackageBoundary.FEATURE_PACKAGE,
        PackageBoundary.PROVIDER_PACKAGE,
        PackageBoundary.STANDALONE_TOOL,
    }
    for info in load_registry().values():
        if info.boundary not in non_bundled_boundaries:
            continue
        stable_id = info.name.casefold()
        aliases.setdefault(
            stable_id,
            tuple(dict.fromkeys((stable_id, stable_id.replace("_", " ")))),
        )
    return aliases


def _strip_protected_contract(
    text: str,
    start_marker: str,
    end_marker: str,
) -> str:
    """Remove an echoed protected block before deterministic insertion."""
    pattern = re.compile(
        rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
        re.DOTALL,
    )
    return pattern.sub("", text).strip()


def _strip_protected_contracts(text: str) -> str:
    """Remove every deterministic contract from transformed prose."""
    text = _strip_protected_contract(text, BOUNDARY_START, BOUNDARY_END)
    return _strip_protected_contract(text, CONTEXT_START, CONTEXT_END)


def find_boundary_promotions(
    text: str,
    non_bundled_aliases: dict[str, tuple[str, ...]],
) -> list[str]:
    """Find prose that promotes a non-bundled surface to bundled/core.

    This intentionally targets ownership claims, not ordinary capability
    descriptions. The exact boundary contract is inserted separately and is
    not passed to this validator.
    """
    violations: list[str] = []
    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n|(?<=[.!?])\s+", text)
        if block.strip()
    ]
    for block in blocks:
        normalized = block.casefold()
        promotion = next(
            (term for term in OWNERSHIP_PROMOTION_TERMS if term in normalized),
            None,
        )
        if promotion is None:
            continue
        for surface, aliases in non_bundled_aliases.items():
            if any(alias in normalized for alias in aliases):
                violations.append(
                    f"{surface}: ownership promotion using {promotion!r}: "
                    f"{block[:180]}"
                )
                break

    core_section = re.compile(
        r"(?ims)^#{2,6}\s+[^\n]*(?:core|bundled)[^\n]*"
        r"(?:feature|capabilit|module)[^\n]*\n"
        r"(.*?)(?=^#{1,6}\s|\Z)"
    )
    for section in core_section.findall(text):
        normalized = section.casefold()
        for surface, aliases in non_bundled_aliases.items():
            if any(alias in normalized for alias in aliases):
                violations.append(
                    f"{surface}: listed inside a core/bundled inventory section"
                )
    return violations


def find_context_overclaims(text: str) -> list[str]:
    """Find explicit promotions of projected/conditional context behavior."""
    violations: list[str] = []
    for pattern, explanation in CONTEXT_OVERCLAIM_PATTERNS:
        for match in pattern.finditer(text):
            excerpt = text[match.start() : match.end()]
            violations.append(f"{explanation}: {excerpt}")
    return violations


def normalize_generated_links(text: str) -> str:
    """Rebase repository-root links for files under ``docs/generated``.

    LLM transformations often preserve canonical links such as
    ``kestrel_sovereign/agent/context_manager.py``. Those resolve from the
    repository root in ``KESTREL_FEATURES.md`` but not from the generated
    directory. Keep already-valid links and mechanically rebase only a target
    that exists from the project root.
    """

    def replace(match: re.Match[str]) -> str:
        target = match.group("target")
        if (
            target.startswith(("#", "/", "//", "<"))
            or re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", target)
        ):
            return match.group(0)

        path_part, separator, fragment = target.partition("#")
        if not path_part:
            return match.group(0)
        generated_target = (OUTPUT_DIR / path_part).resolve()
        if generated_target.exists():
            return match.group(0)
        canonical_path = path_part
        canonical_target = (PROJECT_ROOT / canonical_path).resolve()
        if not canonical_target.exists() and path_part.startswith("endpoints/"):
            canonical_path = f"kestrel_sovereign/{path_part}"
            canonical_target = (PROJECT_ROOT / canonical_path).resolve()
        if not canonical_target.exists() and path_part in GENERATED_LINK_ALIASES:
            canonical_path = GENERATED_LINK_ALIASES[path_part]
            canonical_target = (PROJECT_ROOT / canonical_path).resolve()
        if not canonical_target.exists():
            return match.group(0)

        rebased = os.path.relpath(canonical_target, OUTPUT_DIR).replace(os.sep, "/")
        if separator:
            rebased = f"{rebased}#{fragment}"
        prefix = match.group("prefix").replace(path_part, canonical_path)
        return f"{prefix}{rebased}{match.group('suffix')}"

    return _INLINE_LINK_RE.sub(replace, text)


def compose_generated_body(source: str, transformed: str) -> str:
    """Validate audience prose and prepend both canonical contracts."""
    boundary_contract = extract_boundary_contract(source)
    context_contract = extract_context_contract(source)
    aliases = extract_non_bundled_surface_aliases(boundary_contract)
    transformed = _strip_protected_contracts(transformed)
    violations = find_boundary_promotions(transformed, aliases)
    violations.extend(find_context_overclaims(transformed))
    if violations:
        raise ValueError(
            "Generated audience document contradicts package ownership or "
            "context honesty:\n- "
            + "\n- ".join(violations)
        )
    transformed = normalize_generated_links(transformed)
    return f"{boundary_contract}\n\n{context_contract}\n\n{transformed}\n"


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a generated document into its unchanged header and body."""
    match = re.match(r"\A---\n.*?\n---\n\n?", text, re.DOTALL)
    if match is None:
        raise ValueError("generated document is missing OKF frontmatter")
    return match.group(0), text[match.end() :]


def sync_protected_contracts() -> list[Path]:
    """Mechanically sync protected blocks and root-relative links.

    This does not call an LLM or rewrite audience prose/frontmatter.
    """
    source = SOURCE_FILE.read_text(encoding="utf-8")
    updated: list[Path] = []
    for audience in AUDIENCES:
        path = OUTPUT_DIR / f"FEATURES_{audience}.md"
        current = path.read_text(encoding="utf-8")
        header, transformed = _split_frontmatter(current)
        rendered = header + compose_generated_body(source, transformed)
        if current != rendered:
            path.write_text(rendered, encoding="utf-8")
            updated.append(path)
    return updated


def check_generated_docs() -> int:
    """Validate metadata, protected contracts, claims, and local links."""
    failures: list[str] = []
    source = SOURCE_FILE.read_text(encoding="utf-8")
    boundary_contract = extract_boundary_contract(source)
    context_contract = extract_context_contract(source)
    aliases = extract_non_bundled_surface_aliases(boundary_contract)
    from scripts import check_docs_links

    for audience in AUDIENCES:
        path = OUTPUT_DIR / f"FEATURES_{audience}.md"
        metadata = parse_okf_frontmatter(path)
        if metadata is None:
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: missing OKF frontmatter")
            continue
        expected = {
            "type": "Generated Reference",
            "generated": True,
            "canonical": False,
            "source": "/KESTREL_FEATURES.md",
            "audience": audience,
            "generator": "scripts/generate_feature_docs.py",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                failures.append(
                    f"{path.relative_to(PROJECT_ROOT)}: expected {key}={value!r}, got {metadata.get(key)!r}"
                )
        text = path.read_text(encoding="utf-8")
        if text.count(boundary_contract) != 1:
            failures.append(
                f"{path.relative_to(PROJECT_ROOT)}: protected package-boundary "
                "contract is missing, duplicated, or stale"
            )
        if text.count(context_contract) != 1:
            failures.append(
                f"{path.relative_to(PROJECT_ROOT)}: protected context-honesty "
                "contract is missing, duplicated, or stale"
            )
        body = text.replace(boundary_contract, "", 1).replace(
            context_contract, "", 1
        )
        for violation in find_boundary_promotions(body, aliases):
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: {violation}")
        for violation in find_context_overclaims(body):
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: {violation}")
        for broken_link in check_docs_links.check_file(path):
            failures.append(broken_link.format())
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("Generated feature docs metadata, contracts, and links are current.")
    return 0


async def _discover_provider_models(provider: str) -> list[ModelInfo]:
    """Discover live models for the selected provider."""
    if provider == "anthropic":
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

        return await AnthropicAdapter().list_models()

    if provider == "openai":
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        return await OpenAIAdapter().list_models()

    return []


def _refresh_provider_cache(provider: str) -> list[ModelInfo]:
    """Refresh model discovery cache for one provider and return enriched models."""
    if provider in _DISCOVERY_REFRESH_CACHE:
        return _DISCOVERY_REFRESH_CACHE[provider]

    discovered = asyncio.run(_discover_provider_models(provider))
    if not discovered:
        raise RuntimeError(
            f"Live model discovery returned no models for provider '{provider}'. "
            "Check API credentials/network before generating audience docs."
        )

    catalog = get_catalog_service()
    enriched = catalog.enrich_models(discovered)
    existing = catalog.load_cache() or []
    merged = [model for model in existing if model.provider != provider] + enriched
    catalog.write_cache(merged)
    _DISCOVERY_REFRESH_CACHE[provider] = enriched
    return enriched


def get_client_and_model(model_override: str | None, refresh_discovery: bool = True) -> tuple:
    """Auto-detect available LLM provider. Returns (call_fn, model_name, provider_name)."""
    api_key_anthropic = os.environ.get("ANTHROPIC_API_KEY")
    api_key_openai = os.environ.get("OPENAI_API_KEY")

    if model_override and "/" in model_override:
        provider, model = model_override.split("/", 1)
        if provider == "anthropic":
            api_key_anthropic = api_key_anthropic or ""
        elif provider == "openai":
            api_key_openai = api_key_openai or ""

    if api_key_anthropic and (model_override is None or model_override.startswith("anthropic/") or model_override.startswith("claude")):
        import anthropic

        client = anthropic.Anthropic()
        refreshed_models = _refresh_provider_cache("anthropic") if refresh_discovery and not model_override else None
        model = model_override or f"anthropic/{resolve_provider_default('anthropic', cached_models=refreshed_models)}"
        if model.startswith("anthropic/"):
            model = model.split("/", 1)[1]

        def call_anthropic(system: str, user: str) -> str:
            resp = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        return call_anthropic, model, "anthropic"

    if api_key_openai:
        import openai

        client = openai.OpenAI()
        refreshed_models = _refresh_provider_cache("openai") if refresh_discovery and not model_override else None
        model = model_override or f"openai/{resolve_provider_default('openai', cached_models=refreshed_models)}"
        if model.startswith("openai/"):
            model = model.split("/", 1)[1]

        def call_openai(system: str, user: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=8192,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content

        return call_openai, model, "openai"

    print("Error: No LLM API key found.")
    print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment or .env file.")
    sys.exit(1)


def generate(
    audience: str,
    model_override: str | None = None,
    dry_run: bool = False,
    refresh_discovery: bool = True,
) -> Path:
    """Generate a feature doc for the given audience. Returns output path."""
    if audience not in AUDIENCES:
        print(f"Error: Unknown audience '{audience}'. Choose from: {', '.join(AUDIENCES)}")
        sys.exit(1)

    profile = AUDIENCES[audience]
    source = SOURCE_FILE.read_text()
    protected_rules = (
        "\n\nProtected invariants:\n"
        f"- The source section between {BOUNDARY_START} and {BOUNDARY_END} is "
        "normative; do not contradict it.\n"
        f"- The source section between {CONTEXT_START} and {CONTEXT_END} is "
        "normative; do not contradict it.\n"
        "- Never describe an extracted Feature package, provider package, or "
        "standalone tool as bundled, built-in, native, core, or included in "
        "the base install.\n"
        "- Never promote diagnostic context estimates or conditional salvage "
        "and compaction paths to exact or universal behavior.\n"
        "- The generator inserts both protected sections verbatim; do not rely "
        "on paraphrase to preserve either contract."
    )

    user_prompt = (
        f"Transform the following canonical feature document for the "
        f'"{audience}" audience. Produce a complete, standalone markdown document.\n\n'
        f"--- BEGIN SOURCE DOCUMENT ---\n{source}\n--- END SOURCE DOCUMENT ---"
    )

    if dry_run:
        print(f"=== DRY RUN: {audience} ===")
        print(f"System prompt ({len(profile['system'])} chars):")
        print(profile["system"] + protected_rules)
        print(f"\nUser prompt: {len(user_prompt)} chars ({len(source)} from source)")
        print(f"Output would go to: {OUTPUT_DIR / f'FEATURES_{audience}.md'}")
        return OUTPUT_DIR / f"FEATURES_{audience}.md"

    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    call_fn, model_name, provider = get_client_and_model(
        model_override,
        refresh_discovery=refresh_discovery,
    )

    print(f"Generating {audience} version via {provider}/{model_name}...")
    result = call_fn(profile["system"] + protected_rules, user_prompt)
    body = compose_generated_body(source, result)

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = build_okf_frontmatter(
        audience,
        provider=provider,
        model_name=model_name,
        generated_at=datetime.now(timezone.utc),
    )

    out_path = OUTPUT_DIR / f"FEATURES_{audience}.md"
    out_path.write_text(header + body)
    print(f"Wrote {out_path} ({len(body)} chars)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate audience-specific feature docs from KESTREL_FEATURES.md"
    )
    parser.add_argument(
        "--audience",
        choices=list(AUDIENCES.keys()),
        help="Target audience",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all audiences",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model override (e.g. openai/gpt-5.1, anthropic/claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the LLM",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check checked-in generated docs without calling the LLM",
    )
    parser.add_argument(
        "--sync-protected-contracts",
        action="store_true",
        help=(
            "Sync protected contracts and local-link rebasing without calling "
            "the LLM or changing audience prose/frontmatter"
        ),
    )
    parser.add_argument(
        "--skip-discovery-refresh",
        action="store_true",
        help="Use the existing discovery cache instead of refreshing live provider models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available audiences",
    )

    args = parser.parse_args()

    if args.list:
        for name, profile in AUDIENCES.items():
            print(f"  {name:12s}  {profile['title']}")
        return

    if args.check:
        raise SystemExit(check_generated_docs())

    if not SOURCE_FILE.exists():
        print(f"Error: {SOURCE_FILE} not found. Generate it first.")
        sys.exit(1)

    if args.sync_protected_contracts:
        updated = sync_protected_contracts()
        for path in updated:
            print(f"Updated {path.relative_to(PROJECT_ROOT)}")
        if not updated:
            print("Protected contracts and generated links are already current.")
        return

    if args.all:
        for audience in AUDIENCES:
            generate(
                audience,
                model_override=args.model,
                dry_run=args.dry_run,
                refresh_discovery=not args.skip_discovery_refresh,
            )
    elif args.audience:
        generate(
            args.audience,
            model_override=args.model,
            dry_run=args.dry_run,
            refresh_discovery=not args.skip_discovery_refresh,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
