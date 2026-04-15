#!/usr/bin/env python3
"""Generate audience-specific feature docs from the canonical KESTREL_FEATURES.md.

Reads the canonical source of truth and uses an LLM to transform it for different audiences.
No annotations needed in the canonical file — audience profiles are system prompts here.

Usage:
    uv run python scripts/generate_feature_docs.py --audience investor
    uv run python scripts/generate_feature_docs.py --all
    uv run python scripts/generate_feature_docs.py --audience user --model openai/gpt-5.1
    uv run python scripts/generate_feature_docs.py --audience investor --dry-run
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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

    user_prompt = (
        f"Transform the following canonical feature document for the "
        f'"{audience}" audience. Produce a complete, standalone markdown document.\n\n'
        f"--- BEGIN SOURCE DOCUMENT ---\n{source}\n--- END SOURCE DOCUMENT ---"
    )

    if dry_run:
        print(f"=== DRY RUN: {audience} ===")
        print(f"System prompt ({len(profile['system'])} chars):")
        print(profile["system"])
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
    result = call_fn(profile["system"], user_prompt)

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    header = (
        f"<!-- AUTO-GENERATED from KESTREL_FEATURES.md — do not edit manually -->\n"
        f"<!-- Audience: {audience} | Generated: {now} | Model: {provider}/{model_name} -->\n"
        f"<!-- Regenerate: uv run python scripts/generate_feature_docs.py --audience {audience} -->\n\n"
    )

    out_path = OUTPUT_DIR / f"FEATURES_{audience}.md"
    out_path.write_text(header + result)
    print(f"Wrote {out_path} ({len(result)} chars)")
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

    if not SOURCE_FILE.exists():
        print(f"Error: {SOURCE_FILE} not found. Generate it first.")
        sys.exit(1)

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
