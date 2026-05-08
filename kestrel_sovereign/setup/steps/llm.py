"""LLM provider step: pick vendor(s), capture keys, write ``[llm]``.

v1 supports three vendors only — Ollama (local), OpenAI, Anthropic.
Each is modeled after the canonical entries in ``kestrel.toml.example``
so the wizard's output is byte-identical to a hand-written config for
those vendors. Other providers (Gemini, OpenRouter, xAI, Groq) are
deferred to the feature step in v2.

The ``[llm]`` block we write is a deep-merged subset of the example —
existing user-authored vendors and routes outside our managed keys are
preserved by the toml writer's deep-merge.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env
from kestrel_sovereign.setup.toml_file import read_toml, write_toml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Vendor:
    """Static description of a vendor we can configure."""

    key: str
    label: str
    route_id: str  # full ``vendor:route`` form used by route_priority
    is_cloud: bool
    api_key_env: str | None
    toml_block: dict


_OLLAMA = _Vendor(
    key="ollama",
    label="Ollama (local, no API key)",
    route_id="ollama:local",
    is_cloud=False,
    api_key_env=None,
    toml_block={
        "vendors": {
            "ollama": {
                "is_cloud": False,
                "routes": {
                    "local": {
                        "adapter": "OllamaAdapter",
                        "host": "http://localhost:11434",
                        "model": "auto",
                        "selection_hints": ["llama3.2", "qwen", "gpt-oss"],
                    }
                },
            }
        }
    },
)

_OPENAI = _Vendor(
    key="openai",
    label="OpenAI (cloud — needs OPENAI_API_KEY)",
    route_id="openai:api",
    is_cloud=True,
    api_key_env="OPENAI_API_KEY",
    toml_block={
        "vendors": {
            "openai": {
                "is_cloud": True,
                "routes": {
                    "api": {
                        "adapter": "OpenAIAdapter",
                        "api_key_env": "OPENAI_API_KEY",
                        "model": "auto",
                        "selection_hints": ["gpt-5", "mini"],
                    }
                },
            }
        }
    },
)

_ANTHROPIC = _Vendor(
    key="anthropic",
    label="Anthropic Claude (cloud — needs ANTHROPIC_API_KEY)",
    route_id="anthropic:api",
    is_cloud=True,
    api_key_env="ANTHROPIC_API_KEY",
    toml_block={
        "vendors": {
            "anthropic": {
                "is_cloud": True,
                "routes": {
                    "api": {
                        "adapter": "AnthropicAdapter",
                        "api_key_env": "ANTHROPIC_API_KEY",
                        "model": "auto",
                        "selection_hints": ["sonnet", "opus", "haiku"],
                    }
                },
            }
        }
    },
)

# Google Gemini via the public API (NOT Vertex — Vertex needs ADC + project,
# which is too much to gather in a wizard prompt; users wanting Vertex can
# add the [llm.vendors.vertex_ai] block by hand).
_GOOGLE = _Vendor(
    key="google",
    label="Google Gemini (cloud — needs GOOGLE_API_KEY)",
    route_id="google:api",
    is_cloud=True,
    api_key_env="GOOGLE_API_KEY",
    toml_block={
        "vendors": {
            "google": {
                "is_cloud": True,
                "routes": {
                    "api": {
                        "adapter": "GoogleAdapter",
                        "api_key_env": "GOOGLE_API_KEY",
                        "model": "auto",
                        "selection_hints": ["flash", "pro"],
                    }
                },
            }
        }
    },
)

# OpenRouter — meta-provider that proxies many model families. Users get
# Anthropic/Google/Mistral/etc. behind one key.
_OPENROUTER = _Vendor(
    key="openrouter",
    label="OpenRouter (multi-vendor proxy — needs OPENROUTER_API_KEY)",
    route_id="openrouter:api",
    is_cloud=True,
    api_key_env="OPENROUTER_API_KEY",
    toml_block={
        "vendors": {
            "openrouter": {
                "is_cloud": True,
                "routes": {
                    "api": {
                        "adapter": "OpenRouterAdapter",
                        "api_key_env": "OPENROUTER_API_KEY",
                        "model": "auto",
                        "selection_hints": ["sonnet", "gpt", "gemini"],
                    }
                },
            }
        }
    },
)

_ALL_VENDORS = (_OLLAMA, _OPENAI, _ANTHROPIC, _GOOGLE, _OPENROUTER)
_VENDORS_BY_KEY = {v.key: v for v in _ALL_VENDORS}


def run(ctx: SetupContext) -> None:
    """Configure at least one LLM route."""
    existing_toml = read_toml(ctx.kestrel_toml_path)
    existing_llm = existing_toml.get("llm", {})
    existing_priority: list[str] = list(existing_llm.get("route_priority", []) or [])
    existing_keys = read_env(ctx.env_path)

    if ctx.flow is Flow.CHECK:
        # Check mode reports on-disk state only — never selects, never prompts.
        if not existing_priority:
            ctx.block("[llm] route_priority is empty — no LLM provider configured")
            return
        existing_vendors = existing_llm.get("vendors") or {}
        for route_id in existing_priority:
            vendor_key, _, route_key = route_id.partition(":")
            route = (
                (existing_vendors.get(vendor_key) or {}).get("routes") or {}
            ).get(route_key) or {}
            api_key_env = route.get("api_key_env")
            if api_key_env and not existing_keys.get(api_key_env):
                ctx.block(
                    f"{api_key_env} not set in .env (required for {route_id})"
                )
        return

    selected = _select_vendors(ctx, existing_priority)
    if not selected:
        ctx.block("No LLM provider selected — agent will not be able to think")
        return

    env_updates = _gather_api_keys(ctx, selected, existing_keys)

    toml_updates = _build_llm_toml(selected, existing_priority, existing_llm)

    if env_updates:
        env_result = write_env(ctx.env_path, env_updates)
        if env_result.backup_path is not None:
            ctx.record(f"Backed up existing .env to {env_result.backup_path.name}")
        for key in env_result.added:
            ctx.record(f"Set {key} in .env")
        for key in env_result.updated:
            ctx.record(f"Updated {key} in .env")

    if toml_updates:
        toml_result = write_toml(
            ctx.kestrel_toml_path, {"llm": toml_updates}
        )
        if toml_result.backup_path is not None:
            ctx.record(
                f"Backed up existing kestrel.toml to {toml_result.backup_path.name}"
            )
        if toml_result.changed:
            labels = ", ".join(v.key for v in selected)
            ctx.record(f"Wrote [llm] in kestrel.toml ({labels})")


def _detect_available_vendors(existing_keys: dict[str, str]) -> list[_Vendor]:
    """Auto-detect which vendors are usable on this machine.

    Walks both the live process env and ``existing_keys`` (the .env
    file the wizard's keys-step may have already populated) to find
    cloud API keys, and probes Ollama at ``localhost:11434`` with a
    short timeout. Returns the detected vendors in priority order:

      OpenRouter > Anthropic > OpenAI > Google > Ollama (if reachable)

    OpenRouter wins among cloud vendors because the README/CLI
    documentation recommends it (one key, many model families). Empty
    return → caller falls back to a sensible default (Ollama only).
    """
    detected: list[_Vendor] = []

    # Cloud vendors first, in documented preference order. Each entry
    # is the env var the wizard would persist to .env; if either the
    # parent shell or the .env file already has a non-empty value,
    # treat the vendor as available.
    cloud_priority = (_OPENROUTER, _ANTHROPIC, _OPENAI, _GOOGLE)
    for vendor in cloud_priority:
        env_var = vendor.api_key_env
        if not env_var:
            continue
        value = os.environ.get(env_var) or existing_keys.get(env_var)
        if value:
            detected.append(vendor)

    if _is_ollama_reachable():
        detected.append(_OLLAMA)

    return detected


def _is_ollama_reachable(host: str = "http://localhost:11434", timeout: float = 1.5) -> bool:
    """``True`` iff ``<host>/api/tags`` responds 200 within ``timeout``.

    Uses :mod:`urllib` (stdlib only — no new deps) and never raises:
    a network failure, timeout, non-200, or parse error all just
    mean "Ollama isn't usable right now". The default 1.5 s keeps
    quickstart snappy on machines without Ollama installed.
    """
    import urllib.error
    import urllib.request

    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return False


def _select_vendors(
    ctx: SetupContext, existing_priority: list[str]
) -> list[_Vendor]:
    """Decide which vendors are active for this run.

    Quickstart auto-detects cloud API keys in env / a running Ollama
    instance, falling back to Ollama-only if nothing is reachable.
    Check flow never selects (it only validates on-disk state).
    Interactive flow asks once, with the auto-detected primary
    pre-selected as the default.
    """
    if existing_priority:
        already = [
            _VENDORS_BY_KEY[r.split(":", 1)[0]]
            for r in existing_priority
            if r.split(":", 1)[0] in _VENDORS_BY_KEY
        ]
        if ctx.flow in (Flow.QUICKSTART, Flow.CHECK):
            return already
        if already:
            keep = ctx.prompter.confirm(
                f"Keep existing LLM order ({', '.join(v.key for v in already)})?",
                default=True,
            )
            if keep:
                return already

    if ctx.flow in (Flow.QUICKSTART, Flow.CHECK):
        # Read existing .env so the wizard sees keys the keys-step
        # already wrote in this run, not just the ones in the parent
        # shell.
        existing_keys = read_env(ctx.env_path)
        detected = _detect_available_vendors(existing_keys)
        if detected:
            labels = ", ".join(v.key for v in detected)
            logger.info(f"Auto-detected LLM vendors for quickstart: {labels}")
            return detected
        # Nothing usable — fall back to the historical default so the
        # wizard still produces a config (operator can install Ollama
        # later or rerun the interactive wizard to pick a cloud vendor).
        logger.info(
            "No LLM vendors auto-detected; defaulting to Ollama. "
            "Install Ollama or export a cloud API key (OPENROUTER_API_KEY, "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY) and rerun "
            "`kestrel setup --quickstart` to use a different provider."
        )
        return [_OLLAMA]

    choice = ctx.prompter.select(
        "Default LLM provider?",
        choices=[v.label for v in _ALL_VENDORS],
        default=_OLLAMA.label,
    )
    primary = next(v for v in _ALL_VENDORS if v.label == choice)

    fallbacks: list[_Vendor] = []
    for vendor in _ALL_VENDORS:
        if vendor.key == primary.key:
            continue
        if ctx.prompter.confirm(
            f"Add {vendor.label} as a fallback?", default=False
        ):
            fallbacks.append(vendor)

    return [primary, *fallbacks]


def _gather_api_keys(
    ctx: SetupContext, selected: list[_Vendor], existing_keys: dict[str, str]
) -> dict[str, str]:
    """Prompt for API keys for cloud vendors that don't yet have one set.

    Resolution order for each vendor's key:
      1. Already in ``.env`` (existing_keys) — leave alone.
      2. QUICKSTART only: exported in the parent shell
         (``os.environ``) — persist to .env so the runtime (which only
         reads .env) can use it. Keeps non-interactive setups working
         when operators have keys exported in their shell, which is
         the common case. INTERACTIVE always asks (the operator opted
         in to being prompted, even if they have keys exported — they
         may want to override).
      3. Quickstart with neither: block (operator must set the key
         and re-run).
      4. Interactive with neither: prompt; blank → block.
    """
    updates: dict[str, str] = {}
    for vendor in selected:
        if not vendor.api_key_env:
            continue
        current = existing_keys.get(vendor.api_key_env, "")
        if current:
            continue
        # Promote a shell-exported key into .env (quickstart only) so
        # the runtime sees it. The autodetect pass treated this vendor
        # as available because of the env var; we honor that here by
        # persisting the value. Interactive setup falls through to the
        # prompt — the operator chose to be asked.
        if ctx.flow is Flow.QUICKSTART:
            from_environ = os.environ.get(vendor.api_key_env, "")
            if from_environ:
                updates[vendor.api_key_env] = from_environ
                continue
        if ctx.flow is Flow.QUICKSTART:
            ctx.block(
                f"{vendor.api_key_env} not set in .env "
                f"(required for {vendor.label}). Set it and re-run setup."
            )
            continue
        value = ctx.prompter.secret(
            f"Paste your {vendor.api_key_env} (leave blank to skip)",
            default="",
        )
        if value:
            updates[vendor.api_key_env] = value
        else:
            ctx.block(
                f"{vendor.api_key_env} left blank — {vendor.label} disabled "
                f"until you add it"
            )
    return updates


def _build_llm_toml(
    selected: list[_Vendor],
    existing_priority: list[str],
    existing_llm: dict,
) -> dict:
    """Produce the deep-merge payload for the ``[llm]`` table."""
    selected_route_ids = [v.route_id for v in selected]

    # Preserve any priority entries the user added for vendors we don't
    # manage in v1, appended after our managed ones.
    managed_route_prefixes = {f"{v.key}:" for v in _VENDORS_BY_KEY.values()}
    preserved_tail = [
        r for r in existing_priority
        if not any(r.startswith(p) for p in managed_route_prefixes)
    ]
    route_priority = selected_route_ids + preserved_tail

    vendors_block: dict = {}
    for vendor in selected:
        vendors_block.setdefault("vendors", {}).update(vendor.toml_block["vendors"])

    payload: dict = {"route_priority": route_priority, **vendors_block}

    # Preserve user-authored ensemble settings, etc.
    for key in ("ensemble", "catalog", "mandate"):
        if key in existing_llm:
            payload[key] = existing_llm[key]

    return payload
