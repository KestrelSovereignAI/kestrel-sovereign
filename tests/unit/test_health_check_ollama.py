"""Unit tests for kestrel_sovereign.health_check._resolve_expected_ollama_model.

Caught in #944 codex review: the helper was reading a flat ``llm_config["ollama"]``
block that no longer exists post-vendor/route refactor (#688). The route config
now lives at ``vendors.ollama.routes.local``; on configs without the legacy
flat block, the fallback resolution path silently returned ``None`` for the
selection-hints branch even when ``kestrel.toml`` carried hints.
"""

from __future__ import annotations

from unittest.mock import patch

from kestrel_sovereign.health_check import _resolve_expected_ollama_model


def test_explicit_model_in_route_returns_that_model():
    cfg = {
        "vendors": {
            "ollama": {
                "routes": {
                    "local": {"model": "llama3.2:1b"},
                },
            },
        },
    }
    with patch("kestrel_sovereign.health_check.load_section", return_value=cfg):
        assert _resolve_expected_ollama_model(["llama3.2:1b", "qwen3:4b"]) == "llama3.2:1b"


def test_auto_model_uses_route_selection_hints_on_resolver_failure():
    """When discovery resolution fails, the helper must walk
    ``vendors.ollama.routes.local.selection_hints`` (post-#688 shape) — not
    a flat ``[ollama].selection_hints`` block, which no longer exists."""
    cfg = {
        "vendors": {
            "ollama": {
                "routes": {
                    "local": {
                        "model": "auto",
                        "selection_hints": ["qwen", "latest"],
                    },
                },
            },
        },
    }
    installed = ["llama3.2:3b", "qwen3:4b", "gpt-oss:20b"]

    with patch("kestrel_sovereign.health_check.load_section", return_value=cfg), \
         patch(
             "kestrel_sovereign.health_check.resolve_provider_default",
             side_effect=RuntimeError("no cache"),
         ):
        assert _resolve_expected_ollama_model(installed) == "qwen3:4b"


def test_no_route_config_falls_back_to_single_installed():
    """No ollama vendor declared at all: the single-installed-model fallback
    is the last line of defence."""
    cfg = {"vendors": {}}
    with patch("kestrel_sovereign.health_check.load_section", return_value=cfg), \
         patch(
             "kestrel_sovereign.health_check.resolve_provider_default",
             side_effect=RuntimeError("no cache"),
         ):
        assert _resolve_expected_ollama_model(["only-model:latest"]) == "only-model:latest"


def test_multiple_installed_no_hints_no_resolver_returns_none():
    """Ambiguous: multiple models installed, no hints, resolver fails → None
    (the caller surfaces the warning)."""
    cfg = {
        "vendors": {
            "ollama": {"routes": {"local": {"model": "auto"}}},
        },
    }
    with patch("kestrel_sovereign.health_check.load_section", return_value=cfg), \
         patch(
             "kestrel_sovereign.health_check.resolve_provider_default",
             side_effect=RuntimeError("no cache"),
         ):
        assert _resolve_expected_ollama_model(["a:latest", "b:latest"]) is None


def test_legacy_flat_ollama_block_is_ignored():
    """Defensive: if a stale config still has a flat ``[ollama]`` block, we
    must NOT honour it — that's the bug this regression test pins. The
    route-shaped block is the source of truth."""
    cfg = {
        # Legacy shape that should be ignored.
        "ollama": {"model": "stale:legacy", "selection_hints": ["legacy"]},
        "vendors": {
            "ollama": {
                "routes": {
                    "local": {"model": "current:correct"},
                },
            },
        },
    }
    with patch("kestrel_sovereign.health_check.load_section", return_value=cfg):
        assert _resolve_expected_ollama_model(["current:correct"]) == "current:correct"
