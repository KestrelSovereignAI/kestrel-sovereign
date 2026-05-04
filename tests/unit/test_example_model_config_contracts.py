"""Contracts for example model config files and catalog source-of-truth shape.

Pins the schema of `kestrel.toml.example`'s `[llm]` block to the current
vendor/route/model architecture. The standalone `llm_config.toml.example`
was removed in #940; tests pinning its shape went with it.
"""

from pathlib import Path

import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _first_route(vendors: dict) -> tuple[str, str, dict]:
    """Return (vendor, route, route_config) for the first concrete route found.

    Raises if no vendor has a routes.<name> subsection — the whole point of the
    new schema is that at least one route must be wired.
    """
    for vendor_name, vendor_block in vendors.items():
        routes = vendor_block.get("routes", {})
        for route_name, route_cfg in routes.items():
            return vendor_name, route_name, route_cfg
    raise AssertionError("No vendors.*.routes.* entries found")


def test_unified_example_llm_block_uses_vendor_route_schema():
    """Pin the vendor/route schema for `kestrel.toml.example`'s `[llm]` block.

    Regression for #732 (originally written against `llm_config.toml.example`,
    repointed at the unified file in #940). Prior to the vendor/route refactor
    the example used flat `[openai]` / `[ollama]` sections, so clean installs
    copied in a file that the new provider_registry couldn't initialize. Pin
    the new shape so this cannot silently drift back.
    """
    config = tomllib.loads((PROJECT_ROOT / "kestrel.toml.example").read_text(encoding="utf-8"))
    llm = config["llm"]

    assert "route_priority" in llm, "[llm] must define route_priority"
    assert llm["route_priority"], "route_priority cannot be empty"
    assert all(":" in entry for entry in llm["route_priority"]), \
        "route_priority entries must be in 'vendor:route' form"

    assert "vendors" in llm, "[llm] must define a [llm.vendors.*] namespace"
    vendor, route, route_cfg = _first_route(llm["vendors"])
    assert "adapter" in route_cfg, f"{vendor}:{route} must declare an adapter"
    assert route_cfg.get("model") == "auto", \
        f"{vendor}:{route} should use auto model selection in the shipped example"
    assert route_cfg.get("selection_hints") or "api_key_env" in route_cfg or "host" in route_cfg, \
        f"{vendor}:{route} needs either selection_hints or a concrete endpoint/auth hint"


def test_unified_example_priorities_match_declared_vendors():
    """Every entry in route_priority must resolve to a real
    `[llm.vendors.<vendor>.routes.<route>]` block — otherwise the priority list
    references routes that cannot initialize and the example misleads."""
    config = tomllib.loads((PROJECT_ROOT / "kestrel.toml.example").read_text(encoding="utf-8"))
    llm = config["llm"]

    for entry in llm["route_priority"]:
        vendor, _, route = entry.partition(":")
        vendor_block = llm.get("vendors", {}).get(vendor)
        assert vendor_block is not None, \
            f"route_priority entry '{entry}' references undefined vendor '{vendor}'"
        assert route in vendor_block.get("routes", {}), \
            f"route_priority entry '{entry}' references undefined route '{route}' under vendor '{vendor}'"


def test_root_model_catalog_is_manual_overrides_only():
    catalog_text = (PROJECT_ROOT / "model_catalog.toml").read_text(encoding="utf-8")
    catalog = tomllib.loads(catalog_text)

    assert "Manual Overrides Only" in catalog_text
    assert "featured" not in catalog
    assert "display_name_overrides" in catalog
    assert "context_limits_override" in catalog


def test_package_local_model_catalog_duplicate_is_removed():
    assert not (PROJECT_ROOT / "kestrel_sovereign/model_catalog.toml").exists()
