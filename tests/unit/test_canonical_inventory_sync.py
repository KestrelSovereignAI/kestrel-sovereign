"""Sync checks between the canonical inventory and the live code surface."""

from pathlib import Path
import re

from kestrel_sovereign.feature_inventory import (
    discover_app_routes,
    discover_core_feature_modules,
    discover_endpoint_router_files,
    discover_exported_feature_classes,
    discover_router_routes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_inventory_keeps_feature_snapshot_counts_in_sync():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text(encoding="utf-8")
    modules = discover_core_feature_modules()
    classes = discover_exported_feature_classes()
    expected = f"Current audited snapshot: `{len(modules)}` discoverable modules and `{len(classes)}` exported `Feature` subclasses."
    assert expected in text


def test_canonical_inventory_mentions_all_router_files():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text(encoding="utf-8")
    for router_file in discover_endpoint_router_files():
        assert f"`kestrel_sovereign/endpoints/{router_file}`" in text


def test_canonical_inventory_mentions_all_discoverable_feature_modules():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text(encoding="utf-8")
    for module in discover_core_feature_modules():
        assert f"`{module}`" in text


def test_canonical_inventory_mentions_all_router_routes():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text(encoding="utf-8")
    for route in discover_router_routes():
        assert f"`{route.method} {route.path}`" in text


def test_canonical_inventory_mentions_all_app_routes():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text(encoding="utf-8")
    for route in discover_app_routes():
        assert f"`{route.method} {route.path}`" in text


def test_canonical_inventory_links_point_to_existing_paths():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text(encoding="utf-8")
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

    for target in link_pattern.findall(text):
        if target.startswith(("http://", "https://", "#")):
            continue
        assert (PROJECT_ROOT / target).exists(), target
