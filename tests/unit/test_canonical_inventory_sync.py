"""Sync checks between the canonical inventory and the live code surface."""

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_ROOT = PROJECT_ROOT / "kestrel_sovereign" / "features"
ENDPOINTS_ROOT = PROJECT_ROOT / "endpoints"


def _discover_feature_modules() -> list[str]:
    modules = []
    for item in sorted(FEATURES_ROOT.iterdir()):
        if item.name.startswith(("_", ".")):
            continue
        if item.is_dir():
            if (item / "feature.py").exists() or (item / "__init__.py").exists():
                modules.append(item.name)
        elif item.is_file() and item.suffix == ".py" and item.name != "base.py":
            modules.append(item.stem)
    return modules


def _discover_exported_feature_classes() -> list[str]:
    names = []
    pattern = re.compile(r"^class\s+(\w+)\(Feature\):", re.M)
    for path in FEATURES_ROOT.rglob("*.py"):
        if path.name == "base.py":
            continue
        text = path.read_text()
        names.extend(match.group(1) for match in pattern.finditer(text))
    return sorted(names)


def _discover_endpoint_router_files() -> list[str]:
    return sorted(
        path.name
        for path in ENDPOINTS_ROOT.glob("*.py")
        if path.name not in {"__init__.py", "agent_helpers.py"}
    )


def _discover_router_routes() -> list[str]:
    routes = []
    pattern = re.compile(r'@router\.(get|post|put|delete|patch|head)\("([^"]+)"')
    prefix_pattern = re.compile(r'APIRouter\(prefix="([^"]*)"')

    for path in sorted(ENDPOINTS_ROOT.glob("*.py")):
        if path.name in {"__init__.py", "agent_helpers.py"}:
            continue
        text = path.read_text()
        prefix_match = prefix_pattern.search(text)
        prefix = prefix_match.group(1) if prefix_match else ""
        for match in pattern.finditer(text):
            method = match.group(1).upper()
            route_path = f"{prefix}{match.group(2)}"
            routes.append(f"{method} {route_path}")
    return sorted(routes)


def _discover_app_routes() -> list[str]:
    text = (PROJECT_ROOT / "server.py").read_text()
    pattern = re.compile(r'@app\.(get|post|put|delete|patch|head)\("([^"]+)"')
    return sorted(f"{match.group(1).upper()} {match.group(2)}" for match in pattern.finditer(text))


def test_canonical_inventory_keeps_feature_snapshot_counts_in_sync():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text()
    modules = _discover_feature_modules()
    classes = _discover_exported_feature_classes()
    expected = f"Current audited snapshot: `{len(modules)}` discoverable modules and `{len(classes)}` exported `Feature` subclasses."
    assert expected in text


def test_canonical_inventory_mentions_all_router_files():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text()
    for router_file in _discover_endpoint_router_files():
        assert f"`endpoints/{router_file}`" in text


def test_canonical_inventory_mentions_all_discoverable_feature_modules():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text()
    for module in _discover_feature_modules():
        assert f"`{module}`" in text


def test_canonical_inventory_mentions_all_router_routes():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text()
    for route in _discover_router_routes():
        assert f"`{route}`" in text


def test_canonical_inventory_mentions_all_app_routes():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text()
    for route in _discover_app_routes():
        assert f"`{route}`" in text


def test_canonical_inventory_links_point_to_existing_paths():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text()
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

    for target in link_pattern.findall(text):
        if target.startswith(("http://", "https://", "#")):
            continue
        assert (PROJECT_ROOT / target).exists(), target
