"""Sync checks between the canonical inventory and the live code surface."""

from pathlib import Path

from kestrel_sovereign.feature_inventory import (
    GENERATED_END,
    GENERATED_START,
    build_inventory,
    discover_app_routes,
    discover_core_feature_modules,
    discover_endpoint_router_files,
    discover_exported_feature_classes,
    discover_router_routes,
    render_inventory_markdown,
    replace_generated_inventory,
)
from scripts import check_docs_links


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_inventory_generated_region_is_exact():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text(encoding="utf-8")
    start = text.index(GENERATED_START)
    end = text.index(GENERATED_END, start) + len(GENERATED_END)
    checked_in = text[start:end].strip()
    rendered = render_inventory_markdown(build_inventory()).strip()

    assert checked_in == rendered


def test_writing_the_inventory_twice_changes_nothing_the_second_time():
    """#3116: ``--write`` appended one blank line on every invocation.

    The existing region test cannot see this: it compares both sides
    ``.strip()``ed, so whitespace inside the region is invisible to it
    and the drift was unbounded and silent. Byte equality is the only
    assertion that catches it.

    A synthetic document rather than the real file, so the test states
    the property (a second write is a no-op) without depending on the
    checked-in file being current — that is a different claim, and
    test_canonical_inventory_generated_region_is_exact already makes it.
    """
    generated = f"{GENERATED_START}\n\n| a | b |\n\n{GENERATED_END}\n"
    document = (
        "# Title\n\nPreamble.\n\n"
        f"{GENERATED_START}\n\nstale\n\n{GENERATED_END}\n"
        "\n## Authentication Surface\n\nTail text.\n"
    )

    once = replace_generated_inventory(document, generated)
    twice = replace_generated_inventory(once, generated)

    assert twice == once, "a second write must be a no-op"
    assert once.endswith("Tail text.\n"), "the trailing newline must survive"
    assert "\n\n\n" not in once, "no blank line may accumulate at the boundary"
    # The property has to hold for a document that already carries the
    # accumulated damage, since that is what every checkout has.
    damaged = once.replace(
        f"{GENERATED_END}\n\n", f"{GENERATED_END}\n\n\n\n\n\n", 1
    )
    assert replace_generated_inventory(damaged, generated) == once, (
        "the splice must collapse blank lines a previous run left, not "
        "preserve them"
    )


def test_writing_the_inventory_keeps_a_region_that_ends_the_file():
    """The region can be the last thing in the file.

    Then there is no tail to separate from, and adding a blank line
    would leave the file ending in one. Dropping the newline instead is
    the failure the first attempt at #3116 produced — git reports
    ``\\ No newline at end of file``.
    """
    generated = f"{GENERATED_START}\n\n| a | b |\n\n{GENERATED_END}\n"
    document = f"# Title\n\n{GENERATED_START}\n\nstale\n\n{GENERATED_END}\n"

    once = replace_generated_inventory(document, generated)

    assert once.endswith(f"{GENERATED_END}\n"), once[-60:]
    assert replace_generated_inventory(once, generated) == once


def test_the_checked_in_inventory_is_a_fixed_point():
    """The file on disk must already be what another write would produce.

    The region test proves the generated content matches; this proves
    the splice around it does too, so `--write` on a clean checkout
    leaves the tree clean — which is the whole point of a generated
    file being checked in.
    """
    path = PROJECT_ROOT / "KESTREL_FEATURES.md"
    existing = path.read_text(encoding="utf-8")
    rewritten = replace_generated_inventory(
        existing, render_inventory_markdown(build_inventory())
    )
    assert rewritten == existing, (
        "python -m scripts.generate_feature_inventory --write would change "
        "the checked-in file"
    )


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
    inventory = PROJECT_ROOT / "KESTREL_FEATURES.md"
    broken_links = check_docs_links.check_file(inventory)

    assert broken_links == [], "\n".join(link.format() for link in broken_links)
