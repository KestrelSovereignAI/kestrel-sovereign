"""Tests for CLI extension discovery via the ``kestrel_sovereign.cli`` group (#2046).

Covers: callable discovery, registration + dispatch, name-collision rejection
(core wins), broken-extension isolation, and missing-feature no-op.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign import cli_extensions
from kestrel_sovereign.cli_extensions import register_cli_extensions
from kestrel_sovereign.entrypoints import discover_entry_point_callables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry_point(name: str, obj):
    ep = MagicMock()
    ep.name = name
    ep.value = f"mock_package:{name}"
    ep.load.return_value = obj
    return ep


def _patch_entry_points(entry_points: list):
    mock_eps = MagicMock()
    mock_eps.select.return_value = entry_points
    return patch(
        "kestrel_sovereign.entrypoints.importlib.metadata.entry_points",
        return_value=mock_eps,
    )


def _fresh_subparsers():
    parser = argparse.ArgumentParser(prog="kestrel")
    return parser, parser.add_subparsers(dest="command")


# ===========================================================================
# discover_entry_point_callables
# ===========================================================================

class TestDiscoverCallables:
    def test_discovers_callable(self):
        def fn(subparsers):
            pass

        ep = _make_entry_point("myfeature", fn)
        with _patch_entry_points([ep]):
            result = discover_entry_point_callables("kestrel_sovereign.cli")
        assert result == [("myfeature", fn)]

    def test_skips_non_callable(self):
        ep = _make_entry_point("bad", 123)
        with _patch_entry_points([ep]):
            result = discover_entry_point_callables("kestrel_sovereign.cli")
        assert "bad" not in dict(result)

    def test_load_failure_is_skipped(self):
        ep = MagicMock()
        ep.name = "boom"
        ep.value = "mock_package:boom"
        ep.load.side_effect = ImportError("no such module")
        with _patch_entry_points([ep]):
            result = discover_entry_point_callables("kestrel_sovereign.cli")
        assert result == []

    def test_missing_group_is_noop(self):
        with _patch_entry_points([]):
            result = discover_entry_point_callables("kestrel_sovereign.cli")
        assert result == []

    def test_preserves_duplicate_names_in_order(self):
        def first(subparsers):
            pass

        def second(subparsers):
            pass

        eps = [
            _make_entry_point("dup", first),
            _make_entry_point("dup", second),
        ]
        with _patch_entry_points(eps):
            result = discover_entry_point_callables("kestrel_sovereign.cli")
        # Both registrations survive discovery, in order — a dict would have
        # collapsed them and lost the first callable before the caller could
        # apply "first wins" collision handling.
        assert result == [("dup", first), ("dup", second)]


# ===========================================================================
# register_cli_extensions
# ===========================================================================

class TestRegisterCliExtensions:
    def test_registers_and_dispatches(self):
        def run(args):
            return 0

        def add_subparser(subparsers):
            p = subparsers.add_parser("myfeature")
            sub = p.add_subparsers(dest="myfeature_command")
            sub.add_parser("status")
            p.set_defaults(_handler=run)

        parser, subparsers = _fresh_subparsers()
        ep = _make_entry_point("myfeature", add_subparser)
        with _patch_entry_points([ep]):
            registered = register_cli_extensions(subparsers)

        assert registered == ["myfeature"]
        # The subparser is wired and its handler dispatches via _handler.
        args = parser.parse_args(["myfeature", "status"])
        assert args.command == "myfeature"
        assert args._handler is run

    def test_collision_with_core_command_rejected(self):
        registrations = []

        def add_subparser(subparsers):
            registrations.append(True)
            subparsers.add_parser("status")

        parser, subparsers = _fresh_subparsers()
        # Simulate a core command already claiming "status".
        subparsers.add_parser("status")

        ep = _make_entry_point("status", add_subparser)
        with _patch_entry_points([ep]):
            registered = register_cli_extensions(subparsers)

        assert registered == []
        # The colliding extension's callable was never invoked (core wins).
        assert registrations == []

    def test_broken_extension_is_isolated(self):
        def good_run(args):
            return 0

        def good(subparsers):
            p = subparsers.add_parser("good")
            p.set_defaults(_handler=good_run)

        def broken(subparsers):
            raise RuntimeError("registration blew up")

        parser, subparsers = _fresh_subparsers()
        eps = [
            _make_entry_point("broken", broken),
            _make_entry_point("good", good),
        ]
        with _patch_entry_points(eps):
            registered = register_cli_extensions(subparsers)

        # Broken one skipped; good one still registered.
        assert registered == ["good"]
        args = parser.parse_args(["good"])
        assert args._handler is good_run

    def test_second_extension_colliding_with_first_rejected(self):
        calls = []

        def first(subparsers):
            calls.append("first")
            subparsers.add_parser("dup")

        def second(subparsers):
            calls.append("second")
            subparsers.add_parser("dup")

        parser, subparsers = _fresh_subparsers()
        eps = [
            _make_entry_point("dup", first),
            _make_entry_point("dup", second),
        ]
        with _patch_entry_points(eps):
            registered = register_cli_extensions(subparsers)

        # Only the first registration of the name wins; the second callable is
        # rejected before invocation (it would otherwise raise an argparse
        # "conflicting subparser" error). This only works because discovery
        # preserves both duplicate pairs rather than collapsing them in a dict.
        assert registered == ["dup"]
        assert calls == ["first"]

    def test_no_extensions_is_noop(self):
        parser, subparsers = _fresh_subparsers()
        with _patch_entry_points([]):
            registered = register_cli_extensions(subparsers)
        assert registered == []


# ===========================================================================
# Integration with the real build_parser / main dispatch
# ===========================================================================

class TestBuildParserIntegration:
    def test_build_parser_registers_extension(self):
        from kestrel_sovereign import cli

        sentinel = {}

        def run(args):
            sentinel["called"] = True
            return 7

        def add_subparser(subparsers):
            p = subparsers.add_parser("demoext")
            sub = p.add_subparsers(dest="demoext_command")
            sub.add_parser("status")
            p.set_defaults(_handler=run)

        ep = _make_entry_point("demoext", add_subparser)
        with _patch_entry_points([ep]):
            parser = cli.build_parser()
            args = parser.parse_args(["demoext", "status"])

        assert args._handler is run

    def test_main_dispatches_extension_handler(self):
        from kestrel_sovereign import cli

        calls = {}

        def run(args):
            calls["ran"] = True
            return 7

        def add_subparser(subparsers):
            p = subparsers.add_parser("demoext")
            sub = p.add_subparsers(dest="demoext_command")
            sub.add_parser("status")
            p.set_defaults(_handler=run)

        ep = _make_entry_point("demoext", add_subparser)
        with _patch_entry_points([ep]), patch.object(
            cli.sys, "argv", ["kestrel", "demoext", "status"]
        ):
            rc = cli.main()

        assert rc == 7
        assert calls.get("ran") is True

    def test_core_command_unaffected_by_extension(self):
        from kestrel_sovereign import cli

        def add_subparser(subparsers):
            subparsers.add_parser("demoext").set_defaults(_handler=lambda a: 0)

        ep = _make_entry_point("demoext", add_subparser)
        with _patch_entry_points([ep]):
            parser = cli.build_parser()
            args = parser.parse_args(["list"])
        assert args.command == "list"


# ===========================================================================
# Reference example module
# ===========================================================================

class TestExampleModule:
    def test_example_registers_status(self):
        from kestrel_sovereign import cli_extension_example

        parser, subparsers = _fresh_subparsers()
        cli_extension_example.add_example_subparser(subparsers)
        args = parser.parse_args(["example", "status"])
        assert args.command == "example"
        assert args.example_command == "status"
        assert args._handler is cli_extension_example.run
