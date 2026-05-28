"""Every signal source's prompt_template must ship inside the package (#1415).

Before this guard rail, four of five signal sources resolved their
``prompt_template`` via ``Path(__file__).resolve().parents[3]`` — which
points at the repo root, not the package. On any wheel install the
file isn't there, ``_render_prompt`` raises FileNotFoundError, the
COGNITION dispatch fails, and the recipient agent never wakes. Only
``channel.message`` worked because ``channels.py`` used ``parents[2]``
(package-internal) and the template lived inside the package.

These tests are the future-proofing — they walk every registered
source's ``prompt_template`` and assert:

  1. It is a real readable file (catches missing files).
  2. It resolves inside the ``kestrel_sovereign/`` package tree
     (catches future additions that drift back to the repo-root path).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import kestrel_sovereign


def _all_signal_source_templates():
    """Discover ``PROMPT_TEMPLATE`` attributes on every in-tree signal source.

    We inspect the module-level attribute directly rather than calling
    ``build_*_registration()`` because some builders take required
    keyword arguments (e.g. ``heartbeat`` needs interval / active-hours
    config). Every source we ship today exposes its template path as
    a module-level ``PROMPT_TEMPLATE`` constant — that's the contract
    we pin.
    """
    import importlib
    import pkgutil

    import kestrel_sovereign.signals.sources as sources_pkg

    triples = []
    for info in pkgutil.iter_modules(sources_pkg.__path__):
        mod = importlib.import_module(f"{sources_pkg.__name__}.{info.name}")
        tmpl = getattr(mod, "PROMPT_TEMPLATE", None)
        if tmpl is not None:
            triples.append((mod.__name__, "PROMPT_TEMPLATE", tmpl))
    return triples


_REGISTRATIONS = _all_signal_source_templates()


@pytest.fixture(scope="module")
def package_root() -> Path:
    """Resolved root of the ``kestrel_sovereign`` package on disk."""
    return Path(kestrel_sovereign.__file__).resolve().parent


@pytest.mark.parametrize(
    "mod_name,attr,tmpl",
    _REGISTRATIONS,
    ids=[f"{m}.{a}" for m, a, _ in _REGISTRATIONS],
)
class TestPromptTemplateShipsWithWheel:
    def test_prompt_template_path_resolves_to_a_file(
        self, mod_name, attr, tmpl
    ):
        """A COGNITION source without a renderable template silently
        breaks the dispatcher's ``_render_prompt`` — the template file
        MUST exist on disk wherever the package is installed."""
        assert isinstance(tmpl, Path), (
            f"{mod_name}.{attr} must be a Path, got {type(tmpl).__name__}"
        )
        assert tmpl.is_file(), (
            f"{mod_name}.{attr} path {tmpl!s} does not resolve to a "
            "file. The wake-up signal for this source will raise "
            "FileNotFoundError in _render_prompt and the recipient "
            "agent will never be woken. Did you forget to add the .md "
            "file inside the package, or use parents[N] that resolves "
            "outside it?"
        )

    def test_prompt_template_lives_inside_the_package(
        self, mod_name, attr, tmpl, package_root
    ):
        """Files outside ``kestrel_sovereign/`` are not packaged in the
        wheel (see [tool.hatch.build.targets.sdist] include in
        pyproject.toml). Any prompt that resolves outside the package
        works in a source checkout but disappears on PyPI install —
        same failure mode as #1415."""
        resolved = tmpl.resolve()
        try:
            resolved.relative_to(package_root)
        except ValueError:
            pytest.fail(
                f"{mod_name}.{attr} resolved to {resolved!s} — OUTSIDE "
                f"the kestrel_sovereign package ({package_root}). The "
                "wheel's [tool.hatch.build.targets.sdist] include only "
                "ships files inside kestrel_sovereign/**/* — this "
                "template would resolve correctly in a source checkout "
                "but raise FileNotFoundError on every PyPI install. "
                "Move the .md file into "
                "kestrel_sovereign/prompts/signals/ and switch the "
                "source's parents[N] to parents[2]."
            )


class TestPackagedTemplateInventory:
    """Catches a different drift: an .md file dropped into
    ``kestrel_sovereign/prompts/signals/`` that no signal source
    actually registers. Useful while we still have repo-root cruft
    around — flags any future leftover that masquerades as a signal
    template but isn't referenced."""

    def test_every_packaged_signal_template_is_referenced(self, package_root):
        signals_dir = package_root / "prompts" / "signals"
        if not signals_dir.is_dir():
            pytest.skip("no prompts/signals directory in the package")
        referenced = {tmpl.resolve() for _, _, tmpl in _REGISTRATIONS}
        packaged = {p.resolve() for p in signals_dir.glob("*.md")}
        orphaned = packaged - referenced
        assert not orphaned, (
            "Packaged signal-prompt templates not referenced by any "
            f"signal source PROMPT_TEMPLATE constant: {sorted(orphaned)}. "
            "Either wire them into a source module or remove them."
        )
