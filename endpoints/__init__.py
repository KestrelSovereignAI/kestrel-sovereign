"""Repo-root re-export shim — endpoints/ moved into the package.

The endpoints package now lives at ``kestrel_sovereign.endpoints``;
this stub keeps the documented source-clone forms working
(``from endpoints.X import Y``, ``import endpoints.X``). Pip-installed
users import the in-package path directly.

Implementation note (codex review v2 on PR #1097): a naive shim that
just shares ``__path__`` with the canonical package would let
``importlib`` *re-execute* a submodule under the ``endpoints.*`` name
on first import, producing two distinct module objects with diverging
globals/state. The fix is a meta-path finder that intercepts every
``endpoints.X`` lookup, imports the canonical
``kestrel_sovereign.endpoints.X``, and aliases it back into
``sys.modules['endpoints.X']``. After that, ``import endpoints.foo as
a; import kestrel_sovereign.endpoints.foo as b; a is b`` is True.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from typing import Sequence

_CANONICAL = "kestrel_sovereign.endpoints"


class _LegacyEndpointsLoader(importlib.abc.Loader):
    """Resolve ``endpoints.X`` to the already-loaded ``kestrel_sovereign.endpoints.X``.

    On ``exec_module`` we delegate to the canonical importer and then
    alias the result back into ``sys.modules`` under the legacy name.
    Doing it this way (rather than copy-and-execute) guarantees that
    every legacy import returns the same module object that the
    in-package code uses — patches and globals stay in sync.
    """

    def __init__(self, canonical_fullname: str) -> None:
        self._canonical = canonical_fullname

    def create_module(self, spec):  # noqa: D401 — Loader contract
        # Returning None tells the import machinery to use a default
        # module object; we replace it in exec_module().
        return None

    def exec_module(self, module) -> None:
        canonical = importlib.import_module(self._canonical)
        sys.modules[module.__name__] = canonical


class _LegacyEndpointsFinder(importlib.abc.MetaPathFinder):
    """Find ``endpoints.<sub>`` and route it to ``kestrel_sovereign.endpoints.<sub>``."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target=None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith("endpoints."):
            return None
        suffix = fullname[len("endpoints."):]
        canonical_full = f"{_CANONICAL}.{suffix}"
        # Make sure the canonical target actually exists. If it
        # doesn't, fall through to the default finders so the operator
        # gets the normal ModuleNotFoundError.
        if importlib.util.find_spec(canonical_full) is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _LegacyEndpointsLoader(canonical_full),
        )


# Idempotent registration — repeated imports of the shim shouldn't
# stack duplicate finders.
if not any(isinstance(f, _LegacyEndpointsFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _LegacyEndpointsFinder())


# Re-export the canonical package's public namespace so
# ``from endpoints import router_factory`` (or attribute-style
# ``endpoints.foo``) sees what the canonical package exposes.
_pkg = importlib.import_module(_CANONICAL)
globals().update({k: v for k, v in vars(_pkg).items() if not k.startswith("_")})
