"""Repo-root re-export shim — endpoints/ moved into the package.

The endpoints package now lives at ``kestrel_sovereign.endpoints``;
this stub keeps the documented source-clone forms working
(``from endpoints.X import Y``). Pip-installed users import the
in-package path directly.

The shim is intentionally minimal — don't add logic here.
"""

from __future__ import annotations

import importlib
import sys

# Import the real package and alias each submodule so
# ``from endpoints.foo import bar`` works exactly the same as
# ``from kestrel_sovereign.endpoints.foo import bar``.
_pkg = importlib.import_module("kestrel_sovereign.endpoints")

# Mirror the package's path so ``importlib`` can find subpackages
# under the bare name. Without this, ``import endpoints.auth_oauth``
# would fail with ModuleNotFoundError because Python would only see
# the shim's own (non-existent) submodules.
__path__ = list(_pkg.__path__)

# Also alias already-loaded submodules so attribute-style access
# (``endpoints.foo``) sees them.
for _name, _mod in list(sys.modules.items()):
    if _name.startswith("kestrel_sovereign.endpoints."):
        _suffix = _name[len("kestrel_sovereign.endpoints."):]
        sys.modules[f"endpoints.{_suffix}"] = _mod

# Re-export the package's public namespace.
globals().update({k: v for k, v in vars(_pkg).items() if not k.startswith("__")})
