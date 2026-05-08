"""Backward-compat shim for ``uvicorn server:app`` from a source clone.

The real module lives at :mod:`kestrel_sovereign.server` so it ships in the
pip-installed wheel (``pyproject.toml`` only includes ``kestrel_sovereign/``
in ``[tool.hatch.build.targets.sdist]``). This file exists only so the
documented source-clone workflow ``uvicorn server:app`` keeps working from
the repo root.

Do not put logic here. Add it to :mod:`kestrel_sovereign.server`.

Implementation note: we re-export the *full* module namespace (including
underscore-prefixed names like ``_DEFAULT_CORS_ORIGINS`` that the test
suite reaches into) by copying ``vars(...)`` rather than ``from ... import
*``.  ``import *`` honors ``__all__`` and would otherwise drop the private
attributes the existing tests depend on.
"""
from kestrel_sovereign import server as _impl

# Mirror every public + private attribute so ``from server import X`` works
# for any X that the in-package module exposes.
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

# Explicit re-export so ASGI loaders (uvicorn server:app) don't have to
# discover ``app`` through the dict-merge above.
app = _impl.app


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8888))
    uvicorn.run(app, host="0.0.0.0", port=port)
