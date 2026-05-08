"""Backward-compat shim for ``uvicorn host:app`` from a source clone.

The real module lives at :mod:`kestrel_sovereign.host` so it ships in the
pip-installed wheel (``pyproject.toml`` only includes ``kestrel_sovereign/``
in ``[tool.hatch.build.targets.sdist]``). This file exists only so the
documented source-clone workflow ``uvicorn host:app`` keeps working from
the repo root.

Do not put logic here. Add it to :mod:`kestrel_sovereign.host`.

Implementation note: we re-export the *full* module namespace (including
underscore-prefixed names like ``_oauth_required`` that the test suite or
ops scripts may reach into) by copying ``vars(...)`` rather than
``from ... import *``. ``import *`` honors ``__all__`` and would otherwise
drop private attributes.
"""
from kestrel_sovereign import host as _impl

# Mirror every attribute so ``from host import X`` works for any X.
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

# Explicit re-export so ASGI loaders (uvicorn host:app) don't have to
# discover ``app`` through the dict-merge above.
app = _impl.app


if __name__ == "__main__":
    import uvicorn
    cfg = _impl.load_multi_agent_config()
    uvicorn.run(app, host=cfg.host.bind, port=cfg.host.port)
