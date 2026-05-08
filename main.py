"""Backward-compat shim for ``python main.py`` from a source clone.

The real module lives at :mod:`kestrel_sovereign.main` so it ships in the
pip-installed wheel. This file exists only so the documented source-clone
workflow (``uv run python main.py <agent_dir>``, plus ``from main import
get_agent_did_async`` from any in-tree code) keeps working from the repo
root.

Do not put logic here. Add it to :mod:`kestrel_sovereign.main`.
"""
from kestrel_sovereign.main import (  # noqa: F401  (re-export public surface)
    get_agent_did_async,
    get_agent_by_did,
    main,
)


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nForced exit.")
