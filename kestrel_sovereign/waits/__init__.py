"""Generic wait engine — one poll loop for every feature's blocking wait.

Every "wait" in the tree is the same shape: poll some external state
until it reaches a terminal condition or a timeout expires, then map the
outcome onto a :class:`ToolResult`. Historically each feature shipped its
own copy of that loop (``wait_for_task``, ``talon_wait``, CI polling),
each with its own cap, interval, terminal vocabulary, and result mapping.

This package is the single engine. A feature implements the SDK
:class:`~kestrel_sdk.tools.Waitable` contract for its handle kind and the
engine owns everything else:

* :func:`run_wait_loop` — the pure poll loop (no agent dependency).
  Named tools call it directly with a provider they construct.
* :class:`WaitRegistry` — the per-agent set of registered providers,
  so the generic ``wait("<kind>:<handle>")`` tool (and the Wave-2
  reconciler cron) can dispatch to a provider owned by another feature.

Core must never import features; features import from here.
"""

from .engine import WaitRegistry, run_wait_loop

__all__ = ["WaitRegistry", "run_wait_loop"]
