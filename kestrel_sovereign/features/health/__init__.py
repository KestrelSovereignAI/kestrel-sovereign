"""Health Feature - periodic liveness probes for the agent.

Not to be confused with :mod:`kestrel_sovereign.heartbeat` — in the OpenClaw /
kestrel-claw tradition a *heartbeat* is a scheduled LLM turn that reads
``HEARTBEAT.md`` and surfaces work that needs attention. That lives in
``HeartbeatRunner``.

This feature is a *liveness probe*: cheap structured checks of the agent's
subsystems (database, LLM-service wiring, memory system, disk, context
budget). It runs on a short interval (default 60s) and never invokes the
LLM. Results land in the ``health_log`` table and surface via the
``/agent/health/*`` endpoints and the ``!health`` commands.
"""

from .feature import HealthFeature

__all__ = ["HealthFeature"]
