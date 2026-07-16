"""Shared state passed through every wizard step.

Each step takes the context, mutates it (via the prompter and file
writers), and returns a list of human-readable changes. The orchestrator
prints those changes at the end so the user sees what setup actually did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from kestrel_sovereign.setup.prompts import Prompter


class Flow(str, Enum):
    """How chatty the wizard is.

    INTERACTIVE : ask everything, default to existing values where present.
    QUICKSTART  : accept defaults silently; only prompt for missing secrets.
    CHECK       : never prompt, never write — report only.
    """

    INTERACTIVE = "interactive"
    QUICKSTART = "quickstart"
    CHECK = "check"


@dataclass
class SetupContext:
    """Pass-through state for the wizard.

    ``project_dir`` is the kestrel-sovereign repo root (where ``.env``
    and ``kestrel.toml`` live). ``agent_data_root`` is where new
    agents are inceptioned (``<project_dir>/agent_data`` by default).
    """

    project_dir: Path
    agent_data_root: Path
    flow: Flow
    prompter: Prompter
    reset: bool = False
    is_test_instance: bool = False
    changes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    # Set by a step that hits an unrecoverable, workflow-wide condition (e.g. an
    # unresolved KESTREL_DATA_KEY custody conflict): the orchestrator stops
    # running later steps so no key-dependent mutation proceeds under it (#2468).
    halted: bool = False
    halt_reason: str | None = None

    @property
    def env_path(self) -> Path:
        return self.project_dir / ".env"

    @property
    def kestrel_toml_path(self) -> Path:
        return self.project_dir / "kestrel.toml"

    @property
    def multi_agent_toml_path(self) -> Path:
        return self.project_dir / "multi_agent.toml"

    def record(self, message: str) -> None:
        """Add a human-readable note to the end-of-run summary."""
        self.changes.append(message)

    def block(self, message: str) -> None:
        """Mark something the wizard could not complete."""
        self.blockers.append(message)

    def halt(self, reason: str) -> None:
        """Stop the wizard before any subsequent step runs.

        Use for conditions that make every later step unsafe — chiefly an
        unresolved ``KESTREL_DATA_KEY`` custody conflict, after which a
        key-dependent step (e.g. ``payments`` encrypting operator credentials)
        would store data under a key the next boot cannot reproduce (#2468).
        The caller should also ``block(reason)`` so it surfaces in the summary.
        """
        self.halted = True
        self.halt_reason = reason
