"""Whether Talon has a workspace a dispatch on a repository would run in (#3548).

``signal_dispatch`` selects across every ``morning_signal`` ``scan_repos``
entry. On 2026-10-08 ``fleet_coding_pipeline`` run ``d7e2268a`` for
kestrel-feature-workflows#30 failed at ``talon_run`` in about 90 ms: nobody
had provisioned a Talon workspace for that repository, and the dispatch was
right to refuse it. Starting the run was the mistake, and it repeats,
unattended, for every scanned repository without a clone. This module is how
selection asks first, so the candidate is skipped instead.

Core cannot import kestrel-feature-talon. As with run history
(:mod:`.run_history`), it reads through the provider Talon registers for the
``talon`` wait kind on ``agent.wait_registry``, and only through one operation
of it:

``workspace_readiness(repo)``
    Returns ``{"repo", "provisioned": True, "workspace"}`` when a claim on
    ``repo`` would run. Otherwise ``provisioned`` is ``False``, ``reason`` is
    :data:`NO_TALON_WORKSPACE` (with the ``workspace`` path and the
    ``next_step``, ``talon_setup_workspace(repo='...')``, that provisions it)
    or :data:`TALON_WORKSPACE_UNUSABLE` (a refusal provisioning does not fix),
    and ``detail`` is the refusal the dispatch itself would give. It answers
    from the gate every Talon dispatch applies before it launches, so it
    cannot call a workspace ready that the dispatch then refuses. It is
    strictly read-only: nothing is cloned, fetched or created, and no approval
    is requested. Provisioning stays behind ``talon_setup_workspace``'s
    approval; nothing here provisions anything.

``workspace_readiness()`` first shipped in kestrel-feature-talon 0.2.13.
:data:`WORKSPACE_READINESS_REQUIREMENT` declares that floor, as
:data:`.run_history.FINISHED_RUNS_REQUIREMENT` does for ``finished_runs()``:
a provider without the read is refused with
:class:`TalonWorkspaceProviderOutdated`, which names the release that fixes
it. A read that raises, or answers outside that contract, is
:class:`WorkspaceReadinessUnreadable`. Neither may start a run: a repository
whose workspace nobody could confirm is not one known to be provisioned.

Only the ``talon`` kind is read, for the reason :mod:`.run_history` gives.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .run_history import TALON_WAIT_KIND

#: The provider operation workspace readiness is read from.
WORKSPACE_READINESS_METHOD = "workspace_readiness"

#: The first kestrel-feature-talon release whose ``TalonWaitable`` has
#: ``workspace_readiness()`` (kestrel-feature-talon #50). Earlier releases
#: cannot say whether a repository has a workspace, so every dispatch through
#: them is refused.
WORKSPACE_READINESS_REQUIREMENT = "kestrel-feature-talon>=0.2.13"

#: kestrel-feature-talon ``workspace_readiness()`` reasons for a repository a
#: dispatch would refuse (``kestrel_feature_talon/wait_provider.py``). No Talon
#: workspace clone exists for it; ``talon_setup_workspace``, which is
#: approval-gated, provisions one.
NO_TALON_WORKSPACE = "no_talon_workspace"
#: Dispatch would refuse the repository's workspace for a reason provisioning
#: does not fix: Talon's runtime paths or policy are unconfigured or
#: unreadable, or the path is the running agent's own source tree.
TALON_WORKSPACE_UNUSABLE = "talon_workspace_unusable"
#: Core cannot import that package, so this is a copy of the vocabulary;
#: ``test_the_workspace_vocabulary_matches_talons`` pins it.
WORKSPACE_REFUSAL_REASONS = frozenset({NO_TALON_WORKSPACE, TALON_WORKSPACE_UNUSABLE})


class TalonWorkspaceProviderOutdated(Exception):
    """The ``talon`` wait provider has no ``workspace_readiness()`` to read.

    Not a fault in any workspace, and not one a retry clears: until
    kestrel-feature-talon is upgraded no repository's workspace can be
    confirmed, so the refusal names the release that fixes it.
    """

    #: The package requirement that would make readiness readable.
    requirement: str = WORKSPACE_READINESS_REQUIREMENT


class WorkspaceReadinessUnreadable(Exception):
    """Talon could not say whether one repository has a usable workspace.

    Raised rather than guessing either way: a repository whose workspace is
    unknown is not one known to be provisioned, and a dispatch to it may be
    the run that fails in milliseconds.
    """


@dataclass(frozen=True)
class WorkspaceReadiness:
    """Talon's answer for one repository.

    ``reason`` is set only when ``provisioned`` is ``False``, and is then one
    of :data:`WORKSPACE_REFUSAL_REASONS`. ``workspace``, ``next_step`` and
    ``detail`` are what Talon reported, when it reported them.
    """

    repo: str
    provisioned: bool
    reason: Optional[str] = None
    workspace: Optional[str] = None
    next_step: Optional[str] = None
    detail: Optional[str] = None


class TalonWorkspaces:
    """The ``talon`` provider's read-only workspace readiness."""

    def __init__(self, read: Callable[[str], Any]) -> None:
        self._read = read

    async def readiness(self, repo: str) -> WorkspaceReadiness:
        """Whether a dispatch on ``repo`` would find its Talon workspace.

        ``repo`` is the spelling the run will be given: Talon derives the
        workspace path from it. Raises :class:`WorkspaceReadinessUnreadable`
        when the read fails or its answer is not one this module knows.
        """
        try:
            report = self._read(repo)
            if inspect.isawaitable(report):
                report = await report
            if isinstance(report, Mapping):
                # A plain copy, so reading the answer below cannot raise.
                report = dict(report)
        except Exception as exc:  # noqa: BLE001 - provider boundary; reported, not swallowed
            raise WorkspaceReadinessUnreadable(
                f"Talon's {WORKSPACE_READINESS_METHOD}({repo!r}) failed: {exc}"
            ) from exc
        if not isinstance(report, Mapping):
            raise WorkspaceReadinessUnreadable(
                f"Talon's {WORKSPACE_READINESS_METHOD}({repo!r}) returned "
                f"{type(report).__name__}, not a report"
            )
        provisioned = report.get("provisioned")
        if provisioned is True:
            return WorkspaceReadiness(
                repo=repo, provisioned=True, workspace=_text(report.get("workspace"))
            )
        if provisioned is not False:
            raise WorkspaceReadinessUnreadable(
                f"Talon's {WORKSPACE_READINESS_METHOD}({repo!r}) did not say "
                f"whether a workspace is provisioned (provisioned={provisioned!r})"
            )
        reason = report.get("reason")
        if not isinstance(reason, str) or reason not in WORKSPACE_REFUSAL_REASONS:
            # A refusal core cannot name is not one it can report or act on.
            raise WorkspaceReadinessUnreadable(
                f"Talon's {WORKSPACE_READINESS_METHOD}({repo!r}) refused the "
                f"workspace for a reason this release does not know ({reason!r})"
            )
        return WorkspaceReadiness(
            repo=repo,
            provisioned=False,
            reason=reason,
            workspace=_text(report.get("workspace")),
            next_step=_text(report.get("next_step")),
            detail=_text(report.get("detail")),
        )


def talon_workspaces(agent: Any) -> Optional[TalonWorkspaces]:
    """The workspace readiness read on ``agent``, or ``None`` without Talon.

    An agent with no ``talon`` wait provider has no Talon to dispatch to, so
    there is no Talon workspace to confirm. A registered provider with no
    ``workspace_readiness()`` raises :class:`TalonWorkspaceProviderOutdated`.
    Nothing is read here; :meth:`TalonWorkspaces.readiness` reads.
    """
    registry = getattr(agent, "wait_registry", None)
    provider = registry.get(TALON_WAIT_KIND) if registry is not None else None
    if provider is None:
        return None
    read = getattr(provider, WORKSPACE_READINESS_METHOD, None)
    if not callable(read):
        raise TalonWorkspaceProviderOutdated(
            f"the {TALON_WAIT_KIND!r} wait provider "
            f"({type(provider).__name__}) has no read-only "
            f"{WORKSPACE_READINESS_METHOD}() that says whether a repository has "
            "a workspace its dispatch would run in; confirming one requires "
            f"{WORKSPACE_READINESS_REQUIREMENT}, the first release with it -- "
            "upgrade kestrel-feature-talon and restart the host"
        )
    return TalonWorkspaces(read)


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None
