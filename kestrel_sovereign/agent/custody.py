"""The resources an agent holds, each held until its release is confirmed (#3522).

An agent's serving record tells a guard (``kestrel update --no-restart``, an
offline reanchor) that a process may still hold the agent. Removing it is only
honest once every resource the agent acquired has been released. Teardown used
to count a resource released whenever its close returned, and several owners
return normally from a close that failed: they log the error, or abandon a
step, and carry on. Each such close released the record while the resource
was still held.

:class:`ResourceCustody` inverts that default. A resource is named here when it
is acquired, with the owner that releases it, and stays held until a release
step reports :attr:`ReleaseOutcome.RELEASED` for an owner in
:data:`TRUTHFUL_CLOSE_OWNERS`. An exception, a timeout, a ``RETAINED`` or
``UNKNOWN`` result, a step that returns anything else, a step that never runs,
and any report from an owner not in that list all leave it held. The boot
rollback and the shutdown both release through this tracker, and the agent
removes its serving record only when :attr:`ResourceCustody.held` is empty.
"""

from __future__ import annotations

import enum
from typing import Awaitable, Callable, Collection, Dict, Optional, Tuple

#: Owners whose release step reports a failed release truthfully: by raising,
#: or by returning a :class:`ReleaseOutcome` other than ``RELEASED``. Only a
#: ``RELEASED`` reported for one of these owners ends a resource's custody.
#:
#: Until every owner a boot acquires from is listed, an agent keeps its serving
#: record until its process exits. An owner joins only once its own close
#: stops swallowing failures; that it returned without raising never makes it
#: truthful. ``task_manager`` joined with #3558 and ``llm_service`` with #3559.
#: The conversion still filed is #3560 (``feature``); every other owner a
#: boot acquires (``storage``, ``signal_dispatcher``, ``memory_system``,
#: ``sync_service``, ``heartbeat_runner``, ``resume_monitor``,
#: ``salvage_worker``, ``background_tasks``) needs its close shown truthful
#: the same way first.
TRUTHFUL_CLOSE_OWNERS: frozenset[str] = frozenset({"task_manager", "llm_service"})

#: The owner of every per-feature resource.
FEATURE_OWNER = "feature"

#: Prefix of the custody name of one loaded feature.
FEATURE_RESOURCE_PREFIX = "feature:"


class ReleaseOutcome(enum.Enum):
    """What a release step reports about the resource it released."""

    #: The step released the resource. Ends custody only for a truthful owner.
    RELEASED = "released"
    #: The step kept the resource deliberately, or could not release it.
    RETAINED = "retained"
    #: Whether the resource is released cannot be told.
    UNKNOWN = "unknown"


#: A zero-argument coroutine function that releases one resource.
ReleaseStep = Callable[[], Awaitable[ReleaseOutcome]]


#: The custody name of a Hold context a standalone helper hands the agent.
STANDALONE_HOLD_CONTEXT = "standalone_hold_context"


def feature_resource_name(feature_key: str) -> str:
    """The custody name of the feature registered under ``feature_key``."""
    return f"{FEATURE_RESOURCE_PREFIX}{feature_key}"


def reported_outcome(result: object) -> ReleaseOutcome:
    """What a close that returned ``result`` without raising reports.

    A close reports a failed release by raising or by returning a
    :class:`ReleaseOutcome` other than ``RELEASED``, so each release step
    passes that value on rather than replacing it. Any other return reports
    ``RELEASED``, which custody believes only from a truthful owner.
    """
    if isinstance(result, ReleaseOutcome):
        return result
    return ReleaseOutcome.RELEASED


class ResourceCustody:
    """Which resources are held, released only on a truthful confirmation."""

    def __init__(self, truthful_owners: Optional[Collection[str]] = None) -> None:
        # None reads TRUTHFUL_CLOSE_OWNERS at each release.
        self._truthful_owners = (
            None if truthful_owners is None else frozenset(truthful_owners)
        )
        # Resource name -> owner, in the order the resources were acquired.
        self._held: Dict[str, str] = {}

    def acquire(self, name: str, owner: Optional[str] = None) -> None:
        """Hold ``name``, released by ``owner`` (default: ``name`` itself).

        A resource already held keeps the owner it was acquired with.
        """
        self._held.setdefault(name, owner if owner is not None else name)

    @property
    def held(self) -> Tuple[str, ...]:
        """The names of the resources still held, in acquisition order."""
        return tuple(self._held)

    def is_held(self, name: str) -> bool:
        return name in self._held

    def is_truthful(self, owner: str) -> bool:
        """Whether ``owner`` reports a failed release instead of hiding it."""
        owners = self._truthful_owners
        if owners is None:
            owners = TRUTHFUL_CLOSE_OWNERS
        return owner in owners

    async def release(self, name: str, step: ReleaseStep) -> ReleaseOutcome:
        """Run ``step`` to release ``name``, then :meth:`settle` what it reported.

        The step runs whether or not ``name`` is held, so a teardown also
        releases what was set up without custody.

        Raises:
            BaseException: Whatever the step raises. The resource stays held.
        """
        return self.settle(name, await step())

    def settle(self, name: str, reported: object) -> ReleaseOutcome:
        """Record what a release of ``name`` reported; end custody if confirmed.

        A ``RELEASED`` report ends custody only when the resource's owner is
        truthful; from any other owner it counts as ``UNKNOWN`` and the
        resource stays held. A report that is not a :class:`ReleaseOutcome`
        is ``UNKNOWN`` too.

        Returns:
            ``RELEASED`` only when ``name`` is not held afterwards, because the
            release was confirmed or it was never held; otherwise what the
            release counts as.
        """
        if not isinstance(reported, ReleaseOutcome):
            reported = ReleaseOutcome.UNKNOWN
        owner = self._held.get(name)
        if owner is None or reported is not ReleaseOutcome.RELEASED:
            return reported
        if not self.is_truthful(owner):
            return ReleaseOutcome.UNKNOWN
        del self._held[name]
        return ReleaseOutcome.RELEASED
