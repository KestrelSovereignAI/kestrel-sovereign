"""Exact original human negative result; no agent or ordinary SQL authority."""
from dataclasses import dataclass

from .original_action_result import _OriginalActionResultCustody


@dataclass(frozen=True, eq=False, repr=False)
class _AcceptedHumanActionResult:
    task: object
    origin: object
    identity: tuple
    payload: bytes
    stored_json: str
    binding: tuple
    private_identity: tuple
    private_payload: bytes


class _OriginalHumanActionResult(_OriginalActionResultCustody):
    """Only Core's admitted human row may own this fixed negative write."""
    def __init__(self, repository, origin, owner, session_key, payload, stored, *, admission):
        self._capture(repository, origin, owner, session_key, payload, stored, admission=admission)
