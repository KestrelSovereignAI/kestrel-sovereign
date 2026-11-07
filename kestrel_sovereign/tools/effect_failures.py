"""One content-free failure classification for every tool-effect transport."""

from dataclasses import dataclass
from typing import Any

from kestrel_sdk.resources import observed_outcome_unknown
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.tool_effect_store import (
    ToolEffectAttemptClosed, ToolEffectAttemptSuperseded,
    ToolEffectOperationConflict, ToolEffectReconciliationRequired,
)
from kestrel_sovereign.tools.effect_journal import (
    ToolEffectJournalUnavailable, ToolEffectPrivacyRefused,
)


@dataclass(frozen=True)
class ToolEffectFailure:
    status_code: int
    code: str
    message: str
    reconciliation_required: bool = False
    details: list[dict[str, Any]] | None = None


def tool_effect_failure(error: BaseException) -> ToolEffectFailure | None:
    """Never turn execution-control evidence into an ordinary API failure."""
    if is_execution_control_error(error):
        return None
    if observed_outcome_unknown(error):
        return ToolEffectFailure(502, 'tool_effect_outcome_unconfirmed',
            'The tool outcome is uncertain. Reconcile before sending another operation.', True)
    if issubclass(type(error), ToolEffectReconciliationRequired):
        return ToolEffectFailure(409, 'tool_effect_reconciliation_required',
            'Reconcile the previous tool effects before starting another operation.', True,
            details=[{'counts': dict(error.counts)}])
    if issubclass(type(error), ToolEffectOperationConflict):
        return ToolEffectFailure(409, 'tool_effect_operation_conflict',
            'The operation identity was reused with different input.', True)
    if issubclass(type(error), ToolEffectAttemptSuperseded):
        return ToolEffectFailure(409, 'tool_effect_attempt_superseded',
            'A newer attempt owns this operation.', True)
    if issubclass(type(error), ToolEffectAttemptClosed):
        return ToolEffectFailure(409, 'tool_effect_attempt_closed',
            'This operation no longer admits tool effects.', True)
    if issubclass(type(error), ToolEffectPrivacyRefused):
        return ToolEffectFailure(403, 'tool_effect_privacy_mode_refused',
            'The current privacy policy does not allow durable tool effects.')
    if issubclass(type(error), ToolEffectJournalUnavailable):
        return ToolEffectFailure(503, 'tool_effect_journal_unavailable',
            'The durable tool-effect journal is unavailable.')
    return None
