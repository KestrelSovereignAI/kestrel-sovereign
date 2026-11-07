"""One error boundary for durable encrypted repository initialization."""

from contextlib import contextmanager

from kestrel_sdk.resources import ResourceError, ResourceOutcomeUnknown
from kestrel_sdk.storage.database.interface import TransactionError
from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error


@contextmanager
def repository_initialization_errors(*, integrity_error: type[ResourceError], unknown_message: str):
    """Expose an original key failure without erasing native transaction custody."""
    try:
        yield
    except TransactionError as error:
        if execution_commit_outcome(error) is not None:
            raise ResourceOutcomeUnknown(unknown_message) from error
        if isinstance(error.__cause__, integrity_error) and not is_execution_control_error(error):
            raise error.__cause__ from None
        raise
