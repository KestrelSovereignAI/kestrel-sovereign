"""Scoped callback-secret redaction without changing native request or control."""

import asyncio
import logging
from contextvars import ContextVar

from kestrel_sdk.native_a2a.types import TaskPushNotificationConfig

from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
)


class PushLogRedaction(logging.Filter):
    """Protect one callback's logs and retained exception messages.

    Each instance owns its context variable. Original HTTP/cleanup children
    inherit it; concurrent unrelated requests do not. Exception identity,
    explicit causes, native custody facts and actual request objects stay
    intact. Only human-readable strings are redacted.
    """

    def __init__(self, config: TaskPushNotificationConfig):
        super().__init__()
        self._active = ContextVar("native-push-log-redaction", default=False)
        self._values: tuple[str, ...] = ()
        self._replacement = "[redacted]"
        self._bindings: list[logging.Filterer] = []
        self._token = None
        self.protect(config.url, config.token, config.authentication.credentials)

    def protect(self, *values: str) -> None:
        """Include the validated HTTPX URL spelling before any network effect."""
        protected = set(self._values)
        for value in values:
            if value:
                protected.add(value)
                protected.add(repr(value)[1:-1])
        self._values = tuple(sorted(protected, key=len, reverse=True))
        # A legitimate short credential may occur in the marker itself.
        self._replacement = (
            "" if any(value in "[redacted]" for value in self._values) else "[redacted]"
        )

    def _text(self, value: str) -> str:
        for secret in self._values:
            value = value.replace(secret, self._replacement)
        return value

    def _error(self, error: BaseException) -> None:
        # Custody uses only explicit ancestry for authority. Logging also
        # renders implicit contexts, so include those for redaction only.
        pending, seen = [error], set()
        while pending:
            for carrier in _explicit_control_chain(pending.pop()):
                if id(carrier) in seen:
                    continue
                seen.add(id(carrier))
                carrier.args = tuple(
                    self._text(value) if isinstance(value, str) else value
                    for value in carrier.args
                )
                notes = vars(carrier).get("__notes__")
                if isinstance(notes, list):
                    notes[:] = [self._text(value) for value in notes]
                if carrier.__context__ is not None:
                    pending.append(carrier.__context__)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._active.get():
            return True
        if record.exc_info and isinstance(record.exc_info[1], BaseException):
            self._error(record.exc_info[1])
        if record.exc_text:
            record.exc_text = self._text(record.exc_text)
        rendered = record.getMessage()
        redacted = self._text(rendered)
        if redacted != rendered:
            record.msg, record.args = redacted, ()
        return True

    def __enter__(self):
        self._token = self._active.set(True)
        # HTTPX's origin filter protects handlers installed during the send.
        # Existing root handlers also cover HTTP-core diagnostics and plugins.
        bindings = [logging.getLogger("httpx"), *logging.getLogger().handlers]
        try:
            for binding in bindings:
                binding.addFilter(self)
                self._bindings.append(binding)
        except BaseException:  # restore partially registered logging ownership
            self._detach()
            self._active.reset(self._token)
            raise
        return self

    def _detach(self) -> None:
        for binding in self._bindings:
            binding.removeFilter(self)
        self._bindings.clear()

    def __exit__(self, exc_type, error, tb):
        try:
            if error is not None:
                self._error(error)
            pending = {
                owner
                for _, owner in execution_cleanup_debts(error)
                if isinstance(owner, asyncio.Future) and not owner.done()
            }
            if pending:
                # Keep the filter for the actual retained producers. These
                # callbacks observe existing owners; no new worker is created.
                def retired(_):
                    if all(owner.done() for owner in pending):
                        self._detach()

                for owner in pending:
                    owner.add_done_callback(retired)
            else:
                self._detach()
        finally:
            self._active.reset(self._token)
        return False
