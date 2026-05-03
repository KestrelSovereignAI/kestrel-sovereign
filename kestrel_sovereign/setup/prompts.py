"""Thin wrapper over ``questionary`` with a non-TTY fallback.

The wizard must work in three environments:

  1. Interactive terminal — full questionary UX (arrow keys, masking).
  2. Non-TTY (CI, redirected stdin, ``CI=true``) — never block; use
     defaults silently and record any unset required value as a blocker.
  3. Tests — inject :class:`StubPrompter` with scripted answers.

The :class:`Prompter` protocol is the seam. Tests substitute
:class:`StubPrompter`; the CLI builds :class:`QuestionaryPrompter` for
interactive runs and :class:`NonInteractivePrompter` otherwise.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Protocol


def is_tty() -> bool:
    """True if stdin and stdout are real terminals and CI is unset."""
    if os.environ.get("CI", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("KESTREL_NONINTERACTIVE", "").lower() in ("1", "true", "yes"):
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


class Prompter(Protocol):
    """The minimum interface every wizard step uses."""

    def text(self, message: str, *, default: str = "") -> str: ...

    def secret(self, message: str, *, default: str = "") -> str: ...

    def confirm(self, message: str, *, default: bool = True) -> bool: ...

    def select(
        self,
        message: str,
        *,
        choices: Iterable[str],
        default: str | None = None,
    ) -> str: ...

    def info(self, message: str) -> None: ...


@dataclass
class QuestionaryPrompter:
    """Real questionary-backed prompter for interactive terminals."""

    def text(self, message: str, *, default: str = "") -> str:
        import questionary

        return questionary.text(message, default=default).unsafe_ask()

    def secret(self, message: str, *, default: str = "") -> str:
        import questionary

        # questionary.password masks input; default is shown as a hint
        # in the prompt label since password() can't accept a default.
        label = f"{message} [press enter to keep current]" if default else message
        answer = questionary.password(label).unsafe_ask()
        return answer if answer else default

    def confirm(self, message: str, *, default: bool = True) -> bool:
        import questionary

        return questionary.confirm(message, default=default).unsafe_ask()

    def select(
        self,
        message: str,
        *,
        choices: Iterable[str],
        default: str | None = None,
    ) -> str:
        import questionary

        return questionary.select(
            message, choices=list(choices), default=default
        ).unsafe_ask()

    def info(self, message: str) -> None:
        print(message)


@dataclass
class NonInteractivePrompter:
    """Used when there is no TTY. Returns defaults; never blocks."""

    def text(self, message: str, *, default: str = "") -> str:
        return default

    def secret(self, message: str, *, default: str = "") -> str:
        return default

    def confirm(self, message: str, *, default: bool = True) -> bool:
        return default

    def select(
        self,
        message: str,
        *,
        choices: Iterable[str],
        default: str | None = None,
    ) -> str:
        choices_list = list(choices)
        if default is not None and default in choices_list:
            return default
        return choices_list[0] if choices_list else ""

    def info(self, message: str) -> None:
        print(message)


@dataclass
class StubPrompter:
    """Test double: scripted answers consumed in order.

    ``answers`` is a list of typed values. Each prompt method pops the
    next one; mismatched type raises so a wrong test fails loudly rather
    than silently returning the default.
    """

    answers: list[object] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def _next(self, expected_type: type) -> object:
        if not self.answers:
            raise AssertionError("StubPrompter ran out of answers")
        value = self.answers.pop(0)
        if not isinstance(value, expected_type):
            raise AssertionError(
                f"StubPrompter expected {expected_type.__name__}, "
                f"got {type(value).__name__}: {value!r}"
            )
        return value

    def text(self, message: str, *, default: str = "") -> str:
        self.log.append(f"text: {message}")
        return self._next(str)  # type: ignore[return-value]

    def secret(self, message: str, *, default: str = "") -> str:
        self.log.append(f"secret: {message}")
        return self._next(str)  # type: ignore[return-value]

    def confirm(self, message: str, *, default: bool = True) -> bool:
        self.log.append(f"confirm: {message}")
        return self._next(bool)  # type: ignore[return-value]

    def select(
        self,
        message: str,
        *,
        choices: Iterable[str],
        default: str | None = None,
    ) -> str:
        self.log.append(f"select: {message}")
        return self._next(str)  # type: ignore[return-value]

    def info(self, message: str) -> None:
        self.log.append(f"info: {message}")
