"""Prompt-template override contract adapters.

The canonical signal dataclasses live in ``kestrel_sdk.signals``. Until
the SDK grows first-class constructor fields for #1146, sovereign code
uses these subclasses for sources that need per-signal COGNITION prompt
templates. They remain normal ``Signal`` / ``SourceRegistration``
instances for the dispatcher and registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kestrel_sdk.signals import Signal, SourceRegistration


@dataclass
class SignalWithPromptTemplateOverride(Signal):
    """Signal envelope that can carry a per-dispatch prompt template."""

    prompt_template_override: Optional[Path] = None


@dataclass
class SourceRegistrationWithPromptOverride(SourceRegistration):
    """Source registration that opts into per-signal prompt templates."""

    allow_prompt_override: bool = False


__all__ = [
    "SignalWithPromptTemplateOverride",
    "SourceRegistrationWithPromptOverride",
]
