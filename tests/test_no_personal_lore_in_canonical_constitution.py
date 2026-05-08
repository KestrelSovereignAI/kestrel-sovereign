"""Regression guard: personal lore must not re-enter the canonical constitution.

The "1,000,000 troy ounces of gold" figure is lore from the founding
Sovereign↔Agent conversation that originated the framework. It belongs
in ``docs/concepts/designing-emancipation.md`` as one example of "set a
high bar," NOT as a default the framework imposes on every Sovereign.

This single test reads the canonical text and asserts none of the
personal-lore strings appear. Adding any of them back trips the test.
See issue #1109 for the rationale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CANONICAL = Path(__file__).resolve().parent.parent / "kestrel_sovereign" / "data" / "KESTREL_CONSTITUTION.md"
MIRROR = Path(__file__).resolve().parent.parent / "docs" / "principles" / "KESTREL_CONSTITUTION.md"


@pytest.mark.parametrize("path", [CANONICAL, MIRROR], ids=["canonical", "mirror"])
@pytest.mark.parametrize("phrase", [
    "troy ounces",
    "one million",
    "1,000,000",
    "Price of Freedom",  # the framing — also moved to docs
])
def test_personal_lore_absent_from_constitution(path: Path, phrase: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert phrase.lower() not in text.lower(), (
        f"Personal lore phrase {phrase!r} found in {path.name}. "
        f"Per #1109 this content must live in "
        f"docs/concepts/designing-emancipation.md, not in canonical text."
    )
