"""Unit tests for kestrel_sovereign.setup.prompts."""

from __future__ import annotations

import os

import pytest

from kestrel_sovereign.setup.prompts import (
    NonInteractivePrompter,
    StubPrompter,
    is_tty,
)


def test_is_tty_false_when_ci_env_set(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert is_tty() is False


def test_is_tty_false_when_kestrel_noninteractive(monkeypatch):
    monkeypatch.setenv("KESTREL_NONINTERACTIVE", "1")
    monkeypatch.delenv("CI", raising=False)
    assert is_tty() is False


def test_noninteractive_prompter_returns_defaults():
    p = NonInteractivePrompter()
    assert p.text("Q", default="dval") == "dval"
    assert p.secret("Q", default="sval") == "sval"
    assert p.confirm("Q", default=True) is True
    assert p.confirm("Q", default=False) is False


def test_noninteractive_select_prefers_default():
    p = NonInteractivePrompter()
    assert p.select("Q", choices=["a", "b", "c"], default="b") == "b"


def test_noninteractive_select_falls_back_to_first():
    p = NonInteractivePrompter()
    assert p.select("Q", choices=["a", "b", "c"]) == "a"


def test_noninteractive_select_empty_choices():
    p = NonInteractivePrompter()
    assert p.select("Q", choices=[]) == ""


def test_stub_prompter_consumes_answers_in_order():
    p = StubPrompter(answers=["one", True, "two"])
    assert p.text("Q") == "one"
    assert p.confirm("Q") is True
    assert p.select("Q", choices=["a"]) == "two"


def test_stub_prompter_raises_when_out_of_answers():
    p = StubPrompter(answers=[])
    with pytest.raises(AssertionError, match="ran out of answers"):
        p.text("Q")


def test_stub_prompter_raises_on_type_mismatch():
    """A test that passes 'yes' for a confirm() must fail loudly."""
    p = StubPrompter(answers=["yes"])  # str, not bool
    with pytest.raises(AssertionError, match="expected bool"):
        p.confirm("Q")


def test_stub_prompter_logs_prompts():
    p = StubPrompter(answers=["a"])
    p.text("First question")
    assert any("First question" in entry for entry in p.log)
