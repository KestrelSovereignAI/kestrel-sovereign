"""Core and the SDK summarize log content with one function and one key (#3952).

``content_log_summary`` digests text with a process-random HMAC key. While Core
kept its own copy beside the SDK's (SDK 0.48), one process held two keys: the
same text got two digests, so a Core log line and an SDK-wrapper log line about
it could not be matched, and the two copies could drift. Core re-exports the
SDK's function, so there is one implementation and one key per process.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from kestrel_sdk import logging_summary
from kestrel_sdk.features.base import Feature as SdkFeature
from kestrel_sdk.tools.base import ToolCategory

from kestrel_sovereign import logging_config
from kestrel_sovereign.features.base import Feature, tool

# None of its letters is a hex digit, so no digest can spell it by chance.
MARKER = "ZqXjVw"
TEXT = f"{MARKER} the heron came back to the mill pond"


def test_core_content_log_summary_is_the_sdk_function():
    assert logging_config.content_log_summary is logging_summary.content_log_summary
    assert logging_config.CONTENT_DIGEST_HEX_CHARS == logging_summary.CONTENT_DIGEST_HEX_CHARS


def test_core_and_sdk_give_one_text_the_same_digest():
    assert logging_config.content_log_summary(TEXT) == logging_summary.content_log_summary(TEXT)
    assert logging_config.exception_log_summary(ValueError(TEXT)) == (
        f"ValueError {logging_summary.content_log_summary(TEXT)}"
    )


class _RaisingTools:
    """One failing tool body; only the wrapping base differs."""

    tool_name = "summary_probe"

    @property
    def tool_description(self) -> str:
        return "Raise an error that quotes its input."

    async def initialize(self) -> None:
        return None

    @tool(
        name="probe_raise",
        description="Raise an error quoting the text it failed on.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def probe_raise(self) -> dict:
        raise ValueError(TEXT)


class _BuiltinProbe(_RaisingTools, Feature):
    """A built-in feature: Core's ``DynamicTool`` wraps and logs the failure."""


class _SdkProbe(_RaisingTools, SdkFeature):
    """An external feature: the SDK's ``DynamicTool`` wraps and logs it."""


async def _failure_log_line(feature_class, caplog) -> str:
    feature = feature_class(SimpleNamespace(did="did:test:summary-3952"))
    [probe] = [found for found in feature.get_tools() if found.name == "probe_raise"]
    caplog.clear()

    result = await probe.execute()

    assert result["success"] is False
    [line] = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Error executing tool probe_raise")
    ]
    return line


@pytest.mark.asyncio
async def test_core_and_sdk_wrapper_log_lines_for_one_failure_correlate(caplog):
    """The same failure, logged by Core's wrapper and by the SDK's, carries
    the same digest, and neither line quotes the text."""
    caplog.set_level(logging.DEBUG)

    core_line = await _failure_log_line(_BuiltinProbe, caplog)
    sdk_line = await _failure_log_line(_SdkProbe, caplog)

    summary = logging_summary.content_log_summary(TEXT)
    assert core_line == sdk_line == f"Error executing tool probe_raise: ValueError {summary}"
    assert MARKER not in core_line
