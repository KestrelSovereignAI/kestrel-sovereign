"""#4207: :class:`ProviderAttemptReceipt`, one provider attempt's evidence.

The adapters' use of it, over the real ``openai`` SDK, is covered in
``test_openai_responses_adapter.py`` and, in booted turns,
``test_openai_responses_continuation_turn.py``.
"""

from __future__ import annotations

import asyncio

from kestrel_sovereign.execution_custody import (
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.output_ceiling import attach_reported_usage, reported_usage
from kestrel_sovereign.llm.provider_served import (
    ProviderAttemptReceipt,
    ProviderServed,
    attach_provider_served,
    provider_served,
)

SERVED = ProviderServed(api="responses", model="gpt-6.2-sol-2026-11-01", response_id="resp_1")


def _counts(usage):
    return None if usage is None else (usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens)


def test_an_empty_receipt_records_nothing_on_the_error():
    """Nothing reported is nothing recorded: no identity, no usage, never
    a zero, and the error is returned as itself."""
    error = RuntimeError("before the provider answered")
    receipt = ProviderAttemptReceipt()
    assert (receipt.served, receipt.usage, receipt.reported()) == (None, None, None)
    assert receipt.retain(error) is error
    assert provider_served(error) is None and reported_usage(error) is None


def test_a_reported_zero_stays_zero_and_a_missing_report_never_erases_it():
    receipt = ProviderAttemptReceipt()
    receipt.report_usage(LLMResponse(input_tokens=0, output_tokens=0, total_tokens=0))
    receipt.report_usage(None)
    error = receipt.retain(ValueError("after a zero usage report"))
    assert _counts(reported_usage(error)) == (0, 0, None)


def test_the_latest_identification_is_recorded_with_the_usage():
    receipt = ProviderAttemptReceipt()
    receipt.identify(ProviderServed(api="responses", response_id="resp_1"))
    receipt.identify(SERVED)
    receipt.report_usage(LLMResponse(input_tokens=100, output_tokens=30, cache_read_input_tokens=20))
    assert provider_served(receipt.reported()) == SERVED
    error = receipt.retain(asyncio.CancelledError())
    assert provider_served(error) == SERVED
    assert _counts(reported_usage(error)) == (100, 30, 20)
    assert provider_served(reported_usage(error)) == SERVED


def test_retain_leaves_a_control_group_whole():
    """An execution-control exception group keeps its identity, members,
    cause, context and native debt; only the receipt is added to it."""
    member = OSError("closing the stream failed")
    debt = RuntimeError("original native stream retirement failed")
    resource, owner = object(), object()
    debt._kestrel_native_retirement_debts = ((resource, owner),)
    cause = KeyError("cause")
    context = ValueError("context")
    group = BaseExceptionGroup("original execution terminal evidence", [member, debt])
    group.__cause__ = cause
    group.__context__ = context
    receipt = ProviderAttemptReceipt()
    receipt.identify(SERVED)
    receipt.report_usage(LLMResponse(input_tokens=100, output_tokens=30, cache_read_input_tokens=20))

    assert receipt.retain(group) is group
    assert len(group.exceptions) == 2 and all(now is then for now, then in zip(group.exceptions, (member, debt)))
    assert group.__cause__ is cause and group.__context__ is context
    assert is_execution_control_error(group)
    assert execution_cleanup_debts(group) == ((resource, owner),)
    assert provider_served(group) == SERVED
    assert _counts(reported_usage(group)) == (100, 30, 20)
    # Members are not decorated: the evidence is on the group that escapes.
    assert provider_served(member) is None and reported_usage(member) is None


def test_the_caller_reads_the_attempt_it_ran_before_anything_wrapped():
    """An error raised again by a later attempt (a shared carrier) reads as
    that attempt, and the attempt's own receipt wins over evidence on an
    error it wraps."""
    earlier = ProviderAttemptReceipt()
    earlier.identify(ProviderServed(api="responses", response_id="resp_earlier"))
    earlier.report_usage(LLMResponse(input_tokens=7, output_tokens=3))
    wrapped = ValueError("wrapped")
    attach_provider_served(wrapped, ProviderServed(api="responses", response_id="resp_wrapped"))
    attach_reported_usage(wrapped, LLMResponse(input_tokens=1, output_tokens=1))
    shared = RuntimeError("raised by more than one attempt")
    shared.__cause__ = wrapped
    earlier.retain(shared)

    later = ProviderAttemptReceipt()
    later.identify(SERVED)
    later.report_usage(LLMResponse(input_tokens=100, output_tokens=30, cache_read_input_tokens=20))
    later.retain(shared)
    assert provider_served(shared) == SERVED
    assert _counts(reported_usage(shared)) == (100, 30, 20)
    # What the wrapped error carries is left as it is.
    assert provider_served(wrapped).response_id == "resp_wrapped"


def test_receipts_share_nothing():
    first, second = ProviderAttemptReceipt(), ProviderAttemptReceipt()
    first.identify(SERVED)
    first.report_usage(LLMResponse(input_tokens=1))
    assert (second.served, second.usage) == (None, None)
