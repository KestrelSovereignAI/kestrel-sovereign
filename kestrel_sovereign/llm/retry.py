"""
Retry Utilities for LLM Adapters

Provides exponential backoff with jitter for handling transient errors
like rate limiting (429) and server errors (5xx).
"""
import asyncio
import logging
import math
import random
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Iterable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Retry configuration.
#
# The DEFAULT budget is tight (5 attempts, 60s cap): a transient infra error
# (503 while an ollama model loads, a 5xx, a timeout) on ANY route should fail
# over to the next route quickly — waiting minutes buys nothing when a healthy
# fallback exists. with_retry is the default for EVERY adapter call site
# (ollama/openai-compat/vertex/anthropic), so a patient default would stall
# failover on all of them (that was the #2074 regression).
MAX_RETRIES = 5
BASE_DELAY = 1.0      # seconds
MAX_DELAY = 60.0      # seconds

# THROTTLE budget — applied only to a 429 / rate-limit throttle. A 429 on a
# subscription plan route (Claude Max / ChatGPT plan) is "you're going too fast
# — wait", NOT "this route is dead". Riding out a normal rate-limit window on
# the plan route (patient: 8 attempts, 2-minute cap) avoids silently downgrading
# to a metered :api route. Honoring the server's ``Retry-After`` keeps the waits
# accurate. Scoped to throttles by error type so it does NOT slow non-throttle
# failover on local/metered routes.
THROTTLE_MAX_RETRIES = 8
THROTTLE_MAX_DELAY = 120.0

# The furthest ahead a server-advised cool-down is believed. A provider that
# puts an epoch timestamp or a stray exponent in ``Retry-After`` must not
# produce a reset time in 2082 or an OverflowError in date arithmetic: the
# advised seconds are kept as reported, but the reset time is computed from
# ``min(advised, horizon)`` and the surfaces say "at least until". A week is
# beyond any published plan-limit window (daily and weekly resets).
ADVISED_WAIT_HORIZON_SECONDS = 7 * 24 * 60 * 60

# PLAN-LIMIT budget — deliberately much tighter than the throttle budget.
#
# The throttle budget sleeps 127-134s across its 8 attempts. The orchestrator
# wraps the whole provider call in `asyncio.timeout(ORCHESTRATOR_TURN_TIMEOUT_SECS)`
# = 180s (orchestrator_engine.py:94,3081,3093), and that watchdog is NOT lifted
# for a cloud route (`effective_request_timeout` returns None unless EVERY
# candidate is local, service.py:2158). Spending 131s of a 180s turn on sleep
# leaves under 50s for the HTTP round-trips AND the streamed generation, so a
# window that cleared on attempt 7 or 8 — exactly the case patience exists for —
# would be killed by the watchdog and reported as `timeout after 180s`. That is
# worse than the hard failure this retry replaces: it converts a diagnosable
# error into a hang marker, and it defeats the fix precisely when it works.
#
# So this budget is sized to be a small fraction of the watchdog: 5 attempts,
# sleeps 1+2+4+8 = 15s (+ up to 4s jitter). It covers a brief blip and nothing
# more.
#
# It does NOT cover the ~3-minute window measured on 2026-09-04 (16:31 open,
# closed by 16:34), and it deliberately does not try: no retry budget can ride
# out a 180s window from inside a 180s watchdog. Widening the watchdog to suit
# is a turn-latency decision that belongs to the orchestrator, not to this
# module. When the budget is exhausted the provider's own message ("You're out
# of extra usage...") propagates, which names the real cause.
#
# The tight bound is also what keeps a GENUINELY exhausted account from
# stalling: this message is not always spurious — with credits enabled and
# drained, or a real cap reached, Anthropic returns the same 400 for hours.
# At 8 attempts / 131s every call would burn two minutes forever, which is the
# "retrying a dead key burns wall-time" pattern this module exists to avoid.
PLAN_LIMIT_MAX_RETRIES = 5
PLAN_LIMIT_MAX_DELAY = 16.0

# HTTP status codes that warrant retry. 401/403/404 are permanent and MUST
# NOT appear here — retrying a dead API key burns wall-time on an error that
# will never recover.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# HTTP status codes that MUST NOT retry (caller error or auth failure).
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}

# Error message patterns that indicate retryable (transient) errors.
RETRYABLE_PATTERNS = [
    "rate",
    "timeout",
    "resource_exhausted",
    "unavailable",
    "internal",
    "overloaded",
    "capacity",
    "try again",
]

# Subscription/plan-route limit reported with a NON-429 status.
#
# Anthropic's subscription endpoint reports a transient plan-limit window as
# ``400 invalid_request_error`` with billing wording ("You're out of extra
# usage..."), NOT as the ``429 rate_limit_error`` that #2074 taught the retry
# layer to ride out. Both are the same condition — a window that clears on its
# own — so both must get the patient throttle budget.
#
# Measured 2026-09-04: the 400 window lasted ~3 minutes and cleared with no
# intervention, while the account's plan usage sat at 3% (session) / 4%
# (weekly) and its usage credits were switched OFF with a $0.00 balance. So
# the message names a billing surface the request never actually reached; the
# status code is the only thing that differs from the 429 form. Without this,
# NON_RETRYABLE_STATUS_CODES short-circuits a 400 to zero retries and the
# window surfaces to the operator as a hard route failure.
#
# Matched against the message BEFORE the status-code short-circuit, and kept
# narrow on purpose: a genuine malformed-request 400 must stay permanent.
PLAN_LIMIT_PATTERNS = [
    "out of extra usage",
]


# Error message patterns that indicate permanent failure. Matched FIRST —
# takes precedence over RETRYABLE_PATTERNS to prevent a 401 message that
# happens to contain "internal server error text" from being retried.
NON_RETRYABLE_PATTERNS = [
    "user not found",          # OpenRouter dead-key / anti-abuse
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "unauthorized",
    "permission_denied",
    "permission denied",
    "quota exceeded",           # Distinct from rate-limit: permanent until refill
    "model_not_found",
    "model not found",
    "insufficient_quota",
]

T = TypeVar('T')


class AdvisedWaitExceedsRetryBudget(Exception):
    """The server said how long until a request can succeed, and it is longer
    than this retry loop is willing to wait.

    Raised by :func:`with_retry` instead of sleeping, so a turn does not spend
    its whole throttle budget on attempts that cannot succeed (a 429 advising
    a 13-hour wait was retried 8 times at 120 s each, 112 times out of 130
    advised waits on one host; #3127). The wait is fact, not guess: retrying
    before ``retry_at`` is futile by the server's own account.

    It classifies as the error it stands for at every downstream door:
    ``status_code`` is the cause's (429 for a throttle; a 503 that advised a
    cool-down stays a 503, so an overload is never reported as a quota
    problem), ``retry_after`` is the advised wait, ``response`` is the
    provider's, and the message names the reset time. It is not itself
    retryable: :func:`is_retryable_error` refuses it, so an outer loop cannot
    re-enter the wait the inner one declined.
    """

    def __init__(
        self,
        cause: Exception,
        *,
        advised_seconds: float,
        budget_seconds: float,
        retry_at: datetime,
    ) -> None:
        self.throttled = _is_throttle_error(cause)
        cause_status = getattr(cause, "status_code", None)
        self.status_code = cause_status if isinstance(cause_status, int) else (429 if self.throttled else 503)
        self.advised_seconds = float(advised_seconds)
        self.budget_seconds = float(budget_seconds)
        #: When a retry can succeed, computed from the advice clamped to
        #: ``ADVISED_WAIT_HORIZON_SECONDS``; ``beyond_horizon`` says the
        #: advice exceeded it, so this is a floor, not the exact time.
        self.retry_at = retry_at
        self.beyond_horizon = self.advised_seconds > ADVISED_WAIT_HORIZON_SECONDS
        self.retry_after = self.advised_seconds
        self.response = getattr(cause, "response", None)
        what = "rate limit" if self.throttled else f"{type(cause).__name__}"
        super().__init__(
            f"{self.status_code} {what}: the provider advised waiting "
            f"{self.advised_seconds:.0f}s ({self.reset_phrase()}), more than the "
            f"{self.budget_seconds:.0f}s this call could still wait; not retrying"
        )
        self.__cause__ = cause

    def reset_phrase(self) -> str:
        """``until <iso>``, or the floor when the advice exceeded the horizon.

        The only value either surface interpolates: a time derived from the
        provider's number, never its prose.
        """
        stamp = self.retry_at.isoformat(timespec="seconds")
        if self.beyond_horizon:
            days = ADVISED_WAIT_HORIZON_SECONDS // (24 * 60 * 60)
            return f"for more than {days} days, until at least {stamp}"
        return f"until {stamp}"

    def retry_after_header_seconds(self, now: Optional[datetime] = None) -> int:
        """Whole seconds for ``Retry-After`` as of ``now``: what remains until
        ``retry_at``, rounded up, at least one.

        The decline is raised on the first route that declines and may
        surface only after the remaining routes spent their own budgets, so
        the header is measured when the response is built, not at decline
        time; it agrees with the reset time in the message.
        """
        now = now or datetime.now(UTC)
        remaining = (self.retry_at - now).total_seconds()
        return max(1, math.ceil(min(remaining, float(ADVISED_WAIT_HORIZON_SECONDS))))


def advised_wait_exceeding_budget(error: BaseException) -> Optional[AdvisedWaitExceedsRetryBudget]:
    """The :class:`AdvisedWaitExceedsRetryBudget` ``error`` explicitly wraps, if any.

    Only explicit links are followed: ``__cause__`` (``raise X from e``), an
    ``LLMProviderError.original_error`` and an ``LLMStreamingError.underlying``;
    an error carrying a ``declined_wait`` attribute is an aggregate whose
    verdict is final, whatever its links hold.
    Implicit ``__context__`` is not: a chain severed with ``from None`` stays
    severed, and an unrelated exception raised while a decline was being
    handled (a cleanup failure, a fallback that itself broke) is not rendered
    as a rate limit it has nothing to do with. The service's wrappers link
    explicitly for that reason.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, AdvisedWaitExceedsRetryBudget):
            return current
        if hasattr(current, "declined_wait"):
            # An aggregate of several routes' errors states its own verdict
            # (``common_declined_wait``); its links are the routes' errors and
            # must not be read past that verdict, or the last route's decline
            # would surface through ``underlying``/``__cause__`` when another
            # route failed for a reason a retry could clear at once.
            verdict = current.declined_wait
            return verdict if isinstance(verdict, AdvisedWaitExceedsRetryBudget) else None
        for link in (
            current.__cause__,
            getattr(current, "original_error", None),
            getattr(current, "underlying", None),
        ):
            if isinstance(link, BaseException):
                pending.append(link)
    return None


def common_declined_wait(
    errors: Iterable[BaseException],
) -> Optional[AdvisedWaitExceedsRetryBudget]:
    """The decline an aggregate of several routes' errors may carry, if any.

    Only when EVERY route declined an advised wait is "come back at" honest
    advice for the whole call; then the soonest reset is the earliest any
    retry could succeed. If some route failed for another reason (a reset
    connection, a 5xx without advice) a retry may succeed there at once, so
    the aggregate carries no reset time.
    """
    errors = list(errors)
    if not errors:
        return None
    earliest: Optional[AdvisedWaitExceedsRetryBudget] = None
    for error in errors:
        declined = advised_wait_exceeding_budget(error)
        if declined is None:
            return None
        if earliest is None or declined.retry_at < earliest.retry_at:
            earliest = declined
    return earliest


def retry_after_seconds(error: Exception) -> Optional[float]:
    """Extract the server-advised cool-down from a provider exception.

    Honors the actual ``Retry-After`` (or ``retry-after-ms``) response header
    the SDK exception carries, so we wait the amount the *server* asked for
    instead of guessing with exponential backoff. Returns ``None`` when no
    usable value is present (caller falls back to exponential backoff).
    """
    # SDK exceptions (anthropic/openai) carry the httpx response.
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            getter = headers.get
        except AttributeError:
            getter = None
        if getter is not None:
            ms = getter("retry-after-ms")
            if ms:
                try:
                    wait = _finite_wait(float(ms) / 1000.0)
                except (TypeError, ValueError):
                    wait = None
                if wait is not None:
                    return wait
            secs = getter("retry-after")
            if secs:
                try:
                    # Numeric seconds. (An HTTP-date form is rare here and not
                    # worth parsing — exponential backoff covers that case.)
                    wait = _finite_wait(float(secs))
                except (TypeError, ValueError):
                    wait = None
                if wait is not None:
                    return wait
    # Some SDKs surface a parsed attribute directly. A source that is absent
    # or unusable (``nan``, ``inf``, prose) never hides the next one.
    attr = getattr(error, "retry_after", None)
    if isinstance(attr, (int, float)) and not isinstance(attr, bool):
        return _finite_wait(float(attr))
    return None


def _finite_wait(value: float) -> Optional[float]:
    """A usable advised wait, or ``None`` for ``inf``/``nan`` (no advice)."""
    if not math.isfinite(value):
        return None
    return max(0.0, value)


def is_plan_limit_error(error: Exception) -> bool:
    """True when ``error`` is a subscription plan-limit window reported with a
    non-429 status (see ``PLAN_LIMIT_PATTERNS``).

    Classified on the message because the status code is precisely what makes
    this shape indistinguishable from a caller error.
    """
    # Scoped to 400 on purpose. The wording alone is not enough: a dead key
    # or a missing model whose message happens to carry it ("401 Invalid API
    # key. You're out of extra usage.") would otherwise be handed a retry
    # budget, and `with_retry` is shared by every adapter, not just Anthropic.
    # Verified before scoping: 401/403/404 carrying the phrase all classified
    # retryable. Reachability was low, but the narrowness the comment above
    # claims has to be in the code, not only in the intent.
    status_code = getattr(error, "status_code", None)
    error_str = str(error).lower()
    if isinstance(status_code, int):
        if status_code != 400:
            return False
    elif not _mentions_status_400(error_str):
        return False
    return any(pattern in error_str for pattern in PLAN_LIMIT_PATTERNS)


def _mentions_status_400(error_str: str) -> bool:
    """Whether an unstructured error text names HTTP 400, with the same
    word-boundary care ``is_retryable_error`` uses for status tokens."""
    return (" 400" in error_str or "400 " in error_str or "400:" in error_str
            or "400," in error_str or "400-" in error_str
            or "code: 400" in error_str or error_str.startswith("400"))


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is transient and should be retried.

    Order of checks:
      0. PLAN_LIMIT_PATTERNS (plan window wearing a non-429 status) → retry.
      1. NON_RETRYABLE_STATUS_CODES (401/403/404/422/400) → no retry.
      2. NON_RETRYABLE_PATTERNS (auth/quota/invalid) → no retry.
      3. RETRYABLE_STATUS_CODES (429/5xx) → retry.
      4. RETRYABLE_PATTERNS (rate/timeout/etc) → retry.
      5. Default → no retry.

    Permanent-failure checks win first so a 401 error message that happens
    to contain the word "internal" in a nested detail doesn't trigger a
    misguided retry loop.

    The provider's *structured* status code (``error.status_code`` on the
    Anthropic/OpenAI SDK exceptions) is consulted before any string matching,
    so a real ``RateLimitError`` is recognized as a 429 even when its message
    text doesn't literally contain the number — we classify on the actual
    error, not a hopeful substring.
    """
    # 0a. Plan-limit window wearing a non-429 status. Checked before the
    #     status-code short-circuit below, which would otherwise classify the
    #     400 form as a permanent caller error and skip retry entirely.
    if is_plan_limit_error(error):
        return True

    # 0b. Structured status code from the SDK exception (most authoritative).
    # The retry loop's own verdict: it has already declined to wait this out.
    if isinstance(error, AdvisedWaitExceedsRetryBudget):
        return False

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return False
        if status_code in RETRYABLE_STATUS_CODES:
            return True

    error_str = str(error).lower()

    # 1. Hard non-retryable status codes — word-boundary check to avoid
    #    matching "429" inside a 64-character token or hash.
    for code in NON_RETRYABLE_STATUS_CODES:
        token = str(code)
        # Match "401 - ..." or "code: 401" or "401,"/"401 " etc.
        if f" {token}" in error_str or f"{token} " in error_str or f"{token}:" in error_str or f"{token}," in error_str or f"{token}-" in error_str or f"code: {token}" in error_str or error_str.startswith(f"{token}"):
            return False

    # 2. Explicit non-retryable message patterns.
    for pattern in NON_RETRYABLE_PATTERNS:
        if pattern in error_str:
            return False

    # 3. Retryable status codes.
    for code in RETRYABLE_STATUS_CODES:
        if str(code) in error_str:
            return True

    # 4. Retryable message patterns.
    for pattern in RETRYABLE_PATTERNS:
        if pattern in error_str:
            return True

    # 5. Default: don't retry unknown errors.
    return False


def _is_throttle_error(error: Exception) -> bool:
    """True when ``error`` is a 429 / rate-limit throttle (a "slow down", not a
    dead route). Only throttles get the patient plan-route budget; every other
    retryable error (5xx, timeout, "unavailable") uses the tight default so
    failover to a healthy route isn't stalled for minutes (#2074 regression)."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429
    error_str = str(error).lower()
    # No structured code: match 429 with the same word-boundary care
    # is_retryable_error uses, plus the explicit rate-limit phrasings.
    if (" 429" in error_str or "429 " in error_str or "429:" in error_str
            or "429," in error_str or error_str.startswith("429")):
        return True
    return "rate limit" in error_str or "rate_limit" in error_str


async def with_retry(
    func: Callable[..., Any],
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    throttle_max_retries: int = THROTTLE_MAX_RETRIES,
    throttle_max_delay: float = THROTTLE_MAX_DELAY,
    **kwargs
) -> T:
    """
    Execute an async function with exponential backoff retry.

    Retries on rate limiting (429) and server errors (5xx).
    Uses exponential backoff with jitter to prevent thundering herd.

    Args:
        func: The async function to execute
        *args: Positional arguments to pass to func
        max_retries: Retry attempts for a NON-throttle transient error (default 5)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Delay cap for a non-throttle error (default: 60.0)
        throttle_max_retries: Retry attempts for a 429/rate-limit throttle
            (default 8 — patient, to ride out a plan-route rate limit)
        throttle_max_delay: Delay cap for a throttle (default 120.0)
        **kwargs: Keyword arguments to pass to func

    The budget is chosen per error: a 429/rate-limit throttle gets the patient
    plan-route budget; every other retryable error (5xx/timeout/unavailable) gets
    the tight default so failover to a healthy route isn't stalled for minutes.

    A server-advised cool-down (``Retry-After``) is a fact about when the next
    attempt can succeed, so it is honoured as one wait when it fits the budget
    the loop has left (the remaining attempts times the delay cap), and the
    loop stops at once with :class:`AdvisedWaitExceedsRetryBudget` when it does
    not: eight capped waits against advice to come back in hours are attempts
    that cannot succeed, and they held a turn's conversation lock for the
    whole budget before failing anyway (#3127). Only a guessed delay (no
    advice) is clamped to the per-attempt cap.

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail, or
        AdvisedWaitExceedsRetryBudget when the advised wait cannot fit.
    """
    attempt = 0
    # Seconds this call has already slept. The budget below bounds the SUM
    # of waits, not a price per attempt: pricing each remaining attempt at
    # the cap and then sleeping advice uncapped let one throttle hold a turn
    # for 3360 s (#3127 review).
    waited = 0.0

    while True:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if not is_retryable_error(e):
                raise

            # Pick the budget from the error type: patient for a throttle,
            # tight for any other transient error.
            # A plan-limit window is checked FIRST: it is neither a 429
            # throttle (whose patient budget would overrun the turn watchdog)
            # nor an ordinary transient error.
            if is_plan_limit_error(e):
                budget_kind = "plan_limit"
                eff_max_retries = PLAN_LIMIT_MAX_RETRIES
                eff_max_delay = PLAN_LIMIT_MAX_DELAY
            elif _is_throttle_error(e):
                budget_kind = "throttle"
                eff_max_retries = throttle_max_retries
                eff_max_delay = throttle_max_delay
            else:
                budget_kind = "default"
                eff_max_retries = max_retries
                eff_max_delay = max_delay

            # What this call may still sleep in total: the loop's classic
            # worst case (every remaining attempt at the cap) less what it
            # has slept already; nothing once no attempt remains. Advice is
            # read BEFORE the attempt count ends the loop, so a Retry-After
            # that arrives on the final attempt still reaches the caller as
            # a reset time rather than as the raw provider error.
            attempts_left = eff_max_retries - attempt - 1
            remaining_budget = (
                0.0 if attempts_left <= 0
                else max(0.0, eff_max_delay * (eff_max_retries - 1) - waited)
            )

            # Prefer the server-advised cool-down; otherwise exponential
            # backoff + jitter. Both are capped at the effective max delay.
            advised = retry_after_seconds(e)
            if advised is not None:
                # Advice is not clamped: a wait that fits what the loop could
                # still spend is taken whole; one that does not ends the loop.
                if advised <= 0 and remaining_budget <= 0:
                    # "Retry now" with nothing left to wait is the hot loop
                    # the advice-less branch below refuses; end here too.
                    logger.warning(
                        "LLM retry budget spent after %.0fs of waits (status=%s): %s: %s",
                        waited, getattr(e, "status_code", None), type(e).__name__, e,
                    )
                    raise
                if advised > remaining_budget:
                    retry_at = datetime.now(UTC) + timedelta(
                        seconds=min(advised, float(ADVISED_WAIT_HORIZON_SECONDS))
                    )
                    logger.warning(
                        "LLM retry declined: advised wait %.0fs exceeds the %.0fs "
                        "this call could still wait (status=%s, retry_at=%s): %s: %s",
                        advised, remaining_budget, getattr(e, "status_code", None),
                        retry_at.isoformat(timespec="seconds"), type(e).__name__, e,
                    )
                    raise AdvisedWaitExceedsRetryBudget(
                        e,
                        advised_seconds=advised,
                        budget_seconds=remaining_budget,
                        retry_at=retry_at,
                    ) from e
                if attempts_left <= 0:
                    raise
                delay = advised + random.uniform(0, 1)
            else:
                if attempts_left <= 0:
                    raise
                # A spent budget ends the loop: an attempt without a wait is
                # a request fired back-to-back at a provider that just
                # refused, with no backoff and no jitter. (Advice beyond the
                # budget is declined above, with its reset time.)
                if remaining_budget <= 0:
                    logger.warning(
                        "LLM retry budget spent after %.0fs of waits (status=%s): %s: %s",
                        waited, getattr(e, "status_code", None), type(e).__name__, e,
                    )
                    raise
                delay = min(
                    base_delay * (2 ** attempt) + random.uniform(0, 1),
                    eff_max_delay,
                    remaining_budget,
                )

            # Log the *actual* error — type, status code, and advised wait —
            # so a throttle that's being ridden out (rather than billed via
            # fallback) is visible, not silent.
            status_code = getattr(e, "status_code", None)
            logger.warning(
                "LLM retry %d/%d after %.1fs (status=%s, budget=%s, retry_after=%s): %s: %s",
                attempt + 1, eff_max_retries, delay, status_code, budget_kind,
                advised, type(e).__name__, e,
            )
            await asyncio.sleep(delay)
            waited += delay
            attempt += 1


async def retry_with_backoff(
    func: Callable[..., Any],
    *args,
    **kwargs
) -> T:
    """
    Alias for with_retry for backwards compatibility.

    This maintains compatibility with the existing Vertex adapter.
    """
    return await with_retry(func, *args, **kwargs)
