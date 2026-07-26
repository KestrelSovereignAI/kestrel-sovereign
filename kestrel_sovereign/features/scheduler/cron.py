"""
Lightweight cron expression parser -- pure Python, no dependencies.

Supports standard 5-field cron syntax:
    minute hour day_of_month month day_of_week

Field values:
    *       any value
    N       exact match
    N-M     range (inclusive)
    N,M,O   list
    */N     step (every Nth value)
    N-M/S   range with step

Special shorthand strings:
    @hourly     -> 0 * * * *
    @daily      -> 0 0 * * *
    @weekly     -> 0 0 * * 0
    @monthly    -> 0 0 1 * *
    @yearly     -> 0 0 1 1 *
    @midnight   -> 0 0 * * *
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Optional, Set


# Shorthand aliases
ALIASES = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
}

# Valid ranges per field (min, max)
FIELD_RANGES = [
    (0, 59),   # minute
    (0, 23),   # hour
    (1, 31),   # day of month
    (1, 12),   # month
    (0, 6),    # day of week (0 = Sunday)
]

FIELD_NAMES = ["minute", "hour", "day_of_month", "month", "day_of_week"]


class CronParseError(ValueError):
    """Raised when a cron expression cannot be parsed."""


def get_timezone(timezone_name: str = "UTC") -> ZoneInfo:
    """Resolve an IANA time-zone name used by a cron schedule.

    Scheduler timestamps are always persisted as UTC instants.  A zone only
    controls which local wall-clock times match a cron expression.  We reject
    invalid names at schedule creation instead of accepting a schedule that
    can never be evaluated.
    """
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        raise CronParseError("timezone must be a non-empty IANA timezone name")
    try:
        return ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise CronParseError(f"Unknown IANA timezone '{timezone_name}'") from e


def _parse_field(field: str, min_val: int, max_val: int) -> Set[int]:
    """
    Parse a single cron field into the set of matching integer values.

    Args:
        field: The cron field string (e.g. "*/5", "1-3", "1,3,5", "10")
        min_val: Minimum valid value for this field
        max_val: Maximum valid value for this field

    Returns:
        Set of integers that this field matches

    Raises:
        CronParseError: If the field cannot be parsed
    """
    result: Set[int] = set()

    for part in field.split(","):
        part = part.strip()
        if not part:
            raise CronParseError(f"Empty sub-expression in field '{field}'")

        # Handle step: either */N or range/N
        step = 1
        if "/" in part:
            base, step_str = part.split("/", 1)
            try:
                step = int(step_str)
            except ValueError:
                raise CronParseError(f"Invalid step value '{step_str}' in '{field}'")
            if step < 1:
                raise CronParseError(f"Step must be >= 1, got {step} in '{field}'")
            part = base

        if part == "*":
            result.update(range(min_val, max_val + 1, step))
        elif "-" in part:
            # Range
            try:
                low_str, high_str = part.split("-", 1)
                low, high = int(low_str), int(high_str)
            except ValueError:
                raise CronParseError(f"Invalid range '{part}' in field '{field}'")
            if low < min_val or high > max_val or low > high:
                raise CronParseError(
                    f"Range {low}-{high} out of bounds [{min_val}-{max_val}] in '{field}'"
                )
            result.update(range(low, high + 1, step))
        else:
            # Single value
            try:
                val = int(part)
            except ValueError:
                raise CronParseError(f"Invalid value '{part}' in field '{field}'")
            if val < min_val or val > max_val:
                raise CronParseError(
                    f"Value {val} out of bounds [{min_val}-{max_val}] in '{field}'"
                )
            # If step was specified with a single value (e.g. "5/2"), generate
            # from that value through max_val with given step
            if step > 1:
                result.update(range(val, max_val + 1, step))
            else:
                result.add(val)

    return result


def parse(expression: str) -> list:
    """
    Parse a full cron expression into a list of 5 sets of matching values.

    Args:
        expression: A 5-field cron string or a shorthand alias

    Returns:
        List of 5 sets: [minutes, hours, days_of_month, months, days_of_week]

    Raises:
        CronParseError: If the expression is invalid
    """
    expression = expression.strip()

    # Resolve aliases
    if expression.startswith("@"):
        alias = ALIASES.get(expression.lower())
        if alias is None:
            raise CronParseError(f"Unknown cron alias '{expression}'")
        expression = alias

    fields = expression.split()
    if len(fields) != 5:
        raise CronParseError(
            f"Expected 5 fields, got {len(fields)} in '{expression}'"
        )

    result = []
    for i, field in enumerate(fields):
        min_val, max_val = FIELD_RANGES[i]
        try:
            result.append(_parse_field(field, min_val, max_val))
        except CronParseError:
            raise
        except Exception as e:
            raise CronParseError(f"Error parsing {FIELD_NAMES[i]} field '{field}': {e}")

    return result


def matches(expression: str, dt: datetime) -> bool:
    """
    Check whether a datetime matches a cron expression.

    Args:
        expression: A cron expression string (5 fields or alias)
        dt: The datetime to test

    Returns:
        True if dt matches the cron schedule
    """
    fields = parse(expression)
    minutes, hours, days_of_month, months, days_of_week = fields

    # isoweekday(): Monday=1 .. Sunday=7; cron uses Sunday=0 .. Saturday=6
    cron_dow = dt.isoweekday() % 7  # Sunday=0

    return (
        dt.minute in minutes
        and dt.hour in hours
        and dt.day in days_of_month
        and dt.month in months
        and cron_dow in days_of_week
    )


def next_run(
    expression: str,
    after: Optional[datetime] = None,
    timezone_name: str = "UTC",
) -> datetime:
    """
    Calculate the next datetime that matches the cron expression.

    Advances minute-by-minute from *after* (exclusive) until a match is found.
    For safety, stops searching after scanning roughly 4 years (2,102,400
    minutes) to avoid infinite loops on impossible expressions.

    Args:
        expression: A cron expression string
        after: Starting point (exclusive). Defaults to now (UTC).
        timezone_name: IANA zone whose local wall clock is matched.  The
            returned timestamp remains an aware UTC instant.  A non-existent
            local time during a DST spring-forward gap is skipped.  During a
            fall-back fold, a matching local time runs once at its earlier
            occurrence (``fold=0``), never twice.

    Returns:
        The next matching datetime (timezone-aware UTC)

    Raises:
        CronParseError: If no match is found within the safety limit
    """
    fields = parse(expression)  # parse once, reuse below
    minutes_set, hours_set, dom_set, months_set, dow_set = fields
    schedule_tz = get_timezone(timezone_name)

    if after is None:
        after = datetime.now(timezone.utc)
    elif after.tzinfo is None:
        # Historic callers supplied naive UTC values.  Retain that compatible
        # interpretation while returning a UTC instant below.
        after = after.replace(tzinfo=timezone.utc)
    else:
        after = after.astimezone(timezone.utc)

    # Advance *UTC instants*, then test their local representations.  This
    # avoids manufacturing non-existent gap times and lets us explicitly skip
    # the second representation of an ambiguous fold minute.
    candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

    # Safety limit: ~4 years of minutes
    max_iterations = 4 * 366 * 24 * 60

    for _ in range(max_iterations):
        local_candidate = candidate.astimezone(schedule_tz)
        # ``fold=1`` is the repeated wall-clock hour after a fall-back.  The
        # scheduler's contract is one run per local matching minute, at the
        # first (earlier UTC) occurrence.
        if local_candidate.fold:
            candidate += timedelta(minutes=1)
            continue
        cron_dow = local_candidate.isoweekday() % 7
        if (
            local_candidate.minute in minutes_set
            and local_candidate.hour in hours_set
            and local_candidate.day in dom_set
            and local_candidate.month in months_set
            and cron_dow in dow_set
        ):
            return candidate
        candidate += timedelta(minutes=1)

    raise CronParseError(
        f"No matching time found within safety limit for expression '{expression}'"
    )
