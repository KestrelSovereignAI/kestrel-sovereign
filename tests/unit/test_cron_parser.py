"""
Unit tests for the lightweight cron expression parser.

Tests parsing, matching, and next_run computation for:
- Standard 5-field cron expressions
- Wildcards, ranges, lists, steps
- Shorthand aliases (@hourly, @daily, @weekly, etc.)
- Error handling for invalid expressions
"""

import pytest
from datetime import datetime, timezone

from kestrel_sovereign.features.scheduler.cron import (
    CronParseError,
    _parse_field,
    matches,
    next_run,
    parse,
)


# =========================================================================
# _parse_field tests
# =========================================================================


class TestParseField:
    """Tests for individual field parsing."""

    def test_wildcard(self):
        result = _parse_field("*", 0, 59)
        assert result == set(range(0, 60))

    def test_single_value(self):
        result = _parse_field("5", 0, 59)
        assert result == {5}

    def test_range(self):
        result = _parse_field("1-5", 0, 59)
        assert result == {1, 2, 3, 4, 5}

    def test_list(self):
        result = _parse_field("1,15,30,45", 0, 59)
        assert result == {1, 15, 30, 45}

    def test_step_wildcard(self):
        result = _parse_field("*/15", 0, 59)
        assert result == {0, 15, 30, 45}

    def test_step_range(self):
        result = _parse_field("1-10/3", 0, 59)
        assert result == {1, 4, 7, 10}

    def test_step_with_single_value(self):
        """A single value with step generates from value to max."""
        result = _parse_field("5/10", 0, 59)
        assert result == {5, 15, 25, 35, 45, 55}

    def test_out_of_range_raises(self):
        with pytest.raises(CronParseError):
            _parse_field("60", 0, 59)

    def test_invalid_range_raises(self):
        with pytest.raises(CronParseError):
            _parse_field("10-5", 0, 59)

    def test_non_numeric_raises(self):
        with pytest.raises(CronParseError):
            _parse_field("abc", 0, 59)

    def test_invalid_step_raises(self):
        with pytest.raises(CronParseError):
            _parse_field("*/0", 0, 59)

    def test_empty_part_raises(self):
        with pytest.raises(CronParseError):
            _parse_field(",", 0, 59)

    def test_day_of_week_range(self):
        result = _parse_field("0-6", 0, 6)
        assert result == {0, 1, 2, 3, 4, 5, 6}


# =========================================================================
# parse tests
# =========================================================================


class TestParse:
    """Tests for full expression parsing."""

    def test_all_wildcards(self):
        fields = parse("* * * * *")
        assert len(fields) == 5
        assert fields[0] == set(range(0, 60))
        assert fields[1] == set(range(0, 24))
        assert fields[2] == set(range(1, 32))
        assert fields[3] == set(range(1, 13))
        assert fields[4] == set(range(0, 7))

    def test_specific_values(self):
        fields = parse("30 14 1 6 3")
        assert fields[0] == {30}
        assert fields[1] == {14}
        assert fields[2] == {1}
        assert fields[3] == {6}
        assert fields[4] == {3}

    def test_mixed_fields(self):
        fields = parse("0,30 */6 1-15 * 1-5")
        assert fields[0] == {0, 30}
        assert fields[1] == {0, 6, 12, 18}
        assert fields[2] == set(range(1, 16))
        assert fields[3] == set(range(1, 13))
        assert fields[4] == {1, 2, 3, 4, 5}

    def test_too_few_fields_raises(self):
        with pytest.raises(CronParseError, match="Expected 5 fields"):
            parse("* * *")

    def test_too_many_fields_raises(self):
        with pytest.raises(CronParseError, match="Expected 5 fields"):
            parse("* * * * * *")


# =========================================================================
# Alias tests
# =========================================================================


class TestAliases:
    """Tests for shorthand cron aliases."""

    def test_hourly(self):
        fields = parse("@hourly")
        assert fields[0] == {0}
        assert fields[1] == set(range(0, 24))

    def test_daily(self):
        fields = parse("@daily")
        assert fields[0] == {0}
        assert fields[1] == {0}
        assert fields[2] == set(range(1, 32))

    def test_weekly(self):
        fields = parse("@weekly")
        assert fields[0] == {0}
        assert fields[1] == {0}
        assert fields[4] == {0}  # Sunday

    def test_monthly(self):
        fields = parse("@monthly")
        assert fields[0] == {0}
        assert fields[1] == {0}
        assert fields[2] == {1}

    def test_yearly(self):
        fields = parse("@yearly")
        assert fields[0] == {0}
        assert fields[1] == {0}
        assert fields[2] == {1}
        assert fields[3] == {1}

    def test_midnight(self):
        fields = parse("@midnight")
        assert fields[0] == {0}
        assert fields[1] == {0}

    def test_annually(self):
        """@annually is an alias for @yearly."""
        assert parse("@annually") == parse("@yearly")

    def test_case_insensitive(self):
        assert parse("@DAILY") == parse("@daily")
        assert parse("@Hourly") == parse("@hourly")

    def test_unknown_alias_raises(self):
        with pytest.raises(CronParseError, match="Unknown cron alias"):
            parse("@every5min")


# =========================================================================
# matches tests
# =========================================================================


class TestMatches:
    """Tests for datetime matching against cron expressions."""

    def test_every_minute_matches_any(self):
        dt = datetime(2026, 3, 5, 14, 30, 0, tzinfo=timezone.utc)
        assert matches("* * * * *", dt) is True

    def test_specific_time_matches(self):
        # 2026-03-05 14:30 is a Thursday (isoweekday=4, cron dow=4)
        dt = datetime(2026, 3, 5, 14, 30, 0, tzinfo=timezone.utc)
        assert matches("30 14 5 3 4", dt) is True

    def test_specific_time_no_match_minute(self):
        dt = datetime(2026, 3, 5, 14, 31, 0, tzinfo=timezone.utc)
        assert matches("30 14 5 3 4", dt) is False

    def test_specific_time_no_match_hour(self):
        dt = datetime(2026, 3, 5, 15, 30, 0, tzinfo=timezone.utc)
        assert matches("30 14 5 3 4", dt) is False

    def test_wildcard_day_of_week(self):
        dt = datetime(2026, 3, 5, 14, 30, 0, tzinfo=timezone.utc)
        assert matches("30 14 5 3 *", dt) is True

    def test_hourly_at_zero_minutes(self):
        dt = datetime(2026, 3, 5, 10, 0, 0, tzinfo=timezone.utc)
        assert matches("@hourly", dt) is True

    def test_hourly_not_at_zero_minutes(self):
        dt = datetime(2026, 3, 5, 10, 15, 0, tzinfo=timezone.utc)
        assert matches("@hourly", dt) is False

    def test_daily_at_midnight(self):
        dt = datetime(2026, 3, 5, 0, 0, 0, tzinfo=timezone.utc)
        assert matches("@daily", dt) is True

    def test_daily_not_at_midnight(self):
        dt = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)
        assert matches("@daily", dt) is False

    def test_every_15_minutes(self):
        dt0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt15 = datetime(2026, 1, 1, 0, 15, 0, tzinfo=timezone.utc)
        dt7 = datetime(2026, 1, 1, 0, 7, 0, tzinfo=timezone.utc)

        assert matches("*/15 * * * *", dt0) is True
        assert matches("*/15 * * * *", dt15) is True
        assert matches("*/15 * * * *", dt7) is False

    def test_sunday(self):
        # 2026-03-01 is a Sunday
        dt = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert matches("0 0 * * 0", dt) is True

    def test_saturday(self):
        # 2026-02-28 is a Saturday
        dt = datetime(2026, 2, 28, 0, 0, 0, tzinfo=timezone.utc)
        assert matches("0 0 * * 6", dt) is True

    def test_weekdays(self):
        # 2026-03-02 is a Monday
        dt_mon = datetime(2026, 3, 2, 9, 0, 0, tzinfo=timezone.utc)
        dt_sun = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
        assert matches("0 9 * * 1-5", dt_mon) is True
        assert matches("0 9 * * 1-5", dt_sun) is False


# =========================================================================
# next_run tests
# =========================================================================


class TestNextRun:
    """Tests for next execution time computation."""

    def test_next_run_every_minute(self):
        after = datetime(2026, 3, 5, 14, 30, 0, tzinfo=timezone.utc)
        nxt = next_run("* * * * *", after=after)
        assert nxt == datetime(2026, 3, 5, 14, 31, 0, tzinfo=timezone.utc)

    def test_next_run_specific_minute(self):
        after = datetime(2026, 3, 5, 14, 10, 0, tzinfo=timezone.utc)
        nxt = next_run("30 * * * *", after=after)
        assert nxt == datetime(2026, 3, 5, 14, 30, 0, tzinfo=timezone.utc)

    def test_next_run_wraps_to_next_hour(self):
        after = datetime(2026, 3, 5, 14, 45, 0, tzinfo=timezone.utc)
        nxt = next_run("30 * * * *", after=after)
        assert nxt == datetime(2026, 3, 5, 15, 30, 0, tzinfo=timezone.utc)

    def test_next_run_daily(self):
        after = datetime(2026, 3, 5, 0, 1, 0, tzinfo=timezone.utc)
        nxt = next_run("@daily", after=after)
        assert nxt == datetime(2026, 3, 6, 0, 0, 0, tzinfo=timezone.utc)

    def test_next_run_hourly(self):
        after = datetime(2026, 3, 5, 14, 30, 0, tzinfo=timezone.utc)
        nxt = next_run("@hourly", after=after)
        assert nxt == datetime(2026, 3, 5, 15, 0, 0, tzinfo=timezone.utc)

    def test_next_run_is_exclusive_of_after(self):
        """next_run should not return the 'after' time itself even if it matches."""
        after = datetime(2026, 3, 5, 14, 0, 0, tzinfo=timezone.utc)
        nxt = next_run("0 14 * * *", after=after)
        # Should skip to the next day's 14:00
        assert nxt == datetime(2026, 3, 6, 14, 0, 0, tzinfo=timezone.utc)

    def test_next_run_uses_utc_now_when_no_after(self):
        nxt = next_run("* * * * *")
        assert nxt.tzinfo is not None

    def test_next_run_weekly_sunday(self):
        # 2026-03-05 is Thursday; next Sunday is 2026-03-08
        after = datetime(2026, 3, 5, 0, 1, 0, tzinfo=timezone.utc)
        nxt = next_run("@weekly", after=after)
        assert nxt.weekday() == 6  # Sunday in Python's weekday()
        assert nxt == datetime(2026, 3, 8, 0, 0, 0, tzinfo=timezone.utc)

    def test_next_run_every_15_min(self):
        after = datetime(2026, 3, 5, 10, 7, 0, tzinfo=timezone.utc)
        nxt = next_run("*/15 * * * *", after=after)
        assert nxt == datetime(2026, 3, 5, 10, 15, 0, tzinfo=timezone.utc)


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_string_raises(self):
        with pytest.raises(CronParseError):
            parse("")

    def test_whitespace_only_raises(self):
        with pytest.raises(CronParseError):
            parse("   ")

    def test_leading_trailing_whitespace_ok(self):
        fields = parse("  0 0 * * *  ")
        assert fields[0] == {0}
        assert fields[1] == {0}

    def test_day_31_valid(self):
        fields = parse("0 0 31 * *")
        assert 31 in fields[2]

    def test_day_0_invalid(self):
        with pytest.raises(CronParseError):
            parse("0 0 0 * *")

    def test_month_0_invalid(self):
        with pytest.raises(CronParseError):
            parse("0 0 * 0 *")

    def test_month_13_invalid(self):
        with pytest.raises(CronParseError):
            parse("0 0 * 13 *")

    def test_multiple_ranges_and_lists(self):
        """Complex expression with multiple sub-expressions."""
        fields = parse("0,15,30,45 9-17 1-15,20-25 1,6 1-5")
        assert fields[0] == {0, 15, 30, 45}
        assert fields[1] == set(range(9, 18))
        assert fields[2] == set(range(1, 16)) | set(range(20, 26))
        assert fields[3] == {1, 6}
        assert fields[4] == {1, 2, 3, 4, 5}


# =========================================================================
# Run tests
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
