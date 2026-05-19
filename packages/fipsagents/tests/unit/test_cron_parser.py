"""Tests for the CronExpression parser and CronSource event source."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from fipsagents.server.sources.cron import CronExpression, CronSource

UTC = timezone.utc


# ── CronExpression parsing ────────────────────────────────────────────


class TestCronExpressionParsing:
    def test_every_minute(self):
        cron = CronExpression("* * * * *")
        assert cron.minutes == set(range(60))
        assert cron.hours == set(range(24))

    def test_weekday_mornings(self):
        cron = CronExpression("0 9 * * 1-5")
        assert cron.minutes == {0}
        assert cron.hours == {9}
        assert cron.days_of_week == {1, 2, 3, 4, 5}

    def test_every_15_minutes(self):
        cron = CronExpression("*/15 * * * *")
        assert cron.minutes == {0, 15, 30, 45}

    def test_midnight_jan_first(self):
        cron = CronExpression("0 0 1 1 *")
        assert cron.minutes == {0}
        assert cron.hours == {0}
        assert cron.days_of_month == {1}
        assert cron.months == {1}

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            CronExpression("")

    def test_wrong_field_count_raises(self):
        with pytest.raises(ValueError, match="Expected 5 fields, got 3"):
            CronExpression("* * *")

    def test_minute_out_of_range(self):
        with pytest.raises(ValueError, match="out of bounds"):
            CronExpression("60 * * * *")

    def test_hour_out_of_range(self):
        with pytest.raises(ValueError, match="out of bounds"):
            CronExpression("0 24 * * *")

    def test_yearly_macro_raises(self):
        with pytest.raises(ValueError, match="macros.*not supported"):
            CronExpression("@yearly")

    def test_reboot_macro_raises(self):
        with pytest.raises(ValueError, match="macros.*not supported"):
            CronExpression("@reboot")

    def test_l_in_field_raises(self):
        with pytest.raises(ValueError, match="Invalid value"):
            CronExpression("0 0 L * *")

    def test_whitespace_is_stripped(self):
        cron = CronExpression("  */5 * * * *  ")
        assert cron.minutes == {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}


# ── Field-level parsing ──────────────────────────────────────────────


class TestCronFieldParsing:
    def test_wildcard_full_range(self):
        result = CronExpression._parse_field("*", 0, 59, "minute")
        assert result == set(range(60))

    def test_range(self):
        result = CronExpression._parse_field("1-5", 0, 59, "minute")
        assert result == {1, 2, 3, 4, 5}

    def test_list(self):
        result = CronExpression._parse_field("1,3,5", 0, 59, "minute")
        assert result == {1, 3, 5}

    def test_step_on_wildcard(self):
        result = CronExpression._parse_field("*/15", 0, 59, "minute")
        assert result == {0, 15, 30, 45}

    def test_step_on_range(self):
        result = CronExpression._parse_field("1-5/2", 0, 59, "minute")
        assert result == {1, 3, 5}

    def test_invalid_range_start_gt_end(self):
        with pytest.raises(ValueError, match="start > end"):
            CronExpression._parse_field("5-1", 0, 59, "minute")

    def test_step_zero_raises(self):
        with pytest.raises(ValueError, match="Step must be positive"):
            CronExpression._parse_field("*/0", 0, 59, "minute")

    def test_mixed_list_and_range(self):
        result = CronExpression._parse_field("1-3,7,10-12", 0, 59, "minute")
        assert result == {1, 2, 3, 7, 10, 11, 12}

    def test_single_value(self):
        result = CronExpression._parse_field("30", 0, 59, "minute")
        assert result == {30}

    def test_range_out_of_bounds(self):
        with pytest.raises(ValueError, match="out of bounds"):
            CronExpression._parse_field("0-25", 0, 23, "hour")

    def test_negative_step_raises(self):
        with pytest.raises(ValueError, match="Step must be positive"):
            CronExpression._parse_field("*/-1", 0, 59, "minute")


# ── next_fire_time ────────────────────────────────────────────────────


class TestCronNextFireTime:
    def test_every_minute_fires_next_minute(self):
        cron = CronExpression("* * * * *")
        base = datetime(2026, 5, 18, 10, 30, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2026, 5, 18, 10, 31, 0, tzinfo=UTC)

    def test_weekday_mornings_skips_weekend(self):
        cron = CronExpression("0 9 * * 1-5")
        # 2026-05-17 is a Sunday
        base = datetime(2026, 5, 17, 10, 0, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        # Next weekday is Monday 2026-05-18
        assert nxt == datetime(2026, 5, 18, 9, 0, 0, tzinfo=UTC)

    def test_quarter_hours(self):
        cron = CronExpression("*/15 * * * *")
        base = datetime(2026, 5, 18, 10, 1, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2026, 5, 18, 10, 15, 0, tzinfo=UTC)

    def test_leap_year_feb_29(self):
        cron = CronExpression("0 0 29 2 *")
        # 2025-03-01: next leap year with Feb 29 is 2028
        base = datetime(2025, 3, 1, 0, 0, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2028, 2, 29, 0, 0, 0, tzinfo=UTC)

    def test_day_31_skips_30_day_month(self):
        cron = CronExpression("0 0 31 * *")
        # April has 30 days -- skip to May 31
        base = datetime(2026, 4, 1, 0, 0, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2026, 5, 31, 0, 0, 0, tzinfo=UTC)

    def test_sunday_as_0(self):
        cron = CronExpression("0 12 * * 0")
        # 2026-05-18 is Monday; next Sunday is 2026-05-24
        base = datetime(2026, 5, 18, 13, 0, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)

    def test_sunday_as_7(self):
        cron = CronExpression("0 12 * * 7")
        # Should behave identically to day-of-week 0
        base = datetime(2026, 5, 18, 13, 0, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)

    def test_exact_fire_time_returns_next(self):
        """When from_dt is exactly on a fire time, return the NEXT one."""
        cron = CronExpression("0 9 * * *")
        base = datetime(2026, 5, 18, 9, 0, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2026, 5, 19, 9, 0, 0, tzinfo=UTC)

    def test_hour_rollover(self):
        cron = CronExpression("0 * * * *")
        base = datetime(2026, 5, 18, 23, 30, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC)

    def test_month_rollover(self):
        cron = CronExpression("0 0 1 * *")
        base = datetime(2026, 5, 31, 12, 0, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)

    def test_year_rollover(self):
        cron = CronExpression("0 0 1 1 *")
        base = datetime(2026, 12, 31, 23, 59, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2027, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_specific_day_and_time(self):
        cron = CronExpression("30 14 15 6 *")
        base = datetime(2026, 6, 15, 14, 0, 0, tzinfo=UTC)
        nxt = cron.next_fire_time(base)
        assert nxt == datetime(2026, 6, 15, 14, 30, 0, tzinfo=UTC)

    def test_impossible_date_raises(self):
        """Feb 30 never exists; should raise after scanning 4 years."""
        cron = CronExpression("0 0 30 2 *")
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        with pytest.raises(RuntimeError, match="No matching time found"):
            cron.next_fire_time(base)


# ── CronSource ────────────────────────────────────────────────────────


class TestCronSource:
    def _make_config(
        self,
        schedule: str = "*/5 * * * *",
        event_type: str = "heartbeat",
        max_events_per_second: float = 10.0,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            schedule=schedule,
            event_type=event_type,
            max_events_per_second=max_events_per_second,
        )

    def test_construction(self):
        cfg = self._make_config()
        source = CronSource("test-cron", config=cfg)
        assert source.source_id == "test-cron"
        assert source._event_type == "heartbeat"

    def test_missing_config_raises(self):
        with pytest.raises(ValueError, match="requires a config"):
            CronSource("test-cron", config=None)

    def test_invalid_schedule_raises(self):
        cfg = self._make_config(schedule="not a cron")
        with pytest.raises(ValueError):
            CronSource("test-cron", config=cfg)

    async def test_consume_yields_event(self):
        cfg = self._make_config(event_type="daily_check")
        source = CronSource("cron-1", config=cfg)

        now = datetime.now(tz=UTC)

        with (
            patch.object(
                source._cron, "next_fire_time", return_value=now,
            ),
            patch("fipsagents.server.sources.cron.asyncio.sleep", new_callable=AsyncMock),
        ):
            events = []
            async for event in source.consume():
                events.append(event)
                if len(events) >= 1:
                    break

        evt = events[0]
        assert evt.event_type == "daily_check"
        assert evt.source == "cron-1"
        assert evt.session_key == "event:cron:daily_check"
        assert "scheduled_time" in evt.payload

    async def test_session_key_format(self):
        cfg = self._make_config(event_type="nightly_scan")
        source = CronSource("cron-2", config=cfg)

        now = datetime.now(tz=UTC)

        with (
            patch.object(
                source._cron, "next_fire_time", return_value=now,
            ),
            patch("fipsagents.server.sources.cron.asyncio.sleep", new_callable=AsyncMock),
        ):
            async for event in source.consume():
                assert event.session_key == "event:cron:nightly_scan"
                break

    async def test_cancellation_safe(self):
        cfg = self._make_config()
        source = CronSource("cron-cancel", config=cfg)

        async def _consume_one():
            async for _ in source.consume():
                break

        task = asyncio.create_task(_consume_one())
        # Give the task a moment to start, then cancel it.
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    def test_repr(self):
        cron = CronExpression("0 9 * * 1-5")
        assert repr(cron) == "CronExpression('0 9 * * 1-5')"
