"""Strict timezone handling at every system boundary."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Naive datetime is forbidden; provide an explicit timezone")
    return value.astimezone(UTC)


def parse_datetime(value: str) -> datetime:
    return as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def seconds_remaining(end_time: datetime, at: datetime) -> float:
    return max(0.0, (as_utc(end_time) - as_utc(at)).total_seconds())
