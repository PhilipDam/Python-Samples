from __future__ import annotations
from datetime import datetime, timezone
import math


JULIAN_EPOCH_J2000 = 2451545.0
DAYS_PER_CENTURY = 36525.0
DEGREES_PER_CIRCLE = 360.0

SECONDS_PER_MINUTE = 60.0
MINUTES_PER_HOUR = 60.0
HOURS_PER_DAY = 24.0


def normalize_angle(angle_degrees: float) -> float:
    angle_degrees %= DEGREES_PER_CIRCLE
    if angle_degrees < 0.0:
        angle_degrees += DEGREES_PER_CIRCLE
    return angle_degrees


def normalize_longitude(longitude_degrees: float) -> float:
    longitude_degrees %= DEGREES_PER_CIRCLE
    if longitude_degrees < -180.0:
        longitude_degrees += DEGREES_PER_CIRCLE
    elif longitude_degrees > 180.0:
        longitude_degrees -= DEGREES_PER_CIRCLE
    return longitude_degrees


def to_utc_if_needed(dt: datetime) -> datetime:
    """
    Match C# DateTime.Kind behavior:
      - naive datetime => treat as UTC (Unspecified)
      - tz-aware => convert to UTC
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def datetime_to_julian_date(utc_time: datetime) -> float:
    """
    Convert a datetime to Julian Date.
    - naive treated as UTC
    - tz-aware must be UTC (or convertible to UTC via to_utc_if_needed beforehand)
    """
    dt = to_utc_if_needed(utc_time)

    year = dt.year
    month = dt.month

    seconds = dt.second + (dt.microsecond / 1_000_000.0)
    day = (
        dt.day
        + (dt.hour + (dt.minute + seconds / SECONDS_PER_MINUTE) / MINUTES_PER_HOUR) / HOURS_PER_DAY
    )

    if month <= 2:
        year -= 1
        month += 12

    A = year // 100
    B = 2 - A + (A // 4)

    jd = (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + B
        - 1524.5
    )
    return jd


def calculate_gmst(utc_time: datetime) -> float:
    """
    Greenwich Mean Sidereal Time (degrees), using the same family of approximation
    as in your C# HelperClass.
    """
    dt = to_utc_if_needed(utc_time)

    julian_date = datetime_to_julian_date(dt)
    T = (julian_date - JULIAN_EPOCH_J2000) / DAYS_PER_CENTURY

    gmst = (
        280.46061837
        + 360.98564736629 * (julian_date - JULIAN_EPOCH_J2000)
        + 0.000387933 * T * T
        - (T * T * T) / 38710000.0
    )
    return normalize_angle(gmst)
