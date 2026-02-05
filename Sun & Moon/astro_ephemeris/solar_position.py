from __future__ import annotations

from datetime import datetime, date as date_type, timezone
from typing import Optional, Tuple
import math

from .models import GeoPosition
from . import helper


class SolarPosition:
    """
    Strict translation of SolarPosition.cs (behavior-preserving).
    """

    JulianEpochJ2000 = 2451545.0
    DegreesPerCircle = 360.0
    RadiansToDegrees = 180.0 / math.pi
    DegreesToRadians = math.pi / 180.0

    @staticmethod
    def get_position(time: Optional[datetime] = None) -> GeoPosition:
        if time is None:
            time = datetime.now(timezone.utc)

        utc_time = helper.to_utc_if_needed(time)

        julian_date = helper.datetime_to_julian_date(utc_time)
        n = julian_date - SolarPosition.JulianEpochJ2000

        mean_solar_longitude = helper.normalize_angle(280.460 + 0.9856474 * n)
        mean_anomaly = helper.normalize_angle(357.528 + 0.9856003 * n)
        mean_anomaly_rad = mean_anomaly * SolarPosition.DegreesToRadians

        ecliptic_longitude = (
            mean_solar_longitude
            + 1.915 * math.sin(mean_anomaly_rad)
            + 0.020 * math.sin(2 * mean_anomaly_rad)
        )

        ecliptic_obliquity = 23.439 - 0.0000004 * n

        ecliptic_longitude_rad = ecliptic_longitude * SolarPosition.DegreesToRadians
        ecliptic_obliquity_rad = ecliptic_obliquity * SolarPosition.DegreesToRadians

        right_ascension = math.atan2(
            math.cos(ecliptic_obliquity_rad) * math.sin(ecliptic_longitude_rad),
            math.cos(ecliptic_longitude_rad),
        )

        declination = math.asin(
            math.sin(ecliptic_obliquity_rad) * math.sin(ecliptic_longitude_rad)
        )

        gmst = helper.calculate_gmst(utc_time)
        right_ascension_deg = helper.normalize_angle(right_ascension * SolarPosition.RadiansToDegrees)

        longitude = right_ascension_deg - gmst
        longitude = helper.normalize_longitude(longitude)

        latitude = declination * SolarPosition.RadiansToDegrees
        return GeoPosition(latitude, longitude)

    @staticmethod
    def get_solar_coordinates(utc_time: datetime) -> Tuple[float, float]:
        julian_date = helper.datetime_to_julian_date(helper.to_utc_if_needed(utc_time))
        n = julian_date - SolarPosition.JulianEpochJ2000

        mean_solar_longitude = helper.normalize_angle(280.460 + 0.9856474 * n)
        mean_anomaly = helper.normalize_angle(357.528 + 0.9856003 * n)
        mean_anomaly_rad = mean_anomaly * SolarPosition.DegreesToRadians

        ecliptic_longitude = (
            mean_solar_longitude
            + 1.915 * math.sin(mean_anomaly_rad)
            + 0.020 * math.sin(2 * mean_anomaly_rad)
        )

        ecliptic_obliquity = 23.439 - 0.0000004 * n

        ecliptic_longitude_rad = ecliptic_longitude * SolarPosition.DegreesToRadians
        ecliptic_obliquity_rad = ecliptic_obliquity * SolarPosition.DegreesToRadians

        right_ascension = math.atan2(
            math.cos(ecliptic_obliquity_rad) * math.sin(ecliptic_longitude_rad),
            math.cos(ecliptic_longitude_rad),
        )

        declination = math.asin(
            math.sin(ecliptic_obliquity_rad) * math.sin(ecliptic_longitude_rad)
        )

        right_ascension_deg = helper.normalize_angle(right_ascension * SolarPosition.RadiansToDegrees)
        declination_deg = declination * SolarPosition.RadiansToDegrees

        return right_ascension_deg, declination_deg

    @staticmethod
    def get_azimuth_elevation(
        observer_position: GeoPosition,
        time: Optional[datetime] = None
    ) -> Tuple[float, float]:
        if time is None:
            time = datetime.now(timezone.utc)

        utc_time = helper.to_utc_if_needed(time)
        right_ascension, declination = SolarPosition.get_solar_coordinates(utc_time)

        gmst = helper.calculate_gmst(utc_time)
        local_sidereal_time = helper.normalize_angle(gmst + observer_position.Longitude)

        hour_angle = helper.normalize_angle(local_sidereal_time - right_ascension)
        if hour_angle > 180.0:
            hour_angle -= 360.0

        hour_angle_rad = hour_angle * SolarPosition.DegreesToRadians
        declination_rad = declination * SolarPosition.DegreesToRadians
        latitude_rad = observer_position.Latitude * SolarPosition.DegreesToRadians

        elevation_rad = math.asin(
            math.sin(latitude_rad) * math.sin(declination_rad)
            + math.cos(latitude_rad) * math.cos(declination_rad) * math.cos(hour_angle_rad)
        )
        elevation = elevation_rad * SolarPosition.RadiansToDegrees

        azimuth_rad = math.atan2(
            -math.sin(hour_angle_rad),
            math.tan(declination_rad) * math.cos(latitude_rad)
            - math.sin(latitude_rad) * math.cos(hour_angle_rad),
        )
        azimuth = helper.normalize_angle(azimuth_rad * SolarPosition.RadiansToDegrees)

        return azimuth, elevation

    @staticmethod
    def get_rise_set(
        observer_position: GeoPosition,
        date: datetime | date_type,
        sunrise_elevation: float = -0.833,
    ):
        if isinstance(date, datetime):
            y, m, d = date.year, date.month, date.day
        else:
            y, m, d = date.year, date.month, date.day

        noon_utc = datetime(y, m, d, 12, 0, 0, tzinfo=timezone.utc)

        _, declination = SolarPosition.get_solar_coordinates(noon_utc)

        latitude_rad = observer_position.Latitude * SolarPosition.DegreesToRadians
        declination_rad = declination * SolarPosition.DegreesToRadians
        sunrise_elevation_rad = sunrise_elevation * SolarPosition.DegreesToRadians

        cos_hour_angle = (
            (math.sin(sunrise_elevation_rad) - math.sin(latitude_rad) * math.sin(declination_rad))
            / (math.cos(latitude_rad) * math.cos(declination_rad))
        )

        if cos_hour_angle > 1.0:
            return None, None
        if cos_hour_angle < -1.0:
            return None, None

        hour_angle = math.acos(cos_hour_angle) * SolarPosition.RadiansToDegrees

        solar_noon = 12.0 - (observer_position.Longitude / 15.0)

        sunrise_hour = (solar_noon - (hour_angle / 15.0) + 24.0) % 24.0
        sunset_hour = (solar_noon + (hour_angle / 15.0) + 24.0) % 24.0

        def hour_to_dt(hr: float) -> datetime:
            h = int(hr)
            minutes_f = (hr - h) * 60.0
            mi = int(minutes_f)
            sec_f = (minutes_f - mi) * 60.0
            s = int(sec_f)
            return datetime(y, m, d, h, mi, s, tzinfo=timezone.utc)

        return hour_to_dt(sunrise_hour), hour_to_dt(sunset_hour)

    @staticmethod
    def get_rise_set_today(observer_position: GeoPosition, sunrise_elevation: float = -0.833):
        today_utc = datetime.now(timezone.utc).date()
        return SolarPosition.get_rise_set(observer_position, today_utc, sunrise_elevation)
