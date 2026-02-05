from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, Tuple
import math

from .models import GeoPosition, FundamentalArguments, EclipticCoordinates, EquatorialCoordinates
from . import helper


class LunarPosition:
    """
    Translation of LunarPosition.cs
    """

    JULIAN_EPOCH_J2000 = 2451545.0
    DAYS_PER_CENTURY = 36525.0
    RADIANS_TO_DEGREES = 180.0 / math.pi
    DEGREES_TO_RADIANS = math.pi / 180.0

    @staticmethod
    def get_position(time: Optional[datetime] = None) -> GeoPosition:
        if time is None:
            time = datetime.utcnow()

        utc_time = helper.to_utc_if_needed(time)

        julian_date = helper.datetime_to_julian_date(utc_time)
        T = (julian_date - LunarPosition.JULIAN_EPOCH_J2000) / LunarPosition.DAYS_PER_CENTURY

        args = LunarPosition._calculate_fundamental_arguments(T)
        ecl = LunarPosition._calculate_ecliptic_coordinates(args)
        eq = LunarPosition._convert_to_equatorial_coordinates(ecl, T)

        return LunarPosition._calculate_sublunar_point(eq, utc_time)

    @staticmethod
    def get_azimuth_elevation(observer: GeoPosition, time: datetime) -> Tuple[float, float]:
        utc_time = helper.to_utc_if_needed(time)

        julian_date = helper.datetime_to_julian_date(utc_time)
        T = (julian_date - LunarPosition.JULIAN_EPOCH_J2000) / LunarPosition.DAYS_PER_CENTURY

        args = LunarPosition._calculate_fundamental_arguments(T)
        ecl = LunarPosition._calculate_ecliptic_coordinates(args)
        eq = LunarPosition._convert_to_equatorial_coordinates(ecl, T)

        lat_rad = observer.Latitude * LunarPosition.DEGREES_TO_RADIANS

        gmst = helper.calculate_gmst(utc_time)  # degrees
        lst = helper.normalize_angle(gmst + observer.Longitude)  # degrees

        ha = (lst - (eq.RightAscension * LunarPosition.RADIANS_TO_DEGREES)) * LunarPosition.DEGREES_TO_RADIANS
        dec = eq.Declination

        sin_alt = math.sin(dec) * math.sin(lat_rad) + math.cos(dec) * math.cos(lat_rad) * math.cos(ha)
        alt = math.asin(sin_alt)

        cos_az = (math.sin(dec) - math.sin(alt) * math.sin(lat_rad)) / (math.cos(alt) * math.cos(lat_rad))
        if cos_az < -1.0:
            cos_az = -1.0
        elif cos_az > 1.0:
            cos_az = 1.0

        az = math.acos(cos_az)
        if math.sin(ha) > 0.0:
            az = 2.0 * math.pi - az

        return az * LunarPosition.RADIANS_TO_DEGREES, alt * LunarPosition.RADIANS_TO_DEGREES

    @staticmethod
    def get_azimuth_elevation_now(observer: GeoPosition) -> Tuple[float, float]:
        return LunarPosition.get_azimuth_elevation(observer, datetime.utcnow())

    @staticmethod
    def get_next_rise_set(
        observer: GeoPosition,
        date: datetime,
        step: int = 1
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        date = helper.to_utc_if_needed(date)
        step = max(1, int(step))

        start_utc = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=date.tzinfo)
        end_utc = start_utc + timedelta(days=1)

        moonrise: Optional[datetime] = None
        moonset: Optional[datetime] = None

        prev_elevation: Optional[float] = None
        prev_time: Optional[datetime] = None

        t = start_utc
        while t <= end_utc:
            _, elevation = LunarPosition.get_azimuth_elevation(observer, t)

            if prev_elevation is not None and prev_time is not None:
                if prev_elevation < 0.0 <= elevation and moonrise is None:
                    moonrise = LunarPosition._interpolate_rise_set(observer, prev_time, t)
                elif prev_elevation > 0.0 >= elevation and moonset is None:
                    moonset = LunarPosition._interpolate_rise_set(observer, prev_time, t)

                if moonrise is not None and moonset is not None:
                    break

            prev_elevation = elevation
            prev_time = t
            t = t + timedelta(minutes=step)

        return moonrise, moonset

    @staticmethod
    def get_next_rise_set_today(observer: GeoPosition, step: int = 1) -> Tuple[Optional[datetime], Optional[datetime]]:
        return LunarPosition.get_next_rise_set(observer, datetime.utcnow(), step)

    # -------------------------
    # Private helpers
    # -------------------------

    @staticmethod
    def _calculate_fundamental_arguments(T: float) -> FundamentalArguments:
        mean_longitude = 218.316 + 13.176396 * LunarPosition.DAYS_PER_CENTURY * T
        mean_elongation = (
            297.8502
            + 445267.1115 * T
            - 0.0016300 * T * T
            + (T * T * T) / 545868.0
            - (T * T * T * T) / 113065000.0
        )
        sun_mean_anomaly = (
            357.5291
            + 35999.0503 * T
            - 0.0001559 * T * T
            - 0.00000048 * T * T * T
        )
        moon_mean_anomaly = (
            134.9634
            + 477198.8675 * T
            + 0.0087414 * T * T
            + (T * T * T) / 69699.0
            - (T * T * T * T) / 14712000.0
        )
        argument_of_latitude = (
            93.2720
            + 483202.0175 * T
            - 0.0036539 * T * T
            - (T * T * T) / 3526000.0
            + (T * T * T * T) / 863310000.0
        )

        return FundamentalArguments(
            L=helper.normalize_angle(mean_longitude),
            D=helper.normalize_angle(mean_elongation),
            M=helper.normalize_angle(sun_mean_anomaly),
            M_moon=helper.normalize_angle(moon_mean_anomaly),
            F=helper.normalize_angle(argument_of_latitude),
        )

    @staticmethod
    def _calculate_ecliptic_coordinates(args: FundamentalArguments) -> EclipticCoordinates:
        D_rad = args.D * LunarPosition.DEGREES_TO_RADIANS
        M_rad = args.M * LunarPosition.DEGREES_TO_RADIANS
        M_moon_rad = args.M_moon * LunarPosition.DEGREES_TO_RADIANS
        F_rad = args.F * LunarPosition.DEGREES_TO_RADIANS

        lambd = (
            args.L
            + 6.289 * math.sin(M_moon_rad)
            + 1.274 * math.sin(2.0 * D_rad - M_moon_rad)
            + 0.658 * math.sin(2.0 * D_rad)
            + 0.214 * math.sin(2.0 * M_moon_rad)
            - 0.186 * math.sin(M_rad)
            - 0.114 * math.sin(2.0 * F_rad)
            + 0.059 * math.sin(2.0 * D_rad - 2.0 * M_moon_rad)
            + 0.057 * math.sin(2.0 * D_rad - M_rad - M_moon_rad)
            + 0.053 * math.sin(2.0 * D_rad + M_moon_rad)
            + 0.046 * math.sin(2.0 * D_rad - M_rad)
            + 0.041 * math.sin(D_rad)
            - 0.035 * math.sin(D_rad + M_moon_rad)
            - 0.030 * math.sin(D_rad - M_moon_rad)
        )

        beta = (
            5.128 * math.sin(F_rad)
            + 0.280 * math.sin(M_moon_rad + F_rad)
            + 0.277 * math.sin(M_moon_rad - F_rad)
            + 0.173 * math.sin(2.0 * D_rad - F_rad)
            + 0.055 * math.sin(2.0 * D_rad - M_moon_rad + F_rad)
            + 0.046 * math.sin(2.0 * D_rad - M_moon_rad - F_rad)
            + 0.033 * math.sin(2.0 * D_rad + F_rad)
            + 0.017 * math.sin(2.0 * M_moon_rad + F_rad)
        )

        return EclipticCoordinates(
            Lambda=helper.normalize_angle(lambd),
            Beta=beta,
        )

    @staticmethod
    def _convert_to_equatorial_coordinates(ecliptic: EclipticCoordinates, T: float) -> EquatorialCoordinates:
        obliquity = (
            23.439291
            - 0.0130042 * T
            - 0.00000016 * T * T
            + 0.000000504 * T * T * T
        )

        lambda_rad = ecliptic.Lambda * LunarPosition.DEGREES_TO_RADIANS
        beta_rad = ecliptic.Beta * LunarPosition.DEGREES_TO_RADIANS
        obliquity_rad = obliquity * LunarPosition.DEGREES_TO_RADIANS

        sin_ra = math.sin(lambda_rad) * math.cos(obliquity_rad) - math.tan(beta_rad) * math.sin(obliquity_rad)
        cos_ra = math.cos(lambda_rad)

        right_ascension = math.atan2(sin_ra, cos_ra)
        declination = math.asin(
            math.sin(beta_rad) * math.cos(obliquity_rad)
            + math.cos(beta_rad) * math.sin(obliquity_rad) * math.sin(lambda_rad)
        )

        return EquatorialCoordinates(RightAscension=right_ascension, Declination=declination)

    @staticmethod
    def _calculate_sublunar_point(equatorial: EquatorialCoordinates, utc_time: datetime) -> GeoPosition:
        ra_deg = helper.normalize_angle(equatorial.RightAscension * LunarPosition.RADIANS_TO_DEGREES)
        gmst = helper.calculate_gmst(utc_time)

        longitude = helper.normalize_longitude(ra_deg - gmst)
        latitude = equatorial.Declination * LunarPosition.RADIANS_TO_DEGREES
        return GeoPosition(latitude, longitude)

    @staticmethod
    def _interpolate_rise_set(observer: GeoPosition, t1: datetime, t2: datetime) -> datetime:
        _, el1 = LunarPosition.get_azimuth_elevation(observer, t1)
        _, el2 = LunarPosition.get_azimuth_elevation(observer, t2)

        frac = el1 / (el1 - el2)
        total_seconds = (t2 - t1).total_seconds()
        return t1 + timedelta(seconds=frac * total_seconds)
