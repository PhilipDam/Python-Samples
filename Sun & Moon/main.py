# Program.py
# Translation of Program.cs (C# top-level statements) to Python.

from __future__ import annotations
from astro_ephemeris import GeoPosition, SolarPosition, LunarPosition


def main() -> None:
    print("Solar Position Calculator Demo\n")

    kagoshima = GeoPosition(31.35, 130.33)
    tokyo = GeoPosition(35.68, 139.76)
    london = GeoPosition(51.51, -0.13)
    dhaka = GeoPosition(23.81, 90.41)
    copenhagen = GeoPosition(55.712139, 12.547889)

    observer = copenhagen
    print(f"Observer: {observer.Latitude:.4f}°, {observer.Longitude:.4f}°")

    # 1. Get current sun subsolar point
    sun_position = SolarPosition.get_position()
    print("1. Current Sun Subsolar Point:")
    print(
        f"   Sun is directly overhead at: "
        f"{sun_position.Latitude:.2f}°, {sun_position.Longitude:.2f}°\n"
    )

    # 2. Get sun's azimuth and elevation from observer
    azimuth, elevation = SolarPosition.get_azimuth_elevation(observer)
    print("2. Sun's Position from Observer:")
    print(f"   Azimuth: {azimuth:.1f}° (from North)")
    print(f"   Elevation: {elevation:.1f}° (above horizon)\n")

    # 3. Get today's sunrise and sunset
    sunrise, sunset = SolarPosition.get_rise_set_today(observer)
    print("3. Today's Sunrise & Sunset:")
    print(f"   Sunrise: {sunrise:%H:%M:%S}" if sunrise is not None else "   No sunrise today")
    print(f"   Sunset: {sunset:%H:%M:%S}" if sunset is not None else "   No sunset today")

    print("\n\nLunar Position Calculator Demo\n")

    # 1. Get current moon sublunar point
    moon_position = LunarPosition.get_position()
    print("1. Current Moon Subsolar Point:")
    print(
        f"   Moon is directly overhead at: "
        f"{moon_position.Latitude:.2f}°, {moon_position.Longitude:.2f}°\n"
    )

    # 2. Get moon's azimuth and elevation from observer
    moon_azimuth, moon_elevation = LunarPosition.get_azimuth_elevation_now(observer)
    print("2. Moon's Position from Observer:")
    print(f"   Azimuth: {moon_azimuth:.1f}° (from North)")
    print(f"   Elevation: {moon_elevation:.1f}° (above horizon)\n")

    # 3. Get the next moonrise and moonset times occurring after now (within the next 24 hours)
    moonrise, moonset = LunarPosition.get_next_rise_set_today(observer)
    print("3. Today's Moonrise & Moonset:")
    print(f"   Moonrise: {moonrise:%H:%M:%S}" if moonrise is not None else "   No moonrise today")
    print(f"   Moonset: {moonset:%H:%M:%S}" if moonset is not None else "   No moonset today")


if __name__ == "__main__":
    main()
