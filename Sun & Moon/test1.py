from astro_ephemeris import GeoPosition, SolarPosition, LunarPosition
from datetime import datetime, timezone

cph = GeoPosition(55.6761, 12.5683)
t = datetime.now(timezone.utc)

print("Sun subsolar:", SolarPosition.get_position(t))
print("Sun az/el:", SolarPosition.get_azimuth_elevation(cph, t))

print("Moon sublunar:", LunarPosition.get_position(t))
print("Moon az/el:", LunarPosition.get_azimuth_elevation(cph, t))
print("Moon rise/set:", LunarPosition.get_next_rise_set(cph, t, step=5))
