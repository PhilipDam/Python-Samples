
"""
Sun & Moon CLI — Corrected & Heavily Documented
------------------------------------------------
Computes:
  • Sun: sunrise, sunset, day length, and change vs. solstice reference.
  • Moon: moonrise, moonset, synodic age (days since New Moon), illuminated fraction, and phase name.

IMPORTANT FIX (relative to earlier version)
------------------------------------------
• Implemented the truncated lunar LATITUDE series (sumB). The previous script omitted sumB’s main terms
  (notably the dominant 5,128,122 * sin(F) term), which caused declination – and therefore rise/set –
  to be significantly wrong. This fix restores realistic lunar declinations and rise/set times.

References (as in the Swift code)
---------------------------------
• NOAA Solar Calculator notes (rise/set, solar noon, equation of time): https://gml.noaa.gov/grad/solcalc/
• Jean Meeus, Astronomical Algorithms (2nd ed.)
  - Ch. 7        : Time scales & Julian Day
  - Ch. 12       : Sidereal time (GMST/LST)
  - Ch. 13,21–22 : Coordinate transforms & obliquity
  - Ch. 24–26    : Solar coordinates
  - Ch. 28–29    : Equation of Time & declination approximations (NOAA-style fractional year)
  - Ch. 15       : Rising, setting, and transit
  - Ch. 46,49    : Lunar illumination & elongation
  - Ch. 47–49    : Lunar longitude/latitude/distance series (used here in truncated form)

Notes on Accuracy
-----------------
• Sun: Typically within ~1–2 minutes at mid-latitudes.
• Moon: With the truncated series below (top terms of L, B, R), rise/set is generally within a few minutes,
  but can drift more under extreme circumstances. For professional-grade ephemerides, consult USNO/JPL.

"""
from math import pi, sin, cos, tan, asin, acos, atan2, floor
from datetime import datetime, timedelta, timezone
import sys

# ---------------- Math helpers ----------------
def deg2rad(x): return x * pi / 180.0
def rad2deg(x): return x * 180.0 / pi
def clamp(x, a, b): return max(a, min(b, x))
def frac(x): return x - floor(x)

# ---------------- JD & Sidereal ----------------
def date_to_jd(dt_utc: datetime) -> float:
    if dt_utc.tzinfo is None: dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)
    Y, M = dt_utc.year, dt_utc.month
    D = dt_utc.day + (dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600)/24.0
    y, m = float(Y), float(M)
    if m <= 2: y -= 1; m += 12
    A = floor(y/100.0); B = 2 - A + floor(A/4.0)
    return floor(365.25*(y+4716.0)) + floor(30.6001*(m+1.0)) + D + B - 1524.5

def gmst_deg(jd: float) -> float:
    T = (jd - 2451545.0) / 36525.0
    theta = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T*T - T*T*T/38710000.0
    theta %= 360.0
    return theta if theta >= 0 else theta + 360.0

def lst_deg(jd: float, lon_deg: float) -> float:
    LST = (gmst_deg(jd) + lon_deg) % 360.0
    return LST if LST >= 0 else LST + 360.0

# ---------------- Formatting ----------------
def fmt_hm(dt_utc, tz_offset_hours: float) -> str:
    if dt_utc is None: return "—"
    local = dt_utc + timedelta(hours=tz_offset_hours)
    return f"{local.hour:02d}:{local.minute:02d}"

def fmt_len(seconds: float) -> str:
    total = int(round(seconds)); h = total // 3600; m = (total % 3600)//60
    return f"{h}:{m:02d}"

# ---------------- Solar (NOAA-style) ----------------
def day_of_year_utc(dt_utc: datetime) -> int:
    return int(dt_utc.strftime("%j"))

def eot_and_decl(N: int):
    # NOAA fractional year gamma using 12:00 local assumption (as in NOAA examples)
    gamma = 2.0*pi/365.0 * (N - 1 + (12 - 12)/24.0)
    eqtime = 229.18 * (0.000075 + 0.001868*cos(gamma) - 0.032077*sin(gamma)
                       - 0.014615*cos(2*gamma) - 0.040849*sin(2*gamma))
    decl = (0.006918 - 0.399912*cos(gamma) + 0.070257*sin(gamma)
            - 0.006758*cos(2*gamma) + 0.000907*sin(2*gamma)
            - 0.002697*cos(3*gamma) + 0.00148*sin(3*gamma))
    return eqtime, decl

def sun_times(date_utc: datetime, lat_deg: float, lon_deg: float):
    h0 = deg2rad(-0.833)  # standard apparent altitude
    N = day_of_year_utc(date_utc)
    eqt_min, decl = eot_and_decl(N)
    lat = deg2rad(lat_deg)
    cos_ha = (sin(h0) - sin(lat)*sin(decl)) / (cos(lat)*cos(decl))
    if cos_ha < -1 or cos_ha > 1: return None, None, 0.0
    ha = acos(clamp(cos_ha, -1, 1)); ha_deg = rad2deg(ha)
    solar_noon_min = 720 - 4*lon_deg - eqt_min
    daylight_min = 8 * ha_deg
    sunrise_min = solar_noon_min - daylight_min/2
    sunset_min  = solar_noon_min + daylight_min/2
    day_start = datetime(date_utc.year, date_utc.month, date_utc.day, tzinfo=timezone.utc)
    sr = day_start + timedelta(minutes=round(sunrise_min))
    ss = day_start + timedelta(minutes=round(sunset_min))
    return sr, ss, max(0.0, (ss - sr).total_seconds())

# ---------------- Lunar (Meeus truncated) ----------------
class Equatorial:
    def __init__(self, ra_hours: float, dec_rad: float): self.ra, self.dec = ra_hours, dec_rad
class Ecliptic:
    def __init__(self, lon_rad: float, lat_rad: float): self.lon, self.lat = lon_rad, lat_rad

def mean_obliquity(jd: float) -> float:
    T = (jd - 2451545.0) / 36525.0
    seconds = 21.448 - T*(46.8150 + T*(0.00059 - 0.001813*T))
    return deg2rad(23.0 + 26.0/60.0 + seconds/3600.0)

def sun_ecl_lon(jd: float) -> float:
    T = (jd - 2451545.0)/36525.0
    L0 = deg2rad((280.46646 + 36000.76983*T + 0.0003032*T*T)%360.0)
    M  = deg2rad(357.52911 + 35999.05029*T - 0.0001537*T*T)
    C  = (1.914602 - 0.004817*T - 0.000014*T*T)*sin(M) + (0.019993 - 0.000101*T)*sin(2*M) + 0.000289*sin(3*M)
    trueLon = deg2rad((rad2deg(L0) + C) % 360.0)
    omega = deg2rad(125.04 - 1934.136*T)
    return trueLon - deg2rad(0.00569) - deg2rad(0.00478)*sin(omega)

def moon_ecliptic(jd: float):
    T = (jd - 2451545.0)/36525.0
    Lp = deg2rad((218.3164477 + 481267.88123421*T - 0.0015786*T*T + T*T*T/538841.0 - T*T*T*T/65194000.0) % 360.0)  # mean lon
    D  = deg2rad((297.8501921 + 445267.1114034*T - 0.0018819*T*T + T*T*T/545868.0 - T*T*T*T/113065000.0) % 360.0)  # elongation
    M  = deg2rad(357.5291092 + 35999.0502909*T - 0.0001536*T*T + T*T*T/24490000.0)                                  # Sun anomaly
    Mp = deg2rad((134.9633964 + 477198.8675055*T + 0.0087414*T*T + T*T*T/69699.0 - T*T*T*T/14712000.0) % 360.0)     # Moon anomaly
    F  = deg2rad((93.2720950 + 483202.0175233*T - 0.0036539*T*T - T*T*T/3526000.0 + T*T*T*T/863310000.0) % 360.0)   # arg of lat

    A1 = deg2rad(119.75) + deg2rad(131.849)*T
    A2 = deg2rad(53.09)  + deg2rad(479264.290)*T
    A3 = deg2rad(313.45) + deg2rad(481266.484)*T

    # Longitude series (arcsec) – top terms
    sumL = (6288774*sin(Mp) + 1274027*sin(2*D - Mp) + 658314*sin(2*D) + 213618*sin(2*Mp)
            - 185116*sin(M) - 114332*sin(2*F) + 58793*sin(2*D - 2*Mp) + 57066*sin(2*D - M - Mp)
            + 53322*sin(2*D + Mp) + 45758*sin(2*D - M) - 40923*sin(M - Mp) - 34720*sin(D)
            - 30383*sin(M + Mp) + 15327*sin(2*D - 2*F) - 12528*sin(Mp + 2*F) + 10980*sin(Mp - 2*F)
            + 10675*sin(4*D - Mp) + 10034*sin(3*Mp))

    # Latitude series (arcsec) – TOP DOMINANT TERMS (truncated)
    # These are the largest contributors; adding them corrects declination significantly.
    sumB = (5128122*sin(F)
            + 280602*sin(Mp + F)
            + 277693*sin(Mp - F)
            + 173237*sin(2*D - F)
            + 55413*sin(2*D - Mp + F)
            + 46271*sin(2*D - Mp - F)
            + 32573*sin(2*D + F)
            + 17198*sin(2*Mp + F)
            + 9266*sin(2*D + Mp - F)
            + 8822*sin(2*Mp - F)
            + 8216*sin(2*D - M - F)
            + 4324*sin(2*D - 2*Mp - F)
            + 4200*sin(2*D + Mp + F)
            - 3359*sin(2*D + M - F)
            + 2463*sin(2*D - M - Mp + F)
            + 2211*sin(2*D - M + F)
            + 2065*sin(2*D - Mp + F)  # note: appears again with small variants in full table
    )

    # Distance series (meters) – top terms
    sumR = (-20905355*cos(Mp) - 3699111*cos(2*D - Mp) - 2955968*cos(2*D) - 569925*cos(2*Mp)
            + 48888*cos(M) - 3149*cos(2*F) + 246158*cos(2*D - 2*Mp) - 152138*cos(2*D - M - Mp)
            - 170733*cos(2*D + Mp) - 204586*cos(2*D - M) - 129620*cos(M - Mp) + 108743*cos(D)
            + 104755*cos(M + Mp) + 10321*cos(2*D - 2*F) + 79661*cos(Mp - 2*F))

    lon = Lp + (sumL + 3958*sin(A1) + 1962*sin(Lp - F) + 318*sin(A2)) / 1296000.0
    lat = (sumB - 2235*sin(Lp) + 382*sin(A3) + 175*sin(A1 - F) + 175*sin(A1 + F)
           + 127*sin(Lp - Mp) - 115*sin(Lp + Mp)) / 1296000.0

    distance_km = 385000.56 + (sumR / 1000.0)
    return Ecliptic(lon, lat), distance_km

def ecl_to_equ(ecl, jd: float):
    eps = mean_obliquity(jd)
    sinE, cosE = sin(eps), cos(eps)
    sinLon, cosLon = sin(ecl.lon), cos(ecl.lon)
    ra = atan2(cosE * sinLon - sinE * tan(ecl.lat), cosLon)
    dec = asin(sin(ecl.lat)*cosE + cos(ecl.lat)*sinE*sinLon)
    ra_hours = (rad2deg(ra)/15.0) % 24.0
    if ra_hours < 0: ra_hours += 24.0
    return Equatorial(ra_hours, dec)

def moon_equatorial(jd: float):
    ecl, dist = moon_ecliptic(jd)
    eq = ecl_to_equ(ecl, jd)
    return eq, ecl.lon, dist

def moon_phase_illum_age(jd: float):
    lam_sun = sun_ecl_lon(jd)
    _, lam_moon, _ = moon_equatorial(jd)
    dlon = (rad2deg(lam_moon - lam_sun)) % 360.0
    if dlon < 0: dlon += 360.0
    phase = dlon/360.0
    synodic = 29.530588861
    age = phase * synodic
    illum = 0.5 * (1 - cos(deg2rad(dlon)))
    if phase < 0.03 or phase >= 0.97: name = "New Moon"
    elif phase < 0.22: name = "Waxing Crescent"
    elif phase < 0.28: name = "First Quarter"
    elif phase < 0.47: name = "Waxing Gibbous"
    elif phase < 0.53: name = "Full Moon"
    elif phase < 0.72: name = "Waning Gibbous"
    elif phase < 0.78: name = "Last Quarter"
    else: name = "Waning Crescent"
    return illum, age, name

# ---------------- Generic rise/set ----------------
def rise_set_for_body(date_utc: datetime, lat_deg: float, lon_deg: float, body_fn, h0_deg: float, iterations: int = 2):
    day_start = datetime(date_utc.year, date_utc.month, date_utc.day, tzinfo=timezone.utc)
    jd0 = date_to_jd(day_start)
    def eq_at(frac_day: float): return body_fn(jd0 + frac_day)
    phi = deg2rad(lat_deg); sinphi, cosphi = sin(phi), cos(phi)
    h0 = deg2rad(h0_deg)
    eq0 = eq_at(0.0); eq1 = eq_at(1.0)
    def interp_ra(t: float):
        a0, a1 = eq0.ra, eq1.ra
        if abs(a1 - a0) > 12:
            if a0 > a1: a1 += 24
            else: a0 += 24
        return (a0 + (a1 - a0)*t) % 24.0
    def interp_dec(t: float): return eq0.dec + (eq1.dec - eq0.dec)*t
    def hour_angle_for(dec: float):
        cosH = (sin(h0) - sinphi*sin(dec)) / (cosphi*cos(dec))
        if cosH < -1 or cosH > 1: return None
        return acos(clamp(cosH, -1, 1))
    theta0 = lst_deg(jd0, lon_deg)
    def m_transit(alpha_hours: float): return frac(((alpha_hours*15.0) - theta0) / 360.0)
    H0_seed = hour_angle_for(eq0.dec)
    if H0_seed is None: return None, None
    m0 = m_transit(eq0.ra)
    m1 = (m0 - rad2deg(H0_seed)/360.0) % 1.0
    m2 = (m0 + rad2deg(H0_seed)/360.0) % 1.0
    for _ in range(iterations):
        alpha0 = interp_ra(m0)
        d1 = interp_dec(m1); d2 = interp_dec(m2)
        H1 = hour_angle_for(d1); H2 = hour_angle_for(d2)
        if H1 is None or H2 is None: return None, None
        m1 = (m_transit(alpha0) - rad2deg(H1)/360.0) % 1.0
        m2 = (m_transit(alpha0) + rad2deg(H2)/360.0) % 1.0
    rise = day_start + timedelta(seconds=m1*86400.0)
    setv = day_start + timedelta(seconds=m2*86400.0)
    return rise, setv

# ---------------- Aggregation & delta ----------------
def moon_info(date_utc: datetime, lat_deg: float, lon_deg: float):
    jd = date_to_jd(date_utc)
    rise, setv = rise_set_for_body(date_utc, lat_deg, lon_deg, lambda jd: moon_equatorial(jd)[0], 0.125)
    illum, age, name = moon_phase_illum_age(jd)
    return rise, setv, age, illum, name

def sun_day_length_delta(date_utc: datetime, lat: float, lon: float):
    sr, ss, now_len = sun_times(date_utc, lat, lon)
    now_len_sec = now_len
    year = date_utc.year; month = date_utc.month
    is_north = lat >= 0.0
    if month <= 6:
        ref_date = datetime(year-1, 12, 21, tzinfo=timezone.utc) if is_north else datetime(year, 6, 21, tzinfo=timezone.utc)
        ref_label = f"vs shortest day ({'Dec 21, '+str(year-1) if is_north else 'Jun 21, '+str(year)})"
    else:
        ref_date = datetime(year, 6, 21, tzinfo=timezone.utc) if is_north else datetime(year, 12, 21, tzinfo=timezone.utc)
        ref_label = f"vs longest day ({'Jun 21, '+str(year) if is_north else 'Dec 21, '+str(year)})"
    _, _, ref_len = sun_times(ref_date, lat, lon)
    return now_len_sec, now_len_sec - ref_len, ref_label

# ---------------- CLI ----------------
def main(argv):
    if len(argv) < 4 or len(argv) > 5:
        print("Usage: python sun_moon_cli_corrected.py <lat> <lon> <YYYY-MM-DD> [tz_offset_hours]")
        print("Example (Copenhagen winter): python sun_moon_cli_corrected.py 55.6761 12.5683 2026-02-04 +1")
        sys.exit(1)
    lat = float(argv[1]); lon = float(argv[2]); date_str = argv[3]
    tz_off = float(argv[4]) if len(argv) == 5 else None
    try:
        y,m,d = map(int, date_str.split("-"))
        date_utc = datetime(y,m,d,tzinfo=timezone.utc)
    except Exception as e:
        print("Invalid date. Use YYYY-MM-DD.", e); sys.exit(1)
    if tz_off is None:
        local_offset = datetime.now().astimezone().utcoffset()
        tz_off = (local_offset.total_seconds()/3600.0) if local_offset else 0.0

    # Sun
    sr, ss, daylen = sun_times(date_utc, lat, lon)
    sunrise = fmt_hm(sr, tz_off); sunset = fmt_hm(ss, tz_off); length = fmt_len(daylen)
    now_len, delta, ref_label = sun_day_length_delta(date_utc, lat, lon)
    change = ("+" if delta>=0 else "-") + fmt_len(abs(delta))

    # Moon
    mr, ms, age, illum, phase = moon_info(date_utc, lat, lon)
    moonrise = fmt_hm(mr, tz_off); moonset = fmt_hm(ms, tz_off)
    age_str = f"{age:.2f} d"; illum_str = f"{illum*100.0:.1f}%"

    print(f"Location: lat {lat:.6f}, lon {lon:.6f}")
    print(f"Date:     {date_str} (TZ offset {tz_off:+.1f}h)")
    print("")
    print("Sun:")
    print(f"  Sunrise         : {sunrise}")
    print(f"  Sunset          : {sunset}")
    print(f"  Length of day   : {length}")
    print(f"  Change in length: {change} {ref_label}")
    print("")
    print("Moon:")
    print(f"  Moonrise        : {moonrise}")
    print(f"  Moonset         : {moonset}")
    print(f"  Age             : {age_str}")
    print(f"  Illumination    : {illum_str}")
    print(f"  Phase           : {phase}")

if __name__ == "__main__":
    main(sys.argv)

# Example (Copenhagen, CET winter):
# python3 sun_and_moon_calcs.py 55.712694 12.541794 2026-02-04 +1