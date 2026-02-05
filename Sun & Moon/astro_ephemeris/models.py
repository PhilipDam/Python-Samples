from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeoPosition:
    """
    C# equivalent: public readonly struct GeoPosition
    """
    Latitude: float
    Longitude: float


@dataclass(slots=True)
class EquatorialCoordinates:
    """
    C# equivalent: public struct EquatorialCoordinates
    RightAscension, Declination are in radians in this library (matching your C# usage).
    """
    RightAscension: float
    Declination: float


@dataclass(slots=True)
class EclipticCoordinates:
    """
    C# equivalent: public struct EclipticCoordinates
    Lambda is in degrees in the lunar module (matching the C#), Beta is in degrees.
    """
    Lambda: float
    Beta: float


@dataclass(slots=True)
class FundamentalArguments:
    """
    C# equivalent: public struct FundamentalArguments
    All arguments in degrees (matching the C#).
    """
    L: float
    D: float
    M: float
    M_moon: float
    F: float
