"""Small-area geodetic coordinate conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math


EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class GeoOrigin:
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0

    @property
    def lat_rad(self) -> float:
        return math.radians(self.lat_deg)


def latlon_to_enu(lat_deg, lon_deg, alt_m, origin: GeoOrigin):
    """Convert WGS84 latitude/longitude near ``origin`` to local ENU meters."""

    import numpy as np

    lat = np.asarray(lat_deg, dtype="float64")
    lon = np.asarray(lon_deg, dtype="float64")
    alt = np.asarray(alt_m, dtype="float64")
    east = np.radians(lon - origin.lon_deg) * EARTH_RADIUS_M * math.cos(origin.lat_rad)
    north = np.radians(lat - origin.lat_deg) * EARTH_RADIUS_M
    up = alt - origin.alt_m
    return east, north, up


def enu_to_latlon(east_m, north_m, up_m, origin: GeoOrigin):
    """Convert local ENU meters near ``origin`` back to WGS84 lat/lon."""

    import numpy as np

    east = np.asarray(east_m, dtype="float64")
    north = np.asarray(north_m, dtype="float64")
    up = np.asarray(up_m, dtype="float64")
    lat = origin.lat_deg + np.degrees(north / EARTH_RADIUS_M)
    lon = origin.lon_deg + np.degrees(east / (EARTH_RADIUS_M * math.cos(origin.lat_rad)))
    alt = origin.alt_m + up
    return lat, lon, alt


def enu_to_ned(east_m, north_m, up_m):
    """Return NED components from ENU components."""

    return north_m, east_m, -up_m
