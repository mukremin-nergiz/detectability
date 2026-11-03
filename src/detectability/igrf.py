"""Main geomagnetic field from the International Geomagnetic Reference Field, 14th generation (IGRF-14).

The coefficients in ``data/igrf14coeffs.txt`` are the official IAGA release (NOAA NCEI mirror,
https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt), unchanged. Between epochs the coefficients are interpolated
linearly; after 2025.0 the secular variation is applied, up to 2030.0 (the model's validity limit).

The synthesis follows the standard IGRF/WMM algorithm: geodetic (WGS84) to geocentric conversion, Schmidt
semi-normalised associated Legendre functions, and rotation of the field components back to the geodetic frame.

References
----------
- Alken, P. et al. (2021). International Geomagnetic Reference Field: the thirteenth generation. Earth, Planets and Space
  73, 49. (Method; IGRF-14 uses the same formulation.)
- Chulliat, A. et al. (2020). The US/UK World Magnetic Model for 2020-2025, Technical Report, NOAA NCEI (algorithm and
  Legendre recursion).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np

A_REF_KM = 6371.2                      # IGRF reference radius
WGS84_A_KM = 6378.137
WGS84_B_KM = WGS84_A_KM * (1.0 - 1.0 / 298.257223563)
VALID_FROM, VALID_TO = 1900.0, 2030.0
MODEL = "IGRF-14"


@dataclass(frozen=True)
class MainField:
    """Main field at a point. Components in nT: north (X), east (Y), down (Z); angles in degrees."""

    north: float
    east: float
    down: float
    total: float
    horizontal: float
    inclination: float     # positive downwards (northern hemisphere)
    declination: float     # positive east of geographic north
    model: str = MODEL


@lru_cache(maxsize=1)
def _coefficients():
    text = resources.files("detectability").joinpath("data/igrf14coeffs.txt").read_text()
    epochs: list[float] | None = None
    rows: list[tuple[str, int, int, list[float]]] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or line.startswith("#"):
            continue
        if parts[0] == "g/h":
            epochs = [float(p) for p in parts[3:-1]]  # the last column is the secular variation, e.g. "2025-30"
            continue
        if parts[0] in ("g", "h"):
            rows.append((parts[0], int(parts[1]), int(parts[2]), [float(p) for p in parts[3:]]))
    if epochs is None or not rows:
        raise RuntimeError("IGRF coefficient file not recognised")
    nmax = max(n for _, n, _, _ in rows)
    g = np.zeros((len(epochs), nmax + 1, nmax + 1))
    h = np.zeros_like(g)
    g_sv = np.zeros((nmax + 1, nmax + 1))
    h_sv = np.zeros_like(g_sv)
    for kind, n, m, values in rows:
        main, sv = values[:-1], values[-1]
        (g if kind == "g" else h)[:, n, m] = main
        (g_sv if kind == "g" else h_sv)[n, m] = sv
    return np.array(epochs), g, h, g_sv, h_sv, nmax


def decimal_year(date: _dt.date | _dt.datetime | float) -> float:
    if isinstance(date, (int, float)):
        return float(date)
    start = _dt.date(date.year, 1, 1)
    days = (_dt.date(date.year + 1, 1, 1) - start).days
    day = (date.date() if isinstance(date, _dt.datetime) else date) - start
    return date.year + day.days / days


def _coefficients_at(year: float):
    if not VALID_FROM <= year <= VALID_TO:
        raise ValueError(f"{MODEL} is defined from {VALID_FROM:g} to {VALID_TO:g}; got {year:.3f}")
    epochs, g, h, g_sv, h_sv, nmax = _coefficients()
    if year >= epochs[-1]:
        dt = year - epochs[-1]
        return g[-1] + g_sv * dt, h[-1] + h_sv * dt, nmax
    i = int(np.searchsorted(epochs, year, side="right") - 1)
    w = (year - epochs[i]) / (epochs[i + 1] - epochs[i])
    return (1 - w) * g[i] + w * g[i + 1], (1 - w) * h[i] + w * h[i + 1], nmax


def _schmidt_legendre(nmax: int, cos_t: float, sin_t: float):
    """Schmidt semi-normalised P[n, m](cos theta) and dP/dtheta (Gauss-normalised recursion, then scaling)."""
    P = np.zeros((nmax + 1, nmax + 1))
    dP = np.zeros_like(P)
    P[0, 0] = 1.0
    for n in range(1, nmax + 1):
        for m in range(n + 1):
            if n == m:
                P[n, n] = sin_t * P[n - 1, n - 1]
                dP[n, n] = sin_t * dP[n - 1, n - 1] + cos_t * P[n - 1, n - 1]
            else:
                k = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3)) if n > 1 else 0.0
                P[n, m] = cos_t * P[n - 1, m] - (k * P[n - 2, m] if n > 1 else 0.0)
                dP[n, m] = cos_t * dP[n - 1, m] - sin_t * P[n - 1, m] - (k * dP[n - 2, m] if n > 1 else 0.0)
    S = np.zeros_like(P)
    S[0, 0] = 1.0
    for n in range(1, nmax + 1):
        S[n, 0] = S[n - 1, 0] * (2 * n - 1) / n
        for m in range(1, n + 1):
            S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1) * (2.0 if m == 1 else 1.0) / (n + m))
    return P * S, dP * S


def field(latitude: float, longitude: float, date: _dt.date | _dt.datetime | float, altitude_km: float = 0.0) -> MainField:
    """IGRF-14 main field at geodetic ``latitude``/``longitude`` (degrees), ``altitude_km`` above the WGS84 ellipsoid,
    on ``date`` (a date or a decimal year)."""
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude must be within [-90, 90]")
    g, h, nmax = _coefficients_at(decimal_year(date))

    # geodetic -> geocentric (radius r and colatitude theta)
    lat = np.radians(latitude)
    ct, st = np.sin(lat), np.cos(lat)          # cos and sin of the geodetic colatitude
    a2, b2 = WGS84_A_KM**2, WGS84_B_KM**2
    one, two = a2 * st * st, b2 * ct * ct
    rho = np.sqrt(one + two)
    r = np.sqrt(altitude_km * (altitude_km + 2.0 * rho) + (a2 * one + b2 * two) / (one + two))
    cd = (altitude_km + rho) / r
    sd = (a2 - b2) / rho * ct * st / r
    cos_t, sin_t = ct * cd - st * sd, st * cd + ct * sd
    sin_t = max(sin_t, 1e-10)                  # avoid division by zero exactly at the poles

    P, dP = _schmidt_legendre(nmax, cos_t, sin_t)
    phi = np.radians(longitude)
    x = y = z = 0.0
    for n in range(1, nmax + 1):
        ratio = (A_REF_KM / r) ** (n + 2)
        for m in range(n + 1):
            cm, sm = np.cos(m * phi), np.sin(m * phi)
            gh = g[n, m] * cm + h[n, m] * sm
            x += ratio * gh * dP[n, m]
            y += ratio * m * (g[n, m] * sm - h[n, m] * cm) * P[n, m] / sin_t
            z -= ratio * (n + 1) * gh * P[n, m]

    # rotate back from geocentric to geodetic north/down
    north, down = x * cd + z * sd, z * cd - x * sd
    horizontal = float(np.hypot(north, y))
    total = float(np.sqrt(horizontal**2 + down**2))
    return MainField(
        north=float(north),
        east=float(y),
        down=float(down),
        total=total,
        horizontal=horizontal,
        inclination=float(np.degrees(np.arctan2(down, horizontal))),
        declination=float(np.degrees(np.arctan2(y, north))),
    )
