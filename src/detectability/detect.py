"""From a modelled profile to a detectability assessment: signal-to-noise ratio, verdict, station spacing and profile
length, with the assumptions behind them.

The thresholds are conventions, not physics, and are stated in every result:
- signal-to-noise ratio (peak anomaly / effective noise) >= 5: clearly detectable; 2-5: marginal; < 2: not detectable;
- at least ``STATIONS_ACROSS`` stations across the anomaly width so its shape (and so depth) can be resolved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

CLEAR, MARGINAL = 5.0, 2.0
STATIONS_ACROSS = 5

# Instrument repeatability in µGal (manufacturer specifications; to be verified against current data sheets).
GRAVIMETERS_UGAL: dict[str, float] = {
    "Scintrex CG-6": 5.0,
    "Scintrex CG-5": 5.0,
    "LaCoste & Romberg G": 10.0,
}
DEFAULT_FIELD_NOISE_UGAL = 10.0  # typical microgravity field noise (tares, drift residuals, terrain, cultural noise)

# Magnetometer sensitivity in nT by sensor class (typical values; to be verified against current data sheets). Field noise
# (diurnal residuals after base-station correction, heading error, near-surface geology, cultural noise) usually dominates.
MAGNETOMETERS_NT: dict[str, float] = {
    "proton precession": 0.2,
    "Overhauser": 0.05,
    "caesium vapour": 0.01,
    "fluxgate gradiometer": 0.1,
}
DEFAULT_FIELD_NOISE_NT = 2.0     # quiet rural site after diurnal correction; urban and industrial sites are far noisier

UNITS = {"gravity": "µGal", "magnetic": "nT"}


def effective_noise(instrument: float, field_noise: float) -> float:
    """Instrument and field noise added in quadrature."""
    return float(np.hypot(instrument, field_noise))


def _nice(value: float) -> float:
    """Round down to 1, 2 or 5 times a power of ten (practical station spacings)."""
    if value <= 0:
        return 0.0
    exp = np.floor(np.log10(value))
    for m in (5.0, 2.0, 1.0):
        if m * 10**exp <= value:
            return float(m * 10**exp)
    return float(10 ** (exp - 1) * 5)


@dataclass
class Assessment:
    method: str
    peak: float                 # largest |anomaly| (bell) or step amplitude, in ``units``
    peak_at: float              # position of the peak, m
    width: float                # full width at half maximum (bell) or 25-75 % transition width (step), m
    shape: str                  # "bell", "step" or "none" (no contrast)
    noise: float                # effective noise, same units as peak
    snr: float
    verdict: str                # "clearly detectable" | "marginal" | "not detectable"
    station_spacing: float      # recommended maximum spacing, m
    profile_length: float       # recommended minimum profile length, m
    assumptions: list[str] = field(default_factory=list)
    units: str = "µGal"

    def to_dict(self) -> dict:
        return asdict(self)


def _width(x: np.ndarray, a: np.ndarray, shape: str) -> float:
    if shape == "bell":
        i = int(np.argmax(a))
        half = a[i] / 2.0
        above = np.where(a >= half)[0]
        left, right = above.min(), above.max()
        # linear interpolation at both half-maximum crossings
        xl = x[left] if left == 0 else np.interp(half, [a[left - 1], a[left]], [x[left - 1], x[left]])
        xr = x[right] if right == len(x) - 1 else np.interp(half, [a[right + 1], a[right]], [x[right + 1], x[right]])
        return float(xr - xl)
    lo, hi = a.min(), a.max()
    order = np.argsort(x)
    xs, as_ = x[order], a[order]
    increasing = as_[-1] >= as_[0]
    ref = as_ if increasing else as_[::-1]
    xr = xs if increasing else xs[::-1]
    x25 = np.interp(lo + 0.25 * (hi - lo), ref, xr)
    x75 = np.interp(lo + 0.75 * (hi - lo), ref, xr)
    return float(abs(x75 - x25))


def assess(x, anomaly, noise: float, depth: float, method: str = "gravity") -> Assessment:
    """Assess a modelled profile ``anomaly`` (already in the units of ``noise``) along ``x``.

    ``depth`` is the depth to the top (or centre) of the target, used for the profile length.
    """
    x = np.asarray(x, float)
    g = np.asarray(anomaly, float)
    a = np.abs(g)
    unit = UNITS.get(method, "")
    if not np.any(a > 0):
        return Assessment(method, 0.0, 0.0, 0.0, "none", noise, 0.0, "not detectable", 0.0, 0.0,
                          ["No physical-property contrast between target and host for this method."], unit)
    peak_i = int(np.argmax(a))
    peak = float(a[peak_i])
    shape = "bell" if max(a[0], a[-1]) < 0.1 * peak else "step"
    if shape == "step":
        peak = float(g.max() - g.min())  # the step amplitude is what the survey measures
    width = _width(x, a, shape)
    snr = peak / noise if noise > 0 else float("inf")
    verdict = "clearly detectable" if snr >= CLEAR else "marginal" if snr >= MARGINAL else "not detectable"
    spacing = _nice(width / STATIONS_ACROSS)
    length = float(max(6.0 * depth, 4.0 * width))
    assumptions = [
        f"Signal-to-noise thresholds: >= {CLEAR:g} clearly detectable, {MARGINAL:g}-{CLEAR:g} marginal, < {MARGINAL:g} not detectable.",
        f"Effective noise {noise:.3g} {unit} (instrument and field noise added in quadrature).",
        f"Station spacing allows at least {STATIONS_ACROSS} stations across the anomaly width ({width:.3g} m).",
        "Flat topography, homogeneous host and no regional trend; terrain and regional effects reduce the margin.",
    ]
    if method == "magnetic":
        assumptions.append("Peak is the largest absolute value of the (often two-lobed) anomaly; no remanence unless given.")
    return Assessment(method, peak, float(x[peak_i]), width, shape, noise, snr, verdict, spacing, length, assumptions,
                      unit)
