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

@dataclass(frozen=True)
class Instrument:
    """An instrument preset: its noise, what the figure means and where it comes from (checked 4 Nov 2026)."""

    method: str          # "gravity", "magnetic" or "gradient" (magnetic vertical gradient)
    noise: float         # one standard deviation of a reading, in µGal (gravity) or nT (magnetic and gradient)
    basis: str           # what the manufacturer's figure is
    reference: str       # instrument the figure is taken from
    source: str          # data sheet or publication
    baseline: float | None = None   # m, vertical distance between the two sensors (gradiometers)


INSTRUMENTS: dict[str, Instrument] = {
    "Scintrex CG-6": Instrument(
        "gravity", 5.0, "repeatability, standard deviation < 5 µGal", "Scintrex CG-6 Autograv",
        "https://microglacoste.com/wp-content/uploads/2019/01/CG-6-Brochure_R5.pdf"),
    "Scintrex CG-5": Instrument(
        "gravity", 5.0, "standard deviation < 5 µGal (1 µGal resolution)", "Scintrex CG-5 Autograv",
        "https://dmliefer.ru/sites/default/files/cg5.pdf"),
    "LaCoste & Romberg G": Instrument(
        "gravity", 20.0, "manufacturer's repeatability 10-20 µGal; the upper value is used", "LaCoste & Romberg Model G",
        "https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/8th_CISBGf/arq_938.pdf"),
    "proton precession": Instrument(
        "magnetic", 0.1, "resolution and sensitivity 0.1 nT", "Geometrics G-856AX",
        "https://www.aseg.org.au/public/200/files/digital-library-files/856_DataSheet.pdf"),
    "Overhauser": Instrument(
        "magnetic", 0.022, "sensitivity 0.022 nT/√Hz, i.e. at 1 Hz", "GEM Systems GSM-19 v7",
        "https://www.aseg.org.au/public/200/files/digital-library-files/GSM-19%20Overhauser%20v7_0%202%20Pag.pdf"),
    "caesium vapour": Instrument(
        "magnetic", 0.008, "noise < 0.008 nT/√Hz RMS, i.e. at 1 Hz", "Geometrics G-858",
        "https://www.geomatrix.co.uk/land-products/magnetic/g-858/"),
    # Vertical gradiometers: two sensors one above the other; the reading is the lower minus the upper sensor, in nT
    # over the baseline (as Bartington and Foerster report it; nT/m only for a 1 m baseline).
    "Bartington Grad601": Instrument(
        "gradient", 0.1, "noise 100 pT peak-to-peak maximum, taken as one standard deviation to be conservative",
        "Bartington Grad-01-1000L sensor (Grad601)", "https://www.bartingtondownloads.com/wp-content/uploads/DS0060.pdf",
        baseline=1.0),
    "Sensys FGM650": Instrument(
        "gradient", 0.04, "noise < 40 pT rms/√Hz at 1 Hz", "Sensys FGM650 fluxgate gradiometer",
        "https://sensysmagnetometer.com/de/produkte/fgm650-gradiometer/", baseline=0.65),
    "Foerster FEREX 4.034": Instrument(
        "gradient", 0.04, "noise < 40 pT/√Hz at 1 Hz (MG-10 probe)", "Foerster FEREX 4.034 with MG-10-550 probe",
        "https://www.foerstergroup.com/FOERSTER/downloads/product/ferex/FOERSTER_FLY_FEREX4.034_DE.pdf", baseline=0.55),
    "GEM GSM-19G": Instrument(
        "gradient", 0.031, "0.022 nT at 1 Hz per sensor; two sensors in difference give 0.022 × √2",
        "GEM Systems GSM-19G Overhauser gradiometer (one staff section, 0.56 m, per the manual)",
        "https://www.gemsys.ca/wp-content/themes/gemsystems/pdf/GEM_Overhauser_GSM_19_fin.pdf", baseline=0.56),
}

GRAVIMETERS_UGAL: dict[str, float] = {k: v.noise for k, v in INSTRUMENTS.items() if v.method == "gravity"}
MAGNETOMETERS_NT: dict[str, float] = {k: v.noise for k, v in INSTRUMENTS.items() if v.method == "magnetic"}
GRADIOMETERS_NT: dict[str, float] = {k: v.noise for k, v in INSTRUMENTS.items() if v.method == "gradient"}

# Field noise is what remains after corrections and usually dominates the instrument noise.
DEFAULT_FIELD_NOISE_UGAL = 10.0  # microgravity: tares, drift residuals, terrain, cultural noise
DEFAULT_FIELD_NOISE_NT = 2.0     # quiet rural site after diurnal correction; urban and industrial sites are far noisier
# Gradient surveys cancel diurnal variation and the regional field, so their field noise is mainly soil, heading and
# walking noise. 0.5 nT is an assumption, not a published figure (the Grad601 manual asks for a set-up spot that varies
# by less than about 0.5 nT over 1 m); change it for the site.
DEFAULT_FIELD_NOISE_GRADIENT_NT = 0.5

UNITS = {"gravity": "µGal", "magnetic": "nT", "gradient": "nT", "gpr": "× detection limit", "ert": "%"}


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
    if method in ("magnetic", "gradient"):
        assumptions.append("Peak is the largest absolute value of the (often two-lobed) anomaly; induced "
                           "magnetisation only, remanence is not included.")
    return Assessment(method, peak, float(x[peak_i]), width, shape, noise, snr, verdict, spacing, length, assumptions,
                      unit)
