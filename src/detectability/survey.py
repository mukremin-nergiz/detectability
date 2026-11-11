"""One question, one answer: describe a target and a survey, get the modelled response and a detectability assessment
for each method, with every input needed to reproduce it.

    >>> from detectability import evaluate
    >>> report = evaluate({"shape": "sphere", "depth_top": 15, "size": 20, "material": "air", "host": "limestone"},
    ...                   {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"})
    >>> print(report.summary())

The website, the ChatGPT/Claude app and the generated scripts all call this function, so they give the same numbers.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from . import __version__, detect, ert, gpr, gravity, igrf, magnetics, rocks
from .constants import SI_TO_UGAL

METHODS = ("gravity", "magnetic", "gradient", "gpr", "ert")   # gradient: magnetic vertical gradient; gpr: ground-penetrating
# radar; ert: electrical resistivity tomography (needs pyGIMLi or an ERT service)
DEFAULT_METHODS = ("gravity", "magnetic")

SHAPES: dict[str, str] = {
    "sphere": "Sphere (cavity, boulder, buried object). size = diameter.",
    "horizontal_cylinder": "Horizontal cylinder crossing the profile at right angles (tunnel, pipe, gallery). size = diameter.",
    "vertical_cylinder": "Vertical cylinder (shaft, well, sinkhole pipe). size = diameter; needs depth_bottom.",
    "prism": "Rectangular block (room, foundation, tank). size = width along the profile, length = extent across it "
             "(default: size); needs depth_bottom.",
    "dyke": "Dyke, wall or filled trench crossing the profile. size = width; needs depth_bottom; optional dip.",
    "contact": "Vertical contact (fault, edge of a fill or of a rock unit); the target material lies on the +x side. "
               "Needs depth_bottom; size is not used.",
}
NEEDS_BOTTOM = {"vertical_cylinder", "prism", "dyke", "contact"}
NEEDS_SIZE = set(SHAPES) - {"contact"}
TWO_D = {"horizontal_cylinder", "dyke", "contact"}


@dataclass
class Target:
    shape: str
    depth_top: float                      # m below ground to the top of the target
    size: float | None = None             # m, see SHAPES
    depth_bottom: float | None = None     # m, see NEEDS_BOTTOM
    length: float | None = None           # m, prism extent across the profile
    dip: float = 90.0                     # degrees from horizontal, dyke only
    material: str = "air"                 # what the target is made of (fill), see rocks
    host: str = "limestone"               # surrounding material
    density_contrast: float | None = None          # kg/m^3, overrides the material tables
    susceptibility_contrast: float | None = None   # SI, overrides the material tables
    permittivity: float | None = None              # relative, of the target (GPR); overrides the material table
    conductivity: float | None = None              # mS/m, of the target (GPR)
    host_permittivity: float | None = None         # relative, of the host (GPR)
    host_conductivity: float | None = None         # mS/m, of the host (GPR)
    resistivity: float | None = None               # ohm m, of the target (ERT); overrides the material table
    host_resistivity: float | None = None          # ohm m, of the host (ERT)

    def validate(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"unknown shape {self.shape!r}; choose from {', '.join(SHAPES)}")
        if self.depth_top < 0:
            raise ValueError("depth_top must be zero or positive")
        if self.shape in NEEDS_SIZE and not (self.size and self.size > 0):
            raise ValueError(f"{self.shape} needs a positive size")
        if self.shape in NEEDS_BOTTOM and not (self.depth_bottom and self.depth_bottom > self.depth_top):
            raise ValueError(f"{self.shape} needs depth_bottom greater than depth_top")
        if self.shape in ("sphere", "horizontal_cylinder") and self.depth_top <= 0:
            raise ValueError(f"{self.shape} must be buried: depth_top > 0")


@dataclass
class Survey:
    gravimeter: str = "Scintrex CG-6"
    gravity_field_noise: float = detect.DEFAULT_FIELD_NOISE_UGAL     # µGal
    gravity_instrument_noise: float | None = None                     # µGal, overrides the gravimeter preset
    magnetometer: str = "Overhauser"
    magnetic_field_noise: float = detect.DEFAULT_FIELD_NOISE_NT       # nT
    magnetic_instrument_noise: float | None = None                    # nT, overrides the magnetometer preset
    sensor_height: float = 0.5            # m, magnetometer sensor above ground
    gradiometer: str = "Bartington Grad601"
    gradient_field_noise: float = detect.DEFAULT_FIELD_NOISE_GRADIENT_NT   # nT, of the gradiometer reading
    gradient_instrument_noise: float | None = None                         # nT, overrides the gradiometer preset
    gradiometer_height: float = 0.2       # m, lower sensor above ground (Bartington: about 20 cm in normal use)
    gradiometer_baseline: float | None = None   # m between the sensors; default: the gradiometer preset's
    profile_azimuth: float = 0.0          # degrees east of north; 2D targets strike at right angles to it
    latitude: float | None = None         # for the main field (IGRF-14); needed for magnetics
    longitude: float | None = None
    date: str | None = None               # ISO date of the survey; default: today
    gpr_frequency: float | None = None    # MHz, antenna centre frequency; default: from the target depth
    gpr_system_performance: float = gpr.DEFAULT_SYSTEM_PERFORMANCE_DB   # dB, largest two-way loss the system records
    ert_array: str = ert.DEFAULT_ARRAY    # dipole-dipole, wenner, schlumberger or gradient
    ert_electrodes: int = ert.DEFAULT_ELECTRODES
    ert_spacing: float | None = None      # m between electrodes; default: spread about six times the target depth
    ert_noise: float = ert.DEFAULT_NOISE_PERCENT   # % of each apparent resistivity
    ert_invert: bool = True               # also invert noisy synthetic data to see whether the target is recovered
    field_total: float | None = None      # nT; main field given directly instead of IGRF (all three together)
    field_inclination: float | None = None
    field_declination: float | None = None


@dataclass
class MethodResult:
    method: str
    available: bool
    reason: str = ""
    contrast: float | None = None
    contrast_units: str = ""
    instrument: str = ""
    instrument_noise: float | None = None
    field_noise: float | None = None
    main_field: dict | None = None
    x: list[float] = field(default_factory=list)
    anomaly: list[float] = field(default_factory=list)
    assessment: detect.Assessment | None = None
    details: dict | None = None           # method-specific figures (GPR: frequency, velocity, attenuation, ...)


@dataclass
class Report:
    version: str
    target: dict
    survey: dict
    methods: list[str]
    results: list[MethodResult]
    recommended: str | None

    def result(self, method: str) -> MethodResult:
        return next(r for r in self.results if r.method == method)

    def to_dict(self, profile: bool = True) -> dict:
        d = asdict(self)
        if not profile:
            for r in d["results"]:
                r.pop("x"), r.pop("anomaly")
        return d

    def summary(self) -> str:
        lines = [f"Detectability {self.version}: {self.target['shape']} in {self.target['host']}"]
        for r in self.results:
            if not r.available:
                lines.append(f"- {r.method}: not assessed ({r.reason})")
                continue
            a = r.assessment
            if r.method == "gpr":
                d = r.details
                margin = 20 * np.log10(a.snr) if a.snr > 0 else float("-inf")
                lines.append(f"- gpr ({d['frequency_mhz']:g} MHz): echo {margin:+.1f} dB against the detection limit, "
                             f"S/N {a.snr:.3g}: {a.verdict}")
            else:
                lines.append(f"- {r.method}: peak {a.peak:.3g} {a.units}, noise {a.noise:.3g} {a.units}, "
                             f"S/N {a.snr:.3g}: {a.verdict}")
            if a.verdict != "not detectable":
                lines.append(f"  stations every {a.station_spacing:g} m or less, profile at least "
                             f"{a.profile_length:.0f} m long")
        lines.append(f"Recommended method: {self.recommended}" if self.recommended
                     else "No method is expected to detect this target with these assumptions.")
        return "\n".join(lines)

    def script(self) -> str:
        """A standalone Python script that recomputes this report."""
        target = {k: v for k, v in self.target.items() if v is not None}
        if target.get("shape") != "dyke":
            target.pop("dip", None)
        survey = {k: v for k, v in self.survey.items() if v is not None}
        for prefix, method in (("gpr_", "gpr"), ("ert_", "ert")):   # settings of methods not asked for add nothing
            if method not in self.methods:
                survey = {k: v for k, v in survey.items() if not k.startswith(prefix)}

        def literal(d):  # JSON text, but Python booleans
            return re.sub(r"(?<=: )(true|false)(?=,?\n)", lambda m: m.group(1).capitalize(),
                          json.dumps(d, indent=4, ensure_ascii=False))
        return SCRIPT.format(
            version=self.version,
            target=literal(target),
            survey=literal(survey),
            methods=json.dumps(self.methods),
        )


SCRIPT = '''"""Reproduces a Detectability result (detectability {version}).

Install: pip install "detectability=={version}"
"""
import detectability
from detectability import evaluate

target = {target}

survey = {survey}

report = evaluate(target, survey, methods={methods})
print(report.summary())
if detectability.__version__ != "{version}":
    print("Note: computed with detectability", detectability.__version__, "- the original used {version}")

# Optional plot (needs matplotlib)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
if plt:
    shown = [r for r in report.results if r.available]
    fig, axes = plt.subplots(len(shown), 1, figsize=(8, 3 * len(shown)), squeeze=False)
    for ax, r in zip(axes[:, 0], shown):
        ax.plot(r.x, r.anomaly, color="black")
        ax.axhspan(-r.assessment.noise, r.assessment.noise, color="grey", alpha=0.2, label="noise")
        ax.set(xlabel="distance along profile (m)", ylabel=f"{{r.method}} ({{r.assessment.units}})")
        ax.legend()
    plt.tight_layout()
    plt.show()
'''


def _as(cls, value):
    if value is None:
        return cls() if cls is Survey else None
    if isinstance(value, cls):
        return replace(value)
    known = {f.name for f in fields(cls)}
    unknown = set(value) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__.lower()} field(s): {', '.join(sorted(unknown))}")
    return cls(**value)


def _profile(t: Target) -> np.ndarray:
    size = t.size or 0.0
    if t.shape == "contact":
        ref = t.depth_bottom
    elif t.shape in ("sphere", "horizontal_cylinder"):
        ref = t.depth_top + size / 2
    else:
        ref = (t.depth_top + t.depth_bottom) / 2
    half = max(8.0 * ref, 5.0 * size, 20.0)
    return np.linspace(-half, half, 1201)


def _contrast(t: Target, method: str):
    try:
        if method == "gravity":
            value = t.density_contrast if t.density_contrast is not None else rocks.contrast(t.material, t.host)
            return value, "kg/m^3", ""
        value = t.susceptibility_contrast if t.susceptibility_contrast is not None else \
            rocks.susceptibility_contrast(t.material, t.host)
        return value, "SI", ""
    except KeyError as e:
        return None, "", str(e.args[0]) + " (give the contrast directly)"


def _gravity(t: Target, x, drho):
    r = (t.size or 0) / 2
    if t.shape == "sphere":
        return gravity.sphere(x, t.depth_top + r, r, drho)
    if t.shape == "horizontal_cylinder":
        return gravity.horizontal_cylinder(x, t.depth_top + r, r, drho)
    if t.shape == "vertical_cylinder":
        return gravity.vertical_cylinder(x, t.depth_top, t.depth_bottom, r, drho)
    if t.shape == "prism":
        a, c = t.size / 2, (t.length or t.size) / 2
        return gravity.prism(x, 0.0, -a, a, -c, c, t.depth_top, t.depth_bottom, drho)
    if t.shape == "dyke":
        return gravity.dyke(x, t.depth_top, t.depth_bottom, t.size, drho, dip=t.dip)
    return gravity.vertical_step(x, t.depth_top, t.depth_bottom, drho)


def _magnetic(t: Target, x, chi, amb, s: Survey):
    r = (t.size or 0) / 2
    kw = dict(azimuth=s.profile_azimuth, height=s.sensor_height)
    if t.shape == "sphere":
        return magnetics.sphere(x, t.depth_top + r, r, chi, amb, **kw)
    if t.shape == "horizontal_cylinder":
        return magnetics.horizontal_cylinder(x, t.depth_top + r, r, chi, amb, **kw)
    if t.shape == "vertical_cylinder":
        return magnetics.vertical_cylinder(x, t.depth_top, t.depth_bottom, r, chi, amb, **kw)
    if t.shape == "prism":
        a, c = t.size / 2, (t.length or t.size) / 2
        return magnetics.prism(x, -a, a, -c, c, t.depth_top, t.depth_bottom, chi, amb, **kw)
    if t.shape == "dyke":
        return magnetics.dyke(x, t.depth_top, t.depth_bottom, t.size, chi, amb, dip=t.dip, **kw)
    return magnetics.contact(x, t.depth_top, t.depth_bottom, chi, amb, **kw)


def _main_field(s: Survey):
    given = (s.field_total, s.field_inclination, s.field_declination)
    if all(v is not None for v in given):
        return magnetics.Ambient(*given), {"source": "given", "total": given[0], "inclination": given[1],
                                           "declination": given[2]}
    if s.latitude is None or s.longitude is None:
        return None, None
    f = igrf.field(s.latitude, s.longitude, _dt.date.fromisoformat(s.date))
    return magnetics.Ambient.from_igrf(f), {"source": f.model, "total": round(f.total, 1),
                                           "inclination": round(f.inclination, 3), "declination": round(f.declination, 3)}


def _gradient(t: Target, x, chi, amb, s: Survey, baseline: float):
    """Gradiometer reading: total-field anomaly at the lower sensor minus that at the upper sensor, in nT."""
    low, high = replace(s, sensor_height=s.gradiometer_height), replace(s, sensor_height=s.gradiometer_height + baseline)
    return _magnetic(t, x, chi, amb, low) - _magnetic(t, x, chi, amb, high)


def _gpr_medium(name: str, permittivity, conductivity):
    """Medium from given properties or the GPR table; None for a metal (perfect conductor)."""
    if permittivity is not None:
        return gpr.Medium(permittivity, (conductivity or 0.0) / 1000.0)
    if name in gpr.METALS:
        return None
    m = gpr.medium(name)
    return m if conductivity is None else gpr.Medium(m.permittivity, conductivity / 1000.0)


def _gpr(t: Target, s: Survey, x) -> MethodResult:
    """Ground-penetrating radar: echo of the target's top surface against the system's detection limit."""
    if t.shape == "contact":
        return MethodResult("gpr", False, "a vertical contact has no horizontal surface to reflect the radar wave; "
                            "GPR sees it only as a diffraction at its top edge, which is not modelled")
    if t.depth_top <= 0:
        return MethodResult("gpr", False, "GPR needs the target below the surface (depth_top > 0)")
    try:
        if t.host in gpr.METALS:
            raise KeyError(f"the host cannot be a metal for GPR")
        host = _gpr_medium(t.host, t.host_permittivity, t.host_conductivity)
        target = _gpr_medium(t.material, t.permittivity, t.conductivity)
    except KeyError as e:
        return MethodResult("gpr", False, str(e.args[0]) + " (give permittivity and conductivity directly)")
    if s.gpr_system_performance <= 0 or (s.gpr_frequency is not None and s.gpr_frequency <= 0):
        raise ValueError("gpr_frequency and gpr_system_performance must be positive")

    f_mhz = s.gpr_frequency or gpr.frequency_for_depth(t.depth_top)
    f = f_mhz * 1e6
    R = gpr.reflection(host, target, f)
    lam = host.wavelength(f)
    v = host.velocity(f)
    a = (t.size or 0) / 2
    if t.shape in ("sphere", "horizontal_cylinder"):
        centre = t.depth_top + a
        r = np.hypot(x, centre) - a
        factor = gpr.size_factor("sphere" if t.shape == "sphere" else "cylinder", r, lam, a=a)
    else:
        r = np.hypot(x, t.depth_top)
        if t.shape == "vertical_cylinder":
            factor = gpr.size_factor("disc", r, lam, a=a)
        elif t.shape == "prism":
            factor = gpr.size_factor("rectangle", r, lam, width=t.size, length=t.length or t.size)
        else:
            factor = gpr.size_factor("strip", r, lam, width=t.size)
    amplitude = gpr.echo(r, host, R, factor, f)
    floor = 10 ** (-s.gpr_system_performance / 20)
    snr_profile = amplitude / floor
    a_ = detect.assess(x, snr_profile, 1.0, depth=t.depth_top, method="gpr")
    nyquist = lam / 4
    a_.station_spacing = detect._nice(min(a_.width / detect.STATIONS_ACROSS, nyquist))
    t0 = gpr.two_way_time_ns(t.depth_top, v)
    window = 1.3 * gpr.two_way_time_ns(max(t.depth_top + (t.size or 0), t.depth_top), v)
    alpha = float(host.attenuation_db(f))
    k_a = 2 * np.pi * a / lam
    a_.assumptions = [
        f"Signal-to-noise thresholds: >= {detect.CLEAR:g} clearly detectable, {detect.MARGINAL:g}-{detect.CLEAR:g} "
        f"marginal, < {detect.MARGINAL:g} not detectable; S/N is the echo over the system's detection limit.",
        f"System performance {s.gpr_system_performance:g} dB (largest two-way loss recorded, referred to 1 m; "
        "typical systems 90-130 dB). It includes antenna gain and coupling; clutter and noise at the site lower it.",
        f"Antenna {f_mhz:g} MHz" + ("" if s.gpr_frequency else " (chosen from the target depth)") +
        f"; host: relative permittivity {host.permittivity:g}, conductivity {host.conductivity * 1000:g} mS/m, "
        f"velocity {v / 1e9:.3g} m/ns, attenuation {alpha:.3g} dB/m (one way).",
        "Reflection coefficient " + ("-1 (metal)" if target is None else f"{abs(R):.3g} (normal incidence, "
                                     f"target permittivity {target.permittivity:g}, conductivity "
                                     f"{target.conductivity * 1000:g} mS/m)") + ".",
        "Homogeneous host with frequency-independent properties; real soils lose more at high frequency, so wet or "
        "clayey ground above ~500 MHz is optimistic.",
        f"Trace spacing at most a quarter wavelength in the host ({nyquist:.3g} m) and {detect.STATIONS_ACROSS} traces "
        "across the echo.",
        f"Two-way time to the target top {t0:.3g} ns; record at least {window:.3g} ns (30 % margin).",
    ]
    if t.shape in ("sphere", "horizontal_cylinder"):
        a_.assumptions.append("Curved target: geometric optics" + (f"; small against the wavelength (k a = {k_a:.2g}), "
                              "Rayleigh scaling applied" if k_a < 1 else "") + ".")
    else:
        a_.assumptions.append("Echo of the flat top surface; a top smaller than the first Fresnel zone returns the "
                              "fraction of the zone it covers.")
    if (t.size or 0) < nyquist:
        a_.assumptions.append(f"The target ({t.size:g} m) is smaller than the vertical resolution (a quarter "
                              f"wavelength, {nyquist:.3g} m): it shows as a diffraction hyperbola, its size is not resolved.")
    if t.shape in TWO_D:
        a_.assumptions.append("2D target: infinitely long along strike, at right angles to the profile.")
    details = {"frequency_mhz": float(f_mhz), "velocity_m_per_ns": float(v / 1e9), "attenuation_db_per_m": alpha,
               "wavelength_m": float(lam), "reflection_coefficient": float(abs(R)), "two_way_time_ns": float(t0),
               "time_window_ns": float(window), "vertical_resolution_m": float(nyquist),
               "margin_db": float(20 * np.log10(a_.snr)) if a_.snr > 0 else None,
               "system_performance_db": s.gpr_system_performance}
    return MethodResult("gpr", True, contrast=abs(R), contrast_units="reflection coefficient",
                        instrument=f"{f_mhz:g} MHz antenna", x=x.round(4).tolist(), anomaly=snr_profile.tolist(),
                        assessment=a_, details=details)


def _ert(t: Target, s: Survey) -> MethodResult:
    """Electrical resistivity tomography: locally with pyGIMLi, else through DETECTABILITY_ERT_URL, else not available."""
    import os
    try:
        if ert.available():
            d = ert.run(t, s)
        elif os.environ.get("DETECTABILITY_ERT_URL"):
            try:
                d = ert.remote(asdict(t), asdict(s), os.environ["DETECTABILITY_ERT_URL"])
            except Exception as e:  # the service is down or too slow: say so instead of failing the whole report
                return MethodResult("ert", False, f"the ERT service did not answer ({type(e).__name__}); try again later")
            if not d.get("available", False):
                return MethodResult("ert", False, d.get("reason", "the ERT service could not compute this target"))
        else:
            return MethodResult("ert", False, 'needs pyGIMLi: pip install "detectability[ert]"')
    except KeyError as e:
        return MethodResult("ert", False, str(e.args[0]) + " (give resistivity and host_resistivity directly)")
    a = d.pop("assessment")
    d["assessment"] = detect.Assessment(**a)
    return MethodResult(**d)


def evaluate(target, survey=None, methods=DEFAULT_METHODS) -> Report:
    """Model ``target`` (a ``Target`` or dict) for each method and assess it for ``survey`` (a ``Survey`` or dict)."""
    t, s = _as(Target, target), _as(Survey, survey)
    t.validate()
    if s.date is None:
        s.date = _dt.date.today().isoformat()
    methods = list(methods)
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(sorted(unknown))}; choose from {', '.join(METHODS)}")

    x = _profile(t)
    results = []
    for method in methods:
        if method == "gpr":
            results.append(_gpr(t, s, x))
            continue
        if method == "ert":
            results.append(_ert(t, s))
            continue
        contrast, units, problem = _contrast(t, method)
        if contrast is None:
            results.append(MethodResult(method, False, problem))
            continue
        if method == "gravity":
            inst = s.gravity_instrument_noise if s.gravity_instrument_noise is not None else \
                detect.GRAVIMETERS_UGAL.get(s.gravimeter)
            if inst is None:
                results.append(MethodResult(method, False, f"unknown gravimeter {s.gravimeter!r}"))
                continue
            noise = detect.effective_noise(inst, s.gravity_field_noise)
            anomaly = _gravity(t, x, contrast) * SI_TO_UGAL
            res = MethodResult(method, True, contrast=contrast, contrast_units=units, instrument=s.gravimeter,
                               instrument_noise=inst, field_noise=s.gravity_field_noise)
        else:
            amb, info = _main_field(s)
            if amb is None:
                results.append(MethodResult(method, False, "needs the survey location (latitude, longitude) for the "
                                            "main field, or field_total/inclination/declination"))
                continue
            if method == "magnetic":
                inst = s.magnetic_instrument_noise if s.magnetic_instrument_noise is not None else \
                    detect.MAGNETOMETERS_NT.get(s.magnetometer)
                if inst is None:
                    results.append(MethodResult(method, False, f"unknown magnetometer {s.magnetometer!r}"))
                    continue
                noise = detect.effective_noise(inst, s.magnetic_field_noise)
                anomaly = _magnetic(t, x, contrast, amb, s)
                res = MethodResult(method, True, contrast=contrast, contrast_units=units, instrument=s.magnetometer,
                                   instrument_noise=inst, field_noise=s.magnetic_field_noise, main_field=info)
            else:
                preset = detect.INSTRUMENTS.get(s.gradiometer)
                inst = s.gradient_instrument_noise if s.gradient_instrument_noise is not None else \
                    (preset.noise if preset and preset.method == "gradient" else None)
                baseline = s.gradiometer_baseline if s.gradiometer_baseline is not None else \
                    (preset.baseline if preset and preset.method == "gradient" else None)
                if inst is None or baseline is None:
                    results.append(MethodResult(method, False, f"unknown gradiometer {s.gradiometer!r} (or give "
                                                "gradient_instrument_noise and gradiometer_baseline)"))
                    continue
                if baseline <= 0 or s.gradiometer_height < 0:
                    raise ValueError("gradiometer_baseline must be positive and gradiometer_height zero or positive")
                noise = detect.effective_noise(inst, s.gradient_field_noise)
                anomaly = _gradient(t, x, contrast, amb, s, baseline)
                res = MethodResult(method, True, contrast=contrast, contrast_units=units, instrument=s.gradiometer,
                                   instrument_noise=inst, field_noise=s.gradient_field_noise, main_field=info)
        a = detect.assess(x, anomaly, noise, depth=t.depth_top, method=method)
        preset = detect.INSTRUMENTS.get(res.instrument)
        overridden = {"gravity": s.gravity_instrument_noise, "magnetic": s.magnetic_instrument_noise,
                      "gradient": s.gradient_instrument_noise}[method] is not None
        if preset and not overridden:
            a.assumptions.append(f"{res.instrument}: {preset.noise:g} {a.units} per reading ({preset.basis}; "
                                 f"{preset.reference} data sheet).")
        if t.shape in TWO_D:
            a.assumptions.append("2D target: infinitely long along strike, at right angles to the profile.")
        if method == "magnetic":
            a.assumptions.append(f"Sensor {s.sensor_height:g} m above ground; profile azimuth {s.profile_azimuth:g}°; "
                                 "induced magnetisation only, no demagnetisation.")
        if method == "gradient":
            top = s.gradiometer_height + baseline
            a.assumptions.append(f"Reading: lower sensor ({s.gradiometer_height:g} m above ground) minus upper sensor "
                                 f"({top:g} m), in nT over the {baseline:g} m baseline; diurnal variation and the regional "
                                 f"field cancel. Profile azimuth {s.profile_azimuth:g}°; induced magnetisation only.")
        res.x, res.anomaly, res.assessment = x.round(4).tolist(), anomaly.tolist(), a
        results.append(res)

    ranked = sorted((r for r in results if r.available and r.assessment.verdict != "not detectable"),
                    key=lambda r: r.assessment.snr, reverse=True)
    return Report(__version__, asdict(t), asdict(s), methods, results, ranked[0].method if ranked else None)
