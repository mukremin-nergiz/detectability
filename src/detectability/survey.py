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
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from . import __version__, detect, gravity, igrf, magnetics, rocks
from .constants import SI_TO_UGAL

METHODS = ("gravity", "magnetic")

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
    profile_azimuth: float = 0.0          # degrees east of north; 2D targets strike at right angles to it
    latitude: float | None = None         # for the main field (IGRF-14); needed for magnetics
    longitude: float | None = None
    date: str | None = None               # ISO date of the survey; default: today
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
        return SCRIPT.format(
            version=self.version,
            target=json.dumps(target, indent=4, ensure_ascii=False),
            survey=json.dumps(survey, indent=4, ensure_ascii=False),
            methods=json.dumps(self.methods),
        )


SCRIPT = '''"""Reproduces a Detectability result (detectability {version}).

Install: pip install "git+https://github.com/mukremin-nergiz/detectability"
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


def evaluate(target, survey=None, methods=METHODS) -> Report:
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
            inst = s.magnetic_instrument_noise if s.magnetic_instrument_noise is not None else \
                detect.MAGNETOMETERS_NT.get(s.magnetometer)
            if inst is None:
                results.append(MethodResult(method, False, f"unknown magnetometer {s.magnetometer!r}"))
                continue
            noise = detect.effective_noise(inst, s.magnetic_field_noise)
            anomaly = _magnetic(t, x, contrast, amb, s)
            res = MethodResult(method, True, contrast=contrast, contrast_units=units, instrument=s.magnetometer,
                               instrument_noise=inst, field_noise=s.magnetic_field_noise, main_field=info)
        a = detect.assess(x, anomaly, noise, depth=t.depth_top, method=method)
        preset = detect.INSTRUMENTS.get(res.instrument)
        overridden = (s.gravity_instrument_noise if method == "gravity" else s.magnetic_instrument_noise) is not None
        if preset and not overridden:
            a.assumptions.append(f"{res.instrument}: {preset.noise:g} {a.units} per reading ({preset.basis}; "
                                 f"{preset.reference} data sheet).")
        if t.shape in TWO_D:
            a.assumptions.append("2D target: infinitely long along strike, at right angles to the profile.")
        if method == "magnetic":
            a.assumptions.append(f"Sensor {s.sensor_height:g} m above ground; profile azimuth {s.profile_azimuth:g}°; "
                                 "induced magnetisation only, no demagnetisation.")
        res.x, res.anomaly, res.assessment = x.round(4).tolist(), anomaly.tolist(), a
        results.append(res)

    ranked = sorted((r for r in results if r.available and r.assessment.verdict != "not detectable"),
                    key=lambda r: r.assessment.snr, reverse=True)
    return Report(__version__, asdict(t), asdict(s), methods, results, ranked[0].method if ranked else None)
