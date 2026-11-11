"""Validation: every model against a result computed in a different way.

The same cases run as tests (each must stay within its tolerance), feed the website's Validation page, and print as a
Markdown table:

    python -m detectability.validation

Errors are the largest absolute difference along the profile as a percentage of the largest reference value (or, for a
single value, the relative difference).
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from importlib import resources

import numpy as np

from . import gpr, gravity, igrf, magnetics
from .constants import G, SI_TO_UGAL
from .magnetics import MU0, Ambient

MID = Ambient(48567.0, 58.373, 6.313)     # Sivas, Türkiye, 2026 (IGRF-14)
POLE = Ambient(50000.0, 90.0, 0.0)        # vertical field and magnetisation
CHI = 0.01

# British Geological Survey IGRF-14 web service (geomag.bgs.ac.uk), retrieved 3 Nov 2026; rounded to 1 nT and 0.001°.
BGS = [
    ("Sivas, Türkiye", 39.75, 37.02, 0.0, "2026-07-01", 48567, 6.313, 58.373),
    ("Cape Town, South Africa", -33.9, 18.4, 0.0, "2026-07-01", 24996, -26.673, -64.631),
    ("Canadian Arctic", 70.0, -100.0, 0.0, "2026-07-01", 57831, -0.041, 85.147),
    ("Amazon, 5 km altitude", 0.0, -60.0, 5.0, "2026-07-01", 26671, -16.84, 8.983),
    ("London, United Kingdom", 51.5, -0.1, 0.0, "2012-03-15", 48635, -1.345, 66.469),
]
IGRF_TOLERANCE_NT, IGRF_TOLERANCE_DEG = 1.0, 0.002

# Independent code: the same targets computed with Harmonica (Fatiando a Terra Project), a separate open-source package.
# tools/harmonica_reference.py computes them and stores data/harmonica_reference.json, so neither the library nor the
# website has to install Harmonica; CI installs it, recomputes the file and checks it. Profiles run north (azimuth 0),
# so the profile frame (along, across, down) is (north, east, down). A prism 200 km long stands in for a 2D dyke.
HARMONICA_FILE = "harmonica_reference.json"
_FIELD = (MID.total, MID.inclination, MID.declination)
HARMONICA_CASES = {
    "harmonica-prism-gravity": {"source": "prism", "along": (-3.0, 3.0), "across": (-4.0, 4.0), "depth": (2.0, 6.0),
                                "drho": -2200.0, "x": (-40.0, 40.0, 81)},
    "harmonica-prism-magnetic": {"source": "prism", "along": (-3.0, 2.0), "across": (-2.5, 3.5), "depth": (4.0, 7.0),
                                 "chi": CHI, "field": _FIELD, "height": 0.5, "x": (-30.0, 30.0, 61)},
    "harmonica-sphere-gravity": {"source": "sphere", "depth": 12.0, "radius": 3.0, "drho": -2600.0,
                                 "x": (-60.0, 60.0, 121)},
    "harmonica-sphere-magnetic": {"source": "sphere", "depth": 12.0, "radius": 3.0, "chi": CHI, "field": _FIELD,
                                  "height": 0.5, "x": (-60.0, 60.0, 121)},
    "harmonica-dyke-gravity": {"source": "prism", "along": (5.0, 9.0), "across": (-1e5, 1e5), "depth": (3.0, 15.0),
                               "drho": 500.0, "x": (-40.0, 40.0, 81)},
    "harmonica-dyke-magnetic": {"source": "prism", "along": (-2.0, 2.0), "across": (-1e5, 1e5), "depth": (3.0, 20.0),
                                "chi": CHI, "field": _FIELD, "height": 0.5, "x": (-50.0, 50.0, 101)},
}


@dataclass
class Case:
    key: str
    check: str            # kind of check
    geometry: str         # target and method
    model: str            # how Detectability computes it
    reference: str        # the independent result
    setting: str          # the example used
    units: str
    tolerance: float      # %: the test fails above this
    error: float          # %
    x: np.ndarray | None = None
    model_values: np.ndarray | None = None
    reference_values: np.ndarray | None = None

    @property
    def passed(self) -> bool:
        return self.error <= self.tolerance


def _profile_error(model, ref) -> float:
    model, ref = np.asarray(model, float), np.asarray(ref, float)
    return float(np.max(np.abs(model - ref)) / np.max(np.abs(ref)) * 100.0)


def _value_error(model: float, ref: float) -> float:
    return float(abs(model - ref) / abs(ref) * 100.0)


def _point_masses(x, box, drho, n=16):
    """Gauss-Legendre sum of point masses filling a box ((x1, x2), (y1, y2), (z1, z2)); profile along y = 0."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    pts = [((hi - lo) / 2 * nodes + (hi + lo) / 2, (hi - lo) / 2 * weights) for lo, hi in box]
    X, Y, Z = np.meshgrid(pts[0][0], pts[1][0], pts[2][0], indexing="ij")
    Wt = np.einsum("i,j,k->ijk", pts[0][1], pts[1][1], pts[2][1])
    return np.array([np.sum(G * drho * Wt * Z / ((X - xi) ** 2 + Y**2 + Z**2) ** 1.5) for xi in x])


def _point_dipoles(x, box, chi, ambient, azimuth, height=0.0, n=16):
    """Gauss-Legendre sum of point dipoles filling a box given in profile coordinates (along, across, depth)."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    pts = [((hi - lo) / 2 * nodes + (hi + lo) / 2, (hi - lo) / 2 * weights) for lo, hi in box]
    az = np.radians(azimuth)
    along, across = np.array([np.cos(az), np.sin(az), 0.0]), np.array([-np.sin(az), np.cos(az), 0.0])
    m = magnetics.magnetisation(chi, ambient)
    f = magnetics.unit_vector(ambient.inclination, ambient.declination)
    A, C, Z = np.meshgrid(pts[0][0], pts[1][0], pts[2][0], indexing="ij")
    Wt = np.einsum("i,j,k->ijk", pts[0][1], pts[1][1], pts[2][1]).ravel()
    src = A.ravel()[:, None] * along + C.ravel()[:, None] * across + Z.ravel()[:, None] * np.array([0, 0, 1.0])
    out = []
    for xi in x:
        r = xi * along - np.array([0, 0, height]) - src
        dist = np.linalg.norm(r, axis=1, keepdims=True)
        rhat = r / dist
        b = MU0 / (4 * np.pi) * (3 * (rhat @ m)[:, None] * rhat - m) / dist**3
        out.append(np.sum((b @ f) * Wt) * 1e9)
    return np.array(out)


def _dipoles_along_strike(x, z, radius, chi, ambient, azimuth, n=20001):
    m = magnetics.magnetisation(chi, ambient) * np.pi * radius**2
    f = magnetics.unit_vector(ambient.inclination, ambient.declination)
    a = np.radians(azimuth)
    along, strike = np.array([np.cos(a), np.sin(a), 0.0]), np.array([-np.sin(a), np.cos(a), 0.0])
    theta = np.linspace(-np.pi / 2, np.pi / 2, n)[1:-1]
    s, ds = z * np.tan(theta), z / np.cos(theta) ** 2 * (theta[1] - theta[0])
    out = []
    for xi in x:
        r = xi * along - s[:, None] * strike - np.array([0.0, 0.0, z])
        dist = np.linalg.norm(r, axis=1, keepdims=True)
        rhat = r / dist
        b = MU0 / (4 * np.pi) * (3 * (rhat @ m)[:, None] * rhat - m) / dist**3
        out.append(np.sum((b @ f) * ds) * 1e9)
    return np.array(out)


def _disc_line_masses(x, radius, z1, z2, drho, n=200):
    """Vertical cylinder by Gauss-Legendre over its cross-section in polar coordinates about the axis, each node a
    vertical line mass in closed form; profile along y = 0. Independent of the rim integral used by the model."""
    r, wr = np.polynomial.legendre.leggauss(n)
    r, wr = radius / 2 * (r + 1), radius / 2 * wr
    t, wt = np.polynomial.legendre.leggauss(2 * n)
    t, wt = np.pi * (t + 1), np.pi * wt
    rr, tt = np.meshgrid(r, t, indexing="ij")
    w = np.outer(wr, wt) * rr
    out = []
    for xi in x:
        s2 = (xi - rr * np.cos(tt)) ** 2 + (rr * np.sin(tt)) ** 2
        out.append(np.sum(w * (1 / np.sqrt(s2 + z1**2) - 1 / np.sqrt(s2 + z2**2))))
    return G * drho * np.array(out)


def _cylinder_dipoles(x, radius, z1, z2, chi, ambient, azimuth, height=0.0, n=32):
    """Vertical cylinder filled with Gauss-Legendre point dipoles (polar about the axis, and depth)."""
    r, wr = np.polynomial.legendre.leggauss(n)
    r, wr = radius / 2 * (r + 1), radius / 2 * wr
    t, wt = np.polynomial.legendre.leggauss(2 * n)
    t, wt = np.pi * (t + 1), np.pi * wt
    z, wz = np.polynomial.legendre.leggauss(n)
    z, wz = (z2 - z1) / 2 * z + (z1 + z2) / 2, (z2 - z1) / 2 * wz
    rr, tt, zz = np.meshgrid(r, t, z, indexing="ij")
    w = (np.einsum("i,j,k->ijk", wr, wt, wz) * rr).ravel()
    a = np.radians(azimuth)
    along, across = np.array([np.cos(a), np.sin(a), 0.0]), np.array([-np.sin(a), np.cos(a), 0.0])
    src = ((rr * np.cos(tt)).ravel()[:, None] * along + (rr * np.sin(tt)).ravel()[:, None] * across
           + zz.ravel()[:, None] * np.array([0.0, 0.0, 1.0]))
    m = magnetics.magnetisation(chi, ambient)
    f = magnetics.unit_vector(ambient.inclination, ambient.declination)
    out = []
    for xi in x:
        rel = xi * along - np.array([0.0, 0.0, height]) - src
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        rhat = rel / dist
        b = MU0 / (4 * np.pi) * (3 * (rhat @ m)[:, None] * rhat - m) / dist**3
        out.append(np.sum((b @ f) * w) * 1e9)
    return np.array(out)


def _poisson(dgz_dz, chi, ambient, drho):
    """Poisson's relation for vertical field and magnetisation: dT = mu0 M / (4 pi G drho) * dgz/dz, in nT."""
    return MU0 * (chi * ambient.total * 1e-9 / MU0) / (4 * np.pi * G * drho) * dgz_dz * 1e9


def _dgz_dz(fn, top, bottom, d=1e-4):
    """Vertical derivative of a gravity profile: moving the sensor down equals moving the body up."""
    return (fn(top - d, bottom - d) - fn(top + d, bottom + d)) / (2 * d)


def _profile_case(key, check, geometry, model, reference, setting, units, tolerance, x, mv, rv) -> Case:
    return Case(key, check, geometry, model, reference, setting, units, tolerance, _profile_error(mv, rv),
                np.asarray(x, float), np.asarray(mv, float), np.asarray(rv, float))


def cases() -> list[Case]:
    out: list[Case] = []

    # --- analytical limits -----------------------------------------------------------------------------------
    x = np.linspace(-300, 300, 121)
    a, z = 2.0, 100.0
    mass = (2 * a) ** 3 * -2600.0
    out.append(_profile_case(
        "prism-point-mass", "Analytical limit", "Prism, gravity, far from the sensor", "Nagy (1966) closed form",
        "Point mass of the same mass", "4 m cube, centre at 100 m, −2,600 kg/m³", "µGal", 0.01, x,
        gravity.prism(x, 0.0, -a, a, -a, a, z - a, z + a, -2600.0) * SI_TO_UGAL,
        G * mass * z / (x**2 + z**2) ** 1.5 * SI_TO_UGAL))

    t, L = 2.0, 1e5
    out.append(Case("prism-slab", "Analytical limit", "Prism, gravity, wide and thin", "Nagy (1966) closed form",
                    "Infinite Bouguer slab, 2πGΔρt", "200 km wide, 2 m thick, top at 10 m, +1,000 kg/m³", "µGal", 0.1,
                    _value_error(gravity.prism(0.0, 0.0, -L, L, -L, L, 10.0, 10.0 + t, 1000.0),
                                 2 * np.pi * G * 1000.0 * t)))

    out.append(Case("step-slab", "Analytical limit", "Vertical step, gravity, far over the slab", "Faulted-slab closed form",
                    "Infinite Bouguer slab, 2πGΔρt", "Slab from 5 to 25 m, +300 kg/m³, 1,000 km from the edge", "µGal",
                    0.01, _value_error(gravity.vertical_step(1e6, 5.0, 25.0, 300.0), 2 * np.pi * G * 300.0 * 20.0)))

    top, bottom, R = 5.0, 30.0, 3.0
    exact = 2 * np.pi * G * -2600.0 * ((bottom - top) - np.hypot(bottom, R) + np.hypot(top, R))
    out.append(Case("vertical-cylinder-axis", "Analytical limit", "Vertical cylinder, gravity, on its axis",
                    "Rim integral (closed form in depth and distance)", "Exact on-axis cylinder formula",
                    "Shaft of 6 m diameter from 5 to 30 m, −2,600 kg/m³", "µGal", 1e-10,
                    _value_error(gravity.vertical_cylinder(0.0, top, bottom, R, -2600.0), exact)))

    # --- independent closed forms ----------------------------------------------------------------------------
    x = np.linspace(-60, 60, 121)
    z, R, n = 8.0, 2.0, 720
    Rn = R * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack([Rn * np.cos(ang), z + Rn * np.sin(ang)])
    out.append(_profile_case(
        "cylinder-polygon", "Independent formula", "Horizontal cylinder, gravity", "Line-mass closed form",
        "2D polygon, 720 sides (Talwani; Won & Bevis)", "Tunnel of 4 m diameter, axis at 8 m, −2,600 kg/m³", "µGal",
        0.01, x, gravity.horizontal_cylinder(x, z, R, -2600.0) * SI_TO_UGAL,
        gravity.polygon(x, circle, -2600.0) * SI_TO_UGAL))

    x = np.linspace(-40, 40, 81)
    out.append(_profile_case(
        "dyke-polygon", "Independent formula", "Vertical dyke, gravity", "Difference of two vertical steps",
        "2D polygon (Won & Bevis)", "4 m wide dyke from 3 to 15 m, +500 kg/m³", "µGal", 1e-6, x,
        gravity.dyke(x, 3, 15, 4, 500.0, centre=7) * SI_TO_UGAL,
        gravity.polygon(x, [(5, 3), (9, 3), (9, 15), (5, 15)], 500.0) * SI_TO_UGAL))

    # gradiometer reading of a sphere at the pole: closed-form dipole field at the two sensor heights
    x = np.linspace(-20, 20, 81)
    depth, radius, low, base = 3.0, 0.5, 0.2, 1.0
    m = CHI * POLE.total * 1e-9 / MU0 * 4 / 3 * np.pi * radius**3

    def _pole_dipole(h):
        z = depth + h                          # depth of the centre below the sensor
        return MU0 / (4 * np.pi) * m * (2 * z**2 - x**2) / (x**2 + z**2) ** 2.5 * 1e9

    from .survey import _gradient, Survey, Target
    sphere_target = Target("sphere", depth - radius, 2 * radius)
    reading = _gradient(sphere_target, x, CHI, POLE, Survey(gradiometer_height=low), base)
    out.append(_profile_case(
        "gradient-sphere-pole", "Independent formula", "Sphere, magnetic gradient (gradiometer reading)",
        "Total-field model at the lower minus the upper sensor", "Closed-form vertical-dipole field at both sensors",
        "Radius 0.5 m, centre at 3 m, sensors 0.2 and 1.2 m above ground, at the pole", "nT", 1e-10, x, reading,
        _pole_dipole(low) - _pole_dipole(low + base)))

    # --- numerical integration -------------------------------------------------------------------------------
    x = np.linspace(-40, 40, 81)
    box = ((-3.0, 3.0), (-4.0, 4.0), (2.0, 6.0))
    out.append(_profile_case(
        "prism-gravity-quadrature", "Numerical integration", "Prism, gravity", "Nagy (1966) closed form",
        "16³ Gauss–Legendre point masses", "6 × 8 m room from 2 to 6 m, −2,200 kg/m³", "µGal", 0.001, x,
        gravity.prism(x, 0.0, *box[0], *box[1], *box[2], -2200.0) * SI_TO_UGAL,
        _point_masses(x, box, -2200.0) * SI_TO_UGAL))

    x = np.linspace(-40, 40, 41)
    out.append(_profile_case(
        "cylinder-magnetic-strike", "Numerical integration", "Horizontal cylinder, magnetic", "2D line dipole",
        "3D dipoles integrated along strike", "Pipe of 2 m diameter, axis at 6 m, 0.01 SI, main field of Sivas, "
        "profile azimuth 30°", "nT", 0.01, x, magnetics.horizontal_cylinder(x, 6.0, 1.0, CHI, MID, azimuth=30.0),
        _dipoles_along_strike(x, 6.0, 1.0, CHI, MID, 30.0)))

    x = np.linspace(-30, 30, 61)
    box = ((-3.0, 2.0), (-2.5, 3.5), (4.0, 7.0))
    out.append(_profile_case(
        "prism-magnetic-quadrature", "Numerical integration", "Prism, magnetic",
        "Gradient tensor and Poisson's relation", "16³ Gauss–Legendre point dipoles",
        "5 × 6 m block from 4 to 7 m, 0.01 SI, sensor 0.5 m above ground, azimuth 30°", "nT", 0.001, x,
        magnetics.prism(x, *box[0], *box[1], *box[2], CHI, MID, azimuth=30.0, height=0.5),
        _point_dipoles(x, box, CHI, MID, 30.0, height=0.5)))

    x = np.linspace(-30, 30, 61)
    out.append(_profile_case(
        "vertical-cylinder-quadrature", "Numerical integration", "Vertical cylinder, gravity",
        "Rim integral (closed form in depth and distance)", "Gauss–Legendre over the cross-section, line masses in depth",
        "Shaft of 6 m diameter from 5 to 30 m, −2,600 kg/m³", "µGal", 1e-8, x,
        gravity.vertical_cylinder(x, 5.0, 30.0, 3.0, -2600.0) * SI_TO_UGAL,
        _disc_line_masses(x, 3.0, 5.0, 30.0, -2600.0) * SI_TO_UGAL))

    out.append(_profile_case(
        "vertical-cylinder-magnetic-quadrature", "Numerical integration", "Vertical cylinder, magnetic",
        "Gradient tensor (rim integral) and Poisson's relation", "Gauss–Legendre point dipoles over the volume",
        "Shaft of 6 m diameter from 5 to 30 m, 0.01 SI, sensor 0.5 m above ground, azimuth 30°", "nT", 1e-6, x,
        magnetics.vertical_cylinder(x, 5.0, 30.0, 3.0, CHI, MID, azimuth=30.0, height=0.5),
        _cylinder_dipoles(x, 3.0, 5.5, 30.5, CHI, MID, 30.0)))

    # --- Poisson's relation: the magnetic codes against the independent gravity codes -------------------------
    x = np.linspace(-60, 60, 121)
    out.append(_profile_case(
        "sphere-poisson", "Poisson's relation", "Sphere, gravity ↔ magnetic", "Magnetic dipole",
        "Vertical derivative of the gravity sphere", "Radius 3 m, centre at 12 m, at the pole", "nT", 0.001, x,
        magnetics.sphere(x, 12.0, 3.0, CHI, POLE),
        _poisson(_dgz_dz(lambda a, b: gravity.sphere(x, a, 3.0, 1000.0), 12.0, 12.0) , CHI, POLE, 1000.0)))

    x = np.linspace(-50, 50, 101)
    for key, geometry, dip in (("dyke-poisson", "Vertical dyke, gravity ↔ magnetic", 90.0),
                               ("dipping-dyke-poisson", "Dipping dyke (60°), gravity ↔ magnetic", 60.0)):
        out.append(_profile_case(
            key, "Poisson's relation", geometry, "Magnetic 2D polygon (surface charge)",
            "Vertical derivative of the gravity dyke", "4 m wide, from 3 to 20 m, at the pole", "nT", 1e-4, x,
            magnetics.dyke(x, 3.0, 20.0, 4.0, CHI, POLE, dip=dip),
            _poisson(_dgz_dz(lambda a, b: gravity.dyke(x, a, b, 4.0, 1000.0, dip=dip), 3.0, 20.0), CHI, POLE, 1000.0)))

    x = np.linspace(-40, 40, 81)
    out.append(_profile_case(
        "vertical-cylinder-poisson", "Poisson's relation", "Vertical cylinder, gravity ↔ magnetic",
        "Gradient tensor (rim integral)", "Vertical derivative of the gravity cylinder",
        "Shaft of 6 m diameter from 2 to 30 m, at the pole", "nT", 1e-4, x,
        magnetics.vertical_cylinder(x, 2.0, 30.0, 3.0, CHI, POLE),
        _poisson(_dgz_dz(lambda a, b: gravity.vertical_cylinder(x, a, b, 3.0, 1000.0), 2.0, 30.0), CHI, POLE, 1000.0)))

    x = np.linspace(-80, 80, 161)
    out.append(_profile_case(
        "contact-poisson", "Poisson's relation", "Vertical contact, gravity ↔ magnetic", "Magnetic 2D polygon",
        "Vertical derivative of the gravity step", "Slab from 5 to 40 m, at the pole", "nT", 1e-4, x,
        magnetics.contact(x, 5.0, 40.0, CHI, POLE),
        _poisson(_dgz_dz(lambda a, b: gravity.vertical_step(x, a, b, 1000.0), 5.0, 40.0), CHI, POLE, 1000.0)))

    # --- ground-penetrating radar --------------------------------------------------------------------------------
    lime = gpr.medium("limestone")
    out.append(Case("gpr-low-loss", "Analytical limit", "GPR, attenuation in a low-loss host",
                    "Exact complex wavenumber", "Low-loss limit 1636 σ/√εr dB/m (σ in S/m)", "Limestone (εr 6, 1 mS/m), 1 GHz",
                    "dB/m", 0.01, _value_error(lime.attenuation_db(1e9),
                                               gpr.low_loss_attenuation_db(lime.permittivity, lime.conductivity))))
    sand = gpr.medium("dry sand")
    out.append(Case("gpr-velocity", "Analytical limit", "GPR, velocity in a low-loss host", "Exact complex wavenumber",
                    "c/√εr", "Dry sand (εr 4, 0.01 mS/m), 100 MHz", "m/ns", 1e-4,
                    _value_error(sand.velocity(1e8), gpr.C0 / np.sqrt(sand.permittivity))))
    air = gpr.Medium(1.0, 0.0)
    dry_lime = gpr.Medium(lime.permittivity, 0.0)
    out.append(Case("gpr-reflection", "Analytical limit", "GPR, reflection coefficient", "Ratio of complex impedances",
                    "(√ε1 − √ε2)/(√ε1 + √ε2) for lossless media", "Air void in limestone (εr 6 and 1)", "", 1e-10,
                    _value_error(abs(gpr.reflection(dry_lime, air, 1e8)),
                                 (np.sqrt(6.0) - 1.0) / (np.sqrt(6.0) + 1.0))))
    r = np.linspace(1.0, 50.0, 50)
    out.append(Case(
        "gpr-sphere-plane", "Analytical limit", "GPR, very large sphere", "Geometric optics (convex mirror)",
        "Flat reflector at the same range", "Radius 10⁹ m, ranges 1–50 m, 100 MHz in limestone", "", 1e-4,
        _profile_error(gpr.size_factor("sphere", r, lime.wavelength(1e8), a=1e9),
                       gpr.size_factor("plane", r, lime.wavelength(1e8)))))
    # Davis & Annan (1989) table, as reproduced in USACE EM 1110-1-1802 (1995) Table 4-3: velocity in m/ns.
    table_v = {"dry sand": 0.15, "wet sand": 0.06, "limestone": 0.12, "shale": 0.09, "silt": 0.07, "clay": 0.06,
               "granite": 0.13, "rock salt": 0.13, "ice": 0.16, "fresh water": 0.033}
    names = list(table_v)
    model_v = np.array([gpr.medium(n).velocity(1e8) / 1e9 for n in names])
    ref_v = np.array([table_v[n] for n in names])
    out.append(Case("gpr-table-velocity", "Published values", "GPR, velocity of common materials",
                    "Exact velocity from the typical εr and σ, 100 MHz",
                    "Davis & Annan (1989) table velocities (rounded to 0.01 m/ns)",
                    "Ten materials, from dry sand to fresh water", "m/ns", 6.0,
                    float(np.max(np.abs(model_v - ref_v) / ref_v * 100.0))))

    # --- independent code ---------------------------------------------------------------------------------------
    out.extend(_harmonica_cases())
    return out


def harmonica_reference() -> dict | None:
    """The stored Harmonica values and the versions they were computed with (None if the file is missing)."""
    try:
        return json.loads(resources.files("detectability").joinpath("data", HARMONICA_FILE).read_text("utf-8"))
    except FileNotFoundError:
        return None


def _harmonica_cases() -> list[Case]:
    ref = harmonica_reference()
    if ref is None:
        return []
    h = f"Harmonica {ref['harmonica'].lstrip('v')}"
    rows = {
        "harmonica-prism-gravity": ("Prism, gravity", "Nagy (1966) closed form", f"{h} prism_gravity",
                                    "6 × 8 m room from 2 to 6 m, −2,200 kg/m³", "µGal", 1e-8),
        "harmonica-prism-magnetic": ("Prism, magnetic", "Gradient tensor and Poisson's relation", f"{h} prism_magnetic",
                                     "5 × 6 m block from 4 to 7 m, 0.01 SI, main field of Sivas, sensor 0.5 m above "
                                     "ground", "nT", 1e-8),
        "harmonica-sphere-gravity": ("Sphere, gravity", "Point mass", f"{h} point_gravity",
                                     "Radius 3 m, centre at 12 m, −2,600 kg/m³", "µGal", 1e-10),
        "harmonica-sphere-magnetic": ("Sphere, magnetic", "Magnetic dipole", f"{h} dipole_magnetic",
                                      "Radius 3 m, centre at 12 m, 0.01 SI, main field of Sivas, sensor 0.5 m above "
                                      "ground", "nT", 1e-10),
        "harmonica-dyke-gravity": ("Vertical dyke, gravity", "Difference of two vertical steps (2D)",
                                   f"{h} prism_gravity, 200 km along strike", "4 m wide dyke from 3 to 15 m, "
                                   "+500 kg/m³", "µGal", 1e-5),
        "harmonica-dyke-magnetic": ("Vertical dyke, magnetic", "2D polygon (surface charge)",
                                    f"{h} prism_magnetic, 200 km along strike", "4 m wide, from 3 to 20 m, 0.01 SI, "
                                    "main field of Sivas, sensor 0.5 m above ground", "nT", 1e-5),
    }
    amb = Ambient(*_FIELD)
    out = []
    for key, c in HARMONICA_CASES.items():
        x = np.linspace(c["x"][0], c["x"][1], int(c["x"][2]))
        if key == "harmonica-prism-gravity":
            model = gravity.prism(x, 0.0, *c["along"], *c["across"], *c["depth"], c["drho"]) * SI_TO_UGAL
        elif key == "harmonica-prism-magnetic":
            model = magnetics.prism(x, *c["along"], *c["across"], *c["depth"], c["chi"], amb, height=c["height"])
        elif key == "harmonica-sphere-gravity":
            model = gravity.sphere(x, c["depth"], c["radius"], c["drho"]) * SI_TO_UGAL
        elif key == "harmonica-sphere-magnetic":
            model = magnetics.sphere(x, c["depth"], c["radius"], c["chi"], amb, height=c["height"])
        elif key == "harmonica-dyke-gravity":
            width = c["along"][1] - c["along"][0]
            model = gravity.dyke(x, *c["depth"], width, c["drho"], centre=sum(c["along"]) / 2) * SI_TO_UGAL
        else:
            width = c["along"][1] - c["along"][0]
            model = magnetics.dyke(x, *c["depth"], width, c["chi"], amb, centre=sum(c["along"]) / 2,
                                   height=c["height"])
        geometry, how, reference, setting, units, tol = rows[key]
        out.append(_profile_case(key, "Independent code", geometry, how, reference, setting, units, tol, x, model,
                                 np.asarray(ref["cases"][key], float)))
    return out


def igrf_rows() -> list[dict]:
    rows = []
    for site, lat, lon, alt, date, F, D, I in BGS:
        f = igrf.field(lat, lon, _dt.date.fromisoformat(date), alt)
        rows.append({"site": site, "date": date, "F": F, "D": D, "I": I, "F_model": f.total,
                     "D_model": f.declination, "I_model": f.inclination,
                     "dF": f.total - F, "dD": f.declination - D, "dI": f.inclination - I})
    return rows


def _pct(v: float) -> str:
    return f"{v:.1e} %" if v < 0.001 else f"{v:.3g} %"


def markdown_table() -> str:
    lines = ["| Check | Geometry and method | Reference | Max. relative error | Tolerance |",
             "|---|---|---|---:|---:|"]
    for c in cases():
        lines.append(f"| {c.check} | {c.geometry} | {c.reference} | {_pct(c.error)} | {_pct(c.tolerance)} |")
    rows = igrf_rows()
    dF = max(abs(r["dF"]) for r in rows)
    dA = max(max(abs(r["dD"]), abs(r["dI"])) for r in rows)
    lines.append(f"| Reference service | IGRF-14 main field, 5 sites | British Geological Survey IGRF-14 | "
                 f"{dF:.2f} nT, {dA:.4f}° | {IGRF_TOLERANCE_NT:g} nT, {IGRF_TOLERANCE_DEG:g}° |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown_table())
