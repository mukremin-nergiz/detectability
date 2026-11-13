"""Electrical resistivity tomography (ERT): 2D forward model of a target under a line of electrodes, the change it
makes to the apparent resistivities, and whether an inversion of noisy synthetic data recovers it.

The finite-element modelling and the inversion use pyGIMLi (Rücker, Günther & Wagner 2017, Computers & Geosciences
109, 106-123), an optional dependency: ``pip install "detectability[ert]"``. Without it the method is reported as not
available. A service can run it instead: if pyGIMLi is missing and ``DETECTABILITY_ERT_URL`` is set, the request is
sent there (the website does this; its ERT runs on a separate machine).

What is computed, and the conventions:
- Electrodes at equal spacing along a straight line centred over the target; arrays: dipole-dipole, Wenner,
  Schlumberger or multiple gradient.
- Apparent resistivities with and without the target on the same mesh; their ratio removes the mesh error. The data
  anomaly is 100 (rho_a,target / rho_a,host - 1) %, and S/N is its largest absolute value over the data noise (%), with
  the same thresholds as the other methods.
- Recovery: the noisy data (the given noise level) are inverted (smoothness-constrained Gauss-Newton, pyGIMLi's
  ERTManager, at most 6 iterations). The recovered contrast is the median model value inside the target over the median
  outside. The target counts as recovered when that contrast has the right sign and reaches 20 % of the true contrast
  in log scale or a factor of 1.5, whichever is smaller; otherwise a detectable verdict is lowered to marginal.
- 2D: the target is infinitely long across the profile. A sphere is modelled as a horizontal cylinder of the same
  diameter, which overstates its effect; the assumptions say so.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

import numpy as np

# Resistivity ranges (ohm m), from Loke (1999), "Electrical imaging surveys for environmental and engineering
# studies", Table 1, after Keller & Frischknecht (1966) and Daniels & Alberty (1966); ranges checked against that
# table on 14 Nov 2026. "Typical" is the geometric middle of the range. Air (a void) is effectively an insulator; it
# is modelled as 10^6 ohm m, beyond which the anomaly no longer changes. Metal: iron, 9.074e-8 ohm m.
RESISTIVITY: dict[str, tuple[float, float, float]] = {
    "air": (1e6, 1e6, 1e6),
    "granite": (7.07e4, 5e3, 1e6),
    "basalt": (3.16e4, 1e3, 1e6),
    "slate": (1.55e5, 6e2, 4e7),
    "marble": (1.58e5, 1e2, 2.5e8),
    "quartzite": (1.41e5, 1e2, 2e8),
    "sandstone": (1.79e2, 8.0, 4e3),
    "shale": (2.0e2, 20.0, 2e3),
    "limestone": (1.41e2, 50.0, 4e2),
    "clay": (10.0, 1.0, 100.0),
    "alluvium": (89.4, 10.0, 800.0),
    "water": (31.6, 10.0, 100.0),
    "fresh water": (31.6, 10.0, 100.0),
    "sea water": (0.2, 0.2, 0.2),
    "metal": (9.074e-8, 9.074e-8, 9.074e-8),
}

ARRAYS = {"dipole-dipole": "dd", "wenner": "wa", "schlumberger": "slm", "gradient": "gr"}
DEFAULT_ARRAY = "dipole-dipole"
DEFAULT_ELECTRODES = 48
DEFAULT_NOISE_PERCENT = 3.0
RECOVERY_THRESHOLD = 0.2
RECOVERED_FACTOR = 1.5
MIN_RESISTIVITY = 1e-3   # metal is modelled as 10^-3 ohm m (a perfect conductor for the survey; avoids a singular system)
DEFAULT_SERVICE_TIMEOUT = 150


def resistivity(name: str) -> float:
    try:
        return RESISTIVITY[name][0]
    except KeyError:
        raise KeyError(f"no resistivity for {name!r}; known: {', '.join(sorted(RESISTIVITY))}") from None


def available() -> bool:
    try:
        import pygimli  # noqa: F401
        return True
    except ImportError:
        return False


def _nice_down(v: float) -> float:
    e = np.floor(np.log10(v))
    for m in (5.0, 2.0, 1.0):
        if m * 10**e <= v:
            return float(m * 10**e)
    return float(5 * 10 ** (e - 1))


def default_spacing(depth_needed: float, electrodes: int) -> float:
    """Electrode spacing whose spread (electrodes - 1 spacings) is about six times the depth that must be seen."""
    return max(0.1, _nice_down(6.0 * depth_needed / (electrodes - 1)))


@dataclass
class Geometry:
    polygon: list          # target outline (x, z), z negative downwards, in the profile frame centred on the target
    depth_needed: float    # m: the bottom of the target (or of its useful part) that the survey must reach
    area: float            # m^2 of the 2D outline


def geometry(t) -> Geometry:
    """2D outline of a ``survey.Target``."""
    r = (t.size or 0) / 2
    if t.shape in ("sphere", "horizontal_cylinder"):
        zc = t.depth_top + r
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        return Geometry([(r * np.cos(a), -zc + r * np.sin(a)) for a in ang], zc + r, np.pi * r * r)
    if t.shape in ("vertical_cylinder", "prism"):
        w = t.size / 2
        return Geometry([(-w, -t.depth_top), (w, -t.depth_top), (w, -t.depth_bottom), (-w, -t.depth_bottom)],
                        t.depth_bottom, 2 * w * (t.depth_bottom - t.depth_top))
    if t.shape == "dyke":
        h = t.depth_bottom - t.depth_top
        dx = h / np.tan(np.radians(t.dip)) if abs(t.dip - 90) > 1e-9 else 0.0
        w = t.size / 2
        return Geometry([(-w, -t.depth_top), (w, -t.depth_top), (w + dx, -t.depth_bottom), (-w + dx, -t.depth_bottom)],
                        t.depth_bottom, t.size * h)
    if t.shape == "contact":
        return Geometry([], t.depth_bottom, float("inf"))
    raise ValueError(f"unknown shape {t.shape!r}")


def run(t, s) -> dict:
    """Model, assess and invert. Returns a plain dict (MethodResult fields) so a service can send it as JSON."""
    import pygimli as pg
    import pygimli.meshtools as mt
    from pygimli.physics import ert as pgert

    from . import detect

    pg.setThreadCount(int(os.environ.get("DETECTABILITY_ERT_THREADS", "2")))
    started = time.time()
    rho_h = t.host_resistivity if t.host_resistivity is not None else resistivity(t.host)
    rho_t = t.resistivity if t.resistivity is not None else resistivity(t.material)
    rho_t = max(rho_t, MIN_RESISTIVITY)
    if rho_h <= 0:
        raise ValueError("host resistivity must be positive")
    array = s.ert_array or DEFAULT_ARRAY
    if array not in ARRAYS:
        raise ValueError(f"unknown ERT array {array!r}; choose from {', '.join(ARRAYS)}")
    n = int(s.ert_electrodes or DEFAULT_ELECTRODES)
    if not 8 <= n <= 96:
        raise ValueError("ert_electrodes must be between 8 and 96")
    noise = s.ert_noise if s.ert_noise is not None else DEFAULT_NOISE_PERCENT
    if not 0.1 <= noise <= 50:
        raise ValueError("ert_noise must be between 0.1 and 50 %")
    g = geometry(t)
    a = s.ert_spacing or default_spacing(g.depth_needed, n)
    if a <= 0:
        raise ValueError("ert_spacing must be positive")
    L = a * (n - 1)
    x0 = -L / 2
    elecs = np.linspace(x0, x0 + L, n)
    scheme = pgert.createData(elecs=elecs, schemeName=ARRAYS[array])

    pad = max(L, 4 * g.depth_needed)
    depth = max(L / 2, 2.5 * g.depth_needed)
    world = mt.createWorld(start=[x0 - pad, 0], end=[x0 + L + pad, -depth - pad], worldMarker=True)
    if t.shape == "contact":
        body = mt.createPolygon([(0, -t.depth_top), (x0 + L + pad, -t.depth_top), (x0 + L + pad, -t.depth_bottom),
                                 (0, -t.depth_bottom)], isClosed=True, marker=2)
    else:
        body = mt.createPolygon(g.polygon, isClosed=True, marker=2, area=max((a / 4) ** 2, g.area / 200))
    geom = world + body
    for p in scheme.sensors():
        geom.createNode(p)
        geom.createNode(p - [0, a / 10])
    mesh = mt.createMesh(geom, quality=33.5)

    with_target = pgert.simulate(mesh, scheme=scheme, res=[[1, rho_h], [2, rho_t]], noiseLevel=0, verbose=False)
    host_only = pgert.simulate(mesh, scheme=scheme, res=[[1, rho_h], [2, rho_h]], noiseLevel=0, verbose=False)
    ra_t, ra_h = np.array(with_target["rhoa"]), np.array(host_only["rhoa"])
    anomaly = 100.0 * (ra_t / ra_h - 1.0)

    # Profile for the assessment: at each data midpoint, the largest change over the levels.
    mid = np.array([(elecs[int(with_target["a"][i])] + elecs[int(with_target["b"][i])] +
                     elecs[int(with_target["m"][i])] + elecs[int(with_target["n"][i])]) / 4
                    for i in range(with_target.size())])
    xs = np.unique(np.round(mid, 6))
    prof = np.array([anomaly[np.isclose(mid, xv)][np.argmax(np.abs(anomaly[np.isclose(mid, xv)]))] for xv in xs])
    assessment = detect.assess(xs, prof, noise, depth=t.depth_top, method="ert")
    assessment.station_spacing = float(a)
    assessment.profile_length = float(L)

    details = {"array": array, "electrodes": n, "spacing_m": float(a), "spread_m": float(L), "data": int(scheme.size()),
               "host_resistivity": float(rho_h), "target_resistivity": float(rho_t), "noise_percent": float(noise),
               "max_anomaly_percent": float(np.max(np.abs(anomaly))), "inverted": False}

    if s.ert_invert and t.shape != "contact":
        rng = np.random.default_rng(1337)
        data = pg.DataContainerERT(with_target)
        data["rhoa"] = ra_t * (1 + rng.normal(0, noise / 100, len(ra_t)))
        data["err"] = pg.Vector(len(ra_t), noise / 100)
        mgr = pgert.ERTManager(data, verbose=False)
        mgr.invert(lam=20, verbose=False, paraDepth=min(depth, max(1.5 * g.depth_needed, L / 5)), quality=33.6,
                   paraMaxCellSize=max(a * a, g.area / 50), paraDX=0.5, maxIter=6)
        cells = np.array(mgr.paraDomain.cellCenters())[:, :2]
        model = np.array(mgr.model)
        inside = np.array([_inside(p, g.polygon) for p in cells])
        if inside.sum() >= 3:
            rec_t, rec_bg = float(np.median(model[inside])), float(np.median(model[~inside]))
            recovery = float(np.log(rec_t / rec_bg) / np.log(rho_t / rho_h)) if rho_t != rho_h else 0.0
        else:
            rec_t, rec_bg, recovery = None, float(np.median(model)), 0.0
        # Recovered: the inverted contrast points the right way and reaches 20 % of the true contrast (log scale) or a
        # factor of 1.5, whichever is smaller (a void's true contrast is huge; no inversion of surface data reaches it).
        need = min(RECOVERY_THRESHOLD * abs(np.log(rho_t / rho_h)), np.log(RECOVERED_FACTOR))
        recovered = bool(rec_t is not None and rho_t != rho_h and np.sign(np.log(rec_t / rec_bg)) == np.sign(np.log(rho_t / rho_h))
                         and abs(np.log(rec_t / rec_bg)) >= need)
        details.update(inverted=True, recovered_target=rec_t, recovered_background=rec_bg, recovery=recovery,
                       recovered=recovered, chi2=float(mgr.inv.chi2()), inversion_cells=int(len(model)),
                       section={"x": cells[:, 0].round(3).tolist(), "z": cells[:, 1].round(3).tolist(),
                                "rho": np.round(model, 4).tolist()})
        if not recovered and assessment.verdict != "not detectable":
            assessment.verdict = "marginal"
    details["seconds"] = round(time.time() - started, 1)

    unit = "%"
    asm = [
        f"Signal-to-noise thresholds: >= {detect.CLEAR:g} clearly detectable, {detect.MARGINAL:g}-{detect.CLEAR:g} "
        f"marginal, < {detect.MARGINAL:g} not detectable; S/N is the largest change in apparent resistivity over the "
        f"data noise ({noise:g} %).",
        f"{n} electrodes every {a:g} m ({L:g} m spread), {array} array, {scheme.size()} measurements.",
        f"Host {rho_h:.3g} Ωm, target {rho_t:.3g} Ωm" + ("" if t.resistivity is not None or t.host_resistivity is not None
                                                          else " (typical values; resistivity ranges over orders of magnitude, measure on site)") + ".",
        "2D model: the target is infinitely long across the profile; homogeneous host, flat ground, electrodes on the "
        "surface. Finite-element forward model (pyGIMLi); the target/host ratio cancels the mesh error.",
    ]
    if t.shape == "sphere":
        asm.append("A sphere is modelled as a horizontal cylinder of the same diameter: a real 3D sphere gives a "
                   "smaller anomaly than shown.")
    if t.shape in ("prism", "vertical_cylinder"):
        asm.append("A 3D block or shaft is modelled by its 2D cross-section; a short target gives a smaller anomaly.")
    if details["inverted"]:
        rt = details["recovered_target"]
        asm.append(f"Inversion of the noisy data (smoothness constraint, λ 20, χ² {details['chi2']:.2g}): recovered "
                   f"{'%.3g' % rt + ' Ωm in the target' if rt else 'no cells inside the target'} against "
                   f"{details['recovered_background']:.3g} Ωm around it ({100 * details['recovery']:.0f} % of the true contrast "
                   f"in log scale): " + ("recovered." if details["recovered"] else "not recovered (needs 20 % of the "
                   "contrast in log scale or a factor of 1.5)."))
    elif t.shape == "contact":
        asm.append("Inversion not run for a contact (no closed target).")
    if a > (t.size or a):
        asm.append(f"Electrode spacing ({a:g} m) is larger than the target ({t.size:g} m): its shape is not resolved.")
    if t.shape in ("horizontal_cylinder", "dyke", "contact"):
        asm.append("2D target: infinitely long along strike, at right angles to the profile.")
    assessment.assumptions = asm
    assessment.units = unit
    return {"method": "ert", "available": True, "contrast": float(rho_t / rho_h), "contrast_units": "resistivity ratio",
            "instrument": f"{n} electrodes, {array}", "instrument_noise": float(noise), "field_noise": None,
            "x": xs.round(4).tolist(), "anomaly": prof.tolist(), "assessment": assessment.to_dict(), "details": details}


def _inside(p, poly) -> bool:
    x, z = p
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, zi = poly[i]
        xj, zj = poly[j]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            inside = not inside
        j = i
    return inside


def remote(target: dict, survey: dict, url: str) -> dict:
    """Ask an ERT service (another machine with pyGIMLi) for the result."""
    req = urllib.request.Request(url, data=json.dumps({"target": target, "survey": survey}).encode(),
                                 headers={"Content-Type": "application/json"})
    timeout = float(os.environ.get("DETECTABILITY_ERT_TIMEOUT", DEFAULT_SERVICE_TIMEOUT))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
