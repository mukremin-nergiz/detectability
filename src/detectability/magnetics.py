"""Total-field magnetic anomaly of simple near-surface targets.

Conventions
-----------
- Right-handed frame: x north, y east, z down; depths positive downwards; metres.
- The ambient (main) field is given by its total intensity (nT), inclination (degrees, positive down) and declination
  (degrees east of north); ``igrf.field`` provides it from location and date.
- ``susceptibility`` is the contrast (target minus host) in SI units. Induced magnetisation is M = chi * F / mu0;
  self-demagnetisation is neglected (valid for chi << 1). Optional remanent magnetisation is given as
  ``(intensity A/m, inclination, declination)``.
- Profiles run along ``azimuth`` (degrees east of north) through the target; ``x`` is the distance along the profile.
  The sensor is ``height`` metres above the ground.
- Results are the total-field anomaly in nT: the projection of the anomalous field on the main-field direction, valid when
  the anomaly is much smaller than the main field.
- 2D targets (horizontal cylinder, dyke, contact) strike perpendicular to the profile and extend to infinity along strike.

References
----------
- Blakely, R. J. (1996). Potential Theory in Gravity and Magnetic Applications, Cambridge University Press, ch. 4, 5, 9.
- Telford, W. M., Geldart, L. P., Sheriff, R. E. (1990). Applied Geophysics, 2nd ed., Cambridge University Press, ch. 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MU0 = 4e-7 * np.pi
T_TO_NT = 1e9


@dataclass(frozen=True)
class Ambient:
    """Main field: total intensity (nT), inclination and declination (degrees)."""

    total: float
    inclination: float
    declination: float

    @classmethod
    def from_igrf(cls, main_field) -> "Ambient":
        return cls(main_field.total, main_field.inclination, main_field.declination)


def unit_vector(inclination: float, declination: float) -> np.ndarray:
    """Unit vector (north, east, down) for a direction given by inclination and declination in degrees."""
    i, d = np.radians(inclination), np.radians(declination)
    return np.array([np.cos(i) * np.cos(d), np.cos(i) * np.sin(d), np.sin(i)])


def magnetisation(susceptibility: float, ambient: Ambient, remanence: tuple[float, float, float] | None = None) -> np.ndarray:
    """Magnetisation vector (north, east, down) in A/m: induced plus optional remanent."""
    induced = susceptibility * ambient.total / T_TO_NT / MU0 * unit_vector(ambient.inclination, ambient.declination)
    if remanence is None:
        return induced
    intensity, inc, dec = remanence
    return induced + intensity * unit_vector(inc, dec)


def _profile_points(x, azimuth: float, height: float):
    x = np.asarray(x, float)
    a = np.radians(azimuth)
    return np.stack([x * np.cos(a), x * np.sin(a), np.full_like(x, -height)], axis=-1)


def sphere(x, depth: float, radius: float, susceptibility: float, ambient: Ambient, azimuth: float = 0.0,
           height: float = 0.0, remanence=None):
    """Buried sphere (dipole): cavity, ore body, buried object. ``depth`` is to the centre."""
    if depth <= radius:
        raise ValueError("depth to the centre must exceed the radius (the sphere must be buried)")
    m = magnetisation(susceptibility, ambient, remanence) * (4.0 / 3.0 * np.pi * radius**3)
    r = _profile_points(x, azimuth, height) - np.array([0.0, 0.0, depth])
    dist = np.linalg.norm(r, axis=-1, keepdims=True)
    rhat = r / dist
    b = MU0 / (4 * np.pi) * (3.0 * (rhat @ m)[..., None] * rhat - m) / dist**3
    return b @ unit_vector(ambient.inclination, ambient.declination) * T_TO_NT


def _in_plane(vec: np.ndarray, azimuth: float) -> np.ndarray:
    """Components of a (north, east, down) vector along the profile and downwards."""
    a = np.radians(azimuth)
    return np.array([vec[0] * np.cos(a) + vec[1] * np.sin(a), vec[2]])


def horizontal_cylinder(x, depth: float, radius: float, susceptibility: float, ambient: Ambient, azimuth: float = 0.0,
                        height: float = 0.0, remanence=None):
    """Infinite horizontal cylinder (tunnel, pipe, gallery) striking perpendicular to the profile; ``depth`` is to the
    axis. A 2D line dipole: B = mu0 / (2 pi r^2) * (2 (m.r) r - m)."""
    if depth <= radius:
        raise ValueError("depth to the axis must exceed the radius")
    m = _in_plane(magnetisation(susceptibility, ambient, remanence), azimuth) * (np.pi * radius**2)
    u = np.asarray(x, float)
    r = np.stack([u, np.full_like(u, -height - depth)], axis=-1)
    dist2 = np.sum(r**2, axis=-1, keepdims=True)
    rhat = r / np.sqrt(dist2)
    b = MU0 / (2 * np.pi) * (2.0 * (rhat @ m)[..., None] * rhat - m) / dist2
    return b @ _in_plane(unit_vector(ambient.inclination, ambient.declination), azimuth) * T_TO_NT


def polygon(x, vertices, susceptibility: float, ambient: Ambient, azimuth: float = 0.0, height: float = 0.0,
            remanence=None):
    """2D body of polygonal cross-section (any shape), striking perpendicular to the profile.

    ``vertices`` are (distance along profile, depth) pairs in either order. The uniformly magnetised body is replaced by
    the magnetic surface charge M.n on its sides; each straight side has a closed-form field (Blakely 1996, ch. 9;
    equivalent to Talwani & Heirtzler 1964).
    """
    v = np.asarray(vertices, float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        raise ValueError("need at least three (x, depth) vertices")
    if v[:, 1].min() + height <= 0:
        raise ValueError("the body must lie below the sensor (raise the sensor or bury the body)")
    area2 = np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
    if area2 < 0:                              # make the order counter-clockwise in the (x, depth) plane
        v = v[::-1]
    m = _in_plane(magnetisation(susceptibility, ambient, remanence), azimuth)
    f = _in_plane(unit_vector(ambient.inclination, ambient.declination), azimuth)
    obs = np.stack([np.asarray(x, float), np.full(np.shape(x), -height, float)], axis=-1)
    b = np.zeros(obs.shape)
    for p1, p2 in zip(v, np.roll(v, -1, axis=0)):
        side = p2 - p1
        length = np.hypot(*side)
        if length == 0:
            continue
        t = side / length
        n = np.array([t[1], -t[0]])            # outward normal for a counter-clockwise polygon
        sigma = m @ n                          # magnetic surface charge, A/m
        rel = obs - p1
        a, c = rel @ t, rel @ n                # along-side and normal coordinates of the observation point
        r1 = np.hypot(a, c)
        r2 = np.hypot(a - length, c)
        along = np.log(r1 / r2)
        normal = np.arctan2(c * length, c**2 + a * (a - length))
        b += sigma * (along[..., None] * t + normal[..., None] * n)
    return MU0 / (2 * np.pi) * (b @ f) * T_TO_NT


def dyke(x, top: float, bottom: float, width: float, susceptibility: float, ambient: Ambient, azimuth: float = 0.0,
         dip: float = 90.0, centre: float = 0.0, height: float = 0.0, remanence=None):
    """2D dyke (or buried wall, filled trench, vein) of horizontal ``width``, from ``top`` to ``bottom`` depth.
    ``dip`` in degrees from horizontal, towards +x for dips below 90."""
    if not (0 <= top < bottom and width > 0 and 0 < dip <= 180):
        raise ValueError("need 0 <= top < bottom, width > 0 and 0 < dip <= 180")
    shift = (bottom - top) / np.tan(np.radians(dip)) if dip != 90 else 0.0
    w = width / 2.0
    verts = [(centre - w, top), (centre + w, top), (centre + w + shift, bottom), (centre - w + shift, bottom)]
    return polygon(x, verts, susceptibility, ambient, azimuth, height, remanence)


def contact(x, top: float, bottom: float, susceptibility: float, ambient: Ambient, azimuth: float = 0.0,
            edge: float = 0.0, height: float = 0.0, remanence=None, far: float = 1e9):
    """2D vertical contact: a slab from ``top`` to ``bottom`` depth for x > ``edge`` (fault, edge of a fill or of a
    magnetic unit). The slab is closed ``far`` metres away, where its effect is negligible."""
    if not 0 <= top < bottom:
        raise ValueError("need 0 <= top < bottom")
    verts = [(edge, top), (edge + far, top), (edge + far, bottom), (edge, bottom)]
    return polygon(x, verts, susceptibility, ambient, azimuth, height, remanence)
