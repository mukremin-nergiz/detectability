"""Gravity forward models for simple near-surface targets.

Conventions
-----------
- Coordinates in metres. Observation points are on the surface (z = 0); depths are positive downwards.
- ``drho`` is the density contrast in kg/m^3 (target minus host; a void in limestone is about -2600).
- Results are the vertical attraction gz in m/s^2, positive downwards (positive for excess mass).
  Use ``constants.SI_TO_UGAL`` / ``SI_TO_MGAL`` to convert.

References
----------
- Telford, W. M., Geldart, L. P., Sheriff, R. E. (1990). Applied Geophysics, 2nd ed., Cambridge University Press, §2.6.
- Blakely, R. J. (1996). Potential Theory in Gravity and Magnetic Applications, Cambridge University Press, ch. 3 and 9.
- Nagy, D. (1966). The gravitational attraction of a right rectangular prism. Geophysics 31, 362-371.
- Plouff, D. (1976). Gravity and magnetic fields of polygonal prisms and application to magnetic terrain corrections.
  Geophysics 41, 727-741.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import G


def sphere(x, depth: float, radius: float, drho: float, y=0.0):
    """Buried sphere (cavity, ore body, boulder). ``depth`` is to the centre."""
    if depth <= radius:
        raise ValueError("depth to the centre must exceed the radius (the sphere must be buried)")
    x, y = np.asarray(x, float), np.asarray(y, float)
    mass = 4.0 / 3.0 * np.pi * radius**3 * drho
    r2 = x**2 + y**2 + depth**2
    return G * mass * depth / r2**1.5


def horizontal_cylinder(x, depth: float, radius: float, drho: float):
    """Infinite horizontal cylinder (tunnel, pipe, gallery) striking perpendicular to the profile.
    ``depth`` is to the axis."""
    if depth <= radius:
        raise ValueError("depth to the axis must exceed the radius")
    x = np.asarray(x, float)
    return 2.0 * np.pi * G * drho * radius**2 * depth / (x**2 + depth**2)


def _prism_kernel(dx, dy, dz):
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(dy + r > 0, dx * np.log(dy + r), 0.0)
        t2 = np.where(dx + r > 0, dy * np.log(dx + r), 0.0)
        t3 = np.where(dz * r != 0, dz * np.arctan(dx * dy / (dz * r)), 0.0)
    return t1 + t2 - t3


def prism(x, y, x1: float, x2: float, y1: float, y2: float, z1: float, z2: float, drho: float):
    """Right rectangular prism (room, foundation, buried block) with edges parallel to the axes.

    Exact solution (Nagy 1966; Plouff 1976). Horizontal extent [x1, x2] x [y1, y2], top ``z1`` and bottom ``z2`` below
    the surface (0 <= z1 < z2).
    """
    if not (x1 < x2 and y1 < y2 and 0 <= z1 < z2):
        raise ValueError("need x1 < x2, y1 < y2 and 0 <= z1 < z2")
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    total = np.zeros(x.shape)
    for i, xi in enumerate((x1, x2)):
        for j, yj in enumerate((y1, y2)):
            for k, zk in enumerate((z1, z2)):
                sign = (-1) ** (i + j + k)
                total += sign * _prism_kernel(xi - x, yj - y, zk)
    return G * drho * total


def _atan_ratio(num, den):
    """arctan(num / den), taken as 0 where den = 0 (those terms cancel in the vertex sum for points outside the body)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, np.arctan(num / np.where(den != 0, den, 1.0)), 0.0)


def prism_tensor(x, y, x1: float, x2: float, y1: float, y2: float, z1: float, z2: float):
    """Gravity gradient tensor of a right rectangular prism per unit G * drho, as an array (..., 3, 3) in (x, y, z down).

    T_ij = d g_i / d x_j for g = grad U, U = G drho * integral dV / |r - r'| (Nagy, Papp & Benedek 2000, J. Geodesy
    74, 552-560). Observation points on the surface z = 0 above the prism (0 <= z1 < z2; if z1 = 0 keep the points off
    the top face). Multiply by G * drho for s^-2; used by ``magnetics.prism`` through Poisson's relation.
    """
    if not (x1 < x2 and y1 < y2 and 0 <= z1 < z2):
        raise ValueError("need x1 < x2, y1 < y2 and 0 <= z1 < z2")
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    t = np.zeros(x.shape + (3, 3))
    for i, xi in enumerate((x1, x2)):
        for j, yj in enumerate((y1, y2)):
            for k, zk in enumerate((z1, z2)):
                s = (-1) ** (i + j + k)
                dx, dy, dz = xi - x, yj - y, np.full(x.shape, zk)
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                t[..., 0, 0] += s * _atan_ratio(dy * dz, dx * r)
                t[..., 1, 1] += s * _atan_ratio(dx * dz, dy * r)
                t[..., 2, 2] += s * _atan_ratio(dx * dy, dz * r)
                t[..., 0, 1] -= s * np.log(dz + r)
                t[..., 0, 2] -= s * np.log(dy + r)
                t[..., 1, 2] -= s * np.log(dx + r)
    t[..., 1, 0], t[..., 2, 0], t[..., 2, 1] = t[..., 0, 1], t[..., 0, 2], t[..., 1, 2]
    return t


# --- vertical cylinder ----------------------------------------------------------------------------------------
# The volume integral is done in closed form along the depth and along every ray from the station; what is left is one
# integral round the rim (Gauss's theorem in the horizontal plane). Each term is written as its difference from the
# value at zero distance, divided by the matching power of the distance, so the integrand is an analytic periodic
# function of the rim angle, also for stations right above the rim. The trapezoid rule then converges geometrically:
# the nearest complex singularity is at Im(theta) = arccosh(1 + ((R - d)^2 + z1^2) / (2 R d)), which sets the number of
# nodes. Small arguments use Taylor series to avoid cancellation.

_C = np.array([math.comb(2 * n, n) / 4.0**n for n in range(16)])          # coefficients of asinh and 1/sqrt(1 + x^2)
_S_SERIES = np.array([(-1) ** n * _C[n] / (2 * n + 1) for n in range(1, 13)])                 # (z asinh(x) - z x) / (z x)^3 * z^2
_C_SERIES = np.array([(-1) ** (n + 1) * _C[n] * 2 * n / (2 * n + 1) for n in range(1, 13)])  # (asinh x - x / sqrt(1+x^2)) / x^3
_E_SERIES = np.array([2 * (1 - 2.0**-n) / n for n in range(2, 18)])       # (2 ln((1+t)/(2t)) - 1 + t) / eps^2, t = 1 - eps
_L_SERIES = np.array([(1 - 2.0**-n) / n for n in range(1, 17)])           # ln((1+t)/(2t)) / eps


def _rim_terms(rho2, z: float) -> dict:
    """Per-depth terms of the rim integrand at squared distance ``rho2`` from the station, for a face at depth ``z``."""
    u = np.sqrt(rho2 + z * z)
    k = 1.0 / (u * (u + z))               # (1 - z/u) / rho^2
    eps = rho2 * k                        # 1 - z/u, computed without cancellation
    x2 = rho2 / (z * z)
    small, small_eps = x2 < 0.0625, eps < 0.1
    rho = np.sqrt(rho2)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_direct = (np.arcsinh(rho / z) - rho / u) / (rho2 * rho)
        ln_ratio = np.log1p(-eps / 2) - np.log1p(-eps)                       # ln((z + u) / (2 z))
        e_direct = (2 * ln_ratio - eps) / eps**2
        l_direct = ln_ratio / eps
    return {
        "gz": 1.0 / (u + z),                                                   # (u - z) / rho^2
        "tzz": k,
        "tzh": np.where(small, np.polynomial.polynomial.polyval(x2, _C_SERIES) / z**3, c_direct),
        "ww": np.where(small_eps, np.polynomial.polynomial.polyval(eps, _E_SERIES), e_direct) * k * k,
        "ii": np.where(small_eps, np.polynomial.polynomial.polyval(eps, _L_SERIES), l_direct) * k,
    }


def _rim_nodes(d, radius: float, z1: float) -> np.ndarray:
    """Trapezoid nodes per station for about 15 correct digits (a power of two, 16 to 65,536)."""
    with np.errstate(divide="ignore"):
        excess = ((radius - d) ** 2 + z1 * z1) / (2.0 * radius * d)        # inf on the axis
        a = 2.0 * np.arcsinh(np.sqrt(excess / 2.0))                          # arccosh(1 + excess), also when tiny
        n = np.clip(36.0 / a, 16, 65536)
    return (2 ** np.ceil(np.log2(n))).astype(int)


def _cylinder_rim(x, y, radius: float, z1: float, z2: float, tensor: bool):
    """gz (and the gradient tensor) of a vertical cylinder per unit G * drho, by the rim integral."""
    if not (radius > 0 and 0 <= z1 < z2):
        raise ValueError("need radius > 0 and 0 <= top < bottom")
    z1 = max(z1, 1e-9 * max(radius, z2))   # a top exactly at the surface: the limit, within 1e-9 of the body size
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    ox, oy = x.ravel(), y.ravel()
    gz = np.zeros(ox.shape)
    t = np.zeros(ox.shape + (3, 3))
    nodes = _rim_nodes(np.hypot(ox, oy), radius, z1)
    for n in np.unique(nodes):
        theta = (np.arange(n) + 0.5) * (2.0 * np.pi / n)       # half-step grid: never exactly at a station on the x axis
        cos, sin = np.cos(theta), np.sin(theta)
        members = np.flatnonzero(nodes == n)
        for chunk in np.array_split(members, max(1, len(members) * n // 400_000)):
            px, py = ox[chunk, None], oy[chunk, None]
            wx, wy = radius * cos - px, radius * sin - py         # station to rim
            rho2 = wx * wx + wy * wy
            dphi = (radius * radius - radius * (px * cos + py * sin)) * (2.0 * np.pi / n)   # rho^2 dphi / dtheta * h
            top, base = _rim_terms(rho2, z1), _rim_terms(rho2, z2)
            gz[chunk] = np.sum((top["gz"] - base["gz"]) * dphi, axis=1)
            if tensor:
                zh = (top["tzh"] - base["tzh"]) * dphi
                ww = (top["ww"] - base["ww"]) * dphi
                ii = (top["ii"] - base["ii"]) * dphi
                tc = t[chunk]
                tc[:, 2, 2] = np.sum((top["tzz"] - base["tzz"]) * dphi, axis=1)
                tc[:, 0, 2] = tc[:, 2, 0] = np.sum(zh * wx, axis=1)
                tc[:, 1, 2] = tc[:, 2, 1] = np.sum(zh * wy, axis=1)
                tc[:, 0, 0] = np.sum(ww * wx * wx - ii, axis=1)
                tc[:, 1, 1] = np.sum(ww * wy * wy - ii, axis=1)
                tc[:, 0, 1] = tc[:, 1, 0] = np.sum(ww * wx * wy, axis=1)
                t[chunk] = tc
    return gz.reshape(x.shape), t.reshape(x.shape + (3, 3))


def vertical_cylinder(x, top: float, bottom: float, radius: float, drho: float, y=0.0):
    """Vertical cylinder (shaft, sinkhole pipe, well) from ``top`` to ``bottom`` depth, axis at x = y = 0.

    Exact: the volume integral is reduced to one integral round the rim (closed form in depth and distance) and summed
    with the trapezoid rule to about 15 digits. A top at the surface (0) is allowed; right above the rim the sum then
    converges slowly and the value there is good to about 1e-6.
    """
    gz, _ = _cylinder_rim(x, y, radius, top, bottom, tensor=False)
    return G * drho * gz


def vertical_cylinder_tensor(x, y, radius: float, z1: float, z2: float):
    """Gravity gradient tensor of a vertical cylinder per unit G * drho, as an array (..., 3, 3) in (x, y, z down), with
    the same convention as ``prism_tensor``; used by ``magnetics.vertical_cylinder`` through Poisson's relation."""
    _, t = _cylinder_rim(x, y, radius, z1, z2, tensor=True)
    return t


def vertical_step(x, top: float, bottom: float, drho: float, edge: float = 0.0):
    """2D faulted slab / vertical step: a horizontal slab from ``top`` to ``bottom`` depth that exists for x > ``edge``
    and ends at a vertical contact at x = ``edge`` (fault, buried wall, edge of a fill).

    Far over the slab gz tends to the Bouguer value 2*pi*G*drho*t; far on the other side it tends to 0 (Telford et al.,
    §2.6).
    """
    if not 0 <= top < bottom:
        raise ValueError("need 0 <= top < bottom")
    u = np.asarray(x, float) - edge
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(u != 0, 0.5 * u * np.log((u**2 + bottom**2) / (u**2 + top**2)), 0.0)
    at_bottom = bottom * np.arctan2(u, bottom)
    at_top = np.where(top > 0, top * np.arctan2(u, top), 0.0)
    return 2.0 * G * drho * (log_term + np.pi / 2.0 * (bottom - top) + at_bottom - at_top)


def dyke(x, top: float, bottom: float, width: float, drho: float, centre: float = 0.0, dip: float = 90.0):
    """2D dyke (or buried wall, filled trench, vein) of horizontal ``width`` centred at ``centre``. A vertical dyke is
    the difference of two vertical steps; for ``dip`` below 90 degrees (towards +x) the parallelogram section is
    computed with ``polygon``."""
    if width <= 0:
        raise ValueError("width must be positive")
    if dip != 90.0:
        if not 0 < dip < 180:
            raise ValueError("dip must be within (0, 180) degrees")
        shift = (bottom - top) / np.tan(np.radians(dip))
        w = width / 2.0
        verts = [(centre - w, top), (centre + w, top), (centre + w + shift, bottom), (centre - w + shift, bottom)]
        return polygon(x, verts, drho)
    return (vertical_step(x, top, bottom, drho, centre - width / 2.0)
            - vertical_step(x, top, bottom, drho, centre + width / 2.0))


def polygon(x, vertices, drho: float):
    """2D body of polygonal cross-section striking perpendicular to the profile (Talwani, Worzel & Landisman 1959;
    closed form of Won & Bevis 1987, Geophysics 52, 232-238). ``vertices`` are (distance along profile, depth) pairs in
    either order; the body must lie below the surface."""
    v = np.asarray(vertices, float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        raise ValueError("need at least three (x, depth) vertices")
    if v[:, 1].min() < 0:
        raise ValueError("the body must lie below the surface")
    if np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]) < 0:
        v = v[::-1]                            # counter-clockwise in the (x, depth) plane
    x = np.asarray(x, float)
    total = np.zeros(x.shape)
    for (xa, za), (xb, zb) in zip(v, np.roll(v, -1, axis=0)):
        x1, x2 = xa - x, xb - x
        dx, dz = xb - xa, zb - za
        len2 = dx**2 + dz**2
        if len2 == 0:
            continue
        r1, r2 = np.hypot(x1, za), np.hypot(x2, zb)
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = np.arctan2(za * x2 - zb * x1, x1 * x2 + za * zb)   # angle from vertex a to vertex b
            log_term = np.where((r1 > 0) & (r2 > 0), np.log(np.where(r1 > 0, r2, 1.0) / np.where(r1 > 0, r1, 1.0)), 0.0)
        cross = x1 * zb - x2 * za
        total += cross / len2 * (dx * theta + dz * log_term)
    return 2.0 * G * drho * total
