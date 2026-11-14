import numpy as np
import pytest

from detectability import gravity, magnetics
from detectability.constants import G
from detectability.magnetics import MU0, Ambient

POLE = Ambient(50000.0, 90.0, 0.0)        # vertical field
MID = Ambient(48567.0, 58.373, 6.313)     # Sivas, 2026 (IGRF-14)
CHI = 0.01


def test_sphere_at_pole_peak_and_symmetry():
    z, R = 10.0, 2.0
    x = np.linspace(-50, 50, 201)
    t = magnetics.sphere(x, z, R, CHI, POLE)
    m = CHI * POLE.total * 1e-9 / MU0 * 4 / 3 * np.pi * R**3
    assert t[100] == pytest.approx(MU0 / (4 * np.pi) * 2 * m / z**3 * 1e9, rel=1e-12)
    assert np.allclose(t, t[::-1])


def test_sphere_profile_direction():
    # With declination 0 the field lies in the north-down plane: an east-west profile is symmetric, a north-south one is not.
    amb = Ambient(48000.0, 55.0, 0.0)
    x = np.linspace(-40, 40, 161)
    ew = magnetics.sphere(x, 8.0, 2.0, CHI, amb, azimuth=90.0)
    ns = magnetics.sphere(x, 8.0, 2.0, CHI, amb, azimuth=0.0)
    assert np.allclose(ew, ew[::-1])
    assert not np.allclose(ns, ns[::-1])


def _poisson(dgz_dz, chi, ambient, drho):
    """Poisson's relation for vertical field and magnetisation: dT = mu0 M / (4 pi G drho) * dgz/dz."""
    m = chi * ambient.total * 1e-9 / MU0
    return MU0 * m / (4 * np.pi * G * drho) * dgz_dz * 1e9


def test_sphere_poisson_relation():
    x, z, R, drho, d = np.linspace(-60, 60, 121), 12.0, 3.0, 1000.0, 1e-3
    dg = (gravity.sphere(x, z - d, R, drho) - gravity.sphere(x, z + d, R, drho)) / (2 * d)
    assert np.allclose(magnetics.sphere(x, z, R, CHI, POLE), _poisson(dg, CHI, POLE, drho), rtol=1e-5)


def test_horizontal_cylinder_equals_integrated_dipoles():
    # A 2D line dipole must equal 3D dipoles summed along strike (strike is perpendicular to the profile).
    z, R, az = 6.0, 1.0, 30.0
    x = np.linspace(-30, 30, 13)
    t2d = magnetics.horizontal_cylinder(x, z, R, CHI, MID, azimuth=az)
    m_per_m = magnetics.magnetisation(CHI, MID) * np.pi * R**2
    f = magnetics.unit_vector(MID.inclination, MID.declination)
    a = np.radians(az)
    along, strike = np.array([np.cos(a), np.sin(a), 0.0]), np.array([-np.sin(a), np.cos(a), 0.0])
    theta = np.linspace(-np.pi / 2, np.pi / 2, 40001)[1:-1]
    s, ds = z * np.tan(theta), z / np.cos(theta) ** 2 * (theta[1] - theta[0])
    t3d = []
    for xi in x:
        r = xi * along - s[:, None] * strike - np.array([0.0, 0.0, z])
        dist = np.linalg.norm(r, axis=1, keepdims=True)
        rhat = r / dist
        b = MU0 / (4 * np.pi) * (3 * (rhat @ m_per_m)[:, None] * rhat - m_per_m) / dist**3
        t3d.append(np.sum((b @ f) * ds) * 1e9)
    assert np.allclose(t2d, t3d, rtol=1e-4)


def test_polygon_circle_matches_cylinder():
    z, R, n = 8.0, 2.0, 720
    Rn = R * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))  # polygon with the circle's area
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = np.column_stack([Rn * np.cos(ang), z + Rn * np.sin(ang)])
    x = np.linspace(-40, 40, 81)
    cyl = magnetics.horizontal_cylinder(x, z, R, CHI, MID, azimuth=20.0)
    poly = magnetics.polygon(x, verts, CHI, MID, azimuth=20.0)
    assert np.allclose(poly, cyl, rtol=1e-4, atol=1e-6 * np.abs(cyl).max())
    assert np.allclose(magnetics.polygon(x, verts[::-1], CHI, MID, azimuth=20.0), poly)


def test_dyke_poisson_relation():
    x, top, bottom, w, drho, d = np.linspace(-50, 50, 101), 3.0, 20.0, 4.0, 1000.0, 1e-4
    dg = (gravity.dyke(x, top - d, bottom - d, w, drho) - gravity.dyke(x, top + d, bottom + d, w, drho)) / (2 * d)
    t = magnetics.dyke(x, top, bottom, w, CHI, POLE)
    assert np.allclose(t, _poisson(dg, CHI, POLE, drho), rtol=0, atol=1e-6 * np.abs(t).max())


def test_contact_poisson_relation():
    x, top, bottom, drho, d = np.linspace(-80, 80, 161), 5.0, 40.0, 1000.0, 1e-4
    dg = (gravity.vertical_step(x, top - d, bottom - d, drho) - gravity.vertical_step(x, top + d, bottom + d, drho)) / (2 * d)
    t = magnetics.contact(x, top, bottom, CHI, POLE)
    assert np.allclose(t, _poisson(dg, CHI, POLE, drho), rtol=0, atol=1e-6 * np.abs(t).max())


def test_dipping_dyke_is_asymmetric_and_vertical_is_symmetric():
    x = np.linspace(-40, 40, 81)
    vertical = magnetics.dyke(x, 2.0, 15.0, 3.0, CHI, POLE)
    dipping = magnetics.dyke(x, 2.0, 15.0, 3.0, CHI, POLE, dip=45.0)
    assert np.allclose(vertical, vertical[::-1])
    assert not np.allclose(dipping, dipping[::-1])


def test_remanence_adds_linearly():
    x = np.linspace(-20, 20, 41)
    rem = (0.5, 40.0, 10.0)
    induced = magnetics.sphere(x, 5.0, 1.0, CHI, MID)
    remanent_only = magnetics.sphere(x, 5.0, 1.0, 0.0, MID, remanence=rem)
    both = magnetics.sphere(x, 5.0, 1.0, CHI, MID, remanence=rem)
    assert np.allclose(both, induced + remanent_only)


def test_checks():
    with pytest.raises(ValueError):
        magnetics.sphere(0.0, 1.0, 2.0, CHI, MID)
    with pytest.raises(ValueError):
        magnetics.dyke(0.0, 0.0, 5.0, 1.0, CHI, MID)  # outcropping body with the sensor on the ground
    magnetics.dyke(np.array([0.0]), 0.0, 5.0, 1.0, CHI, MID, height=0.3)  # fine with the sensor above ground


def _dipole_quadrature(x, box, chi, ambient, azimuth, height=0.0, n=16):
    """Total-field anomaly of a prism as a Gauss-Legendre sum of point dipoles (independent check)."""
    (a1, a2), (c1, c2), (z1, z2) = box
    nodes, weights = np.polynomial.legendre.leggauss(n)
    pts = [((hi - lo) / 2 * nodes + (hi + lo) / 2, (hi - lo) / 2 * weights) for lo, hi in box]
    az = np.radians(azimuth)
    along, across = np.array([np.cos(az), np.sin(az), 0.0]), np.array([-np.sin(az), np.cos(az), 0.0])
    m = magnetics.magnetisation(chi, ambient)
    f = magnetics.unit_vector(ambient.inclination, ambient.declination)
    A, C, Z = np.meshgrid(pts[0][0], pts[1][0], pts[2][0], indexing="ij")
    W = np.einsum("i,j,k->ijk", pts[0][1], pts[1][1], pts[2][1]).ravel()
    src = A.ravel()[:, None] * along + C.ravel()[:, None] * across + Z.ravel()[:, None] * np.array([0, 0, 1.0])
    out = []
    for xi in np.atleast_1d(x):
        r = xi * along - np.array([0, 0, height]) - src
        dist = np.linalg.norm(r, axis=1, keepdims=True)
        rhat = r / dist
        b = MU0 / (4 * np.pi) * (3 * (rhat @ m)[:, None] * rhat - m) / dist**3
        out.append(np.sum((b @ f) * W) * 1e9)
    return np.array(out)


def test_prism_matches_dipole_quadrature():
    x = np.linspace(-30, 30, 25)
    box = ((-3.0, 2.0), (-2.5, 3.5), (4.0, 7.0))
    t = magnetics.prism(x, *box[0], *box[1], *box[2], CHI, MID, azimuth=30.0, height=0.5)
    q = _dipole_quadrature(x, box, CHI, MID, 30.0, height=0.5)
    assert np.allclose(t, q, rtol=0, atol=1e-5 * np.abs(q).max())


def test_prism_far_field_matches_sphere():
    a, z = 1.0, 60.0
    x = np.linspace(-100, 100, 41)
    t = magnetics.prism(x, -a, a, -a, a, z - a, z + a, CHI, MID, azimuth=15.0)
    R = (3 * (2 * a) ** 3 / (4 * np.pi)) ** (1 / 3)
    s = magnetics.sphere(x, z, R, CHI, MID, azimuth=15.0)
    assert np.allclose(t, s, rtol=0, atol=1e-3 * np.abs(s).max())


def test_dipping_dyke_poisson_relation():
    # Gravity and magnetic polygon codes are independent; at the pole Poisson's relation links them.
    x, top, bottom, w, dip, drho, d = np.linspace(-50, 50, 101), 3.0, 20.0, 4.0, 60.0, 1000.0, 1e-4
    dg = (gravity.dyke(x, top - d, bottom - d, w, drho, dip=dip) - gravity.dyke(x, top + d, bottom + d, w, drho, dip=dip)) / (2 * d)
    t = magnetics.dyke(x, top, bottom, w, CHI, POLE, dip=dip)
    assert np.allclose(t, _poisson(dg, CHI, POLE, drho), rtol=0, atol=1e-6 * np.abs(t).max())


def test_vertical_cylinder_matches_dipole_quadrature():
    from detectability.validation import _cylinder_dipoles
    x = np.array([-25.0, -6.0, -3.0, 0.0, 2.0, 3.2, 8.0, 30.0])
    ours = magnetics.vertical_cylinder(x, 4.0, 25.0, 2.5, 0.02, MID, azimuth=120.0, height=0.3)
    ref = _cylinder_dipoles(x, 2.5, 4.3, 25.3, 0.02, MID, 120.0, n=40)
    assert np.allclose(ours, ref, rtol=1e-9, atol=1e-12 * np.max(np.abs(ref)))


def test_vertical_cylinder_far_field_is_a_dipole():
    R, top, bottom = 2.0, 10.0, 14.0                       # a squat cylinder seen from far away
    x = np.array([400.0, 800.0])
    volume = np.pi * R**2 * (bottom - top)
    dipole = magnetics.sphere(x, (top + bottom) / 2, (3 * volume / (4 * np.pi)) ** (1 / 3), 0.01, MID)
    assert np.allclose(magnetics.vertical_cylinder(x, top, bottom, R, 0.01, MID), dipole, rtol=1e-3)


# --- steel casing as a pair of poles (Frischknecht and others, 1983, USGS Open-File Report 83-702) --------------------

FT = 0.3048


def test_pole_pair_on_axis_at_the_pole_is_the_closed_form():
    p, top, sep, h = 600.0, 3.0, 150.0, 40.0
    expected = MU0 / (4 * np.pi) * p * (1 / (h + top) ** 2 - 1 / (h + top + sep) ** 2) * 1e9
    assert magnetics.pole_pair(0.0, p, top, sep, POLE, height=h) == pytest.approx(expected, rel=1e-12)
    assert expected > 0                    # magnetised downwards: positive above the casing


def test_pole_pair_of_a_long_casing_falls_as_the_inverse_square():
    near, far = (magnetics.pole_pair(0.0, 600.0, 0.0, 1e7, POLE, height=h) for h in (25.0, 50.0))
    assert near / far == pytest.approx(4.0, rel=1e-6)


def test_pole_pair_is_a_thin_cylinder_magnetised_along_its_axis():
    """Two codes that share nothing: the cylinder's gradient tensor, and two point poles of strength M * area."""
    radius, top, bottom, m = 0.05, 3.0, 103.0, 4000.0
    x = np.linspace(-80.0, 80.0, 41)
    cylinder = magnetics.vertical_cylinder(x, top, bottom, radius, 0.0, MID, azimuth=35.0, height=30.0,
                                           remanence=(m, 90.0, 0.0))
    poles = magnetics.pole_pair(x, m * np.pi * radius**2, top, bottom - top, MID, azimuth=35.0, height=30.0)
    assert np.allclose(poles, cylinder, rtol=0, atol=1e-5 * np.abs(cylinder).max())


def test_pole_pair_offset_is_a_parallel_line():
    """A line 20 m to the side of the well is the same as moving the well 20 m the other way: checked against the
    field of the two poles written out for a north-south line 20 m east of the casing."""
    p, top, sep, h, off = 800.0, 4.0, 60.0, 35.0, 20.0
    x = np.linspace(-100.0, 100.0, 21)
    f = magnetics.unit_vector(MID.inclination, MID.declination)

    def pole(strength, depth):
        dz = -(h + depth)
        return MU0 / (4 * np.pi) * strength * (f[0] * x + f[1] * off + f[2] * dz) / (x**2 + off**2 + dz**2) ** 1.5

    expected = (pole(-p, top) + pole(p, top + sep)) * 1e9
    assert np.allclose(magnetics.pole_pair(x, p, top, sep, MID, height=h, offset=off), expected, rtol=1e-12)
    assert magnetics.pole_pair(0.0, p, top, sep, MID, height=h, offset=off) < magnetics.pole_pair(0.0, p, top, sep, MID, height=h)


def test_pole_pair_reproduces_the_calculated_profiles_of_the_report():
    """83-702 Table 2 gives the pole pairs fitted to ground surveys of wells near Denver (hybrid units, divided by
    1076.4 for SI, distances in feet) and Figures 96, 97 and 101 plot the airborne profiles calculated from them. The
    numbers here are the ends of those figures' vertical axes. This compares two implementations of one model; it is
    not a comparison with a measurement."""
    def field(north, down):
        return Ambient(float(np.hypot(north, down)), float(np.degrees(np.arctan2(down, north))), 0.0)

    x = np.linspace(-800.0, 800.0, 6401) * FT
    well4 = field(21324.0, 50902.0)
    for azimuth, plotted in ((0.0, 55211.2 - 55187.5), (90.0, 55210.9 - 55188.1)):        # Figures 96 and 97, 200 ft
        t = (magnetics.pole_pair(x, 2158930 / 1076.4, 18.2 * FT, 12.3 * FT, well4, azimuth=azimuth, height=200 * FT)
             + magnetics.pole_pair(x, 1010510 / 1076.4, 20.7 * FT, 979.2 * FT, well4, azimuth=azimuth, height=200 * FT))
        assert t.max() - t.min() == pytest.approx(plotted, abs=0.5)
    well5 = field(21331.0, 50928.0)
    x = np.linspace(-400.0, 400.0, 3201) * FT
    t = magnetics.pole_pair(x, 311960 / 1076.4, 6.3 * FT, 263.6 * FT, well5, azimuth=90.0, height=150 * FT)
    assert t.max() - t.min() == pytest.approx(55225.0 - 55214.7, abs=0.3)                 # Figure 101, 150 ft


def test_pole_pair_refuses_what_it_cannot_model():
    with pytest.raises(ValueError):
        magnetics.pole_pair(0.0, 100.0, 2.0, 0.0, POLE)
    with pytest.raises(ValueError):
        magnetics.pole_pair(0.0, 100.0, 0.0, 50.0, POLE, height=0.0)

