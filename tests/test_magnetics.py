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
