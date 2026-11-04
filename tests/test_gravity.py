import numpy as np
import pytest

from detectability import gravity
from detectability.constants import G

DRHO = -2600.0  # air-filled void in limestone, kg/m^3


def test_sphere_peak_equals_point_mass():
    z, R = 15.0, 5.0
    mass = 4 / 3 * np.pi * R**3 * DRHO
    assert gravity.sphere(0.0, z, R, DRHO) == pytest.approx(G * mass / z**2, rel=1e-12)


def test_sphere_half_width_rule():
    # For a sphere, gz falls to half its peak at x = 0.766 z (Telford et al. 1990, §2.6).
    z, R = 20.0, 4.0
    peak = gravity.sphere(0.0, z, R, DRHO)
    assert gravity.sphere(0.766 * z, z, R, DRHO) / peak == pytest.approx(0.5, abs=1e-3)


def test_horizontal_cylinder_half_width_equals_depth():
    z, R = 10.0, 2.0
    peak = gravity.horizontal_cylinder(0.0, z, R, DRHO)
    assert peak == pytest.approx(2 * np.pi * G * DRHO * R**2 / z)
    assert gravity.horizontal_cylinder(z, z, R, DRHO) / peak == pytest.approx(0.5)


def test_prism_far_field_matches_point_mass():
    a, z = 2.0, 100.0  # 4 m cube at 100 m: indistinguishable from a point mass
    g = gravity.prism(0.0, 0.0, -a, a, -a, a, z - a, z + a, DRHO)
    mass = (2 * a) ** 3 * DRHO
    assert g == pytest.approx(G * mass / z**2, rel=1e-4)


def test_prism_wide_thin_matches_bouguer_slab():
    t, L = 2.0, 1e5
    g = gravity.prism(0.0, 0.0, -L, L, -L, L, 10.0, 10.0 + t, 1000.0)
    assert g == pytest.approx(2 * np.pi * G * 1000.0 * t, rel=1e-3)


def test_prism_sign_and_symmetry():
    x = np.linspace(-50, 50, 101)
    g = gravity.prism(x, 0.0, -5, 5, -5, 5, 5, 15, 500.0)
    assert np.all(g > 0)
    assert np.allclose(g, g[::-1])
    assert np.argmax(g) == 50


def test_vertical_cylinder_exact_on_axis():
    top, bottom, R = 5.0, 30.0, 3.0
    exact = 2 * np.pi * G * DRHO * ((bottom - top) - np.hypot(bottom, R) + np.hypot(top, R))
    assert gravity.vertical_cylinder(0.0, top, bottom, R, DRHO) == pytest.approx(exact, rel=1e-12)


def test_vertical_cylinder_matches_cross_section_quadrature_also_above_the_rim():
    from detectability.validation import _disc_line_masses
    R = 3.0
    for top in (5.0, 0.3):                                   # deep, and shallow (sharp near the rim)
        x = np.array([0.0, 1.0, 2.5, 2.99, 3.0, 3.01, 3.5, 6.0, 20.0])
        ref = _disc_line_masses(x, R, top, 30.0, DRHO)
        assert np.allclose(gravity.vertical_cylinder(x, top, 30.0, R, DRHO), ref, rtol=1e-10, atol=0)


def test_vertical_cylinder_limits():
    R, top, bottom = 3.0, 5.0, 30.0
    x = np.array([2000.0, 4000.0])                           # far: a point mass
    mass = np.pi * R**2 * (bottom - top) * DRHO
    zc = (top + bottom) / 2
    point = G * mass * zc / (x**2 + zc**2) ** 1.5
    assert np.allclose(gravity.vertical_cylinder(x, top, bottom, R, DRHO), point, rtol=1e-4)
    slab = gravity.vertical_cylinder(0.0, 10.0, 12.0, 1e7, 1000.0)   # very wide: the Bouguer slab, to (z1+z2)/2R
    assert slab == pytest.approx(2 * np.pi * G * 1000.0 * 2.0, rel=1e-5)
    off = gravity.vertical_cylinder(np.array([0.0, 7.0]), top, bottom, R, DRHO, y=np.array([7.0, 0.0]))
    assert off[0] == pytest.approx(off[1], rel=1e-13)       # axisymmetric


def test_vertical_cylinder_open_at_the_surface():
    from detectability.validation import _disc_line_masses
    axis = 2 * np.pi * G * DRHO * (20.0 - np.hypot(20.0, 3.0) + 3.0)
    assert gravity.vertical_cylinder(0.0, 0.0, 20.0, 3.0, DRHO) == pytest.approx(axis, rel=1e-8)   # top kept 1e-9 below
    x = np.array([3.2, 5.0, 12.0])                           # outside the opening
    assert np.allclose(gravity.vertical_cylinder(x, 0.0, 20.0, 3.0, DRHO), _disc_line_masses(x, 3.0, 0.0, 20.0, DRHO),
                       rtol=1e-10)
    rim = gravity.vertical_cylinder(np.array([2.999, 3.0, 3.001]), 0.0, 20.0, 3.0, DRHO)   # steep but continuous
    assert np.all(np.isfinite(rim)) and rim[0] < rim[1] < rim[2]
    assert abs(rim[1] - (rim[0] + rim[2]) / 2) < 0.01 * np.ptp(rim)


def test_vertical_cylinder_tensor():
    x, y = np.linspace(-20, 20, 41), 0.7
    t = gravity.vertical_cylinder_tensor(x, y, 3.0, 5.0, 30.0)
    assert np.allclose(t, np.swapaxes(t, -1, -2))
    assert np.max(np.abs(np.trace(t, axis1=-2, axis2=-1))) < 1e-12 * np.max(np.abs(t))
    h = 1e-3
    gz = lambda xx, yy, a, b: gravity.vertical_cylinder(xx, a, b, 3.0, 1.0, yy) / G
    scale = np.max(np.abs(t[..., 2, 2]))
    assert np.max(np.abs(t[..., 2, 2] - (gz(x, y, 5 - h, 30 - h) - gz(x, y, 5 + h, 30 + h)) / (2 * h))) < 1e-6 * scale
    assert np.max(np.abs(t[..., 2, 0] - (gz(x + h, y, 5, 30) - gz(x - h, y, 5, 30)) / (2 * h))) < 1e-6 * scale
    assert np.max(np.abs(t[..., 2, 1] - (gz(x, y + h, 5, 30) - gz(x, y - h, 5, 30)) / (2 * h))) < 1e-6 * scale


def test_vertical_step_limits():
    top, bottom, drho = 5.0, 25.0, 300.0
    t = bottom - top
    bouguer = 2 * np.pi * G * drho * t
    assert gravity.vertical_step(1e6, top, bottom, drho) == pytest.approx(bouguer, rel=1e-4)
    assert gravity.vertical_step(-1e6, top, bottom, drho) == pytest.approx(0.0, abs=bouguer * 1e-4)
    assert gravity.vertical_step(0.0, top, bottom, drho) == pytest.approx(bouguer / 2, rel=1e-12)


def test_vertical_step_outcropping_slab():
    # top = 0: the slab reaches the surface; the formula must stay finite.
    g = gravity.vertical_step(np.array([-10.0, 0.0, 10.0]), 0.0, 10.0, 200.0)
    assert np.all(np.isfinite(g))


def test_dyke_symmetric_and_matches_prism_when_long():
    x = np.linspace(-40, 40, 81)
    g2d = gravity.dyke(x, 2.0, 12.0, 4.0, 400.0)
    assert np.allclose(g2d, g2d[::-1])
    g3d = gravity.prism(x, 0.0, -2.0, 2.0, -1e5, 1e5, 2.0, 12.0, 400.0)
    assert np.allclose(g2d, g3d, rtol=1e-3)


def test_buried_checks():
    with pytest.raises(ValueError):
        gravity.sphere(0.0, 3.0, 5.0, DRHO)
    with pytest.raises(ValueError):
        gravity.prism(0.0, 0.0, 1, 0, 0, 1, 0, 1, DRHO)


def test_prism_tensor_traceless_symmetric():
    x = np.linspace(-30, 30, 61)
    t = gravity.prism_tensor(x, 3.0, -4, 6, -2, 5, 2.0, 9.0)
    assert np.allclose(np.trace(t, axis1=-2, axis2=-1), 0.0, atol=1e-9)
    assert np.allclose(t, np.swapaxes(t, -1, -2))


def test_prism_tensor_matches_finite_differences_of_gz():
    x, y, d = np.linspace(-25, 25, 51), 1.5, 1e-4
    box = (-4.0, 6.0, -2.0, 5.0, 2.0, 9.0)
    gz = lambda xx, yy, dz=0.0: gravity.prism(xx, yy, *box[:4], box[4] + dz, box[5] + dz, 1.0) / G
    t = gravity.prism_tensor(x, y, *box)
    assert np.allclose(t[:, 2, 2], (gz(x, y, -d) - gz(x, y, d)) / (2 * d), rtol=0, atol=1e-6)  # observer down = body up
    assert np.allclose(t[:, 2, 0], (gz(x + d, y) - gz(x - d, y)) / (2 * d), rtol=0, atol=1e-6)
    assert np.allclose(t[:, 2, 1], (gz(x, y + d) - gz(x, y - d)) / (2 * d), rtol=0, atol=1e-6)


def test_polygon_matches_dyke_and_cylinder():
    x = np.linspace(-40, 40, 81)
    rect = [(5, 3), (9, 3), (9, 15), (5, 15)]
    assert np.allclose(gravity.polygon(x, rect, 500.0), gravity.dyke(x, 3, 15, 4, 500.0, centre=7), rtol=1e-10)
    assert np.allclose(gravity.polygon(x, rect[::-1], 500.0), gravity.polygon(x, rect, 500.0))
    n, R, z = 720, 2.0, 8.0
    Rn = R * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack([Rn * np.cos(a), z + Rn * np.sin(a)])
    assert np.allclose(gravity.polygon(x, circle, 500.0), gravity.horizontal_cylinder(x, z, R, 500.0), rtol=1e-4)


def test_dipping_dyke_shifts_anomaly_down_dip():
    x = np.linspace(-60, 60, 241)
    g = gravity.dyke(x, 2.0, 30.0, 3.0, 400.0, dip=45.0)
    assert x[np.argmax(g)] > 0
