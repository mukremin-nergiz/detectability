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


def test_vertical_cylinder_close_to_exact_on_axis():
    top, bottom, R = 5.0, 30.0, 3.0
    exact = 2 * np.pi * G * DRHO * ((bottom - top) - np.hypot(bottom, R) + np.hypot(top, R))
    assert gravity.vertical_cylinder(0.0, top, bottom, R, DRHO) == pytest.approx(exact, rel=0.03)


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
