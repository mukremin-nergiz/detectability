import numpy as np
import pytest

from detectability import detect, gravity, igrf, magnetics, rocks
from detectability.constants import SI_TO_UGAL


def test_void_in_limestone_example():
    # "A 20 m wide void at 15 m depth in limestone": sphere of radius 10 m, top at 15 m -> centre at 25 m.
    x = np.linspace(-150, 150, 601)
    drho = rocks.contrast("air", "limestone")
    g = gravity.sphere(x, 25.0, 10.0, drho) * SI_TO_UGAL
    noise = detect.effective_noise(detect.GRAVIMETERS_UGAL["Scintrex CG-6"], detect.DEFAULT_FIELD_NOISE_UGAL)
    a = detect.assess(x, g, noise, depth=15.0)
    assert a.shape == "bell"
    assert a.peak == pytest.approx(abs(g).max())
    # G * (4/3 pi R^3 drho) / z^2 = 6.674e-11 * 1.068e7 kg / 625 m^2 = 1.14e-6 m/s^2 = 114 µGal (checked by hand)
    assert a.peak == pytest.approx(114.07, rel=1e-3)
    assert a.snr == pytest.approx(114.07 / noise, rel=1e-3)
    assert a.verdict == "clearly detectable"
    # FWHM of a sphere is 2 * 0.766 * z
    assert a.width == pytest.approx(2 * 0.766 * 25.0, rel=0.01)
    assert a.station_spacing <= a.width / detect.STATIONS_ACROSS
    assert a.profile_length >= 6 * 15.0


def test_small_deep_target_not_detectable():
    x = np.linspace(-200, 200, 801)
    g = gravity.sphere(x, 40.0, 1.0, -2550.0) * SI_TO_UGAL  # 2 m cavity at 40 m
    a = detect.assess(x, g, detect.effective_noise(5, 10), depth=39.0)
    assert a.verdict == "not detectable"


def test_step_shape_and_amplitude():
    x = np.linspace(-300, 300, 1201)
    g = gravity.vertical_step(x, 10.0, 60.0, 200.0) * SI_TO_UGAL
    a = detect.assess(x, g, 11.2, depth=10.0)
    assert a.shape == "step"
    assert a.peak == pytest.approx(g.max() - g.min())


def test_nice_spacing():
    assert detect._nice(7.3) == 5.0
    assert detect._nice(3.1) == 2.0
    assert detect._nice(1.4) == 1.0
    assert detect._nice(0.37) == 0.2


def test_noise_quadrature():
    assert detect.effective_noise(3, 4) == pytest.approx(5)


def test_void_in_limestone_magnetic_example():
    # The same void seen by a magnetometer in Sivas: limestone is almost non-magnetic, so the anomaly is well below noise.
    x = np.linspace(-150, 150, 601)
    amb = magnetics.Ambient.from_igrf(igrf.field(39.75, 37.02, 2026.5))
    chi = rocks.susceptibility_contrast("air", "limestone")
    t = magnetics.sphere(x, 25.0, 10.0, chi, amb, height=0.3)
    noise = detect.effective_noise(detect.MAGNETOMETERS_NT["Overhauser"], detect.DEFAULT_FIELD_NOISE_NT)
    a = detect.assess(x, t, noise, depth=15.0, method="magnetic")
    assert a.units == "nT"
    assert a.peak < 1.0
    assert a.verdict == "not detectable"


def test_instrument_presets_have_sources():
    assert detect.INSTRUMENTS
    for name, inst in detect.INSTRUMENTS.items():
        assert inst.method in ("gravity", "magnetic", "gradient"), name
        assert inst.noise > 0 and inst.source.startswith("https://") and inst.reference, name
    assert detect.GRAVIMETERS_UGAL["Scintrex CG-6"] == 5.0
    assert detect.MAGNETOMETERS_NT["Overhauser"] == 0.022
    assert "Bartington Grad601" not in detect.MAGNETOMETERS_NT  # a gradiometer reading, not the total field


def test_gradiometer_presets_have_a_baseline_and_a_source():
    grads = {k: v for k, v in detect.INSTRUMENTS.items() if v.method == "gradient"}
    assert len(grads) >= 3 and set(grads) == set(detect.GRADIOMETERS_NT)
    for name, inst in grads.items():
        assert inst.baseline and 0.3 <= inst.baseline <= 1.5, name
        assert inst.source.startswith("https://") and inst.noise > 0, name
    assert all(v.baseline is None for v in detect.INSTRUMENTS.values() if v.method != "gradient")
