import numpy as np
import pytest

from detectability import evaluate, gpr

SITE = {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"}


def test_attenuation_within_published_ranges():
    # Davis & Annan (1989) table (USACE 1995 Table 4-3): attenuation ranges in dB/m
    ranges = {"dry sand": (0.005, 0.02), "wet sand": (0.03, 0.3), "limestone": (0.4, 1.0), "shale": (1, 100),
              "silt": (1, 100), "clay": (1, 300), "granite": (0.01, 1), "rock salt": (0.01, 1), "fresh water": (0.05, 0.15)}
    for name, (lo, hi) in ranges.items():
        assert lo <= gpr.medium(name).attenuation_db(1e8) <= hi, name


def test_sea_water_is_opaque():
    assert gpr.medium("sea water").attenuation_db(1e8) > 100


def test_reflection_limits():
    lime = gpr.medium("limestone")
    assert abs(gpr.reflection(lime, lime, 1e8)) == pytest.approx(0, abs=1e-12)
    assert gpr.reflection(lime, None, 1e8) == -1
    # a faster (lower permittivity) target gives a positive coefficient, a slower one negative
    assert gpr.reflection(lime, gpr.medium("air"), 1e8).real > 0
    assert gpr.reflection(lime, gpr.medium("fresh water"), 1e8).real < 0


def test_size_factor_limits():
    r = np.array([1.0, 5.0, 20.0])
    lam = 0.5
    assert np.allclose(gpr.size_factor("sphere", r, lam, a=1e9), 1.0)
    assert np.allclose(gpr.size_factor("cylinder", r, lam, a=1e9), 1.0)
    assert np.allclose(gpr.size_factor("disc", r, lam, a=1e3), 1.0)
    small = gpr.size_factor("sphere", r, lam, a=0.01)
    assert np.all(small < 1e-3)
    # Rayleigh regime joins geometric optics continuously at k a = 1
    a1 = lam / (2 * np.pi)
    assert gpr.size_factor("sphere", r, lam, a=a1 * (1 - 1e-9)) == pytest.approx(gpr.size_factor("sphere", r, lam, a=a1))


def test_frequency_for_depth_follows_the_rule():
    assert gpr.frequency_for_depth(1.0) == 500
    assert gpr.frequency_for_depth(10.0) == 50
    assert gpr.frequency_for_depth(0.2) == 1000


def test_echo_falls_with_depth_and_attenuation():
    lime, clay = gpr.medium("limestone"), gpr.medium("clay")
    R = gpr.reflection(lime, gpr.medium("air"), 2e8)
    one = np.ones(2)
    e = gpr.echo(np.array([2.0, 8.0]), lime, R, one, 2e8)
    assert e[0] > e[1]
    assert gpr.echo(np.array([2.0]), clay, R, np.ones(1), 2e8)[0] < e[0]


def test_evaluate_gpr_void_in_limestone():
    rep = evaluate({"shape": "sphere", "depth_top": 3, "size": 1, "material": "air", "host": "limestone"}, SITE,
                   methods=["gpr"])
    r = rep.result("gpr")
    assert r.available and r.details["frequency_mhz"] == 200
    assert r.assessment.verdict == "clearly detectable"
    assert r.assessment.station_spacing <= r.details["vertical_resolution_m"]
    assert r.details["velocity_m_per_ns"] == pytest.approx(0.1224, abs=1e-3)
    assert "gpr (200 MHz)" in rep.summary()


def test_deep_target_in_clay_is_not_detectable():
    rep = evaluate({"shape": "sphere", "depth_top": 8, "size": 2, "material": "air", "host": "clay"}, SITE,
                   methods=["gpr"])
    assert rep.result("gpr").assessment.verdict == "not detectable"


def test_metal_pipe_and_given_properties():
    rep = evaluate({"shape": "horizontal_cylinder", "depth_top": 1, "size": 0.1, "material": "metal",
                    "host": "soil", "host_permittivity": 9, "host_conductivity": 5}, SITE, methods=["gpr", "gravity"])
    r = rep.result("gpr")
    assert r.available and r.contrast == pytest.approx(1.0)
    assert not rep.result("gravity").available


def test_unavailable_cases():
    contact = evaluate({"shape": "contact", "depth_top": 1, "depth_bottom": 5, "host": "limestone"}, SITE, methods=["gpr"])
    assert not contact.result("gpr").available
    unknown = evaluate({"shape": "sphere", "depth_top": 2, "size": 1, "host": "gabbro"}, SITE, methods=["gpr"])
    assert "permittivity" in unknown.result("gpr").reason


def test_lower_system_performance_lowers_snr():
    t = {"shape": "prism", "depth_top": 4, "depth_bottom": 6, "size": 3, "material": "air", "host": "limestone"}
    hi = evaluate(t, {**SITE, "gpr_system_performance": 120}, methods=["gpr"]).result("gpr").assessment.snr
    lo = evaluate(t, {**SITE, "gpr_system_performance": 80}, methods=["gpr"]).result("gpr").assessment.snr
    assert hi / lo == pytest.approx(100.0)  # 40 dB


def test_script_reproduces_gpr():
    rep = evaluate({"shape": "dyke", "depth_top": 2, "depth_bottom": 6, "size": 1, "material": "wet sand",
                    "host": "granite"}, {**SITE, "gpr_frequency": 250}, methods=["gpr"])
    ns = {}
    exec(rep.script().replace("print(report.summary())", "").split("# Optional plot")[0], ns)
    assert ns["report"].result("gpr").assessment.snr == pytest.approx(rep.result("gpr").assessment.snr)
