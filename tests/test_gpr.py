import numpy as np
import pytest

from detectability import evaluate, gpr

SITE = {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"}
CAVITY = {"shape": "sphere", "size": 2.0, "material": "air", "host": "dry sand"}


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


def test_the_script_pins_the_antenna_it_actually_used():
    """Every other setting is written out; an antenna worked out from the depth has to be written out too.

    Otherwise the script's answer depends on a rule that lives in the library, and a later release that changed the
    rule would quietly give a different result from the one the script was made to reproduce.
    """
    report = evaluate({**CAVITY, "depth_top": 1.2}, SITE, methods=["gpr"])
    used = report.result("gpr").details["frequency_mhz"]
    assert report.result("gpr").details["frequency_source"] == "auto"
    assert f'"gpr_frequency": {used}' in report.script()


def test_an_antenna_that_was_asked_for_is_marked_as_given():
    report = evaluate({**CAVITY, "depth_top": 1.2}, {**SITE, "gpr_frequency": 250}, methods=["gpr"])
    assert report.result("gpr").details["frequency_source"] == "given"
    assert report.result("gpr").details["frequency_mhz"] == 250


def test_the_pinned_script_reproduces_the_same_answer():
    """Run what the script would run, and it must land on the same signal-to-noise."""
    report = evaluate({**CAVITY, "depth_top": 1.2}, SITE, methods=["gpr"])
    survey = {**SITE, "gpr_frequency": report.result("gpr").details["frequency_mhz"]}
    again = evaluate({**CAVITY, "depth_top": 1.2}, survey, methods=["gpr"])
    assert again.result("gpr").assessment.snr == report.result("gpr").assessment.snr


def test_attenuation_is_frequency_independent_in_this_model():
    """The host's properties are taken as frequency-independent, so attenuation is too, in the low-loss limit.

    It is worth pinning: it means the depth limit is set by the ground rather than by the antenna, which is why the
    page says so, and it is also the model's weakest point against real soils, which lose more at high frequency.
    """
    target = {"shape": "sphere", "depth_top": 15, "size": 20, "material": "air", "host": "limestone"}
    att = [evaluate(target, {**SITE, "gpr_frequency": f}, methods=["gpr"]).result("gpr").details["attenuation_db_per_m"]
           for f in (25, 100, 400, 1000)]
    assert max(att) - min(att) < 0.01 * max(att)


# --- what frequency does, and does not do, in this model ------------------------------------------------------------

def _gpr(target, frequency=None):
    survey = {**SITE, **({"gpr_frequency": frequency} if frequency else {})}
    return evaluate(target, survey, methods=["gpr"]).result("gpr")


def test_every_result_states_the_attenuation_of_its_own_host_at_both_ends_of_the_band():
    """The statement is general, the numbers are the host's own: nothing here is copied from another case."""
    for host in ("limestone", "clay", "dry sand"):
        r = _gpr({"shape": "sphere", "depth_top": 2, "size": 2, "material": "air", "host": host})
        medium = gpr.medium(host)
        low, high = medium.attenuation_db(25e6), medium.attenuation_db(1e9)
        assert r.details["attenuation_db_per_m_25mhz"] == pytest.approx(low)
        assert r.details["attenuation_db_per_m_1ghz"] == pytest.approx(high)
        line = next(a for a in r.assessment.assumptions if a.startswith("Frequency is not modelled"))
        assert f"{low:.3g} dB/m at 25 MHz" in line and f"{high:.3g} dB/m at 1 GHz" in line
        assert "not only above 500 MHz" in line


def test_attenuation_does_rise_with_frequency_in_a_conductive_host():
    """Why the wording is not "attenuation is frequency-independent": that holds in a low-loss host only. In clay the
    conduction loss of the exact wavenumber is 17 % higher at 1 GHz than at 25 MHz."""
    clay = gpr.medium("clay")
    assert clay.attenuation_db(1e9) / clay.attenuation_db(25e6) == pytest.approx(1.173, abs=0.002)


def test_the_antenna_moves_the_depth_limit_only_through_the_target_size_factor():
    """Measured, so the sentence in every result stays true: a large target is indifferent to the antenna, a small one
    is seen deeper at the higher frequency (the opposite of real ground, which is what the next assumption says)."""
    from detectability import depth_limit

    def limit(size, frequency):
        return depth_limit({"shape": "sphere", "size": size, "material": "air", "host": "limestone"},
                           {**SITE, "gpr_frequency": frequency}, method="gpr", search=(0.05, 300)).depth

    assert limit(5.0, 25) == pytest.approx(limit(5.0, 1000), rel=0.01)       # 20.35 m and 20.30 m
    assert limit(0.2, 1000) > 4 * limit(0.2, 25)                             # 9.7 m against 2.2 m
