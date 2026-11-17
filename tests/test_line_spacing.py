"""Line spacing: a compact target is assessed where a survey of parallel lines reads least, not on a line through it.

No new physics is involved: the compact shapes were already computed in three dimensions and the across-line
coordinate was fixed at zero. The tests check that coordinate (against a rotated profile and against the independent
pole-pair code), the search for the worst place between two lines, and the limit that inverts it.
"""

import numpy as np
import pytest

from detectability import SpacingLimit, depth_limit, detect, evaluate, gravity, magnetics, spacing_limit
from detectability.survey import LINE_SPACING_CAVEAT

AMBIENT = magnetics.Ambient(48000.0, 61.0, 13.0)
SURVEY = {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"}
CAVITY = {"shape": "sphere", "depth_top": 5, "size": 6, "material": "air", "host": "limestone"}
# A cut-off steel casing seen from a drone. The susceptibility contrast is a stand-in that gives a signal well above
# the noise; these tests are about geometry and no figure here is a prediction for steel.
CASING = {"shape": "vertical_cylinder", "depth_top": 1.5, "depth_bottom": 301.5, "size": 0.2, "material": "air",
          "host": "soil", "susceptibility_contrast": 3000.0}
DRONE = {**SURVEY, "sensor_height": 35.0}


def assessment(target, survey, method):
    return evaluate(target, survey, methods=[method]).result(method).assessment


# --- the across-line coordinate of the forward models -----------------------------------------------------------------

@pytest.mark.parametrize("model, args", [
    (magnetics.sphere, (8.0, 3.0, 0.01)),
    (magnetics.prism, (-2.0, 2.0, -3.0, 3.0, 1.0, 6.0, 0.01)),
    (magnetics.vertical_cylinder, (1.0, 40.0, 0.5, 0.01)),
])
def test_zero_offset_is_the_profile_through_the_body(model, args):
    x = np.linspace(-50, 50, 201)
    np.testing.assert_array_equal(model(x, *args, AMBIENT, azimuth=25.0, height=1.0),
                                  model(x, *args, AMBIENT, azimuth=25.0, height=1.0, offset=0.0))


@pytest.mark.parametrize("model, args", [
    (magnetics.sphere, (8.0, 3.0, 0.01)),
    (magnetics.prism, (-2.0, 2.0, -2.0, 2.0, 1.0, 6.0, 0.01)),        # square in plan: unchanged by a quarter turn
    (magnetics.vertical_cylinder, (1.0, 40.0, 0.5, 0.01)),
])
def test_an_offset_line_reaches_the_same_point_as_a_turned_profile(model, args):
    """The point x along a north profile and d to its right (east) is the point d along an east profile and x to its
    left (north). Two routes through the code to one place in space."""
    for x0, d in ((4.0, 7.0), (-11.0, 3.0), (6.0, -9.0)):
        north = model(np.array([x0]), *args, AMBIENT, azimuth=0.0, height=1.5, offset=d)
        east = model(np.array([d]), *args, AMBIENT, azimuth=90.0, height=1.5, offset=-x0)
        assert north[0] == pytest.approx(east[0], rel=1e-9, abs=1e-12)


def test_offset_cylinder_agrees_with_the_pole_pair():
    """A thin vertical cylinder magnetised along its axis, off the line, against two poles off the line: two codes
    that share nothing, and both have to mean the same side by ``offset``."""
    radius, top, length, m = 0.05, 2.0, 60.0, 1.0e4              # m, m, m, A/m straight down
    x = np.linspace(-80, 80, 161)
    for offset in (12.0, -12.0):
        cylinder = magnetics.vertical_cylinder(x, top, top + length, radius, 0.0, AMBIENT, azimuth=30.0, height=20.0,
                                               remanence=(m, 90.0, 0.0), offset=offset)
        poles = magnetics.pole_pair(x, m * np.pi * radius**2, top, length, AMBIENT, azimuth=30.0, height=20.0,
                                    offset=offset)
        np.testing.assert_allclose(cylinder, poles, rtol=2e-4, atol=1e-6 * np.abs(poles).max())
    left = magnetics.pole_pair(x, 1.0, top, length, AMBIENT, azimuth=30.0, height=20.0, offset=-12.0)
    right = magnetics.pole_pair(x, 1.0, top, length, AMBIENT, azimuth=30.0, height=20.0, offset=12.0)
    assert np.abs(left).max() != pytest.approx(np.abs(right).max(), rel=0.02)   # the two sides do differ


# --- the assessment between two lines ---------------------------------------------------------------------------------

def test_without_a_spacing_nothing_is_added():
    a = assessment(CAVITY, SURVEY, "gravity")
    assert a.line_spacing is None and a.line_offset is None and a.peak_over_target is None and a.peak_best_line is None
    assert not any("ine spacing" in line for line in a.assumptions)


def test_gravity_worst_case_is_midway_between_two_lines():
    spacing = 12.0
    a = assessment(CAVITY, {**SURVEY, "line_spacing": spacing}, "gravity")
    on_line = assessment(CAVITY, SURVEY, "gravity")
    assert abs(a.line_offset) == pytest.approx(spacing / 2)
    radius = CAVITY["size"] / 2
    drho = evaluate(CAVITY, SURVEY, methods=["gravity"]).result("gravity").contrast
    expected = abs(gravity.sphere(0.0, CAVITY["depth_top"] + radius, radius, drho, y=spacing / 2)) * 1e8   # µGal
    assert a.peak == pytest.approx(expected, rel=1e-9)
    assert a.peak_over_target == pytest.approx(on_line.peak, rel=1e-9) == pytest.approx(a.peak_best_line, rel=1e-9)
    assert a.snr == pytest.approx(a.peak / a.noise) and a.snr < on_line.snr
    assert a.width > on_line.width                    # further from the body, the anomaly is wider


@pytest.mark.parametrize("target, survey, method", [
    (CASING, DRONE, "magnetic"),
    (CAVITY, SURVEY, "gravity"),
    ({**CAVITY, "material": "basalt"}, SURVEY, "magnetic"),
    ({"shape": "prism", "depth_top": 1.0, "depth_bottom": 3.0, "size": 2.0, "length": 4.0, "material": "basalt",
      "host": "clay"}, SURVEY, "gradient"),
])
def test_wider_lines_never_read_more_and_close_lines_read_the_best_line(target, survey, method):
    peaks = [assessment(target, {**survey, "line_spacing": s}, method) for s in (0.05, 2.0, 5.0, 10.0, 20.0, 40.0)]
    for a in peaks:
        assert a.peak <= a.peak_best_line * (1 + 1e-9) and a.peak_over_target <= a.peak_best_line * (1 + 1e-9)
    assert peaks[0].peak == pytest.approx(peaks[0].peak_best_line, rel=2e-3)
    readings = [a.peak for a in peaks]
    assert all(later <= earlier * (1 + 1e-6) for earlier, later in zip(readings, readings[1:]))
    assert readings[-1] < 0.9 * readings[0]


def test_the_line_over_a_magnetic_target_is_not_the_best_line():
    """The anomaly's peak lies to one side of the body because the field is inclined: with east-west lines the line
    that passes over the casing reads clearly less than the best line."""
    a = assessment(CASING, {**DRONE, "profile_azimuth": 90.0, "line_spacing": 5.0}, "magnetic")
    assert a.peak_over_target < 0.9 * a.peak_best_line


def test_a_narrow_anomaly_is_sampled_where_it_peaks():
    """Sensor half a metre up, casing top a metre down: the anomaly is a few metres wide on a profile hundreds of
    metres long, and off the line its peak is not at x = 0."""
    ground = {**SURVEY, "profile_azimuth": 90.0}
    a = assessment(CASING, {**ground, "line_spacing": 0.02}, "magnetic")
    assert a.peak == pytest.approx(a.peak_best_line, rel=5e-3)
    x = np.array(evaluate(CASING, {**ground, "line_spacing": 2.0}, methods=["magnetic"]).result("magnetic").x)
    assert np.diff(x).min() < 0.1 and x[-1] >= 8 * 151.5


def test_the_result_says_what_was_assumed():
    a = assessment(CASING, {**DRONE, "line_spacing": 30.0}, "magnetic")
    text = " ".join(a.assumptions)
    assert "Line spacing 30 m" in text and "% of the best line a survey could fly" in text
    summary = evaluate(CASING, {**DRONE, "line_spacing": 30.0}, methods=["magnetic"]).summary()
    assert "worst case between lines 30 m apart" in summary and "see the assumptions" in summary
    assert "worst case" not in evaluate(CASING, DRONE, methods=["magnetic"]).summary()
    assert LINE_SPACING_CAVEAT in a.assumptions and "10.1021/acs.est.6c02007" in LINE_SPACING_CAVEAT
    assert a.verdict == ("clearly detectable" if a.snr >= detect.CLEAR else
                         "marginal" if a.snr >= detect.MARGINAL else "not detectable")


def test_a_2d_target_reads_the_same_on_every_line():
    tunnel = {"shape": "horizontal_cylinder", "depth_top": 4, "size": 3, "material": "air", "host": "limestone"}
    plain = assessment(tunnel, SURVEY, "gravity")
    spaced = assessment(tunnel, {**SURVEY, "line_spacing": 50.0}, "gravity")
    assert spaced.peak == plain.peak and spaced.line_spacing is None
    assert any("has no effect here" in line for line in spaced.assumptions)
    assert LINE_SPACING_CAVEAT not in spaced.assumptions


def test_methods_it_is_not_applied_to_say_so():
    a = assessment(CAVITY, {**SURVEY, "line_spacing": 10.0}, "gpr")
    assert a.line_spacing is None and any("is not applied to this method" in line for line in a.assumptions)


@pytest.mark.parametrize("value", [0, -5.0])
def test_a_spacing_must_be_positive(value):
    with pytest.raises(ValueError, match="line_spacing"):
        evaluate(CAVITY, {**SURVEY, "line_spacing": value})


# --- the widest spacing -----------------------------------------------------------------------------------------------

def test_spacing_limit_brackets_the_threshold():
    fast = dict(search=(20.0, 800.0), steps=10)
    limit = spacing_limit(CASING, DRONE, method="magnetic", **fast)
    assert isinstance(limit, SpacingLimit) and limit.available and limit.spacing is not None and not limit.beyond_search
    assert assessment(CASING, {**DRONE, "line_spacing": limit.spacing * 0.99}, "magnetic").snr >= detect.CLEAR
    assert assessment(CASING, {**DRONE, "line_spacing": limit.spacing * 1.01}, "magnetic").snr < detect.CLEAR
    assert spacing_limit(CASING, DRONE, method="magnetic", threshold=detect.MARGINAL, **fast).spacing > limit.spacing
    assert "line spacing" in str(limit)


def test_spacing_limit_when_no_spacing_is_enough_or_none_is_needed():
    weak = spacing_limit({**CASING, "susceptibility_contrast": 1.0}, DRONE, method="magnetic")
    assert weak.spacing is None and weak.snr_narrowest < detect.CLEAR and "never reaches" in str(weak)
    tunnel = {"shape": "horizontal_cylinder", "depth_top": 4, "size": 3, "material": "air", "host": "limestone"}
    assert not spacing_limit(tunnel, SURVEY, method="gravity").available
    assert not spacing_limit(CAVITY, SURVEY, method="gpr").available


def test_a_spacing_shortens_the_depth_limit():
    body = {"shape": "sphere", "size": 20, "material": "air", "host": "limestone"}
    assert depth_limit(body, {**SURVEY, "line_spacing": 40.0}, method="gravity").depth < \
        depth_limit(body, SURVEY, method="gravity").depth


# --- against the one published line-thinning measurement found --------------------------------------------------------

def test_the_calculation_is_optimistic_against_the_published_line_thinning():
    """Wang and others (2026, Environ. Sci. Technol. 60, 29355-29365, doi:10.1021/acs.est.6c02007), section 4.2.3 and
    Figure 5: the dense 10 m survey over the Wildcat Canyon well (casing 0.1143 m across, 1400 m long, top about 1.5 m
    down; sensor about 35 m above ground, 5 August 2024) was thinned, and with the well midway between the lines kept
    the recovered peak was about 87 % at 20 m, 78 % at 30 m and 57 % at 40 m of the 10 m survey's.

    The same definition here, for north-south lines and induced magnetisation, gives 95, 87 and 77 %. The site's
    coordinates (37.95 N, 122.30 W) are read from a map, the paper does not give the line direction, and its authors
    put the gap at 40 m and beyond down to height variation over steep terrain. One well: a consistency check that
    the calculation is on the optimistic side, which every result says; not a validation.
    """
    from detectability import igrf
    amb = magnetics.Ambient.from_igrf(igrf.field(37.95, -122.30, __import__("datetime").date(2024, 8, 5)))
    x = np.linspace(-400, 400, 3201)

    def midway(spacing):
        return max(np.abs(magnetics.vertical_cylinder(x, 1.5, 1401.5, 0.1143 / 2, 1.0, amb, height=35.0,
                                                      offset=side * spacing / 2)).max() for side in (1, -1))
    published = {20.0: 87.0, 30.0: 78.0, 40.0: 57.0}
    calculated = {s: 100 * midway(s) / midway(10.0) for s in published}
    assert calculated == pytest.approx({20.0: 95.4, 30.0: 87.2, 40.0: 77.1}, abs=0.2)
    assert all(calculated[s] > published[s] for s in published)
