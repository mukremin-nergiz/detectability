import json

import pytest

from detectability import Survey, Target, evaluate

SIVAS = {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"}
VOID = {"shape": "sphere", "depth_top": 15, "size": 20, "material": "air", "host": "limestone"}


def test_void_in_limestone():
    r = evaluate(VOID, SIVAS)
    g, m = r.result("gravity"), r.result("magnetic")
    assert g.assessment.verdict == "clearly detectable"
    assert g.assessment.peak == pytest.approx(114.07, rel=2e-3)
    assert m.assessment.verdict == "not detectable"
    assert m.main_field["source"] == "IGRF-14"
    assert r.recommended == "gravity"
    assert "Recommended method: gravity" in r.summary()


def test_script_reproduces_the_report():
    r = evaluate(VOID, SIVAS)
    namespace = {}
    exec(compile(r.script().replace("import matplotlib.pyplot as plt", "raise ImportError"), "script", "exec"),
         namespace)
    again = namespace["report"]
    assert again.to_dict() == r.to_dict()


def test_every_shape_runs_and_serialises():
    shapes = [
        {"shape": "sphere", "depth_top": 2, "size": 1},
        {"shape": "horizontal_cylinder", "depth_top": 2, "size": 1},
        {"shape": "vertical_cylinder", "depth_top": 0.5, "depth_bottom": 10, "size": 1},
        {"shape": "prism", "depth_top": 1, "depth_bottom": 3, "size": 4, "length": 6},
        {"shape": "dyke", "depth_top": 0.5, "depth_bottom": 3, "size": 1, "dip": 70},
        {"shape": "contact", "depth_top": 2, "depth_bottom": 30, "material": "basalt", "host": "sandstone"},
    ]
    for t in shapes:
        r = evaluate(t, SIVAS)
        assert all(res.available for res in r.results), t
        json.dumps(r.to_dict())


def test_magnetics_needs_a_location_or_a_given_field():
    r = evaluate(VOID, {})
    assert not r.result("magnetic").available
    assert r.result("gravity").available
    r = evaluate(VOID, {"field_total": 50000, "field_inclination": 60, "field_declination": 0})
    assert r.result("magnetic").main_field["source"] == "given"


def test_missing_property_and_override():
    t = {"shape": "sphere", "depth_top": 2, "size": 1, "material": "air", "host": "soil"}
    r = evaluate(t, SIVAS)
    assert r.result("gravity").available
    assert not r.result("magnetic").available and "susceptibility" in r.result("magnetic").reason
    r = evaluate({**t, "susceptibility_contrast": -1e-3}, SIVAS)
    assert r.result("magnetic").available


def test_input_checks():
    with pytest.raises(ValueError):
        evaluate({"shape": "cube", "depth_top": 1, "size": 1})
    with pytest.raises(ValueError):
        evaluate({"shape": "dyke", "depth_top": 1, "size": 1})          # no depth_bottom
    with pytest.raises(ValueError):
        evaluate(VOID, {"colour": "red"})
    with pytest.raises(ValueError):
        evaluate(VOID, methods=["seismic"])


def test_dataclass_inputs_are_not_modified():
    s = Survey(latitude=39.75, longitude=37.02)
    evaluate(Target(**VOID), s)
    assert s.date is None


SITE = {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"}
SHALLOW_DYKE = {"shape": "dyke", "depth_top": 0.4, "depth_bottom": 1.2, "size": 0.8, "material": "basalt",
                "host": "sandstone"}


def test_default_methods_unchanged_and_gradient_on_request():
    from detectability import DEFAULT_METHODS, METHODS
    assert DEFAULT_METHODS == ("gravity", "magnetic") and "gradient" in METHODS
    assert [r.method for r in evaluate(SHALLOW_DYKE, SITE).results] == ["gravity", "magnetic"]
    r = evaluate(SHALLOW_DYKE, SITE, ["gradient"]).results[0]
    assert r.available and r.instrument == "Bartington Grad601" and r.assessment.units == "nT"


def test_gradient_is_the_difference_of_two_sensor_heights():
    import numpy as np
    low = evaluate(SHALLOW_DYKE, {**SITE, "sensor_height": 0.3}, ["magnetic"]).results[0]
    high = evaluate(SHALLOW_DYKE, {**SITE, "sensor_height": 0.95}, ["magnetic"]).results[0]
    g = evaluate(SHALLOW_DYKE, {**SITE, "gradiometer": "Sensys FGM650", "gradiometer_height": 0.3}, ["gradient"])
    assert np.allclose(g.results[0].anomaly, np.subtract(low.anomaly, high.anomaly), rtol=0, atol=1e-12)


def test_gradient_favours_shallow_targets():
    deep = {"shape": "sphere", "depth_top": 15, "size": 20, "material": "air", "host": "limestone"}
    shallow = evaluate(SHALLOW_DYKE, SITE, ["magnetic", "gradient"])
    assert shallow.recommended == "gradient"
    far = evaluate(deep, SITE, ["magnetic", "gradient"]).results
    assert far[1].assessment.snr < far[0].assessment.snr


def test_custom_gradiometer_and_errors():
    import pytest
    r = evaluate(SHALLOW_DYKE, {**SITE, "gradiometer": "home-made", "gradient_instrument_noise": 0.05,
                                "gradiometer_baseline": 0.5}, ["gradient"]).results[0]
    assert r.available and r.instrument_noise == 0.05
    assert not evaluate(SHALLOW_DYKE, {**SITE, "gradiometer": "home-made"}, ["gradient"]).results[0].available
    assert not evaluate(SHALLOW_DYKE, {}, ["gradient"]).results[0].available          # needs the location
    with pytest.raises(ValueError):
        evaluate(SHALLOW_DYKE, {**SITE, "gradiometer_baseline": 0}, ["gradient"])


# --- inputs that cannot be a survey are refused by name, not passed on to the models ---------------------------------

CAVE = {"shape": "sphere", "depth_top": 15, "size": 20, "material": "air", "host": "limestone"}
SIVAS = {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"}


@pytest.mark.parametrize("target, message", [
    ({**CAVE, "depth_top": "5"}, "depth_top must be a number"),
    ({**CAVE, "size": True}, "size must be a number"),
    ({**CAVE, "size": float("nan")}, "size must be a finite number"),
    ({**CAVE, "material": 5}, "material must be text"),
    ({**CAVE, "host_permittivity": -4}, "host_permittivity is relative to a vacuum"),
    ({**CAVE, "permittivity": 0}, "permittivity is relative to a vacuum"),
    ({**CAVE, "conductivity": -1}, "conductivity must be zero or positive"),
    ({**CAVE, "host_resistivity": 0}, "host_resistivity must be positive"),
    ({"shape": "sphere"}, "target needs depth_top"),
    ("a cave", "target must be a dictionary"),
    ({"shape": "dyke", "depth_top": 1, "depth_bottom": 5, "size": 2, "dip": 200, "material": "basalt",
      "host": "sandstone"}, "dip must be within"),
])
def test_a_target_that_makes_no_sense_is_refused_by_field(target, message):
    with pytest.raises(ValueError, match=message):
        evaluate(target, SIVAS)


@pytest.mark.parametrize("survey, message", [
    ({**SIVAS, "gravity_field_noise": -10}, "gravity_field_noise must be zero or positive"),
    ({**SIVAS, "magnetic_instrument_noise": -1}, "magnetic_instrument_noise must be zero or positive"),
    ({**SIVAS, "sensor_height": -30}, "sensor_height is the height above ground"),
    ({**SIVAS, "gradiometer_height": -1}, "gradiometer_height is the height above ground"),
    ({**SIVAS, "latitude": "north"}, "latitude must be a number"),
    ({**SIVAS, "longitude": 500}, "longitude must be within"),
    ({**SIVAS, "date": "next week"}, "date must be an ISO date"),
    ({**SIVAS, "date": 2026}, "date must be text"),
    ({"field_total": 50000, "field_inclination": 500, "field_declination": 0}, "field_inclination must be within"),
    ({"field_total": -1, "field_inclination": 60, "field_declination": 0}, "field_total must be positive"),
    ("here", "survey must be a dictionary"),
])
def test_a_survey_that_makes_no_sense_is_refused_by_field(survey, message):
    with pytest.raises(ValueError, match=message):
        evaluate(CAVE, survey)


def test_a_negative_noise_no_longer_hides_in_the_quadrature_sum():
    """hypot(5, -10) equals hypot(5, 10): before the check, a negative field noise gave the same answer as a positive
    one and nobody was told."""
    with pytest.raises(ValueError):
        evaluate(CAVE, {**SIVAS, "gravity_field_noise": -10}, methods=["gravity"])


def test_zero_noise_is_refused_instead_of_giving_an_infinite_ratio():
    with pytest.raises(ValueError, match="effective gravity noise is zero"):
        evaluate(CAVE, {**SIVAS, "gravity_field_noise": 0, "gravity_instrument_noise": 0}, methods=["gravity"])


@pytest.mark.parametrize("methods", ["gravity", [["gravity"]], [1]])
def test_methods_must_be_a_list_of_names(methods):
    with pytest.raises(ValueError, match="methods must be a list of method names"):
        evaluate(CAVE, SIVAS, methods=methods)


def test_water_is_the_same_water_under_each_table_s_name():
    """The density table says "water", the radar table "fresh water": either name is assessed by every method that has
    the property, instead of one of them coming back as not assessed."""
    flooded = {"shape": "sphere", "depth_top": 3, "size": 4, "host": "limestone"}
    by_name = {name: evaluate({**flooded, "material": name}, SIVAS, methods=["gravity", "magnetic", "gpr"])
               for name in ("water", "fresh water")}
    for method in ("gravity", "magnetic", "gpr"):
        a, b = (by_name[n].result(method) for n in ("water", "fresh water"))
        assert a.available and b.available
        assert a.assessment.snr == pytest.approx(b.assessment.snr)
