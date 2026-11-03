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
