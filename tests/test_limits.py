"""The depth limit must agree with the assessment it inverts: detectable just above it, not just below."""

import pytest

from detectability import Target, depth_limit, detect, evaluate

SURVEY = {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"}
CAVITY = {"shape": "sphere", "size": 20, "material": "air", "host": "limestone"}


def snr_at(target, depth, method):
    return evaluate({**target, "depth_top": depth}, SURVEY, methods=[method]).results[0].assessment.snr


@pytest.mark.parametrize("method", ["gravity", "gpr"])
def test_limit_brackets_the_threshold(method):
    limit = depth_limit(CAVITY, SURVEY, method=method)
    assert limit.available and limit.depth is not None and not limit.beyond_search
    assert snr_at(CAVITY, limit.depth * 0.99, method) >= detect.CLEAR
    assert snr_at(CAVITY, limit.depth * 1.01, method) < detect.CLEAR


def test_marginal_reaches_deeper_than_clear():
    clear = depth_limit(CAVITY, SURVEY, method="gravity")
    marginal = depth_limit(CAVITY, SURVEY, method="gravity", threshold=detect.MARGINAL)
    assert marginal.depth > clear.depth


def test_a_bigger_body_is_seen_deeper():
    small = depth_limit({**CAVITY, "size": 10}, SURVEY, method="gravity")
    large = depth_limit({**CAVITY, "size": 40}, SURVEY, method="gravity")
    assert large.depth > small.depth


def test_depth_top_is_ignored_and_optional():
    without = depth_limit(CAVITY, SURVEY, method="gravity")
    with_depth = depth_limit({**CAVITY, "depth_top": 999}, SURVEY, method="gravity")
    as_object = depth_limit(Target(shape="sphere", depth_top=1, size=20, material="air", host="limestone"),
                            SURVEY, method="gravity")
    assert without.depth == pytest.approx(with_depth.depth, rel=1e-9)
    assert without.depth == pytest.approx(as_object.depth, rel=1e-9)


def test_never_detected_reports_the_shallowest_signal():
    limit = depth_limit(CAVITY, SURVEY, method="magnetic")
    assert limit.depth is None and limit.available
    assert 0 < limit.snr_shallowest < detect.CLEAR
    assert "never reaches" in str(limit)


def test_still_detected_at_the_bottom_of_the_search():
    limit = depth_limit(CAVITY, SURVEY, method="gravity", search=(0.5, 5.0))
    assert limit.beyond_search and limit.depth == 5.0
    assert "at least" in str(limit)


def test_unassessable_method_says_why():
    limit = depth_limit({"shape": "sphere", "size": 20, "material": "magnetite", "host": "granite"},
                        SURVEY, method="gravity")
    assert not limit.available and limit.depth is None
    assert "no density for 'magnetite'" in limit.reason and "not assessed" in str(limit)


def test_body_keeps_its_thickness_while_it_moves_down():
    dyke = {"shape": "dyke", "size": 10, "depth_top": 5, "depth_bottom": 45,
            "material": "air", "host": "limestone"}
    limit = depth_limit(dyke, SURVEY, method="gravity")
    assert limit.depth is not None
    # The 40 m thickness must be kept at the limit, or the answer would be about a different body.
    assert snr_at({**dyke, "depth_bottom": limit.depth * 0.99 + 40}, limit.depth * 0.99, "gravity") >= detect.CLEAR


def test_a_narrow_search_still_lands_on_the_same_answer():
    wide = depth_limit(CAVITY, SURVEY, method="gravity")
    narrow = depth_limit(CAVITY, SURVEY, method="gravity", search=(20.0, 30.0), steps=6)
    assert narrow.depth == pytest.approx(wide.depth, rel=1e-3)


@pytest.mark.parametrize("search", [(0, 100), (-1, 10), (100, 10), (10, 10)])
def test_a_bad_search_range_is_refused(search):
    with pytest.raises(ValueError, match="search must be"):
        depth_limit(CAVITY, SURVEY, method="gravity", search=search)


def test_too_few_steps_is_refused():
    with pytest.raises(ValueError, match="steps must be"):
        depth_limit(CAVITY, SURVEY, method="gravity", steps=1)


def test_an_invalid_target_is_refused():
    with pytest.raises(ValueError):
        depth_limit({"shape": "sphere", "size": -5, "material": "air", "host": "limestone"},
                    SURVEY, method="gravity")
