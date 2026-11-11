import pytest

from detectability import evaluate

TUNNEL = {"shape": "horizontal_cylinder", "depth_top": 3, "size": 2, "material": "air", "host": "limestone"}

pg = pytest.importorskip("pygimli")


def test_ert_with_pygimli_tunnel_in_limestone():
    r = evaluate(TUNNEL, {"ert_invert": False}, methods=["ert"]).result("ert")
    assert r.available and 5 < r.details["max_anomaly_percent"] < 20
    assert r.assessment.station_spacing == 0.5


def test_conductive_dyke_is_clear():
    r = evaluate({"shape": "dyke", "depth_top": 1, "depth_bottom": 10, "size": 2, "material": "clay",
                  "host": "granite"}, {"ert_invert": False}, methods=["ert"]).result("ert")
    assert r.assessment.verdict == "clearly detectable"
