import json

import numpy as np
import pytest

from detectability import ert, evaluate, validation
from detectability.survey import Target

TUNNEL = {"shape": "horizontal_cylinder", "depth_top": 3, "size": 2, "material": "air", "host": "limestone"}


def test_typical_resistivity_is_inside_the_published_range():
    for name, (typ, lo, hi) in ert.RESISTIVITY.items():
        assert lo <= typ <= hi, name
    assert ert.resistivity("limestone") == pytest.approx(np.sqrt(50 * 400), rel=0.01)


def test_geometry_outlines():
    g = ert.geometry(Target(**TUNNEL))
    assert g.depth_needed == 5 and g.area == pytest.approx(np.pi)
    d = ert.geometry(Target(shape="dyke", depth_top=1, depth_bottom=11, size=2, dip=45))
    assert d.polygon[2][0] == pytest.approx(1 + 10)  # bottom shifted by the dip
    assert ert.default_spacing(5, 48) == 0.5


def test_unavailable_without_pygimli_or_service(monkeypatch):
    monkeypatch.setattr(ert, "available", lambda: False)
    monkeypatch.delenv("DETECTABILITY_ERT_URL", raising=False)
    r = evaluate(TUNNEL, {}, methods=["ert"]).result("ert")
    assert not r.available and "pygimli" in r.reason.lower()


def test_remote_service_result_and_failure(monkeypatch):
    monkeypatch.setattr(ert, "available", lambda: False)
    monkeypatch.setenv("DETECTABILITY_ERT_URL", "http://127.0.0.1:9/evaluate")
    fake = {"method": "ert", "available": True, "contrast": 7000.0, "contrast_units": "resistivity ratio",
            "instrument": "48 electrodes, dipole-dipole", "instrument_noise": 3.0, "field_noise": None,
            "x": [-1.0, 0.0, 1.0], "anomaly": [1.0, 10.0, 1.0], "details": {"recovered": True},
            "assessment": {"method": "ert", "peak": 10.0, "peak_at": 0.0, "width": 1.0, "shape": "bell", "noise": 3.0,
                           "snr": 3.3, "verdict": "marginal", "station_spacing": 0.5, "profile_length": 23.5,
                           "assumptions": ["x"], "units": "%"}}
    monkeypatch.setattr(ert, "remote", lambda t, s, url: json.loads(json.dumps(fake)))
    rep = evaluate(TUNNEL, {}, methods=["ert"])
    assert rep.result("ert").assessment.verdict == "marginal" and rep.recommended == "ert"

    def down(t, s, url):
        raise TimeoutError
    monkeypatch.setattr(ert, "remote", down)
    r = evaluate(TUNNEL, {}, methods=["ert"]).result("ert")
    assert not r.available and "did not answer" in r.reason


def test_script_has_python_booleans_and_only_asked_settings():
    rep = evaluate(TUNNEL, {"ert_invert": False}, methods=["gravity"])
    s = rep.script()
    assert "ert_" not in s and "gpr_" not in s and "true" not in s


def test_stored_validation_rows_pass():
    rows = [c for c in validation.cases() if c.key.startswith("ert-")]
    assert len(rows) == 2 and all(c.passed for c in rows)
