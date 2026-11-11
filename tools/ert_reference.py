"""Compute (or check) the stored ERT validation values: pyGIMLi's forward model against analytic solutions.

    python tools/ert_reference.py           # write src/detectability/data/ert_reference.json
    python tools/ert_reference.py --check   # recompute and compare with the stored file

Needs pyGIMLi (pip install -e ".[ert]"). The library reads the stored values, so the website's Validation page does
not need pyGIMLi.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

OUT = Path(__file__).resolve().parents[1] / "src" / "detectability" / "data" / "ert_reference.json"


def _mesh(elecs, depth, layer=None):
    L = elecs[-1] - elecs[0]
    world = mt.createWorld(start=[elecs[0] - 2 * L, 0], end=[elecs[-1] + 2 * L, -depth], worldMarker=True)
    geom = world
    if layer is not None:
        geom = geom + mt.createLine(start=[elecs[0] - 2 * L, -layer], end=[elecs[-1] + 2 * L, -layer])
    for x in elecs:
        geom.createNode([x, 0.0])
        geom.createNode([x, -0.1])
    mesh = mt.createMesh(geom, quality=34, area=4.0)
    if layer is not None:  # regions: above the line marker 1, below marker 2
        for c in mesh.cells():
            c.setMarker(1 if c.center().y() > -layer else 2)
    return mesh


def halfspace():
    elecs = np.arange(48, dtype=float)
    scheme = ert.createData(elecs=elecs, schemeName="dd")
    data = ert.simulate(_mesh(elecs, 200.0), scheme=scheme, res=100.0, noiseLevel=0, verbose=False)
    return float(np.max(np.abs(np.array(data["rhoa"]) / 100.0 - 1.0)) * 100)


def wenner_two_layer(rho1=100.0, rho2=10.0, h=5.0):
    spacings = np.array([1, 2, 3, 5, 8, 12, 16, 20], float)
    k = (rho2 - rho1) / (rho2 + rho1)
    model, analytic = [], []
    for a in spacings:
        elecs = np.array([0, a, 2 * a, 3 * a]) - 1.5 * a
        scheme = pg.DataContainerERT()
        for x in elecs:
            scheme.createSensor([x, 0.0])
        scheme.createFourPointData(0, 0, 3, 1, 2)  # A, B, M, N: Wenner alpha
        scheme["k"] = ert.geometricFactors(scheme)
        mesh = _mesh(np.linspace(elecs[0], elecs[-1], 4), max(400.0, 30 * a), layer=h)
        d = ert.simulate(mesh, scheme=scheme, res=[[1, rho1], [2, rho2]], noiseLevel=0, verbose=False)
        model.append(float(d["rhoa"][0]))
        n = np.arange(1, 400)
        r = 2 * n * h / a
        analytic.append(float(rho1 * (1 + 4 * np.sum(k**n * (1 / np.sqrt(1 + r**2) - 1 / np.sqrt(4 + r**2))))))
    return spacings.tolist(), model, analytic


def compute():
    a, m, ref = wenner_two_layer()
    return {"pygimli": pg.__version__, "doi": "10.1016/j.cageo.2017.07.011",
            "halfspace_error_percent": halfspace(),
            "wenner_two_layer": {"rho1": 100.0, "rho2": 10.0, "h": 5.0, "spacing": a, "model": m, "analytic": ref}}


if __name__ == "__main__":
    new = compute()
    if "--check" in sys.argv:
        old = json.loads(OUT.read_text())
        ok = abs(old["halfspace_error_percent"] - new["halfspace_error_percent"]) < 0.05 and \
            np.allclose(old["wenner_two_layer"]["model"], new["wenner_two_layer"]["model"], rtol=1e-3)
        print("ERT reference matches" if ok else "ERT reference differs")
        sys.exit(0 if ok else 1)
    OUT.write_text(json.dumps(new, indent=1))
    w = new["wenner_two_layer"]
    print("half-space error %", round(new["halfspace_error_percent"], 3),
          "two-layer max error %", round(max(abs(x / y - 1) * 100 for x, y in zip(w["model"], w["analytic"])), 3))
