"""Reference values from Harmonica for the "Independent code" rows of the validation.

Harmonica (Fatiando a Terra Project) is a separate open-source package for gravity and magnetic modelling. This script
computes the targets defined in ``detectability.validation.HARMONICA_CASES`` with it and writes
``src/detectability/data/harmonica_reference.json``; Detectability's own results are compared with these values by
``detectability.validation``. Only the target geometry is taken from Detectability: the forward models, the
magnetisation and the unit conversions here are Harmonica's or written out below.

    pip install -e ".[validation]"
    python tools/harmonica_reference.py           # write the file
    python tools/harmonica_reference.py --check   # recompute and compare with the stored file (used in CI)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import choclo
import harmonica as hm
import numpy as np
from choclo.constants import VACUUM_MAGNETIC_PERMEABILITY

from detectability.validation import HARMONICA_CASES, HARMONICA_FILE

OUT = Path(__file__).resolve().parents[1] / "src" / "detectability" / "data" / HARMONICA_FILE
DOI = "10.5281/zenodo.13308312"            # Harmonica v0.7.0 on Zenodo (concept DOI 10.5281/zenodo.3628741)
# Harmonica's kernels use the CODATA 2018 value of mu0; the induced magnetisation chi F / mu0 uses the same value, so
# the induced anomaly does not depend on it (as in Detectability, which uses 4 pi 1e-7).
MU0 = VACUUM_MAGNETIC_PERMEABILITY
MGAL_TO_UGAL = 1000.0


def unit(inclination: float, declination: float) -> np.ndarray:
    """(north, east, down) unit vector."""
    i, d = np.radians(inclination), np.radians(declination)
    return np.array([np.cos(i) * np.cos(d), np.cos(i) * np.sin(d), np.sin(i)])


def enu(vec: np.ndarray) -> tuple[float, float, float]:
    """(north, east, down) -> Harmonica's (east, north, up)."""
    return vec[1], vec[0], -vec[2]


def compute(c: dict) -> np.ndarray:
    x = np.linspace(c["x"][0], c["x"][1], int(c["x"][2]))
    coords = (np.zeros_like(x), x, np.full_like(x, c.get("height", 0.0)))       # profile along north
    magnetic = "chi" in c
    if magnetic:
        total, inc, dec = c["field"]
        f = unit(inc, dec)
        m = c["chi"] * total * 1e-9 / MU0 * f                                     # induced magnetisation, A/m
    if c["source"] == "prism":
        (s, n), (w, e), (top, bottom) = c["along"], c["across"], c["depth"]
        prisms = np.array([[w, e, s, n, -bottom, -top]])
        if not magnetic:
            return hm.prism_gravity(coords, prisms, np.array([c["drho"]]), field="g_z") * MGAL_TO_UGAL
        me, mn, mu = enu(m)
        be, bn, bu = hm.prism_magnetic(coords, prisms, (np.array([me]), np.array([mn]), np.array([mu])), field="b")
    else:
        volume = 4.0 / 3.0 * np.pi * c["radius"] ** 3
        point = (np.array([0.0]), np.array([0.0]), np.array([-c["depth"]]))
        if not magnetic:
            return hm.point_gravity(coords, point, np.array([c["drho"] * volume]), field="g_z") * MGAL_TO_UGAL
        me, mn, mu = enu(m * volume)
        be, bn, bu = hm.dipole_magnetic(coords, point, (np.array([me]), np.array([mn]), np.array([mu])), field="b")
    return bn * f[0] + be * f[1] - bu * f[2]                                     # total-field anomaly, nT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the stored file instead of writing it")
    args = parser.parse_args()
    values = {key: compute(c) for key, c in HARMONICA_CASES.items()}
    if args.check:
        stored = json.loads(OUT.read_text("utf-8"))
        worst = 0.0
        for key, v in values.items():
            ref = np.asarray(stored["cases"][key], float)
            err = float(np.max(np.abs(v - ref)) / np.max(np.abs(ref)))
            worst = max(worst, err)
            print(f"{key:28s} {err:.2e}")
        ok = worst <= 1e-12 and stored["harmonica"] == hm.__version__
        print("stored file matches Harmonica", hm.__version__, "->", "ok" if ok else "MISMATCH")
        return 0 if ok else 1
    doc = {
        "about": "Reference values computed with Harmonica by tools/harmonica_reference.py; gravity in µGal, "
                 "magnetic total-field anomaly in nT. Geometry: detectability.validation.HARMONICA_CASES.",
        "harmonica": hm.__version__, "choclo": choclo.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "doi": DOI, "computed": dt.date.today().isoformat(),
        "cases": {key: [float(f"{t:.17g}") for t in v] for key, v in values.items()},
    }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
