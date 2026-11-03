# Detectability

[![tests](https://github.com/mukremin-nergiz/detectability/actions/workflows/tests.yml/badge.svg)](https://github.com/mukremin-nergiz/detectability/actions/workflows/tests.yml)
[![licence: BSD-3-Clause](https://img.shields.io/badge/licence-BSD--3--Clause-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23931815.svg)](https://doi.org/10.5281/zenodo.23931815)

**Will your geophysical survey see it?** Describe a buried target and a survey. Detectability models the gravity and
magnetic response of the target, compares it with instrument and field noise, and gives a verdict for each method
(clearly detectable, marginal, not detectable), with the recommended station spacing and profile length, every
assumption, and a Python script that reproduces the result.

- Web tool: <https://detectability.mukreminnergiz.com>
- Method and validation: <https://detectability.mukreminnergiz.com/method>

## Install

```bash
pip install "git+https://github.com/mukremin-nergiz/detectability"
```

Python 3.10 or newer; the only dependency is NumPy.

## Example

A 20 m air-filled cave whose top is 15 m deep in limestone, surveyed near Sivas, Türkiye:

```python
from detectability import evaluate

report = evaluate(
    {"shape": "sphere", "depth_top": 15, "size": 20, "material": "air", "host": "limestone"},
    {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"},
)
print(report.summary())
```

```text
Detectability 0.1.1: sphere in limestone
- gravity: peak 114 µGal, noise 11.2 µGal, S/N 10.2: clearly detectable
  stations every 5 m or less, profile at least 153 m long
- magnetic: peak 0.462 nT, noise 2 nT, S/N 0.231: not detectable
Recommended method: gravity
```

`report.script()` returns a standalone script that recomputes the report; `report.to_dict()` gives the full result,
including the modelled profiles and the assumptions.

Target shapes: `sphere`, `horizontal_cylinder`, `vertical_cylinder`, `prism`, `dyke` (any dip) and `contact`. The main
geomagnetic field comes from IGRF-14 for the survey location and date. Typical densities and susceptibilities are built
in, and measured contrasts can be given instead.

## Modules

| Module | Contents |
|---|---|
| `survey` | `evaluate()`: target and survey in, report and reproduction script out |
| `gravity` | sphere, horizontal cylinder, right rectangular prism (Nagy; Plouff) and its gradient tensor, vertical cylinder, vertical step, dyke, 2D polygon (Talwani; Won & Bevis) |
| `magnetics` | total-field anomaly: sphere, horizontal cylinder, prism, vertical cylinder, 2D polygon, dyke (any dip), vertical contact; induced and remanent magnetisation |
| `igrf` | IGRF-14 main field from location and date, from the official IAGA coefficients |
| `rocks` | typical densities and magnetic susceptibilities with their ranges (Telford et al. 1990) |
| `detect` | instrument and field noise, signal-to-noise ratio, verdict, station spacing, profile length |

## Validation

Every model is checked against a result computed in a different way: analytical limits, numerical integration of
point dipoles, Poisson's relation between the independent gravity and magnetic codes (including dipping dykes), and
the British Geological Survey's IGRF-14 service at five sites. Run the tests with:

```bash
pip install -e ".[test]"
pytest
```

## Limitations

Flat topography, a homogeneous host and no regional trend; 2D targets are infinitely long along strike; the profile
crosses the centre of the target; self-demagnetisation is neglected. Results are expectations under these assumptions,
not guarantees. Instrument presets come from manufacturers' data sheets (`detect.INSTRUMENTS` lists each figure and its
source); field noise, which usually dominates, is a stated assumption that you can change.

## Citation

See [CITATION.cff](CITATION.cff), or:

> Nergiz, M. (2026). *Detectability: will your geophysical survey see it?* (Version 0.1.0) [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.23931815

The DOI above always resolves to the latest version; each release also has its own DOI on Zenodo.

## Author

Mükremin Nergiz, Geophysical Engineer & Software Developer (Independent Researcher, Sivas, Türkiye).
ORCID [0009-0001-6789-2774](https://orcid.org/0009-0001-6789-2774) · <https://mukreminnergiz.com>

Licence: BSD-3-Clause.
