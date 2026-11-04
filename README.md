# Detectability

[![tests](https://github.com/mukremin-nergiz/detectability/actions/workflows/tests.yml/badge.svg)](https://github.com/mukremin-nergiz/detectability/actions/workflows/tests.yml)
[![licence: BSD-3-Clause](https://img.shields.io/badge/licence-BSD--3--Clause-blue.svg)](https://github.com/mukremin-nergiz/detectability/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23931815.svg)](https://doi.org/10.5281/zenodo.23931815)

**Pre-survey detectability, modelled and reproducible.** Describe a buried target and a survey. Detectability models the gravity and
magnetic response of the target, compares it with instrument and field noise, and gives a verdict for each method
(clearly detectable, marginal, not detectable), with the recommended station spacing and profile length, every
assumption, and a Python script that reproduces the result.

- Web tool: <https://detectability.mukreminnergiz.com>
- Method and validation: <https://detectability.mukreminnergiz.com/method>

## Install

```bash
pip install detectability
```

Python 3.10 or newer; the only dependency is NumPy. The development version: `pip install "git+https://github.com/mukremin-nergiz/detectability"`.

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
Detectability 0.2.1: sphere in limestone
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
| `gravity` | sphere, horizontal cylinder, right rectangular prism (Nagy; Plouff) and its gradient tensor, vertical cylinder (exact, by a rim integral) and its gradient tensor, vertical step, dyke, 2D polygon (Talwani; Won & Bevis) |
| `magnetics` | total-field anomaly: sphere, horizontal cylinder, prism, vertical cylinder, 2D polygon, dyke (any dip), vertical contact; induced and remanent magnetisation |
| `igrf` | IGRF-14 main field from location and date, from the official IAGA coefficients |
| `rocks` | typical densities and magnetic susceptibilities with their ranges (Telford et al. 1990) |
| `detect` | instrument and field noise, signal-to-noise ratio, verdict, station spacing, profile length |

## Validation

Every model is checked against a result computed in a different way. Each row below is also a test: it fails if the
error exceeds the tolerance. The error is the largest difference along the profile as a percentage of the largest
reference value. Figures for each row: <https://detectability.mukreminnergiz.com/validation>.

The "Independent code" rows compare with [Harmonica](https://www.fatiando.org/harmonica) 0.7.0 (Fatiando a Terra
Project, [doi:10.5281/zenodo.13308312](https://doi.org/10.5281/zenodo.13308312)), a separate open-source package. Its
values are stored in `src/detectability/data/harmonica_reference.json`, so the library does not depend on it; to
recompute and check them:

```bash
pip install -e ".[validation]"
python tools/harmonica_reference.py --check
```

| Check | Geometry and method | Reference | Max. relative error | Tolerance |
|---|---|---|---:|---:|
| Analytical limit | Prism, gravity, far from the sensor | Point mass of the same mass | 1.9e-05 % | 0.01 % |
| Analytical limit | Prism, gravity, wide and thin | Infinite Bouguer slab, 2πGΔρt | 0.0099 % | 0.1 % |
| Analytical limit | Vertical step, gravity, far over the slab | Infinite Bouguer slab, 2πGΔρt | 4.8e-04 % | 0.01 % |
| Analytical limit | Vertical cylinder, gravity, on its axis | Exact on-axis cylinder formula | 1.7e-13 % | 1.0e-10 % |
| Independent formula | Horizontal cylinder, gravity | 2D polygon, 720 sides (Talwani; Won & Bevis) | 6.3e-12 % | 0.01 % |
| Independent formula | Vertical dyke, gravity | 2D polygon (Won & Bevis) | 1.9e-13 % | 1.0e-06 % |
| Numerical integration | Prism, gravity | 16³ Gauss–Legendre point masses | 2.9e-06 % | 0.001 % |
| Numerical integration | Horizontal cylinder, magnetic | 3D dipoles integrated along strike | 2.2e-07 % | 0.01 % |
| Numerical integration | Prism, magnetic | 16³ Gauss–Legendre point dipoles | 1.3e-12 % | 0.001 % |
| Numerical integration | Vertical cylinder, gravity | Gauss–Legendre over the cross-section, line masses in depth | 5.5e-13 % | 1.0e-08 % |
| Numerical integration | Vertical cylinder, magnetic | Gauss–Legendre point dipoles over the volume | 5.8e-13 % | 1.0e-06 % |
| Poisson's relation | Sphere, gravity ↔ magnetic | Vertical derivative of the gravity sphere | 1.4e-08 % | 0.001 % |
| Poisson's relation | Vertical dyke, gravity ↔ magnetic | Vertical derivative of the gravity dyke | 2.2e-08 % | 1.0e-04 % |
| Poisson's relation | Dipping dyke (60°), gravity ↔ magnetic | Vertical derivative of the gravity dyke | 2.6e-08 % | 1.0e-04 % |
| Poisson's relation | Vertical cylinder, gravity ↔ magnetic | Vertical derivative of the gravity cylinder | 3.6e-08 % | 1.0e-04 % |
| Poisson's relation | Vertical contact, gravity ↔ magnetic | Vertical derivative of the gravity step | 3.9e-06 % | 1.0e-04 % |
| Independent code | Prism, gravity | Harmonica 0.7.0 prism_gravity | 3.0e-12 % | 1.0e-08 % |
| Independent code | Prism, magnetic | Harmonica 0.7.0 prism_magnetic | 4.8e-13 % | 1.0e-08 % |
| Independent code | Sphere, gravity | Harmonica 0.7.0 point_gravity | 3.9e-14 % | 1.0e-10 % |
| Independent code | Sphere, magnetic | Harmonica 0.7.0 dipole_magnetic | 8.8e-14 % | 1.0e-10 % |
| Independent code | Vertical dyke, gravity | Harmonica 0.7.0 prism_gravity, 200 km along strike | 3.6e-07 % | 1.0e-05 % |
| Independent code | Vertical dyke, magnetic | Harmonica 0.7.0 prism_magnetic, 200 km along strike | 5.1e-07 % | 1.0e-05 % |
| Reference service | IGRF-14 main field, 5 sites | British Geological Survey IGRF-14 | 0.38 nT, 0.0005° | 1 nT, 0.002° |

Regenerate this table with `python -m detectability.validation`. Run all tests (including these) with:

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

See [CITATION.cff](https://github.com/mukremin-nergiz/detectability/blob/main/CITATION.cff), or:

> Nergiz, M. (2026). *Detectability: Pre-survey detectability, modelled and reproducible* (Version 0.2.1) [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.23931815

The DOI above always resolves to the latest version; each release also has its own DOI on Zenodo. Changes between
releases: [CHANGELOG.md](https://github.com/mukremin-nergiz/detectability/blob/main/CHANGELOG.md).

## Author

Mükremin Nergiz, Geophysical Engineer & Software Developer (Independent Researcher, Sivas, Türkiye).
ORCID [0009-0001-6789-2774](https://orcid.org/0009-0001-6789-2774) · <https://mukreminnergiz.com>

Licence: BSD-3-Clause.
