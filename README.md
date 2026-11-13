# Detectability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23931815.svg)](https://doi.org/10.5281/zenodo.23931815)
[![tests](https://github.com/mukremin-nergiz/detectability/actions/workflows/tests.yml/badge.svg)](https://github.com/mukremin-nergiz/detectability/actions/workflows/tests.yml)
[![licence: BSD-3-Clause](https://img.shields.io/badge/licence-BSD--3--Clause-blue.svg)](https://github.com/mukremin-nergiz/detectability/blob/main/LICENSE)

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
Detectability 0.7.0: sphere in limestone
- gravity: peak 114 µGal, noise 11.2 µGal, S/N 10.2: clearly detectable
  stations every 5 m or less, profile at least 153 m long
- magnetic: peak 0.462 nT, noise 2 nT, S/N 0.231: not detectable
Recommended method: gravity
```

`report.script()` returns a standalone script that recomputes the report; `report.to_dict()` gives the full result,
including the modelled profiles and the assumptions.

### How deep before a method loses it?

`depth_limit()` asks the inverse question — the one a survey plan turns on. Leave `depth_top` out: it is what the
search solves for.

```python
from detectability import depth_limit

for method in ("gravity", "magnetic", "gpr"):
    print(depth_limit({"shape": "sphere", "size": 20, "material": "air", "host": "limestone"},
                      {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"}, method=method))
```

```text
gravity: detectable (S/N >= 5) down to 25.7 m
magnetic: never reaches S/N 5; S/N 2.88 even at 0.5 m
gpr: detectable (S/N >= 5) down to 25.2 m
```

Pass `threshold=detect.MARGINAL` for the lower threshold, and `search=(shallow, deep)` to bound the sweep.
Sweep the target size and you have a detection-limit chart; the website draws one at
[/limits](https://detectability.mukreminnergiz.com/limits).

Target shapes: `sphere`, `horizontal_cylinder`, `vertical_cylinder`, `prism`, `dyke` (any dip) and `contact`. The main
geomagnetic field comes from IGRF-14 for the survey location and date. Typical densities and susceptibilities are built
in, and measured contrasts can be given instead.

Methods: `gravity`, `magnetic` (total-field anomaly) and, on request, `gradient`: the reading of a vertical gradiometer
(lower minus upper sensor, in nT over the baseline), with presets for the Bartington Grad601, Sensys FGM650, Foerster
FEREX 4.034 and GEM GSM-19G; and, on request, `gpr`: ground-penetrating radar, the echo of the target's top against
the system's detection limit (radar range equation), with velocity, attenuation, antenna frequency, trace spacing and
recording window; and, on request, `ert`: electrical resistivity tomography, a 2D finite-element model of the target
under a line of electrodes (dipole-dipole, Wenner, Schlumberger or gradient array), the change it makes to the
apparent resistivities, and whether an inversion of noisy synthetic data recovers it. ERT needs
[pyGIMLi](https://www.pygimli.org): `pip install "detectability[ert]"`. The defaults are `gravity` and `magnetic`:

```python
evaluate(target, survey, methods=["gravity", "magnetic", "gradient", "gpr", "ert"])
```

## Modules

| Module | Contents |
|---|---|
| `survey` | `evaluate()`: target and survey in, report and reproduction script out |
| `gravity` | sphere, horizontal cylinder, right rectangular prism (Nagy; Plouff) and its gradient tensor, vertical cylinder (exact, by a rim integral) and its gradient tensor, vertical step, dyke, 2D polygon (Talwani; Won & Bevis) |
| `magnetics` | total-field anomaly: sphere, horizontal cylinder, prism, vertical cylinder, 2D polygon, dyke (any dip), vertical contact; induced and remanent magnetisation (a gradiometer reading is the difference at two sensor heights) |
| `gpr` | ground-penetrating radar: exact velocity and attenuation of a lossy medium, normal-incidence reflection coefficient, target-size factor (geometric optics, Rayleigh regime, Fresnel zone), echo against system performance; typical permittivity and conductivity (Davis & Annan 1989) |
| `ert` | electrical resistivity tomography with pyGIMLi (optional): electrode layout, 2D finite-element forward model, inversion and recovery check; typical resistivities (Loke 1999, after Keller & Frischknecht 1966) |
| `igrf` | IGRF-14 main field from location and date, from the official IAGA coefficients |
| `rocks` | typical densities and magnetic susceptibilities with their ranges (Telford et al. 1990) |
| `detect` | instrument presets with their sources (gravimeters, magnetometers, gradiometers), field noise, signal-to-noise ratio, verdict, station spacing, profile length |

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
| Independent formula | Sphere, magnetic gradient (gradiometer reading) | Closed-form vertical-dipole field at both sensors | 7.8e-14 % | 1.0e-10 % |
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
| Analytical limit | GPR, attenuation in a low-loss host | Low-loss limit 1636 σ/√εr dB/m (σ in S/m) | 1.1e-04 % | 0.01 % |
| Analytical limit | GPR, velocity in a low-loss host | c/√εr | 2.5e-06 % | 1.0e-04 % |
| Analytical limit | GPR, reflection coefficient | (√ε1 − √ε2)/(√ε1 + √ε2) for lossless media | 0.0e+00 % | 1.0e-10 % |
| Analytical limit | GPR, very large sphere | Flat reflector at the same range | 5.0e-06 % | 1.0e-04 % |
| Published values | GPR, velocity of common materials | Davis & Annan (1989) table velocities (rounded to 0.01 m/ns) | 4.92 % | 6 % |
| Analytical limit | ERT, homogeneous ground | Apparent resistivity equals the true resistivity | 0.297 % | 1 % |
| Analytical limit | ERT, two layers, Wenner sounding | Image-series solution for two layers | 0.144 % | 2 % |
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
crosses the centre of the target; self-demagnetisation is neglected. Magnetisation is induced only unless a remanent
magnetisation is given, as a vector (`remanence` in A/m, `remanence_inclination`, `remanence_declination`) or as a
Koenigsberger ratio (`koenigsberger_ratio`, taken along the present field: the optimistic case); no material has a
value of its own, and remanence of the host is not modelled. Results are expectations under these assumptions,
not guarantees. Instrument presets come from manufacturers' data sheets (`detect.INSTRUMENTS` lists each figure and its
source); field noise, which usually dominates, is a stated assumption that you can change. For gradiometers the default
field noise (0.5 nT) is an assumption, not a published figure: soil, heading and walking noise depend on the site.

## Citation

See [CITATION.cff](https://github.com/mukremin-nergiz/detectability/blob/main/CITATION.cff), or:

> Nergiz, M. (2026). *Detectability: Pre-survey detectability, modelled and reproducible* (Version 0.7.0) [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.23931815

The DOI above always resolves to the latest version; each release also has its own DOI on Zenodo. Changes between
releases: [CHANGELOG.md](https://github.com/mukremin-nergiz/detectability/blob/main/CHANGELOG.md).

## Author

Mükremin Nergiz, Geophysical Engineer & Software Developer (Independent Researcher, Sivas, Türkiye).
ORCID [0009-0001-6789-2774](https://orcid.org/0009-0001-6789-2774) · <https://mukreminnergiz.com>

Licence: BSD-3-Clause.
