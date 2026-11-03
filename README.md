# detectability

Will your geophysical survey see it? Describe a target (shape, depth, size, physical-property contrast) and get the
forward-modelled response, the expected signal-to-noise ratio for a given instrument and field noise, the station spacing
and profile length needed, and a detectability verdict with its assumptions.

Status: early development (v0.1). Resistivity and GPR later.

| Module | Contents |
|---|---|
| `gravity` | sphere, horizontal cylinder, right rectangular prism (Nagy/Plouff), vertical cylinder, vertical step, dyke |
| `magnetics` | total-field anomaly: sphere, horizontal cylinder, 2D polygon, dyke (any dip), vertical contact; induced and remanent magnetisation |
| `igrf` | IGRF-14 main field (total intensity, inclination, declination) from location and date, official IAGA coefficients |
| `rocks` | typical densities and magnetic susceptibilities with ranges (Telford et al. 1990) |
| `detect` | instrument and field noise, signal-to-noise ratio, verdict, station spacing, profile length, assumptions |

Every model is tested against an independent result: analytical limits, numerical integration, Poisson's relation
between the gravity and magnetic codes, and the British Geological Survey IGRF-14 service.

Run the tests: `PYTHONPATH=src python -m pytest`.

Author: Mükremin Nergiz — Geophysical Engineer & Software Developer (Independent Researcher, Sivas, Türkiye). ORCID: [0009-0001-6789-2774](https://orcid.org/0009-0001-6789-2774).
