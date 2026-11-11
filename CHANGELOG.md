# Changelog

All notable changes to Detectability. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [semantic versioning](https://semver.org/). Each release is archived on Zenodo with its own DOI;
the concept DOI [10.5281/zenodo.23931815](https://doi.org/10.5281/zenodo.23931815) always resolves to the latest.

## [0.4.0] - 2026-11-11

DOI [10.5281/zenodo.24134972](https://doi.org/10.5281/zenodo.24134972)

### Added
- Ground-penetrating radar: a new method, `gpr`, and module `detectability.gpr`. The echo of the target's top surface
  (reflection coefficient, two-way attenuation, spreading and a target-size factor) is compared with the system
  performance (survey field `gpr_system_performance`, default 100 dB). Antenna centre frequency `gpr_frequency` (MHz;
  default from the target depth), velocity and attenuation from the exact complex wavenumber, trace spacing at most a
  quarter wavelength, recording window. Typical relative permittivity and conductivity of fourteen materials (Davis &
  Annan 1989), new materials such as `dry sand`, `wet sand`, `silt`, `ice` and `metal` (perfect conductor); target
  fields `permittivity`, `conductivity`, `host_permittivity` and `host_conductivity` override the table. A vertical
  contact is not assessed by GPR.
- Validation rows: low-loss attenuation limit, lossless velocity, reflection coefficient, very large sphere against a
  flat reflector, and the velocities of ten materials against Davis & Annan's table.
- `MethodResult.details`: method-specific figures (for GPR: frequency, velocity, attenuation, reflection coefficient,
  two-way time, time window, vertical resolution, margin).

## [0.3.0] - 2026-11-04

DOI [10.5281/zenodo.23949047](https://doi.org/10.5281/zenodo.23949047)

### Added
- Magnetic vertical gradient: a new method, `gradient`, models the reading of a two-sensor vertical gradiometer, the
  total-field anomaly at the lower sensor minus that at the upper sensor, in nT over the baseline. Instrument presets
  from the manufacturers' data sheets: Bartington Grad601 (1 m baseline, 0.1 nT), Sensys FGM650 (0.65 m, 0.04 nT),
  Foerster FEREX 4.034 (0.55 m, 0.04 nT) and GEM GSM-19G (0.56 m, 0.031 nT). Survey fields `gradiometer`,
  `gradient_field_noise` (default 0.5 nT, an assumption), `gradient_instrument_noise`, `gradiometer_height` (default
  0.2 m) and `gradiometer_baseline`.
- Validation row: the gradiometer reading of a sphere at the pole against the closed-form dipole field at both sensors.
- `DEFAULT_METHODS`: the default methods stay `gravity` and `magnetic`; `gradient` is added on request.

## [0.2.1] - 2026-11-04

DOI [10.5281/zenodo.23948416](https://doi.org/10.5281/zenodo.23948416)

### Added
- Published on PyPI: `pip install detectability`. Releases are uploaded from GitHub Actions with trusted publishing
  (no stored token); the package is built, checked with `twine check --strict` and tested before upload.
- Package metadata: supported Python versions, and links to the documentation, validation, issue tracker and citation.

### Changed
- The reproduction script from `Report.script()` pins the exact version (`pip install "detectability==0.2.1"`).
- Absolute links in the README, so they also work on PyPI.

No change to the models or results.

## [0.2.0] - 2026-11-04

DOI [10.5281/zenodo.23947287](https://doi.org/10.5281/zenodo.23947287)

### Added
- `gravity.vertical_cylinder_tensor`: the gravity gradient tensor of a vertical cylinder.
- Validation against an independent code: six "Independent code" rows compare the prism, the sphere and the vertical
  dyke (gravity and magnetic) with Harmonica 0.7.0 (Fatiando a Terra Project,
  [doi:10.5281/zenodo.13308312](https://doi.org/10.5281/zenodo.13308312)). The reference values are stored in
  `data/harmonica_reference.json`; `tools/harmonica_reference.py --check` recomputes them, and a CI job does so on
  every change. Harmonica is an optional extra (`pip install -e ".[validation]"`), not a dependency.
- Four validation rows for the vertical cylinder: exact on-axis formula, cross-section quadrature, volume dipole
  quadrature (magnetic) and Poisson's relation.

### Changed
- **The vertical cylinder is now exact** in gravity and magnetics. The volume integral is done in closed form in depth
  and along each ray from the station, leaving one integral round the rim, summed with the trapezoid rule to about 15
  digits, also right above the rim. It replaces a square prism of equal cross-section, which was off by 0.34 % in
  gravity and 0.67 % in the magnetic field on the axis of the validation example: **vertical-cylinder results change
  by up to that amount.** The validation tolerance went from 3 % to 1e-10 %.

## [0.1.1] - 2026-11-03

DOI [10.5281/zenodo.23932438](https://doi.org/10.5281/zenodo.23932438)

### Changed
- Instrument presets checked against the manufacturers' data sheets; each preset records its figure and source
  (`detect.INSTRUMENTS`), and every result states it. LaCoste & Romberg G 10 → 20 µGal (upper value of the stated
  repeatability), proton precession 0.2 → 0.1 nT (Geometrics G-856AX), Overhauser 0.05 → 0.022 nT (GEM GSM-19 v7),
  caesium vapour 0.01 → 0.008 nT (Geometrics G-858). Magnetic verdicts are practically unchanged, because field noise
  dominates.

### Removed
- The fluxgate gradiometer preset: a gradiometer measures a vertical gradient, not the total-field anomaly modelled.

## [0.1.0] - 2026-11-03

DOI [10.5281/zenodo.23931816](https://doi.org/10.5281/zenodo.23931816)

### Added
- First public release. `evaluate()` models the gravity and magnetic anomaly of a described target along a profile,
  takes the main field from IGRF-14 for the location and date, compares the anomaly with instrument and field noise,
  and returns a verdict per method, the recommended method, station spacing, profile length, every assumption and a
  script that reproduces the result.
- Gravity: sphere, horizontal cylinder, rectangular prism (Nagy; Plouff) and its gradient tensor, vertical cylinder,
  vertical step, dyke of any dip, 2D polygon (Talwani; Won & Bevis). Magnetics (induced and remanent magnetisation):
  sphere, horizontal cylinder, prism, vertical cylinder, 2D polygon, dyke of any dip, vertical contact.
- Validation module and tests against independent results; CI on Python 3.10–3.13.

[0.4.0]: https://github.com/mukremin-nergiz/detectability/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mukremin-nergiz/detectability/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/mukremin-nergiz/detectability/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mukremin-nergiz/detectability/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/mukremin-nergiz/detectability/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mukremin-nergiz/detectability/releases/tag/v0.1.0
