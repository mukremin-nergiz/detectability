# Changelog

All notable changes to Detectability. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [semantic versioning](https://semver.org/). Each release is archived on Zenodo with its own DOI;
the concept DOI [10.5281/zenodo.23931815](https://doi.org/10.5281/zenodo.23931815) always resolves to the latest.

## [0.7.0] - 2026-11-14

DOI [10.5281/zenodo.24209921](https://doi.org/10.5281/zenodo.24209921)

### Added
- Remanent magnetisation of the target in `evaluate()` and `depth_limit()`, for the `magnetic` and `gradient`
  methods. **The default is unchanged: with nothing given, the magnetisation is induced only and every earlier result
  stays as it was.** Two ways to give it, as target fields:
  - a vector: `remanence` (A/m), `remanence_inclination` and `remanence_declination` (degrees), all three together,
    added to the induced magnetisation;
  - a Koenigsberger ratio: `koenigsberger_ratio` (Q, remanent over induced). Q is applied to the magnetisation induced
    by the susceptibility contrast and taken along the present field, so the total is (1 + Q) times the induced one.
    That is the textbook ratio where the host's magnetisation is negligible (steel in soil), not in a magnetic host
    rock, and the direction is the optimistic case: a remanence pointing otherwise gives less, and against the field
    it partly cancels the induced part (give it as a vector for another direction).
  No material in the tables has a remanence or a Q of its own, and remanence of the host is not modelled. Every
  result's assumptions say which case it is, and the reproduction script carries the values that were used.
- Validation rows for remanence: a sphere with induced and remanent magnetisation in different directions against an
  independent code (Harmonica `dipole_magnetic`, 4e-14 %); a remanence at right angles to the field against the
  closed-form dipole field; and a vertical remanence against the vertical derivative of the gravity cylinder
  (Poisson's relation: a cross-check with this library's own gravity code, not an independent one).

### Changed
- The vacuum permeability in `detectability.magnetics` is the CODATA 2018 value (1.25663706212e-6 H/m), as in
  `detectability.gpr`, instead of 4π × 10⁻⁷. The two differ by 5.4e-10. Induced anomalies do not depend on it; a
  remanent magnetisation given in A/m does, and the comparison with Harmonica showed it (9e-8 % before, 4e-14 % after).
- The ERT service is sent only the target fields ERT uses, and only those that were given.

## [0.6.1] - 2026-11-14

DOI [10.5281/zenodo.24209492](https://doi.org/10.5281/zenodo.24209492)

### Added
- GPR: the reproduction script now writes the antenna frequency that was used when it was chosen from the target
  depth (`gpr_frequency`), so the script no longer depends on the depth-frequency rule staying the same; the result's
  `details` say where the frequency came from (`frequency_source`: `"auto"` or `"given"`).
- Water under either name: `fresh water` and `distilled water` resolve to the `water` row for density and
  susceptibility, and `water` resolves to `fresh water` for GPR, instead of the method coming back as not assessed.

### Changed
- Inputs that no survey can have are refused with a `ValueError` that names the field, instead of giving a number or
  an error from inside a model: a negative field or instrument noise (it used to disappear in the quadrature sum), a
  negative sensor height, a main field given with an inclination outside ±90°, a relative permittivity below 1, a
  negative conductivity, a resistivity of zero or less, a date that is not an ISO date, a value of the wrong type
  (text where a number belongs, a boolean, NaN or infinity), `methods` that is not a list of names, and a target or
  survey that is not a dictionary. A gravity, magnetic or gradient assessment with zero effective noise is refused as
  well: it used to return an infinite signal-to-noise ratio.
- The magnetic assumption line no longer says "no remanence unless given": `evaluate()` takes no remanence. It now
  says that the result is for induced magnetisation only. (The model functions in `detectability.magnetics` still
  accept a remanent magnetisation.)

## [0.6.0] - 2026-11-13

DOI [10.5281/zenodo.24200875](https://doi.org/10.5281/zenodo.24200875)

### Added
- `depth_limit()`: the inverse of a detectability assessment. Instead of "can this method see the target at this
  depth?", it answers "how deep can the target be before the method loses it?" — the question a survey plan turns on.
  New module `detectability.limits`, exported as `depth_limit` and `Limit`.
  `depth_top` on the target is ignored and may be left out; where the shape needs a `depth_bottom`, the body keeps its
  thickness as it moves down. `threshold` defaults to the clearly-detectable signal-to-noise and takes
  `detect.MARGINAL` for the lower one; `search`, `steps` and `refine` control the sweep.
  A coarse geometric scan brackets the **shallowest** crossing of the threshold and a bisection refines it, so a
  response that steps rather than falls smoothly with depth — as GPR does when the antenna frequency is left to the
  default and chosen from the depth — still gives the conservative answer. The result says when a method never
  reaches the threshold at all, and reports the signal-to-noise at the shallowest depth searched for context.

## [0.5.0] - 2026-11-11

DOI [10.5281/zenodo.24136298](https://doi.org/10.5281/zenodo.24136298)

### Added
- Electrical resistivity tomography: a new method, `ert`, and module `detectability.ert` (optional dependency pyGIMLi,
  `pip install "detectability[ert]"`). Electrodes centred over the target (survey fields `ert_array`: dipole-dipole,
  Wenner, Schlumberger or gradient; `ert_electrodes`, default 48; `ert_spacing`, default a spread about six times the
  depth to reach; `ert_noise`, default 3 %). The apparent resistivities with and without the target, on the same
  finite-element mesh, give the data anomaly and S/N; with `ert_invert` (default on) noisy synthetic data are inverted
  and the result says whether the target is recovered (a detectable verdict is lowered to marginal if not). Typical
  resistivities from Loke (1999, Table 1, after Keller & Frischknecht 1966); target fields `resistivity` and
  `host_resistivity` override them. Without pyGIMLi the method is reported as not available, or computed by an ERT
  service if `DETECTABILITY_ERT_URL` is set.
- Validation rows: pyGIMLi's apparent resistivity over homogeneous ground, and a two-layer Wenner sounding against the
  image-series solution; the values are stored (`data/ert_reference.json`, `tools/ert_reference.py --check`) and CI
  recomputes them in a separate job with pyGIMLi.

### Fixed
- The reproduction script wrote booleans as JSON (`true`); it now writes Python (`True`) and leaves out the settings of
  methods that were not asked for.

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

[0.5.0]: https://github.com/mukremin-nergiz/detectability/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/mukremin-nergiz/detectability/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mukremin-nergiz/detectability/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/mukremin-nergiz/detectability/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mukremin-nergiz/detectability/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/mukremin-nergiz/detectability/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mukremin-nergiz/detectability/releases/tag/v0.1.0
