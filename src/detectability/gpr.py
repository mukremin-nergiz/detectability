"""Ground-penetrating radar (GPR): wave speed and attenuation in the ground, reflection at the target, and the echo of
a buried target compared with what the radar system can still detect.

The model is a radar range (loss budget) calculation, in the spirit of GPR survey-design practice (Annan & Cosway 1992,
"Ground penetrating radar survey design", SAGEEP '92, 329-351; Annan 2009, "Electromagnetic principles of ground
penetrating radar", in Jol (ed.), Ground Penetrating Radar: Theory and Applications, Elsevier):

    echo amplitude (relative to the amplitude 1 m from the antenna) = |R| * S * exp(-2 alpha r) * 1 m / (2 r)

with R the normal-incidence reflection coefficient between host and target, S the target-size factor (1 for a large
flat reflector), alpha the attenuation of the host and r the range from the antenna to the reflecting surface. The echo is
detectable while its loss stays within the system performance Q (dB, the largest two-way loss the system can still
record, referred to 1 m). Signal-to-noise ratio = echo / 10**(-Q/20).

Each step and its limits:
- Wave speed and attenuation are exact for a homogeneous medium with frequency-independent permittivity and
  conductivity. Real soils lose more at high frequency (water relaxation, scattering), so results above ~500 MHz in
  wet or clayey ground are optimistic.
- R is exact for a plane wave at normal incidence on a plane boundary; a perfect conductor (metal) has R = -1.
- S is geometric optics for curved targets: a sphere of radius a at range r returns a / (r + a) of a flat reflector's
  echo and a horizontal cylinder sqrt(a / (r + a)) (one curved direction). Where the target is small compared with the
  wavelength (k a < 1, k = 2 pi / lambda) the echo falls in the Rayleigh regime as (k a)**2 (sphere) and (k a)**1.5
  (cylinder), joined continuously at k a = 1. Flat tops smaller than the first Fresnel zone (radius sqrt(lambda r / 2))
  return the fraction of the zone they cover. Antenna gain, beam pattern, coupling and surface clutter are in Q.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C0 = 299_792_458.0            # speed of light in vacuum, m/s
EPS0 = 8.8541878128e-12       # vacuum permittivity, F/m (CODATA 2018)
MU0 = 1.25663706212e-6        # vacuum permeability, H/m (CODATA 2018)
NEPER_TO_DB = 20.0 / np.log(10.0)

# Typical relative permittivity and conductivity (mS/m), each as (typical, low, high). Ranges from the table of
# Davis & Annan (1989, Geophysical Prospecting 37, 531-551), as reproduced in USACE EM 1110-1-1802 (1995) Table 4-3;
# "typical" is the middle of the permittivity range and the geometric middle of the conductivity range. The same table
# gives velocity and attenuation, which the tests check against these values. Verify against the source before release.
PERMITTIVITY: dict[str, tuple[float, float, float]] = {
    "air": (1.0, 1.0, 1.0),
    "distilled water": (80.0, 80.0, 80.0),
    "fresh water": (80.0, 80.0, 80.0),
    "sea water": (80.0, 80.0, 80.0),
    "dry sand": (4.0, 3.0, 5.0),
    "wet sand": (25.0, 20.0, 30.0),
    "limestone": (6.0, 4.0, 8.0),
    "shale": (10.0, 5.0, 15.0),
    "silt": (17.5, 5.0, 30.0),
    "clay": (22.5, 5.0, 40.0),
    "granite": (5.0, 4.0, 6.0),
    "rock salt": (5.5, 5.0, 6.0),
    "ice": (3.5, 3.0, 4.0),
}
CONDUCTIVITY_MS_M: dict[str, tuple[float, float, float]] = {
    "air": (0.0, 0.0, 0.0),
    "distilled water": (0.01, 0.01, 0.01),
    "fresh water": (0.5, 0.5, 0.5),
    "sea water": (3000.0, 3000.0, 3000.0),
    "dry sand": (0.01, 0.01, 0.01),
    "wet sand": (0.3, 0.1, 1.0),
    "limestone": (1.0, 0.5, 2.0),
    "shale": (10.0, 1.0, 100.0),
    "silt": (10.0, 1.0, 100.0),
    "clay": (45.0, 2.0, 1000.0),
    "granite": (0.1, 0.01, 1.0),
    "rock salt": (0.1, 0.01, 1.0),
    "ice": (0.01, 0.01, 0.01),
}
METALS = {"metal"}  # perfect conductors: R = -1 whatever the host

# Antenna centre frequency for a target depth (Annan 2001, as given by Robinson et al. 2013, Table 2 in USEPA CLU-IN
# "Ground Penetrating Radar"): a rule of thumb that keeps the resolution near a quarter of the depth.
DEPTH_TO_FREQUENCY_MHZ = ((0.5, 1000.0), (1.0, 500.0), (2.0, 200.0), (5.0, 100.0), (10.0, 50.0), (30.0, 25.0),
                          (50.0, 10.0))
ANTENNAS_MHZ = (10.0, 25.0, 50.0, 100.0, 200.0, 250.0, 400.0, 500.0, 800.0, 1000.0, 1600.0)

DEFAULT_SYSTEM_PERFORMANCE_DB = 100.0  # typical dynamic range 90-130 dB (Benson et al. 1984; USACE 1995)


@dataclass(frozen=True)
class Medium:
    permittivity: float       # relative
    conductivity: float       # S/m

    def wavenumber(self, f: float) -> complex:
        """Complex wavenumber beta - i alpha at frequency f (Hz)."""
        w = 2 * np.pi * f
        eps = self.permittivity * EPS0 - 1j * self.conductivity / w
        k = w * np.sqrt(MU0 * eps + 0j)
        return complex(k.real, -abs(k.imag))

    def velocity(self, f: float) -> float:
        """Phase velocity, m/s."""
        return 2 * np.pi * f / self.wavenumber(f).real

    def attenuation_db(self, f: float) -> float:
        """Amplitude attenuation, dB per metre (one way)."""
        return -self.wavenumber(f).imag * NEPER_TO_DB

    def impedance(self, f: float) -> complex:
        w = 2 * np.pi * f
        return np.sqrt(MU0 / (self.permittivity * EPS0 - 1j * self.conductivity / w) + 0j)

    def wavelength(self, f: float) -> float:
        return self.velocity(f) / f


def medium(name: str) -> Medium:
    try:
        return Medium(PERMITTIVITY[name][0], CONDUCTIVITY_MS_M[name][0] / 1000.0)
    except KeyError:
        raise KeyError(f"no GPR properties for {name!r}; known: {', '.join(sorted(set(PERMITTIVITY) | METALS))}") from None


def low_loss_attenuation_db(permittivity: float, conductivity: float) -> float:
    """High-frequency (low-loss) limit of the attenuation, dB/m: 1636 sigma / sqrt(eps_r) with sigma in S/m."""
    return NEPER_TO_DB * conductivity / 2 * np.sqrt(MU0 / EPS0) / np.sqrt(permittivity)


def reflection(host: Medium, target: Medium | None, f: float) -> complex:
    """Normal-incidence reflection coefficient from ``host`` onto ``target`` (None: perfect conductor, R = -1)."""
    if target is None:
        return complex(-1.0)
    z1, z2 = host.impedance(f), target.impedance(f)
    return complex((z2 - z1) / (z2 + z1))


def fresnel_radius(wavelength: float, r: np.ndarray) -> np.ndarray:
    """Radius of the first Fresnel zone on a reflector at range r (two-way path), m."""
    return np.sqrt(wavelength * r / 2 + wavelength**2 / 16)


def size_factor(kind: str, r: np.ndarray, wavelength: float, a: float = 0.0, width: float = 0.0,
                length: float = 0.0) -> np.ndarray:
    """Echo of the target relative to an infinite flat reflector at the same range (see the module notes).

    kind: "sphere" (radius a), "cylinder" (horizontal, radius a), "disc" (flat top of radius a), "strip" (flat top of
    width ``width``, long along strike), "rectangle" (flat top ``width`` x ``length``), "plane" (infinite).
    """
    r = np.asarray(r, float)
    ka = 2 * np.pi * a / wavelength
    if kind == "sphere":
        return a / (r + a) * min(1.0, ka) ** 2
    if kind == "cylinder":
        return np.sqrt(a / (r + a)) * min(1.0, ka) ** 1.5
    rf = fresnel_radius(wavelength, r)
    if kind == "disc":
        return np.minimum(1.0, (a / rf) ** 2)
    if kind == "strip":
        return np.minimum(1.0, width / (2 * rf))
    if kind == "rectangle":
        return np.minimum(1.0, width * length / (np.pi * rf**2))
    if kind == "plane":
        return np.ones_like(r)
    raise ValueError(f"unknown target kind {kind!r}")


def echo(r: np.ndarray, host: Medium, R: complex, factor: np.ndarray, f: float) -> np.ndarray:
    """Echo amplitude relative to the amplitude 1 m from the antenna, for reflecting surfaces at ranges r (m)."""
    r = np.maximum(np.asarray(r, float), 1e-3)
    alpha = host.attenuation_db(f) / NEPER_TO_DB
    return abs(R) * factor * np.exp(-2 * alpha * r) / (2 * r)


def frequency_for_depth(depth: float) -> float:
    """Antenna centre frequency (MHz) for a target depth, from the depth-frequency rule, snapped to a common antenna."""
    d, fr = np.array(DEPTH_TO_FREQUENCY_MHZ).T
    f = float(np.exp(np.interp(np.log(max(depth, 1e-3)), np.log(d), np.log(fr))))
    return min(ANTENNAS_MHZ, key=lambda a: abs(np.log(a / f)))


def two_way_time_ns(depth: float, velocity: float) -> float:
    return 2 * depth / velocity * 1e9
