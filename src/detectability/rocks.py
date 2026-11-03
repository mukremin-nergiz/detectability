"""Typical physical properties of near-surface materials: density (g/cm^3) and magnetic susceptibility (SI), each as
(typical, low, high).

Sources: Telford, Geldart & Sheriff (1990), Applied Geophysics, 2nd ed., Table 6.1 (densities) and Table 3.3
(susceptibilities), rounded. Values vary widely with porosity, saturation, weathering and magnetite content; use site data
when available. Verify against the source before the v1 release.
"""

DENSITY_G_CM3: dict[str, tuple[float, float, float]] = {
    # name: (typical, low, high)
    "air": (0.0, 0.0, 0.0),
    "water": (1.0, 1.0, 1.05),
    "soil": (1.92, 1.2, 2.4),
    "clay": (2.21, 1.63, 2.6),
    "sand": (2.0, 1.7, 2.3),
    "gravel": (2.0, 1.7, 2.4),
    "sandstone": (2.35, 1.61, 2.76),
    "shale": (2.40, 1.77, 3.2),
    "limestone": (2.55, 1.93, 2.90),
    "dolomite": (2.70, 2.28, 2.90),
    "marble": (2.75, 2.6, 2.9),
    "rock salt": (2.22, 2.10, 2.60),
    "gypsum": (2.35, 2.2, 2.6),
    "granite": (2.64, 2.50, 2.81),
    "basalt": (2.99, 2.70, 3.30),
    "gabbro": (3.03, 2.70, 3.50),
}


# Telford et al. Table 3.3 gives susceptibility x 10^3 (SI); stored here in SI. Where the table gives only an average,
# low and high repeat it. Air and water are effectively zero (water is weakly diamagnetic, about -9e-6).
SUSCEPTIBILITY_SI: dict[str, tuple[float, float, float]] = {
    "air": (0.0, 0.0, 0.0),
    "water": (0.0, 0.0, 0.0),
    "clay": (0.2e-3, 0.2e-3, 0.2e-3),
    "dolomite": (0.1e-3, 0.0, 0.9e-3),
    "limestone": (0.3e-3, 0.0, 3e-3),
    "sandstone": (0.4e-3, 0.0, 20e-3),
    "shale": (0.6e-3, 0.01e-3, 15e-3),
    "rock salt": (-0.01e-3, -0.01e-3, -0.01e-3),
    "gypsum": (-0.01e-3, -0.01e-3, -0.01e-3),
    "slate": (6e-3, 0.0, 35e-3),
    "schist": (1.4e-3, 0.3e-3, 3e-3),
    "granite": (2.5e-3, 0.0, 50e-3),
    "diorite": (85e-3, 0.6e-3, 120e-3),
    "gabbro": (70e-3, 1e-3, 90e-3),
    "basalt": (70e-3, 0.2e-3, 175e-3),
    "andesite": (160e-3, 160e-3, 160e-3),
    "peridotite": (150e-3, 90e-3, 200e-3),
    "magnetite": (6.0, 1.2, 19.2),
}


def _typical(table: dict, name: str, what: str) -> float:
    try:
        return table[name][0]
    except KeyError:
        raise KeyError(f"no {what} for {name!r}; known: {', '.join(sorted(table))}") from None


def contrast(target: str, host: str) -> float:
    """Density contrast target - host in kg/m^3 from the typical values (e.g. air-filled void in limestone: -2550)."""
    return (_typical(DENSITY_G_CM3, target, "density") - _typical(DENSITY_G_CM3, host, "density")) * 1000.0


def susceptibility_contrast(target: str, host: str) -> float:
    """Susceptibility contrast target - host in SI from the typical values (e.g. void in limestone: -3e-4)."""
    return _typical(SUSCEPTIBILITY_SI, target, "susceptibility") - _typical(SUSCEPTIBILITY_SI, host, "susceptibility")
