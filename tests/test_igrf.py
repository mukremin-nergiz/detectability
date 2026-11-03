import datetime as dt

import pytest

from detectability import igrf

# Reference values from the British Geological Survey IGRF-14 web service (geomag.bgs.ac.uk), retrieved 3 Nov 2026,
# rounded by the service to 1 nT and 0.001 degree.
BGS = [
    # lat, lon, alt km, date, F nT, D deg, I deg
    (39.75, 37.02, 0.0, dt.date(2026, 7, 1), 48567, 6.313, 58.373),      # Sivas, Türkiye
    (-33.9, 18.4, 0.0, dt.date(2026, 7, 1), 24996, -26.673, -64.631),    # Cape Town
    (70.0, -100.0, 0.0, dt.date(2026, 7, 1), 57831, -0.041, 85.147),     # Canadian Arctic
    (0.0, -60.0, 5.0, dt.date(2026, 7, 1), 26671, -16.84, 8.983),        # Amazon, 5 km altitude
    (51.5, -0.1, 0.0, dt.date(2012, 3, 15), 48635, -1.345, 66.469),      # London, definitive model (DGRF) epoch
]


@pytest.mark.parametrize("lat, lon, alt, date, F, D, I", BGS)
def test_matches_bgs(lat, lon, alt, date, F, D, I):
    f = igrf.field(lat, lon, date, alt)
    assert f.total == pytest.approx(F, abs=1.0)
    assert f.declination == pytest.approx(D, abs=0.002)
    assert f.inclination == pytest.approx(I, abs=0.002)
    assert f.total == pytest.approx((f.north**2 + f.east**2 + f.down**2) ** 0.5)


def test_validity_range():
    with pytest.raises(ValueError):
        igrf.field(0, 0, 2031.0)
    with pytest.raises(ValueError):
        igrf.field(0, 0, 1899.0)


def test_decimal_year():
    assert igrf.decimal_year(dt.date(2026, 1, 1)) == 2026.0
    assert igrf.decimal_year(dt.date(2024, 7, 2)) == pytest.approx(2024 + 183 / 366)
    assert igrf.decimal_year(2026.5) == 2026.5
