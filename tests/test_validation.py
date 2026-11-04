import pytest

from detectability import validation

CASES = validation.cases()


@pytest.mark.parametrize("case", CASES, ids=[c.key for c in CASES])
def test_case_within_tolerance(case):
    assert case.passed, f"{case.key}: {case.error:.3g} % > {case.tolerance:.3g} %"


def test_igrf_within_tolerance():
    for r in validation.igrf_rows():
        assert abs(r["dF"]) <= validation.IGRF_TOLERANCE_NT, r["site"]
        assert abs(r["dD"]) <= validation.IGRF_TOLERANCE_DEG and abs(r["dI"]) <= validation.IGRF_TOLERANCE_DEG, r["site"]


def test_table_lists_every_case():
    table = validation.markdown_table()
    assert table.count("\n") == len(CASES) + 2  # header, rule, cases, IGRF row


def test_harmonica_reference_is_packaged_and_complete():
    ref = validation.harmonica_reference()
    assert ref is not None and ref["doi"].startswith("10.5281/zenodo.")
    assert set(ref["cases"]) == set(validation.HARMONICA_CASES)
    assert sum(c.check == "Independent code" for c in CASES) == len(validation.HARMONICA_CASES)
