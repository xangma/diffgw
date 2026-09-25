"""Independent scalar checks of the native signed-spin QNM interpolation rule."""

import math

import pytest

from dev.native_rom.reference.qnm import (
    BERTI_AUTHOR_220_SAMPLES, COOK_AUTHOR_220_SAMPLES,
    COOK_AUTHOR_COUNTERROTATING_SAMPLES, native_signed_220_spline,
)
from diffgw_core.kerr_qnm_220_data import (
    KNOTS_220, COEFFS_220, KNOTS_2M20, COEFFS_2M20,
)


def native_spline(spin):
    # Production coefficient data are passed into, never imported by, the
    # scalar reference. Author rows below check their physical content.
    return native_signed_220_spline(
        spin, KNOTS_220, COEFFS_220, KNOTS_2M20, COEFFS_2M20)


def test_native_signed_spline_uses_degree_five_and_both_branches():
    knots = (0.0,)*6 + (1.0,)*6
    # With an open degree-five knot vector, only the last positive control
    # point contributes x^5; this would fail under a lower-degree rule.
    positive = tuple((2 + int(i == 5), -3.0) for i in range(6))
    negative = tuple((4 + i/5, -(5 + 2*i/5)) for i in range(6))
    spin = 1 - math.exp(-0.5)
    for signed_spin, expected in ((spin, (2.03125, 3.0)), (-spin, (4.5, 6.0))):
        actual = native_signed_220_spline(
            signed_spin, knots, positive, knots, negative)
        assert tuple(map(float, actual)) == pytest.approx(expected, rel=0, abs=1e-15)


@pytest.mark.parametrize("spin,frequency,damping", BERTI_AUTHOR_220_SAMPLES)
def test_native_signed_spline_against_berti_author_rows(spin, frequency, damping):
    actual = native_spline(spin)
    assert float(actual[0]) == pytest.approx(frequency, rel=0, abs=1e-9)
    assert float(actual[1]) == pytest.approx(damping, rel=0, abs=1e-9)


@pytest.mark.parametrize("spin,frequency,damping",
                         COOK_AUTHOR_220_SAMPLES + COOK_AUTHOR_COUNTERROTATING_SAMPLES)
def test_native_signed_spline_against_cook_author_rows(spin, frequency, damping):
    actual = native_spline(spin)
    assert float(actual[0]) == pytest.approx(frequency, rel=0, abs=1e-10)
    assert float(actual[1]) == pytest.approx(damping, rel=0, abs=1e-10)


@pytest.mark.parametrize("spin,limit", [(1.0, 0.9999), (-1.0, -0.9999)])
def test_native_signed_spline_clamps_to_signed_frontend_domain(spin, limit):
    assert native_spline(spin) == native_spline(limit)


def test_native_signed_spline_rejects_invalid_input():
    with pytest.raises(ValueError, match="finite"):
        native_spline(float("nan"))
    with pytest.raises(ValueError, match="lengths"):
        native_signed_220_spline(0.1, (0.0,), ((1.0, -1.0),),
                                 (0.0,), ((1.0, -1.0),))
