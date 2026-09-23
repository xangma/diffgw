"""Independent SI/IAU audit of the three waveform-adapter unit constants.

Specification: si-iau-waveform-units/v1. See docs/math/units.md for derivations,
source locations, rounding policy, and the nominal solar-mass qualification.
No expected value is constructed from a production constant.
"""

import math

import pytest

from diffgw_core import constants

sp = pytest.importorskip("sympy", reason="install diffgw[reference] for the unit audit")

UNIT_SPEC_VERSION = "si-iau-waveform-units/v1"

# BIPM SI metre definition, read 2026-09-22:
# https://www.bipm.org/en/si-base-units/metre
EXACT_C = sp.Integer(299792458)

# IAU 2015 B3, arXiv:1510.07674v1, p. 3, solar conversion constants table.
# The nominal GM is exact; no measured G or solar mass in kg enters this value.
EXACT_NOMINAL_SOLAR_GM = sp.Rational("1.3271244e20")

# IAU 2012 B2, recommendation 1, PDF p. 3 (B2 printed p. 1):
# https://iauarchive.eso.org/static/resolutions/IAU2012_English.pdf
EXACT_AU = sp.Integer(149597870700)

# IAU 2015 B2, arXiv:1510.06262v1, p. 4, note 4: pc = (648000/pi) au.
EXACT_MEGAPARSEC = sp.Integer(10)**6 * sp.Integer(648000) * EXACT_AU / sp.pi


@pytest.mark.parametrize(
    "name, exact, maximum_ulps",
    [
        pytest.param("C_SI", EXACT_C, sp.S.Zero, id="units.c_si.v1"),
        pytest.param(
            "MSUN_SEC", EXACT_NOMINAL_SOLAR_GM / EXACT_C**3,
            sp.Rational(1, 2), id="units.msun_sec.v1",
        ),
        pytest.param(
            "MPC_SI", EXACT_MEGAPARSEC,
            sp.Rational(1, 2), id="units.mpc_si.v1",
        ),
    ],
)
def test_unit_constant_against_exact_definition(name, exact, maximum_ulps):
    actual = getattr(constants, name)
    assert math.isfinite(actual)
    # Both high-precision evaluations must select the same binary64 value.
    nearest = float(exact.evalf(80))
    assert nearest == float(exact.evalf(120))
    ulp = sp.Rational(*math.ulp(nearest).as_integer_ratio())
    actual_exact = sp.Rational(*actual.as_integer_ratio())
    error = sp.Abs(actual_exact - exact)
    for precision in (80, 120):
        assert error.evalf(precision) <= maximum_ulps * ulp, (
            f"{name}: error={sp.N(error / ulp, 20)} ulp; "
            f"allowed={maximum_ulps}; exact definition={exact}"
        )
