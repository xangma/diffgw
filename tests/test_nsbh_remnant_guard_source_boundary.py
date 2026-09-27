"""Source-only examples showing why Zappa factor clipping is not paper-defined."""

import pytest

from diffgw_reference.nsbh_remnant_rounding import (
    ZAPPA_MASS_TABLE,
    ZAPPA_SPIN_TABLE,
    bbh_remnant_interval,
    zappa_factor_interval,
)


@pytest.mark.parametrize("q,spin", ((3, "0"), (5, "-0.5")))
def test_printed_fit_factors_above_one_with_physical_full_remnant(q, spin):
    mass_bbh, spin_bbh = bbh_remnant_interval(q, spin)
    mass_factor = zappa_factor_interval(q, spin, 100, ZAPPA_MASS_TABLE)
    spin_factor = zappa_factor_interval(q, spin, 100, ZAPPA_SPIN_TABLE)

    assert mass_factor.lower > 1 and spin_factor.lower > 1
    assert 0 < (mass_bbh * mass_factor).lower
    assert (mass_bbh * mass_factor).upper < 1
    assert 0 < (spin_bbh * spin_factor).lower
    assert (spin_bbh * spin_factor).upper < 1
