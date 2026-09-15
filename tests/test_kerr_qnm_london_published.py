import math

from torchwave.approximants.kerr_qnm_london_published import dimensionless_qnm, kappa


def test_printed_domain_map_and_extremal_limits():
    assert math.isclose(kappa(0.0, 2, 2), math.log(2.0, 3.0) ** .5)
    for mode in ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)):
        z = dimensionless_qnm(0.0, *mode)
        assert math.isfinite(z.real) and math.isfinite(z.imag)
    for mode in ((2, 2), (3, 3), (4, 4), (4, 3)):
        assert dimensionless_qnm(1.0, *mode).real == mode[1] / 2


def test_printed_220_and_330_are_distinct_fundamental_fits():
    z22 = dimensionless_qnm(0.7, 2, 2)
    z33 = dimensionless_qnm(0.7, 3, 3)
    assert z22.real > 0 and z33.real > z22.real
    assert dimensionless_qnm(0.7, 2, 2, overtone=1) != z22


def test_unprinted_first_overtone_is_rejected():
    try:
        dimensionless_qnm(0.7, 2, 1, overtone=1)
    except ValueError:
        pass
    else:
        raise AssertionError("(2,1) first-overtone fit is not printed")
