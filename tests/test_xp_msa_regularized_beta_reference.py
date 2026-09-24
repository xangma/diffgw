"""Paper-derived opening-angle identities, without any waveform oracle."""

import math

import pytest

from diffgw_reference.xp_msa_regularized_beta import one_spin_aligned_beta


def test_regularized_invariants_are_physical_not_split_roots():
    result = one_spin_aligned_beta(
        40.0, 20.0, (0.03, -0.04, 0.4), (0.0, 0.0, -0.2),
        v_ref=0.25, l_beta=0.7,
    )
    sx = (40.0 / 60.0) ** 2 * 0.03
    sy = (40.0 / 60.0) ** 2 * -0.04
    sz = (40.0 / 60.0) ** 2 * 0.4 + (20.0 / 60.0) ** 2 * -0.2
    s2 = sx**2 + sy**2 + sz**2
    l0 = result.eta / 0.25
    assert result.spin_squared == pytest.approx(s2, abs=1e-17)
    assert result.s_average == result.spin_squared
    assert result.j_ref_squared == pytest.approx((l0 + sz) ** 2 + sx**2 + sy**2)
    assert result.c1 == pytest.approx(
        0.5 * 0.25 * (result.j_ref_squared - l0**2 - result.s_average),
        abs=1e-16,
    )
    assert result.c1 == pytest.approx(result.eta * sz)
    j2_from_msa = 0.7**2 + 2 * result.c1 * 0.7 / result.eta + result.s_average
    assert result.j_beta**2 == pytest.approx(j2_from_msa, abs=1e-16)
    cos_from_triangle = (j2_from_msa + 0.7**2 - result.spin_squared) / (2 * result.j_beta * 0.7)
    assert result.cos_beta == pytest.approx(cos_from_triangle, abs=1e-15)


@pytest.mark.parametrize("azimuth", (-2.0, 0.4, 2.1))
def test_exact_and_tiny_transverse_spins_share_continuous_limit(azimuth):
    args = (40.0, 20.0)
    spin2 = (0.0, 0.0, -0.2)
    exact = one_spin_aligned_beta(*args, (0.0, 0.0, 0.4), spin2,
                                  v_ref=0.25, l_beta=0.7)
    assert exact.cos_beta == 1.0
    assert exact.beta == 0.0
    for epsilon in (1e-3, 1e-6, 1e-9):
        near = one_spin_aligned_beta(
            *args, (epsilon * math.cos(azimuth), epsilon * math.sin(azimuth), 0.4),
            spin2, v_ref=0.25, l_beta=0.7,
        )
        expected_perp = (40.0 / 60.0) ** 2 * epsilon
        assert near.spin_perpendicular == pytest.approx(expected_perp, rel=2e-15)
        assert near.beta == pytest.approx(
            math.atan2(expected_perp, 0.7 + 0.4 * (40.0 / 60.0) ** 2
                       - 0.2 * (20.0 / 60.0) ** 2), rel=2e-15,
        )
        assert near.beta < epsilon
    assert near.beta < 1e-9


def test_reversed_j_has_pi_opening_angle():
    # The Newtonian reference J is forward, but a smaller supplied L_beta
    # makes J point opposite L. A blanket cos(beta)=1 endpoint would be wrong.
    reversed_j = one_spin_aligned_beta(
        30.0, 30.0, (0.0, 0.0, -0.8), (0.0, 0.0, 0.0),
        v_ref=0.2, l_beta=0.1,
    )
    assert reversed_j.cos_beta == -1.0
    assert reversed_j.beta == math.pi
    tilted = one_spin_aligned_beta(
        30.0, 30.0, (1e-9, 0.0, -0.8), (0.0, 0.0, 0.0),
        v_ref=0.2, l_beta=0.1,
    )
    assert tilted.beta < math.pi
    assert math.pi - tilted.beta == pytest.approx(math.atan2(0.25e-9, 0.1))


def test_zero_total_j_is_not_assigned_an_euler_chart():
    with pytest.raises(ValueError, match="J_beta is zero"):
        one_spin_aligned_beta(
            30.0, 30.0, (0.0, 0.0, -0.8), (0.0, 0.0, 0.0),
            v_ref=0.2, l_beta=0.2,
        )
