"""Independent checks for the clean non-reexpanded ``(2,1)`` SPA branch."""

import math

import pytest
import sympy as sp
import torch

from torchwave.approximants.xhm_spa_clean import (
    normalized_spa_amplitude_21,
    physical_spa_amplitude_21,
    spa_hhat_21,
    spa_hhat_21_printed,
    taylor_t4_a6_3pn,
    taylor_t4_ratio,
    taylor_t4_ratio_printed,
    taylor_t4_xdot,
    td_hhat_21,
)
from torchwave.approximants.taylorf2 import TaylorF2


_ETA, _DELTA, _CHI_S, _CHI_A = 0.21, 0.4, 0.13, 0.04


# These values were evaluated with an independent 80-digit transcription of
# the published TaylorT4 ratio and TD H21 terms.  They are numeric checkpoints,
# rather than calls to the implementation's scalar expressions.
_CHECKPOINTS = {
    0.037: {
        "ratio": 0.99676185685535345914481043781438798,
        "td": 4.7189783287891286972372617910457e-7
        + 4.8681097315512101370878881352073e-3j,
        "spa": 6.6844745714033683117993759680961e-7
        + 6.8957205234093905753069683994121e-3j,
    },
    0.083: {
        "ratio": 0.98730954203327871414242529333795846,
        "td": 1.2004254002653747301986646670931e-5
        + 1.0735734592746002647511504484778e-2j,
        "spa": 1.7085335325789184442293999393813e-5
        + 1.5279885401057932283346931107966e-2j,
    },
    0.147: {
        "ratio": 0.97500265357111638883409437722514430,
        "td": 1.1954542079513573264039811204172e-4
        + 1.8610882241359307567519407517939e-2j,
        "spa": 1.7121627491747481633184228379494e-4
        + 2.6655022911785264183511926505821e-2j,
    },
}


@pytest.mark.parametrize("v, expected", tuple(_CHECKPOINTS.items()))
def test_printed_nonreexpanded_primary_numeric_checkpoints(v, expected):
    """Keep the published checkpoints attached to the named paper branch."""
    got_ratio = taylor_t4_ratio_printed(v, _ETA, _DELTA, _CHI_S, _CHI_A)
    got_td = td_hhat_21(v, _ETA, _DELTA, _CHI_S, _CHI_A)
    got_spa = spa_hhat_21_printed(v, _ETA, _DELTA, _CHI_S, _CHI_A)
    torch.testing.assert_close(
        got_ratio,
        torch.tensor(expected["ratio"], dtype=torch.float64),
        rtol=3e-15,
        atol=3e-16,
    )
    torch.testing.assert_close(
        got_td,
        torch.tensor(expected["td"], dtype=torch.complex128),
        rtol=3e-15,
        atol=3e-16,
    )
    torch.testing.assert_close(
        got_spa,
        torch.tensor(expected["spa"], dtype=torch.complex128),
        rtol=3e-15,
        atol=3e-16,
    )


def test_missing_v6_is_independent_reciprocal_series_coefficient():
    """Check the corrected term by an independent polynomial inversion."""
    v = torch.tensor((0.037, 0.083, 0.147), dtype=torch.float64)
    mass1 = torch.tensor((1.0 + _DELTA) / 2.0, dtype=torch.float64)
    mass2 = torch.tensor((1.0 - _DELTA) / 2.0, dtype=torch.float64)
    spin1 = torch.tensor(_CHI_S + _CHI_A, dtype=torch.float64)
    spin2 = torch.tensor(_CHI_S - _CHI_A, dtype=torch.float64)
    phase = TaylorF2.phase_coefficients(
        mass1,
        mass2,
        spin1,
        spin2,
        spin_order=7,
        phenom_d_calibration=False,
    )
    _, p2, p3, p4, _, _, p6, l6, *_ = phase
    log4v = torch.log(4.0 * v)

    def b(k, p, logarithmic):
        return (
            (k - 5) * (k - 8) * (p + logarithmic * log4v)
            + (2 * k - 13) * logarithmic
        ) / 40.0

    # Build the inverse recursively, instead of using the closed a6 formula.
    b_coefficients = {
        2: b(2, p2, torch.zeros_like(v)),
        3: b(3, p3, torch.zeros_like(v)),
        4: b(4, p4, torch.zeros_like(v)),
        6: b(6, p6, l6),
    }
    inverse = {0: torch.ones_like(v)}
    for order in range(1, 7):
        inverse[order] = -sum(
            (b_coefficients.get(index, torch.zeros_like(v)) * inverse[order - index])
            for index in range(1, order + 1)
        )
    expected_a6 = inverse[6]
    got_a6 = taylor_t4_a6_3pn(v, _ETA, _DELTA, _CHI_S, _CHI_A)
    torch.testing.assert_close(got_a6, expected_a6, rtol=2e-14, atol=2e-13)

    corrected = taylor_t4_ratio(v, _ETA, _DELTA, _CHI_S, _CHI_A)
    printed = taylor_t4_ratio_printed(v, _ETA, _DELTA, _CHI_S, _CHI_A)
    torch.testing.assert_close(
        corrected - printed, got_a6 * v**6, rtol=2e-14, atol=2e-15
    )


def test_leading_limit_has_published_dimensionless_normalization():
    # For m=1, Eq. (spaamp) gives Hhat_21 -> i*sqrt(2)*delta*v/3.
    delta = 0.37
    values = torch.tensor((1e-8, 2e-8, 4e-8), dtype=torch.float64)
    td = td_hhat_21(values, _ETA, delta, _CHI_S, _CHI_A)
    spa = spa_hhat_21(values, _ETA, delta, _CHI_S, _CHI_A)
    td_limit = 1j * delta * values / 3.0
    spa_limit = 1j * math.sqrt(2.0) * delta * values / 3.0
    torch.testing.assert_close(td / td_limit, torch.ones_like(td), rtol=0.0, atol=3e-8)
    torch.testing.assert_close(
        spa / spa_limit, torch.ones_like(spa), rtol=0.0, atol=3e-8
    )


# Independent Appendix-E E10 Hhat checkpoints for eta=.21, delta=.4,
# chi_s=.13, chi_a=.04.  The branch's product form must reproduce these
# coefficients through v^6; its retained ratio terms first leave O(v^7).
_E10_COEFFICIENTS = torch.tensor(
    (
        0.0 + 0.18856180831641267j,
        0.0 - 0.06505382386916237j,
        0.0 + 0.17673179962799012j,
        0.35568307574981892 - 0.52134429166709844j,
        0.0 + 0.24390043103141973j,
        0.79099622094570042 - 0.011295109361428364j,
    ),
    dtype=torch.complex128,
)


def test_printed_reexpansion_matches_verified_e10_through_v6():
    values = torch.tensor((0.004, 0.006, 0.009, 0.013, 0.017), dtype=torch.float64)
    powers = torch.arange(1, 7, dtype=torch.float64)
    e10 = values[:, None].pow(powers).to(torch.complex128) @ _E10_COEFFICIENTS
    branch = spa_hhat_21_printed(values, _ETA, _DELTA, _CHI_S, _CHI_A)
    residual = (branch - e10).abs()
    # No v^1,...,v^6 mismatch is allowed.  The remaining product terms are
    # bounded by a small constant times v^7 on this low-frequency interval.
    assert torch.all(residual / values.pow(7) < 7.0)
    torch.testing.assert_close(branch, e10, rtol=0.0, atol=3e-12)

    # Adding the independently derived ratio term first changes the SPA only
    # at relative order v^7, since the leading H21 term is O(v).  Keep this
    # check separate from the printed-paper coefficient checkpoint.
    corrected = spa_hhat_21(values, _ETA, _DELTA, _CHI_S, _CHI_A)
    correction = (corrected - branch).abs() / values.pow(7)
    assert torch.isfinite(correction).all()
    assert torch.all(correction < 20.0)


def test_bfh_427a_mass_weighted_spin_conversion_identity():
    """Check the physical spin conversion used by rendered BFH Eq. (4.27a).

    This is an independent symbolic derivation from x_i=m_i/M and
    chi_{1,2}=chi_s +/- chi_a.  It does not use the E10 coefficients or the
    implementation's intermediate expressions.
    """
    eta, delta, chi_s, chi_a = sp.symbols("eta delta chi_s chi_a")
    x1, x2 = (1 + delta) / 2, (1 - delta) / 2
    chi1, chi2 = chi_s + chi_a, chi_s - chi_a

    # Aligned-spin projections in BFH Eq. (4.26)-(4.27a).
    s_c = sp.expand(x1**2 * chi1 + x2**2 * chi2)
    sigma_c = sp.expand(x2 * chi2 - x1 * chi1)
    assert sp.expand(s_c - ((1 + delta**2) * chi_s / 2 + delta * chi_a)) == 0
    assert sp.expand(sigma_c + delta * chi_s + chi_a) == 0

    # For physical masses, delta^2=1-4*eta.  Substitution gives the form
    # multiplying i*v^4 in the clean H21 transcription.
    physical_s_c = (1 - 2 * eta) * chi_s + delta * chi_a
    assert sp.simplify(
        (s_c - physical_s_c).subs(delta**2, 1 - 4 * eta)
    ) == 0
    bfh_spin = sp.expand(
        -sp.Rational(43, 21) * delta * s_c
        + (-79 + 139 * eta) / sp.Integer(42) * sigma_c
    )
    expected = sp.expand(
        ((205 * eta - 7) * chi_a + (33 * eta - 7) * delta * chi_s) / 42
    )
    relation = sp.Poly(delta**2 - (1 - 4 * eta), delta)
    remainder = sp.Poly(bfh_spin - expected, delta).rem(relation).as_expr()
    assert sp.simplify(remainder) == 0


def test_exchange_symmetry_and_batch_shapes():
    v = torch.tensor((0.03, 0.07, 0.11), dtype=torch.float64)
    original = spa_hhat_21(v, _ETA, _DELTA, _CHI_S, _CHI_A)
    exchanged = spa_hhat_21(v, _ETA, -_DELTA, _CHI_S, -_CHI_A)
    torch.testing.assert_close(original, -exchanged, rtol=2e-14, atol=2e-16)

    ratio = taylor_t4_ratio(v, _ETA, _DELTA, _CHI_S, _CHI_A)
    exchanged_ratio = taylor_t4_ratio(v, _ETA, -_DELTA, _CHI_S, -_CHI_A)
    torch.testing.assert_close(ratio, exchanged_ratio, rtol=2e-14, atol=2e-16)
    assert taylor_t4_xdot(v**2, _ETA, _DELTA, _CHI_S, _CHI_A).shape == v.shape


def test_phase_sensitive_gradients_and_frequency_batch():
    v = torch.tensor(0.117, dtype=torch.float64)
    eta = torch.tensor(_ETA, dtype=torch.float64, requires_grad=True)
    chi_s = torch.tensor(_CHI_S, dtype=torch.float64, requires_grad=True)
    chi_a = torch.tensor(_CHI_A, dtype=torch.float64, requires_grad=True)
    value = spa_hhat_21(v, eta, _DELTA, chi_s, chi_a)
    real_grad = torch.autograd.grad(value.real, (eta, chi_s, chi_a), retain_graph=True)
    imag_grad = torch.autograd.grad(value.imag, (eta, chi_s, chi_a))
    for gradients in (real_grad, imag_grad):
        assert all(torch.isfinite(g) for g in gradients)
        assert float(torch.linalg.vector_norm(torch.stack(gradients))) > 1e-12

    frequency = torch.tensor((20.0, 40.0, 80.0), dtype=torch.float64, requires_grad=True)
    normalized = normalized_spa_amplitude_21(
        frequency,
        11.0 * 4.925490947641267e-6,
        6.0,
        5.0,
        0.08,
        0.03,
    )
    assert normalized.shape == frequency.shape
    assert torch.isfinite(normalized).all()
    frequency_grad = torch.autograd.grad(normalized.sum(), frequency)[0]
    assert torch.isfinite(frequency_grad).all()


def test_physical_prefactor_keeps_absolute_m2_over_d_scale():
    # Keeping v fixed while doubling M doubles the Fourier prefactor fourfold
    # when D is fixed.  This checks the raw physical normalization directly.
    m_seconds = 4.925490947641267e-6
    distance_seconds = 500e6 * 3.0856775814913673e16 / 299792458.0
    frequency = 30.0
    amplitude = physical_spa_amplitude_21(
        frequency,
        10.0 * m_seconds,
        6.0,
        4.0,
        0.08,
        0.03,
        distance_seconds,
    )
    doubled = physical_spa_amplitude_21(
        frequency / 2.0,
        20.0 * m_seconds,
        12.0,
        8.0,
        0.08,
        0.03,
        distance_seconds,
    )
    torch.testing.assert_close(
        doubled / amplitude,
        torch.tensor(4.0, dtype=torch.float64),
        rtol=2e-14,
        atol=2e-15,
    )
    assert float(amplitude) > 0.0


def test_branch_domain_and_q_cut():
    common = dict(
        frequency=20.0,
        total_mass_seconds=10.0 * 4.925490947641267e-6,
        m1=3.0,
        m2=2.0,
        chi1=0.08,
        chi2=0.03,
    )
    assert torch.isfinite(normalized_spa_amplitude_21(**common))
    with pytest.raises(ValueError, match="q < 40"):
        normalized_spa_amplitude_21(**{**common, "m1": 80.0, "m2": 2.0})
    with pytest.raises(ValueError, match="positive"):
        normalized_spa_amplitude_21(**{**common, "frequency": 0.0})
    with pytest.raises(ValueError, match="positive"):
        spa_hhat_21(0.0, _ETA, _DELTA, _CHI_S, _CHI_A)
    with pytest.raises(ValueError, match="non-positive"):
        # Independent coefficient continuation is allowed at this low-level
        # API; this deliberately exercises its explicit R>0 guard.
        spa_hhat_21(1.0, 0.05, 0.0, 0.5, -1.0)
