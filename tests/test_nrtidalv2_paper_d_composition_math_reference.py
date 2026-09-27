"""Frozen spinning paper-D quotient across the printed merger/taper boundary."""

import math

import mpmath as mp
import numpy as np
import pytest

from diffgw_core.constants import MSUN_SEC
from diffgw_reference import (
    nrtidalv2_amplitude,
    nrtidalv2_phase,
    nrtidalv2_self_spin,
)
from test_nrtidalv2_phase_math_reference import require_module


# (name, m1, m2, chi1, chi2, Lambda1, Lambda2); masses are in solar masses.
CASES = (
    ("equal", 1.35, 1.35, 0.20, -0.12, 350.0, 350.0),
    ("unequal", 1.6, 1.2, 0.08, -0.12, 250.0, 800.0),
    ("exchanged", 1.2, 1.6, -0.12, 0.08, 800.0, 250.0),
)
FRACTIONS = (0.75, 0.99, 1.0, 1.025, 1.05, 1.1, 1.15, 1.195, 1.2, 1.25)
QUOTIENT_ATOL = 1e-10


def _merger(case):
    _, m1, m2, _, _, lam1, lam2 = case
    return float(nrtidalv2_amplitude.merger_frequency(
        (m1 + m2) * MSUN_SEC, (m1 - m2) / (m1 + m2), lam1, lam2))


def _paper_quotient(case, frequency, merger):
    _, m1, m2, chi1, chi2, lam1, lam2 = case
    m_sec = (m1 + m2) * MSUN_SEC
    with mp.workdps(80):
        x = (mp.pi * mp.mpf(str(m_sec)) * mp.mpf(str(frequency))) ** (mp.mpf(2) / 3)
        tidal = nrtidalv2_phase.evaluate(
            frequency, m_sec, (m1 - m2) / (m1 + m2), lam1, lam2)
        spin = nrtidalv2_self_spin.d_phase(
            x, str(m1), str(m2), str(chi1), str(chi2), str(lam1), str(lam2))
        window = nrtidalv2_amplitude.taper(frequency, merger)
        return complex(window * mp.exp(-mp.j * (tidal + spin)))


def _native_quotients(backend, case, frequencies):
    _, m1, m2, chi1, chi2, lam1, lam2 = case
    options = dict(
        mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
        distance=100.0, inclination=0.4, phic=0.0, tc=0.0,
        phasing_convention="coalescence",
    )
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD
        from torchwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidalv2

        f = torch.as_tensor(frequencies, dtype=torch.float64)
        bbh_plus, bbh_cross = IMRPhenomD()(f=f, **options)
        bns_plus, bns_cross = IMRPhenomD_NRTidalv2(
            amplitude_variant="paper_2019")(
                f=f, lambda1=lam1, lambda2=lam2,
                apply_merger_taper=True, **options)
    else:
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD
        from jaxwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidalv2

        f = jnp.asarray(frequencies, dtype=jnp.float64)
        bbh_plus, bbh_cross = IMRPhenomD.evaluate(f=f, **options)
        bns_plus, bns_cross = IMRPhenomD_NRTidalv2.evaluate(
            f=f, lambda1=lam1, lambda2=lam2,
            apply_merger_taper=True, amplitude_variant="paper_2019", **options)
    assert np.all(np.abs(np.asarray(bbh_plus)) > 0)
    assert np.all(np.abs(np.asarray(bbh_cross)) > 0)
    return (np.asarray(bns_plus / bbh_plus),
            np.asarray(bns_cross / bbh_cross))


def test_frozen_spinning_boundary_is_finite_and_body_exchange_symmetric():
    assert math.isclose(_merger(CASES[1]), _merger(CASES[2]), rel_tol=1e-14)
    for case in CASES:
        merger = _merger(case)
        assert 1500.0 < merger < 2300.0
        assert 1.2 * merger > merger
    merger = _merger(CASES[1])
    for fraction in FRACTIONS:
        f = fraction * merger
        assert abs(_paper_quotient(CASES[1], f, merger)
                   - _paper_quotient(CASES[2], f, merger)) <= 1e-12


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_spinning_paper_d_complex_quotient_through_taper(backend, case):
    merger = _merger(case)
    frequencies = np.array([100.0, 800.0, *(fraction * merger for fraction in FRACTIONS)])
    expected = np.array([_paper_quotient(case, f, merger) for f in frequencies])
    plus, cross = _native_quotients(backend, case, frequencies)
    assert np.max(np.abs(plus - expected)) <= QUOTIENT_ATOL
    assert np.max(np.abs(cross - expected)) <= QUOTIENT_ATOL
    assert np.max(np.abs(plus - cross)) <= QUOTIENT_ATOL
    # A float-rounded scalar merger can straddle the native 1.2*f_merger
    # comparison by an ulp; the strictly post-endpoint bin is exact zero.
    assert abs(plus[-2]) <= QUOTIENT_ATOL
    assert abs(cross[-2]) <= QUOTIENT_ATOL
    assert plus[-1] == 0.0
    assert cross[-1] == 0.0
