"""Frozen aligned-frame Pv2 additive finite-size composition, before twist-up."""

import math

import mpmath as mp
import numpy as np
import pytest

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from diffgw_reference import nrtidalv2_amplitude, nrtidalv2_pv2_terms, nrtidalv2_self_spin
from test_nrtidalv2_paper_d_composition_math_reference import CASES, _native_quotients
from test_nrtidalv2_phase_math_reference import require_module


FRACTIONS = (0.75, 0.99, 1.0, 1.025, 1.05, 1.10, 1.15, 1.25)
DISTANCE_MPC = 100.0
RELATIVE_COMPLEX_LIMIT = 1e-10


def _reference(case, frequencies):
    _, m1, m2, chi1, chi2, lam1, lam2 = case
    m_sec = (m1 + m2) * MSUN_SEC
    d_sec = DISTANCE_MPC * MPC_SI / C_SI
    delta = (m1 - m2) / (m1 + m2)
    merger = nrtidalv2_amplitude.merger_frequency(m_sec, delta, lam1, lam2)
    terms, scales = [], []
    for frequency in frequencies:
        terms.append(nrtidalv2_pv2_terms.additive_tidal_term(
            frequency, m1, m2, chi1, chi2, lam1, lam2,
            m_sec, d_sec, merger))
        amplitude = nrtidalv2_amplitude.amplitude(
            frequency, m_sec, d_sec, delta, lam1, lam2)
        window = nrtidalv2_amplitude.taper(frequency, merger)
        scales.append(float(abs(amplitude * window)))
    return np.asarray(terms), np.asarray(scales), float(merger)


def _native(case, frequencies, backend):
    _, m1, m2, chi1, chi2, lam1, lam2 = case
    q_plus, q_cross = _native_quotients(backend, case, frequencies, taper=False)
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd_nrtidal import (
            _bns_merger_frequency, _nrtidal_amplitude,
            _nrtidal_self_spin_components, _planck_taper_window,
        )
        from torchwave.conversions import tidal_combinations
        def array(value):
            return torch.as_tensor(value, dtype=torch.float64)
        power = torch.exp
    else:
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd_nrtidal import (
            _bns_merger_frequency, _nrtidal_amplitude,
            _nrtidal_self_spin_components, _planck_taper_window,
        )
        from jaxwave.conversions import tidal_combinations
        def array(value):
            return jnp.asarray(value, dtype=jnp.float64)
        power = jnp.exp

    f = array(frequencies)
    mass1, mass2 = array(m1), array(m2)
    lambda1, lambda2 = array(lam1), array(lam2)
    x = (math.pi * (mass1 + mass2) * MSUN_SEC * f) ** (2.0 / 3.0)
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    lambda_tilde, _ = tidal_combinations(mass1, mass2, lambda1, lambda2)
    d_sec = array(DISTANCE_MPC * MPC_SI / C_SI)
    amplitude = _nrtidal_amplitude(
        x, (mass1 + mass2) * MSUN_SEC, d_sec, eta, lambda_tilde)
    _, s35 = _nrtidal_self_spin_components(
        x, mass1, mass2, array(chi1), array(chi2), lambda1, lambda2,
        fit="dietrich_2019")
    merger = _bns_merger_frequency(mass1, mass2, lambda1, lambda2)
    window = _planck_taper_window(f, merger, 1.2 * merger)
    factor = np.asarray(amplitude * window * power(-1j * s35))
    return factor * q_plus, factor * q_cross


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_pv2_aligned_additive_terms_through_taper(backend, case):
    _, _, _, _, _, lam1, lam2 = case
    merger = float(nrtidalv2_amplitude.merger_frequency(
        (case[1] + case[2]) * MSUN_SEC,
        (case[1] - case[2]) / (case[1] + case[2]), lam1, lam2))
    frequencies = np.array([100.0, 800.0,
                            *(fraction * merger for fraction in FRACTIONS)])
    expected, scales, _ = _reference(case, frequencies)
    plus, cross = _native(case, frequencies, backend)
    live = scales > 0
    assert np.max(np.abs(plus[live] - expected[live]) / scales[live]) <= RELATIVE_COMPLEX_LIMIT
    assert np.max(np.abs(cross[live] - expected[live]) / scales[live]) <= RELATIVE_COMPLEX_LIMIT
    assert np.max(np.abs(plus[live] - cross[live]) / scales[live]) <= RELATIVE_COMPLEX_LIMIT
    assert plus[-1] == 0j
    assert cross[-1] == 0j


def test_35pn_phase_is_observable_in_frozen_equal_mass_case():
    case = CASES[0]
    _, m1, m2, chi1, chi2, lam1, lam2 = case
    with mp.workdps(80):
        x = (mp.pi * mp.mpf(str((m1 + m2) * MSUN_SEC)) * 800) ** (mp.mpf(2) / 3)
        s35 = nrtidalv2_self_spin.components(
            x, str(m1), str(m2), str(chi1), str(chi2), str(lam1), str(lam2))[2]
        assert abs(mp.exp(-mp.j * s35) - 1) > mp.mpf("1e-4")
