"""Paper-math gate for the nonspinning NRTidalv2 amplitude and taper."""

import math

import numpy as np
import pytest

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from diffgw_reference import nrtidalv2_amplitude as ref
from diffgw_reference.nrtidalv2_phase import lambda_tilde_expr
from test_nrtidalv2_phase_math_reference import CASES, require_module


DISTANCE_MPC = 100.0
INCLINATION = 0.4
AMP_FREQUENCIES = np.array([100.0, 300.0, 800.0, 1200.0])
TAPER_FRACTIONS = np.array([0.90, 1.0, 1.025, 1.05,
                            1.1, 1.15, 1.195, 1.2, 1.25])
MERGER_REL_MAX = 1e-11
TAPER_ABS_MAX = 1e-10


def paper_parameters(case):
    _, m1, m2, lam1, lam2 = case
    total = m1 + m2
    return total*MSUN_SEC, (m1-m2)/total, lam1, lam2


def paper_amplitude(case, f):
    m_sec, delta, lam1, lam2 = paper_parameters(case)
    return float(ref.amplitude(f, m_sec, DISTANCE_MPC*MPC_SI/C_SI,
                               delta, lam1, lam2))


def paper_merger(case):
    m_sec, delta, lam1, lam2 = paper_parameters(case)
    return float(ref.merger_frequency(m_sec, delta, lam1, lam2))


def native(case, frequencies, backend, *, taper, amplitude_variant="binary_compatible"):
    _, m1, m2, lam1, lam2 = case
    options = dict(mass1=m1, mass2=m2, chi1=0.0, chi2=0.0,
                   distance=DISTANCE_MPC, inclination=INCLINATION,
                   phic=0.0, tc=0.0, f_ref=30.0,
                   phasing_convention="reference_frequency")
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD
        from torchwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidalv2
        f = torch.as_tensor(frequencies, dtype=torch.float64)
        bbh, _ = IMRPhenomD()(f=f, **options)
        bns, _ = IMRPhenomD_NRTidalv2()(f=f, lambda1=lam1,
                                         lambda2=lam2,
                                         apply_merger_taper=taper,
                                         amplitude_variant=amplitude_variant, **options)
    else:
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD
        from jaxwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidalv2
        f = jnp.asarray(frequencies, dtype=jnp.float64)
        bbh, _ = IMRPhenomD.evaluate(f=f, **options)
        bns, _ = IMRPhenomD_NRTidalv2.evaluate(
            f=f, lambda1=lam1, lambda2=lam2,
            apply_merger_taper=taper,
            amplitude_variant=amplitude_variant, **options)
    return np.asarray(bbh), np.asarray(bns)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_paper_exchange_zero_tide_and_pn_limit(case):
    if case[0] == "exchanged":
        assert math.isclose(paper_merger(case), paper_merger(CASES[1]),
                            rel_tol=1e-14)
        for f in AMP_FREQUENCIES:
            assert math.isclose(paper_amplitude(case, f),
                                paper_amplitude(CASES[1], f), rel_tol=1e-14)
    if case[0] == "zero_tide":
        assert all(paper_amplitude(case, f) == 0.0 for f in AMP_FREQUENCIES)
    else:
        m_sec, delta, lam1, lam2 = paper_parameters(case)
        f = 1e-3
        x = (math.pi*m_sec*f)**(2/3)
        eta = (1-delta**2)/4
        lt = float(lambda_tilde_expr(delta, lam1, lam2))
        leading = (-math.sqrt(5*math.pi*eta/24)
                   * 9*m_sec**2/(DISTANCE_MPC*MPC_SI/C_SI)
                   * (3*lt/16)*x**(13/4))
        assert math.isclose(paper_amplitude(case, f)/leading,
                            1 + 449*x/108, rel_tol=2e-7)


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_native_merger_and_window_against_paper(backend, case):
    _, m1, m2, lam1, lam2 = case
    f_merger = paper_merger(case)
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd_nrtidal import (
            _bns_merger_frequency, _planck_taper_window)
        array = lambda v: torch.as_tensor(v, dtype=torch.float64)
    else:
        require_module("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd_nrtidal import (
            _bns_merger_frequency, _planck_taper_window)
        array = lambda v: jnp.asarray(v, dtype=jnp.float64)
    merger = float(_bns_merger_frequency(*[array(v) for v in
                                           (m1, m2, lam1, lam2)]))
    f_taper = f_merger*TAPER_FRACTIONS
    window = np.asarray(_planck_taper_window(
        array(f_taper), array(f_merger), array(1.2*f_merger)))
    assert abs(merger/f_merger - 1) <= MERGER_REL_MAX
    expected_window = np.array([float(ref.taper(f, f_merger))
                                for f in f_taper])
    assert np.max(np.abs(window - expected_window)) <= TAPER_ABS_MAX


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_native_waveform_taper_against_paper(backend, case):
    f_merger = paper_merger(case)
    frequencies = f_merger*TAPER_FRACTIONS
    _, untapered = native(case, frequencies, backend, taper=False)
    _, tapered = native(case, frequencies, backend, taper=True)
    assert np.all(np.abs(untapered) > 0)
    actual = np.abs(tapered)/np.abs(untapered)
    expected = np.array([float(ref.taper(f, f_merger)) for f in frequencies])
    assert np.max(np.abs(actual-expected)) <= TAPER_ABS_MAX
