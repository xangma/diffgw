"""Optional LAL binary-oracle check of the TaylorF2 electric-tidal slice.

This compares complete complex polarizations without phase, time, or amplitude
alignment for nonspinning binaries. The separate paper-based reference checks
tidal coefficients and their derivatives, including spinning compositions,
independently of both frontends and this LAL binary.
"""

import math

import numpy as np
import pytest


CASES = (
    ("equal", 1.4, 1.4, 0.0, 0.0, 400.0, 400.0),
    ("unequal", 1.7, 1.2, 0.0, 0.0, 800.0, 200.0),
)


def _lal_polarizations(frequencies, m1, m2, chi1, chi2, lambda1, lambda2, order):
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(params, 7)
    lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(params, 0)
    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(params, 7)
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(params, order)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params, lambda1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params, lambda2)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(params, 0.0)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(params, 0.0)
    grid = lal.CreateREAL8Vector(len(frequencies))
    grid.data[:] = frequencies
    hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
        0.0, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0.0, 0.0, chi1, 0.0, 0.0, chi2,
        0.0, 100.0 * 1e6 * lal.PC_SI, 0.3,
        params, lalsim.TaylorF2, grid,
    )
    return np.stack((np.asarray(hp.data.data), np.asarray(hc.data.data)))


def _frontend_polarizations(backend, frequencies, m1, m2, chi1, chi2,
                            lambda1, lambda2, order):
    options = dict(distance=100.0, inclination=0.3, phic=math.pi / 2,
                   lambda1=lambda1, lambda2=lambda2,
                   spin_order=7, tidal_order=order,
                   phasing_convention="coalescence", f_isco_cutoff=False)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.taylorf2 import TaylorF2

        hp, hc = TaylorF2()(torch.tensor(frequencies, dtype=torch.float64),
                            m1, m2, chi1, chi2, **options)
        return np.stack((hp.detach().cpu().numpy(), hc.detach().cpu().numpy()))

    jax = pytest.importorskip("jax")
    previous_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp
        from jaxwave.approximants.taylorf2 import TaylorF2

        hp, hc = TaylorF2.evaluate(jnp.asarray(frequencies),
                                   m1, m2, chi1, chi2, **options)
        return np.stack((np.asarray(hp), np.asarray(hc)))
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("order", (10, 12, 13))
@pytest.mark.parametrize("name,m1,m2,chi1,chi2,lambda1,lambda2", CASES)
def test_taylorf2_lal_tidal_polarizations(
    backend, order, name, m1, m2, chi1, chi2, lambda1, lambda2
):
    """Check direct plus/cross strain for each supported tidal PN selector."""
    frequencies = np.arange(30.0, 500.0 + 1e-12, 0.5)
    expected = _lal_polarizations(frequencies, m1, m2, chi1, chi2,
                                  lambda1, lambda2, order)
    actual = _frontend_polarizations(backend, frequencies, m1, m2, chi1, chi2,
                                     lambda1, lambda2, order)
    relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    phase_error = np.max(np.abs(np.unwrap(np.angle(expected / actual))))
    assert relative_l2 < 1e-9, (name, order, backend, relative_l2)
    assert phase_error < 1e-8, (name, order, backend, phase_error)
