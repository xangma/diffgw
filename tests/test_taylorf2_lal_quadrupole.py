"""Optional, same-model binary-oracle check for the TaylorF2 quadrupole slice.

The pinned campaign uses LALSuite 7.21. LAL accepts dQuadMon=C_Q-1 and its
TaylorF2 spin-order 7 includes the 2PN and 3PN non-Kerr terms. Its spin-order
8 is unsupported, so the 3.5PN quadrupole terms are checked by the separate
paper-based mathematical-reference gate, not this binary-oracle test.
"""

import math

import numpy as np
import pytest


CASES = (
    ("Kerr", 1.7, 1.2, 0.4, -0.2, 1.0, 1.0),
    ("BH-NS", 10.0, 1.4, 0.3, 0.25, 1.0, 5.0),
    ("NS-NS", 1.7, 1.2, 0.4, -0.2, 5.0, 2.5),
)


def _lal_params(lalsim, lal, cq1, cq2):
    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(params, 7)
    lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(params, 0)
    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(params, 7)
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(params, 0)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(params, cq1 - 1.0)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(params, cq2 - 1.0)
    return params


def _native_polarizations(backend, frequencies, m1, m2, chi1, chi2, cq1, cq2):
    options = dict(distance=100.0, inclination=0.3, phic=math.pi / 2,
                   quad_monopole1=cq1, quad_monopole2=cq2,
                   spin_order=7, tidal_order=0,
                   phasing_convention="coalescence", f_isco_cutoff=False)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.taylorf2 import TaylorF2
        waves = TaylorF2()(torch.tensor(frequencies, dtype=torch.float64),
                           m1, m2, chi1, chi2, **options)
        return tuple(wave.detach().cpu().numpy() for wave in waves)

    jax = pytest.importorskip("jax")
    previous_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp
        from jaxwave.approximants.taylorf2 import TaylorF2
        waves = TaylorF2.evaluate(jnp.asarray(frequencies),
                                  m1, m2, chi1, chi2, **options)
        return tuple(np.asarray(wave) for wave in waves)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("name,m1,m2,chi1,chi2,cq1,cq2", CASES)
def test_taylorf2_lal_quadrupole_coefficients_and_polarizations(
    backend, name, m1, m2, chi1, chi2, cq1, cq2
):
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    pytest.importorskip("sympy")
    from diffgw_reference.taylorf2_spin_quadrupole import quadrupole_corrections
    params = _lal_params(lalsim, lal, cq1, cq2)
    kerr_params = _lal_params(lalsim, lal, 1.0, 1.0)

    args = (m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, chi1, chi2)
    lal_quad = lalsim.SimInspiralTaylorF2AlignedPhasing(*args, params)
    lal_kerr = lalsim.SimInspiralTaylorF2AlignedPhasing(*args, kerr_params)
    eta = m1 * m2 / (m1 + m2) ** 2
    lal_delta = (lal_quad.v - lal_kerr.v) / (3.0 / (128.0 * eta))
    expected = quadrupole_corrections((m1 - m2) / (m1 + m2), chi1, chi2, cq1, cq2)
    for term, index in (("Q4", 4), ("Q6", 6)):
        assert math.isclose(lal_delta[index], float(expected[term]), abs_tol=1e-10), (name, term)

    frequencies = np.arange(30.0, 300.0 + 1e-12, 0.25)
    lal_grid = lal.CreateREAL8Vector(len(frequencies))
    lal_grid.data[:] = frequencies
    lal_plus, lal_cross = lalsim.SimInspiralChooseFDWaveformSequence(
        0.0, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0.0, 0.0, chi1, 0.0, 0.0, chi2,
        0.0, 100.0 * 1e6 * lal.PC_SI, 0.3,
        params, lalsim.TaylorF2, lal_grid,
    )
    # LAL's phiRef=0 carrier differs by a fixed pi from DiffGW's phic=0
    # coalescence carrier. phic=pi/2 removes that convention difference.
    native_plus, native_cross = _native_polarizations(
        backend, frequencies, m1, m2, chi1, chi2, cq1, cq2)
    for lal_pol, native_pol in ((lal_plus, native_plus), (lal_cross, native_cross)):
        reference = np.asarray(lal_pol.data.data)
        relative_l2 = np.linalg.norm(reference - native_pol) / np.linalg.norm(reference)
        phase_error = np.max(np.abs(np.unwrap(np.angle(reference / native_pol))))
        assert relative_l2 < 1e-9, (name, backend, relative_l2)
        assert phase_error < 1e-8, (name, backend, phase_error)
