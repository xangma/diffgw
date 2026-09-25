"""Optional binary-oracle check of smooth IMRPhenomD intrinsic derivatives.

LALSimulation supplies waveform values through its public FD API. A fourth-order
stencil differentiates those values independently of the Torch/JAX autodiff
implementations and the paper-based scalar mathematical reference.
"""

from functools import lru_cache

import numpy as np
import pytest


CASES = ((36.0, 24.0, 0.2, -0.3),
         (45.0, 15.0, 0.45, -0.25),
         (36.0, 9.0, -0.6, -0.4))
FREQUENCIES = np.array((40.0, 120.0, 280.0))
DISTANCE_MPC = 400.0
INCLINATION = 0.7
F_REF = 20.0


@lru_cache(maxsize=None)
def _lal_polarizations(case):
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    m1, m2, chi1, chi2 = case
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0.0, 0.0, chi1, 0.0, 0.0, chi2,
        DISTANCE_MPC * 1e6 * lal.PC_SI, INCLINATION,
        0.0, 0.0, 0.0, 0.0,
        0.25, F_REF, 300.25, F_REF, lal.CreateDict(), lalsim.IMRPhenomD,
    )
    indices = (FREQUENCIES / 0.25).astype(int)
    return np.stack((np.asarray(hp.data.data)[indices],
                     np.asarray(hc.data.data)[indices]))


def _perturb(case, parameter, offset):
    m1, m2, chi1, chi2 = case
    if parameter == "mass_ratio":
        total = m1 + m2
        q = m1 / m2 + offset
        return (total * q / (1.0 + q), total / (1.0 + q), chi1, chi2)
    if parameter == "chi1":
        return m1, m2, chi1 + offset, chi2
    if parameter == "chi2":
        return m1, m2, chi1, chi2 + offset
    raise ValueError(parameter)


def _lal_derivative(case, parameter, step):
    values = {n: _lal_polarizations(_perturb(case, parameter, n * step))
              for n in (-2, -1, 1, 2)}
    return (values[-2] - 8 * values[-1] + 8 * values[1] - values[2]) / (12 * step)


def _frontend_polarizations(backend, case, parameter):
    m1, m2, chi1, chi2 = case
    q0 = m1 / m2
    total = m1 + m2
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD

        def evaluate(value):
            m1p, m2p = ((total * value / (1 + value), total / (1 + value))
                         if parameter == "mass_ratio" else (m1, m2))
            hp, hc = IMRPhenomD()(
                torch.tensor(FREQUENCIES, dtype=torch.float64), m1p, m2p,
                value if parameter == "chi1" else chi1,
                value if parameter == "chi2" else chi2,
                distance=DISTANCE_MPC, inclination=INCLINATION,
                phic=0.0, f_ref=F_REF,
            )
            return torch.stack((hp, hc))

        x = torch.tensor(q0 if parameter == "mass_ratio" else
                         chi1 if parameter == "chi1" else chi2,
                         dtype=torch.float64, requires_grad=True)
        wave = evaluate(x).detach().numpy()
        jac = torch.autograd.functional.jacobian(
            lambda v: torch.view_as_real(evaluate(v)), x)
        return wave, torch.view_as_complex(jac.contiguous()).detach().numpy()

    jax = pytest.importorskip("jax")
    previous_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD

        def evaluate(value):
            m1p, m2p = ((total * value / (1 + value), total / (1 + value))
                         if parameter == "mass_ratio" else (m1, m2))
            hp, hc = IMRPhenomD.evaluate(
                f=jnp.asarray(FREQUENCIES), mass1=m1p, mass2=m2p,
                chi1=value if parameter == "chi1" else chi1,
                chi2=value if parameter == "chi2" else chi2,
                distance=DISTANCE_MPC, inclination=INCLINATION,
                phic=0.0, f_ref=F_REF,
            )
            return jnp.stack((hp, hc))

        x = jnp.asarray(q0 if parameter == "mass_ratio" else
                        chi1 if parameter == "chi1" else chi2)
        wave = np.asarray(evaluate(x))
        jac = np.asarray(jax.jacfwd(lambda v: jnp.stack(
            (evaluate(v).real, evaluate(v).imag)))(x))
        return wave, jac[0] + 1j * jac[1]
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("parameter,step", (("mass_ratio", 1e-3),
                                             ("chi1", 5e-4),
                                             ("chi2", 5e-4)))
def test_lal_intrinsic_derivatives(backend, case, parameter, step):
    """Check direct complex values and converged slopes without alignment."""
    expected_wave = _lal_polarizations(case)
    expected = _lal_derivative(case, parameter, step / 2)
    coarse = _lal_derivative(case, parameter, step)
    wave, actual = _frontend_polarizations(backend, case, parameter)

    wave_norm = np.linalg.norm(expected_wave)
    derivative_norm = np.linalg.norm(expected)
    assert wave_norm > 0 and derivative_norm > 0
    assert np.linalg.norm(wave - expected_wave) / wave_norm < 1e-9
    assert np.linalg.norm(coarse - expected) / derivative_norm < 1e-6
    assert np.linalg.norm(actual - expected) / derivative_norm < 1e-6
