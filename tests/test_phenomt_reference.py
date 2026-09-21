"""Pinned oracle values from the MIT author's independent NumPy implementation.

phenomxpy commit 4121fafa0484f3db2a7352f4eea697faaaf29f65, pWFHM/pPhase/pAmp,
eta=60*6/66**2, chi1=0.9, chi2=0.1, mode=22, numba_ansatze=False.
No LAL installation is needed. The arbitrary phase integration constant is
removed at t=0; frequencies, times and mode amplitudes use total-mass units.
"""

import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
def test_author_reference_high_spin_unequal_masses(backend):
    if backend == "torchwave":
        import torch

        def array(x):
            return torch.tensor(x, dtype=torch.float64)
    else:
        import jax.numpy as jnp

        def array(x):
            return jnp.asarray(x, dtype=jnp.float64)

    model = importlib.import_module(backend).IMRPhenomT()
    amplitude = importlib.import_module(f"{backend}.approximants._phenomt_amplitude").amplitude
    state = model._coeff(*map(array, (60.0, 6.0, 0.9, 0.1)))
    times = array([-2000.0, -150.0, -100.0, 0.0, 50.0])
    omega, phase, amp_state = model._evaluate(times, state)
    amp = amplitude(times, amp_state, lambda t: model._evaluate(t, state)[0])
    omega_ref = [0.07038546522857381, 0.18587972328061905, 0.21361514075038326, 0.4432521717028724, 0.7158728715096001]
    amp_ref = [0.04967921450586258, 0.08848209053404885, 0.09554155331142025, 0.12372734727224127, 0.02621654229480636]
    phase_ref = np.array(
        [-360.50544144608165, -179.26671803764242, -169.33516889932505, -140.97061490500874, -109.41634670327691]
    )
    np.testing.assert_allclose(np.asarray(omega), omega_ref, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(np.asarray(amp), amp_ref, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(np.asarray(phase - phase[3]), phase_ref - phase_ref[3], rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize(
    "mass1,mass2,chi1,chi2",
    [
        (30.0, 25.0, 0.2, -0.1),
        (30.0, 30.0, 0.0, 0.0),
        (50.0, 10.0, 0.5, -0.4),
        (60.0, 6.0, 0.9, 0.1),
        (25.0, 25.0, -0.9, -0.9),
    ],
)
def test_lal_binary_reference_across_mass_ratios_and_spins(mass1, mass2, chi1, chi2):
    """Track the documented LAL residual without fitting to binary outputs."""
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    import torch
    from torchwave import IMRPhenomT

    dt = 1 / 2048
    hp, hc = lalsim.SimIMRPhenomT(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        chi1,
        chi2,
        100e6 * lal.PC_SI,
        0.5,
        dt,
        20.0,
        20.0,
        0.2,
        lal.CreateDict(),
    )
    times = torch.from_numpy(float(hp.epoch) + dt * np.arange(hp.data.length))
    plus, cross = IMRPhenomT().time_domain(
        times,
        mass1,
        mass2,
        chi1,
        chi2,
        distance=100.0,
        inclination=0.5,
        f_ref=20.0,
        phic=0.2,
    )
    actual = (plus - 1j * cross).numpy()
    reference = hp.data.data - 1j * hc.data.data
    # Direct complex-strain L2: no time or phase alignment. The normalized
    # overlap maximizes only constant phase and is not noise weighted.
    relative_l2 = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    overlap = abs(np.vdot(actual, reference)) / (np.linalg.norm(actual) * np.linalg.norm(reference))
    assert relative_l2 < 1e-3
    assert 1 - overlap < 1e-6
