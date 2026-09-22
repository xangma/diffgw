"""Cross-framework parity tests: torchwave vs jaxwave for SEOBNR / EOB engine.

Validates:
1. Nonspinning3PNHamiltonian and SEOBNRv4Hamiltonian conservative dynamics and potentials.
2. Prescribed constant torque to isolate generic solver mechanics.
3. find_quasicircular_initial_conditions solve.
4. EOBTrajectoryIntegrator.integrate (RK4 trajectory integration).
5. _quadrupole_fd Fourier projection.
6. JIT compilation of core EOB components under jax.jit.
"""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

# Enable float64 precision in JAX
jax.config.update("jax_enable_x64", True)


from torchwave.approximants.eob.hamiltonian import Nonspinning3PNHamiltonian as TorchHam
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as TorchSEOBNRv4Ham
from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions as torch_initial_conditions,
)
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.waveform import _quadrupole_fd as torch_quadrupole_fd

from jaxwave.approximants.eob.hamiltonian import Nonspinning3PNHamiltonian as JaxHam
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxSEOBNRv4Ham
from jaxwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions as jax_initial_conditions,
)
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.waveform import _quadrupole_fd as jax_quadrupole_fd


class TorchRR:
    """Prescribed negative torque used only to test solver mechanics."""
    def __init__(self, *unused):
        pass

    def radiation_reaction_forces(self, omega, pr_star, pphi, **kwargs):
        return torch.zeros_like(omega), torch.full_like(omega, -1e-6)


class JaxRR:
    def __init__(self, *unused):
        pass

    def radiation_reaction_forces(self, omega, pr_star, pphi, **kwargs):
        return jnp.zeros_like(omega), jnp.full_like(omega, -1e-6)


@pytest.mark.parametrize("nu", [0.25, 0.22, 0.16])
def test_eob_hamiltonian_parity(nu):
    """Test conservative EOB Hamiltonian potentials and equations of motion."""
    t_ham = TorchHam(
        torch.tensor(nu, dtype=torch.float64),
    )
    j_ham = JaxHam(
        jnp.asarray(nu, dtype=jnp.float64),
    )

    r_vals = np.array([6.0, 10.0, 15.0])
    pr_vals = np.array([-0.02, -0.005, 0.0])
    pphi_vals = np.array([3.1, 3.4, 3.8])

    t_r = torch.tensor(r_vals, dtype=torch.float64)
    t_pr = torch.tensor(pr_vals, dtype=torch.float64)
    t_pphi = torch.tensor(pphi_vals, dtype=torch.float64)

    j_r = jnp.asarray(r_vals, dtype=jnp.float64)
    j_pr = jnp.asarray(pr_vals, dtype=jnp.float64)
    j_pphi = jnp.asarray(pphi_vals, dtype=jnp.float64)

    # 1. Effective Hamiltonian
    t_eff = t_ham.effective_hamiltonian(t_r, t_pr, t_pphi).numpy()
    j_eff = np.asarray(j_ham.effective_hamiltonian(j_r, j_pr, j_pphi))
    np.testing.assert_allclose(t_eff, j_eff, rtol=1e-14, atol=1e-15)

    # 2. Real Hamiltonian
    t_real = t_ham.real_hamiltonian(t_r, t_pr, t_pphi).numpy()
    j_real = np.asarray(j_ham.real_hamiltonian(j_r, j_pr, j_pphi))
    np.testing.assert_allclose(t_real, j_real, rtol=1e-14, atol=1e-15)

    # 3. Equations of motion (dr/dt, Omega, dpr_star/dt)
    t_dr, t_om, t_dpr = [x.numpy() for x in t_ham.equations_of_motion_conservative(t_r, t_pr, t_pphi)]
    j_dr, j_om, j_dpr = [np.asarray(x) for x in j_ham.equations_of_motion_conservative(j_r, j_pr, j_pphi)]

    np.testing.assert_allclose(t_dr, j_dr, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(t_om, j_om, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(t_dpr, j_dpr, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("nu, chi1, chi2", [
    (0.25, 0.0, 0.0),
    (0.22, 0.3, -0.2),
    (0.18, -0.1, 0.4),
])
def test_seobnrv4_hamiltonian_parity(nu, chi1, chi2):
    """Test SEOBNRv4 Hamiltonian potentials and equations of motion."""
    t_ham = TorchSEOBNRv4Ham(
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
    )
    j_ham = JaxSEOBNRv4Ham(
        jnp.asarray(nu, dtype=jnp.float64),
        jnp.asarray(chi1, dtype=jnp.float64),
        jnp.asarray(chi2, dtype=jnp.float64),
    )

    r_vals = np.array([5.0, 8.0, 12.0])
    pr_vals = np.array([-0.01, -0.002, 0.0])
    pphi_vals = np.array([3.0, 3.3, 3.7])

    t_r = torch.tensor(r_vals, dtype=torch.float64)
    t_pr = torch.tensor(pr_vals, dtype=torch.float64)
    t_pphi = torch.tensor(pphi_vals, dtype=torch.float64)

    j_r = jnp.asarray(r_vals, dtype=jnp.float64)
    j_pr = jnp.asarray(pr_vals, dtype=jnp.float64)
    j_pphi = jnp.asarray(pphi_vals, dtype=jnp.float64)

    t_eff = t_ham.effective_hamiltonian(t_r, t_pr, t_pphi).numpy()
    j_eff = np.asarray(j_ham.effective_hamiltonian(j_r, j_pr, j_pphi))
    np.testing.assert_allclose(t_eff, j_eff, rtol=1e-14, atol=1e-15)

    t_real = t_ham.real_hamiltonian(t_r, t_pr, t_pphi).numpy()
    j_real = np.asarray(j_ham.real_hamiltonian(j_r, j_pr, j_pphi))
    np.testing.assert_allclose(t_real, j_real, rtol=1e-14, atol=1e-15)

    t_dr, t_om, t_dpr = [x.numpy() for x in t_ham.equations_of_motion_conservative(t_r, t_pr, t_pphi)]
    j_dr, j_om, j_dpr = [np.asarray(x) for x in j_ham.equations_of_motion_conservative(j_r, j_pr, j_pphi)]

    np.testing.assert_allclose(t_dr, j_dr, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(t_om, j_om, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(t_dpr, j_dpr, rtol=1e-14, atol=1e-15)


def test_radiation_reaction_parity():
    """Prescribed torques agree between array backends."""
    nu, chi1, chi2 = 0.24, 0.1, -0.2
    t_ham = TorchHam(
        torch.tensor(nu, dtype=torch.float64),
    )
    j_ham = JaxHam(
        jnp.asarray(nu, dtype=jnp.float64),
    )

    t_rr = TorchRR(
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
    )
    j_rr = JaxRR(
        jnp.asarray(nu, dtype=jnp.float64),
        jnp.asarray(chi1, dtype=jnp.float64),
        jnp.asarray(chi2, dtype=jnp.float64),
    )

    omega = 0.025
    pr = -0.005
    pphi = 3.4
    r = 9.5

    t_fr, t_fphi = t_rr.radiation_reaction_forces(
        torch.tensor(omega, dtype=torch.float64),
        torch.tensor(pr, dtype=torch.float64),
        torch.tensor(pphi, dtype=torch.float64),
        r=torch.tensor(r, dtype=torch.float64),
        hamiltonian=t_ham,
    )
    j_fr, j_fphi = j_rr.radiation_reaction_forces(
        jnp.asarray(omega, dtype=jnp.float64),
        jnp.asarray(pr, dtype=jnp.float64),
        jnp.asarray(pphi, dtype=jnp.float64),
        r=jnp.asarray(r, dtype=jnp.float64),
        hamiltonian=j_ham,
    )

    np.testing.assert_allclose(t_fr.numpy(), np.asarray(j_fr), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(t_fphi.numpy(), np.asarray(j_fphi), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("omega0, nu", [(0.015, 0.25), (0.02, 0.22), (0.018, 0.18)])
def test_quasicircular_initial_conditions_parity(omega0, nu):
    chi1 = chi2 = 0.0
    """Test 2x2 Newton-Raphson quasicircular initial conditions solver."""
    t_ham = TorchHam(
        torch.tensor(nu, dtype=torch.float64),
    )
    t_rr = TorchRR(
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
    )
    t_r0, t_pr0, t_pphi0 = torch_initial_conditions(
        t_ham, t_rr, torch.tensor(omega0, dtype=torch.float64),
    )

    j_ham = JaxHam(
        jnp.asarray(nu, dtype=jnp.float64),
    )
    j_rr = JaxRR(
        jnp.asarray(nu, dtype=jnp.float64),
        jnp.asarray(chi1, dtype=jnp.float64),
        jnp.asarray(chi2, dtype=jnp.float64),
    )
    j_r0, j_pr0, j_pphi0 = jax_initial_conditions(
        j_ham, j_rr, jnp.asarray(omega0, dtype=jnp.float64),
    )

    np.testing.assert_allclose(t_r0.numpy(), np.asarray(j_r0), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(t_pr0.numpy(), np.asarray(j_pr0), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(t_pphi0.numpy(), np.asarray(j_pphi0), rtol=1e-12, atol=1e-14)


def test_integrator_rk4_parity():
    """Test RK4 trajectory integration."""
    nu, chi1, chi2 = 0.25, 0.0, 0.0
    t_ham = TorchHam(
        torch.tensor(nu, dtype=torch.float64),
    )
    t_rr = TorchRR(
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
    )
    t_r0, t_pr0, t_pphi0 = torch_initial_conditions(
        t_ham, t_rr, torch.tensor(0.02, dtype=torch.float64),
    )
    t_state0 = torch.stack([t_r0, torch.tensor(0.0, dtype=torch.float64), t_pr0, t_pphi0])
    t_integ = TorchIntegrator(t_ham, t_rr)
    t_times, t_traj = t_integ.integrate(t_state0, dt=10.0, num_steps=50)

    j_ham = JaxHam(
        jnp.asarray(nu, dtype=jnp.float64),
    )
    j_rr = JaxRR(
        jnp.asarray(nu, dtype=jnp.float64),
        jnp.asarray(chi1, dtype=jnp.float64),
        jnp.asarray(chi2, dtype=jnp.float64),
    )
    j_r0, j_pr0, j_pphi0 = jax_initial_conditions(
        j_ham, j_rr, jnp.asarray(0.02, dtype=jnp.float64),
    )
    j_state0 = jnp.array([j_r0, 0.0, j_pr0, j_pphi0], dtype=jnp.float64)
    j_integ = JaxIntegrator(j_ham, j_rr)
    j_times, j_traj = j_integ.integrate(j_state0, dt=10.0, num_steps=50)

    np.testing.assert_allclose(t_times.numpy(), np.asarray(j_times), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(t_traj.numpy(), np.asarray(j_traj), rtol=1e-12, atol=1e-14)


def test_quadrupole_fd_parity():
    """Test Fourier-domain quadrupole projection parity."""
    freqs = np.linspace(20.0, 100.0, 32)
    t_len = 128
    t = np.linspace(0, 1, t_len)
    h22_data = (np.cos(2 * np.pi * 30 * t) + 1j * np.sin(2 * np.pi * 30 * t)) * np.exp(-t * 2)

    t_h22 = torch.tensor(h22_data, dtype=torch.complex128)
    t_f = torch.tensor(freqs, dtype=torch.float64)
    t_dt = torch.tensor(1.0 / t_len, dtype=torch.float64)
    t_inc = torch.tensor(0.5, dtype=torch.float64)

    j_h22 = jnp.asarray(h22_data, dtype=jnp.complex128)
    j_f = jnp.asarray(freqs, dtype=jnp.float64)
    j_dt = jnp.asarray(1.0 / t_len, dtype=jnp.float64)
    j_inc = jnp.asarray(0.5, dtype=jnp.float64)

    t_hp, t_hc = torch_quadrupole_fd(t_f, t_h22, t_dt, t_inc)
    j_hp, j_hc = jax_quadrupole_fd(j_f, j_h22, j_dt, j_inc)

    np.testing.assert_allclose(t_hp.numpy(), np.asarray(j_hp), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(t_hc.numpy(), np.asarray(j_hc), rtol=1e-14, atol=1e-15)


def test_eob_components_jit():
    """Verify that EOB components compile and execute cleanly under jax.jit."""
    # 1. Nonspinning 3PN Hamiltonian equations of motion
    ham = JaxHam(0.25)
    jit_eom = jax.jit(ham.equations_of_motion_conservative)
    dr, om, dpr = jit_eom(10.0, -0.01, 3.2)
    assert jnp.isfinite(dr) and jnp.isfinite(om) and jnp.isfinite(dpr)

    # 2. SEOBNRv4Hamiltonian equations of motion
    v4ham = JaxSEOBNRv4Ham(0.25, 0.2, -0.1)
    jit_v4eom = jax.jit(v4ham.equations_of_motion_conservative)
    v4dr, v4om, v4dpr = jit_v4eom(10.0, -0.01, 3.2)
    assert jnp.isfinite(v4dr) and jnp.isfinite(v4om) and jnp.isfinite(v4dpr)

    # 3. Radiation reaction forces
    rr = JaxRR(0.25, 0.2, -0.1)
    jit_rr = jax.jit(lambda o, pr, pp, r: rr.radiation_reaction_forces(o, pr, pp, r=r, hamiltonian=ham))
    fr, fphi = jit_rr(0.03, -0.01, 3.2, 10.0)
    assert jnp.isfinite(fr) and jnp.isfinite(fphi)

    # 4. Trajectory integrator (RK4 scan)
    integ = JaxIntegrator(ham, rr)
    r0, pr0, pp0 = jax_initial_conditions(ham, rr, 0.02)
    state0 = jnp.array([r0, 0.0, pr0, pp0])
    jit_integ = jax.jit(lambda s: integ.integrate(s, dt=0.5, num_steps=20))
    times, traj = jit_integ(state0)
    assert traj.shape == (21, 4)
    assert jnp.all(jnp.isfinite(traj))

    # 5. Fourier domain quadrupole projection
    jit_qfd = jax.jit(jax_quadrupole_fd)
    f = jnp.linspace(20.0, 200.0, 16)
    h22 = jnp.ones(32, dtype=jnp.complex128)
    hp, hc = jit_qfd(f, h22, 1e-4, 0.5)
    assert hp.shape == (16,)
    assert jnp.all(jnp.isfinite(hp))


def test_batched_nonspinning_initial_data_preserves_supplied_torque():
    """A generic solver must preserve model choices without rebuilding constructors."""
    nu = jnp.asarray([.16, .24], dtype=jnp.float64)
    frequency = jnp.asarray(.017, dtype=jnp.float64)
    actual = jax_initial_conditions(JaxHam(nu), JaxRR(), frequency)
    rows = [jax_initial_conditions(JaxHam(eta), JaxRR(), frequency) for eta in nu]
    for component, reference in zip(actual, zip(*rows)):
        np.testing.assert_allclose(component, jnp.stack(reference), rtol=2e-13, atol=2e-14)
