"""Cross-framework parity test: torchwave vs jaxwave for TEOBResumS model."""

import math
import numpy as np
import pytest
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from torchwave.approximants.teobresums import (
    TEOBResumS as TorchTEOBResumS,
    TEOBResumSHamiltonian as TorchHam,
    TEOBResumSRadiationReaction as TorchRR,
    compute_tidal_merger_frequency as torch_f_mrg,
    _piecewise_linear_fourier as torch_plf,
    _teobresums_orbital_potential as torch_orb_pot,
)
from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions as torch_find_init,
)
from torchwave.waveforms import get_fd_waveform as torch_get_fd_waveform


def torch_cent_rad(r, nu, chi1, chi2, cq1=None, cq2=None, delta=None):
    ham = TorchHam(
        nu=nu, chi1=chi1, chi2=chi2,
        lambda1=torch.zeros_like(nu), lambda2=torch.zeros_like(nu),
        cq1=cq1, cq2=cq2, delta=delta,
    )
    return torch.sqrt(ham.centrifugal_radius_squared(r))


def torch_eff_ham(r, pr_star, pphi, nu, chi1, chi2, lambda1=0.0, lambda2=0.0, cq1=None, cq2=None, delta=None):
    from torchwave.conversions import as_tensor
    l1 = as_tensor(lambda1, dtype=nu.dtype, device=nu.device) if isinstance(nu, torch.Tensor) else as_tensor(lambda1)
    l2 = as_tensor(lambda2, dtype=nu.dtype, device=nu.device) if isinstance(nu, torch.Tensor) else as_tensor(lambda2)
    ham = TorchHam(
        nu=nu, chi1=chi1, chi2=chi2,
        lambda1=l1, lambda2=l2,
        cq1=cq1, cq2=cq2, delta=delta,
    )
    return ham.effective_hamiltonian(r, pr_star, pphi)

from jaxwave.approximants.teobresums import (
    TEOBResumS as JaxTEOBResumS,
    TEOBResumSHamiltonian as JaxHam,
    TEOBResumSRadiationReaction as JaxRR,
    compute_tidal_merger_frequency as jax_f_mrg,
    _piecewise_linear_fourier as jax_plf,
    _teobresums_orbital_potential as jax_orb_pot,
    _teobresums_centrifugal_radius as jax_cent_rad,
    _teobresums_effective_hamiltonian as jax_eff_ham,
    find_quasicircular_initial_conditions as jax_find_init,
)
from jaxwave.waveforms import get_fd_waveform as jax_get_fd_waveform


# =====================================================================
# 1. Piecewise Linear Fourier Transform Parity
# =====================================================================

def test_piecewise_linear_fourier_parity_1d():
    """Parity of 1D _piecewise_linear_fourier between JAX and PyTorch."""
    t_grid = np.linspace(0.0, 1.0, 120, dtype=np.float64)
    sig_np = np.sin(2.0 * np.pi * 12.0 * t_grid) + 0.3 * np.cos(2.0 * np.pi * 35.0 * t_grid)
    dt_val = t_grid[1] - t_grid[0]
    ev_val = 0.85
    freqs_np = np.linspace(0.0, 60.0, 30, dtype=np.float64)

    t_res = torch_plf(
        torch.tensor(sig_np, dtype=torch.float64),
        torch.tensor(ev_val, dtype=torch.float64),
        torch.tensor(dt_val, dtype=torch.float64),
        torch.tensor(freqs_np, dtype=torch.float64),
    ).numpy()

    j_res = np.asarray(jax_plf(
        jnp.array(sig_np, dtype=jnp.float64),
        jnp.array(ev_val, dtype=jnp.float64),
        jnp.array(dt_val, dtype=jnp.float64),
        jnp.array(freqs_np, dtype=jnp.float64),
    ))

    np.testing.assert_allclose(t_res, j_res, rtol=1e-14, atol=1e-15)


def test_piecewise_linear_fourier_parity_batched():
    """Parity of 2D batched _piecewise_linear_fourier between JAX and PyTorch."""
    t_grid = np.linspace(0.0, 0.5, 80, dtype=np.float64)
    sig1 = np.sin(2.0 * np.pi * 10.0 * t_grid)
    sig2 = np.cos(2.0 * np.pi * 20.0 * t_grid)
    sig_batched = np.stack([sig1, sig2], axis=1)  # (T, B)

    dt_val = t_grid[1] - t_grid[0]
    ev_batched = np.array([0.42, 0.48], dtype=np.float64)
    dt_batched = np.array([dt_val, dt_val], dtype=np.float64)
    freqs_np = np.linspace(5.0, 45.0, 15, dtype=np.float64)

    t_res = torch_plf(
        torch.tensor(sig_batched, dtype=torch.float64),
        torch.tensor(ev_batched, dtype=torch.float64),
        torch.tensor(dt_batched, dtype=torch.float64),
        torch.tensor(freqs_np, dtype=torch.float64),
    ).numpy()

    j_res = np.asarray(jax_plf(
        jnp.array(sig_batched, dtype=jnp.float64),
        jnp.array(ev_batched, dtype=jnp.float64),
        jnp.array(dt_batched, dtype=jnp.float64),
        jnp.array(freqs_np, dtype=jnp.float64),
    ))

    np.testing.assert_allclose(t_res, j_res, rtol=1e-14, atol=1e-15)


def test_piecewise_linear_fourier_autodiff_parity():
    """Parity of autograd/autodiff gradients of _piecewise_linear_fourier."""
    sig_np = np.sin(np.linspace(0.0, 5.0, 30, dtype=np.float64))
    ev_val = 0.7
    dt_val = 0.02
    freqs_np = np.array([10.0, 25.0], dtype=np.float64)

    # PyTorch gradient w.r.t event_time and dt_sec
    ev_t = torch.tensor(ev_val, dtype=torch.float64, requires_grad=True)
    dt_t = torch.tensor(dt_val, dtype=torch.float64, requires_grad=True)
    out_t = torch_plf(torch.tensor(sig_np, dtype=torch.float64), ev_t, dt_t, torch.tensor(freqs_np, dtype=torch.float64))
    loss_t = torch.sum(out_t.abs() ** 2)
    loss_t.backward()

    # JAX gradient w.r.t event_time and dt_sec
    def jax_loss(ev_j, dt_j):
        out_j = jax_plf(jnp.array(sig_np, dtype=jnp.float64), ev_j, dt_j, jnp.array(freqs_np, dtype=jnp.float64))
        return jnp.sum(jnp.abs(out_j) ** 2)

    g_ev_j = jax.grad(jax_loss, argnums=0)(jnp.array(ev_val, dtype=jnp.float64), jnp.array(dt_val, dtype=jnp.float64))
    g_dt_j = jax.grad(jax_loss, argnums=1)(jnp.array(ev_val, dtype=jnp.float64), jnp.array(dt_val, dtype=jnp.float64))

    np.testing.assert_allclose(ev_t.grad.item(), float(g_ev_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dt_t.grad.item(), float(g_dt_j), rtol=1e-12, atol=1e-14)


# =====================================================================
# 2. Orbital, Centrifugal, and Tidal Potential Parity
# =====================================================================

@pytest.mark.parametrize("nu", [0.0, 0.12, 0.20, 0.24, 0.25])
def test_orbital_potential_parity(nu):
    """Parity of _teobresums_orbital_potential across mass ratios."""
    u_vals = np.array([0.01, 0.05, 0.10, 0.15], dtype=np.float64)
    u_t = torch.tensor(u_vals, dtype=torch.float64)
    nu_t = torch.tensor(nu, dtype=torch.float64)
    res_t = torch_orb_pot(u_t, nu_t).numpy()

    u_j = jnp.array(u_vals, dtype=jnp.float64)
    nu_j = jnp.array(nu, dtype=jnp.float64)
    res_j = np.asarray(jax_orb_pot(u_j, nu_j))

    np.testing.assert_allclose(res_t, res_j, rtol=1e-14, atol=1e-15)


def test_orbital_potential_grad_parity():
    """Parity of orbital potential gradient w.r.t u and nu."""
    u_val = 0.08
    nu_val = 0.22

    u_t = torch.tensor(u_val, dtype=torch.float64, requires_grad=True)
    nu_t = torch.tensor(nu_val, dtype=torch.float64, requires_grad=True)
    pot_t = torch_orb_pot(u_t, nu_t)
    pot_t.backward()

    u_j = jnp.array(u_val, dtype=jnp.float64)
    nu_j = jnp.array(nu_val, dtype=jnp.float64)
    grad_u_j = jax.grad(jax_orb_pot, argnums=0)(u_j, nu_j)
    grad_nu_j = jax.grad(jax_orb_pot, argnums=1)(u_j, nu_j)

    np.testing.assert_allclose(u_t.grad.item(), float(grad_u_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(nu_t.grad.item(), float(grad_nu_j), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("chi1,chi2,cq1,cq2", [
    (0.0, 0.0, None, None),
    (0.2, -0.3, 1.0, 1.0),
    (0.4, 0.1, 3.5, 4.2),
])
def test_centrifugal_radius_parity(chi1, chi2, cq1, cq2):
    """Parity of centrifugal radius and centrifugal_radius_squared."""
    r_np = np.array([6.0, 8.0, 12.0, 20.0], dtype=np.float64)
    nu = 0.23
    r_t = torch.tensor(r_np, dtype=torch.float64)
    r_j = jnp.array(r_np, dtype=jnp.float64)

    rc_t = torch_cent_rad(
        r_t,
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
        cq1=torch.tensor(cq1, dtype=torch.float64) if cq1 is not None else None,
        cq2=torch.tensor(cq2, dtype=torch.float64) if cq2 is not None else None,
    ).numpy()

    rc_j = np.asarray(jax_cent_rad(
        r_j,
        jnp.array(nu, dtype=jnp.float64),
        jnp.array(chi1, dtype=jnp.float64),
        jnp.array(chi2, dtype=jnp.float64),
        cq1=jnp.array(cq1, dtype=jnp.float64) if cq1 is not None else None,
        cq2=jnp.array(cq2, dtype=jnp.float64) if cq2 is not None else None,
    ))

    np.testing.assert_allclose(rc_t, rc_j, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("l1,l2", [(0.0, 0.0), (380.0, 420.0), (800.0, 1000.0)])
def test_tidal_metric_potential_parity(l1, l2):
    """Parity of tidal amplification and metric potentials."""
    nu_val = 0.24
    ham_t = TorchHam(
        torch.tensor(nu_val, dtype=torch.float64),
        torch.tensor(0.1, dtype=torch.float64),
        torch.tensor(-0.05, dtype=torch.float64),
        torch.tensor(l1, dtype=torch.float64),
        torch.tensor(l2, dtype=torch.float64),
    )
    ham_j = JaxHam(
        jnp.array(nu_val, dtype=jnp.float64),
        jnp.array(0.1, dtype=jnp.float64),
        jnp.array(-0.05, dtype=jnp.float64),
        jnp.array(l1, dtype=jnp.float64),
        jnp.array(l2, dtype=jnp.float64),
    )

    r_vals = np.array([7.0, 10.0, 15.0, 25.0], dtype=np.float64)
    r_t = torch.tensor(r_vals, dtype=torch.float64)
    r_j = jnp.array(r_vals, dtype=jnp.float64)

    # Point-mass metric potentials
    a0_t, b0_t, xi0_t = ham_t.point_mass_metric_potentials(r_t)
    a0_j, b0_j, xi0_j = ham_j.point_mass_metric_potentials(r_j)
    np.testing.assert_allclose(a0_t.numpy(), np.asarray(a0_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(b0_t.numpy(), np.asarray(b0_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(xi0_t.numpy(), np.asarray(xi0_j), rtol=1e-14, atol=1e-15)

    # Full metric potentials
    a_t, b_t, xi_t = ham_t.metric_potentials(r_t)
    a_j, b_j, xi_j = ham_j.metric_potentials(r_j)
    np.testing.assert_allclose(a_t.numpy(), np.asarray(a_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(b_t.numpy(), np.asarray(b_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(xi_t.numpy(), np.asarray(xi_j), rtol=1e-14, atol=1e-15)


def test_effective_and_real_hamiltonian_parity():
    """Parity of effective and real Hamiltonian values."""
    nu = 0.235
    ham_t = TorchHam(
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(0.15, dtype=torch.float64),
        torch.tensor(-0.08, dtype=torch.float64),
        torch.tensor(350.0, dtype=torch.float64),
        torch.tensor(450.0, dtype=torch.float64),
    )
    ham_j = JaxHam(
        jnp.array(nu, dtype=jnp.float64),
        jnp.array(0.15, dtype=jnp.float64),
        jnp.array(-0.08, dtype=jnp.float64),
        jnp.array(350.0, dtype=jnp.float64),
        jnp.array(450.0, dtype=jnp.float64),
    )

    r_val, pr_val, pphi_val = 8.5, -0.02, 3.4
    heff_t = ham_t.effective_hamiltonian(
        torch.tensor(r_val, dtype=torch.float64),
        torch.tensor(pr_val, dtype=torch.float64),
        torch.tensor(pphi_val, dtype=torch.float64),
    ).item()
    heff_j = float(ham_j.effective_hamiltonian(
        jnp.array(r_val, dtype=jnp.float64),
        jnp.array(pr_val, dtype=jnp.float64),
        jnp.array(pphi_val, dtype=jnp.float64),
    ))
    np.testing.assert_allclose(heff_t, heff_j, rtol=1e-14, atol=1e-15)

    hreal_t = ham_t.real_hamiltonian(
        torch.tensor(r_val, dtype=torch.float64),
        torch.tensor(pr_val, dtype=torch.float64),
        torch.tensor(pphi_val, dtype=torch.float64),
    ).item()
    hreal_j = float(ham_j.real_hamiltonian(
        jnp.array(r_val, dtype=jnp.float64),
        jnp.array(pr_val, dtype=jnp.float64),
        jnp.array(pphi_val, dtype=jnp.float64),
    ))
    np.testing.assert_allclose(hreal_t, hreal_j, rtol=1e-14, atol=1e-15)


def test_equations_of_motion_conservative_parity():
    """Parity of conservative equations of motion (dr/dt, omega, dpr_star/dt)."""
    ham_t = TorchHam(
        torch.tensor(0.24, dtype=torch.float64),
        torch.tensor(0.1, dtype=torch.float64),
        torch.tensor(-0.05, dtype=torch.float64),
        torch.tensor(380.0, dtype=torch.float64),
        torch.tensor(420.0, dtype=torch.float64),
    )
    ham_j = JaxHam(
        jnp.array(0.24, dtype=jnp.float64),
        jnp.array(0.1, dtype=jnp.float64),
        jnp.array(-0.05, dtype=jnp.float64),
        jnp.array(380.0, dtype=jnp.float64),
        jnp.array(420.0, dtype=jnp.float64),
    )

    r_val, pr_val, pphi_val = 9.0, -0.015, 3.6
    dr_t, om_t, dpr_t = ham_t.equations_of_motion_conservative(
        torch.tensor(r_val, dtype=torch.float64),
        torch.tensor(pr_val, dtype=torch.float64),
        torch.tensor(pphi_val, dtype=torch.float64),
    )
    dr_j, om_j, dpr_j = ham_j.equations_of_motion_conservative(
        jnp.array(r_val, dtype=jnp.float64),
        jnp.array(pr_val, dtype=jnp.float64),
        jnp.array(pphi_val, dtype=jnp.float64),
    )

    np.testing.assert_allclose(dr_t.numpy(), float(dr_j), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(om_t.numpy(), float(om_j), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(dpr_t.numpy(), float(dpr_j), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("m1,m2,l1,l2", [
    (1.4, 1.4, 0.0, 0.0),
    (1.4, 1.3, 380.0, 420.0),
    (1.5, 1.25, 600.0, 200.0),
])
def test_merger_frequency_heuristic_parity(m1, m2, l1, l2):
    """Parity of compute_tidal_merger_frequency between JAX and PyTorch."""
    fmrg_t = torch_f_mrg(m1, m2, lambda1=l1, lambda2=l2, chi1=0.05, chi2=-0.02)
    fmrg_j = jax_f_mrg(m1, m2, lambda1=l1, lambda2=l2, chi1=0.05, chi2=-0.02)
    np.testing.assert_allclose(fmrg_t.numpy(), float(fmrg_j), rtol=1e-14, atol=1e-15)


def test_initial_conditions_parity():
    """Parity of quasicircular initial conditions solve."""
    ham_t = TorchHam(
        torch.tensor(0.24, dtype=torch.float64),
        torch.tensor(0.05, dtype=torch.float64),
        torch.tensor(-0.02, dtype=torch.float64),
        torch.tensor(380.0, dtype=torch.float64),
        torch.tensor(420.0, dtype=torch.float64),
    )
    rr_t = TorchRR(ham_t.nu, ham_t.chi1, ham_t.chi2)
    r0_t, pr0_t, pphi0_t = torch_find_init(ham_t, rr_t, torch.tensor(0.025, dtype=torch.float64))

    ham_j = JaxHam(
        jnp.array(0.24, dtype=jnp.float64),
        jnp.array(0.05, dtype=jnp.float64),
        jnp.array(-0.02, dtype=jnp.float64),
        jnp.array(380.0, dtype=jnp.float64),
        jnp.array(420.0, dtype=jnp.float64),
    )
    rr_j = JaxRR(ham_j.nu, ham_j.chi1, ham_j.chi2)
    r0_j, pr0_j, pphi0_j = jax_find_init(ham_j, rr_j, jnp.array(0.025, dtype=jnp.float64))

    np.testing.assert_allclose(r0_t.numpy(), float(r0_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(pr0_t.numpy(), float(pr0_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(pphi0_t.numpy(), float(pphi0_j), rtol=1e-12, atol=1e-14)


# =====================================================================
# 3. Waveform Evaluation Parity: BNS, BBH, and Batched
# =====================================================================

def test_teobresums_waveform_parity_bns():
    """Verify BNS waveform evaluations parity between JAX and PyTorch."""
    f_np = np.linspace(25.0, 120.0, 40, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    hp_t, hc_t = torch_get_fd_waveform(
        approximant="teobresums",
        mass1=1.4,
        mass2=1.3,
        chi1=0.05,
        chi2=-0.02,
        lambda1=380.0,
        lambda2=420.0,
        sample_frequencies=f_t,
        distance=40.0,
        inclination=0.0,
        dt_geom=1.0,
        num_steps=150,
    )

    hp_j, hc_j = jax_get_fd_waveform(
        approximant="teobresums",
        mass1=1.4,
        mass2=1.3,
        chi1=0.05,
        chi2=-0.02,
        lambda1=380.0,
        lambda2=420.0,
        sample_frequencies=f_j,
        distance=40.0,
        inclination=0.0,
        dt_geom=1.0,
        num_steps=150,
    )

    assert np.all(np.isfinite(np.asarray(hp_j)))
    assert np.all(np.isfinite(np.asarray(hc_j)))
    assert np.any(np.abs(np.asarray(hp_j)) > 0.0)

    # Cross-framework match to high numerical precision
    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-10, atol=1e-28)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-10, atol=1e-28)


def test_teobresums_waveform_parity_bbh():
    """Verify BBH (zero tides) waveform parity between JAX and PyTorch."""
    f_np = np.linspace(25.0, 100.0, 30, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    model_t = TorchTEOBResumS()
    hp_t, hc_t = model_t(
        f=f_t,
        mass1=30.0,
        mass2=20.0,
        chi1=0.1,
        chi2=-0.1,
        lambda1=0.0,
        lambda2=0.0,
        distance=100.0,
        dt_geom=1.0,
        num_steps=100,
    )

    model_j = JaxTEOBResumS()
    hp_j, hc_j = model_j(
        f=f_j,
        mass1=30.0,
        mass2=20.0,
        chi1=0.1,
        chi2=-0.1,
        lambda1=0.0,
        lambda2=0.0,
        distance=100.0,
        dt_geom=1.0,
        num_steps=100,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-10, atol=1e-28)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-10, atol=1e-28)


def test_teobresums_waveform_parity_batched():
    """Verify batched waveform generation parity."""
    f_np = np.linspace(30.0, 110.0, 25, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    m1_t = torch.tensor([1.4, 1.5], dtype=torch.float64)
    m2_t = torch.tensor([1.3, 1.2], dtype=torch.float64)
    l1_t = torch.tensor([380.0, 300.0], dtype=torch.float64)
    l2_t = torch.tensor([420.0, 500.0], dtype=torch.float64)

    m1_j = jnp.array([1.4, 1.5], dtype=jnp.float64)
    m2_j = jnp.array([1.3, 1.2], dtype=jnp.float64)
    l1_j = jnp.array([380.0, 300.0], dtype=jnp.float64)
    l2_j = jnp.array([420.0, 500.0], dtype=jnp.float64)

    hp_t, hc_t = torch_get_fd_waveform(
        approximant="teobresums",
        mass1=m1_t,
        mass2=m2_t,
        lambda1=l1_t,
        lambda2=l2_t,
        sample_frequencies=f_t,
        dt_geom=1.0,
        num_steps=80,
    )

    hp_j, hc_j = jax_get_fd_waveform(
        approximant="teobresums",
        mass1=m1_j,
        mass2=m2_j,
        lambda1=l1_j,
        lambda2=l2_j,
        sample_frequencies=f_j,
        dt_geom=1.0,
        num_steps=80,
    )

    assert hp_j.shape == (2, len(f_np))
    assert hc_j.shape == (2, len(f_np))
    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-10, atol=1e-28)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-10, atol=1e-28)


# =====================================================================
# 4. JIT Compilation and Autodiff Tests
# =====================================================================

def test_teobresums_jax_jit_compilation():
    """Verify jax.jit compilation of TEOBResumS.evaluate."""
    f = jnp.linspace(25.0, 100.0, 30, dtype=jnp.float64)

    @jax.jit
    def run_eval(m1, m2, l1, l2):
        return JaxTEOBResumS.evaluate(
            f=f,
            mass1=m1,
            mass2=m2,
            chi1=0.05,
            chi2=-0.02,
            lambda1=l1,
            lambda2=l2,
            dt_geom=1.0,
            num_steps=80,
        )

    hp_jitted, hc_jitted = run_eval(1.4, 1.3, 380.0, 420.0)
    hp_ref, hc_ref = JaxTEOBResumS.evaluate(
        f=f,
        mass1=1.4,
        mass2=1.3,
        chi1=0.05,
        chi2=-0.02,
        lambda1=380.0,
        lambda2=420.0,
        dt_geom=1.0,
        num_steps=80,
    )

    np.testing.assert_allclose(np.asarray(hp_jitted), np.asarray(hp_ref), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(np.asarray(hc_jitted), np.asarray(hc_ref), rtol=1e-14, atol=1e-15)


def test_teobresums_hamiltonian_autodiff():
    """Verify JAX autodiff w.r.t masses and tidal parameters matches PyTorch."""
    nu_val, chi1_val, chi2_val, l1_val, l2_val = 0.24, 0.2, 0.1, 350.0, 450.0
    r_val, pr_val, pphi_val = 10.0, -0.01, 3.5

    # PyTorch gradients
    nu_t = torch.tensor(nu_val, dtype=torch.float64, requires_grad=True)
    chi1_t = torch.tensor(chi1_val, dtype=torch.float64, requires_grad=True)
    chi2_t = torch.tensor(chi2_val, dtype=torch.float64, requires_grad=True)
    l1_t = torch.tensor(l1_val, dtype=torch.float64, requires_grad=True)
    l2_t = torch.tensor(l2_val, dtype=torch.float64, requires_grad=True)
    r_t = torch.tensor(r_val, dtype=torch.float64, requires_grad=True)
    pr_t = torch.tensor(pr_val, dtype=torch.float64, requires_grad=True)
    pphi_t = torch.tensor(pphi_val, dtype=torch.float64, requires_grad=True)

    ham_t = TorchHam(nu=nu_t, chi1=chi1_t, chi2=chi2_t, lambda1=l1_t, lambda2=l2_t)
    h_real_t = ham_t.real_hamiltonian(r_t, pr_t, pphi_t)
    h_real_t.backward()

    # JAX gradients
    def jax_h_real(nu, chi1, chi2, l1, l2, r, pr, pphi):
        ham = JaxHam(nu=nu, chi1=chi1, chi2=chi2, lambda1=l1, lambda2=l2)
        return ham.real_hamiltonian(r, pr, pphi)

    grads_j = jax.grad(jax_h_real, argnums=(0, 1, 2, 3, 4, 5, 6, 7))(
        jnp.array(nu_val, dtype=jnp.float64),
        jnp.array(chi1_val, dtype=jnp.float64),
        jnp.array(chi2_val, dtype=jnp.float64),
        jnp.array(l1_val, dtype=jnp.float64),
        jnp.array(l2_val, dtype=jnp.float64),
        jnp.array(r_val, dtype=jnp.float64),
        jnp.array(pr_val, dtype=jnp.float64),
        jnp.array(pphi_val, dtype=jnp.float64),
    )

    t_grads = [nu_t.grad.item(), chi1_t.grad.item(), chi2_t.grad.item(), l1_t.grad.item(),
               l2_t.grad.item(), r_t.grad.item(), pr_t.grad.item(), pphi_t.grad.item()]

    for got_t, got_j in zip(t_grads, grads_j):
        assert np.isfinite(float(got_j))
        np.testing.assert_allclose(got_t, float(got_j), rtol=1e-12, atol=1e-14)


def test_merger_frequency_autodiff():
    """Verify d(f_mrg)/d(Lambda) < 0 and d(f_mrg)/d(mass) < 0 in JAX matching PyTorch."""
    m1, m2 = 1.4, 1.4
    lam = 400.0

    lam_t = torch.tensor(lam, dtype=torch.float64, requires_grad=True)
    f_t = torch_f_mrg(m1, m2, lambda1=lam_t, lambda2=lam_t)
    f_t.backward()

    def jax_fmrg_lam(l_val):
        return jax_f_mrg(m1, m2, lambda1=l_val, lambda2=l_val)

    def jax_fmrg_m1(m1_val):
        return jax_f_mrg(m1_val, m2, lambda1=lam, lambda2=lam)

    g_lam_j = jax.grad(jax_fmrg_lam)(jnp.array(lam, dtype=jnp.float64))
    g_m1_j = jax.grad(jax_fmrg_m1)(jnp.array(m1, dtype=jnp.float64))

    assert float(g_lam_j) < 0.0
    assert float(g_m1_j) < 0.0
    np.testing.assert_allclose(lam_t.grad.item(), float(g_lam_j), rtol=1e-13, atol=1e-15)
