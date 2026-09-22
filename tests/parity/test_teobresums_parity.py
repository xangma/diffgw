"""Cross-framework parity test: torchwave vs jaxwave for published TEOB components."""

import math
import numpy as np
import pytest
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from torchwave.approximants.teobresums import (
    TEOBResumSHamiltonian as TorchHam,
    _piecewise_linear_fourier as torch_plf,
    _teobresums_orbital_potential as torch_orb_pot,
    _teobresums_centrifugal_radius as torch_cent_rad,
)
from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions as torch_find_init,
)
from torchwave.waveforms import get_fd_waveform as torch_get_fd_waveform


from jaxwave.approximants.teobresums import (
    TEOBResumSHamiltonian as JaxHam,
    _piecewise_linear_fourier as jax_plf,
    _teobresums_orbital_potential as jax_orb_pot,
    _teobresums_centrifugal_radius as jax_cent_rad,
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


def test_effective_and_real_hamiltonian_parity():
    """Parity of effective and real Hamiltonian values."""
    nu = 0.235
    ham_t = TorchHam(
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    ham_j = JaxHam(
        jnp.array(nu, dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
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
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    ham_j = JaxHam(
        jnp.array(0.24, dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
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


def test_nonspinning_hamiltonian_jit_and_mass_gradient():
    r, pr, j = 10., -.01, 3.5
    nu = torch.tensor(.24, dtype=torch.float64, requires_grad=True)
    energy = TorchHam(nu).real_hamiltonian(torch.tensor(r, dtype=torch.float64), torch.tensor(pr, dtype=torch.float64), torch.tensor(j, dtype=torch.float64))
    gradient, = torch.autograd.grad(energy, nu)
    def field(value):
        return JaxHam(value).real_hamiltonian(jnp.asarray(r), jnp.asarray(pr), jnp.asarray(j))
    np.testing.assert_allclose(float(gradient), float(jax.jit(jax.grad(field))(.24)), rtol=1e-6)
