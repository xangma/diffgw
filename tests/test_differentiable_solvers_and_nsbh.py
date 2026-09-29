"""Tests for differentiable JAX solvers and NSBH smooth continuation.

Tests:
1. Pure JAX autodiff with respect to Kerr spin a and complex oblateness c on the
   spheroidal angular projection without pure_callback.
2. Smooth C^1 polynomial continuation of IMRPhenomNSBH premerger amplitude
   across Lambda in [0, 2] ensuring no discontinuous jumps at Lambda=1 in both
   PyTorch and JAX (contrasting with the Thompson 2020 Eq. (26) published paper
   jump discontinuity).
3. Verification of NSBH ringdown width window sign convention (+0.81248 omega^+
   BBH continuity limit vs paper minus window).
"""

import math
import numpy as np
import pytest

import torch
import jax
import jax.numpy as jnp

from diffgw_core.windowing import (
    nsbh_gamma1_continuation,
    torch_nsbh_gamma1_continuation,
    jax_nsbh_gamma1_continuation,
    nsbh_ringdown_width_window,
    torch_nsbh_ringdown_width_window,
    jax_nsbh_ringdown_width_window,
)
from diffgw_reference.nsbh_amplitude_variants import gamma1_small_lambda, delta2_large_lambda
from torchwave.approximants.imrphenom_nsbh import IMRPhenomNSBH as TorchNSBH
from jaxwave.approximants.imrphenom_nsbh import IMRPhenomNSBH as JaxNSBH
from jaxwave.approximants.kerr_ringdown import (
    KerrRingdown,
    spheroidal_coefficients,
    _ringdown_projection,
    spin_weighted_spheroidal_harmonic,
)
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220


# ---------------------------------------------------------------------------
# 1. Kerr Spheroidal Angular Differentiable Solver Tests (JAX)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def enable_jax_x64():
    """Ensure JAX float64 is active for high-precision autodiff checks."""
    prev = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_jax_spheroidal_eigensolver_accuracy():
    """Verify pure JAX inverse-iteration eigensolver against independent scalar reference."""
    # Test (2, 2) mode at oblateness c = 0.42 - 0.10j
    c = 0.42 - 0.10j
    eigenvalue, coefficients = spheroidal_coefficients(2, 2, jnp.asarray(c, dtype=jnp.complex128))

    # Compare with known benchmark from kerr-angular-projection spec
    # Benchmark: A(2, 2; c=0.42 - 0.10j) approx 2.7540812545 + 0.3313863217j
    expected_real = 2.7540812545
    expected_imag = 0.3313863217
    assert abs(float(jnp.real(eigenvalue)) - expected_real) < 1e-9
    assert abs(float(jnp.imag(eigenvalue)) - expected_imag) < 1e-9

    # Unit norm check
    norm_sq = float(jnp.sum(jnp.abs(coefficients) ** 2))
    assert abs(norm_sq - 1.0) < 1e-12

    # Phase anchor check: first component must be real positive
    anchor = coefficients[0]
    assert float(jnp.imag(anchor)) == pytest.approx(0.0, abs=1e-14)
    assert float(jnp.real(anchor)) > 0.0

    # At c = 0, exactly recover spherical eigenvalue A_0 = l*(l+1) - 2 = 4
    eig_0, coeff_0 = spheroidal_coefficients(2, 2, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    assert abs(float(jnp.real(eig_0)) - 4.0) < 1e-14
    assert abs(float(jnp.imag(eig_0))) < 1e-14
    assert abs(float(jnp.real(coeff_0[0])) - 1.0) < 1e-14
    assert float(jnp.linalg.norm(coeff_0[1:])) < 1e-14


def test_jax_spheroidal_autodiff_wrt_oblateness():
    """Verify JAX reverse-mode autodiff with respect to complex oblateness c."""
    c_real = 0.42
    c_imag = -0.10

    # Differentiate eigenvalue A with respect to real part of c
    def eig_loss_real(cr):
        c = jnp.asarray(cr + 1j * c_imag, dtype=jnp.complex128)
        val, _ = spheroidal_coefficients(2, 2, c)
        return jnp.real(val)

    grad_real_ad = jax.grad(eig_loss_real)(c_real)
    eps = 1e-7
    grad_real_fd = (eig_loss_real(c_real + eps) - eig_loss_real(c_real - eps)) / (2 * eps)
    assert abs(grad_real_ad - grad_real_fd) / abs(grad_real_fd) < 1e-6

    # Differentiate projected angular factor with respect to oblateness
    def proj_loss(cr):
        c = jnp.asarray(cr + 1j * c_imag, dtype=jnp.complex128)
        theta = jnp.asarray(1.07, dtype=jnp.float64)
        phi = jnp.asarray(-0.31, dtype=jnp.float64)
        dp, _ = _ringdown_projection(2, 2, c, theta, phi, "spheroidal")
        return jnp.real(dp)

    grad_proj_ad = jax.grad(proj_loss)(c_real)
    grad_proj_fd = (proj_loss(c_real + eps) - proj_loss(c_real - eps)) / (2 * eps)
    assert abs(grad_proj_ad - grad_proj_fd) / abs(grad_proj_fd) < 1e-6


def test_jax_spheroidal_autodiff_wrt_spin():
    """Verify JAX reverse-mode autodiff with respect to dimensionless Kerr spin a without pure_callback."""
    # Function evaluating projected spheroidal harmonic d_plus(a) through QNM frequency
    def projected_plus_from_spin(spin_a):
        wr, damp = evaluate_kerr_qnm_220(spin_a)
        # Dimensionless oblateness c = a * (M*omega - i * M*gamma)
        c = spin_a * (wr - 1j * damp)
        theta = jnp.asarray(0.85, dtype=jnp.float64)
        phi = jnp.asarray(0.25, dtype=jnp.float64)
        d_plus, _ = _ringdown_projection(2, 2, c, theta, phi, "spheroidal")
        return jnp.real(d_plus)

    test_spins = [0.2, 0.5, 0.7, 0.85]
    eps = 1e-6

    for a in test_spins:
        grad_ad = jax.grad(projected_plus_from_spin)(float(a))
        grad_fd = (projected_plus_from_spin(a + eps) - projected_plus_from_spin(a - eps)) / (2 * eps)

        assert jnp.isfinite(grad_ad), f"Gradient at a={a} must be finite"
        rel_err = abs(grad_ad - grad_fd) / abs(grad_fd)
        assert rel_err < 1e-5, f"Grad error at a={a}: ad={grad_ad}, fd={grad_fd}, rel_err={rel_err}"

    # Verify that the computation can be JIT-compiled
    jit_grad_fn = jax.jit(jax.grad(projected_plus_from_spin))
    g_jit = jit_grad_fn(0.7)
    assert jnp.isfinite(g_jit)
    assert abs(g_jit - jax.grad(projected_plus_from_spin)(0.7)) < 1e-12


def test_jax_kerr_ringdown_waveform_autodiff():
    """Verify end-to-end autodiff through KerrRingdown time and frequency domain waveforms."""
    # Test frequency-domain autodiff w.r.t final_spin
    def fd_loss(spin):
        freqs = jnp.array([150.0, 250.0, 350.0], dtype=jnp.float64)
        hp, _ = KerrRingdown.frequency_domain(
            frequencies=freqs,
            final_mass=65.0,
            final_spin=spin,
            inclination=0.5,
            phi=0.1,
            angular_basis="spheroidal",
        )
        return jnp.sum(jnp.real(hp))

    g_spin = jax.grad(fd_loss)(0.68)
    assert jnp.isfinite(g_spin)
    eps = 1e-6
    g_fd = (fd_loss(0.68 + eps) - fd_loss(0.68 - eps)) / (2 * eps)
    assert abs(g_spin - g_fd) / abs(g_fd) < 1e-4

    # Test time-domain autodiff w.r.t final_spin
    def td_loss(spin):
        times = jnp.linspace(0.0, 0.05, 100, dtype=jnp.float64)
        hp, _ = KerrRingdown.time_domain(
            times=times,
            final_mass=65.0,
            final_spin=spin,
            inclination=0.5,
            phi=0.1,
            angular_basis="spheroidal",
        )
        return jnp.sum(hp)

    g_td_spin = jax.grad(td_loss)(0.68)
    assert jnp.isfinite(g_td_spin)
    g_td_fd = (td_loss(0.68 + eps) - td_loss(0.68 - eps)) / (2 * eps)
    assert abs(g_td_spin - g_td_fd) / abs(g_td_fd) < 1e-4


# ---------------------------------------------------------------------------
# 2. NSBH Smooth Continuation & Jump Discontinuity Tests
# ---------------------------------------------------------------------------

def test_nsbh_paper_discontinuity_vs_smooth_continuation():
    """Verify that paper Eq. (26) has a jump discontinuity of +1.0 at Lambda=1, while project_correction is C^1 smooth."""
    eps = 1e-7

    # 1. Evaluate published paper Eq. (26) variant
    paper_left = nsbh_gamma1_continuation(1.0 - eps, variant="paper")
    paper_right = nsbh_gamma1_continuation(1.0 + eps, variant="paper")
    paper_jump = paper_right - paper_left

    # Paper prints: 1 - 0.5(1) - 0.25(1)^2 = 0.25 for Lambda <= 1, and 1.25 for Lambda > 1
    assert abs(paper_left - 0.25) < 1e-6
    assert abs(paper_right - 1.25) < 1e-6
    assert abs(paper_jump - 1.0) < 1e-6, "Paper Eq. (26) must exhibit the documented Delta=+1.0 jump discontinuity"

    # 2. Evaluate project_correction smooth continuation variant: 1 + 0.5*Lambda - 0.25*Lambda^2
    corr_left = nsbh_gamma1_continuation(1.0 - eps, variant="project_correction")
    corr_right = nsbh_gamma1_continuation(1.0 + eps, variant="project_correction")
    corr_at_1 = nsbh_gamma1_continuation(1.0, variant="project_correction")

    # At Lambda=1: 1 + 0.5(1) - 0.25(1)^2 = 1.25, exactly matching Lambda > 1
    assert abs(corr_at_1 - 1.25) < 1e-15
    assert abs(corr_left - 1.25) < 1e-6
    assert abs(corr_right - 1.25) < 1e-15
    assert abs(corr_right - corr_left) < 1e-6, "project_correction variant must have zero jump discontinuity"

    # 3. Smooth BBH connection at Lambda=0
    corr_at_0 = nsbh_gamma1_continuation(0.0, variant="project_correction")
    assert abs(corr_at_0 - 1.0) < 1e-15, "At Lambda=0, correction factor must equal 1.0 (exact BBH baseline)"

    # 4. Check first derivative continuity (C^1 smoothness)
    # d/dLambda (1 + 0.5*Lambda - 0.25*Lambda^2) = 0.5 - 0.5*Lambda = 0.0 at Lambda=1
    d_left = (nsbh_gamma1_continuation(1.0, variant="project_correction") -
              nsbh_gamma1_continuation(1.0 - eps, variant="project_correction")) / eps
    d_right = (nsbh_gamma1_continuation(1.0 + eps, variant="project_correction") -
               nsbh_gamma1_continuation(1.0, variant="project_correction")) / eps
    assert abs(d_left) < 1e-6
    assert abs(d_right) < 1e-6
    assert abs(d_left - d_right) < 1e-6


def test_nsbh_smooth_continuation_grid_torch_and_jax():
    """Verify that both PyTorch and JAX evaluate smooth C^1 continuation across Lambda in [0, 2] without jumps."""
    lambda_grid_np = np.linspace(0.0, 2.0, 501, dtype=np.float64)

    # PyTorch evaluation
    lambda_t = torch.from_numpy(lambda_grid_np)
    gamma1_t = torch_nsbh_gamma1_continuation(lambda_t, variant="project_correction").numpy()

    # JAX evaluation
    lambda_j = jnp.asarray(lambda_grid_np)
    gamma1_j = np.asarray(jax_nsbh_gamma1_continuation(lambda_j, variant="project_correction"))

    # Agreement between Torch and JAX
    np.testing.assert_allclose(gamma1_t, gamma1_j, rtol=1e-14, atol=1e-14)

    # Check that maximum consecutive step difference is small (no discontinuous steps)
    diffs_t = np.abs(np.diff(gamma1_t))
    diffs_j = np.abs(np.diff(gamma1_j))
    # Step size is 2.0 / 500 = 0.004. Max slope is 0.5 (at Lambda=0), so max step is ~0.002
    assert np.max(diffs_t) < 0.003
    assert np.max(diffs_j) < 0.003

    # Check exactly at Lambda=1 boundary index (index 250 in 501 points)
    idx_1 = 250
    assert abs(lambda_grid_np[idx_1] - 1.0) < 1e-14
    assert abs(gamma1_t[idx_1] - 1.25) < 1e-14
    assert abs(gamma1_j[idx_1] - 1.25) < 1e-14
    # Check that jump between left of 1 and right of 1 is purely the smooth slope, not a jump of 1.0
    step_at_1 = abs(gamma1_t[idx_1 + 1] - gamma1_t[idx_1 - 1])
    assert step_at_1 < 1e-4


def test_nsbh_waveform_continuity_across_lambda_one():
    """Verify that full IMRPhenomNSBH waveform outputs in Torch and JAX vary continuously across Lambda=1."""
    f = np.linspace(20.0, 500.0, 200, dtype=np.float64)
    lambdas = [0.8, 0.95, 0.999, 1.0, 1.001, 1.05, 1.2]

    torch_model = TorchNSBH()
    torch_f = torch.from_numpy(f)

    jax_f = jnp.asarray(f)

    torch_hplus_norms = []
    jax_hplus_norms = []

    for lam in lambdas:
        # Torch waveform
        hp_t, _ = torch_model(
            torch_f,
            mass1=6.0,
            mass2=1.4,
            chi1=0.2,
            chi2=0.0,
            lambda2=lam,
            distance=100.0,
            inclination=0.0,
        )
        torch_norm = float(torch.linalg.vector_norm(hp_t).detach())
        torch_hplus_norms.append(torch_norm)

        # JAX waveform
        hp_j, _ = JaxNSBH.evaluate(
            jax_f,
            mass1=6.0,
            mass2=1.4,
            chi1=0.2,
            chi2=0.0,
            lambda2=lam,
            distance=100.0,
            inclination=0.0,
        )
        jax_norm = float(jnp.linalg.norm(hp_j))
        jax_hplus_norms.append(jax_norm)

    # Check parity between Torch and JAX norms
    np.testing.assert_allclose(torch_hplus_norms, jax_hplus_norms, rtol=1e-12)

    # Check that across Lambda in [0.999, 1.001] there is no discontinuous jump
    # Index 2 is 0.999, Index 3 is 1.0, Index 4 is 1.001
    jump_across_1 = abs(torch_hplus_norms[4] - torch_hplus_norms[2]) / torch_hplus_norms[3]
    assert jump_across_1 < 1e-3, f"Waveform norm should be continuous across Lambda=1, got relative variation {jump_across_1}"


# ---------------------------------------------------------------------------
# 3. NSBH Ringdown Width Window Sign Convention Tests
# ---------------------------------------------------------------------------

def test_nsbh_ringdown_width_window_sign_convention():
    """Verify NSBH ringdown width window: +0.81248 omega^+ BBH continuity vs paper minus window."""
    f_rd_tilde = 800.0

    # In the high disruption frequency limit (f_tide >> f_rd_tilde), e.g. non-disruptive merger:
    f_tide_high = 8000.0

    # Project correction (+0.81248 omega^+ plus window)
    win_plus = nsbh_ringdown_width_window(f_tide_high, f_rd_tilde, variant="project_correction")
    # Paper published Eq. (29) (minus window omega^-)
    win_minus = nsbh_ringdown_width_window(f_tide_high, f_rd_tilde, variant="paper")

    # The plus window smoothly recovers the BBH ringdown constant delta_2 = 0.81248
    assert abs(win_plus - 0.81248) < 1e-4, "Plus window must approach 0.81248 in BBH / large-tidal-frequency limit"

    # The paper's minus window unphysically collapses to 0.0
    assert abs(win_minus - 0.0) < 1e-4, "Paper minus window collapses to zero as f_tide >> f_rd"

    # Verify PyTorch and JAX window implementations match
    ft_t = torch.tensor(f_tide_high, dtype=torch.float64)
    fr_t = torch.tensor(f_rd_tilde, dtype=torch.float64)
    win_torch = float(torch_nsbh_ringdown_width_window(ft_t, fr_t, variant="project_correction"))

    ft_j = jnp.array(f_tide_high, dtype=jnp.float64)
    fr_j = jnp.array(f_rd_tilde, dtype=jnp.float64)
    win_jax = float(jax_nsbh_ringdown_width_window(ft_j, fr_j, variant="project_correction"))

    assert abs(win_torch - win_plus) < 1e-12
    assert abs(win_jax - win_plus) < 1e-12
