"""Tests for precession boundary singularities and edge cases across PyTorch and JAX.

Covers:
1. Aligned equal-mass MSA remnant and nonaligned mode-3 rejection.
2. Equal-mass common-mass derivatives with explicit remnant mode 0.
3. Up-down near-aligned turning root coalescence (S^2 turning root gap < 1e-20).
4. IMRPhenomXP equal-mass waveform parity with prec_version=223, final_spin_mod=0.
5. SpinTaylor beta continuation damping beyond attachment frequency (damp_beta=True).
"""

import math
import numpy as np
import pytest
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffgw_core.constants import PI, MSUN_SEC
import torchwave.approximants.precession_msa as torch_msa
import jaxwave.approximants.precession_msa as jax_msa
from torchwave.approximants.precession_spintaylor import (
    evaluate_spintaylor_precession as torch_eval_spintaylor,
)
from jaxwave.approximants.precession_spintaylor import (
    evaluate_spintaylor_precession as jax_eval_spintaylor,
)
from torchwave.approximants.imrphenom_xp import IMRPhenomXP as TorchPhenomXP
from jaxwave.approximants.imrphenom_xp import IMRPhenomXP as JaxPhenomXP


def test_equal_mass_msa_remnant_torch_and_jax():
    """Aligned q=1 remnants agree; nonaligned MSA remnants are unsupported."""
    v0_val = (PI * 60.0 * MSUN_SEC * 20.0) ** (1.0 / 3.0)
    v0_t = torch.tensor(v0_val, dtype=torch.float64)
    v0_j = jnp.array(v0_val, dtype=jnp.float64)

    # 1. Nonspinning equal mass
    m_t = torch.tensor(30.0, dtype=torch.float64)
    m_j = jnp.array(30.0, dtype=jnp.float64)
    zero_t = torch.tensor(0.0, dtype=torch.float64)
    zero_j = jnp.array(0.0, dtype=jnp.float64)

    spin_t_non = torch_msa.compute_msa_final_spin(
        v0_t, m_t, m_t, zero_t, zero_t, zero_t, zero_t, zero_t, zero_t
    )
    spin_j_non = jax_msa.compute_msa_final_spin(
        v0_j, m_j, m_j, zero_j, zero_j, zero_j, zero_j, zero_j, zero_j
    )

    assert torch.isfinite(spin_t_non)
    assert jnp.isfinite(spin_j_non)
    np.testing.assert_allclose(spin_t_non.item(), float(spin_j_non), atol=1e-14, rtol=1e-14)

    # 2. Precessing equal mass (different mass scales and spin configurations)
    cases = [
        (30.0, (0.2, 0.1, 0.3), (0.1, 0.2, -0.1)),
        (15.0, (0.3, -0.2, 0.4), (-0.1, 0.3, -0.2)),
        (10.0, (0.2, 0.1, 0.3), (-0.1, 0.2, -0.2)),
    ]

    for mass, s1_vec, s2_vec in cases:
        m_t = torch.tensor(mass, dtype=torch.float64)
        m_j = jnp.array(mass, dtype=jnp.float64)
        s1x_t, s1y_t, s1z_t = [torch.tensor(x, dtype=torch.float64) for x in s1_vec]
        s2x_t, s2y_t, s2z_t = [torch.tensor(x, dtype=torch.float64) for x in s2_vec]
        s1x_j, s1y_j, s1z_j = [jnp.array(x, dtype=jnp.float64) for x in s1_vec]
        s2x_j, s2y_j, s2z_j = [jnp.array(x, dtype=jnp.float64) for x in s2_vec]

        with pytest.raises(ValueError, match="valid nonaligned MSA remnant root"):
            torch_msa.compute_msa_final_spin(
                v0_t, m_t, m_t, s1x_t, s1y_t, s1z_t, s2x_t, s2y_t, s2z_t
            )
        with pytest.raises(ValueError, match="valid nonaligned MSA remnant root"):
            jax_msa.compute_msa_final_spin(
                v0_j, m_j, m_j, s1x_j, s1y_j, s1z_j, s2x_j, s2y_j, s2z_j
            )


def test_equal_mass_mode0_autograd_and_jax_grad():
    """Mode-0 derivatives along q=1 agree with central finite differences."""
    f = np.array([25.0, 40.0, 80.0])
    params = dict(
        spin1x=0.2, spin1y=0.1, spin1z=0.3,
        spin2x=-0.1, spin2y=0.2, spin2z=-0.2, distance=150.0,
        inclination=0.7, phic=0.3, f_ref=20.0,
        prec_version=223, final_spin_mod=0,
    )
    m1_t = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    # Vary both masses together: changing their ordering at q=1 also crosses
    # the effective in-plane-spin maximum for this equal-magnitude fixture.
    hp, hc = TorchPhenomXP()(torch.as_tensor(f), mass1=m1_t, mass2=m1_t, **params)
    loss = 1e44 * (hp.abs().square() + hc.abs().square()).sum()
    torch_grad = torch.autograd.grad(loss, m1_t)[0].item()

    def j_fn(m1):
        hp, hc = JaxPhenomXP.evaluate(jnp.asarray(f), mass1=m1, mass2=m1, **params)
        return 1e44 * jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2)

    jax_grad = float(jax.grad(j_fn)(jnp.asarray(10.0)))
    assert np.isfinite(torch_grad)
    assert np.isfinite(jax_grad)
    np.testing.assert_allclose(torch_grad, jax_grad, atol=1e-12, rtol=1e-12)

    # 3. Central finite difference comparison
    h = 1e-6
    f_plus = float(j_fn(jnp.asarray(10.0 + h)))
    f_minus = float(j_fn(jnp.asarray(10.0 - h)))
    fd_grad = (f_plus - f_minus) / (2.0 * h)

    np.testing.assert_allclose(jax_grad, fd_grad, rtol=1e-4)


def test_near_aligned_up_down_root_coalescence():
    """Verify S^2 turning roots for near-aligned up-down spins with root gap < 1e-20.

    Ensures real, non-NaN roots with s3_sq <= sm_sq <= sp_sq, and finite autodiff
    gradients without arccos/sqrt branch cut infinities.
    """
    m1, m2 = 30.0, 20.0
    total = m1 + m2
    m1_n, m2_n = m1 / total, m2 / total
    eta = m1_n * m2_n
    delta = m1_n - m2_n
    q_c = m2_n / m1_n

    # Strictly aligned up-down spins: turning roots coalesce
    s1z = 0.5
    s2z = -0.5
    s1 = m1_n**2 * s1z
    s2 = m2_n**2 * s2z
    s1_sq = s1**2
    s2_sq = s2**2
    xi = (1.0 + q_c) * s1 + (1.0 + 1.0 / q_c) * s2
    v0 = 0.2
    l0 = eta / v0
    j0_sq = (l0 + s1 + s2) ** 2

    # 1. PyTorch solve_spin_roots
    l0_t = torch.tensor(l0, dtype=torch.float64, requires_grad=True)
    s3_t, sm_t, sp_t, valid_t = torch_msa.solve_spin_roots(
        l0_t,
        torch.tensor(j0_sq, dtype=torch.float64),
        torch.tensor(s1_sq, dtype=torch.float64),
        torch.tensor(s2_sq, dtype=torch.float64),
        torch.tensor(xi, dtype=torch.float64),
        torch.tensor(q_c, dtype=torch.float64),
        torch.tensor(eta, dtype=torch.float64),
        torch.tensor(delta, dtype=torch.float64),
    )

    assert torch.all(torch.isfinite(torch.stack([s3_t, sm_t, sp_t])))
    assert s3_t.item() <= sm_t.item() + 1e-15
    assert sm_t.item() <= sp_t.item() + 1e-15
    assert not bool(valid_t)

    # Autograd through solve_spin_roots at coalesced roots
    loss_t = s3_t + sm_t + sp_t
    loss_t.backward()
    assert torch.isfinite(l0_t.grad)
    assert not math.isnan(l0_t.grad.item())

    # 2. JAX solve_spin_roots
    def j_roots(l0_val):
        s3, sm, sp, valid = jax_msa.solve_spin_roots(
            l0_val,
            jnp.array(j0_sq, dtype=jnp.float64),
            jnp.array(s1_sq, dtype=jnp.float64),
            jnp.array(s2_sq, dtype=jnp.float64),
            jnp.array(xi, dtype=jnp.float64),
            jnp.array(q_c, dtype=jnp.float64),
            jnp.array(eta, dtype=jnp.float64),
            jnp.array(delta, dtype=jnp.float64),
        )
        return s3, sm, sp, valid

    s3_j, sm_j, sp_j, valid_j = j_roots(jnp.array(l0, dtype=jnp.float64))
    assert jnp.all(jnp.isfinite(jnp.stack([s3_j, sm_j, sp_j])))
    assert float(s3_j) <= float(sm_j) + 1e-15
    assert float(sm_j) <= float(sp_j) + 1e-15
    assert not bool(valid_j)

    # Parity of roots
    np.testing.assert_allclose(
        [s3_t.item(), sm_t.item(), sp_t.item()],
        [float(s3_j), float(sm_j), float(sp_j)],
        atol=1e-14, rtol=1e-14,
    )

    # JAX grad through solve_spin_roots
    jax_grad = jax.grad(lambda l: sum(j_roots(l)[:3]))(jnp.array(l0, dtype=jnp.float64))
    assert jnp.isfinite(jax_grad)
    np.testing.assert_allclose(l0_t.grad.item(), float(jax_grad), atol=1e-12, rtol=1e-12)

    # 3. msa_oscillatory_angles at tiny root gap (< 1e-20)
    v_eval = torch.tensor(0.25, dtype=torch.float64)
    j_eval = torch.tensor(1.2, dtype=torch.float64)
    l_eval = torch.tensor(1.0, dtype=torch.float64)
    psi_eval = torch.tensor(0.5, dtype=torch.float64)

    alpha_t, eps_t = torch_msa.msa_oscillatory_angles(
        v_eval, j_eval, l_eval, torch.tensor(xi), torch.tensor(eta),
        torch.tensor(s1_sq), torch.tensor(s2_sq), torch.tensor(delta),
        s3_t.detach(), sm_t.detach(), sm_t.detach(),  # sp == sm: gap = 0
        psi_eval,
    )
    assert torch.isfinite(alpha_t)
    assert torch.isfinite(eps_t)
    assert alpha_t.item() == 0.0
    assert eps_t.item() == 0.0

    alpha_j, eps_j = jax_msa.msa_oscillatory_angles(
        jnp.array(0.25, dtype=jnp.float64),
        jnp.array(1.2, dtype=jnp.float64),
        jnp.array(1.0, dtype=jnp.float64),
        jnp.array(xi, dtype=jnp.float64),
        jnp.array(eta, dtype=jnp.float64),
        jnp.array(s1_sq, dtype=jnp.float64),
        jnp.array(s2_sq, dtype=jnp.float64),
        jnp.array(delta, dtype=jnp.float64),
        s3_j, sm_j, sm_j,  # sp == sm: gap = 0
        jnp.array(0.5, dtype=jnp.float64),
    )
    assert jnp.isfinite(alpha_j)
    assert jnp.isfinite(eps_j)
    assert float(alpha_j) == 0.0
    assert float(eps_j) == 0.0


def test_imrphenom_xp_equal_mass_mode0_parity():
    """Verify the supported q=1 waveform with MSA angles and remnant mode 0."""
    f = np.linspace(25.0, 300.0, 64)
    f_t = torch.tensor(f, dtype=torch.float64)
    f_j = jnp.array(f, dtype=jnp.float64)

    params = dict(
        mass1=30.0, mass2=30.0,
        spin1x=0.2, spin1y=0.1, spin1z=0.3,
        spin2x=0.1, spin2y=0.2, spin2z=-0.1,
        prec_version=223,
        final_spin_mod=0,
    )

    hp_t, hc_t = TorchPhenomXP()(f=f_t, **params)
    hp_j, hc_j = JaxPhenomXP()(f=f_j, **params)

    assert torch.all(torch.isfinite(hp_t))
    assert torch.all(torch.isfinite(hc_t))
    assert jnp.all(jnp.isfinite(hp_j))
    assert jnp.all(jnp.isfinite(hc_j))

    max_p = np.max(np.abs(hp_t.numpy()))
    max_c = np.max(np.abs(hc_t.numpy()))
    assert max_p > 0.0
    assert max_c > 0.0

    diff_p = np.max(np.abs(hp_t.numpy() - np.array(hp_j))) / max_p
    diff_c = np.max(np.abs(hc_t.numpy() - np.array(hc_j))) / max_c

    assert diff_p < 1e-10, f"hp mismatch at q=1: {diff_p}"
    assert diff_c < 1e-10, f"hc mismatch at q=1: {diff_c}"


def test_spintaylor_beta_damping():
    """Verify damp_beta=True exponentially damps beta post-inspiral while preserving continuity."""
    m1, m2 = 30.0, 20.0
    total_m = m1 + m2
    m_sec = total_m * MSUN_SEC
    f_trans = 1.0 / (6.0**1.5 * PI * m_sec)

    f = np.linspace(20.0, 300.0, 200)
    f_t = torch.tensor(f, dtype=torch.float64)
    f_j = jnp.array(f, dtype=jnp.float64)

    st_params = dict(
        mass1=m1, mass2=m2,
        spin1x=0.2, spin1y=0.1, spin1z=0.3,
        spin2x=0.1, spin2y=0.2, spin2z=-0.1,
        inclination=0.4, phi_ref=0.0, f_ref=20.0,
    )

    # Undamped
    _, b_undamped, _, _, _, _ = torch_eval_spintaylor(f=f_t, damp_beta=False, **st_params)
    # Damped with rate 2.0
    _, b_damped, _, _, _, _ = torch_eval_spintaylor(
        f=f_t, damp_beta=True, damping_rate=2.0, **st_params
    )
    # JAX damped
    _, b_j, _, _, _, _ = jax_eval_spintaylor(
        f=f_j, damp_beta=True, damping_rate=2.0, **st_params
    )

    # 1. Pre-transition (f < f_trans): damped == undamped
    pre_mask = f < f_trans
    np.testing.assert_allclose(
        b_damped[pre_mask].numpy(),
        b_undamped[pre_mask].numpy(),
        rtol=1e-15, atol=1e-15,
    )

    # 2. Post-transition (f > f_trans): damped < undamped, exponential decay
    post_mask = f > f_trans + 10.0
    assert np.all(b_damped[post_mask].numpy() < b_undamped[post_mask].numpy())

    # Check exponential decay relation:
    # log(beta_damped / beta_undamped) = -damping_rate * (f - f_trans) / f_trans
    # Testing two frequencies f1=200 and f2=300 Hz in the post-inspiral regime:
    f_post = torch.tensor([200.0, 300.0], dtype=torch.float64)
    _, b_d_post, _, _, _, _ = torch_eval_spintaylor(
        f=f_post, damp_beta=True, damping_rate=2.0, **st_params
    )
    _, b_u_post, _, _, _, _ = torch_eval_spintaylor(
        f=f_post, damp_beta=False, damping_rate=2.0, **st_params
    )
    r1 = (b_d_post[0] / b_u_post[0]).item()
    r2 = (b_d_post[1] / b_u_post[1]).item()
    # Recovered f_trans from the exponential decay slope
    f_trans_recovered = -2.0 * (300.0 - 200.0) / (math.log(r2) - math.log(r1))
    # f_trans is 0.98 * f_meco ~ 93 Hz for total mass 50 Msun
    assert 80.0 < f_trans_recovered < 110.0
    expected_r1 = math.exp(-2.0 * (200.0 - f_trans_recovered) / f_trans_recovered)
    np.testing.assert_allclose(r1, expected_r1, rtol=1e-8)

    # 3. Continuity across f_trans
    f_near = torch.tensor(
        [f_trans_recovered - 1e-3, f_trans_recovered + 1e-3], dtype=torch.float64
    )
    _, b_near, _, _, _, _ = torch_eval_spintaylor(
        f=f_near, damp_beta=True, damping_rate=2.0, **st_params
    )
    diff_trans = abs(b_near[1].item() - b_near[0].item())
    assert diff_trans < 1e-3

    # 4. Parity between Torch and JAX damped beta
    np.testing.assert_allclose(np.array(b_j), b_damped.numpy(), rtol=1e-12, atol=1e-12)
