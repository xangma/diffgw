"""Cross-framework parity test: torchwave vs jaxwave for IMRPhenomP / IMRPhenomPv2."""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from diffgw_core.constants import PI, MSUN_SEC
from torchwave.conversions import (
    component_masses_to_eta as torch_masses_to_eta,
    component_masses_to_delta as torch_masses_to_delta,
)
from jaxwave.conversions import (
    component_masses_to_eta as jax_masses_to_eta,
    component_masses_to_delta as jax_masses_to_delta,
)
from torchwave.approximants.imrphenomd import IMRPhenomD as TorchPhenomD
from jaxwave.approximants.imrphenomd import IMRPhenomD as JaxPhenomD
from torchwave.approximants.imrphenomp import (
    IMRPhenomP as TorchPhenomP,
    IMRPhenomPv2 as TorchPhenomPv2,
)
from jaxwave.approximants.imrphenomp import (
    IMRPhenomP as JaxPhenomP,
    IMRPhenomPv2 as JaxPhenomPv2,
    nnlo_angle_coefficients as jax_nnlo_coeffs,
    evaluate_angles as jax_eval_angles,
    orbital_angular_momentum as jax_orbital_angular_momentum,
    pv2_opening_angle as jax_pv2_opening_angle,
    _safe_in_plane_norm as jax_safe_in_plane_norm,
)


def test_imrphenomp_aliases():
    """Verify IMRPhenomPv2 aliases and module exports."""
    assert JaxPhenomPv2 is JaxPhenomP
    assert TorchPhenomPv2 is TorchPhenomP
    model = JaxPhenomP()
    leaves, treedef = jax.tree_util.tree_flatten(model)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, JaxPhenomP)


@pytest.mark.parametrize(
    "l_val, chip_val, m1_norm_val, s_par_val",
    [
        (0.8, 0.2, 35.0 / 60.0, 0.03),
        (0.5, 0.6, 0.7, -0.1),
        (1.2, 0.0, 0.5, 0.0),
        (0.3, 0.8, 0.8, 0.2),
    ],
)
def test_pv2_opening_angle_parity(l_val, chip_val, m1_norm_val, s_par_val):
    """Parity of pv2_opening_angle between torchwave and jaxwave."""
    # PyTorch
    l_t = torch.tensor(l_val, dtype=torch.float64)
    chip_t = torch.tensor(chip_val, dtype=torch.float64)
    m1_norm_t = torch.tensor(m1_norm_val, dtype=torch.float64)
    s_par_t = torch.tensor(s_par_val, dtype=torch.float64)
    beta_t = TorchPhenomP.pv2_opening_angle(l_t, chip_t, m1_norm_t, s_par_t)

    # JAX
    l_j = jnp.array(l_val, dtype=jnp.float64)
    chip_j = jnp.array(chip_val, dtype=jnp.float64)
    m1_norm_j = jnp.array(m1_norm_val, dtype=jnp.float64)
    s_par_j = jnp.array(s_par_val, dtype=jnp.float64)
    beta_j = jax_pv2_opening_angle(l_j, chip_j, m1_norm_j, s_par_j)
    beta_j_cls = JaxPhenomP.pv2_opening_angle(l_j, chip_j, m1_norm_j, s_par_j)

    np.testing.assert_allclose(float(beta_t), float(beta_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(float(beta_j), float(beta_j_cls), rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize(
    "m1, m2, chip, chi_l",
    [
        (35.0, 25.0, 0.3, 0.2),
        (50.0, 10.0, 0.6, -0.4),
        (30.0, 30.0, 0.0, 0.0),
        (60.0, 15.0, 0.8, 0.5),
    ],
)
def test_nnlo_angle_coefficients_parity(m1, m2, chip, chi_l):
    """Parity of nnlo_angle_coefficients between torchwave and jaxwave."""
    # Torch
    m1_t = torch.tensor(m1, dtype=torch.float64)
    m2_t = torch.tensor(m2, dtype=torch.float64)
    eta_t = torch_masses_to_eta(m1_t, m2_t)
    delta_t = torch_masses_to_delta(m1_t, m2_t)
    m1_norm_t = 0.5 * (1.0 + delta_t)
    chip_t = torch.tensor(chip, dtype=torch.float64)
    chil_t = torch.tensor(chi_l, dtype=torch.float64)
    coeffs_t = TorchPhenomP.nnlo_angle_coefficients(eta_t, delta_t, m1_norm_t, chip_t, chil_t)

    # JAX
    m1_j = jnp.array(m1, dtype=jnp.float64)
    m2_j = jnp.array(m2, dtype=jnp.float64)
    eta_j = jax_masses_to_eta(m1_j, m2_j)
    delta_j = jax_masses_to_delta(m1_j, m2_j)
    m1_norm_j = 0.5 * (1.0 + delta_j)
    chip_j = jnp.array(chip, dtype=jnp.float64)
    chil_j = jnp.array(chi_l, dtype=jnp.float64)
    coeffs_j = jax_nnlo_coeffs(eta_j, delta_j, m1_norm_j, chip_j, chil_j)
    coeffs_j_cls = JaxPhenomP.nnlo_angle_coefficients(eta_j, delta_j, m1_norm_j, chip_j, chil_j)

    for ct, cj, cj_cls in zip(coeffs_t, coeffs_j, coeffs_j_cls):
        np.testing.assert_allclose(float(ct), float(cj), rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(float(cj), float(cj_cls), rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("anchor", [True, False])
def test_evaluate_angles_parity(anchor):
    """Parity of evaluate_angles between torchwave and jaxwave."""
    m1, m2 = 36.0, 24.0
    chip, chi_l = 0.4, 0.1
    m1_t = torch.tensor(m1, dtype=torch.float64)
    m2_t = torch.tensor(m2, dtype=torch.float64)
    eta_t = torch_masses_to_eta(m1_t, m2_t)
    delta_t = torch_masses_to_delta(m1_t, m2_t)
    m1n_t = 0.5 * (1.0 + delta_t)
    chip_t = torch.tensor(chip, dtype=torch.float64)
    chil_t = torch.tensor(chi_l, dtype=torch.float64)
    coeffs_t = TorchPhenomP.nnlo_angle_coefficients(eta_t, delta_t, m1n_t, chip_t, chil_t)

    m1_j = jnp.array(m1, dtype=jnp.float64)
    m2_j = jnp.array(m2, dtype=jnp.float64)
    eta_j = jax_masses_to_eta(m1_j, m2_j)
    delta_j = jax_masses_to_delta(m1_j, m2_j)
    m1n_j = 0.5 * (1.0 + delta_j)
    chip_j = jnp.array(chip, dtype=jnp.float64)
    chil_j = jnp.array(chi_l, dtype=jnp.float64)
    coeffs_j = jax_nnlo_coeffs(eta_j, delta_j, m1n_j, chip_j, chil_j)

    m_tot = m1 + m2
    f_arr = np.linspace(20.0, 300.0, 50)
    v_arr = (np.pi * m_tot * MSUN_SEC * f_arr) ** (1.0 / 3.0)
    v_ref_val = (np.pi * m_tot * MSUN_SEC * 20.0) ** (1.0 / 3.0)

    v_t = torch.tensor(v_arr, dtype=torch.float64)
    v_ref_t = torch.tensor(v_ref_val, dtype=torch.float64)
    alpha_t, eps_t = TorchPhenomP.evaluate_angles(v_t, v_ref_t, *coeffs_t, anchor_reference=anchor)

    v_j = jnp.asarray(v_arr)
    v_ref_j = jnp.asarray(v_ref_val)
    alpha_j, eps_j = jax_eval_angles(v_j, v_ref_j, *coeffs_j, anchor_reference=anchor)
    alpha_j_cls, eps_j_cls = JaxPhenomP.evaluate_angles(v_j, v_ref_j, *coeffs_j, anchor_reference=anchor)

    np.testing.assert_allclose(alpha_t.numpy(), np.asarray(alpha_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(eps_t.numpy(), np.asarray(eps_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(alpha_j), np.asarray(alpha_j_cls), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(np.asarray(eps_j), np.asarray(eps_j_cls), rtol=1e-15, atol=1e-15)


def test_imrphenomp_aligned_spin_limit():
    """Verify that in the aligned-spin limit (chip=0), IMRPhenomP reduces to IMRPhenomD."""
    f_np = np.linspace(20.0, 300.0, 100, dtype=np.float64)
    f_j = jnp.asarray(f_np)
    m1, m2 = 25.0, 15.0
    s1z, s2z = 0.3, -0.2

    # JAX PhenomD face-on
    hp_d, hc_d = JaxPhenomD.evaluate(
        f=f_j, mass1=m1, mass2=m2, chi1=s1z, chi2=s2z, distance=100.0, inclination=0.0
    )

    # JAX PhenomP face-on with zero in-plane spins
    hp_p, hc_p = JaxPhenomP.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=s1z,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=s2z,
        distance=100.0,
        inclination=0.0,
    )

    np.testing.assert_allclose(
        np.abs(np.asarray(hp_p)), np.abs(np.asarray(hp_d)), rtol=1e-5, atol=1e-25
    )
    np.testing.assert_allclose(
        np.abs(np.asarray(hc_p)), np.abs(np.asarray(hc_d)), rtol=1e-5, atol=1e-25
    )


@pytest.mark.parametrize(
    "m1, m2, s1, s2, inc",
    [
        (30.0, 30.0, (0.2, 0.1, 0.3), (0.0, 0.2, -0.1), 0.4),
        (40.0, 20.0, (0.3, -0.2, 0.1), (-0.1, 0.3, 0.2), 0.7),
        (60.0, 15.0, (0.4, 0.0, -0.2), (0.1, -0.2, 0.1), 0.5),
        (35.0, 25.0, (0.5, 0.2, 0.0), (0.0, 0.0, 0.0), 0.6),
        (20.0, 35.0, (0.1, 0.2, -0.1), (0.3, -0.1, 0.2), 0.3),  # inverted masses
    ],
)
def test_imrphenomp_waveform_parity(m1, m2, s1, s2, inc):
    """Full 3D precessing waveform parity comparing Torch vs JAX across configurations."""
    f_np = np.linspace(20.0, 400.0, 200, dtype=np.float64)
    f_t = torch.tensor(f_np, dtype=torch.float64)
    f_j = jnp.asarray(f_np)

    # PyTorch
    hp_t, hc_t = TorchPhenomP.evaluate(
        f=f_t,
        mass1=m1,
        mass2=m2,
        spin1x=s1[0],
        spin1y=s1[1],
        spin1z=s1[2],
        spin2x=s2[0],
        spin2y=s2[1],
        spin2z=s2[2],
        distance=120.0,
        inclination=inc,
        phic=0.3,
        tc=0.015,
        f_ref=20.0,
    )

    # JAX static evaluate
    hp_j, hc_j = JaxPhenomP.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        spin1x=s1[0],
        spin1y=s1[1],
        spin1z=s1[2],
        spin2x=s2[0],
        spin2y=s2[1],
        spin2z=s2[2],
        distance=120.0,
        inclination=inc,
        phic=0.3,
        tc=0.015,
        f_ref=20.0,
    )

    # JAX instance call
    model = JaxPhenomP()
    hp_inst, hc_inst = model(
        f=f_j,
        mass1=m1,
        mass2=m2,
        spin1x=s1[0],
        spin1y=s1[1],
        spin1z=s1[2],
        spin2x=s2[0],
        spin2y=s2[1],
        spin2z=s2[2],
        distance=120.0,
        inclination=inc,
        phic=0.3,
        tc=0.015,
        f_ref=20.0,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(hp_j), np.asarray(hp_inst), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(np.asarray(hc_j), np.asarray(hc_inst), rtol=1e-15, atol=1e-15)


def test_imrphenomp_jax_jit_and_autodiff():
    """Verify jax.jit compilation, autodiff w.r.t masses and spins, and jax.vmap."""
    f = jnp.linspace(20.0, 300.0, 128, dtype=jnp.float64)

    @jax.jit
    def scalar_loss(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z):
        hp, hc = JaxPhenomP.evaluate(
            f=f,
            mass1=m1,
            mass2=m2,
            spin1x=s1x,
            spin1y=s1y,
            spin1z=s1z,
            spin2x=s2x,
            spin2y=s2y,
            spin2z=s2z,
            distance=100.0,
            inclination=0.5,
            f_ref=20.0,
        )
        return jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2)

    # 1. JIT execution
    val = scalar_loss(35.0, 25.0, 0.2, 0.1, 0.3, -0.1, 0.2, 0.1)
    assert jnp.isfinite(val)

    # 2. Autodiff w.r.t masses and 3D spins in precessing case
    grad_fn = jax.jit(jax.grad(scalar_loss, argnums=(0, 1, 2, 3, 4, 5, 6, 7)))
    grads_prec = grad_fn(35.0, 25.0, 0.2, 0.1, 0.3, -0.1, 0.2, 0.1)
    for i, g in enumerate(grads_prec):
        assert jnp.isfinite(g), f"Precessing gradient index {i} is not finite: {g}"

    # 3. Autodiff in aligned-spin limit (in-plane spins = 0)
    grads_aligned = grad_fn(35.0, 25.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1)
    for i, g in enumerate(grads_aligned):
        assert jnp.isfinite(g), f"Aligned-limit gradient index {i} is not finite: {g}"

    # 4. jax.vmap vectorization over batches
    b_m1 = jnp.array([30.0, 40.0, 25.0])
    b_m2 = jnp.array([20.0, 20.0, 15.0])
    b_s1x = jnp.array([0.1, 0.2, 0.0])
    b_s1z = jnp.array([0.2, -0.1, 0.3])

    def single_eval(m1, m2, s1x, s1z):
        return JaxPhenomP.evaluate(
            f=f,
            mass1=m1,
            mass2=m2,
            spin1x=s1x,
            spin1z=s1z,
            inclination=0.4,
        )

    vmapped_fn = jax.jit(jax.vmap(single_eval))
    hp_batch, hc_batch = vmapped_fn(b_m1, b_m2, b_s1x, b_s1z)

    assert hp_batch.shape == (3, len(f))
    assert hc_batch.shape == (3, len(f))

    for b in range(3):
        hp_single, hc_single = single_eval(b_m1[b], b_m2[b], b_s1x[b], b_s1z[b])
        np.testing.assert_allclose(hp_batch[b], hp_single, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(hc_batch[b], hc_single, rtol=1e-14, atol=1e-15)
