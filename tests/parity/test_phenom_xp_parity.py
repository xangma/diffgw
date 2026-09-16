"""Cross-framework parity test: torchwave vs jaxwave for Precession MSA & IMRPhenomXP."""

import math
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from diffgw_core.constants import PI, MSUN_SEC
import jaxwave
from jaxwave import IMRPhenomXP as JaxPhenomXP, get_fd_waveform as jax_get_fd_waveform
from torchwave import get_fd_waveform as torch_get_fd_waveform
from torchwave.approximants.imrphenom_xp import IMRPhenomXP as TorchPhenomXP
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS as TorchPhenomXAS
from jaxwave.approximants.imrphenom_xas import IMRPhenomXAS as JaxPhenomXAS
import torchwave.approximants.precession_msa as torch_msa
import jaxwave.approximants.precession_msa as jax_msa
from jaxwave.approximants.imrphenom_xp import wigner_small_d_matrix as jax_wigner_d
from torchwave.approximants.imrphenom_xphm import wigner_small_d_matrix as torch_wigner_d


def test_elliptic_and_jacobi_parity():
    """Verify parity of elliptic_f and jacobi_sn_cn_dn between torchwave and jaxwave."""
    phi_vals = np.linspace(-math.pi / 2 + 1e-4, math.pi / 2 - 1e-4, 25)
    m_vals = [0.0, 0.1, 0.5, 0.9, 0.99, 0.999]

    for m in m_vals:
        phi_t = torch.tensor(phi_vals, dtype=torch.float64)
        m_t = torch.tensor(m, dtype=torch.float64)
        f_torch = torch_msa.elliptic_f(phi_t, m_t).numpy()

        phi_j = jnp.array(phi_vals, dtype=jnp.float64)
        m_j = jnp.array(m, dtype=jnp.float64)
        f_jax = np.array(jax_msa.elliptic_f(phi_j, m_j))

        np.testing.assert_allclose(f_jax, f_torch, atol=1e-14, rtol=1e-14)

    # Jacobi elliptic functions
    u_vals = np.linspace(-3.0, 3.0, 30)
    for m in [0.0, 0.25, 0.5, 0.75, 0.95]:
        u_t = torch.tensor(u_vals, dtype=torch.float64)
        m_t = torch.tensor(m, dtype=torch.float64)
        sn_t, cn_t, dn_t = torch_msa.jacobi_sn_cn_dn(u_t, m_t)

        u_j = jnp.array(u_vals, dtype=jnp.float64)
        m_j = jnp.array(m, dtype=jnp.float64)
        sn_j, cn_j, dn_j = jax_msa.jacobi_sn_cn_dn(u_j, m_j)

        np.testing.assert_allclose(np.array(sn_j), sn_t.numpy(), atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(np.array(cn_j), cn_t.numpy(), atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(np.array(dn_j), dn_t.numpy(), atol=1e-14, rtol=1e-14)


def test_wigner_small_d_parity():
    """Verify parity of Wigner small-d matrix for l in {2, 3, 4}."""
    beta_vals = np.linspace(0.0, math.pi, 20)
    beta_t = torch.tensor(beta_vals, dtype=torch.float64)
    beta_j = jnp.array(beta_vals, dtype=jnp.float64)

    for l in (2, 3, 4):
        d_torch = torch_wigner_d(l, beta_t).numpy()
        d_jax = np.array(jax_wigner_d(l, beta_j))
        np.testing.assert_allclose(d_jax, d_torch, atol=1e-14, rtol=1e-14)


def test_precession_primitives_parity():
    """Verify compute_effective_chip, compute_l_3pn, compute_msa_final_spin parity."""
    m1, m2 = 35.0, 20.0
    s1x, s1y, s1z = 0.2, 0.15, 0.1
    s2x, s2y, s2z = 0.1, -0.05, -0.2

    # Chip
    chip_t = torch_msa.compute_effective_chip(
        torch.tensor(m1, dtype=torch.float64), torch.tensor(m2, dtype=torch.float64),
        torch.tensor(s1x, dtype=torch.float64), torch.tensor(s1y, dtype=torch.float64),
        torch.tensor(s2x, dtype=torch.float64), torch.tensor(s2y, dtype=torch.float64)
    ).item()
    chip_j = float(jax_msa.compute_effective_chip(
        jnp.array(m1, dtype=jnp.float64), jnp.array(m2, dtype=jnp.float64),
        jnp.array(s1x, dtype=jnp.float64), jnp.array(s1y, dtype=jnp.float64),
        jnp.array(s2x, dtype=jnp.float64), jnp.array(s2y, dtype=jnp.float64)
    ))
    np.testing.assert_allclose(chip_j, chip_t, atol=1e-15, rtol=1e-15)

    # L_3pn
    v = 0.2
    total = m1 + m2
    eta = (m1 * m2) / (total**2)
    delta = (m1 - m2) / total
    s_ell = ((m1/total)**2)*s1z + ((m2/total)**2)*s2z
    sigma_ell = (m2/total)*s2z - (m1/total)*s1z

    l_t = torch_msa.compute_l_3pn(
        torch.tensor(v, dtype=torch.float64), torch.tensor(eta, dtype=torch.float64),
        torch.tensor(delta, dtype=torch.float64), torch.tensor(s_ell, dtype=torch.float64),
        torch.tensor(sigma_ell, dtype=torch.float64)
    ).item()
    l_j = float(jax_msa.compute_l_3pn(
        jnp.array(v, dtype=jnp.float64), jnp.array(eta, dtype=jnp.float64),
        jnp.array(delta, dtype=jnp.float64), jnp.array(s_ell, dtype=jnp.float64),
        jnp.array(sigma_ell, dtype=jnp.float64)
    ))
    np.testing.assert_allclose(l_j, l_t, atol=1e-15, rtol=1e-15)

    # Final spin
    v0 = (PI * total * MSUN_SEC * 20.0)**(1/3)
    sfin_t = torch_msa.compute_msa_final_spin(
        torch.tensor(v0, dtype=torch.float64), torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64), torch.tensor(s1x, dtype=torch.float64),
        torch.tensor(s1y, dtype=torch.float64), torch.tensor(s1z, dtype=torch.float64),
        torch.tensor(s2x, dtype=torch.float64), torch.tensor(s2y, dtype=torch.float64),
        torch.tensor(s2z, dtype=torch.float64)
    ).item()
    sfin_j = float(jax_msa.compute_msa_final_spin(
        jnp.array(v0, dtype=jnp.float64), jnp.array(m1, dtype=jnp.float64),
        jnp.array(m2, dtype=jnp.float64), jnp.array(s1x, dtype=jnp.float64),
        jnp.array(s1y, dtype=jnp.float64), jnp.array(s1z, dtype=jnp.float64),
        jnp.array(s2x, dtype=jnp.float64), jnp.array(s2y, dtype=jnp.float64),
        jnp.array(s2z, dtype=jnp.float64)
    ))
    np.testing.assert_allclose(sfin_j, sfin_t, atol=1e-15, rtol=1e-15)


def test_evaluate_msa_precession_angles_parity():
    """Verify evaluate_msa_precession angles (alpha, beta, gamma, theta_jn, zeta) parity."""
    f = np.linspace(25.0, 250.0, 50)
    m1, m2 = 36.0, 22.0
    s1x, s1y, s1z = 0.25, 0.15, 0.2
    s2x, s2y, s2z = -0.1, 0.2, -0.15

    m_sec = (m1 + m2) * MSUN_SEC
    v = (PI * m_sec * f) ** (1.0 / 3.0)
    v0 = (PI * m_sec * 20.0) ** (1.0 / 3.0)

    res_t = torch_msa.evaluate_msa_precession(
        v=torch.tensor(v, dtype=torch.float64),
        v0=torch.tensor(v0, dtype=torch.float64),
        m1=torch.tensor(m1, dtype=torch.float64),
        m2=torch.tensor(m2, dtype=torch.float64),
        s1x=torch.tensor(s1x, dtype=torch.float64),
        s1y=torch.tensor(s1y, dtype=torch.float64),
        s1z=torch.tensor(s1z, dtype=torch.float64),
        s2x=torch.tensor(s2x, dtype=torch.float64),
        s2y=torch.tensor(s2y, dtype=torch.float64),
        s2z=torch.tensor(s2z, dtype=torch.float64),
        iota=torch.tensor(0.5, dtype=torch.float64),
        phi_ref=torch.tensor(0.2, dtype=torch.float64),
        prec_version=223,
        convention=1,
        return_carrier_phase=True,
    )
    res_j = jax_msa.evaluate_msa_precession(
        v=jnp.array(v, dtype=jnp.float64),
        v0=jnp.array(v0, dtype=jnp.float64),
        m1=jnp.array(m1, dtype=jnp.float64),
        m2=jnp.array(m2, dtype=jnp.float64),
        s1x=jnp.array(s1x, dtype=jnp.float64),
        s1y=jnp.array(s1y, dtype=jnp.float64),
        s1z=jnp.array(s1z, dtype=jnp.float64),
        s2x=jnp.array(s2x, dtype=jnp.float64),
        s2y=jnp.array(s2y, dtype=jnp.float64),
        s2z=jnp.array(s2z, dtype=jnp.float64),
        iota=jnp.array(0.5, dtype=jnp.float64),
        phi_ref=jnp.array(0.2, dtype=jnp.float64),
        prec_version=223,
        convention=1,
        return_carrier_phase=True,
    )

    for val_j, val_t in zip(res_j, res_t):
        np.testing.assert_allclose(np.array(val_j), val_t.numpy(), atol=1e-13, rtol=1e-13)


def test_phenom_xp_aligned_spin_parity():
    """Verify IMRPhenomXP matches IMRPhenomXAS and matches PyTorch in aligned limit."""
    f = np.linspace(25.0, 300.0, 64)
    m1, m2 = 36.0, 22.0
    s1z, s2z = 0.35, -0.25
    inc = 0.65
    dist = 150.0

    torch_xp = TorchPhenomXP()
    jax_xp = JaxPhenomXP()
    jax_xas = JaxPhenomXAS()

    hp_t, hc_t = torch_xp(
        f=torch.tensor(f, dtype=torch.float64),
        mass1=m1, mass2=m2,
        spin1z=s1z, spin2z=s2z,
        distance=dist, inclination=inc,
    )
    hp_j, hc_j = jax_xp(
        f=jnp.array(f, dtype=jnp.float64),
        mass1=m1, mass2=m2,
        spin1z=s1z, spin2z=s2z,
        distance=dist, inclination=inc,
    )
    hp_xas, hc_xas = jax_xas(
        f=jnp.array(f, dtype=jnp.float64),
        mass1=m1, mass2=m2,
        chi1=s1z, chi2=s2z,
        distance=dist, inclination=inc,
    )

    # Collapse to XAS
    np.testing.assert_allclose(np.array(hp_j), np.array(hp_xas), atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(np.array(hc_j), np.array(hc_xas), atol=1e-14, rtol=1e-14)

    # Parity with PyTorch
    np.testing.assert_allclose(np.array(hp_j), hp_t.numpy(), atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(np.array(hc_j), hc_t.numpy(), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize(
    "m1, m2, s1, s2, inc, prec_method",
    [
        (35.0, 20.0, (0.2, 0.15, 0.1), (0.1, 0.05, -0.1), 0.5, "msa"),
        (45.0, 15.0, (0.3, 0.0, 0.2), (0.0, 0.2, -0.1), 0.8, "msa"),
        (30.0, 25.0, (0.1, 0.2, 0.0), (0.15, -0.1, 0.1), 0.3, "nnlo"),
        (50.0, 10.0, (0.25, -0.1, 0.15), (-0.1, 0.15, -0.2), 0.6, "nnlo"),
    ]
)
def test_phenom_xp_precessing_parity(m1, m2, s1, s2, inc, prec_method):
    """Verify full waveform parity between Torch and JAX for precessing configurations."""
    f = np.linspace(25.0, 250.0, 64)
    f_torch = torch.tensor(f, dtype=torch.float64)
    f_jax = jnp.array(f, dtype=jnp.float64)

    torch_xp = TorchPhenomXP()
    jax_xp = JaxPhenomXP()

    hp_t, hc_t = torch_xp(
        f=f_torch, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=inc, precession_method=prec_method,
        _force_legacy=True,
    )
    hp_j, hc_j = jax_xp(
        f=f_jax, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=inc, precession_method=prec_method,
    )

    max_amp = np.max(np.abs(hp_t.numpy()))
    diff_p = np.max(np.abs(np.array(hp_j) - hp_t.numpy())) / max_amp
    diff_c = np.max(np.abs(np.array(hc_j) - hc_t.numpy())) / max_amp

    assert diff_p < 1e-10, f"h+ rel mismatch {diff_p} exceeds 1e-10 for {prec_method}"
    assert diff_c < 1e-10, f"hx rel mismatch {diff_c} exceeds 1e-10 for {prec_method}"


def test_phenom_xp_conventions():
    """Verify parity across conventions (0, 1, 5, 6, 7)."""
    f = np.linspace(25.0, 200.0, 32)
    m1, m2 = 35.0, 22.0
    s1 = (0.2, 0.1, 0.15)
    s2 = (0.1, -0.05, -0.1)

    for conv in (0, 1, 5, 6, 7):
        hp_t, hc_t = TorchPhenomXP()(
            f=torch.tensor(f, dtype=torch.float64),
            mass1=m1, mass2=m2,
            spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
            spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
            convention=conv,
            _force_legacy=True,
        )
        hp_j, hc_j = JaxPhenomXP()(
            f=jnp.array(f, dtype=jnp.float64),
            mass1=m1, mass2=m2,
            spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
            spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
            convention=conv,
        )
        max_amp = np.max(np.abs(hp_t.numpy()))
        diff_p = np.max(np.abs(np.array(hp_j) - hp_t.numpy())) / max_amp
        assert diff_p < 1e-10, f"Convention {conv} mismatch: {diff_p}"


def test_phenom_xp_get_fd_waveform_dispatch():
    """Verify get_fd_waveform dispatches IMRPhenomXP with exact parity."""
    f = np.linspace(25.0, 200.0, 32)
    m1, m2 = 32.0, 24.0
    s1x, s1y, s1z = 0.2, 0.1, 0.1
    s2x, s2y, s2z = 0.1, 0.05, -0.1

    hp_t, hc_t = torch_get_fd_waveform(
        approximant="IMRPhenomXP",
        mass1=m1, mass2=m2,
        spin1x=s1x, spin1y=s1y, spin1z=s1z,
        spin2x=s2x, spin2y=s2y, spin2z=s2z,
        sample_frequencies=torch.tensor(f, dtype=torch.float64),
        _force_legacy=True,
    )
    hp_j, hc_j = jax_get_fd_waveform(
        approximant="IMRPhenomXP",
        mass1=m1, mass2=m2,
        spin1x=s1x, spin1y=s1y, spin1z=s1z,
        spin2x=s2x, spin2y=s2y, spin2z=s2z,
        sample_frequencies=jnp.array(f, dtype=jnp.float64),
    )

    max_amp = np.max(np.abs(hp_t.numpy()))
    diff_p = np.max(np.abs(np.array(hp_j) - hp_t.numpy())) / max_amp
    assert diff_p < 1e-10, f"get_fd_waveform mismatch: {diff_p}"


def test_phenom_xp_pytree_and_jit():
    """Verify PyTree registration, compilation with @jax.jit, and execution."""
    model = JaxPhenomXP()
    leaves, treedef = jax.tree_util.tree_flatten(model)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, JaxPhenomXP)

    f = jnp.linspace(25.0, 200.0, 32)
    jit_eval = jax.jit(reconstructed.evaluate)

    hp_jit, hc_jit = jit_eval(
        f=f, mass1=35.0, mass2=20.0,
        spin1x=0.2, spin1y=0.15, spin1z=0.1,
        spin2x=0.1, spin2y=0.05, spin2z=-0.1,
    )
    hp_eager, hc_eager = model.evaluate(
        f=f, mass1=35.0, mass2=20.0,
        spin1x=0.2, spin1y=0.15, spin1z=0.1,
        spin2x=0.1, spin2y=0.05, spin2z=-0.1,
    )

    np.testing.assert_allclose(np.array(hp_jit), np.array(hp_eager), atol=1e-15, rtol=1e-15)
    np.testing.assert_allclose(np.array(hc_jit), np.array(hc_eager), atol=1e-15, rtol=1e-15)


def test_phenom_xp_autodiff_all_8_parameters():
    """Verify reverse-mode autodiff with jax.grad w.r.t all 8 parameters."""
    f = jnp.linspace(25.0, 200.0, 32)

    def loss(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z):
        hp, hc = JaxPhenomXP.evaluate(
            f=f, mass1=m1, mass2=m2,
            spin1x=s1x, spin1y=s1y, spin1z=s1z,
            spin2x=s2x, spin2y=s2y, spin2z=s2z,
        )
        return jnp.sum(jnp.abs(hp)**2 + jnp.abs(hc)**2)

    args = (35.0, 20.0, 0.20, 0.15, 0.10, 0.10, 0.05, -0.10)
    grads = jax.grad(loss, argnums=range(8))(*args)

    param_names = ["m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z"]
    for name, g in zip(param_names, grads):
        val = float(g)
        assert not math.isnan(val), f"Gradient for {name} is NaN"
        assert abs(val) > 0.0, f"Gradient for {name} is zero"


def test_phenom_xp_vmap():
    """Verify vectorized evaluation with jax.vmap across parameter batches."""
    f = jnp.linspace(25.0, 200.0, 32)
    B = 3

    m1_b = jnp.array([32.0, 40.0, 28.0])
    m2_b = jnp.array([20.0, 25.0, 16.0])
    s1x_b = jnp.array([0.2, 0.1, 0.3])
    s1y_b = jnp.array([0.1, 0.2, 0.0])
    s1z_b = jnp.array([0.2, -0.1, 0.1])
    s2x_b = jnp.array([0.0, 0.1, 0.0])
    s2y_b = jnp.array([0.0, 0.0, 0.1])
    s2z_b = jnp.array([-0.1, 0.0, 0.2])

    vmapped_fn = jax.vmap(
        lambda m1, m2, s1x, s1y, s1z, s2x, s2y, s2z: JaxPhenomXP.evaluate(
            f=f, mass1=m1, mass2=m2,
            spin1x=s1x, spin1y=s1y, spin1z=s1z,
            spin2x=s2x, spin2y=s2y, spin2z=s2z,
        )
    )

    hp_vmap, hc_vmap = vmapped_fn(m1_b, m2_b, s1x_b, s1y_b, s1z_b, s2x_b, s2y_b, s2z_b)
    assert hp_vmap.shape == (B, 32)
    assert hc_vmap.shape == (B, 32)

    for i in range(B):
        hp_single, hc_single = JaxPhenomXP.evaluate(
            f=f, mass1=m1_b[i], mass2=m2_b[i],
            spin1x=s1x_b[i], spin1y=s1y_b[i], spin1z=s1z_b[i],
            spin2x=s2x_b[i], spin2y=s2y_b[i], spin2z=s2z_b[i],
        )
        np.testing.assert_allclose(np.array(hp_vmap[i]), np.array(hp_single), atol=1e-15, rtol=1e-15)
        np.testing.assert_allclose(np.array(hc_vmap[i]), np.array(hc_single), atol=1e-15, rtol=1e-15)
