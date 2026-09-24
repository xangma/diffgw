"""Cross-framework parity test: torchwave vs jaxwave for IMRPhenomXPHM and Precession Engines."""

import math
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from diffgw_core.constants import PI, MSUN_SEC
import jaxwave
from jaxwave import (
    IMRPhenomXPHM as JaxPhenomXPHM,
    get_fd_waveform as jax_get_fd_waveform,
)
from torchwave import get_fd_waveform as torch_get_fd_waveform
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM as TorchPhenomXPHM
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM as TorchPhenomXHM
from jaxwave.approximants.imrphenom_xhm import IMRPhenomXHM as JaxPhenomXHM
import torchwave.approximants.precession_spintaylor as torch_st
import jaxwave.approximants.precession_spintaylor as jax_st
import torchwave.approximants.precession_pnr as torch_pnr
import jaxwave.approximants.precession_pnr as jax_pnr


@pytest.mark.parametrize('mode', [(2, 2), (3, 2), (3, 3)])
def test_xhm_observer_azimuth_parity(mode):
    f = np.array([25., 40., 80.])
    common = dict(mass1=40., mass2=20., inclination=.7,
                  phic=.3, f_ref=40., phi0=.4)
    torch_model = TorchPhenomXHM(modes=[mode])
    jax_model = JaxPhenomXHM(modes=[mode])
    spins = dict(chi1=.4, chi2=-.2)
    torch_output = torch_model(torch.tensor(f, dtype=torch.float64),
                               **common, **spins)
    jax_output = jax_model(jnp.asarray(f, dtype=jnp.float64),
                           **common, **spins)
    for actual, expected in zip(jax_output, torch_output):
        np.testing.assert_allclose(np.asarray(actual), expected.detach().numpy(),
                                   atol=1e-34, rtol=2e-11)


@pytest.mark.parametrize('mode', [(2, 2), (3, 2), (3, 3)])
def test_xphm_aligned_mode_parity(mode):
    f = np.array([25., 40., 80.])
    common = dict(mass1=40., mass2=20., inclination=.7, phic=.3,
                  f_ref=40., spin1x=0., spin1y=0., spin1z=.4,
                  spin2x=0., spin2y=0., spin2z=-.2)
    torch_output = TorchPhenomXPHM(modes=[mode])(
        torch.tensor(f, dtype=torch.float64), **common)
    jax_output = JaxPhenomXPHM(modes=[mode])(
        jnp.asarray(f, dtype=jnp.float64), **common)
    for actual, expected in zip(jax_output, torch_output):
        np.testing.assert_allclose(np.asarray(actual), expected.detach().numpy(),
                                   atol=1e-34, rtol=2e-11)


def test_chi_tot_perp_parity():
    """Verify compute_chi_tot_perp parity between torchwave and jaxwave."""
    m1, m2 = 35.0, 20.0
    s1x, s1y = 0.2, 0.3
    s2x, s2y = -0.1, 0.15

    c_torch = torch_st.compute_chi_tot_perp(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        torch.tensor(s1x, dtype=torch.float64),
        torch.tensor(s1y, dtype=torch.float64),
        torch.tensor(s2x, dtype=torch.float64),
        torch.tensor(s2y, dtype=torch.float64),
    ).item()

    c_jax = float(jax_st.compute_chi_tot_perp(m1, m2, s1x, s1y, s2x, s2y))
    np.testing.assert_allclose(c_jax, c_torch, atol=1e-14, rtol=1e-14)


def test_spintaylor_engine_parity():
    """Verify SpinTaylorT4Engine integration trajectory parity between torchwave and jaxwave."""
    m1, m2 = 35.0, 20.0
    s1 = [0.2, 0.3, 0.4]
    s2 = [0.1, -0.2, 0.2]

    t_eng = torch_st.SpinTaylorT4Engine(
        m1, m2,
        torch.tensor(s1, dtype=torch.float64),
        torch.tensor(s2, dtype=torch.float64),
        f_ref=20.0,
    )
    j_eng = jax_st.SpinTaylorT4Engine(m1, m2, s1, s2, f_ref=20.0)

    v0 = t_eng.v_ref.item()
    tv_nodes, tL, tS1, tS2 = t_eng.integrate_nodes(v_min=v0 * 0.8, v_max=0.35, num_steps=150)
    jv_nodes, jL, jS1, jS2 = j_eng.integrate_nodes(v_min=v0 * 0.8, v_max=0.35, num_steps=150)

    np.testing.assert_allclose(np.asarray(jv_nodes), tv_nodes.detach().numpy(), atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(jL), tL.detach().numpy(), atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(jS1), tS1.detach().numpy(), atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(jS2), tS2.detach().numpy(), atol=1e-13, rtol=1e-13)


def test_spintaylor_precession_angles_parity():
    """Verify evaluate_spintaylor_precession parity between torchwave and jaxwave."""
    f_arr = np.linspace(20.0, 300.0, 100)
    f_torch = torch.tensor(f_arr, dtype=torch.float64)
    f_jax = jnp.array(f_arr, dtype=jnp.float64)

    m1, m2 = 36.0, 29.0
    s1 = [0.3, 0.2, -0.1]
    s2 = [0.1, -0.2, 0.2]

    out_t = torch_st.evaluate_spintaylor_precession(
        f=f_torch, f_ref=20.0, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=0.6, phi_ref=0.2,
    )
    out_j = jax_st.evaluate_spintaylor_precession(
        f=f_jax, f_ref=20.0, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=0.6, phi_ref=0.2,
    )

    for val_j, val_t in zip(out_j, out_t):
        np.testing.assert_allclose(np.asarray(val_j), val_t.detach().numpy(), atol=1e-12, rtol=1e-12)


def test_pnr_precession_primitives_parity():
    """Verify compute_pnr_single_spin and coefficient fits parity."""
    m1, m2 = 36.0, 24.0
    s1 = [0.2, 0.1, 0.3]
    s2 = [0.1, -0.1, -0.2]

    chi_t, ct_t, ctf_t, chieff_t = torch_pnr.compute_pnr_single_spin(
        torch.tensor(m1, dtype=torch.float64), torch.tensor(m2, dtype=torch.float64),
        torch.tensor(s1[0], dtype=torch.float64), torch.tensor(s1[1], dtype=torch.float64), torch.tensor(s1[2], dtype=torch.float64),
        torch.tensor(s2[0], dtype=torch.float64), torch.tensor(s2[1], dtype=torch.float64), torch.tensor(s2[2], dtype=torch.float64),
    )
    chi_j, ct_j, ctf_j, chieff_j = jax_pnr.compute_pnr_single_spin(
        jnp.array(m1, dtype=jnp.float64), jnp.array(m2, dtype=jnp.float64),
        jnp.array(s1[0], dtype=jnp.float64), jnp.array(s1[1], dtype=jnp.float64), jnp.array(s1[2], dtype=jnp.float64),
        jnp.array(s2[0], dtype=jnp.float64), jnp.array(s2[1], dtype=jnp.float64), jnp.array(s2[2], dtype=jnp.float64),
    )

    np.testing.assert_allclose(float(chi_j), chi_t.item(), atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(float(ct_j), ct_t.item(), atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(float(ctf_j), ctf_t.item(), atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(float(chieff_j), chieff_t.item(), atol=1e-14, rtol=1e-14)

    q = m1 / m2
    eta = (m1 * m2) / ((m1 + m2) ** 2)

    a_t = torch_pnr.compute_pnr_alpha_coefficients(
        torch.tensor(eta, dtype=torch.float64), chi_t, ct_t, torch.tensor(q, dtype=torch.float64),
        torch.device("cpu"), torch.float64,
    )
    a_j = jax_pnr.compute_pnr_alpha_coefficients(
        jnp.array(eta, dtype=jnp.float64), chi_j, ct_j, jnp.array(q, dtype=jnp.float64),
    )
    for aj, at in zip(a_j, a_t):
        np.testing.assert_allclose(float(aj), at.item(), atol=1e-13, rtol=1e-13)

    b_t = torch_pnr.compute_pnr_beta_coefficients(
        torch.tensor(eta, dtype=torch.float64), chi_t, ct_t, ctf_t, torch.tensor(q, dtype=torch.float64),
        torch.device("cpu"), torch.float64,
    )
    b_j = jax_pnr.compute_pnr_beta_coefficients(
        jnp.array(eta, dtype=jnp.float64), chi_j, ct_j, ctf_j, jnp.array(q, dtype=jnp.float64),
    )
    for bj, bt in zip(b_j, b_t):
        np.testing.assert_allclose(float(bj), bt.item(), atol=1e-12, rtol=1e-12)


def test_pnr_precession_angles_parity():
    """Verify evaluate_pnr_precession parity between torchwave and jaxwave."""
    f_arr = np.linspace(20.0, 300.0, 150)
    f_torch = torch.tensor(f_arr, dtype=torch.float64)
    f_jax = jnp.array(f_arr, dtype=jnp.float64)

    m1, m2 = 36.0, 24.0
    s1 = [0.2, 0.1, 0.3]
    s2 = [0.1, -0.1, -0.2]

    pnr_t = torch_pnr.evaluate_pnr_precession(
        f=f_torch, f_ref=20.0, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=0.5, phi_ref=0.1,
    )
    pnr_j = jax_pnr.evaluate_pnr_precession(
        f=f_jax, f_ref=20.0, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=0.5, phi_ref=0.1,
    )

    for val_j, val_t in zip(pnr_j, pnr_t):
        np.testing.assert_allclose(np.asarray(val_j), val_t.detach().numpy(), atol=1e-12, rtol=1e-12)


def test_imrphenom_xphm_msa_parity():
    """Verify IMRPhenomXPHM with MSA precession parity against torchwave."""
    f_arr = np.linspace(25.0, 300.0, 100)
    f_torch = torch.tensor(f_arr, dtype=torch.float64)
    f_jax = jnp.array(f_arr, dtype=jnp.float64)

    m1, m2 = 35.0, 20.0
    s1 = [0.2, 0.1, 0.3]
    s2 = [0.1, -0.1, -0.2]
    inc = 0.7
    phic = 0.4
    fref = 20.0

    tx_msa = TorchPhenomXPHM(precession_method="msa")
    jx_msa = JaxPhenomXPHM(precession_method="msa")

    hp_t, hc_t = tx_msa(
        f=f_torch, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=inc, phic=phic, f_ref=fref,
    )
    hp_j, hc_j = jx_msa(
        f=f_jax, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=inc, phic=phic, f_ref=fref,
    )

    np.testing.assert_allclose(np.asarray(hp_j), hp_t.detach().numpy(), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.detach().numpy(), atol=1e-12, rtol=1e-12)


def test_imrphenom_xphm_spintaylor_parity():
    """Verify IMRPhenomXPHM with SpinTaylor precession parity against torchwave."""
    f_arr = np.linspace(25.0, 300.0, 100)
    f_torch = torch.tensor(f_arr, dtype=torch.float64)
    f_jax = jnp.array(f_arr, dtype=jnp.float64)

    m1, m2 = 35.0, 20.0
    s1 = [0.2, 0.1, 0.3]
    s2 = [0.1, -0.1, -0.2]
    inc = 0.7
    phic = 0.4
    fref = 20.0

    tx_st = TorchPhenomXPHM(precession_method="spintaylor", prec_version=320, final_spin_mod=2)
    jx_st = JaxPhenomXPHM(precession_method="spintaylor", prec_version=320, final_spin_mod=2)

    hp_t, hc_t = tx_st(
        f=f_torch, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=inc, phic=phic, f_ref=fref,
    )
    hp_j, hc_j = jx_st(
        f=f_jax, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=inc, phic=phic, f_ref=fref,
    )

    np.testing.assert_allclose(np.asarray(hp_j), hp_t.detach().numpy(), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.detach().numpy(), atol=1e-12, rtol=1e-12)


def test_imrphenom_xphm_pnr_parity():
    """Verify IMRPhenomXPHM with PNR precession parity against torchwave."""
    f_arr = np.linspace(25.0, 300.0, 100)
    f_torch = torch.tensor(f_arr, dtype=torch.float64)
    f_jax = jnp.array(f_arr, dtype=jnp.float64)

    m1, m2 = 35.0, 20.0
    s1 = [0.2, 0.1, 0.3]
    s2 = [0.1, -0.1, -0.2]
    inc = 0.7
    phic = 0.4
    fref = 20.0

    tx_pnr = TorchPhenomXPHM(precession_method="pnr", prec_version=330, final_spin_mod=2)
    jx_pnr = JaxPhenomXPHM(precession_method="pnr", prec_version=330, final_spin_mod=2)

    hp_t, hc_t = tx_pnr(
        f=f_torch, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=inc, phic=phic, f_ref=fref,
    )
    hp_j, hc_j = jx_pnr(
        f=f_jax, mass1=m1, mass2=m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        inclination=inc, phic=phic, f_ref=fref,
    )

    np.testing.assert_allclose(np.asarray(hp_j), hp_t.detach().numpy(), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.detach().numpy(), atol=1e-12, rtol=1e-12)


def test_imrphenom_xphm_pytree():
    """Verify PyTree flatten/unflatten behavior for IMRPhenomXPHM."""
    model = JaxPhenomXPHM(modes=((2, 2), (3, 3)), precession_method="spintaylor")
    leaves, treedef = jax.tree_util.tree_flatten(model)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, JaxPhenomXPHM)
    assert reconstructed.modes == ((2, 2), (3, 3))
    assert reconstructed.default_precession_method == "spintaylor"


def test_imrphenom_xphm_jit_vmap_grad():
    """Verify JIT compilation, vmap vectorization, and grad reverse-mode autodiff."""
    f = jnp.linspace(25.0, 150.0, 30)

    # 1. JIT
    @jax.jit
    def jit_eval(m1, m2):
        return JaxPhenomXPHM.evaluate(
            f=f, mass1=m1, mass2=m2,
            spin1x=0.2, spin1z=0.3,
            precession_method="msa",
        )

    hp, hc = jit_eval(35.0, 20.0)
    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert jnp.all(jnp.isfinite(hp))
    assert jnp.all(jnp.isfinite(hc))

    # 2. vmap
    m1_vec = jnp.array([32.0, 36.0])
    m2_vec = jnp.array([20.0, 22.0])
    vmap_eval = jax.vmap(lambda m1, m2: JaxPhenomXPHM.evaluate(
        f=f, mass1=m1, mass2=m2, spin1x=0.2, spin1z=0.3, precession_method="msa"
    ))
    hp_v, hc_v = vmap_eval(m1_vec, m2_vec)
    assert hp_v.shape == (2, 30)
    assert hc_v.shape == (2, 30)
    assert jnp.all(jnp.isfinite(hp_v))

    # 3. grad
    def loss_fn(m1):
        hp, hc = JaxPhenomXPHM.evaluate(
            f=f, mass1=m1, mass2=20.0, spin1x=0.2, spin1z=0.3, precession_method="msa"
        )
        return jnp.sum(jnp.abs(hp) ** 2)

    grad_fn = jax.grad(loss_fn)
    g = grad_fn(35.0)
    assert jnp.isfinite(g)


def test_waveforms_dispatch_parity():
    """Verify get_fd_waveform dispatches correctly for all XPHM aliases."""
    f = jnp.linspace(25.0, 200.0, 50)
    for approx in [
        "imrphenomxphm",
        "phenomxphm",
        "xphm",
        "imrphenomxphmspintaylor",
        "imrphenomxpnr",
        "xpnr",
        "pnr",
    ]:
        hp, hc = jax_get_fd_waveform(
            approximant=approx,
            mass1=35.0, mass2=20.0,
            spin1x=0.2, spin1y=0.1, spin1z=0.3,
            spin2x=0.1, spin2y=-0.1, spin2z=-0.2,
            inclination=0.5, phic=0.2,
            sample_frequencies=f,
        )
        assert hp.shape == f.shape
        assert hc.shape == f.shape
        assert jnp.all(jnp.isfinite(hp))
        assert jnp.all(jnp.isfinite(hc))
