"""Cross-framework parity test: torchwave vs jaxwave for IMRPhenomXAS."""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.imrphenom_xas import (
    IMRPhenomXAS as TorchPhenomXAS,
    XASCarrierState as TorchCarrierState,
    XASParameterFits as TorchParameterFits,
)
from jaxwave.approximants.imrphenom_xas import (
    IMRPhenomXAS as JaxPhenomXAS,
    XASCarrierState as JaxCarrierState,
    XASParameterFits as JaxParameterFits,
)
from jaxwave import IMRPhenomXAS as RootJaxPhenomXAS


def test_imrphenom_xas_exports_and_pytree():
    """Verify IMRPhenomXAS exports and PyTree flatten/unflatten behavior."""
    assert JaxPhenomXAS is RootJaxPhenomXAS
    model = JaxPhenomXAS()
    leaves, treedef = jax.tree_util.tree_flatten(model)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, JaxPhenomXAS)

    # Also test XASCarrierState PyTree registration
    st = JaxPhenomXAS.build_state(mass1=35.0, mass2=25.0, chi1=0.2, chi2=-0.1, f_ref=20.0)
    st_leaves, st_treedef = jax.tree_util.tree_flatten(st)
    st_rec = jax.tree_util.tree_unflatten(st_treedef, st_leaves)
    assert isinstance(st_rec, JaxCarrierState)
    f_test = jnp.linspace(20.0, 100.0, 10, dtype=jnp.float64)
    hp1, hc1 = st(f_test)
    hp2, hc2 = st_rec(f_test)
    np.testing.assert_allclose(np.asarray(hp1), np.asarray(hp2), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(np.asarray(hc1), np.asarray(hc2), rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize(
    "q, m_tot, chi1, chi2",
    [
        # q = 1
        (1.0, 60.0, 0.0, 0.0),
        (1.0, 60.0, 0.6, 0.6),
        (1.0, 60.0, -0.8, 0.8),
        # q = 2
        (2.0, 60.0, 0.3, -0.2),
        (2.0, 60.0, -0.5, 0.7),
        (2.0, 60.0, 0.8, -0.8),
        # q = 4
        (4.0, 75.0, 0.5, -0.4),
        (4.0, 75.0, -0.7, 0.3),
        (4.0, 75.0, 0.8, 0.8),
        # q = 8
        (8.0, 90.0, 0.0, 0.0),
        (8.0, 90.0, 0.8, -0.8),
        (8.0, 90.0, -0.6, 0.4),
    ],
)
def test_xas_waveform_parity(q, m_tot, chi1, chi2):
    """Parity of full waveform polarizations across mass ratios q in {1, 2, 4, 8} and aligned spins."""
    m1 = (q / (1.0 + q)) * m_tot
    m2 = (1.0 / (1.0 + q)) * m_tot

    f_np = np.linspace(20.0, 512.0, 400, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    model_t = TorchPhenomXAS()
    model_j = JaxPhenomXAS()

    hp_t, hc_t = model_t(
        sample_frequencies=f_t,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=100.0,
        inclination=0.4,
        phi_c=0.2,
        t_c=0.01,
        f_ref=20.0,
    )

    hp_j, hc_j = model_j(
        sample_frequencies=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=100.0,
        inclination=0.4,
        phi_c=0.2,
        t_c=0.01,
        f_ref=20.0,
    )

    scale_p = np.max(np.abs(hp_t.numpy()))
    scale_c = np.max(np.abs(hc_t.numpy()))

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=5e-12, atol=1e-12 * scale_p)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=5e-12, atol=1e-12 * scale_c)


@pytest.mark.parametrize(
    "m1, m2, chi1, chi2",
    [
        (30.0, 30.0, 0.0, 0.0),
        (40.0, 20.0, 0.4, -0.3),
        (60.0, 15.0, -0.6, 0.5),
        (80.0, 10.0, 0.7, -0.7),
    ],
)
def test_xas_regime_amplitude_phase_parity(m1, m2, chi1, chi2):
    """Verify amplitude and phase parity across inspiral, intermediate, and ringdown regimes."""
    model_t = TorchPhenomXAS()
    model_j = JaxPhenomXAS()

    st_t = model_t.build_state(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, f_ref=20.0)
    st_j = model_j.build_state(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, f_ref=20.0)

    # Verify precomputed transition boundaries match
    f_trans_amp_t = float(st_t.f_trans_amp)
    f_trans_amp_j = float(st_j.f_trans_amp)
    np.testing.assert_allclose(f_trans_amp_t, f_trans_amp_j, rtol=1e-13)

    f_peak_t = float(st_t.f_peak)
    f_peak_j = float(st_j.f_peak)
    np.testing.assert_allclose(f_peak_t, f_peak_j, rtol=1e-13)

    f_trans_ph_t = float(st_t.f_trans_phase)
    f_trans_ph_j = float(st_j.f_trans_phase)
    np.testing.assert_allclose(f_trans_ph_t, f_trans_ph_j, rtol=1e-13)

    f_join_rd_t = float(st_t.f_join_rd)
    f_join_rd_j = float(st_j.f_join_rd)
    np.testing.assert_allclose(f_join_rd_t, f_join_rd_j, rtol=1e-13)

    m_sec = (m1 + m2) * 4.925490947641267e-06

    # 1. Inspiral regime test (frequencies below f_trans_amp and f_trans_phase)
    f_ins_max = min(f_trans_amp_t, f_trans_ph_t) / m_sec * 0.95
    f_ins_np = np.linspace(20.0, f_ins_max, 50, dtype=np.float64)
    hp_t_ins, hc_t_ins, ph_t_ins = st_t.eval_waveform(torch.from_numpy(f_ins_np), return_phase=True)
    hp_j_ins, hc_j_ins, ph_j_ins = st_j.eval_waveform(jnp.asarray(f_ins_np), return_phase=True)

    np.testing.assert_allclose(hp_t_ins.numpy(), np.asarray(hp_j_ins), rtol=1e-12)
    np.testing.assert_allclose(hc_t_ins.numpy(), np.asarray(hc_j_ins), rtol=1e-12)
    np.testing.assert_allclose(ph_t_ins.numpy(), np.asarray(ph_j_ins), rtol=1e-12)

    # 2. Intermediate regime test (frequencies between transitions)
    f_int_min = max(f_trans_amp_t, f_trans_ph_t) / m_sec * 1.02
    f_int_max = min(f_peak_t, f_join_rd_t) / m_sec * 0.98
    if f_int_max > f_int_min:
        f_int_np = np.linspace(f_int_min, f_int_max, 50, dtype=np.float64)
        hp_t_int, hc_t_int, ph_t_int = st_t.eval_waveform(torch.from_numpy(f_int_np), return_phase=True)
        hp_j_int, hc_j_int, ph_j_int = st_j.eval_waveform(jnp.asarray(f_int_np), return_phase=True)

        np.testing.assert_allclose(hp_t_int.numpy(), np.asarray(hp_j_int), rtol=1e-12)
        np.testing.assert_allclose(hc_t_int.numpy(), np.asarray(hc_j_int), rtol=1e-12)
        np.testing.assert_allclose(ph_t_int.numpy(), np.asarray(ph_j_int), rtol=1e-12)

    # 3. Merger-ringdown regime test (frequencies above peak/join)
    f_mr_min = max(f_peak_t, f_join_rd_t) / m_sec * 1.05
    f_mr_np = np.linspace(f_mr_min, f_mr_min + 300.0, 50, dtype=np.float64)
    hp_t_mr, hc_t_mr, ph_t_mr = st_t.eval_waveform(torch.from_numpy(f_mr_np), return_phase=True)
    hp_j_mr, hc_j_mr, ph_j_mr = st_j.eval_waveform(jnp.asarray(f_mr_np), return_phase=True)

    scale_mr = np.max(np.abs(hp_t_mr.numpy()))
    np.testing.assert_allclose(hp_t_mr.numpy(), np.asarray(hp_j_mr), rtol=2e-11, atol=1e-14 * scale_mr)
    np.testing.assert_allclose(hc_t_mr.numpy(), np.asarray(hc_j_mr), rtol=2e-11, atol=1e-14 * scale_mr)
    np.testing.assert_allclose(ph_t_mr.numpy(), np.asarray(ph_j_mr), rtol=2e-11)

    # 4. Verify eval_phase method parity
    phase_only_t = st_t.eval_phase(torch.from_numpy(f_ins_np))
    phase_only_j = st_j.eval_phase(jnp.asarray(f_ins_np))
    np.testing.assert_allclose(phase_only_t.numpy(), np.asarray(phase_only_j), rtol=1e-12)


def test_xas_jax_jit_compilation():
    """Verify that jaxwave IMRPhenomXAS compiles with @jax.jit and yields identical results."""
    f_np = np.linspace(25.0, 400.0, 150, dtype=np.float64)
    f_j = jnp.asarray(f_np)

    @jax.jit
    def jitted_eval(m1, m2, s1, s2):
        return JaxPhenomXAS.evaluate(
            sample_frequencies=f_j,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            f_ref=25.0,
        )

    hp_jit, hc_jit = jitted_eval(36.0, 24.0, 0.3, -0.2)
    hp_raw, hc_raw = JaxPhenomXAS.evaluate(
        sample_frequencies=f_j,
        mass1=36.0,
        mass2=24.0,
        chi1=0.3,
        chi2=-0.2,
        f_ref=25.0,
    )

    np.testing.assert_allclose(np.asarray(hp_jit), np.asarray(hp_raw), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(np.asarray(hc_jit), np.asarray(hc_raw), rtol=1e-15, atol=1e-15)


def test_xas_autodiff_parity_with_torch_and_finite_diff():
    """Verify jax.grad autodiff parity against PyTorch autograd and numerical finite difference."""
    f_np = np.linspace(30.0, 300.0, 60, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    m1_val, m2_val, c1_val, c2_val = 38.0, 28.0, 0.25, -0.35

    # 1. PyTorch autograd gradients
    m1_t = torch.tensor(m1_val, dtype=torch.float64, requires_grad=True)
    m2_t = torch.tensor(m2_val, dtype=torch.float64, requires_grad=True)
    c1_t = torch.tensor(c1_val, dtype=torch.float64, requires_grad=True)
    c2_t = torch.tensor(c2_val, dtype=torch.float64, requires_grad=True)

    model_t = TorchPhenomXAS()
    hp_t, _ = model_t(sample_frequencies=f_t, mass1=m1_t, mass2=m2_t, chi1=c1_t, chi2=c2_t, f_ref=30.0)
    loss_t = torch.sum(hp_t.real)
    loss_t.backward()

    torch_grads = [m1_t.grad.item(), m2_t.grad.item(), c1_t.grad.item(), c2_t.grad.item()]

    # 2. JAX autodiff gradients
    def jax_loss(m1, m2, c1, c2):
        hp, _ = JaxPhenomXAS.evaluate(
            sample_frequencies=f_j,
            mass1=m1,
            mass2=m2,
            chi1=c1,
            chi2=c2,
            f_ref=30.0,
        )
        return jnp.sum(jnp.real(hp))

    grad_fn = jax.jit(jax.grad(jax_loss, argnums=(0, 1, 2, 3)))
    jax_grads = [float(g) for g in grad_fn(m1_val, m2_val, c1_val, c2_val)]

    # Cross-framework gradient parity: verify exact agreement with PyTorch autograd
    for p_name, g_t, g_j in zip(["m1", "m2", "chi1", "chi2"], torch_grads, jax_grads):
        assert np.isfinite(g_j), f"JAX gradient for {p_name} is non-finite"
        assert abs(g_j) > 0.0, f"JAX gradient for {p_name} is zero"
        np.testing.assert_allclose(g_t, g_j, rtol=1e-10, err_msg=f"Grad mismatch for {p_name}")

    # 3. Finite difference convergence check on JAX model
    eps = 1e-5
    base_args = [m1_val, m2_val, c1_val, c2_val]
    for idx, p_name in enumerate(["m1", "m2", "chi1", "chi2"]):
        args_p = list(base_args)
        args_m = list(base_args)
        args_p[idx] += eps
        args_m[idx] -= eps
        fd_grad = float((jax_loss(*args_p) - jax_loss(*args_m)) / (2.0 * eps))
        rel_diff = abs(jax_grads[idx] - fd_grad) / abs(jax_grads[idx])
        assert rel_diff < 1e-4, f"FD check failed for {p_name}: rel_diff={rel_diff}"


def test_xas_batched_and_vmap_parity():
    """Verify batched evaluation and jax.vmap consistency against single evaluations and PyTorch."""
    f_np = np.linspace(20.0, 400.0, 150, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    m1_arr = np.array([32.0, 45.0, 55.0])
    m2_arr = np.array([24.0, 15.0, 10.0])
    c1_arr = np.array([0.2, -0.4, 0.6])
    c2_arr = np.array([-0.3, 0.1, -0.5])

    # PyTorch batched
    model_t = TorchPhenomXAS()
    hp_t, hc_t = model_t(
        sample_frequencies=f_t,
        mass1=torch.from_numpy(m1_arr),
        mass2=torch.from_numpy(m2_arr),
        chi1=torch.from_numpy(c1_arr),
        chi2=torch.from_numpy(c2_arr),
        f_ref=20.0,
    )

    # JAX batched
    hp_j_b, hc_j_b = JaxPhenomXAS.evaluate(
        sample_frequencies=f_j,
        mass1=jnp.asarray(m1_arr),
        mass2=jnp.asarray(m2_arr),
        chi1=jnp.asarray(c1_arr),
        chi2=jnp.asarray(c2_arr),
        f_ref=20.0,
    )

    # JAX vmap
    vmap_fn = jax.vmap(
        lambda m1, m2, c1, c2: JaxPhenomXAS.evaluate(
            sample_frequencies=f_j,
            mass1=m1,
            mass2=m2,
            chi1=c1,
            chi2=c2,
            f_ref=20.0,
        )
    )
    hp_j_v, hc_j_v = vmap_fn(
        jnp.asarray(m1_arr),
        jnp.asarray(m2_arr),
        jnp.asarray(c1_arr),
        jnp.asarray(c2_arr),
    )

    # Assert shape (3, 150)
    assert hp_j_b.shape == (3, 150)
    assert hp_j_v.shape == (3, 150)

    # Parity between JAX batched and JAX vmap
    np.testing.assert_allclose(np.asarray(hp_j_b), np.asarray(hp_j_v), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(np.asarray(hc_j_b), np.asarray(hc_j_v), rtol=1e-14, atol=1e-15)

    # Parity between PyTorch batched and JAX batched
    scale = np.max(np.abs(hp_t.numpy()))
    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j_b), rtol=5e-12, atol=1e-12 * scale)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j_b), rtol=5e-12, atol=1e-12 * scale)
