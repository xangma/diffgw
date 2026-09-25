"""Cross-framework parity tests: torchwave.inference vs jaxwave.inference."""

import math
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

# Automatically enable float64 in JAX
jax.config.update("jax_enable_x64", True)

# PyTorch imports
import torchwave.inference as torch_inf
from torchwave.detectors.psd import aLIGOZeroDetHighPower as torch_aLIGO
from torchwave.approximants.imrphenomd import IMRPhenomD as TorchIMRPhenomD
from torchwave.approximants.taylorf2 import TaylorF2 as TorchTaylorF2

# JAX imports
import jaxwave.inference as jax_inf
from jaxwave.detectors.psd import aLIGOZeroDetHighPower as jax_aLIGO
from jaxwave.approximants.imrphenomd import IMRPhenomD as JaxIMRPhenomD
from jaxwave.approximants.taylorf2 import TaylorF2 as JaxTaylorF2


# ============================================================================
# 1. Noise-Weighted Inner Product Parity
# ============================================================================

def test_noise_weighted_inner_product_parity_uniform():
    """Verify inner product parity on a uniform frequency grid."""
    f_np = np.linspace(20.0, 500.0, 256, dtype=np.float64)
    df = float(f_np[1] - f_np[0])

    rng = np.random.RandomState(42)
    a_np = rng.randn(len(f_np)) + 1j * rng.randn(len(f_np))
    b_np = rng.randn(len(f_np)) + 1j * rng.randn(len(f_np))

    f_t = torch.from_numpy(f_np)
    psd_t = torch_aLIGO(f_t)
    a_t = torch.from_numpy(a_np)
    b_t = torch.from_numpy(b_np)

    f_j = jnp.asarray(f_np)
    psd_j = jax_aLIGO(f_j)
    a_j = jnp.asarray(a_np)
    b_j = jnp.asarray(b_np)

    # Real inner product
    ip_t = torch_inf.noise_weighted_inner_product(a_t, b_t, psd_t, df=df)
    ip_j = jax_inf.noise_weighted_inner_product(a_j, b_j, psd_j, df=df)
    np.testing.assert_allclose(ip_t.numpy(), np.asarray(ip_j), rtol=1e-12, atol=1e-15)

    # Complex inner product
    ip_c_t = torch_inf.noise_weighted_inner_product(a_t, b_t, psd_t, df=df, complex_output=True)
    ip_c_j = jax_inf.noise_weighted_inner_product(a_j, b_j, psd_j, df=df, complex_output=True)
    np.testing.assert_allclose(ip_c_t.numpy(), np.asarray(ip_c_j), rtol=1e-12, atol=1e-15)


def test_noise_weighted_inner_product_parity_trapezoid():
    """Verify inner product parity on non-uniform grid with f_low and f_high cutoffs."""
    f_np = np.geomspace(20.0, 1024.0, 300, dtype=np.float64)
    rng = np.random.RandomState(123)
    a_np = rng.randn(len(f_np)) + 1j * rng.randn(len(f_np))
    b_np = rng.randn(len(f_np)) + 1j * rng.randn(len(f_np))

    f_t = torch.from_numpy(f_np)
    psd_t = torch_aLIGO(f_t)
    a_t = torch.from_numpy(a_np)
    b_t = torch.from_numpy(b_np)

    f_j = jnp.asarray(f_np)
    psd_j = jax_aLIGO(f_j)
    a_j = jnp.asarray(a_np)
    b_j = jnp.asarray(b_np)

    f_low = 30.0
    f_high = 500.0

    ip_t = torch_inf.noise_weighted_inner_product(
        a_t, b_t, psd_t, frequencies=f_t, f_low=f_low, f_high=f_high
    )
    ip_j = jax_inf.noise_weighted_inner_product(
        a_j, b_j, psd_j, frequencies=f_j, f_low=f_low, f_high=f_high
    )
    np.testing.assert_allclose(ip_t.numpy(), np.asarray(ip_j), rtol=1e-12, atol=1e-15)


# ============================================================================
# 2. Log Bessel I0 Parity and Stability
# ============================================================================

def test_log_bessel_i0_parity():
    """Verify log_bessel_i0 matches across PyTorch and JAX across small, moderate, and huge inputs."""
    x_vals = np.array([0.0, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 2000.0, 10000.0], dtype=np.float64)
    x_t = torch.from_numpy(x_vals)
    x_j = jnp.asarray(x_vals)

    res_t = torch_inf.log_bessel_i0(x_t).numpy()
    res_j = np.asarray(jax_inf.log_bessel_i0(x_j))

    np.testing.assert_allclose(res_t, res_j, rtol=1e-12, atol=1e-15)
    # Check asymptotic expansion at large x: log(I0(x)) ~ x - 0.5 * log(2 * pi * x)
    for i, x in enumerate(x_vals):
        if x > 100.0:
            asymp = x - 0.5 * math.log(2.0 * math.pi * x)
            assert math.isclose(res_j[i], asymp, rel_tol=1e-4)


# ============================================================================
# 3. Optimal SNR & Matched Filter SNR Parity
# ============================================================================

def test_snr_parity_single_and_network():
    """Verify optimal_snr and matched_filter_snr parity for single detector and network."""
    f_np = np.linspace(20.0, 400.0, 200, dtype=np.float64)
    df = float(f_np[1] - f_np[0])

    f_t = torch.from_numpy(f_np)
    psd_t = torch_aLIGO(f_t)
    hp_t, _ = TorchIMRPhenomD()(f=f_t, mass1=35.0, mass2=25.0, distance=100.0)

    f_j = jnp.asarray(f_np)
    psd_j = jax_aLIGO(f_j)
    hp_j, _ = JaxIMRPhenomD.evaluate(f=f_j, mass1=35.0, mass2=25.0, distance=100.0)

    # 1. Single detector optimal SNR
    opt_t = torch_inf.optimal_snr(hp_t, psd_t, df=df)
    opt_j = jax_inf.optimal_snr(hp_j, psd_j, df=df)
    np.testing.assert_allclose(opt_t.numpy(), np.asarray(opt_j), rtol=1e-12)

    # 2. Single detector matched filter SNR with exact match
    mf_t = torch_inf.matched_filter_snr(hp_t, hp_t, psd_t, df=df)
    mf_j = jax_inf.matched_filter_snr(hp_j, hp_j, psd_j, df=df)
    np.testing.assert_allclose(mf_t.numpy(), np.asarray(mf_j), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(opt_j), np.asarray(mf_j), rtol=1e-12)

    # 3. Network optimal and matched filter SNR
    net_wf_t = {"H1": hp_t, "L1": hp_t}
    net_psd_t = {"H1": psd_t, "L1": psd_t}
    net_wf_j = {"H1": hp_j, "L1": hp_j}
    net_psd_j = {"H1": psd_j, "L1": psd_j}

    net_opt_t = torch_inf.optimal_snr(net_wf_t, net_psd_t, df=df)
    net_opt_j = jax_inf.optimal_snr(net_wf_j, net_psd_j, df=df)
    np.testing.assert_allclose(net_opt_t.numpy(), np.asarray(net_opt_j), rtol=1e-12)

    net_mf_t = torch_inf.matched_filter_snr(net_wf_t, net_wf_t, net_psd_t, df=df)
    net_mf_j = jax_inf.matched_filter_snr(net_wf_j, net_wf_j, net_psd_j, df=df)
    np.testing.assert_allclose(net_mf_t.numpy(), np.asarray(net_mf_j), rtol=1e-12)


# ============================================================================
# 4. Whittle Likelihood Parity
# ============================================================================

def test_whittle_likelihood_parity():
    """Verify WhittleLikelihood parity across unmarginalized, phase-marginalized, and distance-marginalized modes."""
    f_np = np.linspace(20.0, 400.0, 200, dtype=np.float64)
    df = float(f_np[1] - f_np[0])

    f_t = torch.from_numpy(f_np)
    psd_t = torch_aLIGO(f_t)
    hp_inj_t, _ = TorchIMRPhenomD()(f=f_t, mass1=36.0, mass2=29.0, distance=100.0)
    hp_test_t, _ = TorchIMRPhenomD()(f=f_t, mass1=36.5, mass2=28.5, distance=100.0)

    f_j = jnp.asarray(f_np)
    psd_j = jax_aLIGO(f_j)
    hp_inj_j, _ = JaxIMRPhenomD.evaluate(f=f_j, mass1=36.0, mass2=29.0, distance=100.0)
    hp_test_j, _ = JaxIMRPhenomD.evaluate(f=f_j, mass1=36.5, mass2=28.5, distance=100.0)

    data_t = {"H1": hp_inj_t, "L1": hp_inj_t}
    psds_t = {"H1": psd_t, "L1": psd_t}
    data_j = {"H1": hp_inj_j, "L1": hp_inj_j}
    psds_j = {"H1": psd_j, "L1": psd_j}

    lik_t = torch_inf.WhittleLikelihood(data=data_t, psds=psds_t, df=df, frequencies=f_t)
    lik_j = jax_inf.WhittleLikelihood(data=data_j, psds=psds_j, df=df, frequencies=f_j)

    # 1. Standard Whittle likelihood at injection (should be 0.0)
    ln_l_0_t = lik_t({"H1": hp_inj_t, "L1": hp_inj_t})
    ln_l_0_j = lik_j({"H1": hp_inj_j, "L1": hp_inj_j})
    np.testing.assert_allclose(ln_l_0_t.numpy(), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ln_l_0_j), 0.0, atol=1e-10)

    # 2. Standard Whittle likelihood at perturbed template
    ln_l_test_t = lik_t({"H1": hp_test_t, "L1": hp_test_t})
    # The native waveforms differ at about 1e-12 relative L2 on current
    # backends. Subtraction into a small residual amplifies that difference;
    # compare the likelihood algorithms on identical strains and PSD instead.
    for torch_strain, jax_strain in ((hp_inj_t, hp_inj_j), (hp_test_t, hp_test_j)):
        reference = np.asarray(jax_strain)
        relative_l2 = np.linalg.norm(torch_strain.numpy() - reference) / np.linalg.norm(reference)
        assert relative_l2 < 1e-12
    shared_data_j = jnp.asarray(hp_inj_t.numpy())
    shared_template_j = jnp.asarray(hp_test_t.numpy())
    shared_psd_j = jnp.asarray(psd_t.numpy())
    lik_j_shared = jax_inf.WhittleLikelihood(
        data={"H1": shared_data_j, "L1": shared_data_j},
        psds={"H1": shared_psd_j, "L1": shared_psd_j},
        df=df, frequencies=f_j,
    )
    ln_l_shared_j = lik_j_shared({"H1": shared_template_j, "L1": shared_template_j})
    np.testing.assert_allclose(ln_l_test_t.numpy(), np.asarray(ln_l_shared_j), rtol=5e-11)

    # 3. Phase-marginalized log-likelihood
    pm_t = lik_t.phase_marginalized_log_likelihood({"H1": hp_test_t, "L1": hp_test_t})
    pm_j = lik_j.phase_marginalized_log_likelihood({"H1": hp_test_j, "L1": hp_test_j})
    np.testing.assert_allclose(pm_t.numpy(), np.asarray(pm_j), rtol=1e-10)

    # Verify phase-rotation invariance in JAX
    phase_shift = np.exp(1j * 0.7)
    pm_rot_j = lik_j.phase_marginalized_log_likelihood({
        "H1": hp_test_j * phase_shift,
        "L1": hp_test_j * phase_shift,
    })
    np.testing.assert_allclose(np.asarray(pm_j), np.asarray(pm_rot_j), rtol=1e-11)

    # 4. Distance-marginalized log-likelihood
    dm_t = lik_t.distance_marginalized_log_likelihood(
        {"H1": hp_test_t, "L1": hp_test_t}, d_min=50.0, d_max=1000.0, d_ref=100.0
    )
    dm_j = lik_j.distance_marginalized_log_likelihood(
        {"H1": hp_test_j, "L1": hp_test_j}, d_min=50.0, d_max=1000.0, d_ref=100.0
    )
    np.testing.assert_allclose(dm_t.numpy(), np.asarray(dm_j), rtol=1e-10)


def test_whittle_likelihood_autodiff_parity():
    """Verify gradient of Whittle log-likelihood w.r.t binary parameters matches between PyTorch and JAX."""
    f_np = np.linspace(30.0, 250.0, 100, dtype=np.float64)
    df = float(f_np[1] - f_np[0])

    f_t = torch.from_numpy(f_np)
    psd_t = torch_aLIGO(f_t)
    hp_inj_t, _ = TorchIMRPhenomD()(f=f_t, mass1=30.0, mass2=20.0, distance=100.0)
    lik_t = torch_inf.WhittleLikelihood(data={"H1": hp_inj_t}, psds={"H1": psd_t}, df=df)

    # PyTorch gradient
    m1_t = torch.tensor(31.0, dtype=torch.float64, requires_grad=True)
    m2_t = torch.tensor(20.5, dtype=torch.float64, requires_grad=True)
    hp_eval_t, _ = TorchIMRPhenomD()(f=f_t, mass1=m1_t, mass2=m2_t, distance=100.0)
    ln_l_t = lik_t({"H1": hp_eval_t})
    ln_l_t.backward()
    grad_t = np.array([m1_t.grad.item(), m2_t.grad.item()])

    # JAX gradient
    f_j = jnp.asarray(f_np)
    psd_j = jax_aLIGO(f_j)
    hp_inj_j, _ = JaxIMRPhenomD.evaluate(f=f_j, mass1=30.0, mass2=20.0, distance=100.0)
    lik_j = jax_inf.WhittleLikelihood(data={"H1": hp_inj_j}, psds={"H1": psd_j}, df=df)

    def jax_loss(p):
        hp, _ = JaxIMRPhenomD.evaluate(f=f_j, mass1=p[0], mass2=p[1], distance=100.0)
        return lik_j({"H1": hp})

    p_init = jnp.array([31.0, 20.5], dtype=jnp.float64)
    grad_j = np.asarray(jax.grad(jax_loss)(p_init))

    np.testing.assert_allclose(grad_t, grad_j, rtol=1e-7, atol=1e-8)


# ============================================================================
# 5. Relative Binning Parity
# ============================================================================

def test_relative_binning_bins_parity():
    """Verify compute_relative_binning_bins parity across frameworks."""
    f_dense = np.linspace(20.0, 500.0, 1000, dtype=np.float64)

    edges_t = torch_inf.compute_relative_binning_bins(
        f_min=20.0,
        f_max=500.0,
        n_bins=40,
        gamma=-5.0 / 3.0,
        frequencies=torch.from_numpy(f_dense),
    ).numpy()

    edges_j = np.asarray(jax_inf.compute_relative_binning_bins(
        f_min=20.0,
        f_max=500.0,
        n_bins=40,
        gamma=-5.0 / 3.0,
        frequencies=jnp.asarray(f_dense),
    ))

    np.testing.assert_allclose(edges_t, edges_j, rtol=1e-12, atol=1e-14)


def test_relative_binning_summary_and_likelihood_parity():
    """Verify relative binning summary coefficients and likelihood evaluations match between PyTorch and JAX."""
    n_dense = 2048
    f_np = np.linspace(20.0, 400.0, n_dense, dtype=np.float64)
    df = float(f_np[1] - f_np[0])

    m1_0, m2_0, dist_0 = 36.0, 29.0, 410.0

    # PyTorch setup
    f_t = torch.from_numpy(f_np)
    psd_t = torch_aLIGO(f_t)
    hp_0_t, _ = TorchIMRPhenomD()(f=f_t, mass1=m1_0, mass2=m2_0, distance=dist_0)

    data_t = {"H1": hp_0_t, "L1": hp_0_t}
    psds_t = {"H1": psd_t, "L1": psd_t}
    h0_t = {"H1": hp_0_t, "L1": hp_0_t}

    rel_lik_t = torch_inf.RelativeBinningLikelihood(
        data=data_t, psds=psds_t, h0=h0_t, frequencies=f_t, n_bins=40, df=df
    )

    # JAX setup
    f_j = jnp.asarray(f_np)
    psd_j = jax_aLIGO(f_j)
    hp_0_j, _ = JaxIMRPhenomD.evaluate(f=f_j, mass1=m1_0, mass2=m2_0, distance=dist_0)

    data_j = {"H1": hp_0_j, "L1": hp_0_j}
    psds_j = {"H1": psd_j, "L1": psd_j}
    h0_j = {"H1": hp_0_j, "L1": hp_0_j}

    rel_lik_j = jax_inf.RelativeBinningLikelihood(
        data=data_j, psds=psds_j, h0=h0_j, frequencies=f_j, n_bins=40, df=df
    )

    # 1. Summary coefficients parity
    for det in ["H1", "L1"]:
        a0_t = getattr(rel_lik_t, f"a0_{det}").numpy()
        a1_t = getattr(rel_lik_t, f"a1_{det}").numpy()
        b0_t = getattr(rel_lik_t, f"b0_{det}").numpy()
        b1_t = getattr(rel_lik_t, f"b1_{det}").numpy()
        dd_t = getattr(rel_lik_t, f"dd_{det}").numpy()

        a0_j = np.asarray(rel_lik_j.a0[det])
        a1_j = np.asarray(rel_lik_j.a1[det])
        b0_j = np.asarray(rel_lik_j.b0[det])
        b1_j = np.asarray(rel_lik_j.b1[det])
        dd_j = np.asarray(rel_lik_j.dd[det])

        np.testing.assert_allclose(a0_t, a0_j, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(a1_t, a1_j, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(b0_t, b0_j, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(b1_t, b1_j, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(dd_t, dd_j, rtol=1e-11, atol=1e-14)

    # 2. Fiducial reference evaluation (ln L = 0)
    bc_t = rel_lik_t.bin_centers
    bc_j = rel_lik_j.bin_centers
    np.testing.assert_allclose(bc_t.numpy(), np.asarray(bc_j), rtol=1e-12)

    hp_0_bc_t, _ = TorchIMRPhenomD()(f=bc_t, mass1=m1_0, mass2=m2_0, distance=dist_0)
    hp_0_bc_j, _ = JaxIMRPhenomD.evaluate(f=bc_j, mass1=m1_0, mass2=m2_0, distance=dist_0)

    ln_l_0_t = rel_lik_t({"H1": hp_0_bc_t, "L1": hp_0_bc_t})
    ln_l_0_j = rel_lik_j({"H1": hp_0_bc_j, "L1": hp_0_bc_j})
    np.testing.assert_allclose(ln_l_0_t.numpy(), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(ln_l_0_j), 0.0, atol=1e-8)

    # 3. Perturbed test parameters
    m1_test, m2_test = 36.02, 28.98
    hp_test_bc_t, _ = TorchIMRPhenomD()(f=bc_t, mass1=m1_test, mass2=m2_test, distance=dist_0)
    hp_test_bc_j, _ = JaxIMRPhenomD.evaluate(f=bc_j, mass1=m1_test, mass2=m2_test, distance=dist_0)

    ln_l_test_t = rel_lik_t({"H1": hp_test_bc_t, "L1": hp_test_bc_t})
    ln_l_test_j = rel_lik_j({"H1": hp_test_bc_j, "L1": hp_test_bc_j})
    np.testing.assert_allclose(ln_l_test_t.numpy(), np.asarray(ln_l_test_j), rtol=5e-10, atol=1e-10)

    # 4. Accuracy vs full dense Whittle likelihood (should match within 0.05)
    full_lik_j = jax_inf.WhittleLikelihood(data=data_j, psds=psds_j, df=df, frequencies=f_j)
    hp_test_dense_j, _ = JaxIMRPhenomD.evaluate(f=f_j, mass1=m1_test, mass2=m2_test, distance=dist_0)
    ln_l_full = full_lik_j({"H1": hp_test_dense_j, "L1": hp_test_dense_j})

    diff = abs(float(ln_l_test_j) - float(ln_l_full))
    assert diff < 0.05, f"Relative binning discrepancy {diff} exceeds 0.05"


def test_relative_binning_autodiff_parity():
    """Verify autograd gradient of relative binning matches between PyTorch and JAX."""
    n_dense = 2048
    f_np = np.linspace(25.0, 400.0, n_dense, dtype=np.float64)
    df = float(f_np[1] - f_np[0])

    m1_0, m2_0, dist_0 = 36.0, 29.0, 410.0

    # PyTorch
    f_t = torch.from_numpy(f_np)
    psd_t = torch_aLIGO(f_t)
    hp_0_t, _ = TorchIMRPhenomD()(f=f_t, mass1=m1_0, mass2=m2_0, distance=dist_0)
    rel_t = torch_inf.RelativeBinningLikelihood(
        data={"H1": hp_0_t}, psds={"H1": psd_t}, h0={"H1": hp_0_t}, frequencies=f_t, n_bins=40, df=df
    )
    bc_t = rel_t.bin_centers

    d_t = torch.tensor(405.0, dtype=torch.float64, requires_grad=True)
    phic_t = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
    hp_eval_t, _ = TorchIMRPhenomD()(f=bc_t, mass1=m1_0, mass2=m2_0, distance=d_t, phic=phic_t)
    ln_l_t = rel_t({"H1": hp_eval_t})
    ln_l_t.backward()
    grad_t = np.array([d_t.grad.item(), phic_t.grad.item()])

    # JAX
    f_j = jnp.asarray(f_np)
    psd_j = jax_aLIGO(f_j)
    hp_0_j, _ = JaxIMRPhenomD.evaluate(f=f_j, mass1=m1_0, mass2=m2_0, distance=dist_0)
    rel_j = jax_inf.RelativeBinningLikelihood(
        data={"H1": hp_0_j}, psds={"H1": psd_j}, h0={"H1": hp_0_j}, frequencies=f_j, n_bins=40, df=df
    )
    bc_j = rel_j.bin_centers

    def jax_loss(p):
        hp, _ = JaxIMRPhenomD.evaluate(f=bc_j, mass1=m1_0, mass2=m2_0, distance=p[0], phic=p[1])
        return rel_j({"H1": hp})

    p_init = jnp.array([405.0, 0.05], dtype=jnp.float64)
    grad_j = np.asarray(jax.grad(jax_loss)(p_init))

    np.testing.assert_allclose(grad_t, grad_j, rtol=1e-7, atol=1e-8)


# ============================================================================
# 6. Fisher Information Matrix, Covariance & Sky Parity
# ============================================================================

def test_fisher_information_matrix_parity():
    """Verify Fisher Information Matrix parity between PyTorch and JAX."""
    f_np = np.linspace(20.0, 400.0, 150, dtype=np.float64)
    df = float(f_np[1] - f_np[0])

    f_t = torch.from_numpy(f_np)
    psd_t = torch_aLIGO(f_t)

    f_j = jnp.asarray(f_np)
    psd_j = jax_aLIGO(f_j)

    def wf_torch(p_dict):
        m1 = p_dict["mass1"]
        m2 = p_dict["mass2"]
        dist = p_dict["distance"]
        hp, _ = TorchTaylorF2.evaluate(f=f_t, mass1=m1, mass2=m2, distance=dist, f_isco_cutoff=False)
        return hp

    def wf_jax(p_dict):
        m1 = p_dict["mass1"]
        m2 = p_dict["mass2"]
        dist = p_dict["distance"]
        hp, _ = JaxTaylorF2.evaluate(f=f_j, mass1=m1, mass2=m2, distance=dist, f_isco_cutoff=False)
        return hp

    params = {
        "mass1": 1.4,
        "mass2": 1.4,
        "distance": 100.0,
    }

    gamma_t, names_t = torch_inf.fisher_information_matrix(
        parameters_dict=params,
        waveform_fn=wf_torch,
        psd_dict=psd_t,
        df=df,
        f_low=20.0,
        f_high=400.0,
    )

    gamma_j, names_j = jax_inf.fisher_information_matrix(
        parameters_dict=params,
        waveform_fn=wf_jax,
        psd_dict=psd_j,
        df=df,
        f_low=20.0,
        f_high=400.0,
        method="autodiff",
    )

    assert names_t == names_j
    np.testing.assert_allclose(gamma_t.numpy(), np.asarray(gamma_j), rtol=1e-8)

    # Test covariance matrix parity (small off-diagonals have near-zero values ~1e-9)
    cov_t = torch_inf.covariance_matrix(gamma_t).numpy()
    cov_j = np.asarray(jax_inf.covariance_matrix(gamma_j))
    np.testing.assert_allclose(cov_t, cov_j, rtol=1e-4, atol=1e-8)

    # Test sky localization area formula parity
    synth_cov = np.array([
        [1.0, 0.0, 0.0],
        [0.0, (0.01)**2, 0.5 * 0.01 * 0.02],
        [0.0, 0.5 * 0.01 * 0.02, (0.02)**2],
    ], dtype=np.float64)

    area_t = torch_inf.sky_localization_area(
        torch.from_numpy(synth_cov), dec_idx=2, ra_idx=1, dec=0.5
    ).numpy()
    area_j = np.asarray(jax_inf.sky_localization_area(
        jnp.asarray(synth_cov), dec_idx=2, ra_idx=1, dec=0.5
    ))
    np.testing.assert_allclose(area_t, area_j, rtol=1e-12)


def test_fisher_methods_consistency_jax():
    """Verify consistency across JAX Fisher methods: autodiff (JVP), hessian, and finite differences."""
    f_j = jnp.linspace(20.0, 400.0, 100, dtype=jnp.float64)
    df = float(f_j[1] - f_j[0])
    psd_j = jax_aLIGO(f_j)

    def wf_jax(p_dict):
        m1 = p_dict["mass1"]
        m2 = p_dict["mass2"]
        dist = p_dict["distance"]
        hp, _ = JaxTaylorF2.evaluate(f=f_j, mass1=m1, mass2=m2, distance=dist, f_isco_cutoff=False)
        return hp

    params = {"mass1": 1.4, "mass2": 1.4, "distance": 100.0}

    gamma_ad, _ = jax_inf.fisher_information_matrix(
        params, wf_jax, psd_j, df=df, f_low=20.0, f_high=400.0, method="autodiff"
    )
    gamma_hess, _ = jax_inf.fisher_information_matrix(
        params, wf_jax, psd_j, df=df, f_low=20.0, f_high=400.0, method="hessian"
    )
    gamma_fd, _ = jax_inf.fisher_information_matrix(
        params, wf_jax, psd_j, df=df, f_low=20.0, f_high=400.0, method="finite_diff",
        step_sizes={"mass1": 1e-5, "mass2": 1e-5, "distance": 1e-3},
    )

    # Autodiff vs Hessian should be practically identical
    np.testing.assert_allclose(np.asarray(gamma_ad), np.asarray(gamma_hess), rtol=1e-9)
    # Finite differences should match to ~1e-4
    np.testing.assert_allclose(np.asarray(gamma_ad), np.asarray(gamma_fd), rtol=1e-3)


# ============================================================================
# 7. MAP Optimizer Parity
# ============================================================================

def test_optimize_map_convergence():
    """Verify MAP optimizer converges close to true injection parameters."""
    f = jnp.linspace(20.0, 300.0, 100, dtype=jnp.float64)
    df = float(f[1] - f[0])
    psd = jax_aLIGO(f)

    true_params = {"mass1": 15.0, "mass2": 15.0}

    def waveform_fn(p_dict):
        hp, _ = JaxTaylorF2.evaluate(
            f=f, mass1=p_dict["mass1"], mass2=p_dict["mass2"], distance=20.0, f_isco_cutoff=False
        )
        return hp

    clean_data = waveform_fn({k: jnp.asarray(v) for k, v in true_params.items()})
    init_params = {"mass1": 14.9, "mass2": 14.9}
    bounds = {"mass1": (10.0, 20.0), "mass2": (10.0, 20.0)}

    best_p, history = jax_inf.optimize_map(
        data=clean_data,
        psd=psd,
        initial_params=init_params,
        waveform_fn=waveform_fn,
        bounds=bounds,
        lr=0.01,
        steps=50,
        df=df,
        f_low=20.0,
        f_high=300.0,
    )

    assert history[-1] < history[0]
    rel_err_m1 = abs(best_p["mass1"] - true_params["mass1"]) / true_params["mass1"]
    rel_err_m2 = abs(best_p["mass2"] - true_params["mass2"]) / true_params["mass2"]
    assert rel_err_m1 < 0.05
    assert rel_err_m2 < 0.05


# ============================================================================
# 8. JAX JIT and VMAP Compatibility
# ============================================================================

def test_jax_jit_and_vmap_compatibility():
    """Verify that WhittleLikelihood and RelativeBinningLikelihood compile cleanly under jax.jit and jax.vmap."""
    f_dense = jnp.linspace(20.0, 400.0, 512, dtype=jnp.float64)
    df = float(f_dense[1] - f_dense[0])
    psd = jax_aLIGO(f_dense)

    hp_0, _ = JaxIMRPhenomD.evaluate(f=f_dense, mass1=36.0, mass2=29.0, distance=100.0)
    data = {"H1": hp_0}
    psds = {"H1": psd}
    h0 = {"H1": hp_0}

    whittle = jax_inf.WhittleLikelihood(data=data, psds=psds, df=df, frequencies=f_dense)
    rel_lik = jax_inf.RelativeBinningLikelihood(data=data, psds=psds, h0=h0, frequencies=f_dense, n_bins=30, df=df)
    bc = rel_lik.bin_centers

    # 1. JIT test on Whittle
    @jax.jit
    def eval_whittle(m1, m2):
        hp, _ = JaxIMRPhenomD.evaluate(f=f_dense, mass1=m1, mass2=m2, distance=100.0)
        return whittle({"H1": hp})

    ln_l_jit = eval_whittle(36.0, 29.0)
    np.testing.assert_allclose(float(ln_l_jit), 0.0, atol=1e-8)

    # 2. JIT test on Relative Binning
    @jax.jit
    def eval_rel(m1, m2):
        hp, _ = JaxIMRPhenomD.evaluate(f=bc, mass1=m1, mass2=m2, distance=100.0)
        return rel_lik({"H1": hp})

    ln_l_rel_jit = eval_rel(36.0, 29.0)
    np.testing.assert_allclose(float(ln_l_rel_jit), 0.0, atol=1e-8)

    # 3. VMAP test over batch of parameters
    m1_batch = jnp.array([35.9, 36.0, 36.1], dtype=jnp.float64)
    m2_batch = jnp.array([28.9, 29.0, 29.1], dtype=jnp.float64)

    vmapped_eval_rel = jax.vmap(eval_rel)
    batch_results = vmapped_eval_rel(m1_batch, m2_batch)
    assert batch_results.shape == (3,)
    np.testing.assert_allclose(float(batch_results[1]), 0.0, atol=1e-8)
