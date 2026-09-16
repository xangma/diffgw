"""Cross-framework parity tests: torchwave vs jaxwave for IMRPhenomNSBH.

Verifies:
- Compactness fit: ns_compactness_from_lambda
- Relativistic disruption root: solve_xi_tide
- Remnant mass/spin: zappa_remnant_properties & bbh_remnant_uib2016
- Characteristic frequencies: nsbh_frequencies (f_tide, f_isco, f_rd, c)
- Full waveform parity across disruptive, mildly disruptive, and non-disruptive regimes
- Gauge conventions: legacy, reference_frequency, coalescence
- @jax.jit compilation, jax.vmap batching, and autodiff
"""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.imrphenom_nsbh import (
    IMRPhenomNSBH as TorchNSBH,
    ns_compactness_from_lambda as torch_compactness,
    solve_xi_tide as torch_solve_xi_tide,
    bbh_remnant_uib2016 as torch_bbh_remnant,
    zappa_remnant_properties as torch_zappa_remnant,
    london_ringdown_properties as torch_london_ringdown,
    nsbh_frequencies as torch_nsbh_frequencies,
)
from jaxwave.approximants.imrphenom_nsbh import (
    IMRPhenomNSBH as JaxNSBH,
    ns_compactness_from_lambda as jax_compactness,
    solve_xi_tide as jax_solve_xi_tide,
    bbh_remnant_uib2016 as jax_bbh_remnant,
    zappa_remnant_properties as jax_zappa_remnant,
    london_ringdown_properties as jax_london_ringdown,
    nsbh_frequencies as jax_nsbh_frequencies,
)


# -----------------------------------------------------------------------------
# 1. Compactness Fit Parity
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 50.0, 100.0, 400.0, 800.0, 1500.0, 2600.0])
def test_compactness_parity(lam):
    """Verify NS compactness C(Lambda) matches between torchwave and jaxwave."""
    lam_t = torch.tensor(lam, dtype=torch.float64)
    c_t = torch_compactness(lam_t)

    lam_j = jnp.array(lam, dtype=jnp.float64)
    c_j = jax_compactness(lam_j)

    np.testing.assert_allclose(float(c_t), float(c_j), rtol=1e-14, atol=1e-15)


# -----------------------------------------------------------------------------
# 2. Relativistic Disruption Root Parity
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "q,chi,c",
    [
        (4.2857, 0.2, 0.1726),
        (10.714, 0.0, 0.1805),
        (2.5, 0.75, 0.1583),
        (5.0, -0.3, 0.1650),
    ],
)
def test_solve_xi_tide_parity(q, chi, c):
    """Verify relativistic disruption root xi_tide matches between frameworks."""
    q_t = torch.tensor(q, dtype=torch.float64)
    chi_t = torch.tensor(chi, dtype=torch.float64)
    c_t = torch.tensor(c, dtype=torch.float64)
    xi_t, res_t = torch_solve_xi_tide(q_t, chi_t, c_t)

    q_j = jnp.array(q, dtype=jnp.float64)
    chi_j = jnp.array(chi, dtype=jnp.float64)
    c_j = jnp.array(c, dtype=jnp.float64)
    xi_j, res_j = jax_solve_xi_tide(q_j, chi_j, c_j)

    np.testing.assert_allclose(float(xi_t), float(xi_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(float(res_t), float(res_j), rtol=1e-10, atol=1e-12)


# -----------------------------------------------------------------------------
# 3. Remnant Mass & Spin Fits Parity
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "m1,m2,chi1,chi2",
    [
        (6.0, 1.4, 0.2, 0.0),
        (15.0, 1.4, 0.0, 0.0),
        (3.0, 1.2, 0.8, 0.0),
        (5.0, 1.35, -0.4, 0.0),
    ],
)
def test_bbh_remnant_uib2016_parity(m1, m2, chi1, chi2):
    """Verify BBH remnant properties match between torchwave and jaxwave."""
    mf_t, chif_t = torch_bbh_remnant(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
    )
    mf_j, chif_j = jax_bbh_remnant(
        jnp.array(m1, dtype=jnp.float64),
        jnp.array(m2, dtype=jnp.float64),
        jnp.array(chi1, dtype=jnp.float64),
        jnp.array(chi2, dtype=jnp.float64),
    )

    np.testing.assert_allclose(float(mf_t), float(mf_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(float(chif_t), float(chif_j), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize(
    "m1,m2,chi1,chi2,lambda2",
    [
        (6.0, 1.4, 0.2, 0.0, 400.0),
        (6.0, 1.4, 0.2, 0.0, 800.0),
        (15.0, 1.4, 0.0, 0.0, 300.0),
        (3.0, 1.2, 0.8, 0.0, 800.0),
        (5.0, 1.4, -0.4, 0.0, 500.0),
        (7.0, 1.35, -0.6, 0.0, 350.0),
    ],
)
def test_zappa_remnant_parity(m1, m2, chi1, chi2, lambda2):
    """Verify Zappa et al. (2019) NSBH remnant mass and spin match between frameworks."""
    mf_t, chif_t = torch_zappa_remnant(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
        torch.tensor(lambda2, dtype=torch.float64),
    )
    mf_j, chif_j = jax_zappa_remnant(
        jnp.array(m1, dtype=jnp.float64),
        jnp.array(m2, dtype=jnp.float64),
        jnp.array(chi1, dtype=jnp.float64),
        jnp.array(chi2, dtype=jnp.float64),
        jnp.array(lambda2, dtype=jnp.float64),
    )

    np.testing.assert_allclose(float(mf_t), float(mf_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(float(chif_t), float(chif_j), rtol=1e-14, atol=1e-15)


# -----------------------------------------------------------------------------
# 4. Characteristic Frequencies Parity
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "m1,m2,chi1,lambda2",
    [
        (6.0, 1.4, 0.2, 400.0),
        (6.0, 1.4, 0.2, 800.0),
        (15.0, 1.4, 0.0, 300.0),
        (3.0, 1.2, 0.75, 800.0),
        (5.0, 1.35, -0.3, 500.0),
    ],
)
def test_nsbh_frequencies_parity(m1, m2, chi1, lambda2):
    """Verify disruption frequency, ISCO, ringdown frequency, and compactness match."""
    ft_t, fi_t, fr_t, c_t = torch_nsbh_frequencies(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(lambda2, dtype=torch.float64),
    )
    ft_j, fi_j, fr_j, c_j = jax_nsbh_frequencies(
        jnp.array(m1, dtype=jnp.float64),
        jnp.array(m2, dtype=jnp.float64),
        jnp.array(chi1, dtype=jnp.float64),
        jnp.array(lambda2, dtype=jnp.float64),
    )

    np.testing.assert_allclose(float(ft_t), float(ft_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(float(fi_t), float(fi_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(float(fr_t), float(fr_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(float(c_t), float(c_j), rtol=1e-14, atol=1e-15)


# -----------------------------------------------------------------------------
# 5. Full Waveform Parity (Disruptive, Mildly Disruptive, Non-Disruptive)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case_name,m1,m2,chi1,lambda2,inc",
    [
        ("nondisruptive_plunge", 15.0, 1.4, 0.0, 300.0, 0.4),
        ("nondisruptive_low_spin", 8.0, 1.4, 0.1, 250.0, 0.0),
        ("mildly_disruptive_400", 6.0, 1.4, 0.2, 400.0, 0.6),
        ("mildly_disruptive_800", 6.0, 1.4, 0.2, 800.0, 0.3),
        ("disruptive_high_spin", 3.0, 1.2, 0.75, 600.0, 0.8),
        ("disruptive_large_tide", 3.5, 1.3, 0.5, 800.0, 0.0),
        ("retrograde_spin", 5.0, 1.4, -0.3, 400.0, 0.5),
    ],
)
@pytest.mark.parametrize(
    "phasing_convention",
    ["legacy", "reference_frequency", "coalescence"],
)
def test_nsbh_full_waveform_parity(case_name, m1, m2, chi1, lambda2, inc, phasing_convention):
    """Verify complete (h_plus, h_cross) parity between torchwave and jaxwave."""
    f_np = np.linspace(25.0, 1200.0, 300, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    model_t = TorchNSBH()
    model_j = JaxNSBH()

    hp_t, hc_t = model_t(
        f=f_t,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        lambda2=lambda2,
        distance=80.0,
        inclination=inc,
        phic=0.3,
        tc=0.005,
        f_ref=35.0,
        phasing_convention=phasing_convention,
    )

    hp_j, hc_j = model_j(
        f=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        lambda2=lambda2,
        distance=80.0,
        inclination=inc,
        phic=0.3,
        tc=0.005,
        f_ref=35.0,
        phasing_convention=phasing_convention,
    )

    scale_p = np.max(np.abs(hp_t.numpy()))
    scale_c = np.max(np.abs(hc_t.numpy()))

    np.testing.assert_allclose(hp_t.numpy() / scale_p, np.asarray(hp_j) / scale_p, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(hc_t.numpy() / scale_c, np.asarray(hc_j) / scale_c, rtol=1e-11, atol=1e-11)


# -----------------------------------------------------------------------------
# 6. JIT, Autodiff, and VMAP Tests
# -----------------------------------------------------------------------------

def test_nsbh_jax_jit():
    """Verify that jaxwave IMRPhenomNSBH compiles with jax.jit."""
    f_j = jnp.linspace(20.0, 800.0, 150, dtype=jnp.float64)
    model = JaxNSBH()

    @jax.jit
    def generate(m1, m2, s1, lam2):
        hp, hc = model(
            f=f_j,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            lambda2=lam2,
            f_ref=30.0,
        )
        return jnp.real(hp).sum()

    val = generate(6.0, 1.4, 0.2, 400.0)
    assert jnp.isfinite(val)


def test_nsbh_jax_autodiff():
    """Verify that jaxwave IMRPhenomNSBH supports reverse-mode autodiff across parameters."""
    f_j = jnp.linspace(30.0, 500.0, 60, dtype=jnp.float64)
    model = JaxNSBH()

    def loss_fn(m1, m2, s1, lam2, inc):
        hp, hc = model(
            f=f_j,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            lambda2=lam2,
            inclination=inc,
            f_ref=30.0,
        )
        return jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2) * 1e44

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4)))
    grads = grad_fn(6.0, 1.4, 0.2, 400.0, 0.5)

    param_names = ["m1", "m2", "chi1", "lambda2", "inclination"]
    for name, g in zip(param_names, grads):
        assert jnp.isfinite(g), f"Gradient w.r.t {name} is not finite: {g}"
        assert g != 0.0, f"Gradient w.r.t {name} is zero: {g}"


def test_nsbh_jax_vmap():
    """Verify batched vectorization across binary parameters using jax.vmap."""
    f_j = jnp.linspace(30.0, 600.0, 80, dtype=jnp.float64)
    model = JaxNSBH()

    m1_batch = jnp.array([5.0, 6.0, 7.0], dtype=jnp.float64)
    lam2_batch = jnp.array([300.0, 450.0, 600.0], dtype=jnp.float64)

    vmapped_fn = jax.vmap(
        lambda m1, lam2: model(f_j, mass1=m1, mass2=1.4, chi1=0.2, lambda2=lam2)[0]
    )
    batched_hp = vmapped_fn(m1_batch, lam2_batch)

    assert batched_hp.shape == (3, 80)
    for i in range(3):
        single_hp, _ = model(f_j, mass1=m1_batch[i], mass2=1.4, chi1=0.2, lambda2=lam2_batch[i])
        np.testing.assert_allclose(np.asarray(batched_hp[i]), np.asarray(single_hp), rtol=1e-14, atol=1e-15)
