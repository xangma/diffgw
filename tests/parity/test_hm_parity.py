"""Cross-framework parity tests: torchwave vs jaxwave for Higher-Order Multipole models.

Tests parity for:
- IMRPhenomHM (London et al. 2018)
- TaylorF2_HM (Kidder 2008, García-Quirós et al. 2020)
- Spin-weighted spherical harmonic projection factors
- Individual multipole harmonic modes: (2,2), (2,1), (3,3), (3,2), (4,4), (4,3)
"""

import math
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.imrphenom_hm import (
    IMRPhenomHM as TorchPhenomHM,
    _swsh_projection_coefficients as torch_swsh_hm,
    frequency_mapping_amplitude as torch_fmap_amp,
    frequency_mapping_phase as torch_fmap_phi,
)
from jaxwave.approximants.imrphenom_hm import (
    IMRPhenomHM as JaxPhenomHM,
    _swsh_projection_coefficients as jax_swsh_hm,
    frequency_mapping_amplitude as jax_fmap_amp,
    frequency_mapping_phase as jax_fmap_phi,
)
from torchwave.approximants.taylorf2_hm import (
    TaylorF2_HM as TorchTaylorF2_HM,
    _swsh_projection_coefficients as torch_swsh_t2,
)
from jaxwave.approximants.taylorf2_hm import (
    TaylorF2_HM as JaxTaylorF2_HM,
    _swsh_projection_coefficients as jax_swsh_t2,
)
from torchwave.approximants.imrphenom_hm_qnm_n61 import evaluate_n61_qnm as torch_n61_qnm
from jaxwave.approximants.imrphenom_hm_qnm_n61 import evaluate_n61_qnm as jax_n61_qnm


# -------------------------------------------------------------------------
# SWSH Projection Parity
# -------------------------------------------------------------------------

@pytest.mark.parametrize("l,m", [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)])
@pytest.mark.parametrize("iota", [0.0, 0.4, math.pi / 3.0, math.pi / 2.0])
@pytest.mark.parametrize("phi0", [0.0, 0.3])
def test_swsh_projection_parity_hm(l, m, iota, phi0):
    """Verify IMRPhenomHM SWSH coefficients match between torchwave and jaxwave."""
    dp_t, dc_t = torch_swsh_hm(l, m, torch.tensor(iota, dtype=torch.float64), torch.tensor(phi0, dtype=torch.float64))
    dp_j, dc_j = jax_swsh_hm(l, m, jnp.array(iota, dtype=jnp.float64), jnp.array(phi0, dtype=jnp.float64))

    np.testing.assert_allclose(dp_t.numpy(), np.asarray(dp_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(dc_t.numpy(), np.asarray(dc_j), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("l,m", [(2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4)])
@pytest.mark.parametrize("iota", [0.2, 0.8, math.pi / 2.0])
@pytest.mark.parametrize("phi0", [0.0, 0.5])
def test_swsh_projection_parity_taylorf2(l, m, iota, phi0):
    """Verify TaylorF2_HM SWSH coefficients match between torchwave and jaxwave."""
    dp_t, dc_t = torch_swsh_t2(l, m, torch.tensor(iota, dtype=torch.float64), torch.tensor(phi0, dtype=torch.float64))
    dp_j, dc_j = jax_swsh_t2(l, m, jnp.array(iota, dtype=jnp.float64), jnp.array(phi0, dtype=jnp.float64))

    np.testing.assert_allclose(dp_t.numpy(), np.asarray(dp_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(dc_t.numpy(), np.asarray(dc_j), rtol=1e-14, atol=1e-15)


# -------------------------------------------------------------------------
# QNM & Frequency Mapping Parity
# -------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)])
@pytest.mark.parametrize("spin", [-0.8, -0.2, 0.0, 0.4, 0.9])
def test_n61_qnm_parity(mode, spin):
    """Verify N=61 London QNM polynomial evaluation parity."""
    tr, ti = torch_n61_qnm(mode, torch.tensor(spin, dtype=torch.float64))
    jr, ji = jax_n61_qnm(mode, jnp.array(spin, dtype=jnp.float64))

    np.testing.assert_allclose(float(tr), float(jr), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(float(ti), float(ji), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("mode", [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)])
def test_imrphenom_hm_mode_qnm_frequencies_parity(mode):
    """Verify mode_qnm_frequencies matches between Torch and Jax."""
    m1, m2 = 45.0, 15.0
    c1, c2 = 0.3, -0.2
    fr_t, fd_t = TorchPhenomHM.mode_qnm_frequencies(
        mode[0], mode[1],
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        torch.tensor(c1, dtype=torch.float64),
        torch.tensor(c2, dtype=torch.float64),
    )
    fr_j, fd_j = JaxPhenomHM.mode_qnm_frequencies(
        mode[0], mode[1],
        jnp.array(m1, dtype=jnp.float64),
        jnp.array(m2, dtype=jnp.float64),
        jnp.array(c1, dtype=jnp.float64),
        jnp.array(c2, dtype=jnp.float64),
    )

    np.testing.assert_allclose(float(fr_t), float(fr_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(float(fd_t), float(fd_j), rtol=1e-14, atol=1e-15)


def test_frequency_mapping_parity():
    """Verify London piecewise frequency maps match."""
    mf = np.linspace(0.005, 0.2, 200, dtype=np.float64)
    mf_t = torch.from_numpy(mf)
    mf_j = jnp.asarray(mf)

    f0_a, f0_p = 0.015, 0.020
    fr_lm, fr_22 = 0.08, 0.06

    for m in [1, 2, 3, 4]:
        map_a_t = torch_fmap_amp(mf_t, f0_a, fr_lm, fr_22, m)
        map_a_j = jax_fmap_amp(mf_j, f0_a, fr_lm, fr_22, m)
        np.testing.assert_allclose(map_a_t.numpy(), np.asarray(map_a_j), rtol=1e-14, atol=1e-15)

        map_p_t = torch_fmap_phi(mf_t, f0_p, fr_lm, fr_22, m)
        map_p_j = jax_fmap_phi(mf_j, f0_p, fr_lm, fr_22, m)
        np.testing.assert_allclose(map_p_t.numpy(), np.asarray(map_p_j), rtol=1e-14, atol=1e-15)


# -------------------------------------------------------------------------
# TaylorF2_HM Parity Tests
# -------------------------------------------------------------------------

@pytest.mark.parametrize("m1,m2", [(30.0, 10.0), (20.0, 15.0), (1.4, 1.3)])
@pytest.mark.parametrize("chi1,chi2", [(0.0, 0.0), (0.3, -0.2)])
def test_taylorf2_hm_waveform_parity(m1, m2, chi1, chi2):
    """Verify TaylorF2_HM full polarization strain parity."""
    f_np = np.linspace(30.0, 400.0, 200, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    hp_t, hc_t = TorchTaylorF2_HM()(
        f=f_t,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=100.0,
        inclination=0.7,
        phic=0.3,
        tc=0.005,
        phi0=0.1,
    )
    hp_j, hc_j = JaxTaylorF2_HM.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=100.0,
        inclination=0.7,
        phic=0.3,
        tc=0.005,
        phi0=0.1,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-11, atol=1e-25)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-11, atol=1e-25)


@pytest.mark.parametrize("mode", [(2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4)])
def test_taylorf2_hm_individual_modes_parity(mode):
    """Verify parity for every individual harmonic mode in TaylorF2_HM."""
    f_np = np.linspace(30.0, 350.0, 150, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    _, _, modes_t = TorchTaylorF2_HM(modes=[mode])(
        f=f_t,
        mass1=35.0,
        mass2=15.0,
        chi1=0.2,
        chi2=-0.1,
        inclination=0.6,
        return_modes=True,
        f_isco_cutoff=False,
    )
    _, _, modes_j = JaxTaylorF2_HM(modes=[mode])(
        f=f_j,
        mass1=35.0,
        mass2=15.0,
        chi1=0.2,
        chi2=-0.1,
        inclination=0.6,
        return_modes=True,
        f_isco_cutoff=False,
    )

    hp_t, hc_t = modes_t[mode]
    hp_j, hc_j = modes_j[mode]

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-11, atol=1e-25)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-11, atol=1e-25)


def test_taylorf2_hm_odd_m_vanish_equal_mass():
    """Verify that odd-m modes vanish identically for equal masses in both frameworks."""
    f_j = jnp.linspace(30.0, 300.0, 100, dtype=jnp.float64)
    _, _, modes = JaxTaylorF2_HM.evaluate(
        f=f_j,
        mass1=15.0,
        mass2=15.0,
        inclination=0.8,
        return_modes=True,
        f_isco_cutoff=False,
    )

    for mode in [(2, 1), (3, 1), (3, 3), (4, 3)]:
        hp_m, hc_m = modes[mode]
        np.testing.assert_allclose(np.asarray(hp_m), 0.0, atol=1e-28)
        np.testing.assert_allclose(np.asarray(hc_m), 0.0, atol=1e-28)


def test_taylorf2_hm_jax_jit():
    """Verify TaylorF2_HM compiles cleanly with jax.jit and supports autodiff."""
    f_j = jnp.linspace(30.0, 250.0, 100, dtype=jnp.float64)

    @jax.jit
    def generate(m1, m2, s1, s2):
        hp, hc = JaxTaylorF2_HM.evaluate(
            f=f_j,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            inclination=0.5,
            f_isco_cutoff=False,
        )
        return jnp.real(hp).sum()

    val = generate(30.0, 15.0, 0.1, -0.1)
    assert jnp.isfinite(val)

    grad_fn = jax.jit(jax.grad(generate, argnums=(0, 1, 2, 3)))
    dm1, dm2, ds1, ds2 = grad_fn(30.0, 15.0, 0.1, -0.1)
    assert jnp.isfinite(dm1)
    assert jnp.isfinite(dm2)
    assert jnp.isfinite(ds1)
    assert jnp.isfinite(ds2)


# -------------------------------------------------------------------------
# IMRPhenomHM Parity Tests
# -------------------------------------------------------------------------

@pytest.mark.parametrize("m1,m2", [(40.0, 20.0), (50.0, 15.0), (30.0, 30.0)])
@pytest.mark.parametrize("chi1,chi2", [(0.0, 0.0), (0.2, -0.1)])
@pytest.mark.parametrize("inclination", [0.0, 0.6])
def test_imrphenom_hm_waveform_parity(m1, m2, chi1, chi2, inclination):
    """Verify IMRPhenomHM full polarization strain parity."""
    f_np = np.linspace(25.0, 350.0, 120, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    hp_t, hc_t = TorchPhenomHM()(
        f=f_t,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=100.0,
        inclination=inclination,
        phic=0.2,
        tc=0.005,
        f_ref=25.0,
    )
    hp_j, hc_j = JaxPhenomHM.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=100.0,
        inclination=inclination,
        phic=0.2,
        tc=0.005,
        f_ref=25.0,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-11, atol=1e-25)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-11, atol=1e-25)


@pytest.mark.parametrize("mode", [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)])
def test_imrphenom_hm_individual_modes_parity(mode):
    """Verify parity for every individual harmonic mode in IMRPhenomHM."""
    f_np = np.linspace(25.0, 300.0, 100, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    _, _, modes_t = TorchPhenomHM(modes=[mode])(
        f=f_t,
        mass1=40.0,
        mass2=15.0,
        chi1=0.25,
        chi2=-0.15,
        inclination=0.7,
        f_ref=25.0,
        return_modes=True,
    )
    _, _, modes_j = JaxPhenomHM.evaluate(
        f=f_j,
        mass1=40.0,
        mass2=15.0,
        chi1=0.25,
        chi2=-0.15,
        inclination=0.7,
        f_ref=25.0,
        modes=[mode],
        return_modes=True,
    )

    hp_t, hc_t = modes_t[mode]
    hp_j, hc_j = modes_j[mode]

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-11, atol=1e-25)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-11, atol=1e-25)


def test_imrphenom_hm_odd_m_vanish_equal_mass():
    """Verify odd-m modes (2,1) and (3,3) vanish for equal masses in JAX."""
    f_j = jnp.linspace(25.0, 300.0, 100, dtype=jnp.float64)
    _, _, modes = JaxPhenomHM.evaluate(
        f=f_j,
        mass1=30.0,
        mass2=30.0,
        chi1=0.2,
        chi2=0.2,
        inclination=math.pi / 3.0,
        modes=[(2, 1), (3, 3)],
        return_modes=True,
    )

    for mode in [(2, 1), (3, 3)]:
        hp_m, hc_m = modes[mode]
        np.testing.assert_allclose(np.asarray(hp_m), 0.0, atol=1e-28)
        np.testing.assert_allclose(np.asarray(hc_m), 0.0, atol=1e-28)


def test_imrphenom_hm_jax_jit_lightweight():
    """Lightweight JIT compilation verification for IMRPhenomHM on mode (2,2)."""
    f_j = jnp.linspace(25.0, 300.0, 60, dtype=jnp.float64)

    @jax.jit
    def generate(m1, m2, s1, s2):
        hp, hc = JaxPhenomHM.evaluate(
            f=f_j,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            inclination=0.4,
            f_ref=25.0,
            modes=[(2, 2)],
        )
        return jnp.real(hp).sum()

    val = generate(35.0, 20.0, 0.2, -0.1)
    assert jnp.isfinite(val)
