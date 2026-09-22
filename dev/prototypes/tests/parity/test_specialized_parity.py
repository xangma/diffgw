"""Cross-framework parity tests: torchwave vs jaxwave for Specialized Astrophysical Models.

Validates exact mathematical parity, @jax.jit compilation, jax.vmap, and jax.grad autodiff across:
1. ContinuousWave (characteristic amplitude, Roemer delay, Doppler phase, quadrupolar amplitudes and basis, full strain)
2. EMRIKludge (Kerr geodesic frequencies, Peters-Mathews radiation reaction, orbital trajectory, multipolar polarizations, LISA TDI projection)
3. ParameterizedPostEinstein (ppE phase shift, modified dispersion phase shift, wrapper across TaylorF2, IMRPhenomD, IMRPhenomXAS)
4. SpinPrecessionODE (precession frequency vectors, state derivatives, RK4 step, trajectory integration, Euler angles, frequency grid angles)
"""

import math
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

# Enable float64 precision in JAX
jax.config.update("jax_enable_x64", True)

# Torchwave models
import torchwave.approximants.continuous_wave as t_cw
import torchwave.approximants.emri_kludge as t_emri
import torchwave.approximants.ppe as t_ppe
import torchwave.approximants.precession_ode as t_ode
from torchwave.approximants.taylorf2 import TaylorF2 as TorchTaylorF2
from torchwave.approximants.imrphenomd import IMRPhenomD as TorchIMRPhenomD
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS as TorchIMRPhenomXAS

# Jaxwave models
import jaxwave.approximants.continuous_wave as j_cw
import jaxwave.approximants.emri_kludge as j_emri
import jaxwave.approximants.ppe as j_ppe
import jaxwave.approximants.precession_ode as j_ode
from jaxwave.approximants.taylorf2 import TaylorF2 as JaxTaylorF2
from jaxwave.approximants.imrphenomd import IMRPhenomD as JaxIMRPhenomD
from jaxwave.approximants.imrphenom_xas import IMRPhenomXAS as JaxIMRPhenomXAS


# ==============================================================================
# 1. ContinuousWave Parity Tests
# ==============================================================================

@pytest.mark.parametrize("f0", [10.0, 50.0, 200.0])
@pytest.mark.parametrize("eps", [1e-6, 1e-5, 5e-4])
@pytest.mark.parametrize("dist", [1e19, 1e21])
def test_continuous_wave_h0_parity(f0, eps, dist):
    """Verify characteristic strain amplitude h_0 parity."""
    t_val = t_cw.continuous_wave_h0(frequency=f0, ellipticity=eps, distance=dist, i_zz=1.2e38)
    j_val = j_cw.continuous_wave_h0(frequency=f0, ellipticity=eps, distance=dist, i_zz=1.2e38)

    np.testing.assert_allclose(t_val.item(), float(j_val), rtol=1e-14, atol=1e-30)


@pytest.mark.parametrize("det", [None, "H1", "L1", "V1"])
def test_continuous_wave_roemer_delay_parity(det):
    """Verify geometric Roemer delay parity across Earth orbital and diurnal effects."""
    t_arr = np.linspace(0.0, 1000.0, 100)
    t_torch = torch.from_numpy(t_arr)
    t_jax = jnp.array(t_arr)

    ra, dec = 1.3, -0.4
    t_del = t_cw.detector_roemer_delay(t=t_torch, ra=ra, dec=dec, detector=det, t_epoch=10.0)
    j_del = j_cw.detector_roemer_delay(t=t_jax, ra=ra, dec=dec, detector=det, t_epoch=10.0)

    np.testing.assert_allclose(t_del.numpy(), np.asarray(j_del), rtol=1e-12, atol=1e-14)


def test_continuous_wave_phase_and_amplitudes_parity():
    """Verify Doppler-modulated CW phase and 4 JKS amplitudes parity."""
    t_arr = np.linspace(0.0, 200.0, 150)
    t_torch = torch.from_numpy(t_arr)
    t_jax = jnp.array(t_arr)

    ra, dec = 0.9, 0.3
    f0, f1, f2 = 40.0, -1e-9, 1e-18
    phi0 = 0.7

    # Phase
    t_phi = t_cw.continuous_wave_phase(t=t_torch, ra=ra, dec=dec, f0=f0, f1=f1, f2=f2, phi0=phi0, detector="H1")
    j_phi = j_cw.continuous_wave_phase(t=t_jax, ra=ra, dec=dec, f0=f0, f1=f1, f2=f2, phi0=phi0, detector="H1")
    np.testing.assert_allclose(t_phi.numpy(), np.asarray(j_phi), rtol=1e-12, atol=1e-14)

    # Amplitudes
    t_amp = t_cw.continuous_wave_amplitudes(h0=2e-22, cos_iota=0.75, psi=0.4, phi0=phi0)
    j_amp = j_cw.continuous_wave_amplitudes(h0=2e-22, cos_iota=0.75, psi=0.4, phi0=phi0)
    np.testing.assert_allclose(t_amp.numpy(), np.asarray(j_amp), rtol=1e-14, atol=1e-30)


@pytest.mark.parametrize("det", ["H1", "L1"])
def test_continuous_wave_basis_and_strain_parity(det):
    """Verify four basis functions and full detector strain h(t) parity."""
    t_arr = np.linspace(0.0, 100.0, 200)
    t_torch = torch.from_numpy(t_arr)
    t_jax = jnp.array(t_arr)

    ra, dec = 1.1, -0.2
    f0, f1 = 30.0, -2e-10
    cos_i, psi, phi0 = 0.65, 0.35, 0.5

    # Basis
    t_basis = t_cw.continuous_wave_basis(t=t_torch, ra=ra, dec=dec, f0=f0, f1=f1, detector=det)
    j_basis = j_cw.continuous_wave_basis(t=t_jax, ra=ra, dec=dec, f0=f0, f1=f1, detector=det)
    np.testing.assert_allclose(t_basis.numpy(), np.asarray(j_basis), rtol=1e-8, atol=1e-10)

    # Direct strain with h0
    t_strain = t_cw.continuous_wave_strain(
        t=t_torch, ra=ra, dec=dec, f0=f0, f1=f1, h0=1.5e-22,
        cos_iota=cos_i, psi=psi, phi0=phi0, detector=det,
    )
    j_strain = j_cw.continuous_wave_strain(
        t=t_jax, ra=ra, dec=dec, f0=f0, f1=f1, h0=1.5e-22,
        cos_iota=cos_i, psi=psi, phi0=phi0, detector=det,
    )
    np.testing.assert_allclose(t_strain.numpy(), np.asarray(j_strain), rtol=1e-8, atol=1e-25)

    # ContinuousWave class with ellipticity + distance
    cw_t = t_cw.ContinuousWave(detector=det)
    cw_j = j_cw.ContinuousWave(detector=det)

    t_class = cw_t(
        t=t_torch, ra=ra, dec=dec, f0=f0, f1=f1, ellipticity=1e-5, distance=1e21,
        cos_iota=cos_i, psi=psi, phi0=phi0,
    )
    j_class = cw_j(
        t=t_jax, ra=ra, dec=dec, f0=f0, f1=f1, ellipticity=1e-5, distance=1e21,
        cos_iota=cos_i, psi=psi, phi0=phi0,
    )
    np.testing.assert_allclose(t_class.numpy(), np.asarray(j_class), rtol=1e-8, atol=1e-25)


def test_continuous_wave_zero_ellipticity_limit():
    """Verify h(t) identically vanishes when ellipticity=0 or h0=0."""
    t_jax = jnp.linspace(0.0, 50.0, 100)
    cw = j_cw.ContinuousWave(detector="H1")

    h_zero_eps = cw(t=t_jax, ra=1.0, dec=0.5, f0=30.0, ellipticity=0.0, distance=1e20)
    assert jnp.all(h_zero_eps == 0.0)

    h_zero_h0 = cw(t=t_jax, ra=1.0, dec=0.5, f0=30.0, h0=0.0)
    assert jnp.all(h_zero_h0 == 0.0)


def test_continuous_wave_jit_vmap_grad():
    """Verify ContinuousWave under JAX JIT compilation, vmap, and gradient autodiff."""
    t_grid = jnp.linspace(0.0, 20.0, 50)
    cw = j_cw.ContinuousWave(detector="H1")

    # JIT
    @jax.jit
    def run_cw(ra, dec, f0, h0):
        return cw(t=t_grid, ra=ra, dec=dec, f0=f0, h0=h0, cos_iota=0.8, psi=0.2, phi0=0.1)

    res_jit = run_cw(1.2, 0.4, 35.0, 1e-21)
    res_eager = cw(t=t_grid, ra=1.2, dec=0.4, f0=35.0, h0=1e-21, cos_iota=0.8, psi=0.2, phi0=0.1)
    np.testing.assert_allclose(np.asarray(res_jit), np.asarray(res_eager), rtol=1e-14, atol=1e-30)

    # VMAP over batch of sky coordinates
    ra_batch = jnp.array([0.5, 1.0, 1.5])
    dec_batch = jnp.array([-0.2, 0.0, 0.4])
    vmap_cw = jax.vmap(lambda r, d: cw(t=t_grid, ra=r, dec=d, f0=35.0, h0=1e-21))
    batched_h = vmap_cw(ra_batch, dec_batch)
    assert batched_h.shape == (3, 50)
    for i in range(3):
        h_single = cw(t=t_grid, ra=ra_batch[i], dec=dec_batch[i], f0=35.0, h0=1e-21)
        np.testing.assert_allclose(np.asarray(batched_h[i]), np.asarray(h_single), rtol=1e-14, atol=1e-30)

    # GRAD autodiff with respect to physical parameters
    def loss_cw(ra, dec, f0, h0):
        h = cw(t=t_grid, ra=ra, dec=dec, f0=f0, h0=h0)
        return jnp.sum(h ** 2)

    grad_ra, grad_dec, grad_f0, grad_h0 = jax.grad(loss_cw, argnums=(0, 1, 2, 3))(1.2, 0.4, 35.0, 1e-21)
    for name, g in [("ra", grad_ra), ("dec", grad_dec), ("f0", grad_f0), ("h0", grad_h0)]:
        assert jnp.isfinite(g), f"Grad for {name} is non-finite: {g}"
        assert abs(float(g)) > 0.0, f"Grad for {name} is zero"


# ==============================================================================
# 2. EMRIKludge Parity Tests
# ==============================================================================

@pytest.mark.parametrize("m", [5e5, 1e6, 2e6])
@pytest.mark.parametrize("spin", [-0.7, 0.0, 0.5, 0.9])
@pytest.mark.parametrize("p", [6.0, 10.0, 14.0])
@pytest.mark.parametrize("e", [0.0, 0.2, 0.5])
@pytest.mark.parametrize("in_hz", [True, False])
def test_kerr_geodesic_frequencies_parity(m, spin, p, e, in_hz):
    """Verify Kerr orbital frequencies and precession rates parity."""
    inc = 0.25
    t_out = t_emri.kerr_geodesic_frequencies(total_mass=m, spin=spin, p=p, e=e, inclination=inc, in_hz=in_hz)
    j_out = j_emri.kerr_geodesic_frequencies(total_mass=m, spin=spin, p=p, e=e, inclination=inc, in_hz=in_hz)

    for to, jo in zip(t_out, j_out):
        np.testing.assert_allclose(to.item(), float(jo), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("e", [0.0, 0.15, 0.6])
def test_peters_mathews_radiation_reaction_parity(e):
    """Verify Peters-Mathews radiation reaction rates dp/dt and de/dt parity."""
    m, mu, p = 1e6, 10.0, 8.0
    t_dp, t_de = t_emri.peters_mathews_radiation_reaction(m, mu, p, e)
    j_dp, j_de = j_emri.peters_mathews_radiation_reaction(m, mu, p, e)

    np.testing.assert_allclose(t_dp.item(), float(j_dp), rtol=1e-12, atol=1e-25)
    np.testing.assert_allclose(t_de.item(), float(j_de), rtol=1e-12, atol=1e-25)


@pytest.mark.parametrize("n_harmonics", [3, 5])
def test_emri_polarizations_parity(n_harmonics):
    """Verify EMRI multi-harmonic time-domain polarizations parity."""
    t_arr = np.linspace(0.0, 500.0, 120)
    t_torch = torch.from_numpy(t_arr)
    t_jax = jnp.array(t_arr)

    model_t = t_emri.EMRIKludge(n_harmonics=n_harmonics)
    model_j = j_emri.EMRIKludge(n_harmonics=n_harmonics)

    kwargs = dict(total_mass=1e6, secondary_mass=10.0, spin=0.6, p0=9.0, e0=0.2, inclination=0.3, distance=1500.0)
    hp_t, hc_t = model_t.polarizations(t_torch, **kwargs)
    hp_j, hc_j = model_j.polarizations(t_jax, **kwargs)

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-12, atol=1e-25)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-12, atol=1e-25)


def test_emri_lisa_projection_parity():
    """Verify EMRI signal projection to LISA TDI channels A, E, T parity."""
    t_arr = np.linspace(0.0, 400.0, 80)
    t_torch = torch.from_numpy(t_arr)
    t_jax = jnp.array(t_arr)

    model_t = t_emri.EMRIKludge(n_harmonics=4)
    model_j = j_emri.EMRIKludge(n_harmonics=4)

    theta, phi, psi = math.pi / 4.0, math.pi / 3.0, math.pi / 6.0
    kwargs = dict(total_mass=1.2e6, secondary_mass=15.0, spin=0.4, p0=8.5, e0=0.1, inclination=0.2, distance=1000.0)

    tdi_t = model_t.project_to_lisa(t_torch, theta=theta, phi=phi, psi=psi, channels=("A", "E", "T"), **kwargs)
    tdi_j = model_j.project_to_lisa(t_jax, theta=theta, phi=phi, psi=psi, channels=("A", "E", "T"), **kwargs)

    for ch in ("A", "E", "T"):
        np.testing.assert_allclose(tdi_t[ch].numpy(), np.asarray(tdi_j[ch]), rtol=1e-12, atol=1e-25)


def test_emri_jit_vmap_grad():
    """Verify EMRIKludge under JAX JIT compilation, vmap, and gradient autodiff."""
    t_grid = jnp.linspace(0.0, 200.0, 50)
    model = j_emri.EMRIKludge(n_harmonics=3)

    # JIT
    @jax.jit
    def run_emri(p0, e0, spin):
        return model.polarizations(t_grid, total_mass=1e6, secondary_mass=10.0, spin=spin, p0=p0, e0=e0)

    hp_jit, hc_jit = run_emri(8.0, 0.1, 0.5)
    hp_eager, hc_eager = model.polarizations(t_grid, total_mass=1e6, secondary_mass=10.0, spin=0.5, p0=8.0, e0=0.1)
    np.testing.assert_allclose(np.asarray(hp_jit), np.asarray(hp_eager), rtol=1e-14, atol=1e-30)

    # VMAP over eccentricity batch
    e_batch = jnp.array([0.05, 0.1, 0.2])
    vmap_fn = jax.vmap(lambda e: model.polarizations(t_grid, total_mass=1e6, secondary_mass=10.0, spin=0.5, p0=8.0, e0=e)[0])
    batched_hp = vmap_fn(e_batch)
    assert batched_hp.shape == (3, 50)

    # GRAD autodiff
    def loss_emri(p0, e0, spin):
        hp, hc = model.polarizations(t_grid, total_mass=1e6, secondary_mass=10.0, spin=spin, p0=p0, e0=e0)
        return jnp.sum(hp ** 2 + hc ** 2)

    grad_p0, grad_e0, grad_spin = jax.grad(loss_emri, argnums=(0, 1, 2))(8.0, 0.1, 0.5)
    for name, g in [("p0", grad_p0), ("e0", grad_e0), ("spin", grad_spin)]:
        assert jnp.isfinite(g), f"Grad for {name} is non-finite: {g}"
        assert abs(float(g)) > 0.0, f"Grad for {name} is zero"


# ==============================================================================
# 3. ParameterizedPostEinstein Parity Tests
# ==============================================================================

@pytest.mark.parametrize("b", [-7, -4, -2, 0, 1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.05, -0.1])
def test_ppe_phase_shift_parity(b, beta):
    """Verify ppE phase shift power-law frequency evolution parity."""
    f_arr = np.linspace(20.0, 300.0, 60)
    f_t = torch.from_numpy(f_arr)
    f_j = jnp.array(f_arr)

    shift_t = t_ppe.ppe_phase_shift(f_t, total_mass=45.0, beta_ppe=beta, b_ppe=b)
    shift_j = j_ppe.ppe_phase_shift(f_j, total_mass=45.0, beta_ppe=beta, b_ppe=b)

    np.testing.assert_allclose(shift_t.numpy(), np.asarray(shift_j), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("lambda_g", [float("inf"), 1e15, 5e16])
@pytest.mark.parametrize("z", [0.0, 0.2])
def test_modified_dispersion_phase_shift_parity(lambda_g, z):
    """Verify modified dispersion phase shift for massive graviton parity."""
    f_arr = np.linspace(20.0, 250.0, 50)
    f_t = torch.from_numpy(f_arr)
    f_j = jnp.array(f_arr)

    disp_t = t_ppe.modified_dispersion_phase_shift(f_t, distance=200.0, lambda_g=lambda_g, z=z)
    disp_j = j_ppe.modified_dispersion_phase_shift(f_j, distance=200.0, lambda_g=lambda_g, z=z)

    np.testing.assert_allclose(disp_t.numpy(), np.asarray(disp_j), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize(
    "torch_approx_cls, jax_approx_cls, m1, m2",
    [
        (TorchIMRPhenomD, JaxIMRPhenomD, 35.0, 25.0),
        (TorchIMRPhenomXAS, JaxIMRPhenomXAS, 40.0, 30.0),
        (TorchTaylorF2, JaxTaylorF2, 1.4, 1.3),
    ],
)
@pytest.mark.parametrize("beta_ppe, lambda_g", [(0.0, float("inf")), (0.04, 2e15)])
def test_ppe_waveform_wrapper_parity(torch_approx_cls, jax_approx_cls, m1, m2, beta_ppe, lambda_g):
    """Verify ParameterizedPostEinstein wrapper parity across multiple base models."""
    f_arr = np.linspace(30.0, 250.0, 80)
    f_t = torch.from_numpy(f_arr)
    f_j = jnp.array(f_arr)

    model_t = t_ppe.ParameterizedPostEinstein(approximant=torch_approx_cls())
    model_j = j_ppe.ParameterizedPostEinstein(approximant=jax_approx_cls())

    hp_t, hc_t = model_t(
        f=f_t, mass1=m1, mass2=m2, distance=100.0,
        beta_ppe=beta_ppe, b_ppe=-2, lambda_g=lambda_g, redshift=0.05,
    )
    hp_j, hc_j = model_j(
        f=f_j, mass1=m1, mass2=m2, distance=100.0,
        beta_ppe=beta_ppe, b_ppe=-2, lambda_g=lambda_g, redshift=0.05,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-12, atol=1e-25)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-12, atol=1e-25)


def test_ppe_jit_vmap_grad():
    """Verify ParameterizedPostEinstein under JAX JIT compilation, vmap, and gradient autodiff."""
    f_grid = jnp.linspace(30.0, 250.0, 60)
    model = j_ppe.ParameterizedPostEinstein(approximant=JaxIMRPhenomD())

    # JIT
    @jax.jit
    def run_ppe(m1, m2, beta):
        return model(f=f_grid, mass1=m1, mass2=m2, distance=100.0, beta_ppe=beta, b_ppe=-2)

    hp_jit, hc_jit = run_ppe(30.0, 25.0, 0.05)
    hp_eager, hc_eager = model(f=f_grid, mass1=30.0, mass2=25.0, distance=100.0, beta_ppe=0.05, b_ppe=-2)
    np.testing.assert_allclose(np.asarray(hp_jit), np.asarray(hp_eager), rtol=1e-14, atol=1e-30)

    # VMAP over beta_ppe
    betas = jnp.array([0.0, 0.02, 0.05])
    vmap_fn = jax.vmap(lambda b: model(f=f_grid, mass1=30.0, mass2=25.0, distance=100.0, beta_ppe=b)[0])
    batched_hp = vmap_fn(betas)
    assert batched_hp.shape == (3, 60)

    # GRAD autodiff
    def loss_ppe(m1, beta):
        hp, hc = model(f=f_grid, mass1=m1, mass2=25.0, distance=100.0, beta_ppe=beta, b_ppe=-2)
        return jnp.sum(jnp.real(hp * jnp.conj(hp)))

    grad_m1, grad_beta = jax.grad(loss_ppe, argnums=(0, 1))(30.0, 0.05)
    assert jnp.isfinite(grad_m1) and abs(float(grad_m1)) > 0.0
    assert jnp.isfinite(grad_beta) and abs(float(grad_beta)) > 0.0


# ==============================================================================
# 4. SpinPrecessionODE Parity Tests
# ==============================================================================

def test_spin_precession_vectors_and_derivatives_parity():
    """Verify spin precession frequency vectors and time derivatives parity."""
    s1 = np.array([0.2, -0.1, 0.3])
    s2 = np.array([-0.1, 0.2, -0.2])
    l_vec = np.array([0.0, 0.0, 1.2])
    m1, m2 = 35.0, 25.0

    # Vectors
    om1_t, om2_t = t_ode.SpinPrecessionODE.precession_vectors(
        torch.tensor(s1, dtype=torch.float64), torch.tensor(s2, dtype=torch.float64),
        torch.tensor(l_vec, dtype=torch.float64), torch.tensor(m1, dtype=torch.float64), torch.tensor(m2, dtype=torch.float64),
    )
    om1_j, om2_j = j_ode.SpinPrecessionODE.precession_vectors(
        jnp.array(s1), jnp.array(s2), jnp.array(l_vec), m1, m2
    )
    np.testing.assert_allclose(om1_t.numpy(), np.asarray(om1_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(om2_t.numpy(), np.asarray(om2_j), rtol=1e-14, atol=1e-15)

    # Derivatives with and without radiation reaction
    for rr in [True, False]:
        ds1_t, ds2_t, dl_t = t_ode.SpinPrecessionODE.state_derivatives(
            torch.tensor(s1, dtype=torch.float64), torch.tensor(s2, dtype=torch.float64),
            torch.tensor(l_vec, dtype=torch.float64),
            torch.tensor(m1, dtype=torch.float64), torch.tensor(m2, dtype=torch.float64),
            radiation_reaction=rr,
        )
        ds1_j, ds2_j, dl_j = j_ode.SpinPrecessionODE.state_derivatives(
            jnp.array(s1), jnp.array(s2), jnp.array(l_vec),
            m1, m2, radiation_reaction=rr,
        )
        np.testing.assert_allclose(ds1_t.numpy(), np.asarray(ds1_j), rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(ds2_t.numpy(), np.asarray(ds2_j), rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(dl_t.numpy(), np.asarray(dl_j), rtol=1e-14, atol=1e-15)


def test_spin_precession_rk4_step_and_integration_parity():
    """Verify single RK4 step and multi-step time integration parity and |J| conservation."""
    m1, m2 = 36.0, 24.0
    total_m = m1 + m2
    eta = (m1 * m2) / (total_m * total_m)

    chi1 = np.array([0.3, -0.2, 0.4])
    chi2 = np.array([-0.1, 0.4, -0.3])
    s1_0 = (m1 / total_m) ** 2 * chi1
    s2_0 = (m2 / total_m) ** 2 * chi2

    r_val = 18.0
    v_val = 1.0 / math.sqrt(r_val)
    l_0 = (eta / v_val) * np.array([0.0, 0.0, 1.0])

    # Single RK4 step
    dt = 1.5
    s1_next_t, s2_next_t, l_next_t = t_ode.SpinPrecessionODE.rk4_step(
        torch.tensor(s1_0, dtype=torch.float64), torch.tensor(s2_0, dtype=torch.float64),
        torch.tensor(l_0, dtype=torch.float64),
        dt, torch.tensor(m1, dtype=torch.float64), torch.tensor(m2, dtype=torch.float64),
        radiation_reaction=False,
    )
    s1_next_j, s2_next_j, l_next_j = j_ode.SpinPrecessionODE.rk4_step(
        jnp.array(s1_0), jnp.array(s2_0), jnp.array(l_0),
        dt, m1, m2, radiation_reaction=False,
    )
    np.testing.assert_allclose(s1_next_t.numpy(), np.asarray(s1_next_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(s2_next_t.numpy(), np.asarray(s2_next_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(l_next_t.numpy(), np.asarray(l_next_j), rtol=1e-14, atol=1e-15)

    # Multi-step integration over 500 steps
    sol_t = t_ode.SpinPrecessionODE.integrate_time(
        torch.tensor(s1_0, dtype=torch.float64), torch.tensor(s2_0, dtype=torch.float64),
        torch.tensor(l_0, dtype=torch.float64),
        torch.tensor(m1, dtype=torch.float64), torch.tensor(m2, dtype=torch.float64),
        total_time=800.0, n_steps=500, radiation_reaction=False,
    )
    sol_j = j_ode.SpinPrecessionODE.integrate_time(
        jnp.array(s1_0), jnp.array(s2_0), jnp.array(l_0),
        m1, m2, total_time=800.0, n_steps=500, radiation_reaction=False,
    )

    for key in ("times", "s1", "s2", "l", "j"):
        np.testing.assert_allclose(sol_t[key].numpy(), np.asarray(sol_j[key]), rtol=1e-7, atol=1e-8)

    # Verify conservative limit |J| conservation to < 1e-6
    j0_mag = np.linalg.norm(s1_0 + s2_0 + l_0)
    j_mags = jnp.linalg.norm(sol_j["j"], axis=-1)
    max_rel_err = float(jnp.max(jnp.abs(j_mags - j0_mag) / j0_mag))
    assert max_rel_err < 1e-6, f"Max relative error in |J| {max_rel_err:.2e} exceeds 1e-6"


def test_extract_euler_angles_parity():
    """Verify extraction of Euler precession angles (alpha, beta, epsilon) parity."""
    alpha_np = np.linspace(-2.0, 5.0, 80)
    beta_val = 0.37
    l_np = np.stack(
        (np.sin(beta_val) * np.cos(alpha_np), np.sin(beta_val) * np.sin(alpha_np), np.full_like(alpha_np, np.cos(beta_val))),
        axis=-1,
    )
    j0_np = np.array([0.1, 0.2, 1.0])

    a_t, b_t, e_t = t_ode.SpinPrecessionODE.extract_euler_angles(torch.from_numpy(l_np), torch.from_numpy(j0_np))
    a_j, b_j, e_j = j_ode.SpinPrecessionODE.extract_euler_angles(jnp.array(l_np), jnp.array(j0_np))

    np.testing.assert_allclose(a_t.numpy(), np.asarray(a_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(b_t.numpy(), np.asarray(b_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(e_t.numpy(), np.asarray(e_j), rtol=1e-13, atol=1e-14)


def test_compute_angles_frequency_parity():
    """Verify frequency grid ODE angle evolution parity."""
    f_np = np.array([25.0, 40.0, 55.0, 80.0])
    params = dict(
        mass1=40.0, mass2=20.0,
        spin1x=0.3, spin1y=0.2, spin1z=0.1,
        spin2x=-0.2, spin2y=0.1, spin2z=-0.1,
        f_ref=40.0,
    )

    a_t, b_t, e_t = t_ode.SpinPrecessionODE.compute_angles_frequency(torch.from_numpy(f_np), **params)
    a_j, b_j, e_j = j_ode.SpinPrecessionODE.compute_angles_frequency(jnp.array(f_np), **params)

    np.testing.assert_allclose(a_t.numpy(), np.asarray(a_j), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(b_t.numpy(), np.asarray(b_j), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(e_t.numpy(), np.asarray(e_j), rtol=1e-13, atol=1e-14)


def test_spin_precession_ode_jit_vmap_grad():
    """Verify SpinPrecessionODE under JAX JIT compilation, vmap, and gradient autodiff."""
    f_grid = jnp.array([25.0, 40.0, 55.0, 80.0])
    params = dict(mass1=40.0, mass2=20.0, spin1y=0.2, spin1z=0.1, spin2x=-0.2, spin2y=0.1, spin2z=-0.1, f_ref=40.0)

    # JIT
    jit_angles = jax.jit(j_ode.SpinPrecessionODE.compute_angles_frequency, static_argnames=["max_steps"])
    a_jit, b_jit, e_jit = jit_angles(f_grid, spin1x=0.3, max_steps=100, **params)
    a_eager, b_eager, e_eager = j_ode.SpinPrecessionODE.compute_angles_frequency(f_grid, spin1x=0.3, max_steps=100, **params)
    np.testing.assert_allclose(np.asarray(a_jit), np.asarray(a_eager), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(np.asarray(b_jit), np.asarray(b_eager), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(np.asarray(e_jit), np.asarray(e_eager), rtol=1e-14, atol=1e-15)

    # VMAP over frequency
    vmap_angles = jax.vmap(
        lambda f: j_ode.SpinPrecessionODE.compute_angles_frequency(jnp.array([f]), spin1x=0.3, max_steps=100, **params)[0][0]
    )
    alphas_vmap = vmap_angles(f_grid)
    np.testing.assert_allclose(np.asarray(alphas_vmap), np.asarray(a_jit), rtol=1e-14, atol=1e-15)

    # GRAD autodiff with respect to spin1x and mass1
    def loss_ode(s1x, m1):
        a, b, e = j_ode.SpinPrecessionODE.compute_angles_frequency(
            f_grid, mass1=m1, mass2=20.0, spin1x=s1x, spin1y=0.2, spin1z=0.1,
            spin2x=-0.2, spin2y=0.1, spin2z=-0.1, f_ref=40.0, max_steps=100,
        )
        return jnp.sum(a ** 2 + b ** 2 + e ** 2)

    grad_s1x, grad_m1 = jax.grad(loss_ode, argnums=(0, 1))(0.3, 40.0)
    assert jnp.isfinite(grad_s1x) and abs(float(grad_s1x)) > 0.0
    assert jnp.isfinite(grad_m1) and abs(float(grad_m1)) > 0.0
