"""Cross-framework parity tests: torchwave vs jaxwave for KerrRingdown.

Validates:
1. Time-domain waveform parity against torchwave.approximants.kerr_ringdown.KerrRingdown.time_domain.
2. Frequency-domain analytic Fourier transform parity against torchwave.approximants.kerr_ringdown.KerrRingdown.frequency_domain.
3. Single mode reference parity (sampled and Fourier transforms).
4. Energy fraction to amplitude conversion parity across conventions.
5. Parameter batching parity.
6. JAX @jax.jit compilation, autodiff (jax.grad w.r.t final_mass, final_spin), and jax.vmap.
7. PyTree registration and methods tracer safety.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

# Enable float64 precision in JAX
jax.config.update("jax_enable_x64", True)

from torchwave.approximants.kerr_ringdown import (
    KerrRingdown as TorchRingdown,
    sampled_kerr_single_mode_reference as torch_sampled_single_ref,
    kerr_single_mode_reference as torch_single_ref,
    kerr_single_mode_fourier_reference as torch_single_fourier_ref,
    evaluate_qnm as torch_eval_qnm,
    get_ringdown_waveform as torch_get_waveform,
)
from torchwave.approximants.kerr_ringdown_energy import (
    ringdown_amplitude_from_energy_fraction as torch_energy_amp,
)

from jaxwave.approximants.kerr_ringdown import (
    KerrRingdown as JaxRingdown,
    sampled_kerr_single_mode_reference as jax_sampled_single_ref,
    kerr_single_mode_reference as jax_single_ref,
    kerr_single_mode_fourier_reference as jax_single_fourier_ref,
    ringdown_amplitude_from_energy_fraction as jax_energy_amp,
    evaluate_qnm as jax_eval_qnm,
    get_ringdown_waveform as jax_get_waveform,
)


def test_jax_qnm_zero_spin_gradient_matches_independent_roots():
    from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220

    data = Path(__file__).resolve().parents[1] / "data/kerr_qnm_220_leaver.json"
    rows = {row["a"]: row for row in json.loads(data.read_text())["holdouts"]}
    expected = np.array([
        (rows[1e-7][name] - rows[-1e-7][name]) / 2e-7
        for name in ("wr", "damping")
    ])
    gradient = jax.jacfwd(lambda spin: jnp.stack(evaluate_kerr_qnm_220(spin)))(
        jnp.array(0.0, dtype=jnp.float64)
    )
    np.testing.assert_allclose(np.asarray(gradient), expected, rtol=0.0, atol=5e-8)


@pytest.mark.parametrize("mass", [40.0, 65.0, 100.0])
@pytest.mark.parametrize("spin", [0.0, 0.5, 0.75, 0.92])
@pytest.mark.parametrize("backend", ["independent", "berti"])
def test_evaluate_qnm_parity(mass, spin, backend):
    """Verify physical QNM frequency, damping time, and quality factor match."""
    # (2, 2, 0)
    t_om, t_tau, t_q = torch_eval_qnm(2, 2, 0, mass, spin, qnm_backend=backend)
    j_om, j_tau, j_q = jax_eval_qnm(2, 2, 0, mass, spin, qnm_backend=backend)

    np.testing.assert_allclose(t_om.numpy(), np.asarray(j_om), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(t_tau.numpy(), np.asarray(j_tau), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(t_q.numpy(), np.asarray(j_q), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("mass", [50.0, 65.0])
@pytest.mark.parametrize("spin", [0.2, 0.7])
@pytest.mark.parametrize("inclination", [0.0, 0.6])
@pytest.mark.parametrize("t0", [0.0, 0.003])
@pytest.mark.parametrize("backend", ["independent", "berti"])
def test_kerr_ringdown_time_domain_parity(mass, spin, inclination, t0, backend):
    """Verify time-domain waveform parity between PyTorch and JAX implementations."""
    times_np = np.linspace(-0.005, 0.04, 350, dtype=np.float64)
    times_t = torch.from_numpy(times_np)
    times_j = jnp.asarray(times_np)

    amplitudes = {(2, 2, 0): 1.0, (2, 2, 1): 0.4}
    phases = {(2, 2, 0): 0.1, (2, 2, 1): 0.5}

    hp_t, hc_t = TorchRingdown.time_domain(
        times=times_t,
        final_mass=mass,
        final_spin=spin,
        amplitudes=amplitudes,
        phases=phases,
        t0=t0,
        inclination=inclination,
        phi=0.25,
        qnm_backend=backend,
        angular_basis="spherical",
    )

    hp_j, hc_j = JaxRingdown.time_domain(
        times=times_j,
        final_mass=mass,
        final_spin=spin,
        amplitudes=amplitudes,
        phases=phases,
        t0=t0,
        inclination=inclination,
        phi=0.25,
        qnm_backend=backend,
        angular_basis="spherical",
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-12, atol=1e-14)


def test_kerr_ringdown_spheroidal_basis_parity():
    """Verify spheroidal angular basis parity between PyTorch and JAX implementations."""
    times_np = np.linspace(0.0, 0.03, 100, dtype=np.float64)
    times_t = torch.from_numpy(times_np)
    times_j = jnp.asarray(times_np)

    hp_t, hc_t = TorchRingdown.time_domain(
        times=times_t,
        final_mass=65.0,
        final_spin=0.7,
        amplitudes={(2, 2, 0): 1.0},
        phases={(2, 2, 0): 0.2},
        inclination=0.5,
        phi=0.3,
        angular_basis="spheroidal",
    )

    hp_j, hc_j = JaxRingdown.time_domain(
        times=times_j,
        final_mass=65.0,
        final_spin=0.7,
        amplitudes={(2, 2, 0): 1.0},
        phases={(2, 2, 0): 0.2},
        inclination=0.5,
        phi=0.3,
        angular_basis="spheroidal",
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-12, atol=1e-14)


def test_kerr_ringdown_spheroidal_zero_spin_limit_jax():
    """The JAX angular projection must reduce to its spherical convention."""
    times = jnp.linspace(0.0, 0.02, 31)
    parameters = dict(final_mass=60.0, final_spin=0.0, inclination=0.93,
                      phi=0.27, amplitudes={(2, 2, 0): 0.8,
                                              (3, 3, 0): 0.1})
    spherical = JaxRingdown.time_domain(times, **parameters)
    spheroidal = JaxRingdown.time_domain(times, **parameters,
                                        angular_basis="spheroidal")
    for sphere, spheroid in zip(spherical, spheroidal):
        np.testing.assert_allclose(np.asarray(sphere), np.asarray(spheroid),
                                   rtol=2e-13, atol=1e-15)


@pytest.mark.parametrize("backend", ["independent", "berti"])
def test_kerr_ringdown_frequency_domain_parity(backend):
    """Verify frequency-domain analytic Fourier transform parity."""
    freqs_np = np.linspace(-400.0, 800.0, 250, dtype=np.float64)
    freqs_t = torch.from_numpy(freqs_np)
    freqs_j = jnp.asarray(freqs_np)

    mass = 68.0
    spin = 0.65
    t0 = 0.002
    amplitudes = {(2, 2, 0): 1.0, (2, 2, 1): 0.3}
    phases = {(2, 2, 0): 0.2, (2, 2, 1): -0.4}

    hp_t, hc_t = TorchRingdown.frequency_domain(
        frequencies=freqs_t,
        final_mass=mass,
        final_spin=spin,
        amplitudes=amplitudes,
        phases=phases,
        t0=t0,
        inclination=0.4,
        phi=0.1,
        qnm_backend=backend,
    )

    hp_j, hc_j = JaxRingdown.frequency_domain(
        frequencies=freqs_j,
        final_mass=mass,
        final_spin=spin,
        amplitudes=amplitudes,
        phases=phases,
        t0=t0,
        inclination=0.4,
        phi=0.1,
        qnm_backend=backend,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-12, atol=1e-14)


def test_single_mode_reference_parity():
    """Verify sampled and Fourier single mode reference functions match."""
    times_np = np.linspace(-0.005, 0.05, 200, dtype=np.float64)
    freqs_np = np.linspace(20.0, 600.0, 150, dtype=np.float64)

    mass = 62.0
    spin = 0.72
    amp = 0.85
    mode = (2, 2, 0)
    phase = 0.35
    t0 = 0.001
    inc = 0.45
    phi = 0.2

    # TD sampled reference
    shp_t, shc_t = torch_sampled_single_ref(
        times=torch.from_numpy(times_np),
        final_mass=mass,
        final_spin=spin,
        amplitude=amp,
        mode=mode,
        phase=phase,
        t0=t0,
        inclination=inc,
        phi=phi,
    )
    shp_j, shc_j = jax_sampled_single_ref(
        times=jnp.asarray(times_np),
        final_mass=mass,
        final_spin=spin,
        amplitude=amp,
        mode=mode,
        phase=phase,
        t0=t0,
        inclination=inc,
        phi=phi,
    )
    np.testing.assert_allclose(shp_t.numpy(), np.asarray(shp_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(shc_t.numpy(), np.asarray(shc_j), rtol=1e-12, atol=1e-14)

    # Alias check
    shp_alias, shc_alias = jax_single_ref(
        times=jnp.asarray(times_np),
        final_mass=mass,
        final_spin=spin,
        amplitude=amp,
        mode=mode,
        phase=phase,
        t0=t0,
        inclination=inc,
        phi=phi,
    )
    np.testing.assert_allclose(np.asarray(shp_j), np.asarray(shp_alias), rtol=1e-15, atol=1e-16)

    # FD Fourier reference
    fhp_t, fhc_t = torch_single_fourier_ref(
        frequencies=torch.from_numpy(freqs_np),
        final_mass=mass,
        final_spin=spin,
        amplitude=amp,
        mode=mode,
        phase=phase,
        t0=t0,
        inclination=inc,
        phi=phi,
    )
    fhp_j, fhc_j = jax_single_fourier_ref(
        frequencies=jnp.asarray(freqs_np),
        final_mass=mass,
        final_spin=spin,
        amplitude=amp,
        mode=mode,
        phase=phase,
        t0=t0,
        inclination=inc,
        phi=phi,
    )
    np.testing.assert_allclose(fhp_t.numpy(), np.asarray(fhp_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(fhc_t.numpy(), np.asarray(fhc_j), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("convention", ["single_mode", "pair_member", "native_pair"])
def test_energy_fraction_amplitude_parity(convention):
    """Verify conversion of energy fraction to strain amplitude matches."""
    mass = 65.0
    distance = 410.0
    energy_frac = 0.025
    omega = 1600.0
    tau = 0.0035

    t_amp = torch_energy_amp(mass, distance, energy_frac, omega, tau, convention=convention)
    j_amp = jax_energy_amp(mass, distance, energy_frac, omega, tau, convention=convention)

    np.testing.assert_allclose(t_amp.numpy(), np.asarray(j_amp), rtol=1e-13, atol=1e-16)

    # Batched inputs
    masses = [50.0, 70.0, 90.0]
    distances = [100.0, 300.0, 500.0]
    t_amp_b = torch_energy_amp(torch.tensor(masses), torch.tensor(distances), energy_frac, omega, tau, convention=convention)
    j_amp_b = jax_energy_amp(jnp.array(masses), jnp.array(distances), energy_frac, omega, tau, convention=convention)
    np.testing.assert_allclose(t_amp_b.numpy(), np.asarray(j_amp_b), rtol=1e-13, atol=1e-16)


def test_energy_fraction_waveform_parity():
    """Verify KerrRingdown waveform generation using energy_fraction."""
    times = np.linspace(0.0, 0.02, 100, dtype=np.float64)
    mass = 65.0
    spin = 0.7
    dist = 200.0
    eff = 0.02

    hp_t, hc_t = TorchRingdown.time_domain(
        times=torch.from_numpy(times),
        final_mass=mass,
        final_spin=spin,
        distance=dist,
        energy_fraction=eff,
        angular_basis="spheroidal",
    )

    hp_j, hc_j = JaxRingdown.time_domain(
        times=jnp.asarray(times),
        final_mass=mass,
        final_spin=spin,
        distance=dist,
        energy_fraction=eff,
        angular_basis="spheroidal",
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-12, atol=1e-14)


def test_kerr_ringdown_batching_parity():
    """Verify parameter batching parity across masses, spins, inclinations, and t0s."""
    B = 3
    times_np = np.linspace(0.0, 0.04, 150, dtype=np.float64)
    masses_np = np.array([55.0, 70.0, 85.0], dtype=np.float64)
    spins_np = np.array([0.3, 0.65, 0.85], dtype=np.float64)
    incs_np = np.array([0.2, 0.5, 0.9], dtype=np.float64)
    t0s_np = np.array([0.001, 0.0, 0.003], dtype=np.float64)

    hp_t_b, hc_t_b = TorchRingdown.time_domain(
        times=torch.from_numpy(times_np),
        final_mass=torch.from_numpy(masses_np),
        final_spin=torch.from_numpy(spins_np),
        amplitudes={(2, 2, 0): 1.0, (2, 2, 1): 0.3},
        phases={(2, 2, 0): 0.0, (2, 2, 1): 0.5},
        t0=torch.from_numpy(t0s_np),
        inclination=torch.from_numpy(incs_np),
    )

    hp_j_b, hc_j_b = JaxRingdown.time_domain(
        times=jnp.asarray(times_np),
        final_mass=jnp.asarray(masses_np),
        final_spin=jnp.asarray(spins_np),
        amplitudes={(2, 2, 0): 1.0, (2, 2, 1): 0.3},
        phases={(2, 2, 0): 0.0, (2, 2, 1): 0.5},
        t0=jnp.asarray(t0s_np),
        inclination=jnp.asarray(incs_np),
    )

    assert hp_j_b.shape == (B, 150)
    assert hc_j_b.shape == (B, 150)
    np.testing.assert_allclose(hp_t_b.numpy(), np.asarray(hp_j_b), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_t_b.numpy(), np.asarray(hc_j_b), rtol=1e-12, atol=1e-14)


def test_kerr_ringdown_evaluate_and_call_interface():
    """Verify high-level evaluate and __call__ dispatch interfaces."""
    times = jnp.linspace(0.0, 0.03, 100, dtype=jnp.float64)
    mass = 65.0
    spin = 0.7

    # Static evaluate
    hp_eval, hc_eval = JaxRingdown.evaluate(
        times=times,
        final_mass=mass,
        final_spin=spin,
    )
    hp_td, hc_td = JaxRingdown.time_domain(
        times=times,
        final_mass=mass,
        final_spin=spin,
    )
    np.testing.assert_allclose(np.asarray(hp_eval), np.asarray(hp_td), rtol=1e-15, atol=1e-16)

    # Instance __call__
    model = JaxRingdown()
    hp_inst, hc_inst = model(times=times, final_mass=mass, final_spin=spin)
    np.testing.assert_allclose(np.asarray(hp_inst), np.asarray(hp_td), rtol=1e-15, atol=1e-16)

    # get_ringdown_waveform function
    hp_fn, hc_fn = jax_get_waveform(final_mass=mass, final_spin=spin, times=times)
    np.testing.assert_allclose(np.asarray(hp_fn), np.asarray(hp_td), rtol=1e-15, atol=1e-16)


def test_kerr_ringdown_jax_jit():
    """Verify @jax.jit compilation and autodiff w.r.t final_mass and final_spin."""
    times_np = np.linspace(0.002, 0.04, 64, dtype=np.float64)
    times_j = jnp.asarray(times_np)
    times_t = torch.from_numpy(times_np)

    mass_val = 65.0
    spin_val = 0.68
    t0_val = 0.001
    inc_val = 0.5
    amplitudes = {(2, 2, 0): 1.0, (2, 2, 1): 0.4}

    # JAX JIT function
    @jax.jit
    def jax_loss(m, s):
        hp, hc = JaxRingdown.time_domain(
            times=times_j,
            final_mass=m,
            final_spin=s,
            amplitudes=amplitudes,
            t0=t0_val,
            inclination=inc_val,
        )
        return jnp.sum(hp ** 2 + hc ** 2)

    val = jax_loss(mass_val, spin_val)
    assert jnp.isfinite(val)

    # JAX Autodiff gradients
    grad_fn = jax.jit(jax.grad(jax_loss, argnums=(0, 1)))
    dm_j, ds_j = grad_fn(mass_val, spin_val)

    assert jnp.isfinite(dm_j)
    assert jnp.isfinite(ds_j)
    assert abs(float(dm_j)) > 0.0
    assert abs(float(ds_j)) > 0.0

    # PyTorch Autograd gradients for cross-framework verification
    m_t = torch.tensor(mass_val, dtype=torch.float64, requires_grad=True)
    s_t = torch.tensor(spin_val, dtype=torch.float64, requires_grad=True)
    hp_t, hc_t = TorchRingdown.time_domain(
        times=times_t,
        final_mass=m_t,
        final_spin=s_t,
        amplitudes=amplitudes,
        t0=t0_val,
        inclination=inc_val,
    )
    loss_t = (hp_t ** 2 + hc_t ** 2).sum()
    loss_t.backward()

    np.testing.assert_allclose(m_t.grad.item(), float(dm_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s_t.grad.item(), float(ds_j), rtol=1e-12, atol=1e-14)


def test_kerr_ringdown_frequency_domain_jit_and_autodiff():
    """Verify frequency-domain JIT compilation and autodiff w.r.t final_mass and final_spin."""
    freqs_np = np.linspace(50.0, 500.0, 64, dtype=np.float64)
    freqs_j = jnp.asarray(freqs_np)
    freqs_t = torch.from_numpy(freqs_np)

    mass_val = 65.0
    spin_val = 0.68
    amplitudes = {(2, 2, 0): 1.0, (2, 2, 1): 0.3}

    @jax.jit
    def jax_fd_loss(m, s):
        hp, hc = JaxRingdown.frequency_domain(
            frequencies=freqs_j,
            final_mass=m,
            final_spin=s,
            amplitudes=amplitudes,
            inclination=0.4,
        )
        return jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2)

    val = jax_fd_loss(mass_val, spin_val)
    assert jnp.isfinite(val)

    grad_fn = jax.jit(jax.grad(jax_fd_loss, argnums=(0, 1)))
    dm_j, ds_j = grad_fn(mass_val, spin_val)

    assert jnp.isfinite(dm_j)
    assert jnp.isfinite(ds_j)

    # PyTorch gradient comparison
    m_t = torch.tensor(mass_val, dtype=torch.float64, requires_grad=True)
    s_t = torch.tensor(spin_val, dtype=torch.float64, requires_grad=True)
    hp_t, hc_t = TorchRingdown.frequency_domain(
        frequencies=freqs_t,
        final_mass=m_t,
        final_spin=s_t,
        amplitudes=amplitudes,
        inclination=0.4,
    )
    loss_t = (torch.abs(hp_t) ** 2 + torch.abs(hc_t) ** 2).sum()
    loss_t.backward()

    np.testing.assert_allclose(m_t.grad.item(), float(dm_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s_t.grad.item(), float(ds_j), rtol=1e-12, atol=1e-14)


def test_kerr_ringdown_vmap():
    """Verify that KerrRingdown supports jax.vmap across mass and spin arrays."""
    times = jnp.linspace(0.0, 0.03, 80, dtype=jnp.float64)
    masses = jnp.array([50.0, 65.0, 80.0], dtype=jnp.float64)
    spins = jnp.array([0.3, 0.6, 0.85], dtype=jnp.float64)

    def single_eval(m, s):
        return JaxRingdown.time_domain(
            times=times,
            final_mass=m,
            final_spin=s,
            amplitudes={(2, 2, 0): 1.0},
        )

    vmapped_fn = jax.vmap(single_eval, in_axes=(0, 0))
    hp_v, hc_v = vmapped_fn(masses, spins)

    assert hp_v.shape == (3, 80)
    assert hc_v.shape == (3, 80)

    # Compare each row against unmapped evaluation
    for i in range(3):
        hp_s, hc_s = single_eval(masses[i], spins[i])
        np.testing.assert_allclose(np.asarray(hp_v[i]), np.asarray(hp_s), rtol=1e-15, atol=1e-16)
        np.testing.assert_allclose(np.asarray(hc_v[i]), np.asarray(hc_s), rtol=1e-15, atol=1e-16)


def test_kerr_ringdown_pytree_registration():
    """Verify that KerrRingdown is a valid JAX PyTree node."""
    model = JaxRingdown(qnm_backend="berti", angular_basis="spherical")
    leaves, treedef = jax.tree_util.tree_flatten(model)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, JaxRingdown)
    assert reconstructed.qnm_backend == "berti"
    assert reconstructed.angular_basis == "spherical"
