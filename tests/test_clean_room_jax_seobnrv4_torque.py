"""Same-state JAX/Torch flux-to-torque adapter and JAX transform checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxHam
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as TorchHam

jax.config.update("jax_enable_x64", True)


def _torch(value):
    return torch.as_tensor(value, dtype=torch.float64)


def test_hamiltonian_adapter_follows_float32_state_dtype():
    ham = JaxHam(.24, .2, -.1, delta=.2)
    model = JaxFlux(spin_truncation="tabulated")
    state = jnp.asarray([12., 0., -.02, 3.8], dtype=jnp.float32)
    quantities = model.from_hamiltonian(state, ham)
    assert all(value.dtype == jnp.float32 for value in quantities.values())
    assert model.energy_flux(state, **quantities).dtype == jnp.float32
    radial, torque = model.radiation_reaction_forces(
        quantities["omega"], state[2], state[3], r=state[0], hamiltonian=ham)
    assert radial.dtype == torque.dtype == jnp.float32


@pytest.mark.parametrize("delta,chi1,chi2,r,pr_star,pphi", [
    (.2, 0., 0., 12., -.02, 3.8),
    (.6, .7, -.3, 10., -.015, 3.4),
])
def test_same_state_torch_jax_torque(delta, chi1, chi2, r, pr_star, pphi):
    nu = (1-delta**2)/4
    jham = JaxHam(nu, chi1, chi2, delta=delta)
    tham = TorchHam(_torch(nu), _torch(chi1), _torch(chi2), delta=_torch(delta))
    jmodel = JaxFlux(spin_truncation="tabulated")
    tmodel = TorchFlux(spin_truncation="tabulated")
    jstate = jnp.asarray([r, 0., pr_star, pphi])
    tstate = _torch([r, 0., pr_star, pphi])

    jq = jmodel.from_hamiltonian(jstate, jham)
    tq = tmodel.from_hamiltonian(tstate, tham)
    for key in ("omega", "omega_zero_radial", "heff_over_mu", "hreal_over_M"):
        np.testing.assert_allclose(jq[key], tq[key].detach().numpy(), rtol=2e-13)
    # An integrator-supplied frequency survives the adapter; omega_0 is
    # independently recomputed at the same radius and p_phi.
    supplied = jq["omega"]*1.01
    jr, jphi = jmodel.radiation_reaction_forces(
        supplied, jstate[2], jstate[3], r=jstate[0], hamiltonian=jham)
    tr, tphi = tmodel.radiation_reaction_forces(
        tq["omega"]*1.01, tstate[2], tstate[3], r=tstate[0], hamiltonian=tham)
    np.testing.assert_allclose(jphi, tphi.detach().numpy(), rtol=3e-12)
    np.testing.assert_allclose(jr, tr.detach().numpy(), rtol=3e-12)
    expected_flux = jmodel.energy_flux(jstate, **(jq | {"omega": supplied}))
    np.testing.assert_allclose(-jphi*nu*supplied, expected_flux, rtol=2e-14)
    np.testing.assert_allclose(jr, pr_star/pphi*jphi, rtol=2e-14)


@pytest.mark.parametrize("trajectory_length", [None, 3])
def test_batched_parameter_energy_map_and_jitted_forces(trajectory_length):
    delta = jnp.asarray([.2, .6])
    nu = (1-delta**2)/4
    ham = JaxHam(nu, jnp.asarray([.4, -.2]), jnp.asarray([-.1, .3]), delta=delta)
    model = JaxFlux(spin_truncation="tabulated")
    radii = (jnp.asarray([15., 16.]) if trajectory_length is None else
             jnp.asarray([[15., 13., 11.], [16., 14., 12.]]))
    pr_star = jnp.full_like(radii, -.02)
    pphi = jnp.sqrt(radii)+.6
    state = jnp.stack((radii, jnp.zeros_like(radii), pr_star, pphi), axis=-1)
    quantities = model.from_hamiltonian(state, ham)
    assert quantities["omega"].shape == radii.shape
    assert quantities["hreal_over_M"].shape == radii.shape
    mass_ratio = nu if trajectory_length is None else nu[:, None]
    expected_real = jnp.sqrt(1+2*mass_ratio*(quantities["heff_over_mu"]-1))
    np.testing.assert_allclose(quantities["hreal_over_M"], expected_real, rtol=2e-14)

    supplied = quantities["omega"]*1.01
    jit_forces = jax.jit(lambda w: model.radiation_reaction_forces(
        w, pr_star, pphi, r=radii, hamiltonian=ham))
    radial, torque = jit_forces(supplied)
    expected_flux = model.energy_flux(state, **(quantities | {"omega": supplied}))
    np.testing.assert_allclose(-torque*mass_ratio*supplied, expected_flux, rtol=3e-14)
    np.testing.assert_allclose(radial, pr_star/pphi*torque, rtol=2e-14)

    # Compare each row with an independently constructed scalar-parameter
    # Hamiltonian to catch accidental summation across the parameter batch.
    for row in range(2):
        single = JaxHam(nu[row], ham.chi1[row], ham.chi2[row], delta=delta[row])
        single_state = state[row]
        single_q = model.from_hamiltonian(single_state, single)
        np.testing.assert_allclose(quantities["omega"][row], single_q["omega"], rtol=3e-14)
        np.testing.assert_allclose(quantities["hreal_over_M"][row],
                                   single_q["hreal_over_M"], rtol=3e-14)


def test_jitted_radius_gradient_and_override_validation():
    ham = JaxHam(.24, .4, -.2, delta=.2)
    model = JaxFlux(spin_truncation="tabulated")
    pr_star, pphi = jnp.asarray(-.02), jnp.asarray(3.8)

    def torque_at_radius(r):
        state = jnp.stack((r, r*0, pr_star, pphi))
        frequency = model.from_hamiltonian(state, ham)["omega"]
        return model.radiation_reaction_forces(
            frequency, pr_star, pphi, r=r, hamiltonian=ham)[1]

    radius = jnp.asarray(12.)
    derivative = jax.jit(jax.grad(torque_at_radius))(radius)
    step = 2e-4
    finite_difference = (torque_at_radius(radius+step)-torque_at_radius(radius-step))/(2*step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-8)
    assert np.isfinite(derivative)

    state = jnp.asarray([12., 0., -.02, 3.8])
    cast_override = jax.jit(lambda w: model.from_hamiltonian(state, ham, omega=w)["omega"])(
        jnp.asarray(.025, dtype=jnp.float32))
    assert cast_override.dtype == jnp.float64
    np.testing.assert_allclose(cast_override, np.float32(.025), rtol=0, atol=0)
    with pytest.raises(TypeError):
        model.from_hamiltonian(state, ham, omega=jnp.asarray(.025+.01j))
    with pytest.raises(ValueError):
        model.from_hamiltonian(state, ham, omega=jnp.asarray([.025, .026]))
    override_under_jit = jax.jit(lambda w: model.from_hamiltonian(state, ham, omega=w)["omega"])
    with pytest.raises(TypeError, match="real numeric dtype"):
        override_under_jit(jnp.asarray(.025+.01j))
    with pytest.raises(ValueError, match="broadcast to state batch dimensions"):
        override_under_jit(jnp.asarray([.025, .026]))
    with pytest.raises(ValueError, match="finite omega>0"):
        model.radiation_reaction_forces(
            jnp.asarray(-.01), pr_star, pphi, r=radius, hamiltonian=ham)
    invalid = jax.jit(lambda w: model.radiation_reaction_forces(
        w, pr_star, pphi, r=radius, hamiltonian=ham)[1])(jnp.asarray(-.01))
    assert np.isnan(invalid)
