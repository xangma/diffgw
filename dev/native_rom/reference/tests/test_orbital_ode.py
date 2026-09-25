"""Independent scalar Hamilton equations on supplied aligned-spin states."""

import jax
import jax.numpy as jnp
import pytest
import torch

from dev.native_rom.reference.orbital_ode import (
    supplied_force_vector_field, zero_radial_vector_field,
)
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxHamiltonian
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as TorchHamiltonian

jax.config.update("jax_enable_x64", True)


ROWS = (
    (30., 20., 0., 0., 8., .1, 3.2),
    (30., 20., .6, -.2, 9., -.7, 3.),
    (45., 5., -.8, .3, 7., 1.2, 2.6),
    (20., 20., -.9, .7, 12., 3.4, 3.5),
)
TORQUE = -.0007


def torch_scalar(value):
    return torch.tensor(value, dtype=torch.float64)


def jax_scalar(value):
    return jnp.asarray(value, dtype=jnp.float64)


class SuppliedTorque:
    def radiation_reaction_forces(self, *, omega, pr_star, pphi, r, hamiltonian):
        return 0*omega, 0*omega + TORQUE


class SuppliedForces:
    def __init__(self, radial_force, torque):
        self.radial_force = radial_force
        self.torque = torque

    def radiation_reaction_forces(self, *, omega, pr_star, pphi, r, hamiltonian):
        return 0*omega + self.radial_force, 0*omega + self.torque


@pytest.mark.parametrize("m1,m2,s1,s2,r,phi,j", ROWS)
@pytest.mark.parametrize("frontend", ["torch", "jax"])
def test_zero_radial_vector_field_against_paper_reference(frontend, m1, m2, s1, s2,
                                                           r, phi, j):
    expected = zero_radial_vector_field(m1, m2, s1, s2, r, phi, j, TORQUE)
    nu = m1*m2/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    if frontend == "torch":
        hamiltonian = TorchHamiltonian(torch_scalar(nu), torch_scalar(s1), torch_scalar(s2),
                                       delta=torch_scalar(delta))
        actual = TorchIntegrator(hamiltonian, SuppliedTorque()).vector_field(
            torch_scalar([r, phi, 0., j]))
    else:
        hamiltonian = JaxHamiltonian(jax_scalar(nu), jax_scalar(s1), jax_scalar(s2),
                                     delta=jax_scalar(delta))
        actual = JaxIntegrator(hamiltonian, SuppliedTorque()).vector_field(
            jax_scalar([r, phi, 0., j]))
    for value, reference in zip(actual, expected):
        assert float(value) == pytest.approx(float(reference), rel=2e-10, abs=2e-12)


@pytest.mark.parametrize("m1,m2,s1,s2,r,phi,pr,j,radial_force,torque", [
    (30., 20., 0., 0., 8., .1, -.01, 3.2, 0., -.0007),
    (30., 20., .6, -.2, 9., -.7, .025, 3., .00003, -.0005),
    (45., 5., -.8, .3, 7., 1.2, -.03, 2.6, -.00002, -.0008),
    (20., 20., -.9, .7, 12., 3.4, .04, 3.5, .00001, -.0004),
])
@pytest.mark.parametrize("frontend", ["torch", "jax"])
def test_nonzero_radial_vector_field_against_paper_reference(
    frontend, m1, m2, s1, s2, r, phi, pr, j, radial_force, torque,
):
    expected = supplied_force_vector_field(
        m1, m2, s1, s2, r, phi, pr, j, radial_force, torque)
    nu = m1*m2/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    forces = SuppliedForces(radial_force, torque)
    if frontend == "torch":
        hamiltonian = TorchHamiltonian(torch_scalar(nu), torch_scalar(s1), torch_scalar(s2),
                                       delta=torch_scalar(delta))
        actual = TorchIntegrator(hamiltonian, forces).vector_field(
            torch_scalar([r, phi, pr, j]))
    else:
        hamiltonian = JaxHamiltonian(jax_scalar(nu), jax_scalar(s1), jax_scalar(s2),
                                     delta=jax_scalar(delta))
        actual = JaxIntegrator(hamiltonian, forces).vector_field(
            jax_scalar([r, phi, pr, j]))
    for value, reference in zip(actual, expected):
        assert float(value) == pytest.approx(float(reference), rel=2e-10, abs=2e-12)
