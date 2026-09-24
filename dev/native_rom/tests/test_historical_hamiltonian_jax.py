"""The historical potential-derivative convention stays confined to dev JAX."""

import jax
import jax.numpy as jnp
import pytest
import torch

from dev.native_rom.eob.jax.historical_hamiltonian import (
    HistoricalLogDerivative as JaxHistoricalLogDerivative,
)
from dev.native_rom.eob.torch.historical_hamiltonian import (
    HistoricalLogDerivative as TorchHistoricalLogDerivative,
)
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("m1,m2,chi1,chi2,r", [
    (30., 20., 0., 0., 9.),
    (30., 20., .4, .3, 6.5),
    (45., 5., -.8, .3, 11.),
])
def test_internal_log_derivative_matches_pinned_source_convention(m1, m2, chi1, chi2, r):
    nu = m1*m2/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    args = (jnp.asarray(nu), jnp.asarray(chi1), jnp.asarray(chi2))
    exact = SEOBNRv4Hamiltonian(*args, delta=jnp.asarray(delta))
    historical = JaxHistoricalLogDerivative(*args, delta=jnp.asarray(delta))
    radius = jnp.asarray(r)
    current = exact.deformed_kerr_potentials(radius)
    old = jax.jit(historical.deformed_kerr_potentials)(radius)

    for name in ("delta_t", "delta_r", "lambda_t", "xi"):
        assert float(getattr(old, name)) == pytest.approx(float(getattr(current, name)), abs=2e-13)

    k, _, _ = historical.calibration_parameters(radius)
    a, _, _ = historical.spin_combinations(radius)
    _, d1, d2, d3, d4, d5_constant, d5_log = historical.radial_coefficients(radius)
    u = 1.0/radius
    m = k*nu - 1.0
    d5 = d5_constant + d5_log*jnp.log(u)
    polynomial = u*(d1 + u*(d2 + u*(d3 + u*(d4 + u*d5))))
    horizon = a**2*u**2 + 2.0*u/m + 1.0/m**2
    missing = horizon*nu*d5_log*u**4/(1.0 + polynomial)
    assert float(old.delta_t_prime-current.delta_t_prime) == pytest.approx(float(missing), rel=2e-9)
    assert float(old.lambda_t_prime-current.lambda_t_prime) == pytest.approx(
        -float(a**2*missing), rel=2e-9, abs=4e-13)
    mathematical = jax.grad(lambda x: exact.deformed_kerr_potentials(x).delta_t)(radius)
    assert float(current.delta_t_prime) == pytest.approx(float(mathematical), rel=2e-10)
    assert abs(float(old.delta_t_prime-mathematical)) > 1e-9


@pytest.mark.parametrize("m1,m2,chi1,chi2,r,pr_star,pphi", [
    (30., 20., 0., 0., 9., .002, 3.4),
    (30., 20., .4, .3, 6.5, .01, 3.1),
    (45., 5., -.8, .3, 11., -.003, 3.7),
])
def test_historical_hamiltonian_torch_jax_state_parity(
        m1, m2, chi1, chi2, r, pr_star, pphi):
    nu = m1*m2/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    jax_ham = JaxHistoricalLogDerivative(
        jnp.asarray(nu), jnp.asarray(chi1), jnp.asarray(chi2),
        delta=jnp.asarray(delta))
    torch_ham = TorchHistoricalLogDerivative(
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
        delta=torch.tensor(delta, dtype=torch.float64))
    jax_metric = jax.jit(jax_ham.deformed_kerr_potentials)(jnp.asarray(r))
    torch_metric = torch_ham.deformed_kerr_potentials(torch.tensor(r, dtype=torch.float64))
    for name in ("delta_t", "delta_r", "delta_t_prime", "lambda_t",
                 "lambda_t_prime", "xi"):
        assert float(getattr(jax_metric, name)) == pytest.approx(
            float(getattr(torch_metric, name)), rel=2e-11, abs=2e-11)
    jax_eff = jax.jit(jax_ham.effective_hamiltonian)(
        jnp.asarray(r), jnp.asarray(pr_star), jnp.asarray(pphi))
    torch_eff = torch_ham.effective_hamiltonian(
        torch.tensor(r, dtype=torch.float64), torch.tensor(pr_star, dtype=torch.float64),
        torch.tensor(pphi, dtype=torch.float64))
    assert float(jax_eff) == pytest.approx(float(torch_eff), rel=2e-11, abs=2e-11)
