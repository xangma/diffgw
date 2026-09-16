"""Differentiable circular solve and leading adiabatic radial momentum in JAX.

Buonanno--Damour, arXiv:gr-qc/0001013v2, Eqs. (4.1), (4.13)-(4.16),
and Damour--Nagar, arXiv:0704.3550v2, Eqs. (6)-(11).
"""

from typing import Tuple
import jax
import jax.numpy as jnp

from jaxwave.approximants.eob.hamiltonian import EOBHamiltonian
from jaxwave.approximants.eob.radiation_reaction import EOBRadiationReaction
from jaxwave.conversions import as_array


def _solve_quasicircular_initial_conditions_single(
    hamiltonian: EOBHamiltonian,
    radiation_reaction: EOBRadiationReaction,
    omega_start: jnp.ndarray,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve initial circular conditions for a single frequency and parameter set."""
    # 1. Initial guess from Schwarzschild circular orbit
    r = omega_start ** (-2.0 / 3.0)
    pphi = jnp.sqrt(r / jnp.maximum(1.0 - 3.0 / r, 0.1))

    # 2. 2x2 Newton-Raphson iteration
    for _ in range(max_iter):
        def res_fn(r_val, pphi_val):
            h_eff = hamiltonian.effective_hamiltonian(r_val, 0.0, pphi_val)
            h_real = hamiltonian.real_hamiltonian_from_eff(h_eff)
            dh_dr = jax.grad(lambda r_: hamiltonian.effective_hamiltonian(r_, 0.0, pphi_val))(r_val)
            dh_dpp = jax.grad(lambda pp_: hamiltonian.effective_hamiltonian(r_val, 0.0, pp_))(pphi_val)
            omega = dh_dpp / h_real
            return jnp.array([omega - omega_start, dh_dr])

        residual = res_fn(r, pphi)
        J_cols = jax.jacobian(res_fn, argnums=(0, 1))(r, pphi)
        J = jnp.column_stack(J_cols)
        step = jnp.linalg.solve(J, -residual)
        r = jnp.maximum(r + step[0], 3.0)
        pphi = jnp.maximum(pphi + step[1], 0.1)

        if jnp.max(jnp.abs(residual)) < tol:
            break

    # 3. Leading adiabatic radial momentum correction
    dh_dr_fn = lambda r_, pp_: jax.grad(lambda r__: hamiltonian.effective_hamiltonian(r__, 0.0, pp_))(r_)
    d2h_dr2 = jax.grad(lambda r_: dh_dr_fn(r_, pphi))(r)
    d2h_drdj = jax.grad(lambda pp_: dh_dr_fn(r, pp_))(pphi)
    d2h_dpr2 = jax.grad(lambda pr_: jax.grad(lambda pr__: hamiltonian.effective_hamiltonian(r, pr__, pphi))(pr_))(0.0)

    dp_phi_dr = -d2h_dr2 / d2h_drdj
    _, f_phi = radiation_reaction.radiation_reaction_forces(
        omega_start, 0.0, pphi, r=r, hamiltonian=hamiltonian,
    )
    _, _, xi = hamiltonian.metric_potentials(r)
    h_eff = hamiltonian.effective_hamiltonian(r, 0.0, pphi)
    h_real = hamiltonian.real_hamiltonian_from_eff(h_eff)
    velocity_per_momentum = xi * d2h_dpr2 / h_real
    pr_star_corr = f_phi / (velocity_per_momentum * dp_phi_dr)

    return r, pr_star_corr, pphi


def find_quasicircular_initial_conditions(
    hamiltonian: EOBHamiltonian,
    radiation_reaction: EOBRadiationReaction,
    omega_start: jnp.ndarray,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (r, p_r*, p_phi) at a positive dimensionless orbital frequency."""
    omega_start = as_array(omega_start, dtype=jnp.float64)
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if jnp.any(omega_start <= 0):
        raise ValueError("omega_start must be positive")

    if omega_start.ndim == 0 and hamiltonian.nu.ndim == 0:
        return _solve_quasicircular_initial_conditions_single(
            hamiltonian, radiation_reaction, omega_start, max_iter, tol,
        )

    # Batched inputs: broadcast shapes and vectorize
    batch_shape = jnp.broadcast_shapes(
        omega_start.shape, hamiltonian.nu.shape, hamiltonian.chi1.shape, hamiltonian.chi2.shape
    )
    w_flat = jnp.broadcast_to(omega_start, batch_shape).reshape(-1)
    nu_flat = jnp.broadcast_to(hamiltonian.nu, batch_shape).reshape(-1)
    c1_flat = jnp.broadcast_to(hamiltonian.chi1, batch_shape).reshape(-1)
    c2_flat = jnp.broadcast_to(hamiltonian.chi2, batch_shape).reshape(-1)

    r_list, pr_list, pp_list = [], [], []
    for w, nu, c1, c2 in zip(w_flat, nu_flat, c1_flat, c2_flat):
        ham_i = type(hamiltonian)(nu, c1, c2)
        rr_i = type(radiation_reaction)(nu, c1, c2)
        r_i, pr_i, pp_i = _solve_quasicircular_initial_conditions_single(
            ham_i, rr_i, w, max_iter, tol
        )
        r_list.append(r_i)
        pr_list.append(pr_i)
        pp_list.append(pp_i)

    r_out = jnp.array(r_list).reshape(batch_shape)
    pr_out = jnp.array(pr_list).reshape(batch_shape)
    pp_out = jnp.array(pp_list).reshape(batch_shape)
    return r_out, pr_out, pp_out


__all__ = [
    "find_quasicircular_initial_conditions",
]
