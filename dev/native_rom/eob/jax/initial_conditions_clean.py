"""Development-only traceable scalar quasicircular initial-condition solve.

The algebra and Newton update follow the existing JAX initial-condition
solver. A fixed iteration count and masked converged updates make the
physical input-to-initial-state path available to JIT and autodiff.
"""

import jax
import jax.numpy as jnp


def traceable_quasicircular_initial_conditions(
    hamiltonian, radiation_reaction, omega_start, *,
    max_iter=25, tol=1e-8,
):
    """Return scalar radius, radial momentum, and angular momentum.

    The caller must supply a finite, positive frequency and physical model
    parameters. This development route does not provide a release contract.
    """
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    omega_start = jnp.asarray(omega_start, dtype=jnp.float64)
    if omega_start.ndim != 0 or hamiltonian.nu.ndim != 0:
        raise ValueError("traceable initial conditions require scalar inputs")

    def residual(r, pphi):
        h_eff = hamiltonian.effective_hamiltonian(r, 0.0, pphi)
        h_real = hamiltonian.real_hamiltonian_from_eff(h_eff)
        dh_dr = jax.grad(
            lambda radius: hamiltonian.effective_hamiltonian(radius, 0.0, pphi)
        )(r)
        dh_dpp = jax.grad(
            lambda angular: hamiltonian.effective_hamiltonian(r, 0.0, angular)
        )(pphi)
        return jnp.stack((dh_dpp / h_real - omega_start, dh_dr))

    r0 = omega_start ** (-2.0 / 3.0)
    pphi0 = jnp.sqrt(r0 / jnp.maximum(1.0 - 3.0 / r0, 0.1))

    def advance(_, values):
        r, pphi, done = values
        error = residual(r, pphi)
        columns = jax.jacobian(residual, argnums=(0, 1))(r, pphi)
        jacobian = jnp.column_stack(columns)
        step = jnp.linalg.solve(jacobian, -error)
        return (
            jnp.where(done, r, jnp.maximum(r + step[0], 3.0)),
            jnp.where(done, pphi, jnp.maximum(pphi + step[1], 0.1)),
            done | (jnp.max(jnp.abs(error)) < tol),
        )

    r, pphi, _ = jax.lax.fori_loop(
        0, max_iter, advance, (r0, pphi0, jnp.asarray(False))
    )

    def dh_dr(radius, angular):
        return jax.grad(
            lambda value: hamiltonian.effective_hamiltonian(value, 0.0, angular)
        )(radius)
    d2h_dr2 = jax.grad(lambda radius: dh_dr(radius, pphi))(r)
    d2h_drdj = jax.grad(lambda angular: dh_dr(r, angular))(pphi)
    d2h_dpr2 = jax.grad(lambda momentum: jax.grad(
        lambda value: hamiltonian.effective_hamiltonian(r, value, pphi)
    )(momentum))(0.0)
    dp_phi_dr = -d2h_dr2 / d2h_drdj
    _, f_phi = radiation_reaction.radiation_reaction_forces(
        omega_start, 0.0, pphi, r=r, hamiltonian=hamiltonian
    )
    _, _, xi = hamiltonian.metric_potentials(r)
    h_eff = hamiltonian.effective_hamiltonian(r, 0.0, pphi)
    h_real = hamiltonian.real_hamiltonian_from_eff(h_eff)
    radial_velocity_per_momentum = xi * d2h_dpr2 / h_real
    pr_star = f_phi / (radial_velocity_per_momentum * dp_phi_dr)
    return r, pr_star, pphi


__all__ = ["traceable_quasicircular_initial_conditions"]
