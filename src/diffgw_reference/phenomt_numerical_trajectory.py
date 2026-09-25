"""Independent event, ODE, transport, and angle reference for PhenomTP.

The physical spin and orbital-axis right-hand side is delegated only to the
paper-derived scalar evaluator in :mod:`phenomt_spin_evolution`.  This module
supplies an otherwise independent SciPy root solve, adaptive DOP853 evolution,
minimal-rotation tangent transport, and scalar Euler-angle reconstruction.

Times are dimensionless ``u=t/M`` and the caller supplies the PhenomT carrier
only as a scalar ``v(u)`` function.  No production ODE, interpolation, angle,
or continuation helper is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .phenomt_spin_evolution import evaluate_spin_derivatives


Array = np.ndarray


def _unit(vector: Array) -> Array:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def initial_state(mass1_fraction, mass2_fraction, spin1, spin2) -> Array:
    """Return ``(Lhat, S1/M^2, S2/M^2, e1)`` at the reference frequency."""
    x1, x2 = float(mass1_fraction), float(mass2_fraction)
    if x1 <= 0 or x2 <= 0 or abs(x1 + x2 - 1) > 1e-12:
        raise ValueError("positive mass fractions must sum to one")
    chi1, chi2 = np.asarray(spin1, dtype=float), np.asarray(spin2, dtype=float)
    if chi1.shape != (3,) or chi2.shape != (3,):
        raise ValueError("spin vectors must have shape (3,)")
    return np.concatenate(([0.0, 0.0, 1.0], x1**2 * chi1, x2**2 * chi2, [1.0, 0.0, 0.0]))


def find_reference_time(
    velocity_at_time: Callable[[float], float],
    reference_velocity: float,
    bracket: tuple[float, float],
    *,
    xtol: float = 1e-12,
) -> float:
    """Find the carrier crossing ``v(u)=reference_velocity`` by Brent's method."""
    lower, upper = map(float, bracket)
    if not lower < upper or reference_velocity <= 0:
        raise ValueError("the root bracket must increase and the reference velocity must be positive")
    return float(brentq(
        lambda time: float(velocity_at_time(time)) - reference_velocity,
        lower,
        upper,
        xtol=xtol,
        rtol=4 * np.finfo(float).eps,
    ))


@dataclass(frozen=True)
class NumericalTrajectory:
    """Dense independent trajectory and solver diagnostics."""

    reference_time: float
    end_time: float
    reference_velocity: float
    initial: Array
    nfev: int
    njev: int
    nlu: int
    _dense_output: Callable[[Array], Array]

    def evaluate(self, times) -> Array:
        values = np.asarray(times, dtype=float)
        if np.any(values < self.reference_time) or np.any(values > self.end_time):
            raise ValueError("requested times lie outside the integrated interval")
        return np.moveaxis(np.asarray(self._dense_output(values), dtype=float), 0, -1)


def integrate_trajectory(
    velocity_at_time: Callable[[float], float],
    reference_velocity: float,
    root_bracket: tuple[float, float],
    end_time: float,
    mass1_fraction: float,
    mass2_fraction: float,
    spin1,
    spin2,
    *,
    rtol: float = 2e-11,
    atol: float = 2e-13,
    dps: int = 50,
    max_step: float = np.inf,
) -> NumericalTrajectory:
    """Solve the published trajectory from the reference event to ``end_time``."""
    reference_time = find_reference_time(velocity_at_time, reference_velocity, root_bracket)
    if end_time <= reference_time:
        raise ValueError("end_time must follow the reference event")
    initial = initial_state(mass1_fraction, mass2_fraction, spin1, spin2)

    def rhs(time, state):
        lhat, s1, s2, tangent = np.split(np.asarray(state, dtype=float), 4)
        # Explicit Runge--Kutta stage states do not lie exactly on the unit
        # sphere.  Evaluate the constrained paper RHS on the normalized axis;
        # the accepted dense state remains untouched so invariant drift stays
        # observable rather than being hidden by per-step projection.
        lhat_on_manifold = _unit(lhat)
        reference = evaluate_spin_derivatives(
            velocity_at_time(float(time)),
            lhat_on_manifold,
            s1,
            s2,
            mass1_fraction,
            mass2_fraction,
            include_n3lo_ss=False,
            dps=dps,
        )
        dlhat = np.asarray(reference.dlhat, dtype=float)
        ds1 = np.asarray(reference.ds1, dtype=float)
        ds2 = np.asarray(reference.ds2, dtype=float)
        dtangent = np.cross(np.cross(lhat_on_manifold, dlhat), tangent)
        return np.concatenate((dlhat, ds1, ds2, dtangent))

    solution = solve_ivp(
        rhs,
        (reference_time, float(end_time)),
        initial,
        method="DOP853",
        dense_output=True,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"independent DOP853 evolution failed: {solution.message}")
    return NumericalTrajectory(
        reference_time=reference_time,
        end_time=float(end_time),
        reference_velocity=float(reference_velocity),
        initial=initial,
        nfev=int(solution.nfev),
        njev=int(solution.njev),
        nlu=int(solution.nlu),
        _dense_output=solution.sol,
    )


def inertial_basis(initial: Array, orbital_angular_momentum: float) -> tuple[Array, Array, Array]:
    """Construct the fixed J-frame used to express Euler angles."""
    lhat, spin1, spin2, _ = np.split(np.asarray(initial, dtype=float), 4)
    jhat = _unit(spin1 + spin2 + float(orbital_angular_momentum) * lhat)
    trial = np.array([1.0, 0.0, 0.0]) if abs(jhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ex = _unit(trial - np.dot(trial, jhat) * jhat)
    ey = np.cross(jhat, ex)
    return jhat, ex, ey


def euler_angles(states: Array, basis: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
    """Recover active ``Rz(alpha) Ry(beta) Rz(gamma)`` angles from a triad."""
    values = np.asarray(states, dtype=float)
    if values.shape[-1] != 12:
        raise ValueError("states must end in twelve trajectory components")
    lhat, _, _, tangent = np.split(values, 4, axis=-1)
    lhat = _unit(lhat)
    tangent = _unit(tangent - np.sum(tangent * lhat, axis=-1, keepdims=True) * lhat)
    jhat, ex, ey = (np.asarray(vector, dtype=float) for vector in basis)
    lx, ly, lz = (np.sum(lhat * vector, axis=-1) for vector in (ex, ey, jhat))
    rho = np.hypot(lx, ly)
    regular = rho > 1e-14
    ca = np.divide(lx, rho, out=np.ones_like(rho), where=regular)
    sa = np.divide(ly, rho, out=np.zeros_like(rho), where=regular)
    alpha = np.arctan2(sa, ca)
    beta = np.arctan2(np.where(regular, rho, 0.0), lz)
    etheta = lz[..., None] * (ca[..., None] * ex + sa[..., None] * ey) - rho[..., None] * jhat
    ephi = -sa[..., None] * ex + ca[..., None] * ey
    gamma = np.arctan2(np.sum(tangent * ephi, axis=-1), np.sum(tangent * etheta, axis=-1))
    return alpha, beta, gamma


def rotation_matrix(alpha, beta, gamma) -> Array:
    """Return the active ``Rz(alpha) Ry(beta) Rz(gamma)`` rotation."""
    alpha, beta, gamma = np.broadcast_arrays(alpha, beta, gamma)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return np.stack((
        ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb,
        sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb,
        -sb * cg, sb * sg, cb,
    ), axis=-1).reshape(alpha.shape + (3, 3))


__all__ = [
    "NumericalTrajectory",
    "euler_angles",
    "find_reference_time",
    "inertial_basis",
    "initial_state",
    "integrate_trajectory",
    "rotation_matrix",
]
