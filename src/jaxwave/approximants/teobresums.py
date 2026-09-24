"""Published nonspinning TEOB point-mass and numerical components.

The orbital potential follows Nagar et al., arXiv:1506.08457v2 Eqs. (1)--(3).
The tortoise kinetic terms follow Damour--Nagar, arXiv:0704.3550v2 Eqs. (2)--(11).
The standalone centrifugal-radius map is Nagar et al., arXiv:1806.01772v2
Eqs. (24)--(25), with explicit quadrupole coefficients (black-hole default 1).
No complete TEOBResumS waveform, heuristic tides or spin dynamics are exposed.
"""

from typing import Any, Dict, Optional, Tuple, Union
import math
import numpy as np
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from jaxwave.conversions import (
    as_array,
    component_masses_to_eta,
    component_masses_to_delta,
)


def _interp1d(x_new: jnp.ndarray, x_old: jnp.ndarray, y_old: jnp.ndarray) -> jnp.ndarray:
    """Interpolate along the final axis, with zero outside the sample interval."""
    scalar_query = x_new.ndim == 0
    if scalar_query:
        x_new = jnp.expand_dims(x_new, -1)
    batch_shape = jnp.broadcast_shapes(x_new.shape[:-1], x_old.shape[:-1], y_old.shape[:-1])
    x_old_b = jnp.broadcast_to(x_old, (*batch_shape, x_old.shape[-1]))
    y_old_b = jnp.broadcast_to(y_old, (*batch_shape, y_old.shape[-1]))
    x_new_b = jnp.broadcast_to(x_new, (*batch_shape, x_new.shape[-1]))

    idx = jnp.searchsorted(x_old_b, x_new_b, side="left")
    idx = jnp.maximum(jnp.minimum(idx, x_old_b.shape[-1] - 1), 1)

    x0 = jnp.take_along_axis(x_old_b, idx - 1, axis=-1)
    x1 = jnp.take_along_axis(x_old_b, idx, axis=-1)
    y0 = jnp.take_along_axis(y_old_b, idx - 1, axis=-1)
    y1 = jnp.take_along_axis(y_old_b, idx, axis=-1)

    slope = (y1 - y0) / (x1 - x0 + 1e-30)
    y_new = y0 + slope * (x_new_b - x0)

    mask_low = x_new_b < x_old_b[..., :1]
    mask_high = x_new_b > x_old_b[..., -1:]
    y_new = jnp.where(mask_low | mask_high, 0.0, y_new)
    return jnp.squeeze(y_new, axis=-1) if scalar_query else y_new


def _piecewise_linear_fourier(
    signal: jnp.ndarray,
    event_time: jnp.ndarray,
    dt_sec: jnp.ndarray,
    frequencies: jnp.ndarray,
) -> jnp.ndarray:
    """Integrate a sampled signal, linearly interpolated, through ``event_time``.

    ``signal`` has shape ``(T,)`` or ``(T, B)`` and event_time is in seconds.
    The interval formula is evaluated in normalized coordinates, with series
    for the two entire functions at zero frequency.
    """
    batched = signal.ndim == 2
    if not batched:
        signal = jnp.expand_dims(signal, -1)
        event_time = jnp.reshape(event_time, (1,))
        dt_sec = jnp.reshape(dt_sec, (1,))
    b = signal.shape[1]
    f = jnp.reshape(frequencies, (-1,))
    n_intervals = signal.shape[0] - 1
    idx = jnp.arange(n_intervals, dtype=signal.dtype)
    start = idx[:, None] * dt_sec[None, :]
    end = jnp.minimum((idx[:, None] + 1.0) * dt_sec[None, :], event_time[None, :])
    length = jnp.maximum(end - start, 0.0)
    interval = jnp.floor(event_time / dt_sec).astype(jnp.int64)
    valid = idx.astype(jnp.int64)[:, None] < interval[None, :]
    final = idx.astype(jnp.int64)[:, None] == interval[None, :]
    y0 = signal[:-1]
    y1 = signal[1:]
    alpha = jnp.maximum(jnp.minimum((event_time[None, :] - start) / dt_sec[None, :], 1.0), 0.0)
    y_end = jnp.where(final, y0 + alpha * (y1 - y0), y1)
    y_end = jnp.where(valid | final, y_end, y0)

    q = -2.0j * PI * length[..., None] * jnp.reshape(f, (1, 1, -1))
    small = jnp.abs(q) < 1e-3
    q2 = q * q
    j0_series = 1.0 + q / 2.0 + q2 / 6.0 + q2 * q / 24.0 + q2 * q2 / 120.0
    j1_series = 0.5 + q / 3.0 + q2 / 8.0 + q2 * q / 30.0 + q2 * q2 / 144.0
    expq = jnp.exp(q)

    q_safe = jnp.where(small, jnp.ones_like(q), q)
    q2_safe = q_safe * q_safe
    j0 = jnp.where(small, j0_series, jnp.expm1(q) / q_safe)
    j1 = jnp.where(small, j1_series, ((q - 1.0) * expq + 1.0) / q2_safe)
    phase = jnp.exp(-2.0j * PI * start[..., None] * jnp.reshape(f, (1, 1, -1)))
    result = jnp.sum(
        length[..., None] * phase * (y0[..., None] * j0 + (y_end - y0)[..., None] * j1),
        axis=0,
    )
    return result[0] if not batched else result


def _teobresums_orbital_potential(u: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    """Return the published TEOBResumS ``P^1_5[A_orb^PN]`` potential in JAX.

    Nagar et al., arXiv:1506.08457v2, Eqs. (1)--(3), specify the 5PN-log
    series and the calibrated ``a6c`` polynomial. The exact 4PN coefficient
    is from Bini--Damour, arXiv:1305.4884, Eqs. (2), (6), and the logarithmic
    5PN coefficient is from Damour--Jaranowski--Schaefer,
    arXiv:1502.07245, Eq. (A13).
    """
    if u.ndim > nu.ndim:
        nu = jnp.expand_dims(nu, -1)
    u, nu = jnp.broadcast_arrays(u, nu)
    one = jnp.ones_like(u)
    pi = jnp.asarray(PI, dtype=u.dtype)
    gamma_e = jnp.asarray(0.577215664901532860606512090082402431, dtype=u.dtype)
    ln2 = jnp.asarray(math.log(2.0), dtype=u.dtype)

    a4 = one * ((94.0 / 3.0) - (41.0 / 32.0) * (pi ** 2))
    a50 = (-4237.0 / 60.0 + (2275.0 / 512.0) * (pi ** 2)
           + (256.0 / 5.0) * ln2 + (128.0 / 5.0) * gamma_e)
    a51 = (41.0 / 32.0) * (pi ** 2) - 221.0 / 6.0
    a5c = a50 + a51 * nu
    a5log = one * (64.0 / 5.0)
    a6c = 3097.3 * (nu ** 2) - 1330.6 * nu + 81.38
    a6log = -7004.0 / 105.0 - (144.0 / 5.0) * nu
    log_u = jnp.log(u)

    c = [
        one,
        -2.0 * one,
        jnp.zeros_like(u),
        2.0 * nu,
        nu * a4,
        nu * (a5c + a5log * log_u),
        nu * (a6c + a6log * log_u),
    ]
    rows = []
    rhs = []
    for n in range(2, 7):
        rows.append(jnp.stack([
            c[n - j] if n - j >= 0 else jnp.zeros_like(u)
            for j in range(1, 6)
        ], axis=-1))
        rhs.append(-c[n])
    matrix = jnp.stack(rows, axis=-2)
    zero_mass = jnp.asarray(nu == 0)
    identity = jnp.eye(5, dtype=u.dtype)
    matrix = matrix + identity * zero_mass[..., None, None]
    rhs_vec = jnp.stack(rhs, axis=-1)
    denominator_coeffs = jnp.linalg.solve(matrix, rhs_vec[..., None]).squeeze(-1)
    powers = jnp.stack([u ** j for j in range(1, 6)], axis=-1)
    denominator = one + jnp.sum(denominator_coeffs * powers, axis=-1)
    numerator = one + (c[1] + denominator_coeffs[..., 0]) * u
    pade = numerator / denominator
    return jnp.where(zero_mass, 1.0 - 2.0 * u, pade)


def _teobresums_centrifugal_radius(
    r: jnp.ndarray,
    nu: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
    cq1: Optional[jnp.ndarray] = None,
    cq2: Optional[jnp.ndarray] = None,
    delta: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Return the aligned-spin centrifugal radius r_c = sqrt(r_c^2)."""
    r_arr = as_array(r, dtype=jnp.float64)
    nu_arr = as_array(nu, dtype=jnp.float64)
    s1_arr = as_array(chi1, dtype=jnp.float64)
    s2_arr = as_array(chi2, dtype=jnp.float64)
    if delta is None:
        delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * nu_arr, 0.0))
    else:
        delta = as_array(delta, dtype=jnp.float64)
    x1 = 0.5 * (1.0 + delta)
    x2 = 0.5 * (1.0 - delta)
    a1 = s1_arr * x1
    a2 = s2_arr * x2
    cq1_val = 1.0 if cq1 is None else as_array(cq1, dtype=jnp.float64)
    cq2_val = 1.0 if cq2 is None else as_array(cq2, dtype=jnp.float64)
    if r_arr.ndim > a1.ndim:
        a1 = jnp.expand_dims(a1, -1)
        a2 = jnp.expand_dims(a2, -1)
        cq1_val = jnp.expand_dims(cq1_val, -1)
        cq2_val = jnp.expand_dims(cq2_val, -1)
    a_q_sq = cq1_val * (a1 ** 2) + 2.0 * a1 * a2 + cq2_val * (a2 ** 2)
    rc2 = (r_arr ** 2) + a_q_sq * (1.0 + 2.0 / r_arr)
    return jnp.sqrt(rc2)


def _teobresums_effective_hamiltonian(
    r: jnp.ndarray,
    pr_star: jnp.ndarray,
    pphi: jnp.ndarray,
    nu: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
    lambda1: jnp.ndarray = 0.0,
    lambda2: jnp.ndarray = 0.0,
    cq1: Optional[jnp.ndarray] = None,
    cq2: Optional[jnp.ndarray] = None,
    delta: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Return the effective Hamiltonian H_eff / mu."""
    ham = TEOBResumSHamiltonian(
        nu=nu,
        chi1=chi1,
        chi2=chi2,
        lambda1=lambda1,
        lambda2=lambda2,
        cq1=cq1,
        cq2=cq2,
        delta=delta,
    )
    return ham.effective_hamiltonian(r, pr_star, pphi)


@jax.tree_util.register_pytree_node_class
class TEOBResumSHamiltonian:
    """Nonspinning vacuum Hamiltonian with the published TEOB orbital potential."""

    def __init__(self, nu, chi1=0.0, chi2=0.0, lambda1=0.0, lambda2=0.0,
                 lambda1_oct=None, lambda2_oct=None, cq1=None, cq2=None,
                 delta=None, r_lr=3.0):
        nu = as_array(nu, dtype=jnp.float64)
        for value in (chi1, chi2, lambda1, lambda2, lambda1_oct, lambda2_oct):
            if value is None:
                continue
            # Unsupported parameters are static; construct the model outside
            # JIT or use their literal zero defaults when differentiating nu.
            if isinstance(value, jax.core.Tracer):
                raise ValueError("TEOB Hamiltonian requires static zero spin and tidal parameters.")
            if bool(np.any(np.asarray(value) != 0)):
                raise ValueError("TEOB Hamiltonian supports only nonspinning vacuum parameters; spin and tidal prototypes are unavailable.")
        chi1 = chi2 = lambda1 = lambda2 = jnp.zeros_like(nu)
        if r_lr != 3.0 or cq1 is not None or cq2 is not None:
            raise ValueError("TEOB Hamiltonian supports only nonspinning vacuum parameters; tidal and quadrupole options are unavailable.")
        self.nu, self.chi1, self.chi2 = nu, chi1, chi2
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * nu, 0.0)) if delta is None else delta
        self.x1 = self.m1 = 0.5 * (1.0 + self.delta)
        self.x2 = self.m2 = 0.5 * (1.0 - self.delta)
        self.kappa3_1 = self.kappa3_2 = jnp.zeros_like(nu)

    def tree_flatten(self):
        return (self.nu, self.delta), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        nu, delta = children
        return cls(nu, delta=delta)


    def point_mass_metric_potentials(self, r: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return TEOBResumS point-mass (A, B, xi)."""
        u = 1.0 / r
        a_potential = _teobresums_orbital_potential(u, self.nu)
        nu = self.nu
        if r.ndim > nu.ndim:
            nu = jnp.expand_dims(nu, -1)
        d_potential = 1.0 / (
            1.0 + 6.0 * nu * (u ** 2)
            + 2.0 * nu * (26.0 - 3.0 * nu) * (u ** 3)
        )
        b_potential = d_potential / a_potential
        xi = a_potential / jnp.sqrt(d_potential)
        return a_potential, b_potential, xi

    def centrifugal_radius_squared(self, r):
        """The nonspinning centrifugal radius equals the orbital radius."""
        return r ** 2

    def metric_potentials(self, r):
        """Return published vacuum point-mass (A, B, xi)."""
        return self.point_mass_metric_potentials(r)


    def effective_hamiltonian(self, r, pr_star, pphi):
        """Nonspinning tortoise mass shell, Damour--Nagar Eq. (7)."""
        nu = self.nu
        if r.ndim > nu.ndim:
            nu = jnp.expand_dims(nu, -1)
        a_pot = self.metric_potentials(r)[0]
        z3 = 2.0 * nu * (4.0 - 3.0 * nu)
        return jnp.sqrt(pr_star ** 2 + a_pot * (
            1.0 + (pphi ** 2 + z3 * pr_star ** 4) / r ** 2))

    def real_hamiltonian_from_eff(self, h_eff: jnp.ndarray) -> jnp.ndarray:
        """Compute the real Hamiltonian H_real / M = sqrt(1 + 2*nu*(H_eff - 1))."""
        nu = self.nu
        if h_eff.ndim > nu.ndim:
            nu = jnp.expand_dims(nu, -1)
        radicand = 1.0 + 2.0 * nu * (h_eff - 1.0)
        return jnp.sqrt(jnp.maximum(radicand, 1e-6))

    def real_hamiltonian(
        self,
        r: jnp.ndarray,
        pr_star: jnp.ndarray,
        pphi: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute real Hamiltonian H_real / M."""
        h_eff = self.effective_hamiltonian(r, pr_star, pphi)
        return self.real_hamiltonian_from_eff(h_eff)

    def equations_of_motion_conservative(
        self,
        r: jnp.ndarray,
        pr_star: jnp.ndarray,
        pphi: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute conservative equations of motion (dr/dt, omega, dpr_star/dt)."""
        r_arr, pr_arr, pphi_arr = jnp.broadcast_arrays(
            as_array(r, dtype=jnp.float64),
            as_array(pr_star, dtype=jnp.float64),
            as_array(pphi, dtype=jnp.float64),
        )

        def _single_eom(rx, px, jx):
            h_eff = self.effective_hamiltonian(rx, px, jx)
            h_real = self.real_hamiltonian_from_eff(h_eff)
            dh_dr = jax.grad(self.effective_hamiltonian, argnums=0)(rx, px, jx)
            dh_dpr = jax.grad(self.effective_hamiltonian, argnums=1)(rx, px, jx)
            dh_dpphi = jax.grad(self.effective_hamiltonian, argnums=2)(rx, px, jx)
            _, _, xi = self.metric_potentials(rx)
            dr_dt = (xi / h_real) * dh_dpr
            omega = (1.0 / h_real) * dh_dpphi
            dpr_star_dt = -(xi / h_real) * dh_dr
            return dr_dt, omega, dpr_star_dt

        # Check if Hamiltonian has batched parameters
        sample_heff = self.effective_hamiltonian(r_arr, pr_arr, pphi_arr)
        if sample_heff.ndim > 0:
            if r_arr.ndim == 0:
                r_arr = jnp.broadcast_to(r_arr, sample_heff.shape)
                pr_arr = jnp.broadcast_to(pr_arr, sample_heff.shape)
                pphi_arr = jnp.broadcast_to(pphi_arr, sample_heff.shape)
            return jax.vmap(_single_eom)(r_arr, pr_arr, pphi_arr)

        return _single_eom(r_arr, pr_arr, pphi_arr)

    def equations_of_motion(
        self,
        r: jnp.ndarray,
        pr_star: jnp.ndarray,
        pphi: jnp.ndarray,
        rr: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute full conservative and dissipative time derivatives."""
        dr_dt, omega, dpr_star_cons = self.equations_of_motion_conservative(
            r, pr_star, pphi
        )
        f_r_star, f_phi = rr.radiation_reaction_forces(
            omega, pr_star, pphi, r=r, hamiltonian=self
        )
        dpr_star_dt = dpr_star_cons + f_r_star
        dpphi_dt = f_phi
        return dr_dt, omega, dpr_star_dt, dpphi_dt
