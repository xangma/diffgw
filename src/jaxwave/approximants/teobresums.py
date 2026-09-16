"""Resummed Tidal Effective-One-Body (TEOBResumS) model in JAX.

The point-mass metric and nonspinning tortoise kinetic terms follow Damour
and Nagar, arXiv:0704.3550v2 Eqs. (2)-(11). Tides enter A additively, as in
Nagar et al., PRD 98, 104052 (2018), arXiv:1806.01772v2 Eq. (23).

The aligned-spin centrifugal-radius map follows Nagar et al., arXiv:1806.01772v2,
Eqs. (23)--(25).
"""

from typing import Any, Dict, Optional, Tuple, Union
import math
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


def _pade_tidal_factor(u: jnp.ndarray, r_lr: float = 3.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return the heuristic tidal amplification factor hat{A}_2, hat{A}_3."""
    den = jnp.maximum(1.0 - r_lr * u, 1e-3)
    hat_a2 = 1.0 + (3.0 * u) / den
    hat_a3 = jnp.ones_like(hat_a2)
    return hat_a2, hat_a3


def _tidal_metric_potential(
    u: jnp.ndarray,
    nu: jnp.ndarray,
    lambda1: jnp.ndarray,
    lambda2: jnp.ndarray,
    lambda1_oct: Optional[jnp.ndarray] = None,
    lambda2_oct: Optional[jnp.ndarray] = None,
    delta: Optional[jnp.ndarray] = None,
    r_lr: float = 3.0,
) -> jnp.ndarray:
    """Compute the tidal metric correction A_tidal(u)."""
    if delta is None:
        delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * nu, 0.0))
    x1 = 0.5 * (1.0 + delta)
    x2 = 0.5 * (1.0 - delta)
    kappa2_1 = 3.0 * lambda1 * (x1 ** 4) * x2
    kappa2_2 = 3.0 * lambda2 * (x2 ** 4) * x1
    kappa3_1 = 7.5 * lambda1_oct * (x1 ** 6) * x2 if lambda1_oct is not None else jnp.zeros_like(kappa2_1)
    kappa3_2 = 7.5 * lambda2_oct * (x2 ** 6) * x1 if lambda2_oct is not None else jnp.zeros_like(kappa2_2)

    if u.ndim > kappa2_1.ndim:
        kappa2_1 = jnp.expand_dims(kappa2_1, -1)
        kappa2_2 = jnp.expand_dims(kappa2_2, -1)
        kappa3_1 = jnp.expand_dims(kappa3_1, -1)
        kappa3_2 = jnp.expand_dims(kappa3_2, -1)

    u6 = u ** 6
    u8 = u6 * (u * u)
    hat_a2, hat_a3 = _pade_tidal_factor(u, r_lr=r_lr)
    a_tidal_quad = -(kappa2_1 + kappa2_2) * u6 * hat_a2
    a_tidal_oct = -(kappa3_1 + kappa3_2) * u8 * hat_a3
    return a_tidal_quad + a_tidal_oct


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


def compute_tidal_merger_frequency(
    mass1: Union[float, jnp.ndarray],
    mass2: Union[float, jnp.ndarray],
    lambda1: Union[float, jnp.ndarray] = 0.0,
    lambda2: Union[float, jnp.ndarray] = 0.0,
    chi1: Union[float, jnp.ndarray] = 0.0,
    chi2: Union[float, jnp.ndarray] = 0.0,
) -> jnp.ndarray:
    """Return the inherited tidal merger-frequency heuristic in Hz."""
    m1 = as_array(mass1, dtype=jnp.float64)
    m2 = as_array(mass2, dtype=jnp.float64)
    l1 = as_array(lambda1, dtype=m1.dtype)
    l2 = as_array(lambda2, dtype=m1.dtype)
    s1 = as_array(chi1, dtype=m1.dtype)
    s2 = as_array(chi2, dtype=m1.dtype)

    total_mass = m1 + m2
    m_sec = total_mass * MSUN_SEC
    nu = component_masses_to_eta(m1, m2)
    delta = component_masses_to_delta(m1, m2)
    x1 = 0.5 * (1.0 + delta)
    x2 = 0.5 * (1.0 - delta)

    k2_1 = 3.0 * l1 * (x1 ** 4) * x2
    k2_2 = 3.0 * l2 * (x2 ** 4) * x1
    kappa2_t = k2_1 + k2_2

    omega_pm = 0.36 * (nu ** 0.2) * (1.0 + 0.1 * (s1 * x1 + s2 * x2))
    c1 = 0.0035
    c2 = 1.5e-6
    tidal_shift = 1.0 / (1.0 + c1 * kappa2_t + c2 * (kappa2_t ** 2))
    m_omega_mrg = omega_pm * tidal_shift

    f_mrg = m_omega_mrg / (PI * m_sec)
    return f_mrg


@jax.tree_util.register_pytree_node_class
class TEOBResumSHamiltonian:
    """Prototype tidal EOB Hamiltonian with aligned spins in JAX."""

    def __init__(
        self,
        nu: jnp.ndarray,
        chi1: jnp.ndarray,
        chi2: jnp.ndarray,
        lambda1: jnp.ndarray,
        lambda2: jnp.ndarray,
        lambda1_oct: Optional[jnp.ndarray] = None,
        lambda2_oct: Optional[jnp.ndarray] = None,
        cq1: Optional[jnp.ndarray] = None,
        cq2: Optional[jnp.ndarray] = None,
        delta: Optional[jnp.ndarray] = None,
        r_lr: float = 3.0,
    ):
        nu = as_array(nu, dtype=jnp.float64)
        chi1 = as_array(chi1, dtype=jnp.float64)
        chi2 = as_array(chi2, dtype=jnp.float64)
        lambda1 = as_array(lambda1, dtype=jnp.float64)
        lambda2 = as_array(lambda2, dtype=jnp.float64)

        if delta is None:
            delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * nu, 0.0))
        else:
            delta = as_array(delta, dtype=jnp.float64)

        self.nu = nu
        self.chi1 = chi1
        self.chi2 = chi2
        self.delta = delta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.r_lr = float(r_lr)

        self.x1 = 0.5 * (1.0 + delta)
        self.x2 = 0.5 * (1.0 - delta)
        self.m1 = self.x1
        self.m2 = self.x2

        self.a1 = chi1 * self.x1
        self.a2 = chi2 * self.x2
        self.a_plus = self.a1 + self.a2
        self.a_minus = self.a1 - self.a2

        self.kappa2_1 = 3.0 * self.lambda1 * (self.x1 ** 4) * self.x2
        self.kappa2_2 = 3.0 * self.lambda2 * (self.x2 ** 4) * self.x1
        self.kappa2_tot = self.kappa2_1 + self.kappa2_2

        if lambda1_oct is not None:
            self.kappa3_1 = 7.5 * as_array(lambda1_oct, dtype=jnp.float64) * (self.x1 ** 6) * self.x2
        else:
            self.kappa3_1 = jnp.zeros_like(self.kappa2_1)

        if lambda2_oct is not None:
            self.kappa3_2 = 7.5 * as_array(lambda2_oct, dtype=jnp.float64) * (self.x2 ** 6) * self.x1
        else:
            self.kappa3_2 = jnp.zeros_like(self.kappa2_2)

        if cq1 is None:
            self.cq1 = jnp.where(
                self.lambda1 > 0,
                1.0 + 0.05 * jnp.log(1.0 + self.lambda1),
                jnp.ones_like(self.lambda1),
            )
        else:
            self.cq1 = as_array(cq1, dtype=jnp.float64)

        if cq2 is None:
            self.cq2 = jnp.where(
                self.lambda2 > 0,
                1.0 + 0.05 * jnp.log(1.0 + self.lambda2),
                jnp.ones_like(self.lambda2),
            )
        else:
            self.cq2 = as_array(cq2, dtype=jnp.float64)

    def tree_flatten(self):
        children = (
            self.nu, self.chi1, self.chi2, self.delta,
            self.lambda1, self.lambda2,
            self.kappa2_1, self.kappa2_2, self.kappa2_tot,
            self.kappa3_1, self.kappa3_2,
            self.cq1, self.cq2,
            self.x1, self.x2, self.m1, self.m2,
            self.a1, self.a2, self.a_plus, self.a_minus,
        )
        aux_data = (self.r_lr,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (
            obj.nu, obj.chi1, obj.chi2, obj.delta,
            obj.lambda1, obj.lambda2,
            obj.kappa2_1, obj.kappa2_2, obj.kappa2_tot,
            obj.kappa3_1, obj.kappa3_2,
            obj.cq1, obj.cq2,
            obj.x1, obj.x2, obj.m1, obj.m2,
            obj.a1, obj.a2, obj.a_plus, obj.a_minus,
        ) = children
        obj.r_lr = aux_data[0]
        return obj

    def pade_tidal_factor(self, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the heuristic tidal amplification factor."""
        return _pade_tidal_factor(u, r_lr=self.r_lr)

    def tidal_metric_potential(self, u: jnp.ndarray) -> jnp.ndarray:
        """Compute the tidal metric correction A_tidal(u)."""
        k2_1 = self.kappa2_1
        k2_2 = self.kappa2_2
        k3_1 = self.kappa3_1
        k3_2 = self.kappa3_2

        if u.ndim > k2_1.ndim:
            k2_1 = jnp.expand_dims(k2_1, -1)
            k2_2 = jnp.expand_dims(k2_2, -1)
            k3_1 = jnp.expand_dims(k3_1, -1)
            k3_2 = jnp.expand_dims(k3_2, -1)

        u6 = u ** 6
        u8 = u6 * (u * u)
        hat_a2, hat_a3 = self.pade_tidal_factor(u)
        a_tidal_quad = -(k2_1 + k2_2) * u6 * hat_a2
        a_tidal_oct = -(k3_1 + k3_2) * u8 * hat_a3
        return a_tidal_quad + a_tidal_oct

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

    def centrifugal_radius_squared(self, r: jnp.ndarray) -> jnp.ndarray:
        """Return aligned-spin centrifugal radius squared."""
        a1 = self.a1
        a2 = self.a2
        cq1 = self.cq1
        cq2 = self.cq2
        if r.ndim > a1.ndim:
            a1 = jnp.expand_dims(a1, -1)
            a2 = jnp.expand_dims(a2, -1)
            cq1 = jnp.expand_dims(cq1, -1)
            cq2 = jnp.expand_dims(cq2, -1)
        a_q_sq = cq1 * (a1 ** 2) + 2.0 * a1 * a2 + cq2 * (a2 ** 2)
        return (r ** 2) + a_q_sq * (1.0 + 2.0 / r)

    def metric_potentials(self, r: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return equatorial metric potentials (A, B, xi)."""
        u = 1.0 / r
        rc = jnp.sqrt(self.centrifugal_radius_squared(r))
        u_c = 1.0 / rc
        a0_c, _, _ = self.point_mass_metric_potentials(rc)
        a_tidal = self.tidal_metric_potential(u_c)

        a_orb = a0_c + a_tidal
        a_total = ((1.0 + 2.0 * u_c) / (1.0 + 2.0 * u)) * a_orb
        a0, b0, _ = self.point_mass_metric_potentials(r)
        d_pot = a0 * b0
        b_total = d_pot / a_total
        xi = a_total / jnp.sqrt(d_pot)
        return a_total, b_total, xi

    def gyro_factors(self, r: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute gyro-gravitomagnetic factors g_{a_+} and g_{a_-}."""
        u = 1.0 / r
        nu = self.nu
        if r.ndim > nu.ndim:
            nu = jnp.expand_dims(nu, -1)
        g_plus = 2.0 - (9.0 / 8.0) * nu * u
        g_minus = (5.0 / 4.0) * nu * u
        return g_plus, g_minus

    def effective_hamiltonian(
        self,
        r: jnp.ndarray,
        pr_star: jnp.ndarray,
        pphi: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return H_eff / mu for equatorial aligned-spin orbits."""
        a_plus = self.a_plus
        a_minus = self.a_minus
        delta = self.delta
        nu = self.nu

        if r.ndim > a_plus.ndim:
            a_plus = jnp.expand_dims(a_plus, -1)
            a_minus = jnp.expand_dims(a_minus, -1)
            delta = jnp.expand_dims(delta, -1)
        if r.ndim > nu.ndim:
            nu = jnp.expand_dims(nu, -1)

        r2 = r * r
        r3 = r2 * r
        a_plus2 = a_plus * a_plus
        r_c2 = self.centrifugal_radius_squared(r)

        a_pot, _, _ = self.metric_potentials(r)
        g_plus, g_minus = self.gyro_factors(r)

        so_num = pphi * (g_plus * a_plus + g_minus * delta * a_minus)
        so_den = r3 + a_plus2 * (r + 2.0)
        h_odd = so_num / so_den

        z3 = 2.0 * nu * (4.0 - 3.0 * nu)
        angular_term = 1.0 + (pphi ** 2) / r_c2
        h_even = jnp.sqrt(pr_star ** 2 + a_pot * (angular_term + z3 * (pr_star ** 4) / r_c2))

        return h_odd + h_even

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


@jax.tree_util.register_pytree_node_class
class TEOBResumSRadiationReaction:
    """TEOBResumS quasi-circular radiation-reaction forces in JAX."""

    def __init__(
        self,
        nu: jnp.ndarray,
        chi1: jnp.ndarray,
        chi2: jnp.ndarray,
        *,
        include_horizon_absorption: bool = True,
    ):
        self.nu = as_array(nu, dtype=jnp.float64)
        self.chi1 = as_array(chi1, dtype=jnp.float64)
        self.chi2 = as_array(chi2, dtype=jnp.float64)
        self.include_horizon_absorption = bool(include_horizon_absorption)

    def tree_flatten(self):
        children = (self.nu, self.chi1, self.chi2)
        aux_data = (self.include_horizon_absorption,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        nu, chi1, chi2 = children
        return cls(nu, chi1, chi2, include_horizon_absorption=aux_data[0])

    def energy_flux(self, omega: jnp.ndarray) -> jnp.ndarray:
        """Compute the total gravitational wave energy flux dE/dt."""
        nu = self.nu
        if omega.ndim > nu.ndim:
            nu = jnp.expand_dims(nu, -1)

        v = jnp.maximum(jnp.abs(omega), 1e-8) ** (1.0 / 3.0)
        v2 = v * v
        v3 = v2 * v
        v10 = (v2 * v2) * (v2 * v2) * v2

        f_1pn = -(1247.0 / 336.0) - (35.0 / 12.0) * nu
        f_15pn = 4.0 * PI

        flux_factor = 1.0 + f_1pn * v2 + f_15pn * v3
        flux_factor = jnp.maximum(flux_factor, 0.1)

        f_newtonian = (32.0 / 5.0) * (nu * nu) * v10
        return f_newtonian * flux_factor

    def radiation_reaction_forces(
        self,
        omega: jnp.ndarray,
        pr_star: jnp.ndarray,
        pphi: jnp.ndarray,
        *,
        r=None,
        hamiltonian=None,
        include_horizon_absorption: Optional[bool] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute dissipative radiation-reaction force vector (F_r_star, F_phi)."""
        flux = self.energy_flux(omega)
        safe_omega = jnp.maximum(jnp.abs(omega), 1e-8)
        nu = self.nu
        if omega.ndim > nu.ndim:
            nu = jnp.expand_dims(nu, -1)
        torque = -(flux / (nu * safe_omega))
        return jnp.zeros_like(torque), torque


def find_quasicircular_initial_conditions(
    hamiltonian: TEOBResumSHamiltonian,
    radiation_reaction: TEOBResumSRadiationReaction,
    omega_start: jnp.ndarray,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve for stable quasicircular initial conditions (r0, pr_star0, pphi0)."""
    omega_start = as_array(omega_start, dtype=jnp.float64)

    def _solve_single(omega_s):
        r = omega_s ** (-2.0 / 3.0)
        pphi = jnp.sqrt(r / jnp.maximum(1.0 - 3.0 / r, 0.1))

        # Iterations with stop_gradient to prevent unrolling AD graph
        for _ in range(4):
            rx, px = jax.lax.stop_gradient(r), jax.lax.stop_gradient(pphi)

            def res_fn(xy):
                r_eval, p_eval = xy[0], xy[1]
                heff = hamiltonian.effective_hamiltonian(r_eval, 0.0, p_eval)
                hreal = hamiltonian.real_hamiltonian_from_eff(heff)
                dhdr = jax.grad(hamiltonian.effective_hamiltonian, argnums=0)(r_eval, 0.0, p_eval)
                dhdpphi = jax.grad(hamiltonian.effective_hamiltonian, argnums=2)(r_eval, 0.0, p_eval)
                return jnp.array([dhdpphi / hreal - omega_s, dhdr])

            res = res_fn(jnp.array([rx, px]))
            jac = jax.jacobian(res_fn)(jnp.array([rx, px]))
            step = - jnp.linalg.solve(jac, res)
            r = rx + step[0]
            pphi = px + step[1]

        # One final Newton step retaining parameter gradients
        r_root, pphi_root = jax.lax.stop_gradient(r), jax.lax.stop_gradient(pphi)

        def res_fn_final(xy):
            r_eval, p_eval = xy[0], xy[1]
            heff = hamiltonian.effective_hamiltonian(r_eval, 0.0, p_eval)
            hreal = hamiltonian.real_hamiltonian_from_eff(heff)
            dhdr = jax.grad(hamiltonian.effective_hamiltonian, argnums=0)(r_eval, 0.0, p_eval)
            dhdpphi = jax.grad(hamiltonian.effective_hamiltonian, argnums=2)(r_eval, 0.0, p_eval)
            return jnp.array([dhdpphi / hreal - omega_s, dhdr])

        res = res_fn_final(jnp.array([r_root, pphi_root]))
        jac = jax.lax.stop_gradient(jax.jacobian(res_fn_final)(jnp.array([r_root, pphi_root])))
        step = - jnp.linalg.solve(jac, res)
        r = r_root + step[0]
        pphi = pphi_root + step[1]

        heff = hamiltonian.effective_hamiltonian(r, 0.0, pphi)
        hreal = hamiltonian.real_hamiltonian_from_eff(heff)
        d2h_dr2 = jax.grad(jax.grad(hamiltonian.effective_hamiltonian, argnums=0), argnums=0)(r, 0.0, pphi)
        d2h_drdj = jax.grad(jax.grad(hamiltonian.effective_hamiltonian, argnums=0), argnums=2)(r, 0.0, pphi)
        d2h_dpr2 = jax.grad(jax.grad(hamiltonian.effective_hamiltonian, argnums=1), argnums=1)(r, 0.0, pphi)
        dp_phi_dr = -d2h_dr2 / d2h_drdj
        _, f_phi = radiation_reaction.radiation_reaction_forces(omega_s, 0.0, pphi, r=r, hamiltonian=hamiltonian)
        _, _, xi = hamiltonian.metric_potentials(r)
        velocity_per_momentum = xi * d2h_dpr2 / hreal
        pr_star_corr = f_phi / (velocity_per_momentum * dp_phi_dr)
        return r, pr_star_corr, pphi

    if omega_start.ndim > 0:
        return jax.vmap(_solve_single)(omega_start)
    return _solve_single(omega_start)


@jax.tree_util.register_pytree_node_class
class TEOBResumS:
    """Prototype tidal waveform generator in JAX retaining its historical API name."""

    def __new__(cls, *args, **kwargs):
        if args and not isinstance(args[0], str) and (len(args) >= 3 or "f" in kwargs or "mass1" in kwargs):
            return cls.evaluate(*args, **kwargs)
        if "f" in kwargs or "mass1" in kwargs:
            return cls.evaluate(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, matching_method: Optional[str] = None, **kwargs):
        self.default_matching_method = matching_method
        self.init_kwargs = kwargs

    def tree_flatten(self):
        return ((), (self.default_matching_method, tuple(sorted(self.init_kwargs.items()))))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        matching_method, items = aux_data
        return cls(matching_method=matching_method, **dict(items))

    def __call__(self, *args, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if not isinstance(self, TEOBResumS):
            return TEOBResumS.evaluate(self, *args, **kwargs)
        call_kwargs = dict(self.init_kwargs)
        if self.default_matching_method is not None and "matching_method" not in kwargs:
            call_kwargs["matching_method"] = self.default_matching_method
        call_kwargs.update(kwargs)
        return self.evaluate(*args, **call_kwargs)

    @staticmethod
    def evaluate(
        f: jnp.ndarray,
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        chi1: Union[float, jnp.ndarray] = 0.0,
        chi2: Union[float, jnp.ndarray] = 0.0,
        lambda1: Union[float, jnp.ndarray] = 0.0,
        lambda2: Union[float, jnp.ndarray] = 0.0,
        cq1: Optional[Union[float, jnp.ndarray]] = None,
        cq2: Optional[Union[float, jnp.ndarray]] = None,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        dt_geom: float = 0.5,
        num_steps: int = 1500,
        taper_width_ratio: float = 0.1,
        f_lower: Optional[Union[float, jnp.ndarray]] = None,
        integration_mode: str = "diagnostic_fixed",
        matching_method: Optional[str] = None,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate prototype tidal polarizations with a heuristic merger taper."""
        f = as_array(f, dtype=jnp.float64)
        m1 = as_array(mass1, dtype=jnp.float64)
        m2 = as_array(mass2, dtype=jnp.float64)
        s1 = as_array(chi1, dtype=jnp.float64)
        s2 = as_array(chi2, dtype=jnp.float64)
        l1 = as_array(lambda1, dtype=jnp.float64)
        l2 = as_array(lambda2, dtype=jnp.float64)
        dist = as_array(distance, dtype=jnp.float64)
        inc = as_array(inclination, dtype=jnp.float64)
        phi_c = as_array(phic, dtype=jnp.float64)
        t_c = as_array(tc, dtype=jnp.float64)

        is_batched = m1.ndim > 0

        def _single_waveform(m1_i, m2_i, s1_i, s2_i, l1_i, l2_i, dist_i, inc_i, phi_ci, t_ci, f_i, f_low_i):
            total_mass = m1_i + m2_i
            eta = component_masses_to_eta(m1_i, m2_i)
            delta = component_masses_to_delta(m1_i, m2_i)

            hamiltonian = TEOBResumSHamiltonian(
                nu=eta,
                chi1=s1_i,
                chi2=s2_i,
                lambda1=l1_i,
                lambda2=l2_i,
                cq1=as_array(cq1, dtype=jnp.float64) if cq1 is not None else None,
                cq2=as_array(cq2, dtype=jnp.float64) if cq2 is not None else None,
                delta=delta,
            )
            rr = TEOBResumSRadiationReaction(
                nu=eta,
                chi1=s1_i,
                chi2=s2_i,
                include_horizon_absorption=True,
            )

            m_sec = total_mass * MSUN_SEC
            if f_low_i is None:
                f_start_hz = jnp.min(jnp.where(f_i > 0, f_i, jnp.inf))
            else:
                f_start_hz = jnp.where(
                    jnp.isnan(f_low_i),
                    jnp.min(jnp.where(f_i > 0, f_i, jnp.inf)),
                    as_array(f_low_i, dtype=jnp.float64),
                )

            omega_start = PI * m_sec * f_start_hz
            r0, pr0, pphi0 = find_quasicircular_initial_conditions(hamiltonian, rr, omega_start)
            init_state = jnp.array([r0, 0.0, pr0, pphi0])

            dt = dt_geom
            half_dt = 0.5 * dt
            sixth_dt = dt / 6.0

            def vector_field(s):
                dr_dt, omega, dpr_star_dt, dpphi_dt = hamiltonian.equations_of_motion(s[0], s[2], s[3], rr)
                return jnp.array([dr_dt, omega, dpr_star_dt, dpphi_dt])

            def rk4_step(state, _):
                k1 = vector_field(state)
                k2 = vector_field(state + half_dt * k1)
                k3 = vector_field(state + half_dt * k2)
                k4 = vector_field(state + dt * k3)
                candidate_state = state + sixth_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                return candidate_state, candidate_state

            _, states_body = jax.lax.scan(rk4_step, init_state, None, length=num_steps)
            states = jnp.concatenate([jnp.expand_dims(init_state, 0), states_body], axis=0)

            r_traj = states[:, 0]
            phi_traj = states[:, 1]

            def get_omega(s):
                return hamiltonian.equations_of_motion(s[0], s[2], s[3], rr)[1]

            omega_traj = jax.vmap(get_omega)(states)
            v_traj = jnp.maximum(omega_traj, 1e-8) ** (1.0 / 3.0)

            h22_amp = 8.0 * math.sqrt(PI / 5.0) * eta * (v_traj * v_traj)
            phase_gw = 2.0 * (phi_traj + phi_ci)

            h22_real = h22_amp * jnp.cos(phase_gw)
            h22_imag = -h22_amp * jnp.sin(phase_gw)

            d_sec = (dist_i * MPC_SI) / C_SI
            phys_scale = m_sec / d_sec
            h22_phys_real = h22_real * phys_scale
            h22_phys_imag = h22_imag * phys_scale
            dt_sec = dt_geom * m_sec

            cos_i = jnp.cos(inc_i)
            plus_factor = math.sqrt(5.0 / (16.0 * PI)) * (1.0 + cos_i * cos_i)
            cross_factor = -math.sqrt(5.0 / (4.0 * PI)) * cos_i
            hp_time = plus_factor * h22_phys_real
            hc_time = cross_factor * h22_phys_imag

            if integration_mode == "physical_event":
                # Find orbital frequency maximum from omega_traj
                event_idx = jnp.argmax(omega_traj)
                event_time = event_idx * dt
                event_time_sec = event_time * m_sec
                hp_interp = _piecewise_linear_fourier(hp_time, event_time_sec, dt_sec, f_i)
                hc_interp = _piecewise_linear_fourier(hc_time, event_time_sec, dt_sec, f_i)
                phase_shift = -2.0 * PI * f_i * t_ci
                shift_factor = jnp.cos(phase_shift) + 1j * jnp.sin(phase_shift)
                f_mrg = compute_tidal_merger_frequency(m1_i, m2_i, l1_i, l2_i, s1_i, s2_i)
                delta_f = taper_width_ratio * f_mrg
                is_tidal = (l1_i > 0) | (l2_i > 0)
                taper = jnp.where(
                    is_tidal,
                    1.0 / (1.0 + jnp.exp(jnp.minimum((f_i - f_mrg) / delta_f, 50.0))),
                    1.0,
                )
                return hp_interp * shift_factor * taper, hc_interp * shift_factor * taper

            # Default diagnostic_fixed mode
            fft_len = 2 ** math.ceil(math.log2(hp_time.shape[0] * 4))
            hp_fd = jnp.fft.rfft(hp_time, n=fft_len, axis=0)
            hc_fd = jnp.fft.rfft(hc_time, n=fft_len, axis=0)
            num_pos = fft_len // 2 + 1
            hp_fd_scaled = hp_fd * dt_sec
            hc_fd_scaled = hc_fd * dt_sec
            k_indices = jnp.arange(num_pos, dtype=jnp.float64)
            fft_freqs = k_indices / (fft_len * dt_sec)

            hp_interp = _interp1d(f_i, fft_freqs, hp_fd_scaled)
            hc_interp = _interp1d(f_i, fft_freqs, hc_fd_scaled)

            phase_shift = -2.0 * PI * f_i * t_ci
            shift_factor = jnp.cos(phase_shift) + 1j * jnp.sin(phase_shift)
            hp_shifted = hp_interp * shift_factor
            hc_shifted = hc_interp * shift_factor

            f_mrg = compute_tidal_merger_frequency(m1_i, m2_i, l1_i, l2_i, s1_i, s2_i)
            delta_f = taper_width_ratio * f_mrg
            is_tidal = (l1_i > 0) | (l2_i > 0)
            taper = jnp.where(
                is_tidal,
                1.0 / (1.0 + jnp.exp(jnp.minimum((f_i - f_mrg) / delta_f, 50.0))),
                1.0,
            )
            return hp_shifted * taper, hc_shifted * taper

        if is_batched:
            # Broadcast parameters across batch shape
            b_shape = m1.shape
            m2_b = jnp.broadcast_to(m2, b_shape)
            s1_b = jnp.broadcast_to(s1, b_shape)
            s2_b = jnp.broadcast_to(s2, b_shape)
            l1_b = jnp.broadcast_to(l1, b_shape)
            l2_b = jnp.broadcast_to(l2, b_shape)
            dist_b = jnp.broadcast_to(dist, b_shape)
            inc_b = jnp.broadcast_to(inc, b_shape)
            phi_cb = jnp.broadcast_to(phi_c, b_shape)
            t_cb = jnp.broadcast_to(t_c, b_shape)
            f_low_b = jnp.broadcast_to(as_array(f_lower, dtype=jnp.float64), b_shape) if f_lower is not None else None

            # Flatten to 1D batch if needed
            flat_m1 = m1.reshape(-1)
            flat_m2 = m2_b.reshape(-1)
            flat_s1 = s1_b.reshape(-1)
            flat_s2 = s2_b.reshape(-1)
            flat_l1 = l1_b.reshape(-1)
            flat_l2 = l2_b.reshape(-1)
            flat_dist = dist_b.reshape(-1)
            flat_inc = inc_b.reshape(-1)
            flat_phic = phi_cb.reshape(-1)
            flat_tc = t_cb.reshape(-1)
            flat_flow = flat_m1 * 0.0 + f_lower if f_lower is not None else jnp.full(flat_m1.shape, jnp.nan)

            def vmap_worker(m1_i, m2_i, s1_i, s2_i, l1_i, l2_i, dist_i, inc_i, phi_ci, t_ci, flow_i):
                flow_val = None if f_lower is None else flow_i
                return _single_waveform(m1_i, m2_i, s1_i, s2_i, l1_i, l2_i, dist_i, inc_i, phi_ci, t_ci, f, flow_val)

            hp_out, hc_out = jax.vmap(vmap_worker)(
                flat_m1, flat_m2, flat_s1, flat_s2, flat_l1, flat_l2,
                flat_dist, flat_inc, flat_phic, flat_tc, flat_flow,
            )
            return hp_out.reshape(*b_shape, f.shape[-1]), hc_out.reshape(*b_shape, f.shape[-1])

        return _single_waveform(m1, m2, s1, s2, l1, l2, dist, inc, phi_c, t_c, f, f_lower)
