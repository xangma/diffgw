"""Clean Kerr (2,2) ringdown TD/FD pair in JAX.

Port of torchwave.approximants.kerr_ringdown_public_clean.
Analytic Fourier transform of a damped complex mode and its mirror.
The default angular shape uses the separately documented inferred
public-compatible convention; ``source_spectral`` selects the angular
Teukolsky eigensolution. No public implementation or calibration values are used.
"""

import math
from typing import Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from jaxwave.approximants.kerr_ringdown import ringdown_amplitude_from_energy_fraction

_K21_NODES = (
    -0.9956571630258081, -0.9739065285171717, -0.9301574913557082,
    -0.8650633666889845, -0.7808177265864169, -0.6794095682990244,
    -0.5627571346686047, -0.43339539412924716, -0.2943928627014602,
    -0.14887433898163122, 0.0, 0.14887433898163122,
    0.2943928627014602, 0.43339539412924716, 0.5627571346686047,
    0.6794095682990244, 0.7808177265864169, 0.8650633666889845,
    0.9301574913557082, 0.9739065285171717, 0.9956571630258081,
)
_K21_WEIGHTS = (
    0.011694638867371862, 0.03255816230796478, 0.054755896574351974,
    0.07503967481091994, 0.0931254545836976, 0.10938715880229763,
    0.12349197626206586, 0.13470921731147337, 0.14277593857706,
    0.14773910490133854, 0.14944555400291687, 0.14773910490133854,
    0.14277593857706, 0.13470921731147337, 0.12349197626206586,
    0.10938715880229763, 0.0931254545836976, 0.07503967481091994,
    0.054755896574351974, 0.03255816230796478, 0.011694638867371862,
)
_G10_INDICES = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
_G10_WEIGHTS = (
    0.06667134430868814, 0.1494513491505806, 0.21908636251598204,
    0.26926671930999635, 0.29552422471475287, 0.29552422471475287,
    0.26926671930999635, 0.21908636251598204, 0.1494513491505806,
    0.06667134430868814,
)


def _np_leaver_frobenius_coefficients(ell: int, c: np.ndarray, n_terms: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(c, dtype=np.complex128)
    orig_shape = c.shape
    c_flat = c.reshape(-1)

    n = np.arange(n_terms, dtype=np.float64)

    def recurrence_matrix(cval):
        out = np.zeros((n_terms, n_terms), dtype=np.complex128)
        diag = n * (n + 5.0) - 4.0 * cval * n
        np.fill_diagonal(out, diag)
        for i in range(n_terms - 1):
            out[i, i + 1] = -(n[i] + 1.0) * (n[i] + 5.0)
            out[i + 1, i] = 4.0 * cval * n[i + 1]
        return out

    A_list = []
    a_list = []
    target = (ell - 2) * (ell + 3)

    for cval in c_flat:
        mat = recurrence_matrix(cval)
        values, vectors = np.linalg.eig(mat)
        previous = target + 0.0j
        for step in range(1, 9):
            vals = np.linalg.eigvals(recurrence_matrix(cval * (step / 8.0)))
            pick = np.argmin(np.abs(vals - previous))
            previous = vals[pick]
        idx = np.argmin(np.abs(values - previous))
        q = values[idx]

        terms = [1.0 + 0.0j]
        for j in range(n_terms - 1):
            b = q - n[j] * (n[j] + 5.0) + 4.0 * cval * n[j]
            if j > 0:
                b = b * terms[j] - 4.0 * cval * n[j] * terms[j - 1]
            else:
                b = b * terms[j]
            terms.append(-b / ((j + 1.0) * (j + 5.0)))
        a = np.array(terms, dtype=np.complex128)
        A = q - cval**2 - 6.0 * cval + 4.0
        A_list.append(A)
        a_list.append(a)

    A_arr = np.array(A_list, dtype=np.complex128).reshape(orig_shape)
    a_arr = np.array(a_list, dtype=np.complex128).reshape(orig_shape + (n_terms,))
    return A_arr, a_arr


def _np_raw(mu: np.ndarray, c: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    z = (1.0 + mu) / 2.0
    active = np.ones_like(mu, dtype=bool)
    total = np.zeros_like(mu, dtype=np.complex128) + coeff[..., 0]
    for n in range(1, coeff.shape[-1]):
        term = coeff[..., n] * (z**n)
        total = total + np.where(active, term, 0.0)
        active = active & (np.abs(term) >= 0.01)
    return np.exp(c * mu) * ((1.0 + mu) ** 2) * total


def _np_norm(c: np.ndarray, coeff: np.ndarray, tol: float = 1e-6, max_intervals: int = 1000) -> np.ndarray:
    nodes = np.array(_K21_NODES, dtype=np.float64)
    weights = np.array(_K21_WEIGHTS, dtype=np.float64)
    gweights = np.array(_G10_WEIGHTS, dtype=np.float64)
    out = []
    for ci, ai in zip(c.reshape(-1), coeff.reshape(-1, coeff.shape[-1])):
        def leaf(lo, hi):
            x = (hi - lo) / 2.0 * nodes + (hi + lo) / 2.0
            y = np.abs(_np_raw(x, ci, ai)) ** 2
            k = (hi - lo) / 2.0 * np.sum(weights * y)
            g = (hi - lo) / 2.0 * np.sum(gweights * y[list(_G10_INDICES)])
            return [lo, hi, k, np.abs(k - g)]

        leaves = [leaf(-1.0, 1.0)]
        while sum(q[3] for q in leaves) > max(tol, tol * abs(sum(q[2] for q in leaves))):
            if len(leaves) >= max_intervals:
                raise RuntimeError("K21/G10 norm exceeded interval limit")
            j = max(range(len(leaves)), key=lambda k: leaves[k][3])
            lo, hi, _, _ = leaves.pop(j)
            mid = (lo + hi) / 2.0
            leaves.extend((leaf(lo, mid), leaf(mid, hi)))
        total = sum(q[2] for q in leaves)
        out.append(np.sqrt(2.0 * math.pi * total).real)
    return np.array(out, dtype=np.float64).reshape(c.shape)


def _np_swsh(spin: np.ndarray, theta: np.ndarray, phi: np.ndarray = 0.0, *, n_terms: int = 48) -> np.ndarray:
    spin = np.asarray(spin, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    bspin, btheta, bphi = np.broadcast_arrays(spin, theta, phi)

    # QNM evaluation
    r_jax, d_jax = evaluate_kerr_qnm_220(bspin)
    real = np.asarray(r_jax, dtype=np.float64)
    damp = np.asarray(d_jax, dtype=np.float64)

    c = bspin * (real - 1j * damp)
    _, coeff = _np_leaver_frobenius_coefficients(2, c, n_terms=n_terms)

    cb = np.broadcast_to(c, bspin.shape)
    ab = np.broadcast_to(coeff, bspin.shape + (coeff.shape[-1],))
    raw = _np_raw(np.cos(btheta), cb, ab)
    norm = _np_norm(c, coeff)
    raw0 = _np_raw(np.zeros_like(c), c, coeff)
    gauge = np.conj(raw0) / np.abs(raw0)
    gauge = np.broadcast_to(gauge.reshape(spin.shape), bspin.shape)
    norm_b = np.broadcast_to(norm.reshape(spin.shape), bspin.shape)
    out = raw * gauge / norm_b * np.exp(2j * bphi)
    return out


def spin_weighted_spheroidal_harmonic_table_compat(
    spin: Union[float, jnp.ndarray],
    theta: Union[float, jnp.ndarray],
    phi: Union[float, jnp.ndarray] = 0.0,
    *,
    n_terms: int = 48,
) -> jnp.ndarray:
    """Evaluate inferred public-compatible normalized ``s=-2, l=m=2`` shape in JAX."""
    spin_arr = jnp.asarray(spin, dtype=jnp.float64)
    theta_arr = jnp.asarray(theta, dtype=jnp.float64)
    phi_arr = jnp.asarray(phi, dtype=jnp.float64)
    bspin, btheta, bphi = jnp.broadcast_arrays(spin_arr, theta_arr, phi_arr)
    res_struct = jax.ShapeDtypeStruct(bspin.shape, jnp.complex128)
    return jax.pure_callback(
        lambda s, th, ph: _np_swsh(s, th, ph, n_terms=n_terms).astype(np.complex128),
        res_struct,
        bspin, btheta, bphi,
    )


def _params(final_mass, final_spin, distance, energy_fraction, inclination, phic, tc, grid=None):
    vals = [final_mass, final_spin, distance, energy_fraction, inclination, phic, tc]
    out = [jnp.asarray(v, dtype=jnp.float64) for v in vals]
    params = list(jnp.broadcast_arrays(*out))
    return params


def _grid(g, params):
    g = jnp.asarray(g, dtype=params[0].dtype)
    params = list(jnp.broadcast_arrays(*params))
    if g.ndim:
        grid_ndim = g.ndim
        batch = params[0].shape
        g = g.reshape((1,) * len(batch) + tuple(g.shape))
        params = [v.reshape(v.shape + (1,) * grid_ndim) for v in params]
    return g, params


def _angular_pair(a, wr, damp, inc, pc, angular_convention):
    if angular_convention == "public_compat":
        return (spin_weighted_spheroidal_harmonic_table_compat(a, inc, pc),
                spin_weighted_spheroidal_harmonic_table_compat(a, math.pi - inc, pc))
    if angular_convention != "source_spectral":
        raise ValueError("angular_convention must be 'public_compat' or 'source_spectral'")
    from jaxwave.approximants.kerr_ringdown import spin_weighted_spheroidal_harmonic as spectral
    c = a * (wr - 1j * damp)
    equator = spectral(2, 2, c, jnp.full_like(inc, math.pi / 2), 0., ell_max=18)
    gauge = jnp.conj(equator) / jnp.abs(equator)
    return (spectral(2, 2, c, inc, pc, ell_max=18) * gauge,
            spectral(2, 2, c, math.pi - inc, pc, ell_max=18) * gauge)


def kerr220_td(
    t: jnp.ndarray,
    final_mass: Union[float, jnp.ndarray],
    final_spin: Union[float, jnp.ndarray],
    distance: Union[float, jnp.ndarray],
    energy_fraction: Union[float, jnp.ndarray] = 0.01,
    inclination: Union[float, jnp.ndarray] = 0.0,
    phic: Union[float, jnp.ndarray] = 0.0,
    tc: Union[float, jnp.ndarray] = 0.0,
    *,
    angular_convention: str = "public_compat",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(hp,hc)`` for u=t-tc>=0; zero before onset in JAX."""
    M, a, D, e, inc, pc, t0 = _params(final_mass, final_spin, distance, energy_fraction, inclination, phic, tc, t)
    t, (M, a, D, e, inc, pc, t0) = _grid(t, [M, a, D, e, inc, pc, t0])
    wr, damp = evaluate_kerr_qnm_220(a)
    sec = M * MSUN_SEC
    omega = wr / sec
    lam = damp / sec
    A = ringdown_amplitude_from_energy_fraction(M, D, e, omega, 1.0 / lam, convention="single_mode")
    Sp, Sm = _angular_pair(a, wr, damp, inc, pc, angular_convention)
    u = t - t0
    up = jnp.maximum(u, 0.0)
    z = jnp.exp(-lam * up - 1j * omega * up)
    mask = (u >= 0.0)
    hp = -A / math.sqrt(2.0) * jnp.real(z * (Sp + Sm)) * mask
    hc = A / math.sqrt(2.0) * jnp.imag(z * (Sp - Sm)) * mask
    return hp, hc


def kerr220_fd(
    f: jnp.ndarray,
    final_mass: Union[float, jnp.ndarray],
    final_spin: Union[float, jnp.ndarray],
    distance: Union[float, jnp.ndarray],
    energy_fraction: Union[float, jnp.ndarray] = 0.01,
    inclination: Union[float, jnp.ndarray] = 0.0,
    phic: Union[float, jnp.ndarray] = 0.0,
    tc: Union[float, jnp.ndarray] = 0.0,
    *,
    angular_convention: str = "public_compat",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return Fourier-minus-frequency ``(hp,hc)`` for frequency f in Hz in JAX."""
    M, a, D, e, inc, pc, t0 = _params(final_mass, final_spin, distance, energy_fraction, inclination, phic, tc, f)
    f, (M, a, D, e, inc, pc, t0) = _grid(f, [M, a, D, e, inc, pc, t0])
    wr, damp = evaluate_kerr_qnm_220(a)
    sec = M * MSUN_SEC
    omega = wr / sec
    lam = damp / sec
    A = ringdown_amplitude_from_energy_fraction(M, D, e, omega, 1.0 / lam, convention="single_mode")
    Sp, Sm = _angular_pair(a, wr, damp, inc, pc, angular_convention)
    P = Sp + Sm
    C = Sp - Sm
    zp = 1.0 / (lam + 1j * (2.0 * math.pi * f + omega))
    zm = 1.0 / (lam + 1j * (2.0 * math.pi * f - omega))
    phase = jnp.exp(-2j * math.pi * f * t0)
    hp = -A / (2.0 * math.sqrt(2.0)) * (P * zp + jnp.conj(P) * zm) * phase
    hc = A / (2j * math.sqrt(2.0)) * (C * zp - jnp.conj(C) * zm) * phase
    return hp, hc


__all__ = [
    "kerr220_td",
    "kerr220_fd",
    "spin_weighted_spheroidal_harmonic_table_compat",
]
