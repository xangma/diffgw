"""Clean-room tensor-product B-splines and reduced-basis reconstruction in JAX.

Port of torchwave.approximants.rom.bspline for JAX / jaxwave.
Native JAX operations (jax.numpy, float64 precision).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import math
import numpy as np
import jax
import jax.numpy as jnp

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC, PI

TensorLike = Union[float, Sequence[float], jnp.ndarray, np.ndarray]


def linear_interpolate(x: TensorLike, grid: TensorLike, values: jnp.ndarray) -> jnp.ndarray:
    """Piecewise-linear interpolation on a strictly increasing ROM grid in JAX."""
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    grid_arr = jnp.asarray(grid, dtype=jnp.float64)
    val_arr = jnp.asarray(values, dtype=jnp.float64)
    if val_arr.ndim == 1:
        return jnp.interp(x_arr, grid_arr, val_arr)
    # Batched interpolation along final axis
    return jax.vmap(lambda v: jnp.interp(x_arr, grid_arr, v))(val_arr)


def cubic_spline_interpolate(x: TensorLike, grid: TensorLike, values: TensorLike) -> jnp.ndarray:
    """Evaluate the not-a-knot cubic spline through tabulated values in JAX."""
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    grid_arr = jnp.asarray(grid, dtype=jnp.float64)
    val_arr = jnp.asarray(values, dtype=jnp.float64)

    n = grid_arr.size
    if n == 2:
        return linear_interpolate(x_arr, grid_arr, val_arr)
    if n == 3:
        vand = jnp.stack((jnp.ones_like(grid_arr), grid_arr, grid_arr**2), axis=-1)
        coef = jnp.linalg.solve(vand, val_arr)
        return coef[0] + coef[1] * x_arr + coef[2] * (x_arr**2)

    h = grid_arr[1:] - grid_arr[:-1]

    mat = jnp.zeros((n, n), dtype=jnp.float64)
    mat = mat.at[0, 0].set(h[1])
    mat = mat.at[0, 1].set(-(h[0] + h[1]))
    mat = mat.at[0, 2].set(h[0])
    mat = mat.at[-1, -3].set(h[-1])
    mat = mat.at[-1, -2].set(-(h[-2] + h[-1]))
    mat = mat.at[-1, -1].set(h[-2])

    slope = (val_arr[1:] - val_arr[:-1]) / h
    interior = jnp.arange(1, n - 1)
    mat = mat.at[interior, interior - 1].set(h[:-1])
    mat = mat.at[interior, interior].set(2.0 * (h[:-1] + h[1:]))
    mat = mat.at[interior, interior + 1].set(h[1:])

    rhs = jnp.zeros((n,), dtype=jnp.float64)
    rhs = rhs.at[1:-1].set(6.0 * (slope[1:] - slope[:-1]))
    second = jnp.linalg.solve(mat, rhs)

    idx = jnp.clip(jnp.searchsorted(grid_arr, x_arr), 1, n - 1)
    j = idx - 1
    dx = x_arr - grid_arr[j]
    hj = h[j]
    a = (grid_arr[j + 1] - x_arr) / hj
    b = dx / hj
    return (
        a * val_arr[j]
        + b * val_arr[j + 1]
        + ((a**3 - a) * second[j] + (b**3 - b) * second[j + 1]) * (hj**2) / 6.0
    )


def _knots(breaks: jnp.ndarray, order: int) -> jnp.ndarray:
    return jnp.concatenate([
        jnp.broadcast_to(breaks[:1], (order,)),
        breaks[1:-1],
        jnp.broadcast_to(breaks[-1:], (order,)),
    ])


def bspline_basis(
    x: TensorLike,
    breaks: TensorLike,
    *,
    order: int = 4,
    validate_domain: bool = True,
) -> jnp.ndarray:
    """Evaluate all clamped B-spline basis functions at ``x`` in JAX."""
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    breaks_arr = jnp.asarray(breaks, dtype=jnp.float64)
    knots = _knots(breaks_arr, order)

    flat = x_arr.reshape(-1, 1)
    ncoeff = breaks_arr.size + order - 2
    n0 = knots.size - 1
    left, right = knots[:-1], knots[1:]
    values = (flat >= left) & (flat < right)
    endpoint_interval = (right == knots[-1]) & (left < right)
    values = values | ((flat[:, 0:1] == knots[-1]) & endpoint_interval)
    values = values.astype(jnp.float64)

    for degree in range(1, order):
        count = n0 - degree
        den_left = knots[degree:degree + count] - knots[:count]
        den_right = knots[degree + 1:degree + 1 + count] - knots[1:count + 1]
        prev = values
        safe_left = jnp.where(den_left != 0, den_left, 1.0)
        safe_right = jnp.where(den_right != 0, den_right, 1.0)
        a = (flat - knots[:count]) / safe_left
        b = (knots[degree + 1:degree + 1 + count] - flat) / safe_right
        out = jnp.where(den_left != 0, a * prev[:, :count], 0.0)
        out = out + jnp.where(den_right != 0, b * prev[:, 1:count + 1], 0.0)
        values = out

    return values[:, :ncoeff].reshape(x_arr.shape + (ncoeff,))


def tensor_product_basis(
    eta: TensorLike,
    chi1: TensorLike,
    chi2: TensorLike,
    grids: Tuple[TensorLike, TensorLike, TensorLike],
    *,
    shifts: Optional[Tuple[float, float, float]] = None,
    order: int = 4,
) -> jnp.ndarray:
    """Return flattened tensor-product basis, with the last axis C-packed in JAX."""
    if shifts is None:
        shifts = (0.0, 0.0, 0.0)
    a = bspline_basis(jnp.asarray(eta) + shifts[0], jnp.asarray(grids[0]) + shifts[0], order=order)
    b = bspline_basis(jnp.asarray(chi1) + shifts[1], jnp.asarray(grids[1]) + shifts[1], order=order)
    c = bspline_basis(jnp.asarray(chi2) + shifts[2], jnp.asarray(grids[2]) + shifts[2], order=order)
    leading = np.broadcast_shapes(a.shape[:-1], b.shape[:-1], c.shape[:-1])
    a = jnp.broadcast_to(a, leading + (a.shape[-1],))
    b = jnp.broadcast_to(b, leading + (b.shape[-1],))
    c = jnp.broadcast_to(c, leading + (c.shape[-1],))
    return (a[..., :, None, None] * b[..., None, :, None] * c[..., None, None, :]).reshape(
        leading + (a.shape[-1] * b.shape[-1] * c.shape[-1],)
    )


def reconstruct_reduced_basis(basis: jnp.ndarray, coefficients: jnp.ndarray) -> jnp.ndarray:
    """Project coefficients through basis rows shaped ``(n_basis, n_frequency)``."""
    return jnp.matmul(coefficients, basis)


class OfficialROMSurrogate:
    """Evaluate one externally supplied official ROM HDF5 submodel in JAX."""

    def __init__(self, group, *, name: str = ""):
        self.name = name
        self.eta_grid = jnp.asarray(group["etavec"][...], dtype=jnp.float64)
        self.chi1_grid = jnp.asarray(group["chi1vec"][...], dtype=jnp.float64)
        self.chi2_grid = jnp.asarray(group["chi2vec"][...], dtype=jnp.float64)
        self.bamp = jnp.asarray(group["Bamp"][...], dtype=jnp.float64)
        self.bphase = jnp.asarray(group["Bphase"][...], dtype=jnp.float64)
        self.mf_grid_amp = jnp.asarray(group["Mf_grid_Amp"][...], dtype=jnp.float64)
        self.mf_grid_phase = jnp.asarray(group["Mf_grid_Phi"][...], dtype=jnp.float64)

        parameter_shape = (self.eta_grid.size + 2, self.chi1_grid.size + 2, self.chi2_grid.size + 2)
        amp_shape = (self.bamp.shape[0],) + parameter_shape
        phase_shape = (self.bphase.shape[0],) + parameter_shape
        self.amp_coeff = jnp.asarray(group["Amp_ciall"][...], dtype=jnp.float64).reshape(amp_shape)
        self.phase_coeff = jnp.asarray(group["Phase_ciall"][...], dtype=jnp.float64).reshape(phase_shape)

    def evaluate(self, eta: TensorLike, chi1: TensorLike, chi2: TensorLike) -> Tuple[jnp.ndarray, jnp.ndarray]:
        p = tensor_product_basis(
            eta, chi1, chi2,
            (self.eta_grid, self.chi1_grid, self.chi2_grid),
            shifts=(0.0, 0.0, 0.0),
        )
        amp_c = jnp.einsum("...p,fp->...f", p, self.amp_coeff.reshape(self.amp_coeff.shape[0], -1))
        phase_c = jnp.einsum("...p,fp->...f", p, self.phase_coeff.reshape(self.phase_coeff.shape[0], -1))
        return reconstruct_reduced_basis(self.bamp, amp_c), reconstruct_reduced_basis(self.bphase, phase_c)

    def evaluate_frequency(self, mf: TensorLike, eta: TensorLike, chi1: TensorLike, chi2: TensorLike) -> Tuple[jnp.ndarray, jnp.ndarray]:
        amp_nodes, phase_nodes = self.evaluate(eta, chi1, chi2)
        amp_nodes = amp_nodes.reshape(-1)
        phase_nodes = phase_nodes.reshape(-1)
        return (
            cubic_spline_interpolate(mf, self.mf_grid_amp, amp_nodes),
            cubic_spline_interpolate(mf, self.mf_grid_phase, phase_nodes),
        )

    def evaluate_complex(self, mf: TensorLike, eta: TensorLike, chi1: TensorLike, chi2: TensorLike) -> jnp.ndarray:
        amp, phase = self.evaluate_frequency(mf, eta, chi1, chi2)
        return amp * jnp.exp(1j * phase)

    def evaluate_polarizations(
        self, frequency_hz: TensorLike, total_mass_msun: TensorLike,
        distance_mpc: TensorLike, eta: TensorLike, chi1: TensorLike,
        chi2: TensorLike, *, inclination: TensorLike = 0.0,
        phic: TensorLike = 0.0, tc: TensorLike = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        f = jnp.asarray(frequency_hz, dtype=jnp.float64)
        mass = jnp.asarray(total_mass_msun, dtype=jnp.float64)
        distance = jnp.asarray(distance_mpc, dtype=jnp.float64)
        inc = jnp.asarray(inclination, dtype=jnp.float64)
        mf = f * mass * MSUN_SEC
        mode = self.evaluate_complex(mf, eta, chi1, chi2)
        phase_shift = jnp.exp(1j * (2.0 * PI * f * jnp.asarray(tc, dtype=jnp.float64) - 2.0 * jnp.asarray(phic, dtype=jnp.float64)))
        m_sec = mass * MSUN_SEC
        scale = m_sec * (C_SI * m_sec) / (distance * MPC_SI)
        mode = mode * scale * phase_shift
        ci = jnp.cos(inc)
        hplus = 0.5 * (1.0 + ci**2) * (0.5 * mode)
        hcross = ci * (0.5j * mode)
        return hplus, hcross


def load_official_rom(path: Union[str, Path], submodel: str = "sub1") -> OfficialROMSurrogate:
    """Load one submodel from the official SEOBNRv4 ROM HDF5 file."""
    import h5py
    with h5py.File(path, "r") as handle:
        if submodel not in handle:
            raise KeyError(f"submodel {submodel} not found in {path}")
        return OfficialROMSurrogate(handle[submodel], name=submodel)


def evaluate_stitched_frequency(
    low: OfficialROMSurrogate, high: OfficialROMSurrogate,
    mf: TensorLike, eta: TensorLike, chi1: TensorLike, chi2: TensorLike,
    *, transition: float = 0.01, ringdown_frequency: Optional[float] = None,
    overlap: Optional[Tuple[float, float]] = (0.0074, 0.014), fit_points: Optional[int] = None,
    fit_nodes: Optional[TensorLike] = None, reference_mf: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Glue low/high official ROM patches using Pürrer's published rule in JAX."""
    if ringdown_frequency is None:
        raise ValueError("ringdown_frequency must be supplied explicitly")
    rd = jnp.asarray(ringdown_frequency, dtype=jnp.float64)
    tr = jnp.asarray(transition, dtype=jnp.float64)

    if overlap is None:
        lo = max(float(low.mf_grid_phase[0]), float(high.mf_grid_phase[0]))
        hi = min(float(low.mf_grid_phase[-1]), float(high.mf_grid_phase[-1]))
    else:
        lo, hi = map(float, overlap)

    zmid = (lo + hi) / 2.0
    zscale = (hi - lo) / 2.0

    if fit_nodes is not None:
        nodes = jnp.asarray(fit_nodes, dtype=jnp.float64)
    else:
        grid = high.mf_grid_phase
        nodes = grid[(grid >= lo) & (grid <= hi)]

    z = (nodes - zmid) / zscale
    design = jnp.stack((jnp.ones_like(z), z, z**2, z**3), axis=-1)

    _, low_phase_fit = low.evaluate_frequency(nodes, eta, chi1, chi2)
    _, high_phase_fit = high.evaluate_frequency(nodes, eta, chi1, chi2)

    cl = jnp.linalg.lstsq(design, low_phase_fit, rcond=None)[0]
    ch = jnp.linalg.lstsq(design, high_phase_fit, rcond=None)[0]

    def _poly(c, x_val):
        zz = (x_val - zmid) / zscale
        p = c[0] + zz * (c[1] + zz * (c[2] + zz * c[3]))
        dp = (c[1] + zz * (2.0 * c[2] + zz * 3.0 * c[3])) / zscale
        return p, dp

    pl, dl = _poly(cl, tr)
    ph, dh = _poly(ch, tr)
    slope = dl - dh
    intercept = pl - ph - slope * tr

    x = jnp.asarray(mf, dtype=jnp.float64)

    amp_low_grid = low.mf_grid_amp[low.mf_grid_amp <= tr]
    amp_high_grid = high.mf_grid_amp[high.mf_grid_amp > tr]
    amp_low_values, _ = low.evaluate_frequency(amp_low_grid, eta, chi1, chi2)
    amp_high_values, _ = high.evaluate_frequency(amp_high_grid, eta, chi1, chi2)
    amp_grid = jnp.concatenate((amp_low_grid, amp_high_grid))
    amp_values = jnp.concatenate((amp_low_values, amp_high_values))
    amp = cubic_spline_interpolate(x, amp_grid, amp_values)

    phase_low_grid = low.mf_grid_phase[low.mf_grid_phase <= tr]
    phase_high_grid = high.mf_grid_phase[high.mf_grid_phase > tr]
    phase_low_values = low.evaluate_frequency(phase_low_grid, eta, chi1, chi2)[1]
    phase_high_values = high.evaluate_frequency(phase_high_grid, eta, chi1, chi2)[1]

    phase_grid = jnp.concatenate((phase_low_grid, phase_high_grid))
    phase_values = jnp.concatenate((phase_low_values, phase_high_values + intercept + slope * phase_high_grid))

    def _eval_rd(r):
        return cubic_spline_interpolate(r, phase_grid, phase_values)

    gauge = jax.grad(_eval_rd)(rd)

    phase = cubic_spline_interpolate(x, phase_grid, phase_values) - gauge * (x - rd)
    if reference_mf is not None:
        ref = jnp.asarray(reference_mf, dtype=jnp.float64)
        phase = phase - (cubic_spline_interpolate(ref, phase_grid, phase_values) - gauge * (ref - rd))

    return amp, phase


__all__ = [
    "bspline_basis",
    "tensor_product_basis",
    "reconstruct_reduced_basis",
    "linear_interpolate",
    "cubic_spline_interpolate",
    "OfficialROMSurrogate",
    "load_official_rom",
    "evaluate_stitched_frequency",
]
