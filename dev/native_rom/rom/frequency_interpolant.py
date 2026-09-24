"""Development-only frequency interpolation of an in-memory reduced ROM.

Not-a-knot cubic coefficients are computed once for the reduced basis, not
once per waveform query. The spline is linear in the reduced coefficients, so
parameter interpolation and frequency interpolation commute. No physical
waveforms or serialized tables are provided here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .backend_evaluation import _jax_cardinal, _torch_cardinal
from .reduced_interpolant import _basis_and_derivative


@dataclass(frozen=True)
class SplineField:
    mean: np.ndarray  # (4, n_intervals), descending polynomial powers
    basis: np.ndarray  # (rank, 4, n_intervals)


def _spline_field(frequency, field):
    mean = CubicSpline(frequency, field.mean, bc_type="not-a-knot").c
    basis = np.moveaxis(
        CubicSpline(frequency, field.basis, axis=-1,
                    bc_type="not-a-knot").c, -1, 0)
    return SplineField(mean, basis)


def _spline_values(field, frequency, query):
    flat = np.asarray(query).reshape(-1)
    index = np.clip(np.searchsorted(frequency, flat, side="right") - 1,
                    0, len(frequency) - 2)
    dx = flat - frequency[index]
    mean = field.mean[:, index]
    basis = field.basis[:, :, index]
    mean_value = ((mean[0] * dx + mean[1]) * dx + mean[2]) * dx + mean[3]
    basis_value = ((basis[:, 0] * dx + basis[:, 1]) * dx
                   + basis[:, 2]) * dx + basis[:, 3]
    return mean_value, basis_value


@dataclass(frozen=True)
class FrequencyReducedInterpolant:
    """A reduced parameter interpolant sampled at arbitrary shared frequencies."""

    model: object
    log_amplitude: SplineField
    phase: SplineField

    @classmethod
    def from_model(cls, model):
        return cls(model,
                   _spline_field(model.frequency, model.log_amplitude),
                   _spline_field(model.frequency, model.phase))

    def evaluate(self, frequency, q, chi1, chi2, *, derivatives=False):
        """Return complex spectrum and optional parameter Jacobian.

        ``frequency`` is shared across the broadcast parameter batch; its
        shape is appended to the parameter batch shape. No extrapolation is
        allowed. Frequency derivatives are available through backend autodiff.
        """
        f = np.asarray(frequency, dtype=np.float64)
        if (not np.isfinite(f).all() or np.any(f < self.model.frequency[0])
                or np.any(f > self.model.frequency[-1])):
            raise ValueError("frequency outside the finite ROM domain")
        queries = np.broadcast_arrays(q, chi1, chi2)
        if any(not np.isfinite(x).all() for x in queries):
            raise ValueError("ROM parameters must be finite")
        factors = [_basis_and_derivative(nodes, x)
                   for nodes, x in zip(self.model.nodes, queries)]
        amp_spline = _spline_values(self.log_amplitude, self.model.frequency, f)
        phase_spline = _spline_values(self.phase, self.model.frequency, f)

        def field(part, spline_values, axes, *, include_mean=True):
            reduced = np.einsum("...i,...j,...k,ijkl->...l", *axes,
                                part.coefficients, optimize=True)
            mean, basis = spline_values
            result = np.einsum("...r,rf->...f", reduced, basis, optimize=True)
            if include_mean:
                result = result + mean
            return result.reshape(queries[0].shape + f.shape)

        cardinal = [pair[0] for pair in factors]
        amp = field(self.model.log_amplitude, amp_spline, cardinal)
        phase = field(self.model.phase, phase_spline, cardinal)
        waveform = np.exp(amp + 1j * phase)
        if not derivatives:
            return waveform
        jacobian = []
        for axis in range(3):
            differentiated = [pair[1] if i == axis else pair[0]
                                for i, pair in enumerate(factors)]
            damp = field(self.model.log_amplitude, amp_spline, differentiated,
                         include_mean=False)
            dphase = field(self.model.phase, phase_spline, differentiated,
                           include_mean=False)
            jacobian.append(waveform * (damp + 1j * dphase))
        return waveform, np.stack(jacobian, axis=len(queries[0].shape))


def _torch_spline_values(torch, spline, grid, frequency, device):
    flat = frequency.reshape(-1)
    knots = torch.as_tensor(grid, dtype=torch.float64, device=device)
    index = torch.searchsorted(knots, flat.contiguous(), right=True).sub(1).clamp(
        0, len(grid) - 2)
    dx = flat - knots[index]
    mean = torch.as_tensor(spline.mean, dtype=torch.float64, device=device)[:, index]
    basis = torch.as_tensor(spline.basis, dtype=torch.float64, device=device)[:, :, index]
    return (((mean[0] * dx + mean[1]) * dx + mean[2]) * dx + mean[3],
            ((basis[:, 0] * dx + basis[:, 1]) * dx + basis[:, 2]) * dx
            + basis[:, 3])


def evaluate_torch_frequency(interpolant, frequency, q, chi1, chi2):
    """Evaluate a reduced spline with Torch parameter/frequency autograd."""
    import torch

    inputs = (frequency, q, chi1, chi2)
    device = next((x.device for x in inputs if isinstance(x, torch.Tensor)),
                  torch.device("cpu"))
    f = torch.as_tensor(frequency, dtype=torch.float64, device=device)
    queries = torch.broadcast_tensors(*(
        torch.as_tensor(x, dtype=torch.float64, device=device)
        for x in (q, chi1, chi2)))
    cardinal = [_torch_cardinal(torch, nodes, x)
                for nodes, x in zip(interpolant.model.nodes, queries)]

    def field(part, spline):
        coefficients = torch.as_tensor(part.coefficients, dtype=torch.float64,
                                       device=device)
        reduced = torch.einsum("...i,...j,...k,ijkl->...l", *cardinal,
                               coefficients)
        mean, basis = _torch_spline_values(
            torch, spline, interpolant.model.frequency, f, device)
        result = mean + torch.einsum("...r,rf->...f", reduced, basis)
        return result.reshape(queries[0].shape + f.shape)

    amp = field(interpolant.model.log_amplitude, interpolant.log_amplitude)
    phase = field(interpolant.model.phase, interpolant.phase)
    waveform = torch.exp(amp + 1j * phase)
    outside = ((~torch.isfinite(f)) | (f < interpolant.model.frequency[0]) | (
        f > interpolant.model.frequency[-1])).reshape(
            (1,) * queries[0].ndim + f.shape)
    for nodes, query in zip(interpolant.model.nodes, queries):
        outside = outside | ((~torch.isfinite(query)) | (query < nodes[0]) | (
            query > nodes[-1])).reshape(query.shape + (1,) * f.ndim)
    return torch.where(outside, torch.full_like(waveform, complex(float("nan"), 0)),
                       waveform)


def _jax_spline_values(jnp, spline, grid, frequency):
    flat = jnp.reshape(frequency, (-1,))
    knots = jnp.asarray(grid, dtype=jnp.float64)
    index = jnp.clip(jnp.searchsorted(knots, flat, side="right") - 1,
                     0, len(grid) - 2)
    dx = flat - knots[index]
    mean = jnp.asarray(spline.mean, dtype=jnp.float64)[:, index]
    basis = jnp.asarray(spline.basis, dtype=jnp.float64)[:, :, index]
    return (((mean[0] * dx + mean[1]) * dx + mean[2]) * dx + mean[3],
            ((basis[:, 0] * dx + basis[:, 1]) * dx + basis[:, 2]) * dx
            + basis[:, 3])


def evaluate_jax_frequency(interpolant, frequency, q, chi1, chi2):
    """Evaluate a reduced spline with JAX JIT and parameter/frequency autodiff."""
    import jax
    import jax.numpy as jnp

    if not jax.config.jax_enable_x64:
        raise RuntimeError("development ROM evaluation requires JAX float64 enabled")
    f = jnp.asarray(frequency, dtype=jnp.float64)
    queries = jnp.broadcast_arrays(*(
        jnp.asarray(x, dtype=jnp.float64) for x in (q, chi1, chi2)))
    cardinal = [_jax_cardinal(jnp, nodes, x)
                for nodes, x in zip(interpolant.model.nodes, queries)]

    def field(part, spline):
        coefficients = jnp.asarray(part.coefficients, dtype=jnp.float64)
        reduced = jnp.einsum("...i,...j,...k,ijkl->...l", *cardinal,
                             coefficients)
        mean, basis = _jax_spline_values(
            jnp, spline, interpolant.model.frequency, f)
        result = mean + jnp.einsum("...r,rf->...f", reduced, basis)
        return jnp.reshape(result, queries[0].shape + f.shape)

    amp = field(interpolant.model.log_amplitude, interpolant.log_amplitude)
    phase = field(interpolant.model.phase, interpolant.phase)
    waveform = jnp.exp(amp + 1j * phase)
    outside = jnp.reshape(
        (~jnp.isfinite(f)) | (f < interpolant.model.frequency[0]) | (
            f > interpolant.model.frequency[-1]),
        (1,) * queries[0].ndim + f.shape)
    for nodes, query in zip(interpolant.model.nodes, queries):
        outside = outside | jnp.reshape(
            (~jnp.isfinite(query)) | (query < nodes[0]) | (query > nodes[-1]),
            query.shape + (1,) * f.ndim)
    return jnp.where(outside, jnp.nan + 0j, waveform)
