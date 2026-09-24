"""Development-only SVD ROM with tensor-product cubic parameter splines.

The not-a-knot endpoint rule is an explicit candidate, not a claim about the
historical ROM's unpublished training coefficients or patch layout. Inputs
are already conditioned frequency-domain fields from a caller-owned generator.
The optional ``nu`` / ``linear`` configuration matches the published choice
of parameter coordinate and amplitude field; it does not reproduce the
authors' tables. The ``log_amplitude`` field name is retained for the older
development API and stores linear amplitude coefficients in that configuration.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from .reduced_interpolant import ReducedField, _fit_field


def _grid(values):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all()
            or not np.all(np.diff(values) > 0)):
        raise ValueError("parameter grids need at least four finite increasing nodes")
    return values.copy()


def _fit_coefficients(field, nodes):
    values = field.coefficients
    knots = []
    for axis, grid in enumerate(nodes):
        spline = make_interp_spline(grid, values, k=3, axis=axis)
        knots.append(spline.t.copy())
        values = np.moveaxis(spline.c, 0, axis)
    return ReducedField(field.mean, field.basis, values,
                        field.training_relative_error), tuple(knots)


def _basis(knots, query):
    n = len(knots) - 4
    spline = BSpline(knots, np.eye(n), 3, extrapolate=False)
    return spline(query), spline.derivative()(query)


@dataclass(frozen=True)
class SplineTensorInterpolant:
    """One parameter patch on a shared frequency grid, without physical data."""

    nodes: tuple[np.ndarray, np.ndarray, np.ndarray]
    frequency: np.ndarray
    knots: tuple[np.ndarray, np.ndarray, np.ndarray]
    log_amplitude: ReducedField
    phase: ReducedField
    first_coordinate: str = "q"
    amplitude_kind: str = "log"

    @classmethod
    def fit(cls, nodes, frequency, log_amplitude, phase, *, ranks,
            first_coordinate="q", amplitude_kind="log"):
        """Fit one in-memory patch; ``nu`` nodes are increasing, queries use q."""
        if len(nodes) != 3 or len(ranks) != 2:
            raise ValueError("expected three parameter grids and two ranks")
        if first_coordinate not in ("q", "nu") or amplitude_kind not in ("log", "linear"):
            raise ValueError("unsupported ROM coordinate or amplitude kind")
        grids = tuple(_grid(values) for values in nodes)
        if first_coordinate == "nu" and (grids[0][0] <= 0 or grids[0][-1] > .25):
            raise ValueError("symmetric-mass-ratio nodes must lie in (0, 0.25]")
        frequency = np.asarray(frequency, dtype=np.float64)
        if (frequency.ndim != 1 or len(frequency) < 2
                or not np.isfinite(frequency).all()
                or not np.all(np.diff(frequency) > 0)):
            raise ValueError("frequency grid must be finite and increasing")
        shape = tuple(map(len, grids)) + (len(frequency),)
        fields = tuple(np.asarray(values, dtype=np.float64)
                       for values in (log_amplitude, phase))
        if any(values.shape != shape or not np.isfinite(values).all()
               for values in fields):
            raise ValueError("training fields must be finite and match the tensor grid")
        if amplitude_kind == "linear" and np.any(fields[0] <= 0):
            raise ValueError("linear training amplitude must be positive")
        if any(not isinstance(rank, int) or rank < 1
               or rank > min(np.prod(shape[:-1]), shape[-1]) for rank in ranks):
            raise ValueError("each rank must fit the sample and frequency dimensions")
        amp, knots = _fit_coefficients(_fit_field(fields[0], ranks[0]), grids)
        phi, phase_knots = _fit_coefficients(_fit_field(fields[1], ranks[1]), grids)
        if any(not np.array_equal(a, b) for a, b in zip(knots, phase_knots)):
            raise RuntimeError("amplitude and phase parameter knots differ")
        return cls(grids, frequency.copy(), knots, amp, phi,
                   first_coordinate, amplitude_kind)

    def evaluate(self, q, chi1, chi2, *, derivatives=False):
        """Evaluate a complex spectrum and optional parameter Jacobian."""
        q = np.asarray(q, dtype=np.float64)
        physical = self.first_coordinate == "nu"
        if physical and np.any(q < 1):
            raise ValueError("mass ratio q must be at least one")
        queries = np.broadcast_arrays(q / (1 + q)**2 if physical else q,
                                      chi1, chi2)
        if any(not np.isfinite(value).all() for value in queries):
            raise ValueError("ROM parameters must be finite")
        if any(np.any((value < grid[0]) | (value > grid[-1]))
               for value, grid in zip(queries, self.nodes)):
            raise ValueError("ROM parameter outside its training domain")
        factors = [_basis(knots, value)
                   for knots, value in zip(self.knots, queries)]

        def field(part, axes, *, include_mean=True):
            reduced = np.einsum("...i,...j,...k,ijkl->...l", *axes,
                                part.coefficients, optimize=True)
            result = reduced @ part.basis
            return result + part.mean if include_mean else result

        cardinal = [pair[0] for pair in factors]
        amp = field(self.log_amplitude, cardinal)
        phi = field(self.phase, cardinal)
        if self.amplitude_kind == "linear" and np.any(amp <= 0):
            raise ValueError("interpolated ROM amplitude must be positive")
        phase_factor = np.exp(1j * phi)
        waveform = (amp if self.amplitude_kind == "linear" else np.exp(amp)) * phase_factor
        if not derivatives:
            return waveform
        jacobian = []
        for axis in range(3):
            differentiated = [pair[1] if i == axis else pair[0]
                                for i, pair in enumerate(factors)]
            damp = field(self.log_amplitude, differentiated, include_mean=False)
            dphi = field(self.phase, differentiated, include_mean=False)
            derivative = (phase_factor * (damp + 1j * amp * dphi)
                          if self.amplitude_kind == "linear"
                          else waveform * (damp + 1j * dphi))
            if axis == 0 and physical:
                derivative = derivative * ((1 - q) / (1 + q)**3)[..., None]
            jacobian.append(derivative)
        return waveform, np.stack(jacobian, axis=-2)


def evaluate_torch_spline(model, q, chi1, chi2):
    """Torch evaluation with parameter autograd; invalid queries return NaN."""
    import torch
    from torchwave.approximants.rom.bspline import bspline_basis

    inputs = (q, chi1, chi2)
    device = next((x.device for x in inputs if isinstance(x, torch.Tensor)),
                  torch.device("cpu"))
    q, chi1, chi2 = (torch.as_tensor(x, dtype=torch.float64, device=device)
                     for x in inputs)
    physical = model.first_coordinate == "nu"
    queries = torch.broadcast_tensors(q / (1 + q)**2 if physical else q,
                                      chi1, chi2)
    factors = [bspline_basis(x, knots[3:-3], order=4, validate_domain=False)
               for knots, x in zip(model.knots, queries)]

    def field(part):
        coefficients = torch.as_tensor(part.coefficients, dtype=torch.float64,
                                       device=device)
        basis = torch.as_tensor(part.basis, dtype=torch.float64, device=device)
        mean = torch.as_tensor(part.mean, dtype=torch.float64, device=device)
        reduced = torch.einsum("...i,...j,...k,ijkl->...l", *factors,
                               coefficients)
        return mean + reduced @ basis

    amp = field(model.log_amplitude)
    phase_factor = torch.exp(1j * field(model.phase))
    waveform = (amp if model.amplitude_kind == "linear" else torch.exp(amp)) * phase_factor
    outside = torch.zeros_like(queries[0], dtype=torch.bool)
    if physical:
        outside = outside | (q < 1)
    for grid, x in zip(model.nodes, queries):
        outside = outside | ~torch.isfinite(x) | (x < grid[0]) | (x > grid[-1])
    invalid = outside[..., None]
    if model.amplitude_kind == "linear":
        invalid = invalid | (amp <= 0)
    return torch.where(invalid,
                       torch.full_like(waveform, complex(float("nan"), 0)), waveform)


def evaluate_jax_spline(model, q, chi1, chi2):
    """JAX JIT/autodiff evaluation; invalid queries return NaN."""
    import jax
    import jax.numpy as jnp
    from jaxwave.approximants.rom.bspline import bspline_basis

    if not jax.config.jax_enable_x64:
        raise RuntimeError("development ROM evaluation requires JAX float64 enabled")
    q, chi1, chi2 = (jnp.asarray(x, dtype=jnp.float64) for x in (q, chi1, chi2))
    physical = model.first_coordinate == "nu"
    queries = jnp.broadcast_arrays(q / (1 + q)**2 if physical else q,
                                   chi1, chi2)
    factors = [bspline_basis(x, knots[3:-3], order=4)
               for knots, x in zip(model.knots, queries)]

    def field(part):
        coefficients = jnp.asarray(part.coefficients, dtype=jnp.float64)
        basis = jnp.asarray(part.basis, dtype=jnp.float64)
        mean = jnp.asarray(part.mean, dtype=jnp.float64)
        reduced = jnp.einsum("...i,...j,...k,ijkl->...l", *factors,
                             coefficients)
        return mean + reduced @ basis

    amp = field(model.log_amplitude)
    phase_factor = jnp.exp(1j * field(model.phase))
    waveform = (amp if model.amplitude_kind == "linear" else jnp.exp(amp)) * phase_factor
    outside = jnp.zeros_like(queries[0], dtype=bool)
    if physical:
        outside = outside | (q < 1)
    for grid, x in zip(model.nodes, queries):
        outside = outside | ~jnp.isfinite(x) | (x < grid[0]) | (x > grid[-1])
    invalid = outside[..., None]
    if model.amplitude_kind == "linear":
        invalid = invalid | (amp <= 0)
    return jnp.where(invalid, jnp.nan + 0j, waveform)
